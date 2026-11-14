"""The MJX env base: navigation controller, possession, observation, episode bookkeeping.

A functional mirror of `mjsoccer.base.BaseEnv`. One `step` is one decision of
`repeat_action` control steps, and each control step is three pieces, kept apart so
tests/test_mjx_task_logic.py can put the CPU's own physics between them:

    control(state, action)  the controller and the dribbler, written into `ctrl`
    physics(data)           `decimation` mjx.step calls on MJX-Warp
    settle(state)           possession, observation, reward, ending, from the stepped data

Per-world state lives in `State.info["world"]`, a pytree: `grace` (n_robots,), `steps` (in
control steps, as the CPU counts them), `drive_cmd` (the energy penalties' input), the
world's RNG key, and the task's own entries. It is nested rather than spread over `info`
because Brax's `EpisodeWrapper` keeps its own `info["steps"]`; the env still owns the
ending: `done` includes truncation at `max_steps`, flagged in `info["truncation"]`, so an
`EpisodeWrapper(episode_length=max_steps / repeat_action)` only agrees with it.

Not yet: the kick and chip (ticket 08: those action dims are ignored here), per-world
failure handling (ticket 09), domain randomisation (ticket 14).
"""

from __future__ import annotations

import functools
import inspect
import math

import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.base import Env, State
from mujoco import mjx

from .. import base as cpu_base
from .. import navigation as nav
from .. import scene

GRAPH_MODES = ("WARP", "WARP_STAGED")
# Warp's contact buffer, per world. It has to hold every broadphase *candidate*, not just
# the contacts the CPU ends up with: ~35 per world with 2 robots and ~120 with 7, parked
# 0.5 m apart (.scratch/gpu-port/issues/04-mjx-install-smoke.md).
NACON_PER_ROBOT = 24
NACON_MIN = 64
NJMAX_PER_CONTACT = 4


# ------------------------------------------------------------ navigation controller


def smallest_angle_diff(a, b):
    """`navigation.smallest_angle_diff`: b - a wrapped to [-pi, pi), fmod then shift."""
    mod = 2 * math.pi
    angle = b - a
    angle = jnp.where((angle < -mod) | (angle > mod), jnp.fmod(angle, mod), angle)
    angle = jnp.where(angle < 0, angle + mod, angle)
    return jnp.where(angle >= math.pi, angle - mod, jnp.where(angle < -math.pi, angle + mod, angle))


def _rotate(v, angle):
    c, s = jnp.cos(angle), jnp.sin(angle)
    return jnp.stack([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _axis(to_target, v_current, v_desired, target, pos, max_velocity):
    """`navigation._axis`, branch for branch."""
    brake_at = (to_target * v_desired < 0) | (to_target * v_current < 0)
    zero_point = target - jnp.sign(v_desired) * v_desired**2 / (2 * nav.MAX_ACCEL)
    to_target = jnp.where(brake_at, zero_point - pos, to_target)
    v_desired = jnp.where(brake_at, 0.0, v_desired)
    braking = jnp.abs(to_target) < jnp.abs(v_current**2 - v_desired**2) / (2 * nav.MAX_ACCEL)
    v = v_current + jnp.where(
        braking, jnp.sign(v_desired - v_current), jnp.sign(to_target)
    ) * nav.MAX_DV
    return jnp.where(jnp.abs(v) > max_velocity, jnp.sign(v) * max_velocity, v)


def go_to_point(pos, vel, angle, target, target_angle, target_velocity=(0.0, 0.0),
                max_velocity=nav.MAX_VELOCITY, k_p=nav.ANGLE_KP):
    """`navigation.go_to_point` in jnp: body-frame (vx, vy, omega) as a (3,) array."""
    pos, vel, target = jnp.asarray(pos), jnp.asarray(vel), jnp.asarray(target)
    target_velocity = jnp.asarray(target_velocity, dtype=pos.dtype)
    heading = jnp.where(
        jnp.linalg.norm(target_velocity) <= nav.EPS,
        jnp.arctan2(target[1] - pos[1], target[0] - pos[0]),
        jnp.arctan2(target_velocity[1], target_velocity[0]),
    )
    tv, v = _rotate(target_velocity, -heading), _rotate(vel, -heading)
    t, p = _rotate(target, -heading), _rotate(pos, -heading)
    vx = _axis(t[0] - p[0], v[0], tv[0], t[0], p[0], max_velocity)
    vy = _axis(t[1] - p[1], v[1], tv[1], t[1], p[1], max_velocity)
    world = _rotate(jnp.stack([vx, vy]), heading)
    c, s = jnp.cos(angle), jnp.sin(angle)
    turn = smallest_angle_diff(angle, target_angle)
    w = jnp.where(jnp.abs(turn) < nav.MIN_ANGLE_TO_ROTATE, 0.0, k_p * turn)
    return jnp.stack([c * world[0] + s * world[1], -s * world[0] + c * world[1], w])


# ------------------------------------------------------------------ placement


def sample_pose(u, taken, margin: float = 0.3, clearance: float = 0.25):
    """`BaseEnv.sample_pose` without the loop: `u` is (N, 2) uniforms in [0, 1), N candidate
    (x, y) inside the field `margin` in from the lines; the first one farther than
    `clearance` from every row of `taken` (k, 2) wins, which is the CPU's rejection
    sampling exactly when N = 200. When none is free there is no raise: the candidate with
    the largest minimum distance comes back, with `overlap` True.

    Returns ((x, y), overlap).
    """
    lo = jnp.array([-scene.FIELD_HALF_X + margin, -scene.FIELD_HALF_Y + margin])
    hi = -lo
    cand = lo + u * (hi - lo)
    taken = jnp.asarray(taken, dtype=cand.dtype).reshape(-1, 2)
    if taken.shape[0] == 0:
        return cand[0], jnp.array(False)
    d2 = ((cand[:, None, :] - taken[None, :, :]) ** 2).sum(-1).min(1)
    free = d2 > clearance**2
    pick = jnp.where(free.any(), jnp.argmax(free), jnp.argmax(d2))
    return cand[pick], ~free.any()


def _default(fn, name):
    """A keyword default off a CPU signature, so the number lives in one place."""
    return inspect.signature(fn).parameters[name].default


# ------------------------------------------------------------------------ env


class MjxEnv(Env):
    """N robots and a ball, batched by `jax.vmap` over MJX-Warp. Subclass per task.

    A task names the CPU class it mirrors (`cpu_cls`) and every config number comes off
    it. It supplies `place` (+ `n_uniform`, the uniforms it draws), optionally
    `reset_physics` and `shape_drive`/`script`, `init_task` and `reward_done`.
    """

    cpu_cls: type[cpu_base.BaseEnv] = cpu_base.BaseEnv
    n_uniform = 0

    def __init__(self, horizon_s: float = 30.0, repeat_action: int = 1,
                 graph_mode: str = "WARP", randomize=None, nworld: int = 1) -> None:
        if randomize is not None:
            raise NotImplementedError("domain randomisation lands in ticket 14")
        if graph_mode not in GRAPH_MODES:
            raise ValueError(f"graph_mode={graph_mode!r}; expected one of {GRAPH_MODES}")
        cpu = self.cpu_cls(repeat_action=repeat_action, horizon_s=horizon_s)
        self.mj_model, self.robots, self.ball = cpu.model, cpu.robots, cpu.ball
        self.n_blue, self.n_controlled = cpu.n_blue, cpu.n_controlled
        self.decimation, self.dt = cpu.decimation, cpu.dt
        self.repeat_action, self.max_steps = cpu.repeat_action, cpu.max_steps
        self.grace_steps = cpu._grace_steps
        self.act_dim, self._obs_size = cpu.act_dim, cpu.observation_space.shape[0]
        cpu.close()
        self.graph_mode = graph_mode
        self.nworld = int(nworld)
        n = len(self.robots)
        self.nacon_per_world = max(NACON_MIN, NACON_PER_ROBOT * n)
        self.njmax = NJMAX_PER_CONTACT * self.nacon_per_world

        r = self.robots
        self._body = np.array([x.body for x in r])
        self._qvel = np.array([x.qvel_adr for x in r])
        self._touch = np.array([x.touch_adr for x in r])

    # ---------------------------------------------------------------- engine

    @functools.cached_property
    def sys(self) -> mjx.Model:
        """The base model on MJX-Warp. Lazy: the task logic alone (the parity test, on
        JAX-CPU) never needs it. Built eagerly even when first asked for inside a trace,
        so no tracer leaks into the cache."""
        from mujoco.mjx.warp import types as wt

        with jax.ensure_compile_time_eval():
            return mjx.put_model(self.mj_model, impl="warp",
                                 graph_mode=getattr(wt.GraphMode, self.graph_mode))

    def make_data(self) -> mjx.Data:
        return mjx.make_data(self.mj_model, impl="warp",
                             naconmax=self.nworld * self.nacon_per_world, njmax=self.njmax)

    def physics(self, data: mjx.Data, n: int | None = None) -> mjx.Data:
        """`n` (default: one control step) physics steps."""
        n = self.decimation if n is None else n
        return jax.lax.fori_loop(0, n, lambda _, d: mjx.step(self.sys, d), data)

    # --------------------------------------------------------------- brax API

    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)
        d = self.make_data()
        d = self.place(d, jax.random.uniform(key, (self.n_uniform,)))
        d = mjx.forward(self.sys, d)
        d = self.reset_physics(d)
        return self.init_state(d, rng)

    def step(self, state: State, action: jax.Array) -> State:
        """One decision: `repeat_action` control steps, rewards summed, cut short when the
        episode ends."""

        def control_step(s):
            s = self.control(s, action)
            return self.settle(s.replace(pipeline_state=self.physics(s.pipeline_state)))

        s = control_step(state)
        if self.repeat_action == 1:
            return s

        # A world that ended keeps its final state through the rest of the decision. Not a
        # while_loop: under vmap its per-world predicate would batch Warp's shared contact
        # buffer, so every world runs every control step and the ended ones are put back
        # with Data.where -- the one merge Warp's buffers allow.
        def body(_, c):
            s, total = c
            new = control_step(s)
            ended = s.done > 0
            merged = jax.tree.map(lambda a, b: jnp.where(ended, a, b),
                                  s.replace(pipeline_state=None), new.replace(pipeline_state=None))
            merged = merged.replace(pipeline_state=new.pipeline_state.where(ended, s.pipeline_state))
            return merged, total + jnp.where(ended, 0.0, new.reward)

        s, total = jax.lax.fori_loop(1, self.repeat_action, body, (s, s.reward))
        return s.replace(reward=total)

    @property
    def observation_size(self) -> int:
        return self._obs_size

    @property
    def action_size(self) -> int:
        return self.n_controlled * self.act_dim

    @property
    def backend(self) -> str:
        return "mjx-warp"

    # ---------------------------------------------------------- the 3 pieces

    def init_state(self, data: mjx.Data, rng: jax.Array) -> State:
        """The first State, from a placed and settled world (`BaseEnv.reset`'s tail)."""
        world = dict(
            grace=jnp.zeros(len(self.robots), jnp.int32),
            steps=jnp.zeros((), jnp.int32),
            drive_cmd=jnp.zeros((self.n_controlled, 3)),
            rng=rng,
            **self.init_task(data),
        )
        world["grace"] = self._possession(data, world["grace"])
        zero = jnp.zeros(())
        return State(data, self._obs(data, world), zero, zero, {},
                     {"world": world, "truncation": zero})

    def control(self, state: State, action: jax.Array) -> State:
        """`BaseEnv._apply` + `_script`: the controlled robots' target poses through the
        navigation controller, and their dribblers, into a zeroed `ctrl`."""
        a = jnp.clip(action, -1.0, 1.0).reshape(self.n_controlled, self.act_dim)
        d = state.pipeline_state
        ctrl = jnp.zeros_like(d.ctrl)
        cmds = []
        for i in range(self.n_controlled):
            r = self.robots[i]
            cmd = self.shape_drive(self._navigate(d, i, a[i, :4]))
            cmds.append(jnp.clip(cmd / r.drive_range[:, 1], -1.0, 1.0))
            # Body-frame command to the world-axis slides, as BaseEnv._drive_robot.
            v = d.xmat[r.body] @ jnp.stack([cmd[0], cmd[1], 0.0])
            ctrl = ctrl.at[jnp.array(r.drive)].set(jnp.stack([v[0], v[1], cmd[2]]))
            ctrl = ctrl.at[r.dribble].set(scene.DRIBBLER_OMEGA * jnp.maximum(a[i, -1], 0.0))
        ctrl = self.script(d, ctrl)
        world = {**state.info["world"], "drive_cmd": jnp.stack(cmds)}
        return state.replace(pipeline_state=d.replace(ctrl=ctrl),
                             info={**state.info, "world": world})

    def settle(self, state: State) -> State:
        """`BaseEnv._control_step`'s tail, on data that has just been stepped."""
        d = state.pipeline_state
        w = dict(state.info["world"])
        w["steps"] = w["steps"] + 1
        w["grace"] = self._possession(d, w["grace"])
        obs = self._obs(d, w)
        reward, terminated, w = self.reward_done(d, w)
        truncated = (w["steps"] >= self.max_steps) & ~terminated
        return state.replace(
            obs=obs,
            reward=jnp.asarray(reward, jnp.float32),
            done=(terminated | truncated).astype(jnp.float32),
            info={**state.info, "world": w, "truncation": truncated.astype(jnp.float32)},
        )

    # ------------------------------------------------------------- helpers

    def _navigate(self, d, i, pose):
        r = self.robots[i]
        q = d.qpos[r.qpos_adr : r.qpos_adr + 3]
        pos = q[:2] + jnp.array(r.xy0)
        target = pose[:2] * jnp.array([scene.FIELD_HALF_X, scene.FIELD_HALF_Y])
        return go_to_point(pos, d.qvel[r.qvel_adr : r.qvel_adr + 2], q[2], target,
                           jnp.arctan2(pose[2], pose[3]))

    def _possession(self, d, grace):
        """`BaseEnv._update_possession`: a dribbler on the ball, per its contact sensors,
        restarts that robot's grace; silence counts it down."""
        touching = (d.sensordata[self._touch] != 0).any(-1)
        return jnp.where(touching, self.grace_steps, jnp.maximum(grace - 1, 0)).astype(jnp.int32)

    def _obs(self, d, world):
        bp = d.xpos[self.ball.body]
        bv = d.qvel[self.ball.qvel_adr : self.ball.qvel_adr + 3]
        p, R = d.xpos[self._body], d.xmat[self._body]
        th = jnp.arctan2(R[:, 1, 0], R[:, 0, 0])
        v = d.qvel[self._qvel[:, None] + jnp.arange(3)]
        robots = jnp.stack([
            p[:, 0] / scene.FIELD_HALF_X, p[:, 1] / scene.FIELD_HALF_Y, jnp.sin(th), jnp.cos(th),
            v[:, 0] / cpu_base.MAX_V, v[:, 1] / cpu_base.MAX_V, v[:, 2] / cpu_base.MAX_W,
            (world["grace"] > 0).astype(p.dtype),
        ], axis=1)
        ball = jnp.concatenate([
            jnp.stack([bp[0] / scene.FIELD_HALF_X, bp[1] / scene.FIELD_HALF_Y,
                       (bp[2] - scene.FLOOR_Z) / scene.GOAL_HEIGHT]),
            bv / cpu_base.MAX_BALL_V,
        ])
        return jnp.concatenate([ball, robots.ravel(), self.task_obs(d, world)]).astype(jnp.float32)

    def xy(self, d, i):
        return d.xpos[self.robots[i].body, :2]

    def ball_xy(self, d):
        return d.xpos[self.ball.body, :2]

    @staticmethod
    def place_robot(d, r: scene.RobotIds, x, y, theta):
        """`RobotIds.place`: the slides are relative to the body's built position."""
        a, v = r.qpos_adr, r.qvel_adr
        qpos = d.qpos.at[a : a + 3].set(jnp.stack([x - r.xy0[0], y - r.xy0[1], theta]))
        return d.replace(qpos=qpos, qvel=d.qvel.at[v : v + 3].set(0.0))

    def seat_ball(self, d, r: scene.RobotIds):
        """`BaseEnv.seat_ball`, the same physical loop: roll the ball into the dribbler at
        `speed` from `distance` with the bar spinning, and let it settle."""
        seat = cpu_base.BaseEnv.seat_ball
        dist, speed = _default(seat, "distance"), _default(seat, "speed")
        settle = _default(seat, "settle")
        f = d.xmat[r.body][:, 0]
        p = d.xpos[r.body]
        a, dv = self.ball.qpos_adr, self.ball.qvel_adr
        qpos = d.qpos.at[a : a + 2].set(p[:2] + f[:2] * dist)
        qpos = qpos.at[a + 3 : a + 7].set(jnp.array([1.0, 0.0, 0.0, 0.0]))
        qvel = d.qvel.at[dv : dv + 6].set(0.0).at[dv : dv + 2].set(-f[:2] * speed)
        d = mjx.forward(self.sys, d.replace(qpos=qpos, qvel=qvel))
        d = d.replace(ctrl=d.ctrl.at[r.dribble].set(scene.DRIBBLER_OMEGA))
        return self.physics(d, round(settle / self.mj_model.opt.timestep))

    # ------------------------------------------------------ subclass surface

    def place(self, d: mjx.Data, u: jax.Array) -> mjx.Data:
        """`_place`, with its `n_uniform` draws passed in (`u` in [0, 1))."""
        raise NotImplementedError

    def reset_physics(self, d: mjx.Data) -> mjx.Data:
        """The physical half of `_reset_task` (seating a ball), after placement."""
        return d

    def init_task(self, d: mjx.Data) -> dict:
        """The task's per-world state at reset (`_reset_task`'s bookkeeping)."""
        return {}

    def shape_drive(self, cmd):
        return cmd

    def script(self, d, ctrl):
        return ctrl

    def task_obs(self, d, world):
        return jnp.zeros((0,), d.qpos.dtype)

    def reward_done(self, d, world):
        """(reward, terminated bool, world) for the control step just taken."""
        raise NotImplementedError
