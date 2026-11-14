"""PROTOTYPE (branch prototype/mjlab, throwaway): ContestedPossession-v0 on mjlab, with the kick.

Answers .scratch/mjlab-port/decisions.md item 13 (throughput, memory, idle world, overflow);
no training, no parity. Mirrors `mjsoccer.mjx.base` + `mjx.tasks.ContestedPossessionEnv` in
torch, following decisions 10-12:

- firing step: `decimation = 48 * repeat_action` at `timestep = 1/960`; `opt.timestep` swapped
  for a per-world `wp.array(nworld)` + `sim.create_graph()`, and the schedule (32 x dt/4 + 16 x dt
  when the world fired, else 24 x dt + 24 x 0) written in place every substep by the action term;
- scene: the whole `scene.build_spec(1, 1)` attached unprefixed in `SceneCfg.spec_fn`, same
  names as the CPU; the action term points at an empty anchor entity;
- reset: `seat_pool()` seats the ball in every world at once (the same physical loop as
  `seat_ball`) and keeps the result as each world's first state; the reset event copies it back.

Horizon, reward and time-out are ours (`scale_rewards_by_dt=False`); repeat_action is 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np
import torch
import warp as wp
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.utils.spec import attach_unprefixed

from . import base as cpu_base
from . import navigation as nav
from . import scene
from . import tasks as cpu_tasks
from .mjx.base import NACON_MIN, NACON_PER_ROBOT, NJMAX_PER_CONTACT

CPU = cpu_tasks.ContestedPossessionEnv
HALF_X, HALF_Y = scene.FIELD_HALF_X, scene.FIELD_HALF_Y
TEAMS = [("blue", "b0_"), ("yellow", "y0_")]


# ------------------------------------------------------------ navigation (torch, batched)


def _angle_diff(a, b):
    mod = 2 * math.pi
    ang = b - a
    ang = torch.where((ang < -mod) | (ang > mod), torch.fmod(ang, mod), ang)
    ang = torch.where(ang < 0, ang + mod, ang)
    return torch.where(ang >= math.pi, ang - mod, torch.where(ang < -math.pi, ang + mod, ang))


def _rot(x, y, ang):
    c, s = torch.cos(ang), torch.sin(ang)
    return c * x - s * y, s * x + c * y


def _axis(to_target, v_cur, v_des, target, pos, vmax):
    brake = (to_target * v_des < 0) | (to_target * v_cur < 0)
    zero_point = target - torch.sign(v_des) * v_des**2 / (2 * nav.MAX_ACCEL)
    to_target = torch.where(brake, zero_point - pos, to_target)
    v_des = torch.where(brake, 0.0, v_des)
    braking = to_target.abs() < (v_cur**2 - v_des**2).abs() / (2 * nav.MAX_ACCEL)
    v = v_cur + torch.where(braking, torch.sign(v_des - v_cur), torch.sign(to_target)) * nav.MAX_DV
    return torch.where(v.abs() > vmax, torch.sign(v) * vmax, v)


def go_to_point(pos, vel, angle, target, target_angle):
    """`mjx.base.go_to_point` with target_velocity 0, over a batch: (N, 2) x4 -> (N, 3)."""
    heading = torch.atan2(target[:, 1] - pos[:, 1], target[:, 0] - pos[:, 0])
    zero = torch.zeros_like(heading)
    vx0, vy0 = _rot(vel[:, 0], vel[:, 1], -heading)
    tx, ty = _rot(target[:, 0], target[:, 1], -heading)
    px, py = _rot(pos[:, 0], pos[:, 1], -heading)
    vx = _axis(tx - px, vx0, zero, tx, px, nav.MAX_VELOCITY)
    vy = _axis(ty - py, vy0, zero, ty, py, nav.MAX_VELOCITY)
    wx, wy = _rot(vx, vy, heading)
    c, s = torch.cos(angle), torch.sin(angle)
    turn = _angle_diff(angle, target_angle)
    w = torch.where(turn.abs() < nav.MIN_ANGLE_TO_ROTATE, 0.0, nav.ANGLE_KP * turn)
    return torch.stack([c * wx + s * wy, -s * wx + c * wy, w], 1)


# ------------------------------------------------------------------------ the env


class Task:
    """Per-world task state and the logic of `MjxEnv.control/settle` + `_ShootTask`, in torch.
    Lives on the env (`env.task`); every manager term below is a thin call into it."""

    def __init__(self, env: ManagerBasedRlEnv) -> None:
        self.env, sim = env, env.sim
        n, dev = env.num_envs, env.device
        self.n = n
        m = sim.mj_model
        _, self.robots, self.ball = scene._resolve(m, TEAMS)
        cpu = CPU(horizon_s=30.0)
        self.decimation, self.max_steps = cpu.decimation, cpu.max_steps
        self.grace_steps, self.cooldown = cpu._grace_steps, cpu._kick_cooldown
        cpu.close()
        assert env.cfg.decimation == 2 * self.decimation, "48 iterations per control step"
        self.dt = float(m.opt.timestep)

        # Per-world opt.timestep: replace the shared array, recapture the graphs (decision 10).
        old = sim.wp_model.opt.timestep
        sim.wp_model.opt.timestep = wp.array(np.full(n, self.dt), dtype=old.dtype, device=old.device)
        sim.create_graph()
        self.timestep = wp.to_torch(sim.wp_model.opt.timestep)
        fine = scene.FIRING_FINE_STEPS
        firing = [self.dt / scene.FIRING_SUBSTEPS] * fine
        firing += [self.dt] * (self.decimation - fine // scene.FIRING_SUBSTEPS)
        idle = [self.dt] * self.decimation + [0.0] * (len(firing) - self.decimation)
        assert len(firing) == len(idle) == env.cfg.decimation
        self.sched = torch.tensor([idle, firing], device=dev, dtype=self.timestep.dtype)  # (2, 48)

        r0, r1 = self.robots
        self.r0, self.r1 = r0, r1
        self.touch = torch.tensor([r.touch_adr for r in self.robots], device=dev)
        self.body = torch.tensor([r.body for r in self.robots], device=dev)
        self.qv = torch.tensor([r.qvel_adr for r in self.robots], device=dev)
        self.kick_n = torch.tensor(scene.KICK_TABLE_N, device=dev)
        self.kick_ms = [torch.tensor(t, device=dev) for t in scene.KICK_TABLE_MS]
        self.drive_max = torch.tensor(r0.drive_range[:, 1], device=dev, dtype=torch.float32)

        z = lambda *s, dt=torch.float32: torch.zeros((n, *s), device=dev, dtype=dt)  # noqa: E731
        self.lock = z(2, dt=torch.int32)  # robot 0's (kick, chip); the yellow never fires
        self.fire = z(dt=torch.bool)
        self.grace = z(2, dt=torch.int32)
        self.drive_cmd = z(3)
        self.prev, self.prev_goal = z(), z()
        self.reward, self.terminated = z(), z(dt=torch.bool)
        self.overflow_seen = z(dt=torch.int32)  # OR of overflow bits since the bench started
        self.ctrl = z(m.nu)
        self.k = 0
        self.pool = None

    # ------------------------------------------------------------- helpers

    def xy(self, r):
        return self.env.sim.data.xpos[:, r.body, :2]

    def ball_xy(self):
        return self.env.sim.data.xpos[:, self.ball.body, :2]

    def possession(self, grace):
        touching = (self.env.sim.data.sensordata[:, self.touch] != 0).any(-1)  # (N, 2)
        return torch.where(touching, self.grace_steps, (grace - 1).clamp(min=0)).int()

    def task_init(self):
        """prev / prev_goal / grace off the current (forwarded) data."""
        to_ball = (self.ball_xy() - self.xy(self.r0)).norm(dim=-1)
        goal = (self.ball_xy() - torch.tensor([HALF_X, 0.0], device=self.env.device)).norm(dim=-1)
        return to_ball, goal, self.possession(torch.zeros_like(self.grace))

    # ---------------------------------------------------------- reset pool

    def seat_pool(self, generator: torch.Generator | None = None) -> float:
        """Decision 12: every world's first state, placed and seated in one batched loop --
        `ContestedPossessionEnv.place` + `seat_ball(yellow)`, `settle` s of plain dt steps.
        Clobbers every world; run it, then `env.reset()`. Returns the wall time."""
        sim, d, n = self.env.sim, self.env.sim.data, self.n
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0[0].record()
        sim.reset()
        u = torch.rand(n, 2, device=self.env.device, generator=generator)
        ox = scene.PENALTY_DEPTH + u[:, 0] * (HALF_X - 2 * scene.PENALTY_DEPTH)
        oy = -scene.PENALTY_HALF_Y + u[:, 1] * (2 * scene.PENALTY_HALF_Y)
        qpos = d.qpos[:].clone()
        for r, x, y, th in ((self.r0, 0.0, 0.0, 0.0), (self.r1, ox, oy, math.pi)):
            a = r.qpos_adr
            qpos[:, a] = x - r.xy0[0]
            qpos[:, a + 1] = y - r.xy0[1]
            qpos[:, a + 2] = th
        d.qpos[:] = qpos
        d.qvel[:] = 0.0
        sim.forward()
        seat = cpu_base.BaseEnv.seat_ball.__defaults__  # (distance, speed, settle)
        dist, speed, settle = seat
        f = d.xmat[:, self.r1.body, :, 0]  # (N, 3) the yellow robot's forward axis
        p = d.xpos[:, self.r1.body]
        a, v = self.ball.qpos_adr, self.ball.qvel_adr
        qpos = d.qpos[:].clone()
        qpos[:, a : a + 2] = p[:, :2] + f[:, :2] * dist
        qpos[:, a + 3 : a + 7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=qpos.device)
        qvel = d.qvel[:].clone()
        qvel[:, v : v + 6] = 0.0
        qvel[:, v : v + 2] = -f[:, :2] * speed
        d.qpos[:], d.qvel[:] = qpos, qvel
        sim.forward()
        ctrl = torch.zeros_like(d.ctrl[:])
        ctrl[:, self.r1.dribble] = scene.DRIBBLER_OMEGA
        d.ctrl[:] = ctrl
        self.timestep.fill_(self.dt)
        for _ in range(round(settle / self.dt)):
            sim.step()
        sim.forward()
        prev, prev_goal, grace = self.task_init()
        self.pool = dict(qpos=d.qpos[:].clone(), qvel=d.qvel[:].clone(), prev=prev, prev_goal=prev_goal,
                         grace=grace)
        t0[1].record()
        torch.cuda.synchronize()
        return t0[0].elapsed_time(t0[1]) / 1000

    def reset(self, env_ids):
        """The reset event: the world's pooled first state, the task's bookkeeping zeroed."""
        if self.pool is None:  # mjlab's own reset during __init__, before the pool exists
            return
        d, p = self.env.sim.data, self.pool
        d.qpos[env_ids] = p["qpos"][env_ids]
        d.qvel[env_ids] = p["qvel"][env_ids]
        self.prev[env_ids], self.prev_goal[env_ids] = p["prev"][env_ids], p["prev_goal"][env_ids]
        self.grace[env_ids] = p["grace"][env_ids]
        self.lock[env_ids] = 0
        self.drive_cmd[env_ids] = 0.0

    # ---------------------------------------------------------- the 3 pieces

    def control(self, action):
        """`MjxEnv.control` for robot 0 (the yellow robot gets ctrl 0). Once per decision."""
        d, r = self.env.sim.data, self.r0
        a = action.clamp(-1.0, 1.0)
        q = d.qpos[:, r.qpos_adr : r.qpos_adr + 3]
        pos = q[:, :2] + torch.tensor(r.xy0, device=a.device)
        target = a[:, :2] * torch.tensor([HALF_X, HALF_Y], device=a.device)
        cmd = go_to_point(pos, d.qvel[:, r.qvel_adr : r.qvel_adr + 2], q[:, 2], target,
                          torch.atan2(a[:, 2], a[:, 3]))
        self.drive_cmd = (cmd / self.drive_max).clamp(-1, 1)
        R = d.xmat[:, r.body]  # (N, 3, 3)
        vxy = R[:, :2, :2] @ cmd[:, :2, None]
        ctrl = torch.zeros_like(self.ctrl)
        ctrl[:, list(r.drive)] = torch.cat([vxy[:, :, 0], cmd[:, 2:]], 1)
        fire = torch.zeros_like(self.fire)
        for j, act in enumerate((r.kick, r.chip)):
            go = (a[:, -3 + j] > 0) & (self.lock[:, j] == 0)
            ms = self.kick_ms[j]
            force = _interp(scene.KICK_SPEED_MAX * a[:, -3 + j], ms, self.kick_n[: len(ms)])
            ctrl[:, act] = torch.where(go, force, 0.0)
            self.lock[:, j] = torch.where(go, self.cooldown, self.lock[:, j])
            fire |= go
        ctrl[:, r.dribble] = scene.DRIBBLER_OMEGA * a[:, -1].clamp(min=0)
        self.ctrl, self.fire = ctrl, fire
        self.k = 0

    def substep(self):
        """Every physics iteration: ctrl, and this iteration's per-world timestep, in place."""
        if self.k == 0:
            self.env.sim.data.ctrl[:] = self.ctrl
        self.timestep.copy_(self.sched[self.fire.long(), self.k])
        self.k += 1

    def settle(self):
        """`MjxEnv.settle` + `_ShootTask.reward_done`, on the stepped data. Terminated (N,)."""
        d = self.env.sim.data
        self.lock = (self.lock - 1).clamp(min=0)
        self.grace = self.possession(self.grace)
        (rx, ry), (bx, by) = self.xy(self.r0).unbind(-1), self.ball_xy().unbind(-1)
        dist, g, _ = self.task_init()
        yv = d.qvel[:, self.r1.qvel_adr : self.r1.qvel_adr + 2]
        collision = (yv.abs().amax(-1) > CPU.collision_speed)
        robot_out = (rx < -0.2) | (ry.abs() > HALF_Y)
        gk = (rx > HALF_X - scene.PENALTY_DEPTH) & (ry.abs() < scene.PENALTY_HALF_Y)
        ball_out = (bx < 0) | (by.abs() > HALF_Y)
        past = bx > HALF_X
        energy = self.drive_cmd.abs().mean(-1) / CPU.energy_steps
        shaping = ((self.prev - dist).clamp(-1, 1) / cpu_tasks.BALL_DIST_SCALE
                   + (self.prev_goal - g).clamp(-1, 1) / cpu_tasks.BALL_GRAD_SCALE - energy)
        bz = d.xpos[:, self.ball.body, 2]
        in_goal = past & (by.abs() < scene.GOAL_HALF_Y) & (bz - scene.FLOOR_Z < scene.GOAL_HEIGHT)
        goal = torch.where(in_goal, cpu_tasks.GOAL_REWARD, 0.0)
        reward = torch.where(robot_out | gk | ball_out, 0.0, torch.where(past, goal, shaping))
        bits = d.overflow[:]
        self.overflow_seen |= bits
        diverged = ~(torch.isfinite(d.qpos[:]).all(-1) & torch.isfinite(d.qvel[:]).all(-1))
        failed = diverged | (bits != 0)
        self.prev, self.prev_goal = dist, g
        self.reward = torch.where(failed, 0.0, reward)
        self.terminated = collision | robot_out | gk | ball_out | past | failed
        return self.terminated

    def obs(self):
        d = self.env.sim.data
        bp = d.xpos[:, self.ball.body]
        bv = d.qvel[:, self.ball.qvel_adr : self.ball.qvel_adr + 3]
        p, R = d.xpos[:, self.body], d.xmat[:, self.body]
        th = torch.atan2(R[..., 1, 0], R[..., 0, 0])
        v = d.qvel[:, self.qv[:, None] + torch.arange(3, device=self.qv.device)]  # (N, 2, 3)
        robots = torch.stack([
            p[..., 0] / HALF_X, p[..., 1] / HALF_Y, torch.sin(th), torch.cos(th),
            v[..., 0] / cpu_base.MAX_V, v[..., 1] / cpu_base.MAX_V, v[..., 2] / cpu_base.MAX_W,
            (self.grace > 0).float(),
        ], -1)
        ball = torch.cat([torch.stack([bp[:, 0] / HALF_X, bp[:, 1] / HALF_Y,
                                       (bp[:, 2] - scene.FLOOR_Z) / scene.GOAL_HEIGHT], 1),
                          bv / cpu_base.MAX_BALL_V], 1)
        return torch.cat([ball, robots.flatten(1)], 1).float()


def _interp(x, xp, fp):
    """np.interp (clamped at the ends) on a batch."""
    i = torch.searchsorted(xp, x.contiguous()).clamp(1, len(xp) - 1)
    x0, x1, f0, f1 = xp[i - 1], xp[i], fp[i - 1], fp[i]
    y = f0 + (x - x0) * (f1 - f0) / (x1 - x0)
    return torch.where(x <= xp[0], fp[0], torch.where(x >= xp[-1], fp[-1], y))


# --------------------------------------------------------------- manager terms


class NavKickAction(ActionTerm):
    """The 7-dim action: target pose (4) through the controller, kick, chip, dribble."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)  # entity_name -> the empty anchor (decision 11)
        self._raw = torch.zeros(env.num_envs, 7, device=env.device)

    @property
    def action_dim(self):
        return 7

    @property
    def raw_action(self):
        return self._raw

    def process_actions(self, actions):
        self._raw[:] = actions
        self._env.task.control(actions)

    def apply_actions(self):
        self._env.task.substep()


@dataclass(kw_only=True)
class NavKickActionCfg(ActionTermCfg):
    entity_name: str = "anchor"

    def build(self, env):
        return NavKickAction(self, env)


def _settle(env):
    return env.task.settle()


def _time_out(env):
    return (env.episode_length_buf >= env.task.max_steps) & ~env.task.terminated


def _reward(env):
    return env.task.reward


def _obs(env):
    return env.task.obs()


def _reset(env, env_ids):
    env.task.reset(env_ids)


class ProtoEnv(ManagerBasedRlEnv):
    """mjlab's env with our `Task` hung on it. `task` exists before the managers load, since
    the reset event and the obs term (sized at load time) read it."""

    def load_managers(self):
        self.task = Task(self)
        super().load_managers()


def _spec_fn(spec: mujoco.MjSpec) -> None:
    """Decision 11: the CPU's exact base model, unprefixed (no Entity)."""
    attach_unprefixed(spec, scene.build_spec(1, 1), spec.worldbody.add_frame())


def make_cfg(nworld: int, nconmax: int | None = None) -> ManagerBasedRlEnvCfg:
    nconmax = nconmax or max(NACON_MIN, NACON_PER_ROBOT * 2)  # the MJX sizing, per world
    return ManagerBasedRlEnvCfg(
        decimation=48,
        scene=SceneCfg(num_envs=nworld, entities={"anchor": EntityCfg(spec_fn=mujoco.MjSpec)},
                       spec_fn=_spec_fn),
        sim=SimulationCfg(nconmax=nconmax, njmax=NJMAX_PER_CONTACT * nconmax,
                          mujoco=MujocoCfg(timestep=1 / 960, cone="elliptic")),
        observations={"actor": ObservationGroupCfg(terms={"obs": ObservationTermCfg(func=_obs)})},
        actions={"nav": NavKickActionCfg()},
        events={"pool": EventTermCfg(func=_reset, mode="reset")},
        rewards={"task": RewardTermCfg(func=_reward, weight=1.0)},
        terminations={"settle": TerminationTermCfg(func=_settle),
                      "time_out": TerminationTermCfg(func=_time_out, time_out=True)},
        episode_length_s=1e9,  # ours: Task.max_steps
        scale_rewards_by_dt=False,
    )


def make(nworld: int, device: str = "cuda:0", **kw) -> ProtoEnv:
    env = ProtoEnv(make_cfg(nworld, **kw), device=device)
    env.task.seat_pool()
    env.reset()
    return env


def warp_stream():
    """Run torch on Warp's stream so our in-place writes and the graph launches are ordered."""
    return torch.cuda.stream(torch.cuda.ExternalStream(wp.get_stream("cuda:0").cuda_stream))


_ = field  # dataclasses.field kept for cfg edits
