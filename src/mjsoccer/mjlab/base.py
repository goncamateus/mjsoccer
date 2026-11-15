"""The torch task base: navigation controller, kicker, possession, observation, bookkeeping.

A line-by-line port of `mjsoccer.mjx.base`, batched over a leading world axis instead of
vmapped. Each control step is the same three pieces, with the physics left to the caller:

    control(d, world, action)   ctrl (N, nu) from the controller, kicker, chipper, dribbler
    <physics>                   one control step on the firing schedule (the env's job)
    settle(d, world, overflow)  possession, observation, reward, ending, on the stepped data

`d` is anything with batched torch tensors `qpos` (N, nq), `qvel` (N, nv), `xpos`
(N, nbody, 3), `xmat` (N, nbody, 3, 3) and `sensordata` (N, nsensordata). `world` is a dict
of per-world tensors: `grace` (N, n_robots) int32, `plunger_lock` (N, n_robots, 2) int32,
`fired` (N, n_robots) bool, `steps` (N,) int32 in control steps, `drive_cmd`
(N, n_controlled, 3), `overlap` (N,) bool and the task's own entries. Every function returns
new tensors and leaves its inputs alone. Randomness comes in as arguments (`place`'s
uniforms), so the per-world RNG, like the physics, lives with the env.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .. import base as cpu_base
from .. import navigation as nav
from .. import scene

# ------------------------------------------------------------ navigation controller


def smallest_angle_diff(a, b):
    """`navigation.smallest_angle_diff`: b - a wrapped to [-pi, pi), fmod then shift."""
    mod = 2 * math.pi
    angle = b - a
    angle = torch.where((angle < -mod) | (angle > mod), torch.fmod(angle, mod), angle)
    angle = torch.where(angle < 0, angle + mod, angle)
    return torch.where(angle >= math.pi, angle - mod,
                       torch.where(angle < -math.pi, angle + mod, angle))


def _rotate(x, y, angle):
    """`navigation._rotate`, rounded as its `np.dot` rounds (one fused multiply-add per
    row). The lateral axis's to-target is identically 0 up to that rounding and the
    controller takes its sign (`_axis`), so a plain `c * x - s * y` (what `mjsoccer.mjx`
    writes, and needs x64 for) would pick the other sign than the CPU on ~1 at-rest step
    in 4; this matches it in float64. In float32 the sign is noise either way."""
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.addcmul(-s * y, c, x), torch.addcmul(c * y, s, x)


def _axis(to_target, v_current, v_desired, target, pos, max_velocity):
    """`navigation._axis`, branch for branch."""
    brake_at = (to_target * v_desired < 0) | (to_target * v_current < 0)
    zero_point = target - torch.sign(v_desired) * v_desired**2 / (2 * nav.MAX_ACCEL)
    to_target = torch.where(brake_at, zero_point - pos, to_target)
    v_desired = torch.where(brake_at, 0.0, v_desired)
    braking = to_target.abs() < (v_current**2 - v_desired**2).abs() / (2 * nav.MAX_ACCEL)
    v = v_current + torch.where(braking, torch.sign(v_desired - v_current),
                                torch.sign(to_target)) * nav.MAX_DV
    return torch.where(v.abs() > max_velocity, torch.sign(v) * max_velocity, v)


def go_to_point(pos, vel, angle, target, target_angle):
    """`navigation.go_to_point` over a batch: pos, vel, target (N, 2), angle and
    target_angle (N,) -> body-frame (vx, vy, omega) as (N, 3)."""
    # ponytail: target velocity is always 0 here (the action space has none), so the
    # heading is the robot->target bearing and the desired axis speeds are 0.
    heading = torch.atan2(target[:, 1] - pos[:, 1], target[:, 0] - pos[:, 0])
    zero = torch.zeros_like(heading)
    vx0, vy0 = _rotate(vel[:, 0], vel[:, 1], -heading)
    tx, ty = _rotate(target[:, 0], target[:, 1], -heading)
    px, py = _rotate(pos[:, 0], pos[:, 1], -heading)
    vx = _axis(tx - px, vx0, zero, tx, px, nav.MAX_VELOCITY)
    vy = _axis(ty - py, vy0, zero, ty, py, nav.MAX_VELOCITY)
    wx, wy = _rotate(vx, vy, heading)
    c, s = torch.cos(angle), torch.sin(angle)
    turn = smallest_angle_diff(angle, target_angle)
    w = torch.where(turn.abs() < nav.MIN_ANGLE_TO_ROTATE, 0.0, nav.ANGLE_KP * turn)
    return torch.stack([c * wx + s * wy, -s * wx + c * wy, w], -1)


def interp(x, xp, fp):
    """`np.interp` on a batch `x`, `xp` increasing: clamped to fp's ends outside xp."""
    i = torch.searchsorted(xp, x.contiguous()).clamp(1, len(xp) - 1)
    x0, x1, f0, f1 = xp[i - 1], xp[i], fp[i - 1], fp[i]
    y = f0 + (x - x0) * (f1 - f0) / (x1 - x0)
    return torch.where(x <= xp[0], fp[0], torch.where(x >= xp[-1], fp[-1], y))


# ------------------------------------------------------------------ placement


def sample_pose(u, taken, margin: float = 0.3, clearance: float = 0.25):
    """`BaseEnv.sample_pose` without the loop: `u` (N, K, 2) uniforms in [0, 1), K candidate
    (x, y) per world inside the field `margin` in from the lines; the first one farther than
    `clearance` from every row of `taken` (N, k, 2) wins, the CPU's rejection sampling
    exactly when K = 200. None free: the roomiest, with `overlap` True.

    Returns ((N, 2), overlap (N,))."""
    lo = torch.tensor([-scene.FIELD_HALF_X + margin, -scene.FIELD_HALF_Y + margin],
                      dtype=u.dtype, device=u.device)
    return first_free(lo + u * (-2 * lo), taken, clearance, strict=True)


def first_free(cand, taken, min_dist: float, strict: bool = False):
    """Per world, the first of the (N, K, 2) candidates `min_dist` or more (more than, if
    `strict`) from every row of `taken` (N, k, 2). None free: the candidate with the
    largest minimum distance, and `overlap` True. Returns ((N, 2), overlap (N,))."""
    n = cand.shape[0]
    rows = torch.arange(n, device=cand.device)
    if taken.shape[1] == 0:
        return cand[:, 0], torch.zeros(n, dtype=torch.bool, device=cand.device)
    d2 = ((cand[:, :, None] - taken[:, None].to(cand.dtype)) ** 2).sum(-1).amin(2)
    free = d2 > min_dist**2 if strict else d2 >= min_dist**2
    ok = free.any(1)
    pick = torch.where(ok, free.byte().argmax(1), d2.argmax(1))
    return cand[rows, pick], ~ok


def check_failures(metrics: dict, overflow=None) -> None:
    """Abort on the host if any world diverged or overflowed: `mjx.base.check_failures`
    for `settle`'s metrics; `overflow` (N,) are Warp's bits, named per type if given."""
    diverged = metrics["diverged"].reshape(-1) > 0
    over = metrics["overflow"].reshape(-1) > 0
    if not (diverged.any() or over.any()):
        return
    n, parts = diverged.numel(), []
    if diverged.any():
        parts.append(f"{int(diverged.sum())} of {n} world(s) diverged (qpos/qvel not finite)")
    if over.any():
        kinds = "bits already cleared by a reset"
        if overflow is not None:
            from mujoco_warp import OverflowType

            bits = overflow.reshape(-1).cpu()
            kinds = ", ".join(f"{t.name} in {int(((bits & t.value) != 0).sum())}"
                              for t in OverflowType if bool((bits & t.value).any())) or kinds
        parts.append(f"{int(over.sum())} of {n} world(s) overflowed Warp ({kinds}): "
                     "naconmax/njmax too small?")
    raise RuntimeError("mjlab: " + "; ".join(parts))


# ------------------------------------------------------------------- the task


class TaskLogic:
    """N robots and a ball, batched over a leading world axis. Subclass per task.

    A task names the CPU class it mirrors (`cpu_cls`) and every config number comes off
    it. It supplies `place` (+ `n_uniform`, the uniforms it draws), optionally `seat_robot`,
    `shape_drive`/`script`, `init_task`, `task_obs` and `reward_done`.
    """

    cpu_cls: type[cpu_base.BaseEnv] = cpu_base.BaseEnv
    n_uniform = 0
    seat_robot: int | None = None  # the robot `_reset_task` seats a ball on (the env's job)

    def __init__(self, horizon_s: float = 30.0, repeat_action: int = 1,
                 device: str | torch.device = "cpu") -> None:
        cpu = self.cpu_cls(repeat_action=repeat_action, horizon_s=horizon_s)
        self.robots, self.ball = cpu.robots, cpu.ball
        self.nu = cpu.model.nu
        self.n_blue, self.n_controlled = cpu.n_blue, cpu.n_controlled
        self.decimation, self.dt = cpu.decimation, cpu.dt
        self.repeat_action, self.max_steps = cpu.repeat_action, cpu.max_steps
        self.grace_steps = cpu._grace_steps
        self.kick_cooldown = cpu._kick_cooldown
        self.act_dim, self.obs_size = cpu.act_dim, cpu.observation_space.shape[0]
        cpu.close()
        self.device = torch.device(device)
        self._kick_ms = [self._t(ms, torch.float64) for ms in scene.KICK_TABLE_MS]
        self._kick_n = self._t(scene.KICK_TABLE_N, torch.float64)
        self._index()

    def _t(self, x, dtype=torch.long):
        return torch.as_tensor(np.asarray(x), dtype=dtype, device=self.device)

    def _index(self) -> None:
        r = self.robots
        self._body = self._t([x.body for x in r])
        self._qvel = self._t([x.qvel_adr for x in r])
        self._touch = self._t([x.touch_adr for x in r])

    def bind(self, model) -> None:
        """Take the robot and ball ids from another compilation of the same scene: mjlab's
        has an extra body (the action term's anchor), so every body id moves."""
        self.robots, self.ball = scene.resolve(model, self.n_blue, len(self.robots) - self.n_blue)
        self._index()

    @property
    def action_size(self) -> int:
        return self.n_controlled * self.act_dim

    # ---------------------------------------------------------- the 3 pieces

    def init_state(self, d, overlap=None):
        """The first (obs, world, metrics), from placed and settled worlds (`BaseEnv.reset`'s
        tail). `overlap` (N,): `place` found no free spot, `metrics["reset_overlap"]`."""
        n, dev = d.qpos.shape[0], self.device
        nr = len(self.robots)
        overlap = (torch.zeros(n, dtype=torch.bool, device=dev) if overlap is None
                   else overlap.to(torch.bool))
        world = dict(
            grace=torch.zeros(n, nr, dtype=torch.int32, device=dev),
            plunger_lock=torch.zeros(n, nr, 2, dtype=torch.int32, device=dev),
            fired=torch.zeros(n, nr, dtype=torch.bool, device=dev),
            steps=torch.zeros(n, dtype=torch.int32, device=dev),
            drive_cmd=torch.zeros(n, self.n_controlled, 3, dtype=d.qpos.dtype, device=dev),
            overlap=overlap,
            **self.init_task(d),
        )
        world["grace"] = self.possession(d, world["grace"])
        return self.obs(d, world), world, self.metrics(n, reset_overlap=overlap)

    def control(self, d, world, action, raw: bool = False):
        """`BaseEnv._apply` + `_script`: the controlled robots' target poses through the
        navigation controller, their kickers, chippers and dribblers, into a zeroed `ctrl`.
        `raw`: each robot's slice is (vx, vy, omega, kick, chip, dribble), straight to the
        servos, as `BaseEnv.step_raw`. Returns (ctrl (N, nu), world)."""
        n, dt = d.qpos.shape[0], d.qpos.dtype
        dim = 6 if raw else self.act_dim
        a = action.to(dt).clamp(-1.0, 1.0).reshape(n, self.n_controlled, dim)
        ctrl = torch.zeros(n, self.nu, dtype=dt, device=self.device)
        lock = world["plunger_lock"].clone()
        fired = torch.zeros_like(world["fired"])
        cmds = []
        for i in range(self.n_controlled):
            r = self.robots[i]
            rng = torch.as_tensor(r.drive_range[:, 1], dtype=dt, device=self.device)
            if raw:
                cmd = self.shape_drive(a[:, i, :3] * rng)
            else:
                cmd = self.shape_drive(self._navigate(d, i, a[:, i, :4]))
            cmds.append((cmd / rng).clamp(-1.0, 1.0))
            # Body-frame command to the world-axis slides, as BaseEnv._drive_robot.
            body = torch.stack([cmd[:, 0], cmd[:, 1], torch.zeros_like(cmd[:, 0])], -1)
            v = (d.xmat[:, r.body].to(dt) @ body[..., None])[..., 0]
            ctrl[:, list(r.drive)] = torch.stack([v[:, 0], v[:, 1], cmd[:, 2]], -1)
            # BaseEnv._actuate: a in (0, 1] on an unlocked plunger asks for that fraction of
            # KICK_SPEED_MAX, mapped to a one-step force by the CPU's table; it locks.
            for j, actuator in enumerate((r.kick, r.chip)):
                go = (a[:, i, -3 + j] > 0.0) & (lock[:, i, j] == 0)
                speeds = self._kick_ms[j].to(dt)
                force = interp(scene.KICK_SPEED_MAX * a[:, i, -3 + j], speeds,
                               self._kick_n[: len(speeds)].to(dt))
                ctrl[:, actuator] = torch.where(go, force, ctrl[:, actuator])
                lock[:, i, j] = torch.where(go, self.kick_cooldown, lock[:, i, j])
                fired[:, i] |= go
            ctrl[:, r.dribble] = scene.DRIBBLER_OMEGA * a[:, i, -1].clamp(min=0.0)
        ctrl = self.script(d, ctrl)
        world = {**world, "drive_cmd": torch.stack(cmds, 1), "plunger_lock": lock,
                 "fired": fired}
        return ctrl, world

    def settle(self, d, world, overflow=None):
        """`BaseEnv._control_step`'s tail, on data that has just been stepped. `overflow`
        (N,): Warp's per-world overflow bits, None for none.

        A world that diverged or overflowed ends here, terminal, with reward and obs zeroed
        so the NaN never reaches a gradient. Returns (obs, reward, terminated, truncated,
        world, metrics), all (N, ...)."""
        n = d.qpos.shape[0]
        w = dict(world)
        w["steps"] = w["steps"] + 1
        w["plunger_lock"] = (w["plunger_lock"] - 1).clamp(min=0)
        w["grace"] = self.possession(d, w["grace"])
        obs = self.obs(d, w)
        reward, terminated, w = self.reward_done(d, w)
        diverged = ~(torch.isfinite(d.qpos).all(-1) & torch.isfinite(d.qvel).all(-1))
        over = (torch.zeros(n, dtype=torch.bool, device=self.device) if overflow is None
                else overflow != 0)
        failed = diverged | over
        terminated = terminated | failed
        truncated = (w["steps"] >= self.max_steps) & ~terminated
        obs = torch.where(failed[:, None], 0.0, obs)
        reward = torch.where(failed, 0.0, reward.to(torch.float32))
        return obs, reward, terminated, truncated, w, self.metrics(n, diverged, over)

    # ------------------------------------------------------------- helpers

    def metrics(self, n, diverged=False, overflow=False, reset_overlap=False):
        """The per-control-step 0/1 counters, (N,) float each."""
        def f(x):
            return torch.as_tensor(x, dtype=torch.float32, device=self.device).expand(n)
        return {"diverged": f(diverged), "overflow": f(overflow),
                "reset_overlap": f(reset_overlap)}

    def _navigate(self, d, i, pose):
        r = self.robots[i]
        q = d.qpos[:, r.qpos_adr : r.qpos_adr + 3].to(pose.dtype)
        pos = q[:, :2] + torch.tensor(r.xy0, dtype=pose.dtype, device=self.device)
        target = pose[:, :2] * torch.tensor([scene.FIELD_HALF_X, scene.FIELD_HALF_Y],
                                            dtype=pose.dtype, device=self.device)
        vel = d.qvel[:, r.qvel_adr : r.qvel_adr + 2].to(pose.dtype)
        return go_to_point(pos, vel, q[:, 2], target, torch.atan2(pose[:, 2], pose[:, 3]))

    def possession(self, d, grace):
        """`BaseEnv._update_possession`: a dribbler on the ball, per its contact sensors,
        restarts that robot's grace; silence counts it down."""
        touching = (d.sensordata[:, self._touch] != 0).any(-1)
        return torch.where(touching, self.grace_steps, (grace - 1).clamp(min=0)).to(torch.int32)

    def obs(self, d, world):
        bp = d.xpos[:, self.ball.body]
        bv = d.qvel[:, self.ball.qvel_adr : self.ball.qvel_adr + 3]
        p, R = d.xpos[:, self._body], d.xmat[:, self._body]
        th = torch.atan2(R[..., 1, 0], R[..., 0, 0])
        v = d.qvel[:, self._qvel[:, None] + torch.arange(3, device=self.device)]
        robots = torch.stack([
            p[..., 0] / scene.FIELD_HALF_X, p[..., 1] / scene.FIELD_HALF_Y,
            torch.sin(th), torch.cos(th),
            v[..., 0] / cpu_base.MAX_V, v[..., 1] / cpu_base.MAX_V, v[..., 2] / cpu_base.MAX_W,
            (world["grace"] > 0).to(p.dtype),
        ], -1)
        ball = torch.cat([
            torch.stack([bp[:, 0] / scene.FIELD_HALF_X, bp[:, 1] / scene.FIELD_HALF_Y,
                         (bp[:, 2] - scene.FLOOR_Z) / scene.GOAL_HEIGHT], -1),
            bv / cpu_base.MAX_BALL_V,
        ], -1)
        return torch.cat([ball, robots.flatten(1), self.task_obs(d, world)], -1).float()

    def xy(self, d, i):
        return d.xpos[:, self.robots[i].body, :2]

    def ball_xy(self, d):
        return d.xpos[:, self.ball.body, :2]

    @staticmethod
    def place_robot(qpos, qvel, r: scene.RobotIds, x, y, theta):
        """`RobotIds.place`, in place on (N, nq)/(N, nv): the slides are relative to the
        body's built position."""
        a, v = r.qpos_adr, r.qvel_adr
        qpos[:, a] = x - r.xy0[0]
        qpos[:, a + 1] = y - r.xy0[1]
        qpos[:, a + 2] = theta
        qvel[:, v : v + 3] = 0.0

    def place_ball(self, qpos, qvel, x, y):
        """`BallIds.place`, at rest, in place on (N, nq)/(N, nv)."""
        a, v = self.ball.qpos_adr, self.ball.qvel_adr
        qpos[:, a] = x
        qpos[:, a + 1] = y
        qpos[:, a + 3 : a + 7] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=qpos.dtype,
                                              device=qpos.device)
        qvel[:, v : v + 6] = 0.0

    # ------------------------------------------------------ subclass surface

    def place(self, qpos, qvel, u):
        """`_place`, with its `n_uniform` draws per world passed in (`u` (N, n_uniform) in
        [0, 1)). Returns new (qpos, qvel, overlap (N,))."""
        raise NotImplementedError

    def init_task(self, d) -> dict:
        """The task's per-world state at reset (`_reset_task`'s bookkeeping)."""
        return {}

    def shape_drive(self, cmd):
        return cmd

    def script(self, d, ctrl):
        return ctrl

    def task_obs(self, d, world):
        return d.qpos.new_zeros(d.qpos.shape[0], 0)

    def reward_done(self, d, world):
        """(reward (N,), terminated (N,) bool, world) for the control step just taken."""
        raise NotImplementedError
