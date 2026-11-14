"""The shared Gymnasium env: action plumbing, observation, episode bookkeeping.

Every task subclasses this and supplies three things -- where the pieces start
(`_place`), what the non-agent robots do (`_script`), and what a step is worth
(`_reward_done`).

Action layout is one flat `Box(-1, 1, (n_controlled * act_dim,))`. `step` reshapes it to
`(n_controlled, act_dim)` and writes each row to that robot's actuators. Only controlled
robots get a slice; scripted ones are driven inside `_script`.

Per-robot slice, in order:

    x, y, sin(theta), cos(theta), kick, chip, dribble

so `act_dim` is 7, the same for every task. The first four are a target pose: x, y scaled
by the half field (6.0 x 4.5 m), heading atan2(sin, cos), target velocity always 0. The
navigation controller (`navigation.go_to_point`) turns it into a body-frame velocity every
control step, from the robot's own joint state. Every robot always exposes every control
it physically has; a task that does not care about one simply does not reward it, which
keeps a single action head transferable across all the envs.

The control step is a fixed 25 ms (40 Hz, 24 physics steps). A decision -- one `step` --
holds its action for `repeat_action` control steps.

`step_raw` is an internal path outside the action space for the envelope tests: the drive
is (vx, vy, omega) in [-1, 1], body frame, scaled by the servos' own ctrlrange.
"""

from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from . import scene
from .navigation import go_to_point

MAX_V = 3.3  # m/s -- the chassis servo ctrlrange (ticket 12), used to normalise observations
MAX_W = 37.1  # rad/s, likewise: the yaw ceiling of a spin about the CoM (ticket 12)
MAX_BALL_V = 8.0  # m/s -- kicks cap at 6.5 (scene.KICK_SPEED_MAX) off a standing robot, ~8.5 moving
CONTROL_DT = 0.025  # s: the navigation controller runs at a fixed 40 Hz
POSSESSION_GRACE = 0.05  # s of no ball/dribbler contact before the ball counts as lost


class BaseEnv(gym.Env):
    """N robots and a ball on an SSL Division A field."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    n_blue = 1
    n_yellow = 0
    n_controlled = 1
    extra_obs = 0  # task-specific observations appended by _task_obs
    extra_obs_bound = np.inf  # |bound| of each, scalar or one per extra

    def __init__(
        self,
        repeat_action: int = 1,
        horizon_s: float = 30.0,
        render_mode: str | None = None,
    ) -> None:
        self.model, self.robots, self.ball = scene.build(self.n_blue, self.n_yellow)
        self.data = mujoco.MjData(self.model)

        # Exact (ticket 13): every horizon, lock and grace is counted in control steps.
        steps = CONTROL_DT / self.model.opt.timestep
        self.decimation = round(steps)
        assert abs(steps - self.decimation) < 1e-6, steps
        self.dt = self.decimation * self.model.opt.timestep  # one control step
        if repeat_action < 1:
            raise ValueError(f"repeat_action={repeat_action} must be >= 1")
        self.repeat_action = int(repeat_action)
        self.metadata = dict(
            self.metadata, render_fps=round(1.0 / (self.dt * self.repeat_action))
        )
        self.max_steps = max(1, round(horizon_s / self.dt))  # in control steps

        self.drive_dim = 4  # x, y, sin, cos: the target pose
        self.act_dim = self.drive_dim + 3  # kick, chip, dribble
        self.action_space = spaces.Box(
            -1.0, 1.0, (self.n_controlled * self.act_dim,), dtype=np.float32
        )
        # Finite only where the simulator guarantees it (ticket 11): nothing clips, so a
        # bound that play can exceed would be a lie, not a normalisation.
        inf = np.inf
        rx = scene.WALL_HALF_X / scene.FIELD_HALF_X
        ry = scene.WALL_HALF_Y / scene.FIELD_HALF_Y
        extra = np.broadcast_to(self.extra_obs_bound, (self.extra_obs,))
        n = len(self.robots)
        # ball x, y, z, v; then per robot x, y, sin, cos, vx, vy, omega, has_ball
        low = np.concatenate(
            [[-inf, -inf, 0, -inf, -inf, -inf], [-rx, -ry, -1, -1, -inf, -inf, -inf, 0] * n, -extra]
        )
        high = np.concatenate([[inf] * 6, [rx, ry, 1, 1, inf, inf, inf, 1] * n, extra])
        self.observation_space = spaces.Box(
            low.astype(np.float32), high.astype(np.float32), dtype=np.float32
        )

        self.render_mode = render_mode
        self._renderer = None
        self._viewer = None
        # Steps a robot must wait before a plunger can re-arm. Without it a held kick
        # action pins the plunger against its 30 mm stop and the ball is never released.
        # One column per plunger (0 = kicker, 1 = chipper): they are independent
        # mechanisms, so a kick must not be able to block a chip.
        self._plunger_lock = np.zeros((len(self.robots), 2), dtype=int)
        self._kick_cooldown = max(1, round(0.25 / self.dt))
        self._grace = np.zeros(len(self.robots), dtype=int)
        self.fired = np.zeros(len(self.robots), dtype=bool)  # a plunger fired this step
        self._grace_steps = max(1, round(POSSESSION_GRACE / self.dt))
        # The drive each controlled robot was commanded on the last control step, body
        # frame, as a fraction of its ctrlrange: what the energy penalties charge for.
        self.drive_cmd = np.zeros((self.n_controlled, 3))
        self.steps = 0

    # ---------------------------------------------------------------- gym API

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._plunger_lock[:] = 0
        self._grace[:] = 0
        self.drive_cmd[:] = 0.0
        self.steps = 0
        self._place()
        mujoco.mj_forward(self.model, self.data)
        self._reset_task()
        return self._obs(), {}

    def step(self, action):
        """One decision: `repeat_action` control steps of the same action, rewards summed,
        cut short when the episode ends."""
        return self._decide(action, raw=False)

    def step_raw(self, action):
        """Internal, for the envelope tests: like `step`, but each robot's slice is
        (vx, vy, omega, kick, chip, dribble) and the drive goes straight to the servos."""
        return self._decide(action, raw=True)

    def _decide(self, action, raw: bool):
        dim = 3 + 3 if raw else self.act_dim
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        action = action.reshape(self.n_controlled, dim)
        total = 0.0
        for _ in range(self.repeat_action):
            obs, reward, terminated, truncated, info = self._control_step(action, raw)
            total += reward
            if terminated or truncated:
                break
        if self.render_mode == "human":
            self.render()
        return obs, total, terminated, truncated, info

    def _control_step(self, action: np.ndarray, raw: bool):
        self.data.ctrl[:] = 0.0
        self.fired[:] = False
        self._apply(action, raw)
        self._script()
        n, dt = self.decimation, self.model.opt.timestep
        if (self._plunger_lock == self._kick_cooldown).any():  # a plunger fired this step
            # K fine steps at dt / FIRING_SUBSTEPS, then the rest of the control step at dt.
            fine = min(scene.FIRING_FINE_STEPS, n * scene.FIRING_SUBSTEPS)
            n -= fine // scene.FIRING_SUBSTEPS
            # ponytail: mutates the shared, cached MjModel for the length of this loop. Safe
            # while envs sharing it step one at a time (Sync/Async vector envs both do);
            # threads would need their own model.
            self.model.opt.timestep = dt / scene.FIRING_SUBSTEPS
            try:
                for _ in range(fine):
                    mujoco.mj_step(self.model, self.data)
            finally:  # never leave the cached model at dt / FIRING_SUBSTEPS for every other env
                self.model.opt.timestep = dt
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        np.maximum(self._plunger_lock - 1, 0, out=self._plunger_lock)

        obs = self._obs()
        reward, terminated, info = self._reward_done(action.ravel())
        truncated = self.steps >= self.max_steps
        if (
            not np.isfinite(self.data.qpos).all()
            or not np.isfinite(self.data.qvel).all()
        ):
            # Loudly, rather than training a policy on NaN observations.
            raise RuntimeError(f"simulation diverged at step {self.steps}")
        return obs, float(reward), bool(terminated), bool(truncated), info

    # -------------------------------------------------------- action plumbing

    def _apply(self, action: np.ndarray, raw: bool = False) -> None:
        for i in range(self.n_controlled):
            if raw:
                cmd = action[i, :3] * self.robots[i].drive_range[:, 1]
            else:
                cmd = self._navigate(i, action[i, :4])
            self._drive_robot(i, self._shape_drive(np.asarray(cmd, dtype=np.float64)))
            self._actuate(i, action[i, -3:])

    def _navigate(self, index: int, pose: np.ndarray) -> tuple[float, float, float]:
        """The navigation controller on robot `index`'s own joint state (not the
        observation): world-frame position, velocity and yaw off the chassis slides and
        hinge, out comes a body-frame (vx, vy, omega)."""
        r = self.robots[index]
        q = self.data.qpos[r.qpos_adr : r.qpos_adr + 3]
        v = self.data.qvel[r.qvel_adr : r.qvel_adr + 2]
        pos = (q[0] + r.xy0[0], q[1] + r.xy0[1])
        target = (pose[0] * scene.FIELD_HALF_X, pose[1] * scene.FIELD_HALF_Y)
        return go_to_point(pos, v, float(q[2]), target, float(np.arctan2(pose[2], pose[3])))

    def _shape_drive(self, cmd: np.ndarray) -> np.ndarray:
        """A task's last word on a robot's body-frame (vx, vy, omega)."""
        return cmd

    def _drive_robot(self, index: int, cmd: np.ndarray) -> None:
        """Body-frame (vx, vy, omega), in m/s and rad/s, into robot `index`'s servos."""
        r = self.robots[index]
        self.drive_cmd[index] = np.clip(cmd / r.drive_range[:, 1], -1.0, 1.0)
        # The chassis joints are *world*-axis slides, but the command is a body-frame
        # velocity -- that is how SSL firmware is commanded, so rotate it here. A diagonal
        # command can ask for 3.3*sqrt(2) on one world axis; MuJoCo clamps it to
        # ctrlrange per axis, same as a real servo would.
        v = self.data.xmat[r.body].reshape(3, 3) @ (cmd[0], cmd[1], 0.0)
        self.data.ctrl[list(r.drive)] = (v[0], v[1], cmd[2])

    def _actuate(self, index: int, a: np.ndarray) -> None:
        """Kick, chip, dribble for robot `index`, each in [-1, 1]."""
        r = self.robots[index]
        # Kicker and chipper are the same mechanism twice over: a force actuator against a
        # return spring. Each gets its own lock so a kick cannot block a chip. Half range
        # like the dribbler: a <= 0 never fires, (0, 1] asks for (0, KICK_SPEED_MAX] m/s.
        for j, actuator in enumerate((r.kick, r.chip)):
            if a[j] > 0.0 and self._plunger_lock[index, j] == 0:
                speeds = scene.KICK_TABLE_MS[j]
                self.data.ctrl[actuator] = np.interp(
                    scene.KICK_SPEED_MAX * float(a[j]),
                    speeds,
                    scene.KICK_TABLE_N[: len(speeds)],
                )
                self._plunger_lock[index, j] = self._kick_cooldown
                self.fired[index] = True
        # Half range (ticket 08): a <= 0 is off, (0, 1] is (0, design speed], so a neutral or
        # sampled action does not dribble. Still a speed, not a switch, because that is what
        # the firmware takes. Never negative: a bar driven backwards throws the ball out of
        # the mouth.
        self.data.ctrl[r.dribble] = scene.DRIBBLER_OMEGA * max(float(a[2]), 0.0)

    # ------------------------------------------------------------ observation

    def holds(self, i: int) -> bool:
        """Is robot `i` carrying the ball? Valid after the last _obs."""
        return bool(self._grace[i] > 0)

    def _update_possession(self) -> None:
        """One pass over the contact list marks every dribbler currently on the ball.

        Cheaper than asking per robot, and contact is the criterion the mjssl retention
        benchmarks score against -- not the arm sensor, which reads the wrong sign here.

        A spinning bar does not touch the ball every step: it sticks, slips and re-grips,
        and a held ball reads as contact on roughly half of them. So a robot keeps
        possession through POSSESSION_GRACE of silence, exactly as mjssl's retention
        sweeps require the bar to be clear for a continuous 50 ms before calling the ball
        lost. Without it any "hold the ball for N seconds" termination is unreachable.
        """
        touching = np.zeros(len(self.robots), dtype=bool)
        by_body = {r.dribbler_body: i for i, r in enumerate(self.robots)}
        for k in range(self.data.ncon):
            c = self.data.contact[k]
            b1 = int(self.model.geom_bodyid[c.geom1])
            b2 = int(self.model.geom_bodyid[c.geom2])
            other = b2 if b1 == self.ball.body else b1 if b2 == self.ball.body else -1
            if other in by_body:
                touching[by_body[other]] = True
        self._grace = np.where(
            touching, self._grace_steps, np.maximum(self._grace - 1, 0)
        )

    def _obs(self) -> np.ndarray:
        self._update_possession()
        d = self.data
        bp, bv = self.ball.pos(d), self.ball.vel(d)
        out = [
            bp[0] / scene.FIELD_HALF_X,
            bp[1] / scene.FIELD_HALF_Y,
            (bp[2] - scene.FLOOR_Z) / scene.GOAL_HEIGHT,
            bv[0] / MAX_BALL_V,
            bv[1] / MAX_BALL_V,
            bv[2] / MAX_BALL_V,
        ]
        for i, r in enumerate(self.robots):
            x, y, th = r.pose(d)
            vx, vy, wz = r.velocity(d)
            out += [
                x / scene.FIELD_HALF_X,
                y / scene.FIELD_HALF_Y,
                np.sin(th),
                np.cos(th),
                vx / MAX_V,
                vy / MAX_V,
                wz / MAX_W,
                float(self._grace[i] > 0),
            ]
        out += list(self._task_obs())
        return np.asarray(out, dtype=np.float32)

    def _task_obs(self):
        return ()

    # ---------------------------------------------------------------- helpers

    def ball_in_frame(self, r: scene.RobotIds) -> np.ndarray:
        """Ball position in a robot's own frame -- the only frame possession means
        anything in."""
        R = self.data.xmat[r.body].reshape(3, 3)
        return R.T @ (self.ball.pos(self.data) - self.data.xpos[r.body])

    def seat_ball(
        self,
        r: scene.RobotIds,
        distance: float = 0.20,
        speed: float = 1.0,
        settle: float = 0.35,
    ) -> None:
        """Roll a ball into a robot's dribbler and let the mechanism catch it.

        Placing the ball at the sphere/bar tangency instead is a launcher, not a seat: the
        contact needs a few tenths of a millimetre of overlap to exist at all, and on a
        46 g ball that overlap is a compressed spring worth several newtons. Below ~0.1 mm
        MuJoCo stops generating the contact at all, so there is no overlap that works.

        So do what the real robot does -- spin the bar up, roll a ball in at 1 m/s, let it
        be caught. Not faster: with the measured ball inertia (beta = 0.5) a ball arriving
        at 2 m/s from 200 mm rebounds off the bar and rolls out of reach.
        Fixed distance, fixed speed, no timestep dependence, and the ball ends up in the
        pocket the dribbler actually holds it in rather than one chosen with a ruler. Must
        be called from _reset_task, after reset() has run mj_forward.
        """
        x, y, _ = r.pose(self.data)
        f = r.forward(self.data)
        self.ball.place(
            self.data,
            x + f[0] * distance,
            y + f[1] * distance,
            -f[0] * speed,
            -f[1] * speed,
        )
        mujoco.mj_forward(self.model, self.data)
        for _ in range(round(settle / self.model.opt.timestep)):
            self.data.ctrl[r.dribble] = scene.DRIBBLER_OMEGA
            mujoco.mj_step(self.model, self.data)

    def sample_pose(self, taken: list, margin: float = 0.3, clearance: float = 0.25):
        """Rejection-sample a free (x, y) inside the field, at least `clearance` from
        everything already placed."""
        for _ in range(200):
            x = self.np_random.uniform(
                -scene.FIELD_HALF_X + margin, scene.FIELD_HALF_X - margin
            )
            y = self.np_random.uniform(
                -scene.FIELD_HALF_Y + margin, scene.FIELD_HALF_Y - margin
            )
            if all((x - px) ** 2 + (y - py) ** 2 > clearance**2 for px, py in taken):
                taken.append((x, y))
                return x, y
        raise RuntimeError("could not place everything without overlap")

    # -------------------------------------------------------- subclass surface

    def _place(self) -> None:
        """Write the reset pose of every robot and the ball into self.data.qpos."""
        raise NotImplementedError

    def _reset_task(self) -> None:
        """Reset any per-episode task state, after the world has been placed."""

    def _script(self) -> None:
        """Drive the robots the policy does not control."""

    def _reward_done(self, action: np.ndarray) -> tuple[float, bool, dict]:
        raise NotImplementedError

    # ------------------------------------------------------------------ render

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, 720, 1280)
            self._renderer.update_scene(self.data, camera="overview")
            return self._renderer.render()
        if self.render_mode == "human":
            import mujoco.viewer

            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
                # The viewer renders on its own thread, so its camera is behind a mutex.
                with self._viewer.lock():
                    self._frame_field(self._viewer.cam)
            self._viewer.sync()
        return None

    def _frame_field(self, cam) -> None:
        """Open the viewer on the whole field rather than on a patch of grass.

        The viewer's default free camera is placed off the model's own extent statistic,
        which the surrounding walls and the goal nets inflate well past the playing area --
        it lands close in and pointed at the middle of the pitch, and the first thing anyone
        does is drag it back out. Match the `overview` camera in the MJCF instead: south
        side, both goals in frame, the field roughly filling the window. Distance is tied to
        the field rather than a constant so it survives a differently sized pitch.

        This is a *free* camera, deliberately -- pinning the viewer to the fixed `overview`
        camera would take the mouse away. It only sets where the view starts.
        """
        cam.lookat[:] = (0.0, 0.0, 0.0)
        cam.distance = 2.2 * scene.FIELD_HALF_X
        cam.azimuth = (
            90.0  # south of the field, so +x (the attacking goal) is on the right
        )
        cam.elevation = -35.0

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
