"""The shared Gymnasium env: action plumbing, observation, episode bookkeeping.

Every task subclasses this and supplies three things -- where the pieces start
(`_place`), what the non-agent robots do (`_script`), and what a step is worth
(`_reward_done`).

Action layout is one flat `Box(-1, 1, (n_controlled * act_dim,))`. `step` reshapes it to
`(n_controlled, act_dim)` and writes each row to that robot's actuators. Only controlled
robots get a slice; scripted ones are driven inside `_script`.

Per-robot slice, in order:

    tier 1 (default):  w1, w2, w3, w4        tier 0:  vx, vy, omega
    then always:  kick, chip, dribble

so `act_dim` is 7 on the default tier 1 and 6 on tier 0, the same for every task. Every robot always
exposes every control it physically has; a task that does not care about one simply does
not reward it, which keeps a single action head transferable across all the envs.

All in [-1, 1] and scaled by the actuator's own ctrlrange, so the bounds come from the
model rather than from a constant here.
"""

from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from . import scene

MAX_V = 4.0  # m/s -- the tier-0 chassis servo ctrlrange, used to normalise observations
MAX_W = 30.0  # rad/s, likewise
MAX_BALL_V = 8.0  # m/s -- a flat kick leaves at 6-7
POSSESSION_GRACE = 0.05  # s of no ball/dribbler contact before the ball counts as lost
# A robot is over when its own +z is more than 60 deg off world +z. Well past anything
# driving produces -- the roller chassis pitches a couple of degrees under hard
# acceleration -- and well short of 90, so a robot on its way down does not have to land
# flat before the episode ends.
TIPPED_COS = 0.5
TIPOVER_PENALTY = 100.0  # same order as the task bonuses, so falling is never worth it


class BaseEnv(gym.Env):
    """N robots and a ball on an SSL Division A field."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    n_blue = 1
    n_yellow = 0
    n_controlled = 1
    extra_obs = 0  # task-specific observations appended by _task_obs

    def __init__(
        self,
        tier: int = 1,
        control_hz: int = 60,
        horizon_s: float = 30.0,
        render_mode: str | None = None,
    ) -> None:
        self.tier = tier
        self.model, self.robots, self.ball = scene.build(
            self.n_blue, self.n_yellow, tier
        )
        self.data = mujoco.MjData(self.model)

        self.decimation = max(1, round(1.0 / (control_hz * self.model.opt.timestep)))
        self.dt = self.decimation * self.model.opt.timestep
        self.metadata = dict(self.metadata, render_fps=round(1.0 / self.dt))
        self.max_steps = max(1, round(horizon_s / self.dt))

        self.drive_dim = 3 if tier == 0 else 4
        self.act_dim = self.drive_dim + 3  # kick, chip, dribble
        self.action_space = spaces.Box(
            -1.0, 1.0, (self.n_controlled * self.act_dim,), dtype=np.float32
        )
        n_obs = 6 + 8 * len(self.robots) + self.extra_obs
        self.observation_space = spaces.Box(-np.inf, np.inf, (n_obs,), dtype=np.float32)

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
        self._grace_steps = max(1, round(POSSESSION_GRACE / self.dt))
        self.steps = 0

    # ---------------------------------------------------------------- gym API

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._plunger_lock[:] = 0
        self._grace[:] = 0
        self.steps = 0
        self._place()
        mujoco.mj_forward(self.model, self.data)
        self._reset_task()
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.data.ctrl[:] = 0.0
        self._apply(action.reshape(self.n_controlled, self.act_dim))
        self._script()
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        np.maximum(self._plunger_lock - 1, 0, out=self._plunger_lock)

        obs = self._obs()
        reward, terminated, info = self._reward_done(action)
        truncated = self.steps >= self.max_steps
        if (
            not np.isfinite(self.data.qpos).all()
            or not np.isfinite(self.data.qvel).all()
        ):
            # Loudly, rather than training a policy on NaN observations.
            raise RuntimeError(f"simulation diverged at step {self.steps}")
        if not scene.in_field(*self.ball.pos(self.data)[:2]) and not info.get("goal"):
            terminated = True
            info["ball_out"] = True
        if self.tipped():
            reward -= TIPOVER_PENALTY
            terminated = True
            info["tipped"] = True
        if self.render_mode == "human":
            self.render()
        return obs, float(reward), bool(terminated), bool(truncated), info

    # -------------------------------------------------------- action plumbing

    def _apply(self, action: np.ndarray) -> None:
        for i in range(self.n_controlled):
            self._drive_robot(i, action[i])

    def _drive_robot(self, index: int, a: np.ndarray) -> None:
        r = self.robots[index]
        lo, hi = r.drive_range[:, 0], r.drive_range[:, 1]
        # Symmetric ranges, so a single scale keeps 0 -> 0 rather than -> midpoint.
        cmd = a[: self.drive_dim] * np.maximum(hi, -lo)
        if self.tier == 0:
            # The tier-0 chassis joints are *world*-axis slides, but the action means a
            # body-frame velocity -- that is how SSL firmware is commanded, and it is what
            # tier 1's wheel speeds inherently produce. Rotating here is what makes the two
            # tiers mean the same thing. A diagonal command can ask for 4*sqrt(2) on one
            # world axis; MuJoCo clamps it to ctrlrange, same as a real servo would.
            v = self.data.xmat[r.body].reshape(3, 3) @ (cmd[0], cmd[1], 0.0)
            cmd = (v[0], v[1], cmd[2])
        self.data.ctrl[list(r.drive)] = cmd
        k = self.drive_dim
        # Kicker and chipper are the same mechanism twice over: a force actuator against a
        # return spring. Each gets its own lock so a kick cannot block a chip.
        for j, actuator in enumerate((r.kick, r.chip)):
            if a[k + j] > 0.0 and self._plunger_lock[index, j] == 0:
                self.data.ctrl[actuator] = scene.KICK_FORCE
                self._plunger_lock[index, j] = self._kick_cooldown
        # [-1, 1] -> [0, design speed]. Never negative: a bar driven backwards throws the
        # ball out of the mouth.
        self.data.ctrl[r.dribble] = scene.DRIBBLER_OMEGA * (a[k + 2] + 1.0) * 0.5

    # ------------------------------------------------------------ observation

    def tipped(self) -> bool:
        """Has any *controlled* robot fallen over?

        Only the controlled ones: a scripted opponent knocked on its side is the agent's
        problem to exploit, not a reason to end its episode. Read off the body frame's
        third column, which is the robot's own +z in world coordinates, so it is one dot
        product rather than a quaternion. Always False on tier 0, whose chassis is a planar
        joint and cannot leave the ground.
        """
        return any(
            self.data.xmat[r.body].reshape(3, 3)[2, 2] < TIPPED_COS
            for r in self.robots[: self.n_controlled]
        )

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
            bp[2] / scene.GOAL_HEIGHT,
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
        speed: float = 2.0,
        settle: float = 0.35,
    ) -> None:
        """Roll a ball into a robot's dribbler and let the mechanism catch it.

        Placing the ball at the sphere/bar tangency instead is a launcher, not a seat: the
        contact needs a few tenths of a millimetre of overlap to exist at all, and on a
        46 g ball that overlap is a compressed spring worth several newtons. Below ~0.1 mm
        MuJoCo stops generating the contact at all, so there is no overlap that works.

        So do what the real robot does -- spin the bar up, roll a ball in at 2 m/s, let it
        be caught. Fixed distance, fixed speed, no timestep dependence, and the ball ends
        up in the pocket the dribbler actually holds it in rather than one chosen with a
        ruler. Must be called from _reset_task, after reset() has run mj_forward.
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
