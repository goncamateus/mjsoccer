"""The task environments.

Each one supplies a reset layout, whatever the scripted robots do, and a reward. Distance
shaping is potential-based -- the *decrease* in some distance over one step, so no cycle
pays. GoToBall and the kick tasks pay facing the same way, 0.5 * (cos - prev_cos), with a 10
bonus; Dribbling and ContestedPossession still pay a per-step 0.1 * cos(front, robot->ball)
against a 100 bonus (.scratch/rl-learning/issues/01-why-nothing-learns.md, 02-reward-redesign.md).

Every env exposes the same per-robot action -- drive, kick, chip, dribble -- whether or
not its reward cares, so one action head transfers across all six.

    id                       robots           action (tier 1 default / tier 0)
    GoToBall-v0              1 blue           7 / 6
    GoToBallShoot-v0         1 blue           7 / 6
    Dribbling-v0             1 blue           7 / 6
    StaticDefenders-v0       1 blue + 6 yel   7 / 6
    ContestedPossession-v0   1 blue + 1 yel   7 / 6
    PassEndurance-v0         2 blue           7 / 6
"""

from __future__ import annotations

import numpy as np

from . import scene as S
from .base import BaseEnv

# The ball sits about 94 mm out from the chassis centre when the dribbler holds it, so
# "at the ball" is a little past the 90 mm robot radius.
REACH = 0.13
# Penalty on ||action||^2 per simulated *second*, paid as ENERGY * dt a step, so it costs the
# same at any control rate. 0.15 is the old 0.01 a step at 15 Hz; paid per step at 40 Hz it
# was 2.7x that, and a random 30 s GoToBall episode (-24) out-weighed the +10 arrival.
ENERGY = 0.15
FACING = 0.1  # per-step weight on cos(robot front, robot->ball), the raw cosine
# GoToBall and the kick tasks pay facing as a potential, which cannot be farmed, so their
# bonus and fouls stay small next to the shaping; against 100 the shaping went faint and SAC
# stopped learning (.scratch/rl-learning/issues/01-why-nothing-learns.md).
GOTO_BONUS = 10.0
KICK_BONUS = 10.0
SETUP_RANGE = 0.5  # m: the lining-up potential fades to 0 this far from the ball
KICK_PENALTY = 10.0  # kick tasks: ball out, or StaticDefenders' collision
BONUS = 100.0  # task completed; large enough that facing the ball and parking never pays
OUT_PENALTY = 100.0  # ball out of the field, a collision foul
# SSL rules 8.4.2 "Crashing": the difference of the two robots' velocities, projected onto
# the line between them at the moment of collision; over 1.5 m/s the faster robot fouled.
CRASH_SPEED = 1.5


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros_like(v)


class _BallTask(BaseEnv):
    """Shared machinery for every task whose first job is to reach the ball."""

    def _agent(self) -> S.RobotIds:
        return self.robots[0]

    def _to_ball(self) -> float:
        r = self._agent()
        return float(
            np.linalg.norm(self.ball.pos(self.data)[:2] - self.data.xpos[r.body][:2])
        )

    def _facing(self) -> float:
        """cos of the angle between the robot's front and the direction to the ball."""
        r = self._agent()
        d = _unit(self.ball.pos(self.data)[:2] - self.data.xpos[r.body][:2])
        return float(d @ r.forward(self.data)[:2])

    def _at_ball(self) -> bool:
        return self._to_ball() < REACH and self._facing() > 0.9

    def _energy(self, action: np.ndarray) -> float:
        return -ENERGY * self.dt * float(np.square(action).sum())

    def _touching(self, j: int) -> bool:
        """Is any part of the agent in contact with any part of robot `j`?"""
        m, d = self.model, self.data
        pair = {self._agent().body, self.robots[j].body}
        for k in range(d.ncon):
            c = d.contact[k]
            roots = {
                int(m.body_rootid[m.geom_bodyid[c.geom1]]),
                int(m.body_rootid[m.geom_bodyid[c.geom2]]),
            }
            if roots == pair:
                return True
        return False


class _KickTask(_BallTask):
    """Reach the ball, then one kick sends it at a target.

    Until the agent kicks, closing on the ball pays, and so does lining up behind it aimed at
    the target (`_setup`); after, the ball closing on the target does. One kick per episode: a kick is the kicker or chipper firing while the robot is
    at the ball (within REACH, facing it), judged on the state the action was taken in.
    A plunger fired at nothing does not count, or random exploration would spend the kick
    on the first step.
    """

    ball_out_penalty = KICK_PENALTY

    def _target_point(self) -> np.ndarray:
        raise NotImplementedError

    def _target_dist(self) -> float:
        return float(np.linalg.norm(self.ball.pos(self.data)[:2] - self._target_point()))

    def _done(self) -> bool:
        raise NotImplementedError

    def _setup(self) -> float:
        """How lined up the kick is: near the ball, facing it, front aimed at the target.

        The kick counts on facing the ball alone, so without this the shot went wherever the
        robot happened to arrive from. Paid as a potential until the kick, then frozen.
        """
        r = self._agent()
        b = self.ball.pos(self.data)[:2]
        aim = float(_unit(self._target_point() - b) @ r.forward(self.data)[:2])
        near = max(0.0, 1.0 - self._to_ball() / SETUP_RANGE)
        return max(self._facing(), 0.0) * near * max(aim, 0.0)

    def _reset_task(self) -> None:
        self.has_kicked = False
        self._was_at_ball = self._at_ball()
        self._prev, self._prev_target = self._to_ball(), self._target_dist()
        self._prev_facing = self._facing_potential()
        self._prev_setup = self._setup()

    def _facing_potential(self) -> float:
        return self._facing() + self._extra_facing()

    def _reward_done(self, action):
        if self.fired[0] and self._was_at_ball:
            self.has_kicked = True
        d, g = self._to_ball(), self._target_dist()
        if self.has_kicked:
            reward = self._prev_target - g  # setup frozen: the ball leaving costs nothing
        else:
            s = self._setup()
            reward = self._prev - d + (s - self._prev_setup)
            self._prev_setup = s
        f = self._facing_potential()
        reward += 0.5 * (f - self._prev_facing) + self._energy(action)
        self._prev, self._prev_target, self._prev_facing = d, g, f
        self._was_at_ball = self._at_ball()
        done = self._done()
        if done:
            reward += KICK_BONUS
        return reward, done, {"is_success": done, "has_kicked": self.has_kicked}

    def _extra_facing(self) -> float:
        """Any second cosine the task wants faced, paid as a potential with the first."""
        return 0.0


class GoToBallEnv(_BallTask):
    """Drive to the ball and end up pointing at it.

    The kicker, chipper and dribbler are in the action space but the reward ignores them.
    """

    def _place(self) -> None:
        taken: list[tuple[float, float]] = []
        x, y = self.sample_pose(taken, clearance=0.5)
        self.robots[0].place(self.data, x, y, self.np_random.uniform(-np.pi, np.pi))
        bx, by = self.sample_pose(taken, clearance=0.5)
        self.ball.place(self.data, bx, by)

    def _reset_task(self) -> None:
        self._prev, self._prev_facing = self._to_ball(), self._facing()

    def _reward_done(self, action):
        d, f = self._to_ball(), self._facing()
        # Facing is a potential here, not the raw per-step cosine the other tasks pay: parking
        # earns at most the one turn, so the bonus can stay small next to the shaping. A raw
        # 0.1 * cos needed the 100 bonus to keep parking unprofitable, and against 100 the
        # shaping went faint: SAC dropped from 0.93 to 0.10-0.20 success, back to 0.95-0.97
        # this way (.scratch/rl-learning/issues/01-why-nothing-learns.md).
        reward = (self._prev - d) + 0.5 * (f - self._prev_facing) + self._energy(action)
        self._prev, self._prev_facing = d, f
        arrived = self._at_ball()
        if arrived:
            reward += GOTO_BONUS
        return reward, arrived, {"is_success": arrived, "dist": d}


class GoToBallShootEnv(_KickTask):
    """Reach the ball and put it in the +x goal. Kicker and dribbler are the agent's."""

    def _place(self) -> None:
        taken: list[tuple[float, float]] = []
        # Start on the defending half so shooting means covering ground.
        x = self.np_random.uniform(-S.FIELD_HALF_X + 0.3, 0.0)
        y = self.np_random.uniform(-S.FIELD_HALF_Y + 0.3, S.FIELD_HALF_Y - 0.3)
        taken.append((x, y))
        self.robots[0].place(self.data, x, y, self.np_random.uniform(-np.pi, np.pi))
        bx, by = self.sample_pose(taken, clearance=0.5)
        self.ball.place(self.data, bx, by)

    def _target_point(self) -> np.ndarray:
        """The nearest point of the goal mouth to the ball."""
        y = self.ball.pos(self.data)[1]
        return np.array([S.FIELD_HALF_X, np.clip(y, -S.GOAL_HALF_Y, S.GOAL_HALF_Y)])

    def _done(self) -> bool:
        return bool(S.in_goal(self.ball.pos(self.data)))

    def _reward_done(self, action):
        reward, done, info = super()._reward_done(action)
        info["goal"] = done
        return reward, done, info


class DribblingEnv(_BallTask):
    """Carry the ball through a sequence of waypoints without losing it.

    The agent can kick or chip its own carried ball away -- the controls are always in the
    action space. It is not worth doing: holding pays every step and waypoint progress is
    gated on `holds(0)`, so the step the ball leaves the mouth is the step both stop paying.
    """

    extra_obs = 2  # the current waypoint, in the robot's own frame
    # No component is longer than the farthest an in-field waypoint can be from a robot
    # inside the walls.
    _reach = np.hypot(S.FIELD_HALF_X + S.WALL_HALF_X, S.FIELD_HALF_Y + S.WALL_HALF_Y)
    extra_obs_bound = (_reach / S.FIELD_HALF_X, _reach / S.FIELD_HALF_Y)
    n_waypoints = 4
    waypoint_tol = 0.3
    gate_bonus = 10.0

    def _place(self) -> None:
        taken: list[tuple[float, float]] = []
        x, y = self.sample_pose(taken, clearance=0.5)
        self.robots[0].place(self.data, x, y, self.np_random.uniform(-np.pi, np.pi))

    def _reset_task(self) -> None:
        # The episode starts with the ball already carried; getting it there is GoToBall's
        # job, not this one's.
        self.seat_ball(self.robots[0])
        self._target = np.array(self.sample_pose([], clearance=0.0))
        self._reached = 0
        self._lost = 0
        self._prev = self._to_target()
        self._prev_ball = self._to_ball()

    def _to_target(self) -> float:
        return float(np.linalg.norm(self.ball.pos(self.data)[:2] - self._target))

    def _task_obs(self):
        r = self._agent()
        R = self.data.xmat[r.body].reshape(3, 3)[:2, :2]
        local = R.T @ (self._target - self.data.xpos[r.body][:2])
        return local / (S.FIELD_HALF_X, S.FIELD_HALF_Y)

    def _reward_done(self, action):
        held = float(self.holds(0))
        d, b = self._to_target(), self._to_ball()
        # Holding dominates on purpose: carry the ball always, through the gates. Waypoint
        # progress only counts while held, or a policy could kick the ball at the waypoint.
        # Approach and facing are what bring the robot back to a ball it dropped.
        reward = (
            (self._prev_ball - b)
            + FACING * self._facing()
            + held
            + (self._prev - d) * held
            + self._energy(action)
        )
        self._prev, self._prev_ball = d, b
        self._lost = 0 if held else self._lost + 1
        if d < self.waypoint_tol and held:
            self._reached += 1
            reward += self.gate_bonus
            self._target = np.array(self.sample_pose([], clearance=0.0))
            self._prev = self._to_target()
        success = self._reached >= self.n_waypoints
        if success:
            # Finishing must never be worth less than parking with the ball: pay the holding
            # the remaining steps would have earned. Without it, standing still out-earned the
            # scripted carry (490 vs 201 at 15 Hz).
            reward += self.max_steps - self.steps
        done = success or self._lost > round(2.0 / self.dt)
        return (
            reward,
            done,
            {
                "is_success": self._reached >= self.n_waypoints,
                "waypoints": self._reached,
            },
        )


class StaticDefendersEnv(GoToBallShootEnv):
    """Score past six stationary yellow robots parked in front of the goal.

    Touching a defender at all ends the episode at -100: they are parked, so any contact is
    the agent driving into them.
    """

    n_yellow = 6

    def _place(self) -> None:
        super()._place()
        # A staggered wall between the ball and the mouth: two ranks, so no straight shot
        # from the centre is free but the flanks stay open.
        for i, r in enumerate(self.robots[self.n_blue :]):
            rank, slot = divmod(i, 3)
            r.place(
                self.data,
                S.FIELD_HALF_X - 1.2 - 0.8 * rank,
                (slot - 1) * 0.7 + 0.35 * rank,
                np.pi,
            )

    def _reward_done(self, action):
        reward, done, info = super()._reward_done(action)
        if not done and any(
            self._touching(j) for j in range(self.n_blue, len(self.robots))
        ):
            reward -= KICK_PENALTY
            done = True
            info["collision"] = True
        return reward, done, info


class ContestedPossessionEnv(_BallTask):
    """Take the ball off a yellow robot that is sitting on it with its dribbler running.

    Contact is how the ball is won, so touching the opponent is allowed; crashing into it
    (SSL rules 8.4.2, the agent the faster robot) ends the episode at -100.
    """

    n_yellow = 1
    hold_seconds = 2.0

    def _place(self) -> None:
        taken: list[tuple[float, float]] = []
        ox, oy = self.sample_pose(taken, clearance=1.0)
        opp = self.robots[self.n_blue]
        opp.place(self.data, ox, oy, self.np_random.uniform(-np.pi, np.pi))
        x, y = self.sample_pose(taken, clearance=1.0)
        self.robots[0].place(self.data, x, y, self.np_random.uniform(-np.pi, np.pi))

    def _reset_task(self) -> None:
        # The premise of the task is that the opponent already has the ball, so seat it in
        # their mouth the way the mechanism actually takes one -- by rolling it in.
        self.seat_ball(self.robots[self.n_blue])
        self._prev = self._to_ball()
        self._held = 0
        self._was_touching = False
        self._prev_v = self._velocities()

    def _script(self) -> None:
        self.data.ctrl[self.robots[self.n_blue].dribble] = S.DRIBBLER_OMEGA

    def _velocities(self) -> tuple[np.ndarray, np.ndarray]:
        return tuple(
            np.array(self.robots[i].velocity(self.data)[:2]) for i in (0, self.n_blue)
        )

    def _crashed(self) -> bool:
        """A crashing foul by the agent, judged at the first step of a contact.

        The velocities are the ones going into the step the contact appeared in: by the end
        of a 1/15 s step the impact has already stopped the robots.
        """
        touching = self._touching(self.n_blue)
        first = touching and not self._was_touching
        self._was_touching = touching
        v_a, v_o = self._prev_v
        self._prev_v = self._velocities()
        if not first:
            return False
        line = _unit(
            self.data.xpos[self.robots[self.n_blue].body][:2]
            - self.data.xpos[self._agent().body][:2]
        )
        closing = abs(float((v_a - v_o) @ line))
        return closing > CRASH_SPEED and np.linalg.norm(v_a) > np.linalg.norm(v_o)

    def _reward_done(self, action):
        d = self._to_ball()
        held = self.holds(0)
        reward = (
            (self._prev - d)
            + FACING * self._facing()
            + float(held)
            + self._energy(action)
        )
        self._prev = d
        self._held = self._held + 1 if held else 0
        info = {}
        if self._crashed():
            reward -= OUT_PENALTY
            info["crash"] = True
            return reward, True, info | {"is_success": False}
        won = self._held >= round(self.hold_seconds / self.dt)
        if won:
            reward += BONUS
        return reward, won, info | {"is_success": won}


class PassEnduranceEnv(_KickTask):
    """Collect the ball and pass it to a stationary team-mate."""

    n_blue = 2

    def _place(self) -> None:
        taken: list[tuple[float, float]] = []
        x, y = self.sample_pose(taken, clearance=1.5)
        self.robots[0].place(self.data, x, y, self.np_random.uniform(-np.pi, np.pi))
        rx, ry = self.sample_pose(taken, clearance=1.5)
        # The receiver faces the passer, which is what makes a pass catchable at all.
        self.robots[1].place(self.data, rx, ry, float(np.arctan2(y - ry, x - rx)))
        bx, by = self.sample_pose(taken, clearance=0.5)
        self.ball.place(self.data, bx, by)

    def _target_point(self) -> np.ndarray:
        return np.array(self.data.xpos[self.robots[1].body][:2])

    def _extra_facing(self) -> float:
        # Face the receiver: cos between the passer's front and passer -> receiver. Not
        # the two fronts against each other, which is -1 when they face each other.
        r = self._agent()
        to_mate = _unit(
            self.data.xpos[self.robots[1].body][:2] - self.data.xpos[r.body][:2]
        )
        return float(to_mate @ r.forward(self.data)[:2])

    def _script(self) -> None:
        self.data.ctrl[self.robots[1].dribble] = S.DRIBBLER_OMEGA

    def _done(self) -> bool:
        return self.holds(1)


# ------------------------------------------------------------------ scripted policy


def goto_ball_policy(obs: np.ndarray, kp: float = 2.0, kw: float = 1.21) -> np.ndarray:
    """A P-controller that drives robot 0 at the ball, in the tier-0 action layout.

    This exists so the smoke test can answer the one question random actions cannot: is
    the reward collectable at all? A task nobody can solve looks identical to a working
    one until you try to train it.
    """
    ball = np.array([obs[0] * S.FIELD_HALF_X, obs[1] * S.FIELD_HALF_Y])
    pos = np.array([obs[6] * S.FIELD_HALF_X, obs[7] * S.FIELD_HALF_Y])
    sin, cos = float(obs[8]), float(obs[9])
    delta = ball - pos
    # Actions are body-frame (see base._drive_robot), so rotate the world error into it.
    local = np.array(
        [cos * delta[0] + sin * delta[1], -sin * delta[0] + cos * delta[1]]
    )
    heading = float(np.arctan2(local[1], local[0]))
    # kw is in action units, so it scales with the yaw ctrlrange: 1.21 of 37.1 rad/s is the
    # 45 rad/s per rad that 1.5 of 30 was before tickets 13 and 12 raised the ceiling.
    return np.clip([kp * local[0], kp * local[1], kw * heading], -1.0, 1.0).astype(
        np.float32
    )


WHEEL_R = 0.030089  # m, effective rolling radius of the direct-drive wheel (ticket 01)


def wheel_command(env: BaseEnv, i: int, vx: float, vy: float, w: float) -> np.ndarray:
    """Body-frame (vx m/s, vy m/s, w rad/s) as robot `i`'s four tier-1 wheel actions.

    The rolling constraint of each wheel, read off the compiled model: a wheel spinning at
    +q about its axis `a` carries its hub along `a x z` at q * WHEEL_R, so it must turn at
    (roll . v_hub) / WHEEL_R. Scaled down as a whole when a wheel would pass its ctrlrange, so
    the direction survives saturation. Kinematics only -- slip is what tier 1 adds.
    """
    r, m, d = env.robots[i], env.model, env.data
    R = d.xmat[r.body].reshape(3, 3)
    q = np.empty(len(r.drive))
    for k, act in enumerate(r.drive):
        j = m.actuator_trnid[act, 0]
        b = m.jnt_bodyid[j]
        p = R.T @ (d.xpos[b] - d.xpos[r.body])
        axis = R.T @ (d.xmat[b].reshape(3, 3) @ m.jnt_axis[j])
        roll = np.cross(axis, (0.0, 0.0, 1.0))
        q[k] = roll @ (vx - w * p[1], vy + w * p[0], 0.0) / WHEEL_R
    action = q / r.drive_range[:, 1]
    return (action / max(1.0, float(np.abs(action).max()))).astype(np.float32)


def main(argv=None) -> int:
    import argparse

    import gymnasium as gym

    import mjsoccer

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", default="GoToBall-v0")
    # Defaults to 0 rather than to the env default of 1: this demo is driven by
    # goto_ball_policy, which is a body-frame velocity controller and says nothing useful
    # to tier 1's four wheel servos.
    ap.add_argument("--tier", type=int, choices=(0, 1), default=0)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--view", action="store_true", help="open the MuJoCo viewer")
    ap.add_argument(
        "--list", action="store_true", help="print the registered ids and exit"
    )
    args = ap.parse_args(argv)

    if args.list:
        for env_id in mjsoccer.ENV_IDS:
            print(env_id)
        return 0

    env = gym.make(args.env, tier=args.tier, render_mode="human" if args.view else None)
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        total, steps = 0.0, 0
        while True:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            # The policy returns the three tier-0 drive dims. On tier 1 those land on three
            # of the four wheels, which under-drives it -- fine for a smoke test.
            action[:3] = goto_ball_policy(obs)
            action[-1] = 1.0  # dribbler on; kick and chip stay at rest
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            steps += 1
            if terminated or truncated:
                print(f"ep {ep}: {steps} steps, return {total:+.2f}, {info}")
                break
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
