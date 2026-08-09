"""The task environments.

Each one supplies a reset layout, whatever the scripted robots do, and a reward. The
rewards are potential-based: every shaping term is the *decrease* in some distance over
one step, so the total shaping a policy can collect on an episode is bounded by the
starting distance and no cycle pays.

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
ENERGY = 0.001  # per-step penalty on ||action||^2, enough to discourage buzzing


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros_like(v)


class _BallTask(BaseEnv):
    """Shared machinery for every task whose first job is to reach the ball."""

    def _agent(self) -> S.RobotIds:
        return self.robots[0]

    def _to_ball(self) -> float:
        r = self._agent()
        return float(np.linalg.norm(self.ball.pos(self.data)[:2] - self.data.xpos[r.body][:2]))

    def _facing(self) -> float:
        """cos of the angle between the robot's front and the direction to the ball."""
        r = self._agent()
        d = _unit(self.ball.pos(self.data)[:2] - self.data.xpos[r.body][:2])
        return float(d @ r.forward(self.data)[:2])

    def _energy(self, action: np.ndarray) -> float:
        return -ENERGY * float(np.square(action).sum())


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
        self._prev = self._to_ball()

    def _reward_done(self, action):
        d = self._to_ball()
        reward = (self._prev - d) + 0.05 * self._facing() + self._energy(action)
        self._prev = d
        arrived = d < REACH and self._facing() > 0.9
        if arrived:
            reward += 10.0
        return reward, arrived, {"is_success": arrived, "dist": d}


class GoToBallShootEnv(_BallTask):
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

    def _goal_dist(self) -> float:
        p = self.ball.pos(self.data)[:2]
        return float(np.linalg.norm(p - (S.FIELD_HALF_X, np.clip(p[1], -S.GOAL_HALF_Y, S.GOAL_HALF_Y))))

    def _reset_task(self) -> None:
        self._prev, self._prev_goal = self._to_ball(), self._goal_dist()

    def _reward_done(self, action):
        d, g = self._to_ball(), self._goal_dist()
        # The ball term dominates: moving the ball goalward is the task, closing on the
        # ball is only how you get to do it.
        reward = 0.2 * (self._prev - d) + (self._prev_goal - g) + self._energy(action)
        self._prev, self._prev_goal = d, g
        scored = S.in_goal(self.ball.pos(self.data))
        if scored:
            reward += 20.0
        return reward, scored, {"is_success": scored, "goal": scored}


class DribblingEnv(_BallTask):
    """Carry the ball through a sequence of waypoints without losing it.

    The agent can kick or chip its own carried ball away -- the controls are always in the
    action space. It is not worth doing: the shaping below is gated on `holds(0)`, so the
    step the ball leaves the mouth is the step progress stops paying.
    """

    extra_obs = 2  # the current waypoint, in the robot's own frame
    n_waypoints = 4
    waypoint_tol = 0.3

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

    def _to_target(self) -> float:
        return float(np.linalg.norm(self.ball.pos(self.data)[:2] - self._target))

    def _task_obs(self):
        r = self._agent()
        R = self.data.xmat[r.body].reshape(3, 3)[:2, :2]
        local = R.T @ (self._target - self.data.xpos[r.body][:2])
        return local / (S.FIELD_HALF_X, S.FIELD_HALF_Y)

    def _reward_done(self, action):
        held = self.holds(0)
        d = self._to_target()
        # Progress only counts while the ball is actually held; otherwise a policy can
        # farm the shaping term by kicking the ball at the waypoint and walking away.
        reward = (self._prev - d) * float(held) + 0.01 * float(held) + self._energy(action)
        self._prev = d
        self._lost = 0 if held else self._lost + 1
        if d < self.waypoint_tol and held:
            self._reached += 1
            reward += 5.0
            self._target = np.array(self.sample_pose([], clearance=0.0))
            self._prev = self._to_target()
        done = self._reached >= self.n_waypoints or self._lost > round(2.0 / self.dt)
        return reward, done, {"is_success": self._reached >= self.n_waypoints, "waypoints": self._reached}


class StaticDefendersEnv(GoToBallShootEnv):
    """Score past six stationary yellow robots parked in front of the goal."""

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


class ContestedPossessionEnv(_BallTask):
    """Take the ball off a yellow robot that is sitting on it with its dribbler running."""

    n_yellow = 1
    hold_seconds = 0.5

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

    def _script(self) -> None:
        self.data.ctrl[self.robots[self.n_blue].dribble] = S.DRIBBLER_OMEGA

    def _reward_done(self, action):
        d = self._to_ball()
        held = self.holds(0)
        reward = (self._prev - d) + 0.05 * float(held) + self._energy(action)
        self._prev = d
        self._held = self._held + 1 if held else 0
        won = self._held >= round(self.hold_seconds / self.dt)
        if won:
            reward += 10.0
        return reward, won, {"is_success": won}


class PassEnduranceEnv(_BallTask):
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

    def _reset_task(self) -> None:
        self._prev, self._prev_mate = self._to_ball(), self._to_mate()

    def _to_mate(self) -> float:
        return float(
            np.linalg.norm(self.ball.pos(self.data)[:2] - self.data.xpos[self.robots[1].body][:2])
        )

    def _script(self) -> None:
        self.data.ctrl[self.robots[1].dribble] = S.DRIBBLER_OMEGA

    def _reward_done(self, action):
        d, m = self._to_ball(), self._to_mate()
        reward = 0.2 * (self._prev - d) + (self._prev_mate - m) + self._energy(action)
        self._prev, self._prev_mate = d, m
        received = self.holds(1)
        if received:
            reward += 10.0
        return reward, received, {"is_success": received}


# ------------------------------------------------------------------ scripted policy


def goto_ball_policy(obs: np.ndarray, kp: float = 2.0, kw: float = 1.5) -> np.ndarray:
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
    local = np.array([cos * delta[0] + sin * delta[1], -sin * delta[0] + cos * delta[1]])
    heading = float(np.arctan2(local[1], local[0]))
    return np.clip([kp * local[0], kp * local[1], kw * heading], -1.0, 1.0).astype(np.float32)


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
    ap.add_argument("--list", action="store_true", help="print the registered ids and exit")
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
