"""The task environments: robocin/rSoccer's six SSL tasks, ported reward for reward.

Each one supplies a reset layout, whatever the scripted robots do, and a reward. Rewards,
episode endings and reset layouts are rSoccer's (ssl_go_to_ball, ssl_go_to_ball_shoot and
ssl_hw_challenge/*), on our field: the formulas read the field's own dimensions, so they
stretch to the Div A carpet the scene builds. What stays ours is the physics, the control
rate and the action space -- every env exposes the same per-robot action, drive, kick,
chip, dribble, whether or not its reward cares, so one action head transfers across all six.

    id                       robots           action
    GoToBall-v0              1 blue + 6 yel   6
    GoToBallShoot-v0         1 blue           6
    Dribbling-v0             1 blue + 4 yel   6
    StaticDefenders-v0       1 blue + 6 yel   6
    ContestedPossession-v0   1 blue + 1 yel   6
    PassEndurance-v0         2 blue           6
"""

from __future__ import annotations

import numpy as np

from . import scene as S
from .base import BaseEnv

HALF_X, HALF_Y = S.FIELD_HALF_X, S.FIELD_HALF_Y
# rSoccer's shaping scales: the largest robot->ball step on the field is 1 after the first,
# and the ball->goal one is a quarter of the half-field diagonal.
BALL_DIST_SCALE = float(np.hypot(2 * HALF_Y, HALF_X))
BALL_GRAD_SCALE = float(np.hypot(HALF_Y, HALF_X)) / 4
GOAL_REWARD = 5.0


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros_like(v)


def _in_gk_area(x: float, y: float) -> bool:
    """Inside the +x (attacked) penalty area."""
    return x > HALF_X - S.PENALTY_DEPTH and abs(y) < S.PENALTY_HALF_Y


class _BallTask(BaseEnv):
    """Shared machinery for every task whose first job is to reach the ball."""

    def _agent(self) -> S.RobotIds:
        return self.robots[0]

    def _xy(self, i: int) -> np.ndarray:
        return np.array(self.data.xpos[self.robots[i].body][:2])

    def _ball_xy(self) -> np.ndarray:
        return np.array(self.ball.pos(self.data)[:2])

    def _to_ball(self) -> float:
        return float(np.linalg.norm(self._ball_xy() - self._xy(0)))

    def _facing(self) -> float:
        """cos of the angle between the robot's front and the direction to the ball."""
        d = _unit(self._ball_xy() - self._xy(0))
        return float(d @ self._agent().forward(self.data)[:2])

    def _yellow_moved(self, speed: float) -> bool:
        """rSoccer's collision test: any yellow robot moving faster than `speed` on x or y."""
        return any(
            max(abs(v) for v in r.velocity(self.data)[:2]) > speed
            for r in self.robots[self.n_blue :]
        )

    def _scatter(self, n: int, taken: list, x_range, y_range, min_dist: float = 0.2):
        """rSoccer's KDTree placement: uniform in a box, resampled while within min_dist."""
        for r in self.robots[n:]:
            while True:
                x, y = (self.np_random.uniform(*x_range), self.np_random.uniform(*y_range))
                if all(np.hypot(x - px, y - py) >= min_dist for px, py in taken):
                    break
            taken.append((x, y))
            r.place(self.data, x, y, self.np_random.uniform(0, 2 * np.pi))


class GoToBallEnv(_BallTask):
    """Get within 0.2 m of the ball, through six parked yellow robots: +1 and done.

    rSoccer's SSLGoToBall-v0 (field_type 2, n_robots_yellow 6). Sparse, no facing, no
    shaping. The kicker, chipper and dribbler are in the action space but the reward ignores
    them.
    """

    n_yellow = 6

    def _place(self) -> None:
        xs, ys = (-HALF_X + 0.1, HALF_X - 0.1), (-HALF_Y + 0.1, HALF_Y - 0.1)
        bx, by = self.np_random.uniform(*xs), self.np_random.uniform(*ys)
        self.ball.place(self.data, bx, by)
        self._scatter(0, [(bx, by)], xs, ys)

    def _reward_done(self, action):
        d = self._to_ball()
        arrived = d < 0.2
        return float(arrived), arrived, {"is_success": arrived, "dist": d}


class _ShootTask(_BallTask):
    """rSoccer's goal-scoring reward, shared by GoToBallShoot, StaticDefenders and
    ContestedPossession.

    Ends when the robot leaves the attacking half (x < -0.2) or the field's width, enters
    the penalty area, the ball leaves the attacking half, or crosses the +x goal line --
    +5 inside the mouth, 0 wide. Every other step pays
        clip(robot->ball closing, +-1) / BALL_DIST_SCALE
      + clip(ball->goal-centre closing, +-1) / BALL_GRAD_SCALE
      - sum |wheel speed| / (4 * max wheel speed * energy_steps).
    """

    energy_steps = 1200  # rSoccer's `max_steps` in its energy scale, not the horizon
    collision_speed: float | None = None  # end on any yellow robot moving this fast

    def _goal_dist(self) -> float:
        return float(np.linalg.norm(self._ball_xy() - (HALF_X, 0.0)))

    def _reset_task(self) -> None:
        self._prev, self._prev_goal = self._to_ball(), self._goal_dist()

    def _energy(self, action: np.ndarray) -> float:
        # ponytail: the commanded drive, as a fraction of its range, stands in for rSoccer's
        # measured |wheel speed| / 160 rad/s; the chassis commands (vx, vy, w), not wheels.
        return float(np.abs(action[: self.drive_dim]).mean()) / self.energy_steps

    def _reward_done(self, action):
        (rx, ry), (bx, by) = self._xy(0), self._ball_xy()
        d, g = self._to_ball(), self._goal_dist()
        reward, done, info = 0.0, False, {}
        if self.collision_speed is not None and self._yellow_moved(self.collision_speed):
            done = info["collision"] = True
        if rx < -0.2 or abs(ry) > HALF_Y:
            done = info["robot_out"] = True
        elif _in_gk_area(rx, ry):
            done = info["gk_area"] = True
        elif bx < 0 or abs(by) > HALF_Y:
            done = info["ball_out"] = True
        elif bx > HALF_X:
            done = True
            info["goal"] = S.in_goal(self.ball.pos(self.data))
            reward = GOAL_REWARD if info["goal"] else 0.0
        else:
            reward = (
                np.clip(self._prev - d, -1, 1) / BALL_DIST_SCALE
                + np.clip(self._prev_goal - g, -1, 1) / BALL_GRAD_SCALE
                - self._energy(action)
            )
        self._prev, self._prev_goal = d, g
        info["is_success"] = bool(info.get("goal", False))
        return float(reward), done, info


class GoToBallShootEnv(_ShootTask):
    """Reach the ball and put it in the +x goal. rSoccer's SSLGoToBallShoot-v0, random_init."""

    def _place(self) -> None:
        xs = (0.3, HALF_X - S.PENALTY_DEPTH - 0.3)
        ys = (-HALF_Y + 0.1, HALF_Y - 0.1)
        u = self.np_random.uniform
        bx, by = u(*xs), u(*ys)
        self.ball.place(self.data, bx, by)
        x, y = u(*xs), -u(*ys)
        sep = (0.0215 + 0.09) * 1.1  # rSoccer's (ball + robot radius) * 1.1, per axis
        while abs(x - bx) < sep and abs(y - by) < sep:
            x, y = u(*xs), u(*ys)
        self.robots[0].place(self.data, x, y, u(0, 2 * np.pi))


class StaticDefendersEnv(_ShootTask):
    """Score past six stationary yellow robots scattered over the attacking half.

    rSoccer's SSLStaticDefenders-v0: no collision rule, the defenders are only in the way.
    """

    n_yellow = 6
    energy_steps = 1000

    def _place(self) -> None:
        xs, ys = (0.2, HALF_X - 0.1), (-HALF_Y + 0.1, HALF_Y - 0.1)
        self.robots[0].place(self.data, 0.0, 0.0, 0.0)
        while True:
            bx, by = self.np_random.uniform(*xs), self.np_random.uniform(*ys)
            if not _in_gk_area(bx, by):
                break
        self.ball.place(self.data, bx, by)
        self._scatter(self.n_blue, [(bx, by), (0.0, 0.0)], xs, ys)


class ContestedPossessionEnv(_ShootTask):
    """Score off a yellow robot that sits on the ball in front of the goal, facing away.

    rSoccer's SSLContestedPossession-v0: the episode ends the moment the yellow robot is
    shoved faster than 0.1 m/s on x or y. It gets no commands, so its dribbler is off.
    """

    n_yellow = 1
    collision_speed = 0.1

    def _place(self) -> None:
        u = self.np_random.uniform
        self.robots[0].place(self.data, 0.0, 0.0, 0.0)
        ox = u(S.PENALTY_DEPTH, HALF_X - S.PENALTY_DEPTH)
        oy = u(-S.PENALTY_HALF_Y, S.PENALTY_HALF_Y)
        self.robots[self.n_blue].place(self.data, ox, oy, np.pi)

    def _reset_task(self) -> None:
        # rSoccer puts the ball 0.1 m in front of the yellow robot's centre, in its mouth.
        # Rolled in instead: a ball placed touching the bar is a compressed spring here.
        self.seat_ball(self.robots[self.n_blue])
        super()._reset_task()


class DribblingEnv(_BallTask):
    """Carry the ball through a zigzag between four yellow robots in a line on -x.

    rSoccer's SSLDribbling-v0. The gaps are checkpoints: the ball crossing y = 0 between
    two robots, in the right direction, pays +1; after the second, it shuttles around the
    last robot until 7 are passed. Ends on 7, on crossing the last gap backwards, on the
    robot leaving x in [-3, 1] or |y| <= 1, or on any yellow robot moving faster than 0.05 m/s.
    """

    n_yellow = 4
    nodes = (-0.5, -1.0, -1.5, -2.0)
    margin = 1.0
    extra_obs = 1  # checkpoints passed, (count / 6) * 2 - 1 as rSoccer observes it
    extra_obs_bound = 4.0 / 3.0

    def _place(self) -> None:
        self.robots[0].place(self.data, 0.0, 0.0, np.pi)
        for r, x in zip(self.robots[self.n_blue :], self.nodes):
            r.place(self.data, x, 0.0, np.pi)

    def _reset_task(self) -> None:
        self.seat_ball(self.robots[0])  # rSoccer: ball at (-0.1, 0), in the mouth
        self._count = 0
        self._last_ball = self._ball_xy()

    def _task_obs(self):
        return ((self._count / 6) * 2 - 1,)

    def _reward_done(self, action):
        n0, n1, n2, n3 = self.nodes
        (rx, ry), (bx, by) = self._xy(0), self._ball_xy()
        ly = self._last_ball[1]
        down, up = ly >= 0 > by, ly < 0 <= by
        reward = 0.0
        done = self._yellow_moved(0.05)
        if rx < n3 - self.margin or rx > self.margin or abs(ry) > self.margin:
            done = True
        elif self._count == 0:
            if n1 < bx < n0 and down:
                reward, self._count = 1.0, 1
        elif self._count == 1:
            if n2 < bx < n1 and up:
                reward, self._count = 1.0, 2
        elif self._count % 2 == 0:
            if n3 < bx < n2:
                if down:
                    reward, self._count = 1.0, self._count + 1
                    done = done or self._count == 7
                elif up:
                    done = True
        elif n3 - self.margin < bx < n3 and up:
            reward, self._count = 1.0, self._count + 1
        self._last_ball = np.array((bx, by))
        return reward, done, {"is_success": self._count >= 7, "checkpoints": self._count}


class PassEnduranceEnv(_BallTask):
    """Pass the ball from a robot that can only turn to a stationary team-mate.

    rSoccer's SSLPassEndurance-v0. The passer starts with the ball in its mouth and takes
    no translation: its drive is reduced to a spin. +1 and done when the receiver holds the
    ball; otherwise clip(ball->receiver closing, +-1) / BALL_GRAD_SCALE a step. -1 and done
    on a wrong ball: out of the box the two robots span (in whole centimetres, as rSoccer
    compares), or its distance to the receiver changing by under 0.01 m for 21 steps.
    """

    n_blue = 2
    stopped_limit = 20

    def _place(self) -> None:
        u = self.np_random.uniform
        bx, by = u(-1.5, 1.5), u(-1.5, 1.5)
        f = 1.0 if by > 0 else -1.0
        sx, sy = bx, by + 0.115 * f
        # 270 deg when the ball is below: rSoccer's thetas, so the passer faces the ball.
        self.robots[0].place(self.data, sx, sy, -np.pi / 2 if f > 0 else np.pi / 2)
        rx = u(-1.5, 1.5)
        while abs(rx - bx) < 1:
            rx = u(-1.5, 1.5)
        ry = -by
        self.robots[1].place(self.data, rx, ry, float(np.arctan2(ry - sy, rx - sx) + np.pi))

    def _reset_task(self) -> None:
        self.seat_ball(self.robots[0])
        self._stopped = 0
        self._prev = self._to_receiver()

    def _to_receiver(self) -> float:
        return float(np.linalg.norm(self._ball_xy() - self._xy(1)))

    def _drive_robot(self, index: int, a: np.ndarray) -> None:
        a = a.copy()
        a[:2] = 0.0  # (vx, vy, w): keep the spin
        super()._drive_robot(index, a)

    def _script(self) -> None:
        self.data.ctrl[self.robots[1].dribble] = S.DRIBBLER_OMEGA

    def _wrong_ball(self, dist: float) -> bool:
        ball, shoot, recv = (
            np.array(p * 100, dtype=int) for p in (self._ball_xy(), self._xy(0), self._xy(1))
        )
        inside = all(
            min(recv[k], shoot[k]) <= ball[k] <= max(recv[k], shoot[k]) for k in (0, 1)
        )
        self._stopped = self._stopped + 1 if abs(self._prev - dist) < 0.01 else 0
        return self._stopped > self.stopped_limit or not inside

    def _reward_done(self, action):
        dist = self._to_receiver()
        received = self.holds(1)
        if received:
            reward, done = 1.0, True
        else:
            reward, done = float(np.clip(self._prev - dist, -1, 1)) / BALL_GRAD_SCALE, False
        if self._wrong_ball(dist):
            reward, done = reward - 1.0, True
        self._prev = dist
        return reward, done, {"is_success": received}


# ------------------------------------------------------------------ scripted policy


def goto_ball_policy(obs: np.ndarray, kp: float = 2.0, kw: float = 1.21) -> np.ndarray:
    """A P-controller that drives robot 0 at the ball: the (vx, vy, omega) drive dims.

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


def main(argv=None) -> int:
    import argparse

    import gymnasium as gym

    import mjsoccer

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", default="GoToBall-v0")
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

    env = gym.make(args.env, render_mode="human" if args.view else None)
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        total, steps = 0.0, 0
        while True:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
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
