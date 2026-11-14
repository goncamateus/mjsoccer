"""The tasks on MJX, one class per CPU class in `mjsoccer.tasks`, of the same name.

Config (`energy_steps`, `collision_speed`, `n_yellow`, ...) is read off the CPU class;
the logic is written twice and held together by tests/test_mjx_task_logic.py. Ported so
far: ContestedPossession and GoToBall (ticket 11 adds the rest).

Placement: the CPU's `_scatter` resamples a robot's (x, y) until it is `min_dist` clear of
everything placed, with no cap. Here each robot gets SCATTER_CANDIDATES (x, y) draws and
takes the first clear one (`first_free`), which is the CPU's answer on the same draws.
With 7 robots and a ball on the Div A field one draw is rejected ~1 % of the time, so all
eight are rejected ~1e-16 of the time; then the roomiest is kept and the world's first
state says `reset_overlap`.
"""

from __future__ import annotations

import jax.numpy as jnp

from .. import scene as S
from .. import tasks as cpu
from .base import MjxEnv, first_free

HALF_X, HALF_Y = S.FIELD_HALF_X, S.FIELD_HALF_Y
SCATTER_CANDIDATES = 8


def _in_gk_area(x, y):
    """`tasks._in_gk_area`: inside the +x penalty area."""
    return (x > HALF_X - S.PENALTY_DEPTH) & (jnp.abs(y) < S.PENALTY_HALF_Y)


def in_goal(pos, sign: float = 1.0):
    """`scene.in_goal`."""
    return (sign * pos[0] > HALF_X) & (jnp.abs(pos[1]) < S.GOAL_HALF_Y) & (
        pos[2] - S.FLOOR_Z < S.GOAL_HEIGHT
    )


class _BallTask(MjxEnv):
    def to_ball(self, d):
        return jnp.linalg.norm(self.ball_xy(d) - self.xy(d, 0))

    def yellow_moved(self, d, speed):
        """`_BallTask._yellow_moved`: any yellow robot faster than `speed` on x or y."""
        adr = jnp.array([r.qvel_adr for r in self.robots[self.n_blue :]])
        v = d.qvel[adr[:, None] + jnp.arange(2)]
        return (jnp.abs(v).max(-1) > speed).any()

    def scatter(self, d, n, taken, u, lo, hi, min_dist=0.2):
        """`_BallTask._scatter`: robots n.. each at its first candidate `min_dist` clear of
        `taken` (k, 2) and the robots before it, heading uniform. `u` holds, per robot,
        SCATTER_CANDIDATES (x, y) pairs then the heading. Returns (data, any overlap)."""
        k = SCATTER_CANDIDATES
        lo, hi = jnp.asarray(lo), jnp.asarray(hi)
        overlap = jnp.array(False)
        for i, r in enumerate(self.robots[n:]):
            block = u[i * (2 * k + 1) : (i + 1) * (2 * k + 1)]
            xy, bad = first_free(lo + block[:-1].reshape(k, 2) * (hi - lo), taken, min_dist)
            taken = jnp.concatenate([taken, xy[None]])
            overlap = overlap | bad
            d = self.place_robot(d, r, xy[0], xy[1], block[-1] * 2 * jnp.pi)
        return d, overlap


class _ShootTask(_BallTask):
    """`tasks._ShootTask`: rSoccer's goal-scoring reward and endings."""

    def goal_dist(self, d):
        return jnp.linalg.norm(self.ball_xy(d) - jnp.array([HALF_X, 0.0]))

    def init_task(self, d):
        return {"prev": self.to_ball(d), "prev_goal": self.goal_dist(d)}

    def reward_done(self, d, w):
        rx, ry = self.xy(d, 0)
        bx, by = self.ball_xy(d)
        dist, g = self.to_ball(d), self.goal_dist(d)
        speed = self.cpu_cls.collision_speed
        collision = self.yellow_moved(d, speed) if speed is not None else jnp.array(False)
        robot_out = (rx < -0.2) | (jnp.abs(ry) > HALF_Y)
        gk = _in_gk_area(rx, ry)
        ball_out = (bx < 0) | (jnp.abs(by) > HALF_Y)
        past = bx > HALF_X
        energy = jnp.abs(w["drive_cmd"][0]).mean() / self.cpu_cls.energy_steps
        shaping = (
            jnp.clip(w["prev"] - dist, -1, 1) / cpu.BALL_DIST_SCALE
            + jnp.clip(w["prev_goal"] - g, -1, 1) / cpu.BALL_GRAD_SCALE
            - energy
        )
        goal = jnp.where(in_goal(d.xpos[self.ball.body]), cpu.GOAL_REWARD, 0.0)
        reward = jnp.where(robot_out | gk | ball_out, 0.0, jnp.where(past, goal, shaping))
        done = collision | robot_out | gk | ball_out | past
        return reward, done, {**w, "prev": dist, "prev_goal": g}


class ContestedPossessionEnv(_ShootTask):
    """`tasks.ContestedPossessionEnv`: score off a yellow robot sitting on the ball."""

    cpu_cls = cpu.ContestedPossessionEnv
    n_uniform = 2

    def place(self, d, u):
        d = self.place_robot(d, self.robots[0], 0.0, 0.0, 0.0)
        ox = S.PENALTY_DEPTH + u[0] * (HALF_X - 2 * S.PENALTY_DEPTH)
        oy = -S.PENALTY_HALF_Y + u[1] * (2 * S.PENALTY_HALF_Y)
        return self.place_robot(d, self.robots[self.n_blue], ox, oy, jnp.pi), jnp.array(False)

    def reset_physics(self, d):
        return self.seat_ball(d, self.robots[self.n_blue])


class GoToBallEnv(_BallTask):
    """`tasks.GoToBallEnv`: within 0.2 m of the ball, through six parked yellow robots."""

    cpu_cls = cpu.GoToBallEnv
    n_uniform = 2 + (1 + cpu.GoToBallEnv.n_yellow) * (2 * SCATTER_CANDIDATES + 1)

    def place(self, d, u):
        lo = jnp.array([-HALF_X + 0.1, -HALF_Y + 0.1])
        ball = lo + u[:2] * (-2 * lo)
        d = self.place_ball(d, ball[0], ball[1])
        return self.scatter(d, 0, ball[None], u[2:], lo, -lo)

    def reward_done(self, d, w):
        arrived = self.to_ball(d) < 0.2
        return arrived.astype(jnp.float32), arrived, w
