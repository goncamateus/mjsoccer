"""The tasks on MJX, one class per CPU class in `mjsoccer.tasks`, of the same name.

Config (`energy_steps`, `collision_speed`, `n_yellow`, ...) is read off the CPU class;
the logic is written twice and held together by tests/test_mjx_task_logic.py. Ported so
far: ContestedPossession (tickets 10 and 11 add the rest).
"""

from __future__ import annotations

import jax.numpy as jnp

from .. import scene as S
from .. import tasks as cpu
from .base import MjxEnv

HALF_X, HALF_Y = S.FIELD_HALF_X, S.FIELD_HALF_Y


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
        return self.place_robot(d, self.robots[self.n_blue], ox, oy, jnp.pi)

    def reset_physics(self, d):
        return self.seat_ball(d, self.robots[self.n_blue])
