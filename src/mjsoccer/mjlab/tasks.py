"""The tasks in torch, one class per CPU class in `mjsoccer.tasks`, of the same name.

A port of `mjsoccer.mjx.tasks`; config is read off the CPU class. ContestedPossession and
GoToBall are ported so far; the other four follow them. Placement redraws as MJX does:
each scattered robot takes the first of SCATTER_CANDIDATES clear draws (`first_free`).
"""

from __future__ import annotations

import math

import torch

from .. import scene
from .. import tasks as cpu
from .base import TaskLogic, first_free

HALF_X, HALF_Y = scene.FIELD_HALF_X, scene.FIELD_HALF_Y
# The same as `mjsoccer.mjx.tasks` (not imported: jax); its module docstring bounds the
# worlds where all eight are rejected at ~1e-16.
SCATTER_CANDIDATES = 8


def _in_gk_area(x, y):
    """`tasks._in_gk_area`: inside the +x penalty area."""
    return (x > HALF_X - scene.PENALTY_DEPTH) & (y.abs() < scene.PENALTY_HALF_Y)


def in_goal(pos, sign: float = 1.0):
    """`scene.in_goal`, on (N, 3)."""
    return (sign * pos[:, 0] > HALF_X) & (pos[:, 1].abs() < scene.GOAL_HALF_Y) & (
        pos[:, 2] - scene.FLOOR_Z < scene.GOAL_HEIGHT
    )


class _BallTask(TaskLogic):
    def to_ball(self, d):
        return (self.ball_xy(d) - self.xy(d, 0)).norm(dim=-1)

    def yellow_moved(self, d, speed):
        """`_BallTask._yellow_moved`: any yellow robot faster than `speed` on x or y."""
        adr = torch.tensor([r.qvel_adr for r in self.robots[self.n_blue :]], device=self.device)
        v = d.qvel[:, adr[:, None] + torch.arange(2, device=self.device)]
        return (v.abs().amax(-1) > speed).any(-1)

    def scatter(self, qpos, qvel, n, taken, u, lo, hi, min_dist=0.2):
        """`_BallTask._scatter`, in place: robots n.. each at its first candidate `min_dist`
        clear of `taken` (N, k, 2) and the robots before it, heading uniform. `u` (N, ...)
        holds, per robot, SCATTER_CANDIDATES (x, y) pairs then the heading. Returns the
        overlap (N,): some robot found no clear candidate (kept the roomiest)."""
        k, n_w = SCATTER_CANDIDATES, u.shape[0]
        lo, hi = (torch.as_tensor(x, dtype=u.dtype, device=u.device) for x in (lo, hi))
        overlap = torch.zeros(n_w, dtype=torch.bool, device=u.device)
        for i, r in enumerate(self.robots[n:]):
            block = u[:, i * (2 * k + 1) : (i + 1) * (2 * k + 1)]
            xy, bad = first_free(lo + block[:, :-1].reshape(n_w, k, 2) * (hi - lo), taken,
                                 min_dist)
            taken = torch.cat([taken, xy[:, None]], 1)
            overlap |= bad
            self.place_robot(qpos, qvel, r, xy[:, 0], xy[:, 1], block[:, -1] * 2 * math.pi)
        return overlap


class _ShootTask(_BallTask):
    """`tasks._ShootTask`: rSoccer's goal-scoring reward and endings."""

    def goal_dist(self, d):
        goal = torch.tensor([HALF_X, 0.0], dtype=d.xpos.dtype, device=self.device)
        return (self.ball_xy(d) - goal).norm(dim=-1)

    def init_task(self, d):
        return {"prev": self.to_ball(d), "prev_goal": self.goal_dist(d)}

    def reward_done(self, d, w):
        (rx, ry), (bx, by) = self.xy(d, 0).unbind(-1), self.ball_xy(d).unbind(-1)
        dist, g = self.to_ball(d), self.goal_dist(d)
        speed = self.cpu_cls.collision_speed
        collision = (self.yellow_moved(d, speed) if speed is not None
                     else torch.zeros_like(rx, dtype=torch.bool))
        robot_out = (rx < -0.2) | (ry.abs() > HALF_Y)
        gk = _in_gk_area(rx, ry)
        ball_out = (bx < 0) | (by.abs() > HALF_Y)
        past = bx > HALF_X
        energy = w["drive_cmd"][:, 0].abs().mean(-1) / self.cpu_cls.energy_steps
        shaping = (
            (w["prev"] - dist).clamp(-1, 1) / cpu.BALL_DIST_SCALE
            + (w["prev_goal"] - g).clamp(-1, 1) / cpu.BALL_GRAD_SCALE
            - energy
        )
        goal = torch.where(in_goal(d.xpos[:, self.ball.body]), cpu.GOAL_REWARD, 0.0)
        reward = torch.where(robot_out | gk | ball_out, 0.0, torch.where(past, goal, shaping))
        done = collision | robot_out | gk | ball_out | past
        return reward, done, {**w, "prev": dist, "prev_goal": g}


class ContestedPossessionEnv(_ShootTask):
    """`tasks.ContestedPossessionEnv`: score off a yellow robot sitting on the ball."""

    cpu_cls = cpu.ContestedPossessionEnv
    n_uniform = 2
    seat_robot = 1  # the yellow robot (n_blue = 1) sits on the ball

    def place(self, qpos, qvel, u):
        qpos, qvel = qpos.clone(), qvel.clone()
        self.place_robot(qpos, qvel, self.robots[0], 0.0, 0.0, 0.0)
        ox = scene.PENALTY_DEPTH + u[:, 0] * (HALF_X - 2 * scene.PENALTY_DEPTH)
        oy = -scene.PENALTY_HALF_Y + u[:, 1] * (2 * scene.PENALTY_HALF_Y)
        self.place_robot(qpos, qvel, self.robots[self.n_blue], ox, oy, math.pi)
        return qpos, qvel, torch.zeros(u.shape[0], dtype=torch.bool, device=u.device)


class GoToBallEnv(_BallTask):
    """`tasks.GoToBallEnv`: within 0.2 m of the ball, through six parked yellow robots."""

    cpu_cls = cpu.GoToBallEnv
    n_uniform = 2 + (1 + cpu.GoToBallEnv.n_yellow) * (2 * SCATTER_CANDIDATES + 1)

    def place(self, qpos, qvel, u):
        qpos, qvel = qpos.clone(), qvel.clone()
        lo = torch.tensor([-HALF_X + 0.1, -HALF_Y + 0.1], dtype=u.dtype, device=u.device)
        ball = lo + u[:, :2] * (-2 * lo)
        self.place_ball(qpos, qvel, ball[:, 0], ball[:, 1])
        overlap = self.scatter(qpos, qvel, 0, ball[:, None], u[:, 2:], lo, -lo)
        return qpos, qvel, overlap

    def reward_done(self, d, w):
        arrived = self.to_ball(d) < 0.2
        return arrived.to(d.qpos.dtype), arrived, w
