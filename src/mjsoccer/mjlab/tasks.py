"""The tasks in torch, one class per CPU class in `mjsoccer.tasks`, of the same name.

A port of `mjsoccer.mjx.tasks`; config is read off the CPU class. All six are ported.
Scripted robots are as on the CPU: yellow ones get `ctrl = 0`, PassEndurance's receiver
only its dribbler and its passer only the spin of its drive. Placement redraws as MJX
does: each scattered robot takes the first of SCATTER_CANDIDATES clear draws
(`first_free`), the other redraw loops their first free candidate (`first_where`).
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
# PassEndurance's receiver x, rejected up to 2/3 of draws: (2/3)^64 ~ 5e-12 run out (MJX's).
PASS_CANDIDATES = 64


def first_where(cand, free, roomy):
    """`first_free` for any rejection test: per world, the first of the candidates
    (N, K, ...) whose `free` (N, K) holds, else the one with the largest `roomy` (N, K),
    and `overlap` (N,) True."""
    ok = free.any(1)
    pick = torch.where(ok, free.byte().argmax(1), roomy.argmax(1))
    return cand[torch.arange(cand.shape[0], device=cand.device), pick], ~ok


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

    def init_task(self, d):
        return {"prev": self.to_ball(d)}

    def reward_done(self, d, w):
        dist = self.to_ball(d)
        arrived = dist < 0.2
        shaping = (w["prev"] - dist).clamp(-1, 1) / cpu.GOTO_STEP_SCALE
        return arrived.to(d.qpos.dtype) + shaping, arrived, {**w, "prev": dist}


class GoToBallShootEnv(_ShootTask):
    """`tasks.GoToBallShootEnv`: ball and robot uniform in the attacking half, the robot
    redrawn while it is within `sep` of the ball on both axes."""

    cpu_cls = cpu.GoToBallShootEnv
    n_uniform = 2 + 2 * SCATTER_CANDIDATES + 1

    def place(self, qpos, qvel, u):
        qpos, qvel = qpos.clone(), qvel.clone()
        k, n = SCATTER_CANDIDATES, u.shape[0]
        lo = torch.tensor([0.3, -HALF_Y + 0.1], dtype=u.dtype, device=u.device)
        hi = torch.tensor([HALF_X - scene.PENALTY_DEPTH - 0.3, HALF_Y - 0.1], dtype=u.dtype,
                          device=u.device)
        ball = lo + u[:, :2] * (hi - lo)
        self.place_ball(qpos, qvel, ball[:, 0], ball[:, 1])
        cand = lo + u[:, 2 : 2 + 2 * k].reshape(n, k, 2) * (hi - lo)
        cand[:, 0, 1] *= -1.0  # the CPU's first y is drawn negated
        sep = (0.0215 + 0.09) * 1.1  # rSoccer's (ball + robot radius) * 1.1, per axis
        gap = (cand - ball[:, None]).abs().amax(-1)
        xy, overlap = first_where(cand, gap >= sep, gap)
        self.place_robot(qpos, qvel, self.robots[0], xy[:, 0], xy[:, 1],
                         u[:, -1] * 2 * math.pi)
        return qpos, qvel, overlap


class StaticDefendersEnv(_ShootTask):
    """`tasks.StaticDefendersEnv`: the robot at the centre, the ball redrawn out of the
    penalty area, six parked yellow robots scattered over the attacking half."""

    cpu_cls = cpu.StaticDefendersEnv
    n_uniform = 2 * SCATTER_CANDIDATES + cpu.StaticDefendersEnv.n_yellow * (
        2 * SCATTER_CANDIDATES + 1)

    def place(self, qpos, qvel, u):
        qpos, qvel = qpos.clone(), qvel.clone()
        k, n = SCATTER_CANDIDATES, u.shape[0]
        lo = torch.tensor([0.2, -HALF_Y + 0.1], dtype=u.dtype, device=u.device)
        hi = torch.tensor([HALF_X - 0.1, HALF_Y - 0.1], dtype=u.dtype, device=u.device)
        self.place_robot(qpos, qvel, self.robots[0], 0.0, 0.0, 0.0)
        cand = lo + u[:, : 2 * k].reshape(n, k, 2) * (hi - lo)
        # None free (~1e-7): the candidate nearest the halfway line, the farthest from it.
        ball, bad = first_where(cand, ~_in_gk_area(cand[..., 0], cand[..., 1]), -cand[..., 0])
        self.place_ball(qpos, qvel, ball[:, 0], ball[:, 1])
        taken = torch.stack([ball, torch.zeros_like(ball)], 1)
        overlap = self.scatter(qpos, qvel, self.n_blue, taken, u[:, 2 * k :], lo, hi)
        return qpos, qvel, overlap | bad


class DribblingEnv(_BallTask):
    """`tasks.DribblingEnv`: the zigzag through four yellow robots on -x, the ball seated on
    the blue robot, `count` checkpoints passed."""

    cpu_cls = cpu.DribblingEnv
    seat_robot = 0

    def place(self, qpos, qvel, u):
        qpos, qvel = qpos.clone(), qvel.clone()
        self.place_robot(qpos, qvel, self.robots[0], 0.0, 0.0, math.pi)
        for r, x in zip(self.robots[self.n_blue :], self.cpu_cls.nodes):
            self.place_robot(qpos, qvel, r, x, 0.0, math.pi)
        return qpos, qvel, torch.zeros(u.shape[0], dtype=torch.bool, device=u.device)

    def init_task(self, d):
        n = d.qpos.shape[0]
        return {"count": torch.zeros(n, dtype=torch.int32, device=self.device),
                "last_ball": self.ball_xy(d).clone()}

    def task_obs(self, d, w):
        return ((w["count"] / 6) * 2 - 1)[:, None].to(d.qpos.dtype)

    def reward_done(self, d, w):
        n0, n1, n2, n3 = self.cpu_cls.nodes
        m = self.cpu_cls.margin
        (rx, ry), (bx, by) = self.xy(d, 0).unbind(-1), self.ball_xy(d).unbind(-1)
        ly, c = w["last_ball"][:, 1], w["count"]
        down, up = (ly >= 0) & (0 > by), (ly < 0) & (0 <= by)
        out = (rx < n3 - m) | (rx > m) | (ry.abs() > m)
        even = (c >= 2) & (c % 2 == 0)
        in_last = (n3 < bx) & (bx < n2)
        passed = ~out & (
            ((c == 0) & (n1 < bx) & (bx < n0) & down)
            | ((c == 1) & (n2 < bx) & (bx < n1) & up)
            | (even & in_last & down)
            | ((c >= 3) & (c % 2 == 1) & (n3 - m < bx) & (bx < n3) & up)
        )
        c = c + passed.to(c.dtype)
        backwards = ~out & even & in_last & ~down & up
        done = self.yellow_moved(d, 0.05) | out | backwards | (passed & even & (c == 7))
        w = {**w, "count": c, "last_ball": torch.stack([bx, by], -1)}
        return passed.to(d.qpos.dtype), done, w


class PassEnduranceEnv(_BallTask):
    """`tasks.PassEnduranceEnv`: a passer that can only turn, the ball seated on it, a
    team-mate receiver whose dribbler is always on."""

    cpu_cls = cpu.PassEnduranceEnv
    n_uniform = 2 + PASS_CANDIDATES
    seat_robot = 0

    def place(self, qpos, qvel, u):
        qpos, qvel = qpos.clone(), qvel.clone()
        b = -1.5 + 3.0 * u[:, :2]
        f = torch.where(b[:, 1] > 0, 1.0, -1.0).to(u.dtype)
        sx, sy = b[:, 0], b[:, 1] + 0.115 * f
        self.place_robot(qpos, qvel, self.robots[0], sx, sy, -f * math.pi / 2)
        cand = -1.5 + 3.0 * u[:, 2:]
        gap = (cand - b[:, :1]).abs()
        rx, overlap = first_where(cand, gap >= 1, gap)
        ry = -b[:, 1]
        self.place_robot(qpos, qvel, self.robots[1], rx, ry,
                         torch.atan2(ry - sy, rx - sx) + math.pi)
        return qpos, qvel, overlap

    def to_receiver(self, d):
        return (self.ball_xy(d) - self.xy(d, 1)).norm(dim=-1)

    def init_task(self, d):
        n = d.qpos.shape[0]
        return {"stopped": torch.zeros(n, dtype=torch.int32, device=self.device),
                "prev": self.to_receiver(d)}

    def shape_drive(self, cmd):
        out = torch.zeros_like(cmd)
        out[:, 2] = cmd[:, 2]
        return out

    def script(self, d, ctrl):
        ctrl[:, self.robots[1].dribble] = scene.DRIBBLER_OMEGA
        return ctrl

    def reward_done(self, d, w):
        dist = self.to_receiver(d)
        received = w["grace"][:, 1] > 0
        reward = torch.where(received, 1.0,
                             (w["prev"] - dist).clamp(-1, 1) / cpu.BALL_GRAD_SCALE)
        # `_wrong_ball`: whole centimetres, truncated as numpy's int cast does.
        ball, shoot, recv = ((p * 100).to(torch.int32)
                             for p in (self.ball_xy(d), self.xy(d, 0), self.xy(d, 1)))
        inside = ((torch.minimum(recv, shoot) <= ball)
                  & (ball <= torch.maximum(recv, shoot))).all(-1)
        stopped = torch.where((w["prev"] - dist).abs() < 0.01, w["stopped"] + 1, 0)
        wrong = (stopped > self.cpu_cls.stopped_limit) | ~inside
        reward = torch.where(wrong, reward - 1.0, reward)
        return reward, received | wrong, {**w, "prev": dist, "stopped": stopped.to(torch.int32)}
