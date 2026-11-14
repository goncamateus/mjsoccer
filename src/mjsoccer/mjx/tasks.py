"""The tasks on MJX, one class per CPU class in `mjsoccer.tasks`, of the same name.

Config (`energy_steps`, `collision_speed`, `n_yellow`, ...) is read off the CPU class;
the logic is written twice and held together by tests/test_mjx_task_logic.py. All six
are ported. Scripted robots are as on the CPU: yellow ones get `ctrl = 0`, PassEndurance's
receiver only its dribbler and its passer only the spin of its drive.

Placement: the CPU's `_scatter` resamples a robot's (x, y) until it is `min_dist` clear of
everything placed, with no cap. Here each robot gets SCATTER_CANDIDATES (x, y) draws and
takes the first clear one (`first_free`), which is the CPU's answer on the same draws.
With 7 robots and a ball on the Div A field one draw is rejected ~1 % of the time, so all
eight are rejected ~1e-16 of the time; then the roomiest is kept and the world's first
state says `reset_overlap`. The other redraw loops (GoToBallShoot's robot, StaticDefenders'
ball, PassEndurance's receiver) take the first free candidate the same way (`first_where`).
"""

from __future__ import annotations

import jax.numpy as jnp

from .. import scene as S
from .. import tasks as cpu
from .base import MjxEnv, first_free

HALF_X, HALF_Y = S.FIELD_HALF_X, S.FIELD_HALF_Y
SCATTER_CANDIDATES = 8
# PassEndurance's receiver x is redrawn while within 1 m of the ball's on a 3 m range: up
# to 2/3 of draws rejected, so (2/3)^64 ~ 5e-12 of worlds run out.
PASS_CANDIDATES = 64


def first_where(cand, free, roomy):
    """`first_free` for any rejection test: the first of the candidates (N, ...) whose
    `free` (N,) holds, else the one with the largest `roomy` (N,), and `overlap` True."""
    pick = jnp.where(free.any(), jnp.argmax(free), jnp.argmax(roomy))
    return cand[pick], ~free.any()


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


class GoToBallShootEnv(_ShootTask):
    """`tasks.GoToBallShootEnv`: ball and robot uniform in the attacking half, the robot
    redrawn while it is within `sep` of the ball on both axes."""

    cpu_cls = cpu.GoToBallShootEnv
    n_uniform = 2 + 2 * SCATTER_CANDIDATES + 1

    def place(self, d, u):
        k = SCATTER_CANDIDATES
        lo = jnp.array([0.3, -HALF_Y + 0.1])
        hi = jnp.array([HALF_X - S.PENALTY_DEPTH - 0.3, HALF_Y - 0.1])
        ball = lo + u[:2] * (hi - lo)
        d = self.place_ball(d, ball[0], ball[1])
        cand = lo + u[2 : 2 + 2 * k].reshape(k, 2) * (hi - lo)
        cand = cand.at[0, 1].multiply(-1.0)  # the CPU's first y is drawn negated
        sep = (0.0215 + 0.09) * 1.1  # rSoccer's (ball + robot radius) * 1.1, per axis
        gap = jnp.abs(cand - ball).max(-1)
        xy, overlap = first_where(cand, gap >= sep, gap)
        return self.place_robot(d, self.robots[0], xy[0], xy[1], u[-1] * 2 * jnp.pi), overlap


class StaticDefendersEnv(_ShootTask):
    """`tasks.StaticDefendersEnv`: the robot at the centre, the ball redrawn out of the
    penalty area, six parked yellow robots scattered over the attacking half."""

    cpu_cls = cpu.StaticDefendersEnv
    n_uniform = 2 * SCATTER_CANDIDATES + cpu.StaticDefendersEnv.n_yellow * (
        2 * SCATTER_CANDIDATES + 1)

    def place(self, d, u):
        k = SCATTER_CANDIDATES
        lo, hi = jnp.array([0.2, -HALF_Y + 0.1]), jnp.array([HALF_X - 0.1, HALF_Y - 0.1])
        d = self.place_robot(d, self.robots[0], 0.0, 0.0, 0.0)
        cand = lo + u[: 2 * k].reshape(k, 2) * (hi - lo)
        # None free (~1e-7): the candidate nearest the halfway line, the farthest from it.
        ball, bad = first_where(cand, ~_in_gk_area(cand[:, 0], cand[:, 1]), -cand[:, 0])
        d = self.place_ball(d, ball[0], ball[1])
        taken = jnp.stack([ball, jnp.zeros(2)])
        d, overlap = self.scatter(d, self.n_blue, taken, u[2 * k :], lo, hi)
        return d, overlap | bad


class DribblingEnv(_BallTask):
    """`tasks.DribblingEnv`: the zigzag through four yellow robots on -x, the ball seated on
    the blue robot, `count` checkpoints passed."""

    cpu_cls = cpu.DribblingEnv

    def place(self, d, u):
        d = self.place_robot(d, self.robots[0], 0.0, 0.0, jnp.pi)
        for r, x in zip(self.robots[self.n_blue :], self.cpu_cls.nodes):
            d = self.place_robot(d, r, x, 0.0, jnp.pi)
        return d, jnp.array(False)

    def reset_physics(self, d):
        return self.seat_ball(d, self.robots[0])

    def init_task(self, d):
        return {"count": jnp.zeros((), jnp.int32), "last_ball": self.ball_xy(d)}

    def task_obs(self, d, w):
        return ((w["count"] / 6) * 2 - 1)[None].astype(d.qpos.dtype)

    def reward_done(self, d, w):
        n0, n1, n2, n3 = self.cpu_cls.nodes
        m = self.cpu_cls.margin
        rx, ry = self.xy(d, 0)
        bx, by = self.ball_xy(d)
        ly, c = w["last_ball"][1], w["count"]
        down, up = (ly >= 0) & (0 > by), (ly < 0) & (0 <= by)
        out = (rx < n3 - m) | (rx > m) | (jnp.abs(ry) > m)
        even = (c >= 2) & (c % 2 == 0)
        in_last = (n3 < bx) & (bx < n2)
        passed = ~out & (
            ((c == 0) & (n1 < bx) & (bx < n0) & down)
            | ((c == 1) & (n2 < bx) & (bx < n1) & up)
            | (even & in_last & down)
            | ((c >= 3) & (c % 2 == 1) & (n3 - m < bx) & (bx < n3) & up)
        )
        c = c + passed
        backwards = ~out & even & in_last & ~down & up
        done = self.yellow_moved(d, 0.05) | out | backwards | (passed & even & (c == 7))
        return passed.astype(jnp.float32), done, {**w, "count": c, "last_ball": jnp.stack([bx, by])}


class PassEnduranceEnv(_BallTask):
    """`tasks.PassEnduranceEnv`: a passer that can only turn, the ball seated on it, a
    team-mate receiver whose dribbler is always on."""

    cpu_cls = cpu.PassEnduranceEnv
    n_uniform = 2 + PASS_CANDIDATES

    def place(self, d, u):
        b = -1.5 + 3.0 * u[:2]
        f = jnp.where(b[1] > 0, 1.0, -1.0)
        sx, sy = b[0], b[1] + 0.115 * f
        d = self.place_robot(d, self.robots[0], sx, sy, -f * jnp.pi / 2)
        cand = -1.5 + 3.0 * u[2:]
        gap = jnp.abs(cand - b[0])
        rx, overlap = first_where(cand, gap >= 1, gap)
        ry = -b[1]
        d = self.place_robot(d, self.robots[1], rx, ry, jnp.arctan2(ry - sy, rx - sx) + jnp.pi)
        return d, overlap

    def reset_physics(self, d):
        return self.seat_ball(d, self.robots[0])

    def to_receiver(self, d):
        return jnp.linalg.norm(self.ball_xy(d) - self.xy(d, 1))

    def init_task(self, d):
        return {"stopped": jnp.zeros((), jnp.int32), "prev": self.to_receiver(d)}

    def shape_drive(self, cmd):
        return jnp.stack([0.0, 0.0, cmd[2]])

    def script(self, d, ctrl):
        return ctrl.at[self.robots[1].dribble].set(S.DRIBBLER_OMEGA)

    def reward_done(self, d, w):
        dist = self.to_receiver(d)
        received = w["grace"][1] > 0
        reward = jnp.where(received, 1.0, jnp.clip(w["prev"] - dist, -1, 1) / cpu.BALL_GRAD_SCALE)
        # `_wrong_ball`: whole centimetres, truncated as numpy's int cast does.
        ball, shoot, recv = ((p * 100).astype(jnp.int32)
                             for p in (self.ball_xy(d), self.xy(d, 0), self.xy(d, 1)))
        inside = ((jnp.minimum(recv, shoot) <= ball) & (ball <= jnp.maximum(recv, shoot))).all()
        stopped = jnp.where(jnp.abs(w["prev"] - dist) < 0.01, w["stopped"] + 1, 0)
        wrong = (stopped > self.cpu_cls.stopped_limit) | ~inside
        reward = jnp.where(wrong, reward - 1.0, reward)
        return reward, received | wrong, {**w, "prev": dist, "stopped": stopped}


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
