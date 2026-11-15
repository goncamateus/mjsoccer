"""Task-logic parity: the torch tasks (mjsoccer.mjlab) against the CPU ones, state by state.

The third implementation of the task logic (ADR 0003), held to the CPU the way
tests/test_mjx_task_logic.py holds the JAX one: CPU rollouts (seeds 0-4) are snapshotted,
each `MjData` becomes a batch of one in torch tensors, and with the same action both sides
must give the same controller output, observation, reward, ending and per-world state, to
float32 tolerance. The physics is the CPU's in both columns, so this runs in torch on the
CPU, with neither Warp nor a GPU.

    uv sync --group mjlab && uv run pytest tests/test_mjlab_task_logic.py

Skipped without torch (the `mjlab` group).
"""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import gymnasium as gym
import mujoco
import numpy as np
import pytest

torch = pytest.importorskip("torch")

import mjsoccer
from mjsoccer.mjlab import tasks
from mjsoccer.mjlab.base import check_failures, go_to_point, sample_pose

SEEDS = range(5)
# float64, as the JAX twin's x64: the controller's lateral axis takes the sign of rounding
# noise near rest, so float32 would compare that noise, not the logic.
DT = torch.float64
TOL = {"rtol": 1e-5, "atol": 1e-5}
# Per task: torch per-world state key -> the CPU env attribute it mirrors.
BASE_STATE = {"grace": "_grace", "steps": "steps", "drive_cmd": "drive_cmd",
              "plunger_lock": "_plunger_lock", "fired": "fired"}
TASK_STATE = {"ContestedPossession-v0": {"prev": "_prev", "prev_goal": "_prev_goal"},
              "GoToBall-v0": {},
              "GoToBallShoot-v0": {"prev": "_prev", "prev_goal": "_prev_goal"},
              "StaticDefenders-v0": {"prev": "_prev", "prev_goal": "_prev_goal"},
              "Dribbling-v0": {"count": "_count", "last_ball": "_last_ball"},
              "PassEndurance-v0": {"prev": "_prev", "stopped": "_stopped"}}
TASKS = list(TASK_STATE)


def _t(x):
    return torch.as_tensor(np.asarray(x)[None], dtype=DT)


def _snap(cpu):
    """The CPU's MjData as a batch of one: the tensors the logic reads, as mjlab's sim
    hands them out ((N, nbody, 3, 3) xmat)."""
    d = cpu.data
    return SimpleNamespace(qpos=_t(d.qpos), qvel=_t(d.qvel), xpos=_t(d.xpos),
                           xmat=_t(d.xmat.reshape(-1, 3, 3)), sensordata=_t(d.sensordata),
                           ctrl=_t(d.ctrl))


def _world_from_cpu(env_id, cpu):
    names = {**BASE_STATE, **TASK_STATE[env_id]}
    return {k: np.asarray(getattr(cpu, a)) for k, a in names.items()}


def _with_cpu_world(env_id, world, cpu):
    """`world` with every mirrored entry overwritten by the CPU's, batch of one."""
    return {**world, **{k: torch.as_tensor(v[None]).to(world[k].dtype)
                        for k, v in _world_from_cpu(env_id, cpu).items()}}


def _assert_world(env_id, world, cpu, when):
    for k, want in _world_from_cpu(env_id, cpu).items():
        np.testing.assert_allclose(world[k][0].double().numpy(), want.astype(float), **TOL,
                                   err_msg=f"world[{k!r}] {when}")


class _Injected:
    """Stands in for the CPU env's `np_random`: `uniform(lo, hi)` hands out the given
    uniforms in order, so both backends place from the same candidates."""

    def __init__(self, u):
        self.u = list(u)

    def uniform(self, lo=0.0, hi=1.0):
        return lo + self.u.pop(0) * (hi - lo)

    @property
    def left(self):
        return self.u


class _Scattered(_Injected):
    """The same for `_scatter`'s rejection loop, which draws a variable number of (x, y)
    per robot and then its heading in [0, 2 pi): the torch side lays out a fixed block per
    robot (`k` candidates, then the heading), so the heading draw skips the rest of the
    block -- the candidates the CPU never needed."""

    def __init__(self, u, head, k, lead=()):
        super().__init__(u[:head])
        n = 2 * k + 1
        # `lead`: sizes of headless candidate blocks before the per-robot ones (a ball
        # redrawn until free), each ended by `end_block` once the CPU accepts a draw.
        self.blocks = []
        for size in lead:
            self.blocks.append(list(u[head : head + size]))
            head += size
        self.blocks += [list(u[i : i + n]) for i in range(head, len(u), n)]

    def end_block(self):
        self.blocks.pop(0)

    def uniform(self, lo=0.0, hi=1.0):
        if self.u:
            return super().uniform(lo, hi)
        if hi == 2 * np.pi:
            return lo + self.blocks.pop(0)[-1] * (hi - lo)
        assert len(self.blocks[0]) > 1, "the CPU drew past the torch side's candidates"
        return lo + self.blocks[0].pop(0) * (hi - lo)

    @property
    def left(self):
        return self.u or self.blocks


def _placement_uniforms(env_id, logic, seed):
    """The task's `n_uniform` draws for a seed and the CPU's stand-in RNG over them, each
    redraw loop's first candidate rigged to be rejected, so the rejection path runs."""
    u = np.random.default_rng(seed).random(logic.n_uniform)
    k = tasks.SCATTER_CANDIDATES
    if env_id == "GoToBall-v0":  # the first robot's first two candidates on the ball
        u[2:4] = u[4:6] = u[0:2]
        return u, _Scattered(u, 2, k)
    if env_id == "GoToBallShoot-v0":
        # The robot's first candidate (whose y the CPU negates) right on the ball.
        u[2], u[3] = u[0], 1 - u[1]
        return u, _Scattered(u, 2, k)
    if env_id == "StaticDefenders-v0":  # the ball's first candidate in the penalty area
        u[0:2] = 0.99, 0.5
        return u, _Scattered(u, 0, k, lead=(2 * k,))
    if env_id == "PassEndurance-v0":  # the receiver's first x on the ball's
        u[2] = u[0]
        return u, _Scattered(u, 2, 0, lead=(tasks.PASS_CANDIDATES,))
    return u, _Injected(u)


def test_import_mjsoccer_imports_neither_torch_nor_mjlab():
    code = ("import sys, mjsoccer, gymnasium; gymnasium.make('ContestedPossession-v0'); "
            "bad = [m for m in ('torch', 'mjlab', 'mjsoccer.mjlab') if m in sys.modules]; "
            "assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_controller_matches_cpu_over_random_states():
    """The navigation controller on its own, over states a rollout rarely visits: moving
    away from the target, overshooting it, inside the 2.5 deg deadzone, unwrapped yaw."""
    from mjsoccer import navigation

    rng = np.random.default_rng(0)
    n = 2000
    pos, vel = rng.uniform(-6, 6, (n, 2)), rng.uniform(-3.5, 3.5, (n, 2))
    target = np.where(rng.random((n, 1)) < 0.1, pos + rng.normal(0, 0.01, (n, 2)),
                      rng.uniform(-6, 6, (n, 2)))
    angle = rng.uniform(-12, 12, n)
    target_angle = np.where(rng.random(n) < 0.2, angle + rng.uniform(-0.05, 0.05, n),
                            rng.uniform(-np.pi, np.pi, n))
    got = go_to_point(*(torch.as_tensor(x) for x in (pos, vel, angle, target, target_angle)))
    want = np.array([navigation.go_to_point(*args) for args in
                     zip(pos, vel, angle, target, target_angle)])
    np.testing.assert_allclose(got.numpy(), want, **TOL)


def _logic(env_id, cpu):
    """The torch task logic for `env_id`, built off the same CPU class as `cpu`."""
    cls = getattr(tasks, type(cpu).__name__)
    return cls(horizon_s=mjsoccer.ENV_IDS[env_id][1])


def test_every_env_id_is_ported_and_covered():
    """Every id the CPU registers has its torch task, and every one is in this file's parity
    runs."""
    assert sorted(TASKS) == sorted(mjsoccer.ENV_IDS)
    for cls_name, _ in mjsoccer.ENV_IDS.values():
        assert getattr(tasks, cls_name).cpu_cls.__name__ == cls_name


def test_logic_has_the_cpu_spaces_and_horizon():
    for env_id in TASKS:
        cpu = gym.make(env_id).unwrapped
        logic = _logic(env_id, cpu)
        assert logic.obs_size == cpu.observation_space.shape[0]
        assert logic.action_size == cpu.action_space.shape[0]
        assert logic.max_steps == cpu.max_steps


@pytest.mark.parametrize("env_id", TASKS)
@pytest.mark.parametrize("seed", SEEDS)
def test_place_matches_cpu_on_the_same_candidates(env_id, seed):
    cpu = gym.make(env_id).unwrapped
    cpu.reset(seed=seed)
    logic = _logic(env_id, cpu)
    u, fake = _placement_uniforms(env_id, logic, seed)
    mujoco.mj_resetData(cpu.model, cpu.data)
    blank = _snap(cpu)
    cpu._np_random = fake
    if env_id == "StaticDefenders-v0":
        # The CPU's ball loop ends on the first draw out of the penalty area: end the
        # injected ball block there too.
        in_gk = mjsoccer.tasks._in_gk_area

        def hooked(x, y):
            inside = in_gk(x, y)
            if not inside:
                fake.end_block()
            return inside

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(mjsoccer.tasks, "_in_gk_area", hooked)
            cpu._place()
    else:
        cpu._place()
    if env_id == "PassEndurance-v0":
        fake.end_block()  # the receiver's x was accepted; the rest are spares
    assert not fake.left, "the torch side declares fewer uniforms than the CPU draws"
    qpos, qvel, overlap = logic.place(blank.qpos, blank.qvel, _t(u))
    assert not overlap.any()
    np.testing.assert_allclose(qpos[0].numpy(), cpu.data.qpos, **TOL)
    np.testing.assert_allclose(qvel[0].numpy(), cpu.data.qvel, **TOL)


def test_scatter_with_no_free_candidate_flags_the_overlap():
    """Every candidate of GoToBall's first robot on the ball, in world 1 of 2: the CPU
    would keep drawing, the batch keeps the roomiest and says so, in that world only."""
    cpu = gym.make("GoToBall-v0").unwrapped
    cpu.reset(seed=0)
    logic = _logic("GoToBall-v0", cpu)
    mujoco.mj_resetData(cpu.model, cpu.data)
    u = np.random.default_rng(0).random((2, logic.n_uniform))
    u[1, 2 : 2 + 2 * tasks.SCATTER_CANDIDATES] = np.tile(u[1, 0:2], tasks.SCATTER_CANDIDATES)
    blank = _snap(cpu)
    _, _, overlap = logic.place(blank.qpos.repeat(2, 1), blank.qvel.repeat(2, 1),
                                torch.as_tensor(u))
    assert overlap.tolist() == [False, True]


def test_sample_pose_takes_the_first_free_candidate_like_the_cpu():
    """200 candidates at once, first free one wins: what the CPU's rejection loop returns
    from the same draws. All five seeds in one batch."""
    cpu = gym.make("ContestedPossession-v0").unwrapped
    taken = [(0.0, 0.0), (1.0, 1.0), (-2.0, 0.5)]
    us, want = [], []
    for seed in SEEDS:
        u = np.random.default_rng(seed).random((200, 2))
        # The first few candidates land on top of what is taken, so the loop must skip them.
        u[:3] = [[0.5, 0.5], [0.5 + 1 / 11.4, 0.5 + 1 / 8.4], [0.5, 0.5]]
        u[3 + seed] = 0.5  # and one more per seed, so each world picks a different index
        cpu._np_random = _Injected(u.ravel())
        want.append(cpu.sample_pose(list(taken)))
        us.append(u)
    taken = torch.tensor([taken] * len(us), dtype=DT)
    got, overlap = sample_pose(torch.as_tensor(np.stack(us)), taken)
    np.testing.assert_allclose(got.numpy(), np.array(want), **TOL)
    assert not overlap.any()


def test_sample_pose_with_no_free_candidate_keeps_the_roomiest():
    """Where the CPU raises, the batch cannot: it keeps the candidate farthest from
    everything placed and flags the overlap -- in that world only."""
    # margin 0 maps u to [-6, 6] x [-4.5, 4.5]: candidates at x = 0.1, 0.2, 0.05, y = 0.
    crowded = [[0.5 + 0.1 / 12, 0.5], [0.5 + 0.2 / 12, 0.5], [0.5 + 0.05 / 12, 0.5]]
    roomy = [[0.5 + 0.1 / 12, 0.5], [0.5 + 3 / 12, 0.5], [0.5 + 4 / 12, 0.5]]
    u = torch.tensor([crowded, roomy], dtype=DT)
    got, overlap = sample_pose(u, torch.zeros(2, 1, 2, dtype=DT), margin=0.0, clearance=0.25)
    np.testing.assert_allclose(got.numpy(), [(0.2, 0.0), (3.0, 0.0)], atol=1e-9)
    assert overlap.tolist() == [True, False]


@pytest.mark.parametrize("env_id", TASKS)
@pytest.mark.parametrize("seed", SEEDS)
def test_initial_state_matches_cpu(env_id, seed):
    """From the CPU's post-reset world (placed, ball seated), the torch side builds the same
    first observation and per-world state."""
    cpu = gym.make(env_id).unwrapped
    obs, _ = cpu.reset(seed=seed)
    logic = _logic(env_id, cpu)
    got, world, metrics = logic.init_state(_snap(cpu))
    check_failures(metrics)
    np.testing.assert_allclose(got[0].numpy(), obs, **TOL)
    _assert_world(env_id, world, cpu, "after reset")


def _action(cpu, obs, rng, t):
    """Random target poses, kick, chip and dribbler for a while, then the scripted chase
    (which ends the episode by shoving the yellow robot or carrying the ball out). Kick and
    chip stay random throughout: about half the steps ask for one, so the cooldown both
    blocks re-arming and runs out many times per rollout."""
    a = rng.uniform(-1, 1, cpu.action_space.shape).astype(np.float32)
    if t >= 15:
        a[:4] = mjsoccer.tasks.goto_ball_policy(obs)
    return a


@pytest.mark.parametrize("env_id", TASKS)
@pytest.mark.parametrize("seed", SEEDS)
def test_control_steps_match_cpu(env_id, seed):
    """Snapshot by snapshot along a CPU rollout: the same action gives the same `ctrl` (the
    navigation controller's output, kicker, chipper and dribbler), and on the CPU-stepped
    data the same observation, reward, ending and per-world state, `plunger_lock` and
    `fired` included."""
    cpu = gym.make(env_id).unwrapped
    obs, _ = cpu.reset(seed=seed)
    logic = _logic(env_id, cpu)
    _, world, _ = logic.init_state(_snap(cpu))
    rng = np.random.default_rng(seed)
    fires = blocked = rearmed = 0
    last_fire = np.full(cpu._plunger_lock.shape, -100)
    for t in range(cpu.max_steps):  # to an ending or, failing one, the horizon
        world = _with_cpu_world(env_id, world, cpu)
        a = _action(cpu, obs, rng, t)
        ctrl, world = logic.control(_snap(cpu), world, torch.as_tensor(a[None]))
        obs, reward, terminated, truncated, _ = cpu._control_step(
            a.reshape(cpu.n_controlled, -1).astype(np.float64), raw=False
        )
        when = f"seed {seed} control step {t}"
        asks = a.reshape(cpu.n_controlled, -1)[:, 4:6] > 0
        fires += int(cpu.fired.any())
        blocked += int((asks & ~cpu.fired[:, None]).any())
        # Per plunger: fired now, and exactly `cooldown` steps after it last did.
        now = world["plunger_lock"][0].numpy() == logic.kick_cooldown
        rearmed += int((now & (t - last_fire == logic.kick_cooldown)).any())
        last_fire[now] = t
        np.testing.assert_allclose(ctrl[0].numpy(), cpu.data.ctrl, **TOL,
                                   err_msg=f"ctrl {when}")
        np.testing.assert_allclose(world["drive_cmd"][0].numpy(), cpu.drive_cmd, **TOL,
                                   err_msg=f"drive_cmd {when}")
        assert world["fired"][0].tolist() == cpu.fired.tolist(), when
        got, r, term, trunc, world, metrics = logic.settle(_snap(cpu), world)
        check_failures(metrics)  # a failed world would hide behind its zeros
        np.testing.assert_allclose(got[0].numpy(), obs, **TOL, err_msg=f"obs {when}")
        np.testing.assert_allclose(float(r[0]), reward, **TOL, err_msg=f"reward {when}")
        assert bool(term[0]) == terminated and bool(trunc[0]) == (truncated and not terminated)
        _assert_world(env_id, world, cpu, f"after settle, {when}")
        if terminated or truncated:
            break
    assert terminated or truncated, "the rollout should reach an ending"
    assert fires > 0 and blocked > 0, f"kick barely exercised: {fires} fired, {blocked} blocked"
    assert logic.kick_cooldown == 10
    # PassEndurance's first kick sends the ball out of the box, ending it within the
    # cooldown; the re-arm is the shared base's, exercised by the other five.
    assert rearmed > 0 or env_id == "PassEndurance-v0", "no plunger re-armed after cooldown 10"


@pytest.mark.parametrize("env_id", TASKS)
def test_raw_control_matches_cpu(env_id):
    """`control(..., raw=True)` is `step_raw`'s half (the envelope scenarios): (vx, vy,
    omega) scaled by the servo range straight to the chassis; kick, chip and dribbler as
    usual."""
    cpu = gym.make(env_id).unwrapped
    cpu.reset(seed=0)
    logic = _logic(env_id, cpu)
    _, world, _ = logic.init_state(_snap(cpu))
    rng = np.random.default_rng(0)
    for t in range(20):
        a = rng.uniform(-1, 1, (cpu.n_controlled, 6))
        world = _with_cpu_world(env_id, world, cpu)
        ctrl, world = logic.control(_snap(cpu), world, torch.as_tensor(a.reshape(1, -1)),
                                    raw=True)
        cpu._control_step(a, raw=True)
        np.testing.assert_allclose(ctrl[0].numpy(), cpu.data.ctrl, **TOL, err_msg=f"step {t}")
        np.testing.assert_allclose(world["drive_cmd"][0].numpy(), cpu.drive_cmd, **TOL)


def _ball_walk(env_id, cpu, logic, path, robot_xy=None, holds=None):
    """Teleport the ball (and robot 0) along `path` and compare each step's reward, ending
    and task state with the CPU's `_reward_done` on the same world. `holds[i]`: robot 1 has
    the ball at point i (its possession grace, set by hand). Yields (reward, done, the
    CPU's per-world state) after each point; an ending resets the task."""
    a = np.zeros(cpu.action_space.shape)
    _, world, _ = logic.init_state(_snap(cpu))
    for i, (bx, by) in enumerate(path):
        if robot_xy is not None:
            cpu.robots[0].place(cpu.data, *robot_xy[i], np.pi)
        if holds is not None:
            cpu._grace[1] = cpu._grace_steps if holds[i] else 0
        cpu.ball.place(cpu.data, bx, by)
        mujoco.mj_forward(cpu.model, cpu.data)
        r, done, w = logic.reward_done(_snap(cpu), _with_cpu_world(env_id, world, cpu))
        want_r, want_done, _ = cpu._reward_done(a)
        when = f"point {i} ({bx:.2f}, {by:.2f})"
        np.testing.assert_allclose(float(r[0]), want_r, **TOL, err_msg=when)
        assert bool(done[0]) == want_done, when
        _assert_world(env_id, w, cpu, when)
        yield want_r, want_done, _world_from_cpu(env_id, cpu)
        if want_done:
            cpu._reset_task()


def test_dribbling_checkpoints_match_cpu():
    """The zigzag, which a random rollout never gets far into: all seven checkpoints, a
    backwards crossing of the last gap, then random ball and robot positions around the
    nodes, each against the CPU's own `_reward_done`."""
    env_id = "Dribbling-v0"
    cpu = gym.make(env_id).unwrapped
    cpu.reset(seed=0)
    logic = _logic(env_id, cpu)
    zigzag = [(-0.75, 0.1), (-0.75, -0.1), (-1.25, 0.1), (-1.75, 0.1), (-1.75, -0.1),
              (-2.5, -0.1), (-2.5, 0.1), (-1.75, -0.1), (-2.5, 0.1), (-1.75, -0.1)]
    out = list(_ball_walk(env_id, cpu, logic, zigzag))
    assert [w["count"] for _, _, w in out] == [0, 1, 2, 2, 3, 3, 4, 5, 6, 7]
    assert out[-1][1] and not any(d for _, d, _ in out[:-1])
    backwards = [(-0.75, 0.1), (-0.75, -0.1), (-1.25, 0.1), (-1.25, -0.1), (-1.75, 0.1)]
    out = list(_ball_walk(env_id, cpu, logic, backwards))
    assert [d for _, d, _ in out] == [False] * 4 + [True] and out[-1][2]["count"] == 2
    rng = np.random.default_rng(0)
    n = 600
    path = np.stack([rng.uniform(-3.2, 0.2, n), rng.uniform(-0.3, 0.3, n)], 1)
    robot = np.where(rng.random((n, 1)) < 0.05,
                     np.stack([rng.uniform(-4.5, 1.5, n), rng.uniform(-1.5, 1.5, n)], 1), 0.0)
    counts = [w["count"] for _, _, w in _ball_walk(env_id, cpu, logic, path, robot)]
    assert max(counts) >= 3


def test_pass_endurance_endings_match_cpu():
    """The two endings a random rollout never reaches (it always knocks the ball out of
    the box first): a ball that stops short, for over `stopped_limit` steps, and one the
    receiver holds; then a ball out of the box. Each against the CPU's `_reward_done`."""
    env_id = "PassEndurance-v0"
    cpu = gym.make(env_id).unwrapped
    cpu.reset(seed=0)
    logic = _logic(env_id, cpu)
    shoot, recv = cpu._xy(0), cpu._xy(1)
    between = [tuple(shoot + t * (recv - shoot)) for t in np.linspace(0.2, 0.8, 7)]
    out = list(_ball_walk(env_id, cpu, logic, [between[3]] * 25))
    ends = [i for i, (_, d, _) in enumerate(out) if d]
    assert ends and out[ends[0]][0] <= -1.0 and out[ends[0] - 1][2]["stopped"] == 20
    held = [False] * 6 + [True]
    out = list(_ball_walk(env_id, cpu, logic, between, holds=held))
    assert [d for _, d, _ in out] == held and out[-1][0] == 1.0
    assert all(0 < r < 1 for r, _, _ in out[1:-1])  # closing in on the receiver
    away = 2 * recv - shoot  # past the receiver, out of the box
    out = list(_ball_walk(env_id, cpu, logic, [between[0], tuple(away)], holds=[False] * 2))
    assert [d for _, d, _ in out] == [False, True] and out[-1][0] < -1.0


# ------------------------------------------------------------------ per-world failures


@pytest.mark.parametrize("field", ["qpos", "qvel"])
def test_non_finite_world_ends_with_zero_reward_and_obs(field):
    """In a batch of two, the world whose state stops being finite ends right there --
    terminal, not truncation -- with reward and observation zeroed, and says so in
    `metrics["diverged"]`; the healthy one carries on. Nothing raises."""
    env_id = TASKS[0]
    cpu = gym.make(env_id).unwrapped
    cpu.reset(seed=0)
    logic = _logic(env_id, cpu)
    one = _snap(cpu)
    d = SimpleNamespace(**{k: v.repeat(2, *[1] * (v.ndim - 1)) for k, v in vars(one).items()})
    _, world, metrics = logic.init_state(d)
    assert all(m.tolist() == [0.0, 0.0] for m in metrics.values())
    getattr(d, field)[1, 3] = float("nan")
    obs, reward, term, trunc, _, metrics = logic.settle(d, world)
    assert term.tolist() == [False, True] and not trunc.any()
    assert float(reward[1]) == 0.0 and (obs[1] == 0).all() and (obs[0] != 0).any()
    assert metrics["diverged"].tolist() == [0.0, 1.0]
    assert metrics["overflow"].tolist() == [0.0, 0.0]
    with pytest.raises(RuntimeError, match=r"1 of 2 world\(s\) diverged"):
        check_failures(metrics)


def test_overflowed_world_ends_and_check_failures_names_it():
    """Warp's overflow bits end a world like a NaN does, and the host-side abort counts it."""
    env_id = TASKS[0]
    cpu = gym.make(env_id).unwrapped
    cpu.reset(seed=0)
    logic = _logic(env_id, cpu)
    d = _snap(cpu)
    _, world, _ = logic.init_state(d)
    _, reward, term, _, _, metrics = logic.settle(d, world, overflow=torch.tensor([4]))
    assert bool(term[0]) and float(reward[0]) == 0.0
    assert metrics["overflow"].tolist() == [1.0]
    with pytest.raises(RuntimeError, match=r"1 of 1 world\(s\) overflowed"):
        check_failures(metrics)
