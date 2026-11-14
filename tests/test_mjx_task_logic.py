"""Task-logic parity: the JAX tasks (mjsoccer.mjx) against the CPU ones, state by state.

The two backends are two implementations of the same task logic (spec, Fase 3). This holds
them together: CPU rollouts (seeds 0-4) are snapshotted, each `MjData` goes to `mjx.Data`
through `put_data`, and with the same action both sides must give the same controller
output, observation, reward, ending and per-world state, to float32 tolerance. The physics
itself is the CPU's in both columns -- only the logic around it is compared -- so this runs
on JAX-CPU and never touches the GPU.

    uv sync --group gpu && uv run pytest tests/test_mjx_task_logic.py

Skipped without the `gpu` group.
"""

from __future__ import annotations

import dataclasses
import os

os.environ["JAX_PLATFORMS"] = "cpu"  # before jax is imported anywhere: never the GPU

import gymnasium as gym  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

jax = pytest.importorskip("jax")
mjx = pytest.importorskip("mujoco.mjx")
pytest.importorskip("brax")
# x64 for the comparison, not for the tolerance: the navigation controller's lateral axis
# (perpendicular to robot->target) has a to-target that is identically 0, so the +-MAX_DV it
# adds near zero lateral speed takes the sign of rounding noise -- in the original as much as
# here. In float32 that noise picks a different sign than the CPU's float64 on ~1 control
# step in 70 (always at rest, step 0). In float64 the two agree everywhere; what is compared
# is then the logic, to float32 tolerance.
jax.config.update("jax_enable_x64", True)

import mjsoccer  # noqa: E402
import mjsoccer.mjx  # noqa: E402

SEEDS = range(5)
TASKS = ["ContestedPossession-v0"]

# Per task: JAX per-world state key -> the CPU env attribute it mirrors. Every task shares
# the base entries; a task that adds state on one side must add it here too.
BASE_STATE = {"grace": "_grace", "steps": "steps", "drive_cmd": "drive_cmd",
              "plunger_lock": "_plunger_lock", "fired": "fired"}
TASK_STATE = {"ContestedPossession-v0": {"prev": "_prev", "prev_goal": "_prev_goal"}}

TOL = dict(rtol=1e-5, atol=1e-5)


_TEMPLATES = {}


def _put(cpu):
    """`mjx.put_data` of the CPU's current MjData. The full put_data runs once per model
    (~0.2 s: it converts the meshes); later snapshots refresh every public `Data` field the
    MjData also has, which is all the state put_data would carry over."""
    key = id(cpu.model)
    if key not in _TEMPLATES:
        _TEMPLATES[key] = mjx.put_data(cpu.model, cpu.data, impl="jax")
    t = _TEMPLATES[key]
    fields = {
        f.name: jax.numpy.asarray(getattr(cpu.data, f.name)).reshape(getattr(t, f.name).shape)
        for f in dataclasses.fields(t)
        if not f.name.startswith("_") and hasattr(cpu.data, f.name)
    }
    return t.replace(**fields)


def _world_from_cpu(env_id, cpu):
    names = {**BASE_STATE, **TASK_STATE[env_id]}
    return {k: np.asarray(getattr(cpu, a)) for k, a in names.items()}


def _assert_world(env_id, world, cpu, when):
    for k, want in _world_from_cpu(env_id, cpu).items():
        np.testing.assert_allclose(  # as float: bool arrays have no subtraction
            np.asarray(world[k], dtype=float), want.astype(float), **TOL,
            err_msg=f"world[{k!r}] {when}",
        )


class _Injected:
    """Stands in for the CPU env's `np_random`: `uniform(lo, hi)` hands out the given
    uniforms in order, so both backends place from the same candidates."""

    def __init__(self, u):
        self.u = list(u)

    def uniform(self, lo=0.0, hi=1.0):
        return lo + self.u.pop(0) * (hi - lo)


@pytest.mark.parametrize("env_id", TASKS)
@pytest.mark.parametrize("seed", SEEDS)
def test_place_matches_cpu_on_the_same_candidates(env_id, seed):
    cpu = gym.make(env_id).unwrapped
    cpu.reset(seed=seed)
    env = mjsoccer.mjx.make(env_id)
    u = np.random.default_rng(seed).random(env.n_uniform)
    mujoco.mj_resetData(cpu.model, cpu.data)
    blank = _put(cpu)
    cpu._np_random = _Injected(u)
    cpu._place()
    assert not cpu.np_random.u, "the JAX side declares fewer uniforms than the CPU draws"
    d = env.place(blank, u)
    np.testing.assert_allclose(np.asarray(d.qpos), cpu.data.qpos, **TOL)
    np.testing.assert_allclose(np.asarray(d.qvel), cpu.data.qvel, **TOL)


def test_sample_pose_takes_the_first_free_candidate_like_the_cpu():
    """200 candidates at once, first free one wins: what the CPU's rejection loop returns
    from the same draws."""
    from mjsoccer.mjx.base import sample_pose

    cpu = gym.make("ContestedPossession-v0").unwrapped
    taken = [(0.0, 0.0), (1.0, 1.0), (-2.0, 0.5)]
    for seed in SEEDS:
        u = np.random.default_rng(seed).random((200, 2))
        # The first few candidates land on top of what is taken, so the loop must skip them.
        u[:3] = [[0.5, 0.5], [0.5 + 1 / 11.4, 0.5 + 1 / 8.4], [0.5, 0.5]]
        cpu._np_random = _Injected(u.ravel())
        want = cpu.sample_pose(list(taken))
        got, overlap = sample_pose(u, np.array(taken))
        np.testing.assert_allclose(np.asarray(got), want, **TOL)
        assert not bool(overlap)


def test_sample_pose_with_no_free_candidate_keeps_the_roomiest():
    """Where the CPU raises, the batch cannot: it keeps the candidate farthest from
    everything placed and flags the overlap."""
    from mjsoccer.mjx.base import sample_pose

    # margin 0 maps u to [-6, 6] x [-4.5, 4.5]: candidates at x = 0.1, 0.2, 0.05, y = 0.
    u = np.array([[0.5 + 0.1 / 12, 0.5], [0.5 + 0.2 / 12, 0.5], [0.5 + 0.05 / 12, 0.5]])
    got, overlap = sample_pose(u, np.array([[0.0, 0.0]]), margin=0.0, clearance=0.25)
    np.testing.assert_allclose(np.asarray(got), (0.2, 0.0), atol=1e-9)
    assert bool(overlap)


@pytest.mark.parametrize("env_id", TASKS)
@pytest.mark.parametrize("seed", SEEDS)
def test_initial_state_matches_cpu(env_id, seed):
    """From the CPU's post-reset world (placed, ball seated), the JAX side builds the same
    first observation and per-world state."""
    cpu = gym.make(env_id).unwrapped
    obs, _ = cpu.reset(seed=seed)
    env = mjsoccer.mjx.make(env_id)
    state = env.init_state(_put(cpu), jax.random.PRNGKey(seed))
    np.testing.assert_allclose(np.asarray(state.obs), obs, **TOL)
    _assert_world(env_id, state.info["world"], cpu, "after reset")
    assert float(state.done) == 0.0 and float(state.reward) == 0.0


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
    navigation controller's output and the dribbler), and on the CPU-stepped data the same
    observation, reward, ending and per-world state."""
    cpu = gym.make(env_id).unwrapped
    obs, _ = cpu.reset(seed=seed)
    env = mjsoccer.mjx.make(env_id)
    control, settle = jax.jit(env.control), jax.jit(env.settle)
    state = env.init_state(_put(cpu), jax.random.PRNGKey(seed))
    rng = np.random.default_rng(seed)
    fires = blocked = 0
    for t in range(300):
        world = state.info["world"]
        world = {**world, **{k: v.astype(world[k].dtype)
                             for k, v in _world_from_cpu(env_id, cpu).items()}}
        state = state.replace(pipeline_state=_put(cpu), info={**state.info, "world": world})
        a = _action(cpu, obs, rng, t)
        state = control(state, a)
        obs, reward, terminated, truncated, _ = cpu._control_step(
            a.reshape(cpu.n_controlled, -1).astype(np.float64), raw=False
        )
        when = f"seed {seed} step {t}"
        fires += int(cpu.fired.any())
        blocked += int(((a.reshape(cpu.n_controlled, -1)[:, 4:6] > 0) & ~cpu.fired[:, None]).any())
        np.testing.assert_allclose(np.asarray(state.pipeline_state.ctrl), cpu.data.ctrl,
                                   **TOL, err_msg=f"ctrl {when}")
        np.testing.assert_allclose(np.asarray(state.info["world"]["drive_cmd"]), cpu.drive_cmd,
                                   **TOL, err_msg=f"drive_cmd {when}")
        state = settle(state.replace(pipeline_state=_put(cpu)))
        np.testing.assert_allclose(np.asarray(state.obs), obs, **TOL, err_msg=f"obs {when}")
        np.testing.assert_allclose(float(state.reward), reward, **TOL, err_msg=f"reward {when}")
        assert bool(state.done) == (terminated or truncated), when
        assert bool(state.info["truncation"]) == (truncated and not terminated), when
        _assert_world(env_id, state.info["world"], cpu, f"after settle, {when}")
        if terminated or truncated:
            break
    assert terminated or truncated, "the rollout should reach an ending"
    assert fires > 0 and blocked > 0, f"kick path barely exercised: {fires} fired, {blocked} blocked"


def test_controller_matches_cpu_over_random_states():
    """The navigation controller on its own, over states a rollout rarely visits: moving
    away from the target, overshooting it, inside the 2.5 deg deadzone, unwrapped yaw."""
    from mjsoccer import navigation
    from mjsoccer.mjx.base import go_to_point

    rng = np.random.default_rng(0)
    n = 2000
    pos, vel = rng.uniform(-6, 6, (n, 2)), rng.uniform(-3.5, 3.5, (n, 2))
    target = np.where(rng.random((n, 1)) < 0.1, pos + rng.normal(0, 0.01, (n, 2)),
                      rng.uniform(-6, 6, (n, 2)))
    angle = rng.uniform(-12, 12, n)
    target_angle = np.where(rng.random(n) < 0.2, angle + rng.uniform(-0.05, 0.05, n),
                            rng.uniform(-np.pi, np.pi, n))
    got = np.asarray(jax.jit(jax.vmap(go_to_point))(pos, vel, angle, target, target_angle))
    want = np.array([navigation.go_to_point(*args) for args in
                     zip(pos, vel, angle, target, target_angle)])
    np.testing.assert_allclose(got, want, **TOL)


@pytest.mark.parametrize("fire", [False, True])
def test_control_step_timesteps(fire):
    """The batch's one physics schedule for a control step (spec, Fase 3): a world that
    fires runs FIRING_FINE_STEPS at dt / FIRING_SUBSTEPS then the rest at dt, as the CPU's
    firing step; an idle one runs `decimation` at dt and is frozen at timestep 0 for the
    rest. Both cover exactly one control step of simulated time."""
    from mjsoccer import scene

    cpu = gym.make("ContestedPossession-v0").unwrapped
    env = mjsoccer.mjx.make("ContestedPossession-v0")
    dt = cpu.model.opt.timestep
    ts = np.asarray(env.timesteps(np.bool_(fire)))
    # 40 Hz on the 1/960 s step: 24 at dt; firing, 32 at dt/4 + 16 at dt -> 48 iterations.
    assert ts.shape == (48,)
    if fire:
        want = [dt / scene.FIRING_SUBSTEPS] * 32 + [dt] * 16
    else:
        want = [dt] * 24 + [0.0] * 24
    np.testing.assert_allclose(ts, want, rtol=1e-12, atol=0)
    np.testing.assert_allclose(ts.sum(), cpu.dt, rtol=1e-12)
