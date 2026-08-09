"""Smoke test for every registered environment.

Runnable either way:
    uv run python tests/test_envs.py
    uv run python -m pytest tests -q
"""

from __future__ import annotations

import re
import sys

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import mjsoccer
from mjsoccer import scene as S
from mjsoccer.tasks import goto_ball_policy

ENV_IDS = list(mjsoccer.ENV_IDS)
TIERS = (0, 1)


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_gym_api(env_id):
    """Gymnasium's own conformance checker: spaces, seeding, reset/step contracts."""
    env = gym.make(env_id).unwrapped
    check_env(env, skip_render_check=True)
    env.close()


@pytest.mark.parametrize("env_id", ENV_IDS)
@pytest.mark.parametrize("tier", TIERS)
def test_steps_finite(env_id, tier):
    """Both tiers compile, and 100 random actions leave the state finite and in-space.

    The direct analogue of a settle check: a model that explodes under load produces NaN
    observations that a policy will happily train on.
    """
    env = gym.make(env_id, tier=tier)
    obs, _ = env.reset(seed=0)
    for i in range(100):
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert np.isfinite(obs).all(), f"non-finite obs at step {i}"
        assert np.isfinite(reward), f"non-finite reward at step {i}"
        assert env.observation_space.contains(obs), f"obs outside its space at step {i}"
        if terminated or truncated:
            obs, _ = env.reset(seed=i)
    env.close()


@pytest.mark.parametrize("tier", TIERS)
def test_ball_pairs_replicated(tier):
    """Every robot must keep its own explicit ball contact pairs.

    Attaching the robot N times does not carry the scene's <pair> elements with it, and a
    model missing them still compiles, still steps, and silently swaps the tuned ball/bar
    contact for the elementwise max of the two geoms' own friction.
    """
    n_blue, n_yellow = 2, 1
    model, robots, _ = S.build(n_blue, n_yellow, tier)

    source = (S.MODELS_DIR / S.SCENES[tier]).read_text()
    want = {
        m["g2"]: [float(v) for v in m["fric"].split()]
        for m in re.finditer(
            r'<pair geom1="ball" geom2="(?P<g2>[^"]+)"[^>]*friction="(?P<fric>[^"]+)"', source
        )
        if m["g2"] != "floor"
    }
    assert want, "the source scene defines no robot ball pairs to compare against"
    assert model.npair == len(robots) * len(want) + 1, "wrong number of contact pairs"

    seen = set()
    for p in range(model.npair):
        g2 = mj_geom_name(model, model.pair_geom2[p])
        for prefix in (r.prefix for r in robots):
            if g2.startswith(prefix) and g2[len(prefix) :] in want:
                np.testing.assert_allclose(
                    model.pair_friction[p], want[g2[len(prefix) :]], rtol=1e-9
                )
                seen.add(g2)
    assert len(seen) == len(robots) * len(want), f"only matched {sorted(seen)}"


def mj_geom_name(model, gid: int) -> str:
    import mujoco

    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or ""


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_deterministic(env_id):
    """The same seed must give the same episode, or nothing measured is reproducible."""
    traces = []
    for _ in range(2):
        env = gym.make(env_id)
        obs, _ = env.reset(seed=1234)
        env.action_space.seed(1234)
        rewards = []
        for _ in range(50):
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            rewards.append(reward)
            if terminated or truncated:
                break
        traces.append((obs.copy(), np.array(rewards)))
        env.close()
    np.testing.assert_array_equal(traces[0][0], traces[1][0])
    np.testing.assert_array_equal(traces[0][1], traces[1][1])


@pytest.mark.parametrize("env_id,index", [("ContestedPossession-v0", 1), ("Dribbling-v0", 0)])
@pytest.mark.parametrize("tier", TIERS)
def test_seated_ball_is_possessed(env_id, index, tier):
    """Tasks that begin with the ball carried must actually begin that way.

    Both the seating and the detection have failed silently here before -- a ball placed
    from a stale orientation lands nowhere near the mouth, and the arm-deflection sensor
    reads the wrong sign on this model, so `has_ball` was False with the ball parked
    square in the mouth. Either way the episode looks fine and the task is unsolvable.

    Both tiers, because they do not retain alike. Measured over 8 seeds, contact steps out
    of 120 idle ones: ContestedPossession 96-103 on tier 0 against 87-95 on tier 1,
    Dribbling a flat 120 against 111-112. Real wheel slip lets the chassis recoil off the
    ball it is holding, which costs about 8% of the contact steps. The threshold sits under
    the worse tier rather than between them -- a ball that is actually dropped scores near
    zero, so there is no risk of that slipping through a lower bar.
    """
    env = gym.make(env_id, tier=tier).unwrapped
    for seed in range(5):
        env.reset(seed=seed)
        # Seated at reset: ~94 mm out along the robot's own +x, at ball radius height.
        r = env.robots[index]
        local = env.data.xmat[r.body].reshape(3, 3).T @ (
            env.ball.pos(env.data) - env.data.xpos[r.body]
        )
        assert 0.07 < local[0] < 0.12, f"seed {seed}: ball at x={local[0]:.3f} in robot frame"
        assert abs(local[1]) < 0.03, f"seed {seed}: ball off-centre by {local[1]:.3f} m"

        held = sum(int(env.holds(index)) for _ in range(120) if env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)) or True)
        assert held > 80, f"seed {seed}: carried only {held}/120 idle steps on tier {tier}"
    env.close()


def test_goto_ball_is_solvable():
    """A P-controller must actually finish GoToBall-v0.

    Random actions cannot tell a working reward from one no controller can collect; this
    is the check that can. It is deliberately a weak controller -- if it needs tuning to
    pass, the task got harder, which is worth knowing.
    """
    # Pinned to tier 0: goto_ball_policy emits a body-frame (vx, vy, omega) command, and
    # tier 1's drive slice is four wheel speeds. What is under test is the reward, not the
    # chassis, so run it on the chassis the controller was written for.
    env = gym.make("GoToBall-v0", tier=0)
    wins, returns = 0, []
    for seed in range(20):
        obs, _ = env.reset(seed=seed)
        total = 0.0
        while True:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            action[:3] = goto_ball_policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                wins += bool(info.get("is_success"))
                returns.append(total)
                break
    env.close()
    assert wins >= 18, f"scripted policy reached the ball {wins}/20 times"
    assert np.mean(returns) > 5.0, f"mean return {np.mean(returns):.2f}"


def test_kicker_rearms():
    """A kick action held down must not pin the plunger out.

    The kicker is a force actuator against a return spring: hold it and the ball is never
    released. The cooldown is what turns a continuous action into a solenoid pulse.
    """
    import mujoco

    env = gym.make("GoToBallShoot-v0").unwrapped
    env.reset(seed=0)
    jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "b0_kicker_slide")
    stroke = int(env.model.jnt_qposadr[jid])

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    # The kick channel sits directly after the drive slice, which is 3 wide on tier 0 and
    # 4 on tier 1 -- index off drive_dim so this follows the default tier wherever it goes.
    action[env.drive_dim] = 1.0  # kick, held down for a second
    fired, travel = 0, []
    for _ in range(60):
        env.step(action)
        # step() decrements the lock after arming it, so a fire this step reads one below.
        fired += int(env._plunger_lock[0, 0] == env._kick_cooldown - 1)
        travel.append(float(env.data.qpos[stroke]))
    env.close()

    assert 2 <= fired <= 8, f"kicker fired {fired} times in 1 s, expected ~4"
    # The spring must actually pull it home between pulses. A pinned plunger sits out at
    # its 30 mm stop and never releases a ball, which the fire count alone cannot see.
    assert max(travel) > 0.005, f"plunger barely moved, max {max(travel) * 1e3:.1f} mm"
    assert min(travel) < 0.002, f"plunger never retracted, min {min(travel) * 1e3:.1f} mm"


def test_chipper_rearms():
    """The chipper is the kicker's twin and needs its own independent pulse.

    Same force-actuator-against-a-spring mechanism, same failure mode if a held action
    pins it out -- and its own lock column, so a kick can never swallow a chip.
    """
    import mujoco

    env = gym.make("GoToBallShoot-v0").unwrapped
    env.reset(seed=0)
    jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "b0_chipper_slide")
    stroke = int(env.model.jnt_qposadr[jid])

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[env.drive_dim + 1] = 1.0  # chip, held down for a second
    fired, travel = 0, []
    for _ in range(60):
        env.step(action)
        fired += int(env._plunger_lock[0, 1] == env._kick_cooldown - 1)
        travel.append(float(env.data.qpos[stroke]))
    env.close()

    assert 2 <= fired <= 8, f"chipper fired {fired} times in 1 s, expected ~4"
    # Measured: the pulse throws the plunger out to ~78 mm and it rings back through
    # ~-40 mm before settling, so both bounds clear by better than an order of magnitude.
    # A plunger pinned by a held action instead sits parked out and trips neither.
    assert max(travel) > 0.005, f"plunger barely moved, max {max(travel) * 1e3:.1f} mm"
    assert min(travel) < 0.002, f"plunger never retracted, min {min(travel) * 1e3:.1f} mm"


@pytest.mark.parametrize("env_id", ENV_IDS)
@pytest.mark.parametrize("tier,dim", [(0, 6), (1, 7)])
def test_action_space_is_uniform(env_id, tier, dim):
    """Every robot exposes drive + kick + chip + dribble, so an action head transfers."""
    env = gym.make(env_id, tier=tier)
    assert env.action_space.shape == (dim,)
    env.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
