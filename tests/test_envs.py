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


@pytest.mark.parametrize("tier", TIERS)
def test_ball_inertia_is_measured(tier):
    """The ball's moment of inertia is the measured golf ball's, not a solid sphere's.

    I = beta * m * r^2 with beta = 0.5 (.scratch/physics-fidelity/research/
    02-physical-constants.md:23, TIGERs BallParameters.java:49). MuJoCo
    infers 0.4 from a sphere geom on its own, which compiles and steps without complaint
    and quietly changes every spin transfer, backspin and kick.
    """
    import mujoco

    model, _, _ = S.build(1, 0, tier)
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    beta = model.body_inertia[b] / (model.body_mass[b] * 0.0215**2)
    np.testing.assert_allclose(model.body_mass[b], 0.046, rtol=1e-9)
    np.testing.assert_allclose(beta, [0.5, 0.5, 0.5], rtol=1e-6)


def test_roller_floor_friction_is_measured():
    """Roller rubber on carpet slides at mu = 0.25 (Williams et al., IEEE T-RA 18(3) 2002).

    There is no explicit pair for it, so MuJoCo takes the elementwise max of the two geoms:
    setting only the roller leaves the floor's own value in charge. That is what bounds the
    chassis to ~2.45 m/s^2, friction-limited rather than torque-limited.
    """
    import mujoco

    model, _, _ = S.build(1, 0, tier=1)  # rollers exist only on tier 1
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    rollers = [g for g in range(model.ngeom) if "roller" in mj_geom_name(model, g)
               and model.geom_contype[g]]
    assert len(rollers) > 4, "found no roller collision geoms"
    for g in rollers:
        mu = max(model.geom_friction[g][0], model.geom_friction[floor][0])
        assert mu == pytest.approx(0.25), f"{mj_geom_name(model, g)}: roller/floor mu {mu}"


def test_full_spin_stays_under_max_w():
    """Every wheel at the top of its ctrlrange must spin the chassis slower than MAX_W.

    Nothing clips yaw in Python (ticket 04): the bound comes from the motor envelope
    (+-95 rad/s, +-0.078 N*m) slipping on mu = 0.25 rollers. Measured 29.2 rad/s peak
    against a kinematic 35.9 -- a 2.6% margin that any contact or actuator change can
    erase, and an observation normalised by MAX_W would then read past its bound.
    """
    import mujoco

    from mjsoccer.base import MAX_W

    model, robots, ball = S.build(1, 0, tier=1)  # the wheel actuators are tier 1 only
    r = robots[0]
    data = mujoco.MjData(model)
    r.place(data, 0.0, 0.0, 0.0)
    ball.place(data, 4.0, 3.0)  # far away: nothing else in contact
    mujoco.mj_forward(model, data)
    peak_w = peak_tilt = 0.0
    for _ in range(round(5.0 / model.opt.timestep)):
        data.ctrl[list(r.drive)] = r.drive_range[:, 1]
        mujoco.mj_step(model, data)
        R = data.xmat[r.body].reshape(3, 3)
        peak_w = max(peak_w, abs((R @ data.qvel[r.qvel_adr + 3 : r.qvel_adr + 6])[2]))
        peak_tilt = max(peak_tilt, np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
    assert peak_w < MAX_W, f"full-command yaw {peak_w:.1f} rad/s exceeds MAX_W = {MAX_W}"
    assert peak_tilt < 15.0, f"chassis tilted {peak_tilt:.1f} deg spinning in place"


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


@pytest.mark.parametrize(
    "env_id,index,min_held", [("ContestedPossession-v0", 1, 60), ("Dribbling-v0", 0, 106)]
)
@pytest.mark.parametrize("tier", TIERS)
def test_seated_ball_is_possessed(env_id, index, min_held, tier):
    """Tasks that begin with the ball carried must actually begin that way.

    Both the seating and the detection have failed silently here before -- a ball placed
    from a stale orientation lands nowhere near the mouth, and the arm-deflection sensor
    reads the wrong sign on this model, so `has_ball` was False with the ball parked
    square in the mouth. Either way the episode looks fine and the task is unsolvable.

    Both tiers, because they do not retain alike. Measured over 8 seeds, holds() steps out
    of 120 idle ones: ContestedPossession 69-95 on tier 0 against 89-96 on tier 1,
    Dribbling 119 against 114. The threshold sits under the worse tier rather than between
    them, with room for seed noise -- a ball that is actually dropped scores near zero, so
    there is no risk of that slipping through a lower bar. ContestedPossession tier 0 is
    chaotic in the ball contact: its worst seed is 82, 75, 69 or 75 as the chassis slide
    damping goes 0.1, 0.01, 0 (ticket 12) or 0.2, so its bar is 60, not 70. Dribbling's is
    tighter: ticket 05's contact changes may not cost it more than 5% of the 111-112 it held
    before them.
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

        # No drive, bar at half speed. a = 0.5 is 157 rad/s, the old affine map's neutral, so
        # the numbers above survive ticket 08 making a = 0 mean off.
        hold = np.zeros(env.action_space.shape, dtype=np.float32)
        hold[-1] = 0.5
        held = sum(int(env.holds(index)) for _ in range(120) if env.step(hold) or True)
        assert held >= min_held, f"seed {seed}: carried only {held}/120 idle steps on tier {tier}"
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


@pytest.mark.parametrize("a,omega", [(-1.0, 0.0), (-0.3, 0.0), (0.0, 0.0), (0.5, 157.0), (1.0, 314.0)])
def test_dribbler_neutral_is_off(a, omega):
    """Half-range map (ticket 08): a <= 0 is off, (0, 1] is (0, DRIBBLER_OMEGA].

    Neutral and the mean of action_space.sample() must not spin the bar.
    """
    env = gym.make("GoToBallShoot-v0").unwrapped
    env.reset(seed=0)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[-1] = a
    env.step(action)
    ctrl = float(env.data.ctrl[env.robots[0].dribble])
    env.close()
    assert ctrl == pytest.approx(omega)


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("plunger", (0, 1), ids=("kick", "chip"))
@pytest.mark.parametrize("a", (0.5, 1.0))
def test_kick_exit_speed(tier, plunger, a):
    """Ticket 09: a in (0, 1] asks for 6.5 * a m/s, the §8.4.2 cap at a = 1.

    Seated and settled ball, robot standing, dribbler off -- the conditions the table is
    calibrated in. Peak 3-D ball speed over half a second. The stock 2 ms step left this at 7.06 m/s
    on tier 0 and wandered +-35 % with force, which is what the 6.5 bound catches.
    """
    env = gym.make("Dribbling-v0", tier=tier).unwrapped
    env.reset(seed=0)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(30):  # let the seated ball settle, as calibrate.py does
        env.step(action)
    action[env.drive_dim + plunger] = a
    best = 0.0
    for _ in range(31):
        env.step(action)
        action[env.drive_dim + plunger] = 0.0
        best = max(best, float(np.linalg.norm(env.ball.vel(env.data))))
    env.close()
    assert best == pytest.approx(S.KICK_SPEED_MAX * a, abs=0.3)


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("a", (-1.0, -0.3, 0.0))
def test_kick_neutral_never_fires(tier, a):
    """Half range (ticket 09, like the dribbler's 08): a <= 0 asks for no kick at all, so
    neither plunger leaves its seat and the ball stays in the mouth."""
    env = gym.make("Dribbling-v0", tier=tier).unwrapped
    env.reset(seed=0)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[env.drive_dim : env.drive_dim + 2] = a
    action[-1] = 0.5  # dribble, or the seated ball drifts out with no kick at all (08)
    for _ in range(30):
        env.step(action)
        assert not env._plunger_lock.any()
    held = env.holds(0)
    env.close()
    assert held


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
def test_tipover_ends_episode(env_id):
    """A controlled robot on its side ends the episode, at a loss.

    Nothing a policy commands tips the chassis, so this rolls it over by hand: a robot
    that keeps collecting shaping while lying on its back is training a behaviour the real
    thing cannot perform, and the penalty is what makes falling never worth it.
    """
    env = gym.make(env_id, tier=1).unwrapped
    env.reset(seed=0)
    assert not env.tipped(), "a fresh reset must be upright"
    a = env.robots[0].qpos_adr
    # 90 deg roll about the chassis' own x -- flat on its side.
    env.data.qpos[a + 3 : a + 7] = (np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0)
    _, reward, terminated, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert terminated and info.get("tipped"), f"episode continued: {info}"
    # The penalty is 100 (ticket 10 kept it); one step of task shaping is nowhere near 50.
    assert reward < -50.0, f"tipping over paid {reward:+.2f}"
    # Logged so tip rate and time-to-tip come from the env itself (ticket 10). One
    # 16 ms step from a 90 deg roll barely moves it.
    assert 80.0 < info["tilt_deg"] < 100.0, f"logged tilt {info['tilt_deg']:.1f} deg"
    assert info["tip_step"] == 1, f"logged step {info['tip_step']}"
    env.close()


def test_chassis_rests_on_carpet():
    """A tier-1 robot dropped on its side lies on the carpet, not under it.

    Only the rollers used to collide with the floor, so a tipped chassis sank through and
    ended up on its back with the CoM at carpet height. Rolling over pivots the chassis
    onto the carpet from ~10 deg of tilt on, well inside the 60 deg termination, so this is
    in-episode physics, not post-mortem. Measured (ticket 06): CoM 87 mm up at 89 deg
    with the contact, 1.6 mm at 180 deg without.
    """
    import mujoco

    model, robots, _ = S.build(1, 0, tier=1)
    r, data = robots[0], mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    r.place(data, 0, 0, 0)
    a = r.qpos_adr
    carpet = model.geom_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")][2]
    data.qpos[a + 2] = carpet + 0.15  # released clear of the carpet
    # 90 deg roll about the chassis' own x -- flat on its side.
    data.qpos[a + 3 : a + 7] = (np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0)
    for _ in range(round(2.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    height = data.xipos[r.body][2] - carpet
    tilt = np.degrees(np.arccos(np.clip(data.xmat[r.body][8], -1, 1)))
    assert height > 0.05, f"CoM {height * 1e3:.1f} mm above the carpet, chassis sank through"
    assert tilt < 120, f"settled at {tilt:.0f} deg, rolled through the floor onto its back"


@pytest.mark.parametrize("tier", (0, 1))
def test_no_buried_collision_hulls(tier):
    """The CAD decomposition's buried hulls stay pruned on both tiers.

    These 16 lie wholly inside the union of the other rigid chassis hulls (exact surface
    test, ticket 07), so no contact can reach them without penetration: the dribbler
    motor and gear housings and hull c02 of each bumper. They took zero contacts in
    random play on all six ids and in robot/wall/ball impacts. They were density="0".
    """
    import mujoco

    model, _, _ = S.build(1, 0, tier=tier)
    buried = re.compile(
        r"r_col_(motorDribbler_c0[1-6]|gear(Motor|Middle)Dribbler_c0[0-3]|bumper(Left|Right)_1_c02)$"
    )
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(model.ngeom)]
    assert not [n for n in names if buried.search(n)]


@pytest.mark.parametrize("env_id", ENV_IDS)
@pytest.mark.parametrize("tier,dim", [(0, 6), (1, 7)])
def test_action_space_is_uniform(env_id, tier, dim):
    """Every robot exposes drive + kick + chip + dribble, so an action head transfers."""
    env = gym.make(env_id, tier=tier)
    assert env.action_space.shape == (dim,)
    env.close()


@pytest.mark.parametrize("tier", TIERS)
def test_ball_height_is_above_the_carpet(tier):
    """A ball resting on the carpet reads its own radius, not its world z.

    The floor plane sits at z = 0.02 (models/field.xml), so dividing world z by
    GOAL_HEIGHT read 0.267 for a ball at rest instead of 0.0215 / 0.155 = 0.139 (ticket 11).
    """
    env = gym.make("GoToBall-v0", tier=tier)
    obs, _ = env.reset(seed=0)
    for _ in range(30):
        obs, *_ = env.step(np.zeros(env.action_space.shape))
    env.close()
    assert obs[2] == pytest.approx(0.0215 / 0.155, abs=0.01)


def test_crossbar_is_measured_from_the_carpet():
    """The goal is 0.155 m tall above the carpet, not above the world origin 20 mm below
    it, where it was 0.135 m of mouth (ticket 11)."""
    x = S.FIELD_HALF_X + 0.05
    assert S.in_goal((x, 0.0, 0.02 + 0.15))
    assert not S.in_goal((x, 0.0, 0.02 + 0.16))


@pytest.mark.parametrize("tier", TIERS)
def test_observation_bounds_are_true(tier):
    """The Box declares a finite bound only where the simulator guarantees one (ticket 11).

    Robots cannot leave the 6.3 x 4.8 m wall box; sin/cos and has_ball are bounded by
    construction; the ball sits on an infinite carpet plane. Everything else -- ball x/y
    (a chip clears the 0.2 m wall), every velocity, ball height -- has no ceiling.
    test_steps_finite then checks the bounds hold under random action.
    """
    inf = np.inf
    rx, ry = 6.3 / 6.0, 4.8 / 4.5
    ball = ([-inf, -inf, 0.0, -inf, -inf, -inf], [inf] * 6)
    robot = ([-rx, -ry, -1, -1, -inf, -inf, -inf, 0], [rx, ry, 1, 1, inf, inf, inf, 1])
    wp = np.hypot(6.0 + 6.3, 4.5 + 4.8)
    for env_id, extra in (("StaticDefenders-v0", 0), ("Dribbling-v0", 2)):
        env = gym.make(env_id, tier=tier)
        n = len(env.unwrapped.robots)
        hi = ball[1] + robot[1] * n + [wp / 6.0, wp / 4.5][:extra]
        lo = ball[0] + robot[0] * n + [-wp / 6.0, -wp / 4.5][:extra]
        np.testing.assert_allclose(env.observation_space.low, lo, rtol=1e-6)
        np.testing.assert_allclose(env.observation_space.high, hi, rtol=1e-6)
        env.close()


def test_velocity_normalisers_are_the_tier0_servo_range():
    """MAX_V and MAX_W are the tier-0 chassis servo ctrlrange (ticket 11), shared by both
    tiers so one observation means one speed. Tier 1 stays under them on its own drive
    (test_full_spin_stays_under_max_w). If ticket 12 retunes the tier-0 servos, this fails
    and the normalisers get revisited rather than drifting apart."""
    from mjsoccer.base import MAX_V, MAX_W

    _, robots, _ = S.build(1, 0, tier=0)
    np.testing.assert_array_equal(robots[0].drive_range[:, 1], (MAX_V, MAX_V, MAX_W))
    np.testing.assert_array_equal(robots[0].drive_range[:, 0], (-MAX_V, -MAX_V, -MAX_W))


def test_tier0_has_the_tier1_envelope():
    """Tier 0 keeps its ideal-kinematics mechanism but gets tier 1's per-axis envelope
    (ticket 12), so a tier-0 command asks for what a tier-1 robot could do. Isotropic, the
    smaller of tier 1's frontal and lateral figure, from tickets 01/04/05:

      v_max 3.30 m/s    frontal (lateral is 4.04)
      a_max 2.30 m/s^2  lateral wheel-torque limit (frontal is min(2.98, mu*g = 2.45))
      alpha 55.1 rad/s^2  roller friction: mu*m*g*R / Izz = 6.602 N * 0.0796 m / 0.009545
      w_max 30 rad/s    the ceiling test_full_spin_stays_under_max_w holds tier 1 under

    Not matched, because they are what tier 1 is for: slip, traction, weight transfer,
    and the coupling between axes (tier 1 at full forward speed has no wheel left to yaw).
    """
    import mujoco

    model, robots, ball = S.build(1, 0, tier=0)
    r = robots[0]
    dt = model.opt.timestep
    for axis, (a_max, v_max) in enumerate(((2.30, 3.30), (2.30, 3.30), (55.1, 30.0))):
        data = mujoco.MjData(model)
        r.place(data, -5.0, -4.0, 0.0)  # room for 7 m before a wall
        ball.place(data, 4.0, 3.0)
        mujoco.mj_forward(model, data)
        cmd = np.zeros(3)
        cmd[axis] = r.drive_range[axis, 1]
        vel = []
        for _ in range(round(2.0 / dt)):
            data.ctrl[list(r.drive)] = cmd
            mujoco.mj_step(model, data)
            vel.append(data.qvel[r.qvel_adr + axis])
        accel = np.diff(vel[: round(0.05 / dt)]).max() / dt
        assert accel == pytest.approx(a_max, rel=0.03), f"axis {axis}: accel {accel:.2f}"
        assert vel[-1] == pytest.approx(v_max, rel=0.03), f"axis {axis}: speed {vel[-1]:.2f}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
