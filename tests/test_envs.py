"""Smoke test for every registered environment.

Runnable either way:
    uv run python tests/test_envs.py
    uv run python -m pytest tests -q
"""

from __future__ import annotations

import re
import sys

import gymnasium as gym
import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import mjsoccer
from mjsoccer import scene as S
from mjsoccer.tasks import goto_ball_policy

ENV_IDS = list(mjsoccer.ENV_IDS)


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_gym_api(env_id):
    """Gymnasium's own conformance checker: spaces, seeding, reset/step contracts."""
    env = gym.make(env_id).unwrapped
    check_env(env, skip_render_check=True)
    env.close()


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_steps_finite(env_id):
    """Every task compiles, and 100 random actions leave the state finite and in-space.

    The direct analogue of a settle check: a model that explodes under load produces NaN
    observations that a policy will happily train on.
    """
    env = gym.make(env_id)
    obs, _ = env.reset(seed=0)
    for i in range(100):
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert np.isfinite(obs).all(), f"non-finite obs at step {i}"
        assert np.isfinite(reward), f"non-finite reward at step {i}"
        assert env.observation_space.contains(obs), f"obs outside its space at step {i}"
        if terminated or truncated:
            obs, _ = env.reset(seed=i)
    env.close()


def test_ball_pairs_replicated():
    """Every robot must keep its own explicit ball contact pairs.

    Attaching the robot N times does not carry the scene's <pair> elements with it, and a
    model missing them still compiles, still steps, and silently swaps the tuned ball/bar
    contact for the elementwise max of the two geoms' own friction. The field's own ball
    pairs -- the carpet and the 12 wall hulls (tickets 17, 21) -- appear once, not per robot.
    """
    n_blue, n_yellow = 2, 1
    model, robots, _ = S.build(n_blue, n_yellow)

    source = (S.MODELS_DIR / S.SCENE).read_text()
    want = {
        m["g2"]: [float(v) for v in m["fric"].split()]
        for m in re.finditer(
            r'<pair geom1="ball" geom2="(?P<g2>[^"]+)"[^>]*friction="(?P<fric>[^"]+)"', source
        )
        if m["g2"] != "floor"
    }
    assert want, "the source scene defines no robot ball pairs to compare against"
    partners = [  # the geom each pair puts the ball against
        ({mj_geom_name(model, model.pair_geom1[p]), mj_geom_name(model, model.pair_geom2[p])}
         - {"ball"}).pop()
        for p in range(model.npair)
    ]
    field = sorted(g for g in partners if not any(g.startswith(r.prefix) for r in robots))
    walls = [f"col_f_walls_1_c{i:02d}" for i in range(12)]
    assert field == sorted(["floor", *walls]), field
    n_robot_pairs = model.npair - len(field)
    assert n_robot_pairs == len(robots) * len(want), "wrong number of robot ball pairs"

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


def test_ball_inertia_is_measured():
    """The ball's moment of inertia is the measured golf ball's, not a solid sphere's.

    I = beta * m * r^2 with beta = 0.5 (.scratch/physics-fidelity/research/
    02-physical-constants.md:23, TIGERs BallParameters.java:49). MuJoCo
    infers 0.4 from a sphere geom on its own, which compiles and steps without complaint
    and quietly changes every spin transfer, backspin and kick.
    """
    import mujoco

    model, _, _ = S.build(1, 0)
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    beta = model.body_inertia[b] / (model.body_mass[b] * 0.0215**2)
    np.testing.assert_allclose(model.body_mass[b], 0.046, rtol=1e-9)
    np.testing.assert_allclose(beta, [0.5, 0.5, 0.5], rtol=1e-6)


@pytest.mark.parametrize("surface", ("floor", "wall"))
def test_ball_bounces_at_its_measured_cor(surface):
    """The ball comes off the carpet and the wall at COR 0.47-0.56, target 0.50 (ticket 05,
    ER-Force/TIGERs). MuJoCo has no literal restitution: the bounce comes out of each
    surface's <pair> solref, found by measurement (ticket 17) and applied by ticket 21.

    Normal impacts at 1 and 6.5 m/s (the top kick speed; MuJoCo's COR rises with speed).
    COR = v_out / v_in, v_in on the last step before contact and v_out on the first step
    with no ball contact left, averaged over four contact-onset gaps: the step phase at
    onset moves it by ~2 %. The wall throw goes at col_f_walls_1_c05 with gravity
    cancelled, so only the wall touches the ball.
    """
    import mujoco

    model, robots, ball = S.build(1, 0)
    r = ball.radius
    # 4.7865: the inner face of col_f_walls_1_c05 at x = 0.865, by contact onset (ticket 17).
    axis, sign, face = (2, -1, S.FLOOR_Z) if surface == "floor" else (1, 1, 4.7865)
    for v in (1.0, 6.5):
        cors = []
        for gap in (0.0010, 0.0013, 0.0016, 0.0019):
            data = mujoco.MjData(model)
            robots[0].place(data, -3.0, 0.0, 0.0)
            ball.place(data, 0.865, 0.0)
            q, d = ball.qpos_adr, ball.qvel_adr
            if surface == "floor":
                data.qpos[q + 2] = face + r + gap
            else:
                data.qpos[q + 1 : q + 3] = (face - r - gap, S.FLOOR_Z + r + 0.05)
                data.xfrc_applied[ball.body, 2] = -model.body_mass[ball.body] * model.opt.gravity[2]
            data.qvel[d + axis] = sign * v
            hit, v_in = False, v
            for _ in range(round(0.2 / model.opt.timestep)):
                v_prev = sign * data.qvel[d + axis]
                mujoco.mj_step(model, data)
                touching = any(ball.geom in (c.geom1, c.geom2) for c in data.contact[: data.ncon])
                if touching and not hit:
                    hit, v_in = True, v_prev
                elif hit and not touching:
                    break
            assert hit, f"{surface}: the ball never reached it"
            cors.append(0.0 if touching else -sign * data.qvel[d + axis] / v_in)  # never left: 0
        assert 0.47 <= np.mean(cors) <= 0.56, f"{surface} at {v} m/s: COR {np.mean(cors):.3f}"


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
    "env_id,index,min_held", [("Dribbling-v0", 0, 111)]
)
def test_seated_ball_is_possessed(env_id, index, min_held):
    """Tasks that begin with the ball carried must actually begin that way.

    Both the seating and the detection have failed silently here before -- a ball placed
    from a stale orientation lands nowhere near the mouth, and the arm-deflection sensor
    reads the wrong sign on this model, so `has_ball` was False with the ball parked
    square in the mouth. Either way the episode looks fine and the task is unsolvable.

    ContestedPossession is out since it took rSoccer's layout: its yellow robot gets no
    commands, so the ball rests in its mouth, untouched by a bar that is not spinning.

    Measured over 8 seeds on the 1/960 s step with the calibrated ball bounce (ticket 21),
    holds() steps out of 120 idle ones: ContestedPossession 120, Dribbling 117, every seed
    alike. The bar sits 5 % under the worst seed -- a ball that is actually dropped scores
    near zero, so the margin is for seed noise only. On the old 2 ms step ContestedPossession scattered
    69-95 and looked chaotic; ticket 13 found it was the step under-resolving the
    ball/dribbler contact (120 at 0.5 ms, 87 at 2 ms, 0 at 4).
    """
    env = gym.make(env_id).unwrapped
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
        assert held >= min_held, f"seed {seed}: carried only {held}/120 idle steps"
    env.close()


def test_seated_ball_survives_a_2_rad_s_turn():
    """Spinning in place at 2 rad/s with the bar at full speed must not drop the ball.

    Ticket 22: with the old ball<->bar solref "0.004 0.5" -- ~10x stiffer than the PU 65A
    sleeve and underdamped -- the bar hammered the ball (normal force p10/p90 0.44/11 N, in
    contact on 83 % of substeps) and a 2 rad/s turn kept it 13-14 of 120 steps. The PU 65A
    solref "0.008 1" keeps it 81, worst of seeds 0-2; the bar sits 5 % under that. 3 rad/s
    still drops it -- an open fidelity gap, not a target.
    """
    from mjsoccer.base import MAX_W

    env = gym.make("Dribbling-v0").unwrapped
    for seed in range(3):
        env.reset(seed=seed)
        held = 0
        for _ in range(120):
            env.step(np.array([0.0, 0.0, 2.0 / MAX_W, 0.0, 0.0, 1.0], dtype=np.float32))
            held += env.holds(0)
        assert held >= 77, f"seed {seed}: held {held}/120 steps of a 2 rad/s turn"
    env.close()


def _dribbler_touch_by_scan(model, robots, ball, data) -> list[bool]:
    """The old definition of possession contact, kept as the independent reference:
    any contact in data.contact between the ball and a geom of the robot's dribbler body."""
    out = []
    for r in robots:
        hit = False
        for c in data.contact[: data.ncon]:
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            if ball.body in (b1, b2) and r.dribbler_body in (b1, b2):
                hit = True
        out.append(hit)
    return out


def test_possession_contact_sensor_matches_the_contact_list():
    """The ball x dribbler contact sensor (GPU port, ticket 03) must call touch exactly
    where a scan of data.contact does, for every robot, step by step.

    MJX can only read contacts through sensors, so the sensor is now the definition of
    possession on both backends; this pins it to the one it replaced. Two robots, each
    seated with the ball in turn, bar spinning and chassis turning, so the run sees
    touch, slip and loss on both.
    """
    model, robots, ball = S.build(1, 1)
    agree = touched = 0
    for seed in range(3):
        rng = np.random.default_rng(seed)
        for holder in range(2):
            data = mujoco.MjData(model)
            for i, r in enumerate(robots):
                r.place(data, -1.0 + 2.0 * i, 0.0, 0.0 if i == 0 else np.pi)
            mujoco.mj_forward(model, data)
            mouth = data.xpos[robots[holder].body] + 0.094 * robots[holder].forward(data)
            ball.place(data, mouth[0], mouth[1])
            for _ in range(600):
                data.ctrl[:] = 0.0
                for r in robots:
                    data.ctrl[r.dribble] = S.DRIBBLER_OMEGA * rng.uniform(0.0, 1.0)
                    data.ctrl[r.drive[2]] = rng.uniform(-3.0, 3.0)
                mujoco.mj_step(model, data)
                ref = _dribbler_touch_by_scan(model, robots, ball, data)
                got = [r.touches_ball(data) for r in robots]
                assert got == ref, f"seed {seed}: sensor {got} vs contact list {ref}"
                agree += 1
                touched += any(ref)
    assert touched > agree // 10, f"only {touched}/{agree} steps had any touch to compare"


def test_possession_sensor_leaves_the_physics_alone():
    """Adding the possession sensors must not change the model's physics (spec,
    restriction 7): same geoms, masses, contact pairs and actuators, and a bit-identical
    rollout against the same scene compiled without them."""
    with_s, robots, ball = S.build(1, 1)
    without, _, _ = S.build.__wrapped__(1, 1, possession_sensors=False)
    assert with_s.nsensor > without.nsensor
    for field in (
        "geom_size", "geom_pos", "geom_friction", "geom_solref", "geom_solimp",
        "geom_contype", "geom_conaffinity", "body_mass", "body_inertia",
        "pair_geom1", "pair_geom2", "pair_friction", "pair_solref", "pair_solimp",
        "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange", "dof_damping",
    ):
        np.testing.assert_array_equal(getattr(with_s, field), getattr(without, field), field)
    traces = []
    for model in (with_s, without):
        data = mujoco.MjData(model)
        robots[0].place(data, -0.5, 0.0, 0.0)
        robots[1].place(data, 0.5, 0.0, np.pi)
        ball.place(data, 0.0, 0.0, 0.0, 0.0)
        rng = np.random.default_rng(0)
        for _ in range(500):
            data.ctrl[:] = rng.uniform(-1, 1, model.nu) * model.actuator_ctrlrange[:, 1] * 0.3
            mujoco.mj_step(model, data)
        traces.append(data.qpos.copy())
    np.testing.assert_array_equal(traces[0], traces[1])


def test_possession_reads_no_contact_list():
    """The task logic must decide possession from sensordata alone -- the one channel MJX
    exposes. Emptying data.contact under a seated ball must not change holds()."""
    env = gym.make("Dribbling-v0").unwrapped
    env.reset(seed=0)
    hold = np.zeros(env.action_space.shape, dtype=np.float32)
    hold[-1] = 0.5
    for _ in range(5):
        env.step(hold)
    assert env.holds(0)

    class NoContacts:
        def __init__(self, data):
            object.__setattr__(self, "_d", data)

        def __getattr__(self, name):
            if name in ("contact", "ncon"):
                raise AssertionError(f"possession read data.{name}")
            return getattr(self._d, name)

    real = env.data
    env.data = NoContacts(real)
    try:
        env._update_possession()
    finally:
        env.data = real
    assert env.holds(0)
    env.close()


def test_goto_ball_is_solvable():
    """A P-controller must actually finish GoToBall-v0.

    Random actions cannot tell a working reward from one no controller can collect; this
    is the check that can. It is deliberately a weak controller -- if it needs tuning to
    pass, the task got harder, which is worth knowing.

    It fails by shoving the ball out of the field, and which seeds do that is chaotic in
    the contact: 53/60 seeds both at 2 ms and at 1/960 s (ticket 13), so 18/20 sat on a
    lucky window and the bar is 16.
    """
    env = gym.make("GoToBall-v0")
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
    assert wins >= 16, f"scripted policy reached the ball {wins}/20 times"
    assert sum(returns) == wins, f"GoToBall is sparse, +1 on arrival only: {returns}"


def _put(env, i, x, y, theta=0.0):
    u = env.unwrapped
    u.robots[i].place(u.data, x, y, theta)
    mujoco.mj_forward(u.model, u.data)


def _ball(env, x, y):
    u = env.unwrapped
    u.ball.place(u.data, x, y)
    mujoco.mj_forward(u.model, u.data)


def test_goto_ball_pays_one_within_20_cm():
    """rSoccer's GoToBall: +1 and done when the robot centre is under 0.2 m from the ball."""
    env = gym.make("GoToBall-v0")
    env.reset(seed=0)
    u = env.unwrapped
    b = u.ball.pos(u.data)
    _put(env, 0, b[0] - 0.15, b[1])
    _, reward, terminated, _, info = env.step(np.zeros(env.action_space.shape, np.float32))
    assert (reward, terminated, info["is_success"]) == (1.0, True, True)
    env.close()


@pytest.mark.parametrize("env_id", ["GoToBallShoot-v0", "StaticDefenders-v0"])
def test_shoot_tasks_pay_five_for_a_goal_and_end_on_leaving_the_half(env_id):
    """rSoccer's shooting reward: +5 past the line inside the mouth, done with 0 when the
    ball leaves the attacking half; neither pays a penalty."""
    from mjsoccer import scene as S

    env = gym.make(env_id)
    zero = np.zeros(env.action_space.shape, np.float32)
    env.reset(seed=0)
    _ball(env, S.FIELD_HALF_X + 0.05, 0.0)
    _, reward, terminated, _, info = env.step(zero)
    assert (reward, terminated, info["is_success"]) == (5.0, True, True)
    env.reset(seed=0)
    _ball(env, -0.5, 0.0)
    _, reward, terminated, _, info = env.step(zero)
    assert (reward, terminated, info.get("ball_out")) == (0.0, True, True)
    env.close()


def test_contested_possession_ends_when_the_opponent_is_shoved():
    """rSoccer's collision rule: the yellow robot moving over 0.1 m/s ends the episode."""
    env = gym.make("ContestedPossession-v0")
    env.reset(seed=0)
    u = env.unwrapped
    opp = u.robots[1]
    u.data.qvel[opp.qvel_adr] = 0.5
    _, _, terminated, _, info = env.step(np.zeros(env.action_space.shape, np.float32))
    assert terminated and info.get("collision")
    env.close()


def test_dribbling_pays_a_checkpoint_crossed_the_right_way():
    """The first gap is between the first two robots, crossed from +y to -y."""
    env = gym.make("Dribbling-v0")
    env.reset(seed=0)
    u = env.unwrapped
    _put(env, 0, -0.75, -0.3, np.pi)
    u._last_ball = np.array((-0.75, 0.05))
    _ball(env, -0.75, -0.05)
    _, reward, _, _, info = env.step(np.zeros(env.action_space.shape, np.float32))
    assert reward == 1.0 and info["checkpoints"] == 1
    env.close()


def test_pass_endurance_ends_a_ball_left_sitting():
    """A ball whose distance to the receiver stays put for 21 steps is a wrong ball, -1."""
    env = gym.make("PassEndurance-v0")
    env.reset(seed=0)
    for n in range(1, 100):
        _, reward, terminated, _, _ = env.step(np.zeros(env.action_space.shape, np.float32))
        if terminated:
            break
    assert n == 21 and reward == pytest.approx(-1.0, abs=0.01), (n, reward)
    env.close()


@pytest.mark.parametrize("env_id", ["StaticDefenders-v0", "ContestedPossession-v0", "PassEndurance-v0"])
def test_place_puts_every_robot_where_asked(env_id):
    """`place` is absolute, whatever offset the scene built the robot at.

    The chassis's slide joints are relative to the body's built position, which is not the origin
    once a scene holds more than one robot: each landed off by it, up to 1.5 m in
    StaticDefenders.
    """
    env = gym.make(env_id)
    env.reset(seed=0)
    u = env.unwrapped
    for i, r in enumerate(u.robots):
        r.place(u.data, 1.0 + 0.3 * i, -0.5, 0.7)
    mujoco.mj_forward(u.model, u.data)
    for i, r in enumerate(u.robots):
        np.testing.assert_allclose(r.pose(u.data), (1.0 + 0.3 * i, -0.5, 0.7), atol=1e-6)
    env.close()


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
    # The kick channel sits directly after the drive slice.
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


@pytest.mark.parametrize("plunger", (0, 1), ids=("kick", "chip"))
@pytest.mark.parametrize("a", (0.25, 0.5, 0.75, 1.0))
@pytest.mark.parametrize("control_hz,tol", [(40, 0.1), (60, 0.3)])
def test_kick_exit_speed(plunger, a, control_hz, tol):
    """Ticket 09: a in (0, 1] asks for 6.5 * a m/s, the §8.4.2 cap at a = 1.

    Seated and settled ball, robot standing, dribbler off -- the conditions the table is
    calibrated in. Peak 3-D ball speed over 31 control steps. The stock 2 ms step left this at
    7.06 m/s and wandered +-35 % with force, which is what the 6.5 bound catches. The table is
    measured at 40 Hz (GPU port, Fase 0), so it holds tight there; 60 Hz only shares the
    first 8.3 ms of fine steps and gets the looser ticket-09 bound.
    """
    env = gym.make("Dribbling-v0", control_hz=control_hz).unwrapped
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
    assert best == pytest.approx(S.KICK_SPEED_MAX * a, abs=tol)


def test_firing_step_is_48_iterations_over_one_control_step(monkeypatch):
    """K = 32 (GPU port, Fase 0): the step a plunger fires on runs 32 physics steps at
    dt / 4 and then 16 at dt -- 48 iterations instead of 96 -- and still lasts the 25 ms
    of a 40 Hz control step, so the clock never drifts."""
    env = gym.make("Dribbling-v0", control_hz=40).unwrapped
    env.reset(seed=0)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    env.step(action)
    steps = []
    real_step = mujoco.mj_step
    monkeypatch.setattr(
        mujoco, "mj_step", lambda m, d: (steps.append(m.opt.timestep), real_step(m, d))
    )
    t0 = env.data.time
    action[env.drive_dim] = 1.0
    env.step(action)
    dt = env.model.opt.timestep
    env.close()
    assert env.data.time - t0 == pytest.approx(0.025, abs=1e-12)
    assert steps == [dt / 4] * 32 + [dt] * 16
    assert dt == pytest.approx(1 / 960)  # the model is handed back at its own step


@pytest.mark.parametrize("a", (-1.0, -0.3, 0.0))
def test_kick_neutral_never_fires(a):
    """Half range (ticket 09, like the dribbler's 08): a <= 0 asks for no kick at all, so
    neither plunger leaves its seat and the ball stays in the mouth."""
    env = gym.make("Dribbling-v0").unwrapped
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


def test_no_buried_collision_hulls():
    """The CAD decomposition's buried hulls stay pruned.

    These 16 lie wholly inside the union of the other rigid chassis hulls (exact surface
    test, ticket 07), so no contact can reach them without penetration: the dribbler
    motor and gear housings and hull c02 of each bumper. They took zero contacts in
    random play on all six ids and in robot/wall/ball impacts. They were density="0".
    """
    import mujoco

    model, _, _ = S.build(1, 0)
    buried = re.compile(
        r"r_col_(motorDribbler_c0[1-6]|gear(Motor|Middle)Dribbler_c0[0-3]|bumper(Left|Right)_1_c02)$"
    )
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(model.ngeom)]
    assert not [n for n in names if buried.search(n)]


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_action_space_is_uniform(env_id):
    """Every robot exposes drive + kick + chip + dribble, so an action head transfers."""
    env = gym.make(env_id)
    assert env.action_space.shape == (6,)
    env.close()


def test_ball_height_is_above_the_carpet():
    """A ball resting on the carpet reads its own radius, not its world z.

    The floor plane sits at z = 0.02 (models/field.xml), so dividing world z by
    GOAL_HEIGHT read 0.267 for a ball at rest instead of 0.0215 / 0.155 = 0.139 (ticket 11).
    """
    env = gym.make("GoToBall-v0")
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


def test_observation_bounds_are_true():
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
    for env_id, extra in (("StaticDefenders-v0", []), ("Dribbling-v0", [4 / 3])):
        env = gym.make(env_id)
        n = len(env.unwrapped.robots)
        hi = ball[1] + robot[1] * n + extra
        lo = ball[0] + robot[0] * n + [-e for e in extra]
        np.testing.assert_allclose(env.observation_space.low, lo, rtol=1e-6)
        np.testing.assert_allclose(env.observation_space.high, hi, rtol=1e-6)
        env.close()


def test_velocity_normalisers_are_the_chassis_servo_range():
    """MAX_V and MAX_W are the chassis servo ctrlrange (ticket 11). If the servos are
    retuned, this fails and the normalisers get revisited rather than drifting apart."""
    from mjsoccer.base import MAX_V, MAX_W

    _, robots, _ = S.build(1, 0)
    np.testing.assert_array_equal(robots[0].drive_range[:, 1], (MAX_V, MAX_V, MAX_W))
    np.testing.assert_array_equal(robots[0].drive_range[:, 0], (-MAX_V, -MAX_V, -MAX_W))


def test_chassis_has_the_hardware_envelope():
    """The chassis is ideal kinematics with the hardware-derived per-axis envelope (ticket
    12, ADR 0001). Isotropic, the smaller of the frontal and lateral figure, from tickets
    01/04/05/19/20:

      v_max 3.30 m/s    frontal kinematic (lateral is 4.04)
      a_max 1.30 m/s^2  lateral wheel friction: the wheel forces that leave no moment about
                        the CoM put the lighter rear wheels at their grip first (ticket 19,
                        exact by linear programme); frontal is 1.87-1.95
      alpha 55.0 rad/s^2  wheel friction about the CoM: mu*sum(N*d) / Izz = 0.5245 / 0.009545
      w_max 37.1 rad/s  a full spin turns about the CoM, 3.45 mm behind the wheel centre
                        (ticket 20): 95 rad/s * r_eff / 77.04 mm, the arm of the wheel nearest
                        the CoM. Faster, every wheel brakes the spin

    Not modelled: slip, traction, weight transfer and the coupling between axes.
    """
    import mujoco

    model, robots, ball = S.build(1, 0)
    r = robots[0]
    dt = model.opt.timestep
    for axis, (a_max, v_max) in enumerate(((1.30, 3.30), (1.30, 3.30), (55.0, 37.1))):
        data = mujoco.MjData(model)
        r.place(data, -5.0, -4.0, 0.0)  # room for 7 m before a wall
        ball.place(data, 4.0, 3.0)
        mujoco.mj_forward(model, data)
        cmd = np.zeros(3)
        cmd[axis] = r.drive_range[axis, 1]
        vel = []
        for _ in range(round(3.0 / dt)):  # 3.30 m/s at 1.30 m/s^2 takes 2.54 s
            data.ctrl[list(r.drive)] = cmd
            mujoco.mj_step(model, data)
            vel.append(data.qvel[r.qvel_adr + axis])
        accel = np.diff(vel[: round(0.05 / dt)]).max() / dt
        assert accel == pytest.approx(a_max, rel=0.03), f"axis {axis}: accel {accel:.2f}"
        assert vel[-1] == pytest.approx(v_max, rel=0.03), f"axis {axis}: speed {vel[-1]:.2f}"


# Seconds are the contract (ticket 13): horizons, the kick lock and the possession grace
# are declared in seconds and converted. These literals are what that conversion gives at
# the default 60 Hz, so a change of dt or control rate fails here instead of silently
# changing every episode length and every reward's rate per second.
@pytest.mark.parametrize(
    "env_id,steps",
    [("GoToBall-v0", 1800), ("GoToBallShoot-v0", 3600), ("Dribbling-v0", 7200),
     ("StaticDefenders-v0", 1500), ("ContestedPossession-v0", 1800), ("PassEndurance-v0", 1800)],
)
def test_horizon_is_seconds_at_exactly_60_hz(env_id, steps):
    env = gym.make(env_id).unwrapped
    assert env.dt == pytest.approx(1 / 60, rel=1e-12)
    assert env.max_steps == steps
    assert env.metadata["render_fps"] == 60
    env.close()


def test_unreachable_control_rate_raises():
    """The control rate is exact or refused, never rounded to a neighbour (ticket 13): 50 Hz
    is 19.2 physics steps of 1/960 s."""
    with pytest.raises(ValueError, match="control_hz"):
        gym.make("GoToBall-v0", control_hz=50)
    assert gym.make("GoToBall-v0", control_hz=120).unwrapped.decimation == 8


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
