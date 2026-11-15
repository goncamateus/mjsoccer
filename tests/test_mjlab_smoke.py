"""mjlab smoke: `mjsoccer.mjlab.make` makes, resets and steps ContestedPossession (2 robots)
and GoToBall (7) on the GPU.

Not a physics check (the parity against the CPU is statistical, `-m validation`): this proves
the env runs end to end -- the first-state pool and its seat_ball loop, the controller, the
firing schedule, possession, observation -- and stays finite; that an idle world sits
bit-for-bit still through the iterations it spends at timestep 0; and that the per-world
failures (a world gone NaN, a world overflowing Warp's constraint buffer) end that world
alone, zeroed, and reset it alone.

    uv sync --group mjlab && uv run pytest tests/test_mjlab_smoke.py

The make errors run in-process; the rest in a fresh GPU subprocess (`_gpu.py`), skipped
without CUDA or without the `mjlab` group.
"""

from __future__ import annotations

import json
import sys

import pytest
from _gpu import SKIP, run_json

pytest.importorskip("torch")
pytest.importorskip("mjlab")

ENV = "ContestedPossession-v0"
PORTED = [ENV, "GoToBall-v0"]
NWORLD = 4


def _run(*args) -> dict:
    return run_json(__file__, *args)


# ------------------------------------------------------------------ in-process


def test_make_rejects_ids_not_ported_or_unknown():
    import mjsoccer.mjlab

    with pytest.raises(NotImplementedError, match="GoToBallShoot-v0"):
        mjsoccer.mjlab.make("GoToBallShoot-v0", nworld=1)
    with pytest.raises(KeyError):
        mjsoccer.mjlab.make("Nope-v0", nworld=1)


def test_make_without_the_mjlab_group_says_how_to_get_it(monkeypatch):
    import mjsoccer.mjlab

    # What a missing torch/mjlab does to the import of the env module.
    monkeypatch.setitem(sys.modules, "mjsoccer.mjlab.env", None)
    monkeypatch.delattr(mjsoccer.mjlab, "env", raising=False)
    with pytest.raises(ImportError, match="uv sync --group mjlab"):
        mjsoccer.mjlab.make(ENV, nworld=1)


def test_make_without_cuda_says_so(monkeypatch):
    import torch

    import mjsoccer.mjlab

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        mjsoccer.mjlab.make(ENV, nworld=1)


def test_firing_schedule_is_the_cpu_firing_step_and_idle_padded_with_zero():
    from mjsoccer import scene

    dt = 1 / 960
    firing, idle = scene.firing_schedule(24, dt)
    assert len(firing) == len(idle) == 48
    assert firing == [dt / 4] * 32 + [dt] * 16 and sum(firing) == pytest.approx(24 * dt)
    assert idle == [dt] * 24 + [0.0] * 24


# ---------------------------------------------------------------- GPU children


@pytest.mark.parametrize("env_id", PORTED)
@pytest.mark.parametrize("repeat_action", [1, 2])
def test_mjlab_smoke(env_id, repeat_action):
    out = _run("smoke", repeat_action, env_id)
    # The CPU's spaces, layout and model, plus the action term's anchor body.
    assert out["obs_dim"] == out["cpu_obs_dim"] and out["act_dim"] == out["cpu_act_dim"], out
    assert out["sizes"] == out["cpu_sizes"], out  # nq, nv, nu, nsensordata
    assert out["nbody"] == out["cpu_nbody"] + 1, out
    assert out["opt"] == out["cpu_opt"], out
    assert out["decimation"] == 48 * repeat_action, out
    assert out["max_steps"] == out["cpu_max_steps"], out
    assert out["n_robots"] == out["cpu_n_robots"], out
    if env_id == ENV:  # the seat_ball loop really ran: the yellow robot holds the ball
        assert all(out["robot1_holds_at_reset"]), out
    else:  # GoToBall seats none: the ball starts placed, at rest
        assert all(out["ball_at_rest"]), out
    assert out["reset_overlap"] == 0, out
    # Every world kicked on the first control step (its lock counted down since).
    assert out["kick_lock"] == [out["cooldown"] - repeat_action] * NWORLD, out
    assert out["finite"], out
    assert out["overflow_bits"] == [0] * NWORLD, out
    assert out["diverged"] == 0 and out["overflow"] == 0, out
    # A decision is repeat_action control steps, cut short only by an ending.
    assert all(10 <= k <= 10 * repeat_action for k in out["control_steps"]), out
    assert max(out["control_steps"]) > 10 or repeat_action == 1, out


def test_idle_worlds_are_frozen_bitwise_at_timestep_zero():
    """A ball seated on the blue robot in every world; world 0 kicks, world 1 chips, 2 and 3
    do neither. Through that decision, the idle worlds' qpos, qvel, act, time and ctrl do
    not change by a bit from iteration 24 to 48, while the firing worlds keep moving."""
    out = _run("idle", 1)
    assert out["fired"] == [True, True, False, False], out
    assert out["zero_iters"] == [0, 0, 24, 24], out
    assert out["idle_bitwise"] == [False, False, True, True], out
    kick, chip, *idle = range(NWORLD)
    assert out["speed"][kick] > 5.0 and out["lift"][kick] < 0.05, out  # asked for 6.5 m/s
    assert out["speed"][chip] > 4.0 and out["lift"][chip] > 0.2, out
    assert all(out["speed"][w] < 1.0 and out["lift"][w] < 0.02 for w in idle), out


def test_mixed_batch_only_the_overflowed_worlds_end():
    """Worlds 0 and 1 get the blue robot jammed into the yellow one, more constraint rows
    than an undersized njmax holds; 2 and 3 fit. Only 0 and 1 end (terminated, reward 0,
    `overflow` 1) and are reset to their first state with their bits cleared; 2 and 3 carry
    on. A masked reset clears only the reset worlds' bits."""
    out = _run("mixed", 1)
    assert out["terminated"] == [True, True, False, False], out
    assert out["time_out"] == [False] * NWORLD, out
    assert out["reward_zero"] == [True, True, False, False], out
    assert out["overflow"] == [1.0, 1.0, 0.0, 0.0], out
    assert out["nefc_bit"] == [True, True, False, False], out
    assert out["episode_overflow"] == 1.0, out  # the logged episode sum over the reset worlds
    assert out["bits_after"] == [0] * NWORLD, out
    assert out["restored"] == [True, True, False, False], out
    assert out["steps"] == [0, 0, 1, 1], out
    assert out["masked_reset_bits"] == [0, 0, 1, 1], out


@pytest.mark.parametrize("repeat_action", [1, 2])
def test_nan_world_ends_zeroed_and_resets_alone(repeat_action):
    """World 0's velocity is poisoned. Its decision ends at the first control step,
    terminated with obs and reward 0 and `diverged` 1; the rest of the decision leaves it
    frozen. The others carry on finite. Its reset brings back its pooled first state."""
    out = _run("nan", repeat_action)
    assert out["terminated"] == [True, False, False, False], out
    assert out["reward0"] == 0.0 and out["obs0_zero"], out
    assert out["diverged"] == [1.0, 0.0, 0.0, 0.0], out
    assert out["steps"] == [1] + [repeat_action] * (NWORLD - 1), out
    assert out["others_finite"], out
    assert "diverged" in out["check_failures"], out
    assert out["restored"] and out["after_finite"], out


def test_a_caller_on_a_side_stream_does_not_race_warp():
    """The caller steps from a non-blocking torch stream. Every world's `data.time` is then
    exactly the float32 sum of the per-iteration timesteps torch wrote: nothing torch wrote
    was read late or early by Warp's graph. Without the env's stream ordering it is off in
    every world (.scratch/mjlab-port/research/04-streams-and-memory.md)."""
    out = _run("streams", 1)
    assert out["checked"] > 0 and out["time_off"] == 0, out


# ------------------------------------------------------------------ the children


def _make(repeat_action=1, env_id=ENV, **kw):
    import mjsoccer.mjlab

    kw.setdefault("nworld", NWORLD)
    return mjsoccer.mjlab.make(env_id, repeat_action=repeat_action, seed=0, **kw)


def _hold(env, robot=0):
    """Hold robot's current pose, plungers and dribblers off: (N, 7)."""
    import torch

    from mjsoccer import scene

    d, r = env.sim.data, env.logic.robots[robot]
    p, R = d.xpos[:, r.body], d.xmat[:, r.body]
    th = torch.atan2(R[:, 1, 0], R[:, 0, 0])
    return torch.stack([p[:, 0] / scene.FIELD_HALF_X, p[:, 1] / scene.FIELD_HALF_Y,
                        torch.sin(th), torch.cos(th), *[-torch.ones_like(th)] * 3], -1)


def smoke(repeat_action: int, env_id: str) -> dict:
    import gymnasium as gym
    import torch

    env = _make(repeat_action, env_id)
    obs, _ = env.reset()
    obs = obs["actor"]
    cpu = gym.make(env_id, repeat_action=repeat_action).unwrapped
    lg = env.logic
    v = lg.ball.qvel_adr
    m, cm = env.sim.mj_model, cpu.model
    opt_fields = ("timestep", "integrator", "cone", "solver", "jacobian", "iterations",
                  "tolerance", "ls_iterations", "ls_tolerance", "ccd_iterations", "impratio",
                  "disableflags", "enableflags")

    def opt(o):
        return [float(getattr(o, f)) for f in opt_fields] + list(map(float, o.gravity))

    out = {
        "obs_dim": obs.shape[1], "act_dim": env.action_space.shape[1],
        "cpu_obs_dim": cpu.observation_space.shape[0], "cpu_act_dim": cpu.action_space.shape[0],
        "sizes": [m.nq, m.nv, m.nu, m.nsensordata], "cpu_sizes": [cm.nq, cm.nv, cm.nu, cm.nsensordata],
        "nbody": m.nbody, "cpu_nbody": cm.nbody, "opt": opt(m.opt), "cpu_opt": opt(cm.opt),
        "decimation": env.cfg.decimation, "max_steps": env.logic.max_steps,
        "cpu_max_steps": cpu.max_steps, "n_robots": len(lg.robots),
        "cpu_n_robots": len(cpu.robots),
        # obs: 6 ball entries, then 8 per robot ending in has_ball.
        "robot1_holds_at_reset": (obs[:, 6 + 8 + 7] > 0).tolist(),
        "ball_at_rest": (env.sim.data.qvel[:, v : v + 6] == 0).all(-1).tolist(),
        "reset_overlap": int(env.pool["world"]["overlap"].sum()),
    }
    cpu.close()
    g = torch.Generator(device=env.device).manual_seed(0)
    finite, bits, diverged, overflow = True, 0, 0.0, 0.0
    for i in range(10):
        a = torch.rand(NWORLD, env.logic.action_size, device=env.device, generator=g) * 2 - 1
        if i == 0:
            a[:, 4] = 1.0  # kick on the first decision, every plunger armed
        obs, _, _, _, _ = env.step(a)
        if i == 0:
            out["kick_lock"] = env.world["plunger_lock"][:, 0, 0].tolist()
        bits = torch.as_tensor(bits) | env.sim.data.overflow[:].cpu()
        diverged += float(env.metrics["diverged"].sum())
        overflow += float(env.metrics["overflow"].sum())
        finite &= bool(torch.isfinite(obs["actor"]).all() and torch.isfinite(env.sim.data.qpos[:]).all())
    return {**out, "finite": finite, "cooldown": env.logic.kick_cooldown, "overflow_bits": bits.tolist(),
            "diverged": diverged, "overflow": overflow,
            "control_steps": env.world["steps"].tolist()}


def idle(_: int) -> dict:
    import torch

    from mjsoccer import scene

    env = _make()
    env.reset()
    blue, ball = env.logic.robots[0], env.logic.ball
    env._seat_ball(blue)  # every world: the ball on the blue robot, then make that the start
    env.pool.update(qpos=env.sim.data.qpos[:].clone(), qvel=env.sim.data.qvel[:].clone())
    env.reset()
    a = _hold(env)
    a[0, 4] = a[1, 5] = 1.0  # world 0 kicks, world 1 chips, at the 6.5 m/s cap

    d, snaps, step = env.sim.data, [], env.sim.step
    fields = ("qpos", "qvel", "act", "time", "ctrl")

    def recorded():  # snaps[i]: the state after iteration i + 1, and its timestep
        step()
        snaps.append({**{f: getattr(d, f)[:].clone() for f in fields},
                      "timestep": env.timestep.clone()})

    env.sim.step = recorded
    env.step(a)
    env.sim.step = step
    zero = torch.stack([s["timestep"] for s in snaps]) == 0
    still = [all(torch.equal(snaps[23][f][w], snaps[47][f][w]) for f in fields)
             for w in range(NWORLD)]
    fired = env.world["fired"].any(-1).tolist()
    z0 = float(scene.FLOOR_Z + ball.radius)
    speed, lift = torch.zeros(NWORLD), torch.zeros(NWORLD)
    rest = _hold(env)
    for _ in range(15):
        speed = torch.maximum(speed, d.qvel[:, ball.qvel_adr : ball.qvel_adr + 3].norm(dim=-1).cpu())
        lift = torch.maximum(lift, d.xpos[:, ball.body, 2].cpu() - z0)
        env.step(rest)
    return {"fired": fired, "zero_iters": zero.sum(0).tolist(), "idle_bitwise": still,
            "speed": speed.tolist(), "lift": lift.tolist()}


def mixed(_: int) -> dict:
    import torch
    from mujoco_warp import OverflowType

    env = _make(nconmax=512, njmax=64, naccdmax=NWORLD * 512)  # EPA room: NEFC overflows
    env.reset()
    d, (blue, yellow) = env.sim.data, env.logic.robots
    q = d.qpos[:].clone()
    jam = d.xpos[:, yellow.body, :2] + torch.tensor([0.16, 0.0], device=env.device)
    q[:2, blue.qpos_adr] = jam[:2, 0] - blue.xy0[0]
    q[:2, blue.qpos_adr + 1] = jam[:2, 1] - blue.xy0[1]
    d.qpos[:] = q
    env.sim.forward()
    seen, settle = [], env.settle

    def watched():
        settle()
        seen.append(d.overflow[:].clone())  # the bits, before the auto-reset clears them

    env.settle = watched
    _, reward, terminated, time_out, extras = env.step(_hold(env))
    p = env.pool
    out = {
        "terminated": terminated.tolist(), "time_out": time_out.tolist(),
        "reward_zero": (reward == 0).tolist(), "overflow": env.metrics["overflow"].tolist(),
        "nefc_bit": ((seen[0] & int(OverflowType.NEFC)) != 0).tolist(),
        "episode_overflow": float(extras["log"]["Episode_Metrics/overflow"]),
        "bits_after": d.overflow[:].tolist(),
        "restored": [bool(torch.equal(d.qpos[w], p["qpos"][w])) for w in range(NWORLD)],
        "steps": env.world["steps"].tolist(),
    }
    d.overflow[:] = 1
    env.reset(env_ids=torch.tensor([0, 1], device=env.device))
    out["masked_reset_bits"] = d.overflow[:].tolist()
    return out


def nan(repeat_action: int) -> dict:
    import torch

    from mjsoccer.mjlab.base import check_failures

    env = _make(repeat_action)
    env.cfg.auto_reset = False  # see the terminal obs
    env.reset()
    d = env.sim.data
    d.qvel[0, 0] = float("nan")
    obs, reward, terminated, _, _ = env.step(_hold(env))
    obs = obs["actor"]
    try:
        check_failures(env.metrics)
        failures = ""
    except RuntimeError as e:
        failures = str(e)
    out = {
        "terminated": terminated.tolist(), "reward0": float(reward[0]),
        "obs0_zero": bool((obs[0] == 0).all()), "diverged": env.metrics["diverged"].tolist(),
        "steps": env.world["steps"].tolist(), "check_failures": failures,
        "others_finite": bool(torch.isfinite(d.qpos[1:]).all() and torch.isfinite(d.qvel[1:]).all()
                              and torch.isfinite(obs[1:]).all()),
    }
    obs, _ = env.reset(env_ids=torch.tensor([0], device=env.device))
    p = env.pool
    out["restored"] = bool(torch.equal(d.qpos[0], p["qpos"][0]) and torch.equal(d.qvel[0], p["qvel"][0])
                           and torch.equal(obs["actor"][0], p["obs"][0])
                           and int(env.world["steps"][0]) == 0)
    obs, *_ = env.step(_hold(env))
    out["after_finite"] = bool(torch.isfinite(obs["actor"]).all() and torch.isfinite(d.qpos[:]).all())
    return out


def streams(_: int) -> dict:
    import torch

    n = 256  # enough work in flight for a race to show
    env = _make(nworld=n)
    side = torch.cuda.Stream()
    plans, ended = [], torch.zeros(n, dtype=torch.bool, device=env.device)
    with torch.cuda.stream(side):
        env.reset()
        t = env.sim.data.time[:].clone()
        for _ in range(20):
            a = torch.rand(n, env.logic.action_size, device=env.device) * 2 - 1
            _, _, term, trunc, _ = env.step(a)
            plans.append(env._plan.clone())
            ended |= term | trunc
        for plan in plans:
            for dt in plan.unbind(1):
                t = t + dt
        off = (env.sim.data.time[:] != t) & ~ended
        side.synchronize()
    return {"checked": int((~ended).sum()), "time_off": int(off.sum())}


def main(mode: str, repeat_action: int, *args) -> None:
    try:
        import mjlab  # noqa: F401
        import torch
    except ImportError:
        sys.exit(SKIP)
    if not torch.cuda.is_available():
        sys.exit(SKIP)
    import warp as wp

    run = {"smoke": smoke, "idle": idle, "mixed": mixed, "nan": nan, "streams": streams}[mode]
    # The test's own reads and writes on Warp's stream too, ordered with the env's.
    with torch.cuda.stream(torch.cuda.ExternalStream(wp.get_stream("cuda:0").cuda_stream)):
        out = run(repeat_action, *args)
    print(json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), *sys.argv[3:])
