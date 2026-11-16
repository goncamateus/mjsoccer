"""examples/ppo_mjx.py: the command line, and `--play` on a saved checkpoint.

The training itself is verified by running it (ticket 13's comments), not here. What is
tested: the flag defaults the spec fixes (`--num-envs` by robot count, `--graph-mode`,
`--repeat-action` to the env, Brax's own repeat at 1), and that a Brax/Orbax checkpoint in
`runs/ppo_mjx_<env>/` plays back on the CPU backend -- through a subprocess with
`JAX_PLATFORMS=cpu`, the way a user runs it, so the GPU is never touched.

    uv sync --group gpu && uv run pytest tests/test_ppo_mjx.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "ppo_mjx.py"


def _script():
    spec = importlib.util.spec_from_file_location("ppo_mjx", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("env_id, num_envs", [
    ("ContestedPossession-v0", 8192),  # 2 robots
    ("GoToBall-v0", 2048),             # 7 robots
    ("GoToBallShoot-v0", 8192),        # 1 robot: the 2-robot default
    ("Dribbling-v0", 2048),            # 5 robots: the 7-robot default
])
def test_num_envs_defaults_to_the_robot_count_rule(env_id, num_envs):
    args = _script().parse_args(["--env", env_id])
    assert args.num_envs == num_envs
    assert args.graph_mode == "WARP"
    assert args.save == f"runs/ppo_mjx_{env_id}"


def test_flags_override_and_reach_the_config():
    args = _script().parse_args(["--env", "GoToBall-v0", "--num-envs", "512",
                                 "--repeat-action", "3", "--graph-mode", "WARP_STAGED"])
    assert (args.num_envs, args.repeat_action, args.graph_mode) == (512, 3, "WARP_STAGED")
    hp = _script().ppo_kwargs(args, episode_length=1200)
    assert hp["num_envs"] == 512
    assert hp["action_repeat"] == 1  # the repeat is the env's, never Brax's
    assert hp["batch_size"] * hp["num_minibatches"] % hp["num_envs"] == 0  # Brax's rule
    with pytest.raises(SystemExit):
        _script().parse_args(["--graph-mode", "WARP_STAGED_EX"])


def test_worlds_redraw_their_first_state_once_per_epoch_by_default():
    """Brax's auto-reset restarts a world from its cached first state; `num_resets_per_eval`
    re-draws every cache. By default once per epoch, and the default budget makes the
    stretch between two resets one whole episode: GoToBall (1 200 decisions) on 2 048
    worlds, 20 x 32 x 256 decisions an iteration = 80 per world, 15 iterations."""
    mod = _script()
    args = mod.parse_args(["--env", "GoToBall-v0", "--epochs", "3"])
    hp = mod.ppo_kwargs(args, episode_length=1200)
    assert hp["num_resets_per_eval"] == 1
    assert hp["num_timesteps"] == 3 * 15 * 163_840
    hp = mod.ppo_kwargs(mod.parse_args(["--env", "GoToBall-v0", "--epochs", "3",
                                        "--num-resets-per-eval", "2"]), episode_length=1200)
    assert (hp["num_resets_per_eval"], hp["num_timesteps"]) == (2, 3 * 2 * 15 * 163_840)
    # Off: the cached first state for the whole run, one iteration an epoch as before.
    hp = mod.ppo_kwargs(mod.parse_args(["--env", "GoToBall-v0", "--epochs", "3",
                                        "--num-resets-per-eval", "0"]), episode_length=1200)
    assert (hp["num_resets_per_eval"], hp["num_timesteps"]) == (0, 3 * 163_840)
    hp = mod.ppo_kwargs(mod.parse_args(["--env", "GoToBall-v0", "--timesteps", "1000"]),
                        episode_length=1200)
    assert hp["num_timesteps"] == 1000  # an explicit budget is kept


def test_brax_ppo_variants_are_flags_and_off_by_default():
    """Brax's own variants at the pinned commit: the KL-adaptive learning rate and the
    distributional (quantile) critic. Off, the keywords are vanilla PPO's."""
    mod = _script()
    hp = mod.ppo_kwargs(mod.parse_args(["--env", "GoToBall-v0"]), episode_length=1200)
    assert hp["learning_rate_schedule"] == "NONE"
    assert hp["use_distributional_critic"] is False
    assert hp["clipping_epsilon_value"] is None
    assert "network_factory" not in hp  # Brax's default networks

    hp = mod.ppo_kwargs(mod.parse_args([
        "--env", "GoToBall-v0", "--learning-rate-schedule", "ADAPTIVE_KL",
        "--desired-kl", "0.02", "--learning-rate-schedule-min-lr", "1e-6",
        "--learning-rate-schedule-max-lr", "1e-3"]), episode_length=1200)
    assert (hp["learning_rate_schedule"], hp["desired_kl"]) == ("ADAPTIVE_KL", 0.02)
    assert (hp["learning_rate_schedule_min_lr"], hp["learning_rate_schedule_max_lr"]) == (
        1e-6, 1e-3)
    with pytest.raises(SystemExit):
        mod.parse_args(["--learning-rate-schedule", "COSINE"])

    with pytest.raises(SystemExit):  # Brax needs the kappa of its quantile Huber loss
        mod.parse_args(["--distributional-critic"])


def test_distributional_critic_builds_quantile_value_networks():
    pytest.importorskip("brax")
    mod = _script()
    hp = mod.ppo_kwargs(mod.parse_args([
        "--env", "GoToBall-v0", "--distributional-critic", "--num-quantiles", "16",
        "--clipping-epsilon-value", "0.2"]), episode_length=1200)
    assert hp["use_distributional_critic"] is True and hp["clipping_epsilon_value"] == 0.2
    net = hp["network_factory"](8, 2)
    import jax

    params = net.value_network.init(jax.random.PRNGKey(0))
    out = net.value_network.apply(None, params, jax.numpy.zeros((3, 8)))
    assert out[1].shape == (3, 16)  # (value, quantiles)


def test_obs_noise_and_action_latency_are_off_by_default():
    args = _script().parse_args(["--env", "GoToBall-v0"])
    assert (args.obs_noise, args.action_latency) == (0.0, 0)
    args = _script().parse_args(["--obs-noise", "0.01", "--action-latency", "2"])
    assert (args.obs_noise, args.action_latency) == (0.01, 2)


def _cpu_env():
    return {**{k: v for k, v in os.environ.items() if k != "JAX_PLATFORMS"},
            "JAX_PLATFORMS": "cpu"}


MAKE_CHECKPOINT = """
import sys, jax
from brax.training.agents.ppo import checkpoint, networks
from brax.training.acme import running_statistics, specs
import gymnasium as gym, mjsoccer
env = gym.make(sys.argv[2]).unwrapped
obs, act = env.observation_space.shape[0], env.action_space.shape[0]
net = networks.make_ppo_networks(obs, act)
k = jax.random.PRNGKey(0)
params = (running_statistics.init_state(specs.Array((obs,), jax.numpy.float32)),
          net.policy_network.init(k), net.value_network.init(k))
# observation_size as ppo.train passes it: a specs.Array, saved as {"shape", "dtype"}.
cfg = checkpoint.network_config(specs.Array((obs,), jax.numpy.float32), act, True,
                                networks.make_ppo_networks)
checkpoint.save(sys.argv[1], 1234, params, cfg)
"""


def test_play_runs_a_saved_checkpoint_on_the_cpu_backend(tmp_path):
    pytest.importorskip("brax")
    run = tmp_path / "ppo_mjx_GoToBall-v0"
    p = subprocess.run([sys.executable, "-c", MAKE_CHECKPOINT, str(run), "GoToBall-v0"],
                       env=_cpu_env(), capture_output=True, text=True, timeout=600, check=False)
    assert p.returncode == 0, p.stderr[-3000:]
    (run / "args.json").write_text(json.dumps({"env": "GoToBall-v0", "repeat_action": 2}))

    p = subprocess.run([sys.executable, str(SCRIPT), "--play", str(run), "--episodes", "1",
                        "--render", "none"],
                       env=_cpu_env(), capture_output=True, text=True, timeout=600, check=False)
    assert p.returncode == 0, p.stderr[-3000:]
    out = p.stdout
    assert "000000001234" in out  # the latest step of the run directory was loaded
    assert "backend cpu" in out and "repeat_action 2" in out
    assert "episode 1:" in out


def test_randomize_trains_under_dr_and_evals_nominal_unless_asked():
    """`--randomize module:sampler` gives the train env the sampler (and Brax its
    randomization_fn); the eval env is nominal unless `--eval-randomize`."""
    pytest.importorskip("brax")
    mod = _script()
    with pytest.raises(SystemExit):
        mod.parse_args(["--eval-randomize"])  # nothing to randomize with
    sampler = "os.path:join"  # any callable: only the plumbing is checked here
    args = mod.parse_args(["--env", "ContestedPossession-v0", "--randomize", sampler])
    env, eval_env = mod.make_envs(args)
    assert env.randomize is os.path.join and env.randomization_fn is not None
    assert eval_env.randomize is None and eval_env.randomization_fn is None
    args = mod.parse_args(["--env", "ContestedPossession-v0", "--randomize", sampler,
                           "--eval-randomize"])
    _, eval_env = mod.make_envs(args)
    assert eval_env.randomize is os.path.join
    args = mod.parse_args(["--env", "ContestedPossession-v0"])
    env, eval_env = mod.make_envs(args)
    assert env.randomize is None and eval_env.randomize is None
