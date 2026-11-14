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
    hp = _script().ppo_kwargs(args)
    assert hp["num_envs"] == 512
    assert hp["action_repeat"] == 1  # the repeat is the env's, never Brax's
    assert hp["batch_size"] * hp["num_minibatches"] % hp["num_envs"] == 0  # Brax's rule
    with pytest.raises(SystemExit):
        _script().parse_args(["--graph-mode", "WARP_STAGED_EX"])


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
                       env=_cpu_env(), capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-3000:]
    (run / "args.json").write_text(json.dumps({"env": "GoToBall-v0", "repeat_action": 2}))

    p = subprocess.run([sys.executable, str(SCRIPT), "--play", str(run), "--episodes", "1",
                        "--render", "none"],
                       env=_cpu_env(), capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-3000:]
    out = p.stdout
    assert "000000001234" in out  # the latest step of the run directory was loaded
    assert "backend cpu" in out and "repeat_action 2" in out
    assert "episode 1:" in out
