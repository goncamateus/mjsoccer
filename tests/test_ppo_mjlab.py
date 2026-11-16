"""examples/ppo_mjlab.py: the command line, the RSL-RL config it builds, and a training smoke.

In-process: the flag defaults (ppo_mjx's PPO defaults, mapped onto RSL-RL), the epoch
budget. On the GPU (a fresh subprocess, `_gpu.py`; skipped without CUDA or the `mjlab`
group): a few training iterations write a checkpoint, the config and the ONNX policy; then
`--eval` loads the run on mjlab and `--play` (headless) runs the ONNX on the CPU backend.

    uv sync --group mjlab && uv run pytest tests/test_ppo_mjlab.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from _gpu import run

pytest.importorskip("torch")
pytest.importorskip("mjlab")

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "ppo_mjlab.py"
ENV = "ContestedPossession-v0"


def _script():
    spec = importlib.util.spec_from_file_location("ppo_mjlab", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_defaults_are_ppo_mjx_s():
    mod = _script()
    args = mod.parse_args(["--env", ENV])
    assert args.num_envs == 8192 and args.num_eval_envs == 128 and args.epochs == 30
    assert args.save == f"runs/ppo_mjlab_{ENV}"
    assert mod.parse_args(["--env", "GoToBall-v0"]).num_envs == 16384  # 7 robots, ticket 11
    # robot counts without a measured default take the next larger one's
    assert mod.parse_args(["--env", "GoToBallShoot-v0"]).num_envs == 8192   # 1 robot
    assert mod.parse_args(["--env", "Dribbling-v0"]).num_envs == 16384      # 5 robots
    cfg = mod.train_cfg(args)
    alg = cfg["algorithm"]
    assert alg["learning_rate"] == 3e-4 and alg["schedule"] == "fixed"
    assert alg["entropy_coef"] == 1e-2 and alg["gamma"] == 0.99 and alg["lam"] == 0.95
    assert alg["clip_param"] == 0.2 and alg["max_grad_norm"] == 0.5
    assert alg["num_learning_epochs"] == 4 and alg["num_mini_batches"] == 32
    assert alg["value_loss_coef"] == 0.25  # Brax's 0.5 * 0.5 on the squared error
    assert alg["use_clipped_value_loss"] is False
    assert cfg["num_steps_per_env"] == 20
    assert cfg["seed"] == 0 and cfg["logger"] == "tensorboard"
    # Brax's default networks: policy 4 x 32, value 5 x 256, swish, normalized obs.
    assert cfg["actor"]["hidden_dims"] == (32,) * 4 and cfg["critic"]["hidden_dims"] == (256,) * 5
    assert cfg["actor"]["activation"] == cfg["critic"]["activation"] == "swish"
    assert cfg["actor"]["obs_normalization"] and cfg["critic"]["obs_normalization"]
    assert cfg["obs_groups"] == {"actor": ("actor",), "critic": ("actor",)}


def test_flags_reach_the_config():
    mod = _script()
    args = mod.parse_args(["--env", ENV, "--seed", "3", "--learning-rate", "1e-3",
                           "--entropy-cost", "0", "--discounting", "0.9", "--gae-lambda", "0.9",
                           "--clipping-epsilon", "0.1", "--unroll-length", "8",
                           "--num-minibatches", "4", "--num-updates-per-batch", "2",
                           "--max-grad-norm", "1.0", "--num-envs", "64", "--repeat-action", "2"])
    cfg = mod.train_cfg(args)
    alg = cfg["algorithm"]
    assert (alg["learning_rate"], alg["entropy_coef"], alg["gamma"], alg["lam"]) == (
        1e-3, 0.0, 0.9, 0.9)
    assert (alg["clip_param"], alg["num_mini_batches"], alg["num_learning_epochs"],
            alg["max_grad_norm"]) == (0.1, 4, 2, 1.0)
    assert cfg["num_steps_per_env"] == 8 and cfg["seed"] == 3
    assert (args.num_envs, args.repeat_action) == (64, 2)


def test_noise_latency_and_dr_flags_reach_the_envs():
    """ppo_mjx's semantics: `--obs-noise` and `--action-latency` (off by default) apply to
    the train env and the per-epoch eval, never to `--eval`; `--randomize module:sampler`
    to the train env, the eval env only with `--eval-randomize`, which needs it."""
    import operator

    mod = _script()
    args = mod.parse_args(["--env", ENV])
    assert (args.obs_noise, args.action_latency, args.randomize) == (0.0, 0, None)
    train, ev = mod.env_kwargs(args)
    assert train == ev == {"obs_noise": 0.0, "action_latency": 0, "randomize": None}
    with pytest.raises(SystemExit):
        mod.parse_args(["--eval-randomize"])  # nothing to randomize with
    args = mod.parse_args(["--env", ENV, "--obs-noise", "0.01", "--action-latency", "2",
                           "--randomize", "operator:is_"])
    train, ev = mod.env_kwargs(args)
    assert train == {"obs_noise": 0.01, "action_latency": 2, "randomize": operator.is_}
    assert ev == {**train, "randomize": None}
    args = mod.parse_args(["--env", ENV, "--randomize", "operator:is_", "--eval-randomize"])
    assert mod.env_kwargs(args)[1]["randomize"] is operator.is_


def test_an_epoch_is_one_episode_per_world_unless_a_budget_is_given():
    """ContestedPossession: 1 200 decisions, 20-step unrolls -> 60 iterations an epoch. An
    explicit `--timesteps` is split over the epochs, at least one iteration each."""
    mod = _script()
    args = mod.parse_args(["--env", ENV, "--epochs", "3"])
    assert mod.iterations_per_epoch(args, episode_length=1200) == 60
    args = mod.parse_args(["--env", ENV, "--epochs", "3", "--num-envs", "100",
                           "--timesteps", "10000"])
    assert mod.iterations_per_epoch(args, episode_length=1200) == 2  # ceil(5 its / 3)
    args = mod.parse_args(["--env", ENV, "--epochs", "3", "--timesteps", "1"])
    assert mod.iterations_per_epoch(args, episode_length=1200) == 1


def test_train_eval_play_smoke(tmp_path):
    """A few iterations on a small batch, then `--eval` and `--play` on what was saved."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    save = tmp_path / "run"
    # With noise, latency and DR on (a sampler that changes nothing: the plumbing only;
    # tests/test_mjlab_randomize.py checks DR itself).
    p = run(SCRIPT, "--env", ENV, "--save", save, "--num-envs", "64", "--num-eval-envs", "4",
            "--epochs", "2", "--timesteps", "1", "--unroll-length", "4",
            "--num-minibatches", "2", "--obs-noise", "0.05", "--action-latency", "2",
            "--randomize", "operator:is_", "--eval-randomize", timeout=3600)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-4000:]
    out = p.stdout
    assert out.count("epoch") >= 2 and "dps" in out and "eval return" in out, out[-3000:]
    assert "DR operator:is_ (eval too)" in out and "obs noise 0.05" in out, out[-3000:]
    assert "action latency 2" in out, out[-3000:]
    for k in ("diverged", "overflow", "reset_overlap"):
        assert k in out
    assert "train failures 0/0/0" in out and " MB " in out, out[-3000:]
    assert json.loads((save / "args.json").read_text())["env"] == ENV
    assert sorted(save.glob("model_*.pt")), list(save.iterdir())
    onnx = list(save.glob("*.onnx"))
    assert len(onnx) == 1, list(save.iterdir())

    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx[0]), providers=["CPUExecutionProvider"])
    import gymnasium as gym

    import mjsoccer  # noqa: F401

    env = gym.make(ENV)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    import numpy as np

    for batch in (1, 3):  # the same obs/action spaces, any batch
        a = sess.run(None, {"obs": np.zeros((batch, obs_dim), np.float32)})[0]
        assert a.shape == (batch, act_dim)

    p = run(SCRIPT, "--eval", save, "--num-eval-envs", "4", timeout=3600)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-4000:]
    assert "return" in p.stdout and "no failures" in p.stdout, p.stdout[-3000:]

    p = run(SCRIPT, "--play", save, "--episodes", "1", "--render", "none", timeout=3600)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-4000:]
    assert "episode 1:" in p.stdout and ENV in p.stdout, p.stdout[-3000:]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
