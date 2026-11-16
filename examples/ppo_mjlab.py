"""Train PPO (RSL-RL, through mjlab's `MjlabOnPolicyRunner`) on an mjsoccer task on mjlab, one GPU.

    uv run --group mjlab examples/ppo_mjlab.py --env ContestedPossession-v0
    uv run --group mjlab examples/ppo_mjlab.py --eval runs/ppo_mjlab_ContestedPossession-v0
    uv run --group mjlab examples/ppo_mjlab.py --play runs/ppo_mjlab_ContestedPossession-v0

The sibling of `ppo_mjx.py` (same flags, same PPO defaults) for the mjlab backend; `--env`
defaults to ContestedPossession, the first task ported to it. Long runs
belong under `systemd-run --user --scope -p MemoryMax=10G`. Training writes the flags to
`args.json`, RSL-RL's checkpoint `model_<iteration>.pt` at the end of every epoch (and
`model_0.pt`, RSL-RL's own save on iteration 0), and the policy in ONNX (`<run dir
name>.onnx`, deterministic: normalizer + MLP -> the action mean) at every checkpoint, a
failed export aborting the run. `--play` runs that ONNX with onnxruntime on the CPU backend (`gymnasium.make`,
the Studio viewer); `--eval` runs it on mjlab once per world and aborts, with Warp's
overflow bits named per type, if any world diverged, overflowed or reset overlapping.

The PPO flags keep `ppo_mjx`'s names and defaults, mapped onto RSL-RL: `--unroll-length` is
`num_steps_per_env`, `--num-updates-per-batch` its learning epochs, `--num-minibatches` its
mini-batches; Brax's value loss (0.5 * 0.5 * squared error) is `value_loss_coef` 0.25, its
networks (policy 4 x 32, value 5 x 256, swish, normalized observations) are the models. Not
carried over: Brax's `--batch-size` (an RSL-RL iteration is one unroll of every world),
`--reward-scaling` (RSL-RL has none), the tanh-squashed policy (RSL-RL's Gaussian is clipped
to the action box instead) and the Brax-only variants.

An epoch is `refill_pool()` (every world's first state drawn anew; it replaces
`--num-resets-per-eval`), then PPO iterations, then an eval on `--num-eval-envs` worlds.
The default budget (`--timesteps` unset) gives an epoch the iterations one episode takes;
`--timesteps` is split over the epochs. Per epoch the log has the decisions so far,
decisions/s, the eval return and the eval episodes' rate of the three failure counters,
the training world-decisions so far with each failure (diverged/overflow/reset_overlap) and
the device memory in use (the whole GPU's, other processes included).

`--randomize module:sampler` trains under domain randomization: the same sampler as
`ppo_mjx`'s, each world its own variant for the run, drawn from `--seed`. The per-epoch eval
runs on the nominal model unless `--eval-randomize`. `--obs-noise` and `--action-latency`
(both off by default) apply to training and the per-epoch eval; `--eval` and `--play` run
without them, nominal (as `ppo_mjx`).

`--num-envs` defaults by robot count, measured in long training runs (`NUM_ENVS`).
"""

from __future__ import annotations

import sys
from pathlib import Path

# examples/random.py shadows stdlib `random` while this directory is sys.path[0].
if sys.path and Path(sys.path[0] or ".").resolve() == Path(__file__).resolve().parent:
    sys.path.pop(0)

import argparse
import json
import math
import time

# Measured in long runs up to 32 768 worlds (ticket 11, .scratch/mjlab-port/research/
# 06-num-envs.md): flat memory everywhere; 2 robots stays at 8 192 (bigger batches barely add
# decisions/s and learn less per decision; rare policy-driven overflows, ~4e-7 per
# world-decision, end only their world), 7 at 16 384 (0 failures, 5.6 GB; 32 768 takes 10.6
# of the 3060's 12 GB).
NUM_ENVS = {2: 8192, 7: 16384}
METRICS = ("diverged", "overflow", "reset_overlap")


def _n_robots(env_id: str) -> int:
    import mjsoccer
    from mjsoccer import tasks

    cls = getattr(tasks, mjsoccer.ENV_IDS[env_id][0])
    return cls.n_blue + cls.n_yellow


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="ContestedPossession-v0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None, help="run directory (default runs/ppo_mjlab_<env>)")
    ap.add_argument("--play", default=None, metavar="DIR",
                    help="watch a saved policy (ONNX) on the CPU backend instead of training")
    ap.add_argument("--eval", default=None, metavar="DIR",
                    help="evaluate a saved policy (ONNX) on mjlab; aborts on any failure")
    ap.add_argument("--episodes", type=int, default=None, help="--play: stop after this many")
    ap.add_argument("--render", choices=("human", "none"), default="human",
                    help="--play: the Studio viewer, or headless")
    ap.add_argument("--repeat-action", type=int, default=1,
                    help="control steps (25 ms each) per decision, passed to the env")
    ap.add_argument("--num-envs", type=int, default=None,
                    help=f"worlds per batch (default by robot count: {NUM_ENVS})")
    ap.add_argument("--epochs", type=int, default=30, help="training epochs, an eval after each")
    ap.add_argument("--timesteps", type=int, default=None,
                    help="decisions in all (default: one episode per world per epoch)")
    ap.add_argument("--num-eval-envs", type=int, default=128)
    ap.add_argument("--obs-noise", type=float, default=0.0,
                    help="std of Gaussian noise on the (normalised) obs, train and eval; 0 off")
    ap.add_argument("--action-latency", type=int, default=0,
                    help="delay each action by this many control steps; 0 off")
    ap.add_argument("--randomize", default=None, metavar="MODULE:SAMPLER",
                    help="domain randomization: a sampler(mj_model, rng) -> None, imported "
                         "by name (mjsoccer.randomize); each world gets its variant")
    ap.add_argument("--eval-randomize", action="store_true",
                    help="evaluate under the same sampler, own variants (default: the nominal model)")
    hp = ap.add_argument_group("PPO (ppo_mjx's flags, on RSL-RL)")
    hp.add_argument("--learning-rate", type=float, default=3e-4)
    hp.add_argument("--entropy-cost", type=float, default=1e-2)
    hp.add_argument("--discounting", type=float, default=0.99)
    hp.add_argument("--gae-lambda", type=float, default=0.95)
    hp.add_argument("--clipping-epsilon", type=float, default=0.2)
    hp.add_argument("--unroll-length", type=int, default=20)
    hp.add_argument("--num-minibatches", type=int, default=32)
    hp.add_argument("--num-updates-per-batch", type=int, default=4)
    hp.add_argument("--max-grad-norm", type=float, default=0.5)
    args = ap.parse_args(argv)
    if args.eval_randomize and not args.randomize:
        ap.error("--eval-randomize needs --randomize")
    if args.num_envs is None:
        # robot counts without a measured default take the next larger one's (fewer worlds)
        n = _n_robots(args.env)
        args.num_envs = NUM_ENVS[min(k for k in NUM_ENVS if k >= n)]
    if args.save is None:
        args.save = f"runs/ppo_mjlab_{args.env}"
    return args


def train_cfg(args) -> dict:
    """The `MjlabOnPolicyRunner` config: ppo_mjx's PPO defaults on RSL-RL."""
    from dataclasses import asdict

    from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

    return asdict(RslRlOnPolicyRunnerCfg(
        seed=args.seed,
        num_steps_per_env=args.unroll_length,
        obs_groups={"actor": ("actor",), "critic": ("actor",)},
        save_interval=10**9,  # the end of every epoch's learn() saves
        logger="tensorboard",
        upload_model=False,
        clip_actions=1.0,
        actor=RslRlModelCfg(hidden_dims=(32,) * 4, activation="swish", obs_normalization=True,
                            distribution_cfg={"class_name": "GaussianDistribution",
                                              "init_std": 1.0, "std_type": "scalar"}),
        critic=RslRlModelCfg(hidden_dims=(256,) * 5, activation="swish", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(
            num_learning_epochs=args.num_updates_per_batch,
            num_mini_batches=args.num_minibatches,
            learning_rate=args.learning_rate,
            schedule="fixed",
            gamma=args.discounting,
            lam=args.gae_lambda,
            entropy_coef=args.entropy_cost,
            max_grad_norm=args.max_grad_norm,
            value_loss_coef=0.25,
            use_clipped_value_loss=False,
            clip_param=args.clipping_epsilon,
        ),
    ))


def iterations_per_epoch(args, episode_length: int) -> int:
    """PPO iterations between two pool refills: one episode's worth of unrolls, or
    `--timesteps` split over the epochs."""
    if args.timesteps:
        total = math.ceil(args.timesteps / (args.num_envs * args.unroll_length))
        return max(1, math.ceil(total / args.epochs))
    return math.ceil(episode_length / args.unroll_length)


def _saved_args(run) -> dict:
    p = Path(run) / "args.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _onnx(run) -> Path:
    found = sorted(Path(run).glob("*.onnx"))
    if not found:
        raise FileNotFoundError(f"no ONNX policy in {run}")
    return found[-1]


def run_episodes(env, act, check: bool = False) -> dict:
    """One episode per world of `env` (a `SoccerEnv` with `auto_reset` off) from a fresh
    pool, `act(obs) -> action` deterministic. Returns the mean return and length and each
    failure counter's rate per episode; `check` aborts on any failure instead, naming
    Warp's overflow bits per type before a reset clears them."""
    import torch

    from mjsoccer.mjlab.base import check_failures

    # Inference tensors throughout: RSL-RL's rollout already made the train env's so.
    with torch.inference_mode():
        env.refill_pool()
        obs = env.reset()[0]["actor"]
        n, dev = env.num_envs, env.device
        active = torch.ones(n, dtype=torch.bool, device=dev)
        ret, length = torch.zeros(n, device=dev), torch.zeros(n, device=dev)
        failed = {k: torch.zeros(n, device=dev) for k in METRICS}
        while active.any():
            obs, reward, term, trunc, _ = env.step(act(obs).clamp(-1.0, 1.0))
            obs = obs["actor"]
            if check:
                check_failures({k: v[active] for k, v in env.metrics.items()},
                               env.sim.data.overflow[:][active])
                if (env.metrics["reset_overlap"][active] > 0).any():
                    raise RuntimeError("mjlab: a world was placed overlapping (reset_overlap)")
            for k in METRICS:
                failed[k] = torch.maximum(failed[k], torch.where(active, env.metrics[k], 0.0))
            ret += torch.where(active, reward, 0.0)
            length += active
            done = term | trunc
            active &= ~done
            if done.any():  # auto_reset is off: the bits above were read before this clears them
                obs = env.reset(env_ids=done.nonzero().squeeze(-1))[0]["actor"]
    return {"return": ret.mean().item(), "return_std": ret.std().item() if n > 1 else 0.0,
            "length": length.mean().item(), **{k: v.mean().item() for k, v in failed.items()}}


def env_kwargs(args) -> tuple[dict, dict]:
    """(train env, per-epoch eval env) `mjsoccer.mjlab.make` kwargs: noise and latency on
    both, the sampler on the eval env only with `--eval-randomize`."""
    import pkgutil

    sampler = pkgutil.resolve_name(args.randomize) if args.randomize else None
    train = {"obs_noise": args.obs_noise, "action_latency": args.action_latency,
             "randomize": sampler}
    return train, {**train, "randomize": sampler if args.eval_randomize else None}


def _make(env_id, nworld, repeat_action, seed, **kw):
    import mjsoccer.mjlab

    return mjsoccer.mjlab.make(env_id, nworld=nworld, repeat_action=repeat_action, seed=seed,
                               **kw)


def train(args) -> int:
    import torch
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx

    class Runner(MjlabOnPolicyRunner):
        export_onnx_on_save = True

        def export_policy_to_onnx(self, path, filename="policy.onnx", verbose=False):
            # mjlab's export fixes the batch at 1; --eval runs a batch.
            model = self.alg.get_policy().as_onnx().to("cpu").eval()
            torch.onnx.export(model, model.get_dummy_inputs(), str(Path(path) / filename),
                              opset_version=18, input_names=model.input_names,
                              output_names=model.output_names, dynamo=False,
                              dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}})

        def _export_onnx_with_metadata(self, checkpoint_path):
            # mjlab's swallows a failed export ("training continues"): a stale ONNX, unnoticed.
            path, filename, onnx_path = self._get_export_paths(checkpoint_path)
            self.export_policy_to_onnx(str(path), filename)
            attach_metadata_to_onnx(str(onnx_path), self.get_export_metadata(self._get_run_name()))

        def get_export_metadata(self, run_name):
            # mjlab's reads a "robot" entity and joint actions; the scene has neither.
            return {"run_path": run_name, "env": args.env, "repeat_action": args.repeat_action}

    torch.manual_seed(args.seed)
    train_kw, eval_kw = env_kwargs(args)
    env = _make(args.env, args.num_envs, args.repeat_action, args.seed, **train_kw)
    eval_env = _make(args.env, args.num_eval_envs, args.repeat_action, args.seed + 1, **eval_kw)
    eval_env.cfg.auto_reset = False
    run = Path(args.save).resolve()
    run.mkdir(parents=True, exist_ok=True)
    (run / "args.json").write_text(json.dumps(vars(args), indent=1))
    length = math.ceil(env.logic.max_steps / env.logic.repeat_action)
    iters = iterations_per_epoch(args, length)
    per_iter = args.num_envs * args.unroll_length
    with torch.inference_mode():  # the rollout's tensors are inference ones
        fails = torch.zeros(len(METRICS), device=env.device)
    env_step = env.step

    def step(action):  # training worlds with a failure, counted on the device (no sync)
        out = env_step(action)
        fails.add_(torch.stack([env.metrics[k] for k in METRICS]).gt(0).sum(1))
        return out

    env.step = step
    vec = RslRlVecEnvWrapper(env, clip_actions=1.0)
    runner = Runner(vec, train_cfg(args), str(run), device=env.device)
    print(f"{args.env} on {env.device}: {args.num_envs} worlds, repeat_action "
          f"{args.repeat_action}, {args.epochs} epochs x {iters} iterations x {per_iter:,} "
          f"decisions, DR {args.randomize or 'off'}"
          f"{' (eval too)' if args.eval_randomize else ''}, obs noise {args.obs_noise}, "
          f"action latency {args.action_latency} -> {run}", flush=True)
    t0, decisions = time.perf_counter(), 0
    for epoch in range(1, args.epochs + 1):
        if epoch > 1:  # the startup event drew the first pool
            with torch.inference_mode():  # RSL-RL's rollout left the env's tensors inference ones
                env.refill_pool()
                vec.reset()
        t = time.perf_counter()
        runner.learn(iters)
        # learn() leaves the counter on its last iteration, which the next call would redo.
        runner.current_learning_iteration += 1
        dps = iters * per_iter / (time.perf_counter() - t)
        decisions += iters * per_iter
        policy = runner.alg.get_policy().as_onnx().eval()  # the graph the ONNX holds
        m = run_episodes(eval_env, policy)
        rates = " ".join(f"{k} {m[k]:.4f}" for k in METRICS)
        train_fails = "/".join(str(int(v)) for v in fails.tolist())
        free, total = torch.cuda.mem_get_info(env.device)
        print(f"epoch {epoch:>3}  decisions {decisions:>13,}  dps {dps:7.0f}  eval return "
              f"{m['return']:8.3f} len {m['length']:6.1f}  {rates}  train failures "
              f"{train_fails}  device {(total - free) // 2**20} MB  "
              f"t {time.perf_counter() - t0:6.0f}s", flush=True)
    print(f"saved {max(run.glob('model_*.pt'), key=lambda p: int(p.stem.split('_')[1]))}, "
          f"{_onnx(run)}")
    return 0


def evaluate(args) -> int:
    """The ONNX policy (onnxruntime, CPU) once per world on mjlab, every step checked."""
    import onnxruntime as ort
    import torch

    path, saved = _onnx(args.eval), _saved_args(args.eval)
    env_id, repeat = saved.get("env", args.env), saved.get("repeat_action", args.repeat_action)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def act(obs):
        a = sess.run(None, {"obs": obs.float().cpu().numpy()})[0]
        return torch.from_numpy(a).to(obs.device)

    env = _make(env_id, args.num_eval_envs, repeat, args.seed)
    env.cfg.auto_reset = False
    print(f"eval {path}: {env_id}, {args.num_eval_envs} worlds, repeat_action {repeat}",
          flush=True)
    m = run_episodes(env, act, check=True)
    print(f"return {m['return']:.3f} +- {m['return_std']:.3f}, length {m['length']:.1f} "
          "decisions, no failures", flush=True)
    return 0


def play(args) -> int:
    """onnxruntime on the CPU, the CPU backend's env and its Studio viewer."""
    import gymnasium as gym
    import numpy as np
    import onnxruntime as ort

    import mjsoccer  # noqa: F401  (registers the ids)

    path, saved = _onnx(args.play), _saved_args(args.play)
    env_id, repeat = saved.get("env", args.env), saved.get("repeat_action", args.repeat_action)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    print(f"play {path}: {env_id}, repeat_action {repeat}, onnxruntime cpu", flush=True)
    env = gym.make(env_id, repeat_action=repeat,
                   render_mode=None if args.render == "none" else args.render)
    obs, _ = env.reset(seed=args.seed)
    episode, ret, length = 0, 0.0, 0
    while args.episodes is None or episode < args.episodes:
        a = sess.run(None, {"obs": obs[None].astype(np.float32)})[0][0]
        obs, reward, terminated, truncated, _ = env.step(np.clip(a, -1.0, 1.0).astype(np.float64))
        ret, length = ret + reward, length + 1
        if terminated or truncated:
            episode += 1
            print(f"episode {episode}: return {ret:.3f}, {length} decisions, "
                  f"{'terminated' if terminated else 'truncated'}", flush=True)
            obs, _ = env.reset()
            ret, length = 0.0, 0
    env.close()
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.play:
        return play(args)
    if args.eval:
        return evaluate(args)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
