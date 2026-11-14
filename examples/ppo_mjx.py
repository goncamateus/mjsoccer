"""Train PPO (Brax, vanilla `ppo.train`) on any mjsoccer task on MJX-Warp, one GPU.

    uv run --group gpu examples/ppo_mjx.py --env GoToBall-v0
    uv run --group gpu examples/ppo_mjx.py --env GoToBall-v0 --eval runs/ppo_mjx_GoToBall-v0
    uv run --group gpu examples/ppo_mjx.py --play runs/ppo_mjx_GoToBall-v0

Long runs belong under `systemd-run --user --scope -p MemoryMax=10G`. Training writes an
Orbax checkpoint per epoch to `runs/ppo_mjx_<env>/<step>/` (Brax's own layout) and the
flags to `args.json` next to them. `--play` loads the latest one and drives the CPU
backend (`gymnasium.make`, the Studio viewer) with JAX on the CPU; `--eval` runs the policy
on MJX once per world and aborts if any world diverged, overflowed or reset overlapping.

One env step is one decision of `--repeat-action` control steps (the env's repeat; Brax's
`action_repeat` stays 1). The hyperparameters are flags; the defaults are a baseline, not
a tuning (the spec leaves tuning out). Per epoch the log has the decisions so far,
decisions/s, the eval return, and the eval episodes' rate of the three failure counters
(per episode: diverged, overflow, reset_overlap), plus the GPU memory in use.

`--num-envs` defaults to 8 192 worlds with 2 robots and 2 048 with 7: the largest batch
that ran 30 epochs with >= 2 GB of the GPU free (ticket 13). In `--graph-mode WARP` the end
of each epoch drops Warp's captured graphs (they pile up otherwise).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# examples/random.py shadows stdlib `random` while this directory is sys.path[0] (run as a
# script). Same guard as ppo_agent.py, but only when it is ours to pop.
if sys.path and Path(sys.path[0] or ".").resolve() == Path(__file__).resolve().parent:
    sys.path.pop(0)

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

# Robots in the scene -> worlds per batch. Measured on the real tasks (ticket 13).
NUM_ENVS = {2: 8192, 7: 2048}
GRAPH_MODES = ("WARP", "WARP_STAGED")


def _n_robots(env_id: str) -> int:
    import mjsoccer
    from mjsoccer import tasks

    cls = getattr(tasks, mjsoccer.ENV_IDS[env_id][0])
    return cls.n_blue + cls.n_yellow


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="GoToBall-v0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None, help="run directory (default runs/ppo_mjx_<env>)")
    ap.add_argument("--play", default=None, metavar="DIR",
                    help="watch a saved policy on the CPU backend instead of training")
    ap.add_argument("--eval", default=None, metavar="DIR",
                    help="evaluate a saved policy on MJX; aborts on any failure counter")
    ap.add_argument("--episodes", type=int, default=None, help="--play: stop after this many")
    ap.add_argument("--render", choices=("human", "none"), default="human",
                    help="--play: the Studio viewer, or headless")
    ap.add_argument("--repeat-action", type=int, default=1,
                    help="control steps (25 ms each) per decision, passed to the env")
    ap.add_argument("--num-envs", type=int, default=None,
                    help="worlds per batch (default by robot count: %s)" % NUM_ENVS)
    ap.add_argument("--graph-mode", choices=GRAPH_MODES, default="WARP")
    ap.add_argument("--epochs", type=int, default=30, help="training epochs, an eval after each")
    ap.add_argument("--timesteps", type=int, default=None,
                    help="decisions in all (default: one PPO iteration per epoch)")
    ap.add_argument("--num-eval-envs", type=int, default=128)
    hp = ap.add_argument_group("PPO (brax ppo.train)")
    hp.add_argument("--learning-rate", type=float, default=3e-4)
    hp.add_argument("--entropy-cost", type=float, default=1e-2)
    hp.add_argument("--discounting", type=float, default=0.99)
    hp.add_argument("--gae-lambda", type=float, default=0.95)
    hp.add_argument("--clipping-epsilon", type=float, default=0.2)
    hp.add_argument("--unroll-length", type=int, default=20)
    hp.add_argument("--num-minibatches", type=int, default=32)
    hp.add_argument("--num-updates-per-batch", type=int, default=4)
    hp.add_argument("--batch-size", type=int, default=None,
                    help="default num_envs // 8: four unrolls of every world per iteration")
    hp.add_argument("--reward-scaling", type=float, default=1.0)
    hp.add_argument("--max-grad-norm", type=float, default=0.5)
    args = ap.parse_args(argv)
    if args.num_envs is None:
        args.num_envs = NUM_ENVS[_n_robots(args.env)]
    if args.batch_size is None:
        args.batch_size = max(1, args.num_envs // 8)
    if args.save is None:
        args.save = f"runs/ppo_mjx_{args.env}"
    return args


def ppo_kwargs(args) -> dict:
    """The `ppo.train` keywords the flags set (the env, eval env and callbacks aside)."""
    per_iteration = args.batch_size * args.num_minibatches * args.unroll_length
    return dict(
        num_timesteps=args.timesteps or args.epochs * per_iteration,
        num_evals=args.epochs + 1,
        num_envs=args.num_envs,
        num_eval_envs=args.num_eval_envs,
        action_repeat=1,
        learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost,
        discounting=args.discounting,
        gae_lambda=args.gae_lambda,
        clipping_epsilon=args.clipping_epsilon,
        unroll_length=args.unroll_length,
        batch_size=args.batch_size,
        num_minibatches=args.num_minibatches,
        num_updates_per_batch=args.num_updates_per_batch,
        reward_scaling=args.reward_scaling,
        max_grad_norm=args.max_grad_norm,
        normalize_observations=True,
        seed=args.seed,
    )


def latest_checkpoint(path) -> Path:
    """A step directory as Brax writes them (`<step:012d>/`): the run's newest, or `path`."""
    path = Path(path).resolve()  # Orbax refuses relative paths
    steps = sorted(p for p in path.iterdir() if p.is_dir() and p.name.isdigit())
    return steps[-1] if steps else path


def _gpu_mb() -> tuple[int, int]:
    """(MB this process holds on the GPU, MB free on it), from nvidia-smi."""
    import subprocess

    def q(*a):
        return subprocess.run(["nvidia-smi", *a, "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.split("\n")

    mine = sum(int(line.split(",")[1]) for line in q("--query-compute-apps=pid,used_memory")
               if line.strip() and int(line.split(",")[0]) == os.getpid())
    used, total = map(int, q("--query-gpu=memory.used,memory.total")[0].split(","))
    return mine, total - used


def _drop_graphs() -> None:
    """Spec, Fase 4: WARP mode recaptures graphs every epoch; let the old ones go."""
    import gc

    import warp as wp

    wp.synchronize()
    wp.clear_jax_callable_graph_cache()
    gc.collect()
    wp.synchronize()


def train(args) -> int:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    from brax.training.agents.ppo import train as ppo

    import mjsoccer.mjx
    from mjsoccer.mjx import wrappers

    def make(n):
        return mjsoccer.mjx.make(args.env, repeat_action=args.repeat_action,
                                 graph_mode=args.graph_mode, nworld=n)

    env, eval_env = make(args.num_envs), make(args.num_eval_envs)
    run = Path(args.save).resolve()
    run.mkdir(parents=True, exist_ok=True)
    (run / "args.json").write_text(json.dumps(vars(args), indent=1))
    hp = ppo_kwargs(args)
    print(f"{args.env} on {jax.devices()[0]}: {args.num_envs} worlds, repeat_action "
          f"{args.repeat_action}, {args.graph_mode}, {hp['num_timesteps']:,} decisions in "
          f"{args.epochs} epochs -> {run}", flush=True)

    t0 = time.perf_counter()
    last = [0, t0]

    def progress(step, m):
        if args.graph_mode == "WARP":
            _drop_graphs()
        now = time.perf_counter()
        dps = (step - last[0]) / (now - last[1]) if step > last[0] else float("nan")
        last[:] = step, now
        mine, free = _gpu_mb()
        rates = " ".join(f"{k} {m.get(f'eval/episode_{k}', float('nan')):.4f}"
                         for k in ("diverged", "overflow", "reset_overlap"))
        print(f"step {step:>11,}  dps {dps:7.0f} (train {m.get('training/sps', float('nan')):7.0f})"
              f"  eval return {m.get('eval/episode_reward', float('nan')):8.3f}"
              f" len {m.get('eval/avg_episode_length', float('nan')):6.1f}  {rates}"
              f"  gpu {mine} MB, free {free} MB  t {now - t0:6.0f}s", flush=True)

    ppo.train(env, eval_env=eval_env, wrap_env_fn=wrappers.wrap,
              episode_length=wrappers.episode_length(env), save_checkpoint_path=str(run),
              progress_fn=progress, **hp)
    print(f"saved {latest_checkpoint(run)}")
    return 0


def evaluate(args) -> int:
    """The policy once per world on MJX (`--num-eval-envs` worlds), host-checked every step."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import numpy as np

    import mjsoccer.mjx

    ckpt = latest_checkpoint(args.eval)
    saved = _saved_args(args.eval)
    env_id, repeat = saved.get("env", args.env), saved.get("repeat_action", args.repeat_action)
    n = args.num_eval_envs
    env = mjsoccer.mjx.make(env_id, repeat_action=repeat, graph_mode=args.graph_mode, nworld=n)
    policy = load_policy(ckpt)
    step = jax.jit(jax.vmap(env.step))
    act = jax.jit(lambda obs, k: policy(obs, k)[0])
    key = jax.random.PRNGKey(args.seed)
    s = jax.jit(jax.vmap(env.reset))(jax.random.split(key, n))
    active, ret, length = np.ones(n, bool), np.zeros(n), np.zeros(n)
    print(f"eval {ckpt}: {env_id}, {n} worlds, repeat_action {repeat}", flush=True)
    while active.any():
        mjsoccer.mjx.check_failures(s)
        if np.asarray(s.metrics["reset_overlap"]).any():
            raise RuntimeError("MJX: a world was placed overlapping (reset_overlap)")
        key, k = jax.random.split(key)
        s = step(s, act(s.obs, k))
        ret += np.where(active, np.asarray(s.reward), 0.0)
        length += active
        active &= np.asarray(s.done) == 0
    mjsoccer.mjx.check_failures(s)
    print(f"return {ret.mean():.3f} +- {ret.std():.3f}, length {length.mean():.1f} decisions, "
          "no failures", flush=True)
    return 0


def load_policy(ckpt: Path):
    """`brax...ppo.checkpoint.load_policy`, deterministic, minus its bug at the pinned
    commit: `load_config` looks up every kernel-init name, and the default factory saves
    `mean_kernel_init_fn` as null (KeyError: None). Null entries are the factory's own
    defaults, so they are dropped before rebuilding the network."""
    from brax.training import checkpoint as bc
    from brax.training import networks as bn
    from brax.training.agents.ppo import checkpoint
    from brax.training.agents.ppo import networks as ppo_networks
    from ml_collections import config_dict

    cfg = json.loads((ckpt / "ppo_network_config.json").read_text())
    # ppo.train saves the observation size as a specs.Array, {"shape": [n], "dtype": ...},
    # which get_network then reads as a dict of observation keys.
    if isinstance(cfg["observation_size"], dict) and "shape" in cfg["observation_size"]:
        cfg["observation_size"] = cfg["observation_size"]["shape"][-1]
    kw ={k: v for k, v in cfg["network_factory_kwargs"].items() if v is not None}
    if "activation" in kw:
        kw["activation"] = bn.ACTIVATION[kw["activation"]]
    for k in bc._KERNEL_INIT_FN_KEYWORDS:
        if k in kw:
            kw[k] = bn.KERNEL_INITIALIZER[kw[k]]
    net = bc.get_network(config_dict.create(**{**cfg, "network_factory_kwargs": kw}),
                         ppo_networks.make_ppo_networks)
    return ppo_networks.make_inference_fn(net)(checkpoint.load(ckpt), deterministic=True)


def _saved_args(run) -> dict:
    p = Path(run) / "args.json"
    return json.loads(p.read_text()) if p.exists() else {}


def play(args) -> int:
    """JAX inference on the CPU, the CPU backend's env and its Studio viewer."""
    os.environ["JAX_PLATFORMS"] = "cpu"
    import gymnasium as gym
    import jax
    import numpy as np

    import mjsoccer  # noqa: F401  -- registers the ids

    ckpt = latest_checkpoint(args.play)
    saved = _saved_args(args.play)
    env_id, repeat = saved.get("env", args.env), saved.get("repeat_action", args.repeat_action)
    policy = load_policy(ckpt)
    act = jax.jit(lambda obs: policy(obs, jax.random.PRNGKey(0))[0])
    print(f"play {ckpt}: {env_id}, repeat_action {repeat}, backend {jax.default_backend()}",
          flush=True)
    env = gym.make(env_id, repeat_action=repeat,
                   render_mode=None if args.render == "none" else args.render)
    obs, _ = env.reset(seed=args.seed)
    episode, ret, length = 0, 0.0, 0
    while args.episodes is None or episode < args.episodes:
        obs, reward, terminated, truncated, info = env.step(
            np.asarray(act(obs.astype(np.float32)), np.float64))
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
