"""Train SAC (stable-baselines3) on any mjsoccer task.

uv run --group rl python examples/sac_agent.py --env GoToBall-v0
uv run --group rl python examples/sac_agent.py --env GoToBall-v0 --play runs/sac_GoToBall-v0.zip
"""

from __future__ import annotations

import sys

sys.path.pop(0)

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

import mjsoccer  # noqa: F401


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", default="GoToBall-v0")
    ap.add_argument(
        "--tier",
        type=int,
        choices=(0, 1),
        default=1,
        help="0 = kinematic chassis, 1 = roller wheels",
    )
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=16, help="parallel rollout workers")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--device", default="auto", help="torch device: auto (default), cuda, or cpu"
    )
    ap.add_argument("--save", default=None, help="default: runs/sac_<env>.zip")
    ap.add_argument(
        "--play",
        default=None,
        metavar="ZIP",
        help="watch a saved policy instead of training",
    )
    args = ap.parse_args(argv)

    if args.play:
        env = gym.make(args.env, tier=args.tier, render_mode="human")
        model = SAC.load(args.play)
        obs, _ = env.reset(seed=args.seed)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(info)
                obs, _ = env.reset()

    envs = make_vec_env(args.env, n_envs=args.n_envs, seed=args.seed, env_kwargs={"tier": args.tier})
    model = SAC(
        "MlpPolicy",
        envs,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=5_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        target_update_interval=1,
        ent_coef="auto_0.2",
        seed=args.seed,
        device=args.device,
        verbose=1,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    path = args.save or f"runs/sac_{args.env}"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    print(f"saved {path}.zip")
    envs.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
