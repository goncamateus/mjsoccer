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
    ap.add_argument(
        "--control-hz",
        type=int,
        default=15,
        help="policy rate, Hz; 15 holds each action for four 60 Hz steps",
    )
    ap.add_argument(
        "--gradient-steps",
        type=int,
        default=4,
        help="updates per vector step; 4 at 15 Hz keeps the update count of 1 at 60 Hz",
    )
    ap.add_argument("--timesteps", type=int, default=250_000, help="policy steps, at --control-hz")
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
        env = gym.make(args.env, tier=args.tier, control_hz=args.control_hz, render_mode="human")
        model = SAC.load(args.play)
        obs, _ = env.reset(seed=args.seed)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(info)
                obs, _ = env.reset()

    # 15 Hz, not 60: gamma 0.99 sees 1/(1-gamma) = 100 policy steps ahead, 1.7 s at 60 Hz
    # against tasks that take 5-8 s; at 15 Hz it is 6.7 s. GoToBall-v0 tier 0, 1M physics-rate
    # steps: 0.15 eval success at 60 Hz, 0.90-1.00 over seeds 0-4 at 15 Hz
    # (.scratch/rl-learning/issues/01-why-nothing-learns.md).
    envs = make_vec_env(
        args.env,
        n_envs=args.n_envs,
        seed=args.seed,
        env_kwargs={"tier": args.tier, "control_hz": args.control_hz},
    )
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
        gradient_steps=args.gradient_steps,
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
