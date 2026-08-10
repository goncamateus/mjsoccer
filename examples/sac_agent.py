"""Train SAC (stable-baselines3) on any mjsoccer task.

    uv run --group rl python examples/sac_agent.py --env GoToBall-v0
    uv run --group rl python examples/sac_agent.py --env GoToBall-v0 --play runs/sac_GoToBall-v0.zip
"""

from __future__ import annotations

import sys

# The sibling examples/random.py shadows stdlib `random` for everything imported below,
# because sys.path[0] is this script's directory. Popping it is safe: nothing here is
# imported from examples/. Same guard as random.py itself.
sys.path.pop(0)

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402

import mjsoccer  # noqa: E402, F401  -- imported for the side effect of registering the ids


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", default="GoToBall-v0")
    ap.add_argument(
        "--tier", type=int, choices=(0, 1), default=1, help="0 = kinematic chassis, 1 = roller wheels"
    )
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="torch device: auto (default), cuda, or cpu")
    ap.add_argument("--save", default=None, help="default: runs/sac_<env>.zip")
    ap.add_argument("--play", default=None, metavar="ZIP", help="watch a saved policy instead of training")
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

    # ponytail: single env -- SAC is off-policy, so extra workers buy far less than they do
    # for PPO and cost a replay buffer per worker. Swap in make_vec_env if sampling is the wall.
    env = gym.make(args.env, tier=args.tier)
    model = SAC("MlpPolicy", env, seed=args.seed, device=args.device, verbose=1)
    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    path = args.save or f"runs/sac_{args.env}"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    print(f"saved {path}.zip")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
