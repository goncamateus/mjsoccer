"""Roll out a random agent on any mjsoccer task, to sanity-check an install.

    python examples/random.py --env GoToBall-v0
"""

from __future__ import annotations

import sys

# ponytail: this file is named random.py per the documented CLI, so its own directory on
# sys.path[0] would shadow stdlib `random` for every library imported below. Popping it is
# safe because this script imports nothing from its own directory. The __future__ import
# above it is a compiler directive the grammar forces to lead, not a runtime lookup.
sys.path.pop(0)

import argparse  # noqa: E402

import gymnasium as gym  # noqa: E402

import mjsoccer  # noqa: E402, F401  -- imported for the side effect of registering the ids


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", default="GoToBall-v0")
    ap.add_argument(
        "--tier", type=int, choices=(0, 1), default=0, help="0 = kinematic chassis, 1 = roller wheels"
    )
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--view", action="store_true", help="open the MuJoCo viewer")
    ap.add_argument("--list", action="store_true", help="print the registered env ids and exit")
    args = ap.parse_args(argv)

    if args.list:
        for env_id in mjsoccer.ENV_IDS:
            print(env_id)
        return 0

    env = gym.make(args.env, tier=args.tier, render_mode="human" if args.view else None)
    print(f"{args.env} tier {args.tier}: action {env.action_space.shape}, obs {env.observation_space.shape}")
    for ep in range(args.episodes):
        # Offset per episode: the same seed every time would replay one layout forever.
        env.reset(seed=args.seed + ep)
        total, steps = 0.0, 0
        while True:
            _, reward, terminated, truncated, info = env.step(env.action_space.sample())
            total += reward
            steps += 1
            if terminated or truncated:
                print(f"ep {ep}: {steps} steps, return {total:+.2f}, {info}")
                break
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
