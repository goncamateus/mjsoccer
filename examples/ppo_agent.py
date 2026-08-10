"""Train PPO (stable-baselines3) on any mjsoccer task.

    uv run --group rl python examples/ppo_agent.py --env GoToBall-v0
    uv run --group rl python examples/ppo_agent.py --env GoToBall-v0 --play runs/ppo_GoToBall-v0.zip
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
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402

import mjsoccer  # noqa: E402, F401  -- imported for the side effect of registering the ids


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", default="GoToBall-v0")
    ap.add_argument(
        "--tier", type=int, choices=(0, 1), default=1, help="0 = kinematic chassis, 1 = roller wheels"
    )
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=8, help="parallel rollout workers")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--device",
        default="cpu",
        help="torch device: cpu (default, see below), cuda, or auto",
    )
    ap.add_argument("--save", default=None, help="default: runs/ppo_<env>.zip")
    ap.add_argument("--play", default=None, metavar="ZIP", help="watch a saved policy instead of training")
    args = ap.parse_args(argv)

    if args.play:
        env = gym.make(args.env, tier=args.tier, render_mode="human")
        model = PPO.load(args.play)
        obs, _ = env.reset(seed=args.seed)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(info)
                obs, _ = env.reset()

    # ponytail: no VecNormalize, no callbacks, no tensorboard dir -- SB3 defaults are the
    # baseline these tasks get compared against. Add them when a run needs tuning, not before.
    venv = make_vec_env(args.env, n_envs=args.n_envs, seed=args.seed, env_kwargs={"tier": args.tier})
    # Defaults to cpu, not auto: SB3 itself warns that an MLP-policy PPO is slower on the GPU
    # than off it -- the nets are tiny and the transfers dominate. --device cuda if you widen them.
    model = PPO("MlpPolicy", venv, seed=args.seed, device=args.device, verbose=1)
    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    path = args.save or f"runs/ppo_{args.env}"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    print(f"saved {path}.zip")
    venv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
