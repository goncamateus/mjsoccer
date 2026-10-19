"""Render one GIF per policy in a folder written by examples/sweep.sh.

    uv run --group rl python examples/visualize_sweep.py runs/sweep/sac_20261018-231040

Each <env>_t<tier>.zip gets a <env>_t<tier>.gif beside it: one deterministic episode seen
from the whole-field `overview` camera. Renders offscreen (EGL), so it runs under screen.
The algorithm comes from the folder name (sac_* / ppo_*), as sweep.sh names it.
"""

from __future__ import annotations

import os
import sys

# Same guard as ppo_agent.py: the sibling examples/random.py shadows stdlib `random`.
sys.path.pop(0)
# Must precede the mujoco import: MuJoCo picks its GL backend at import time.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import itertools  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import mujoco  # noqa: E402
from PIL import Image  # noqa: E402
from stable_baselines3 import PPO, SAC  # noqa: E402

import mjsoccer  # noqa: E402, F401  -- registers the ids

FPS = 20


def render(zip_path: Path, algo) -> str:
    env_id, tier = zip_path.stem.rsplit("_t", 1)
    env = gym.make(env_id, tier=int(tier))
    base = env.unwrapped
    model = algo.load(zip_path, device="cpu")
    every = max(1, round(1.0 / base.dt / FPS))  # keeps playback at real time
    frames, ret = [], 0.0
    with mujoco.Renderer(base.model, 540, 960) as r:
        obs, _ = env.reset(seed=0)
        for step in itertools.count():
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = env.step(action)
            ret += float(rew)
            if step % every == 0:
                r.update_scene(base.data, camera="overview")
                frames.append(Image.fromarray(r.render()).quantize(128))
            if terminated or truncated:
                break
    env.close()
    gif = zip_path.with_suffix(".gif")
    frames[0].save(
        gif, save_all=True, append_images=frames[1:], duration=round(1000 / FPS), loop=0, optimize=True
    )
    return f"{zip_path.stem}: return {ret:.2f}, {step + 1} steps, {info} -> {gif}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    args = ap.parse_args(argv)

    algo = {"sac": SAC, "ppo": PPO}[args.folder.resolve().name.split("_")[0]]
    zips = sorted(args.folder.glob("*.zip"))
    if not zips:
        print(f"no policies in {args.folder}")
        return 1
    for z in zips:
        print(render(z, algo), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
