"""Render one GIF per policy in a folder written by examples/sweep.sh or sweep_gpu.sh.

    uv run --group rl python examples/visualize_sweep.py runs/sweep/sac_20261018-231040
    uv run --group gpu python examples/visualize_sweep.py runs/sweep/mjx_20261116-120000
    uv run --group mjlab python examples/visualize_sweep.py runs/sweep/mjlab_20261116-120000

Each policy gets a <env>.gif beside it: one deterministic episode on the CPU backend, seen
from the whole-field `overview` camera. Renders offscreen (EGL), so it runs under screen.
The algorithm comes from the folder name, as the sweep scripts name it, and picks the loader
(and so the uv group to run under): sac_* / ppo_* load SB3's <env>.zip, mjx_* the latest
Brax checkpoint in <env>/ (JAX on the CPU), mjlab_* the ONNX in <env>/ (onnxruntime).
SB3 policies replay at repeat_action=1, the agents' default, so a sweep trained with
--repeat-action N plays back wrong; the GPU runs replay at the one in their args.json.
The sweep's command, flags included, is in <folder>/cmd.txt.
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
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import mjsoccer  # noqa: E402, F401  -- registers the ids

# Appended, not prepended: the ppo_mjx / ppo_mjlab loaders import from here, and stdlib
# `random` must still win over examples/random.py.
sys.path.append(str(Path(__file__).resolve().parent))

FPS = 20


def load_sb3(algo: str):
    from stable_baselines3 import PPO, SAC

    def load(zip_path: Path):
        model = {"sac": SAC, "ppo": PPO}[algo].load(zip_path, device="cpu")
        # Both agents default to repeat_action=1: a decision every 0.025 s, rSoccer's step.
        return 1, lambda obs: model.predict(obs, deterministic=True)[0]

    return load


def load_mjx(run: Path):
    os.environ["JAX_PLATFORMS"] = "cpu"  # before the first jax import
    import jax
    import ppo_mjx

    policy = ppo_mjx.load_policy(ppo_mjx.latest_checkpoint(run))
    act = jax.jit(lambda obs: policy(obs, jax.random.PRNGKey(0))[0])
    repeat = ppo_mjx._saved_args(run).get("repeat_action", 1)
    return repeat, lambda obs: np.asarray(act(obs.astype(np.float32)), np.float64)


def load_mjlab(run: Path):
    import onnxruntime as ort
    import ppo_mjlab

    sess = ort.InferenceSession(str(ppo_mjlab._onnx(run)), providers=["CPUExecutionProvider"])
    repeat = ppo_mjlab._saved_args(run).get("repeat_action", 1)
    return repeat, lambda obs: np.clip(
        sess.run(None, {"obs": obs[None].astype(np.float32)})[0][0], -1.0, 1.0).astype(np.float64)


def render(path: Path, load) -> str:
    env_id = path.stem  # <env>.zip or the run dir <env>/
    repeat, act = load(path)
    env = gym.make(env_id, repeat_action=repeat)
    base = env.unwrapped
    every = max(1, round(1.0 / base.dt / FPS))  # keeps playback at real time
    frames, ret = [], 0.0
    with mujoco.Renderer(base.model, 540, 960) as r:
        obs, _ = env.reset(seed=0)
        for step in itertools.count():
            obs, rew, terminated, truncated, info = env.step(act(obs))
            ret += float(rew)
            if step % every == 0:
                r.update_scene(base.data, camera="overview")
                # Octree, not the default median cut: median cut splits by pixel count, and the
                # ball is ~3 px in `overview`, so it lost its own palette entry and came out grey.
                frames.append(Image.fromarray(r.render()).quantize(128, method=Image.Quantize.FASTOCTREE))
            if terminated or truncated:
                break
    env.close()
    gif = path.parent / f"{env_id}.gif"
    frames[0].save(
        gif, save_all=True, append_images=frames[1:], duration=round(1000 / FPS), loop=0, optimize=True
    )
    return f"{env_id}: return {ret:.2f}, {step + 1} steps, {info} -> {gif}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    args = ap.parse_args(argv)

    algo = args.folder.resolve().name.split("_")[0]
    if algo in ("sac", "ppo"):
        load, policies = load_sb3(algo), sorted(args.folder.glob("*.zip"))
    else:
        load = {"mjx": load_mjx, "mjlab": load_mjlab}[algo]
        policies = sorted(d for d in args.folder.iterdir() if (d / "args.json").exists())
    if not policies:
        print(f"no policies in {args.folder}")
        return 1
    for p in policies:
        print(render(p, load), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
