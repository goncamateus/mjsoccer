"""Render one GIF per registered environment into docs/assets/, for the docs.

Each clip is driven by the repo's own scripted controller (`goto_ball_policy`) with the
dribbler running, so the documentation shows a robot doing the task rather than a robot
twitching at random.

    MUJOCO_GL=egl uv run --group dev python scripts/make_gifs.py

pillow is a dev-only dependency (`[dependency-groups] dev`): it is needed to write the
GIFs and nothing in `mjsoccer` itself imports it, so it must stay out of the runtime
dependency list.
"""

from __future__ import annotations

import os

# Must precede the mujoco/gymnasium import: MuJoCo picks its GL backend at import time,
# and the default one needs a display this script will not have.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import gymnasium as gym
import mujoco

import mjsoccer
from mjsoccer.tasks import goto_ball_policy


def _aim(cam: mujoco.MjvCamera, env, alpha: float = 1.0) -> None:
    """Point a free camera at the agent and the ball, framed to hold both.

    The model's `overview` camera takes in the whole 12 x 9 m field, which puts a 180 mm
    robot at about five pixels in a GIF small enough to commit -- the clip renders a field
    and nothing legible on it. So chase the action instead. `alpha` is an exponential
    smoother: a hard cut every frame reads as jitter, since the midpoint jumps whenever the
    ball is struck.
    """
    robot = env.data.xpos[env.robots[0].body][:2]
    ball = env.ball.pos(env.data)[:2]
    # Far enough back to hold both plus a margin, never so close that a seated ball fills
    # the frame -- and capped, because the two can start a field apart and framing both at
    # 12 m is the unreadable overview shot again. Past the cap the robot wins and the ball
    # is allowed off-screen: the robot is what the clip is showing.
    want = min(6.0, max(2.5, float(np.linalg.norm(robot - ball)) * 1.3 + 1.5))
    cam.lookat[:2] += alpha * (robot * 0.6 + ball * 0.4 - cam.lookat[:2])
    cam.lookat[2] += alpha * (0.0 - cam.lookat[2])
    cam.distance += alpha * (want - cam.distance)


def _grab_frames(
    env,
    renderer: mujoco.Renderer,
    n_frames: int,
    every: int,
    seed: int,
    max_steps: int,
    smoothing: float,
) -> list[np.ndarray]:
    """Step `env` until `n_frames` frames are captured, resetting on episode end.

    A solved or truncated episode restarts with the next seed instead of ending the clip:
    a one-second GIF of a task that terminated early tells the reader nothing. `max_steps`
    bounds the whole thing so an env that terminates every step cannot spin forever.
    """
    frames: list[np.ndarray] = []
    obs, _ = env.reset(seed=seed)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation = -90.0, -40.0
    _aim(cam, env)  # snap to the opening pose; smooth from there
    episode = 0
    for step in range(max_steps):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[: env.drive_dim] = goto_ball_policy(obs)[: env.drive_dim]
        action[-1] = 1.0  # dribbler on; kick and chip (the two before it) stay at rest
        obs, _, terminated, truncated, _ = env.step(action)
        if step % every == 0:
            _aim(cam, env, smoothing)
            renderer.update_scene(env.data, camera=cam)
            frames.append(renderer.render().copy())
            if len(frames) >= n_frames:
                break
        if terminated or truncated:
            episode += 1
            obs, _ = env.reset(seed=seed + episode)
            _aim(cam, env)  # a reset teleports everything, so cut rather than pan
    return frames


def _write_gif(raw: list[np.ndarray], path: Path, fps: int) -> None:
    # 128 colours is plenty for a green field plus two robot colours, and roughly halves
    # the file next to the default 256.
    frames = [Image.fromarray(f).quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in raw]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=True,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--envs", nargs="*", default=list(mjsoccer.ENV_IDS))
    ap.add_argument("--out", default="docs/assets")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument(
        "--smoothing",
        type=float,
        default=0.12,
        help="camera follow gain per frame, 1.0 to cut instead of pan",
    )
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_frames = round(args.seconds * args.fps)

    for env_id in args.envs:
        env = gym.make(env_id, render_mode="rgb_array").unwrapped
        # base.render() hardcodes 720x1280; build our own renderer so the docs can ask for
        # a size small enough to commit.
        renderer = mujoco.Renderer(env.model, args.height, args.width)
        try:
            control_hz = env.metadata.get("render_fps") or round(1.0 / env.dt)
            every = max(1, round(control_hz / args.fps))  # keeps playback at real time
            frames = _grab_frames(
                env,
                renderer,
                n_frames,
                every,
                args.seed,
                max_steps=n_frames * every * 4,
                smoothing=args.smoothing,
            )
        finally:
            renderer.close()
            env.close()

        if not frames:
            print(f"{env_id}: no frames captured, skipped")
            continue
        path = out / f"{env_id.removesuffix('-v0')}.gif"
        _write_gif(frames, path, args.fps)
        print(f"{env_id}: {len(frames)} frames -> {path} ({path.stat().st_size / 1024:.0f} KB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
