"""Render one GIF per registered environment into docs/assets/, for the docs.

Each clip is scripted rather than random, so the documentation shows a robot doing the
task. Reaching the ball is the repo's own `goto_ball_policy`; the tasks that also need the
ball to go somewhere get the crude striker in `_demo_action`, which lines up on a per-env
target and kicks. Without it every clip is `GoToBall-v0` on a different scene -- the robot
arrives at the ball and sits there, and a shot or a pass never happens.

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

import gymnasium as gym
import mujoco
import numpy as np
from PIL import Image

import mjsoccer
from mjsoccer import scene as S
from mjsoccer.tasks import goto_ball_policy

# What each clip has to show beyond driving at the ball. `goto_ball_policy` only ever
# closes on the ball, so a shooting or passing task rendered with it is indistinguishable
# from GoToBall-v0 -- the robot arrives and then sits there for the rest of the clip.
# Each entry is (where the ball should end up, whether to kick it there).
DEMOS = {
    "GoToBallShoot-v0": (lambda env: np.array([S.FIELD_HALF_X, 0.0]), True),
    "StaticDefenders-v0": (lambda env: np.array([S.FIELD_HALF_X, 0.0]), True),
    "PassEndurance-v0": (lambda env: env.data.xpos[env.robots[1].body][:2].copy(), True),
    # Dribbling must carry the ball to the waypoint, so this one never fires the kicker.
    "Dribbling-v0": (lambda env: env._target.copy(), False),
}

# Tasks whose subject is a spatial relationship spread across the field -- where the wall
# of defenders sits, where the team-mate is standing -- rather than what one robot is doing
# with the ball. A shot that follows the robot crops exactly the thing these two are about,
# so they get the model's own `overview` camera and the whole 12 x 9 m field.
WIDE = {"StaticDefenders-v0", "PassEndurance-v0"}


def _demo_action(env, obs: np.ndarray, target, kick: bool) -> np.ndarray:
    """Drive at the ball, or push it at `target` and kick when lined up.

    A deliberately crude striker: approach the point one ball-radius behind the ball on the
    target line, face the target, and fire once both are true. Good enough to make a clip
    show its task, and not something to mistake for a policy -- it has no notion of the
    defenders in StaticDefenders-v0 and will happily shoot into one.
    """
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    a[-1] = 1.0  # dribbler on; a spinning bar is what holds the ball on the way in
    # goto_ball_policy and the aiming below are both tier-0 (vx, vy, omega) commands; on
    # tier 1 the drive slice is wheel speeds and neither applies.
    if target is None or env.drive_dim != 3:
        a[: min(3, env.drive_dim)] = goto_ball_policy(obs)[: min(3, env.drive_dim)]
        return a

    ball = env.ball.pos(env.data)[:2]
    x, y, th = env.robots[0].pose(env.data)
    to_target = target - ball
    behind = ball - to_target / max(float(np.linalg.norm(to_target)), 1e-9) * 0.09
    err = behind - np.array([x, y])
    c, s = np.cos(th), np.sin(th)
    # Actions are body-frame (see base._drive_robot), so rotate the world error into it.
    a[0], a[1] = np.clip(2.0 * np.array([c * err[0] + s * err[1], -s * err[0] + c * err[1]]), -1, 1)
    face = (np.arctan2(target[1] - y, target[0] - x) - th + np.pi) % (2 * np.pi) - np.pi
    a[2] = float(np.clip(1.5 * face, -1.0, 1.0))
    # Fire only with the ball actually in the mouth and squarely lined up. Kicking on
    # proximity alone punts whatever the robot happens to be facing, and a clip of a robot
    # booting the ball off the pitch documents the task worse than one that never shoots.
    if kick and env.holds(0) and abs(face) < 0.06:
        a[env.drive_dim] = 1.0  # the kick channel sits directly after the drive slice
    return a


def _aim(cam: mujoco.MjvCamera, env, alpha: float = 1.0, target=None) -> None:
    """Point a free camera at the agent and whatever it is currently working towards.

    The model's `overview` camera takes in the whole 12 x 9 m field, which puts a 180 mm
    robot at about five pixels in a GIF small enough to commit -- the clip renders a field
    and nothing legible on it. So chase the action instead. `alpha` is an exponential
    smoother: a hard cut every frame reads as jitter, since the focus jumps whenever the
    ball is struck.

    The second subject is the ball while the robot is chasing it, and the task's target
    once the robot has it. Framing robot and ball throughout looks right until the ball is
    in the mouth, at which point the two are 90 mm apart, the shot collapses to its closest
    distance, and the goal, the defenders and the team-mate all leave the frame -- the clip
    stops showing the task at exactly the moment the task starts.
    """
    robot = env.data.xpos[env.robots[0].body][:2]
    focus = target if target is not None and env.holds(0) else env.ball.pos(env.data)[:2]
    # Far enough back to hold both plus a margin, floored so a seated ball cannot fill the
    # frame, and capped because subject and target can be a field apart and framing both at
    # 12 m is the unreadable overview shot again. Past the cap the robot wins.
    want = min(7.0, max(3.0, float(np.linalg.norm(robot - focus)) * 1.25 + 1.5))
    cam.lookat[:2] += alpha * (robot * 0.55 + focus * 0.45 - cam.lookat[:2])
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
    demo,
    wide: bool,
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
    target_of, kick = demo
    aim_at = lambda: None if target_of is None else target_of(env)
    _aim(cam, env, target=aim_at())  # snap to the opening pose; smooth from there
    episode = 0
    for step in range(max_steps):
        target = aim_at()
        obs, _, terminated, truncated, _ = env.step(_demo_action(env, obs, target, kick))
        if step % every == 0:
            if not wide:
                _aim(cam, env, smoothing, target)
            renderer.update_scene(env.data, camera="overview" if wide else cam)
            frames.append(renderer.render().copy())
            if len(frames) >= n_frames:
                break
        if terminated or truncated:
            episode += 1
            obs, _ = env.reset(seed=seed + episode)
            # A reset teleports everything, so cut rather than pan.
            _aim(cam, env, target=aim_at())
    return frames


def _write_gif(raw: list[np.ndarray], path: Path, fps: int) -> None:
    # 128 colours is plenty for a green field plus two robot colours, and roughly halves
    # the file next to the default 256.
    frames = [
        Image.fromarray(f).quantize(colors=128, method=Image.Quantize.MEDIANCUT)
        for f in raw
    ]
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
        "--wide-scale",
        type=float,
        default=1.5,
        help="resolution multiplier for the whole-field clips (%s)" % ", ".join(sorted(WIDE)),
    )
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
        # a size small enough to commit. The wide clips get more pixels because a fixed
        # camera leaves most of the frame identical between frames and compresses several
        # times better -- they come out smaller at 1.5x than the following shots do at 1x,
        # and a whole-field view needs the resolution to keep a 180 mm robot legible.
        scale = args.wide_scale if env_id in WIDE else 1.0
        renderer = mujoco.Renderer(env.model, round(args.height * scale), round(args.width * scale))
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
                demo=DEMOS.get(env_id, (None, False)),
                wide=env_id in WIDE,
            )
        finally:
            renderer.close()
            env.close()

        if not frames:
            print(f"{env_id}: no frames captured, skipped")
            continue
        path = out / f"{env_id.removesuffix('-v0')}.gif"
        _write_gif(frames, path, args.fps)
        print(
            f"{env_id}: {len(frames)} frames -> {path} ({path.stat().st_size / 1024:.0f} KB)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
