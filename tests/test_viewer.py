"""The `render_mode="human"` viewer (ticket 06): MuJoCo Studio, fed by `render()`.

The native window needs a display, so these tests drive the `web` gfx, which renders in
the browser and needs neither; every test closes its env so no server is left listening.
"""

from __future__ import annotations

import re
import socket
import subprocess
import sys
import time

import gymnasium as gym
import pytest

import mjsoccer  # noqa: F401  (registers the ids)


def test_unknown_viewer_gfx_is_refused():
    with pytest.raises(ValueError, match="viewer_gfx"):
        gym.make("GoToBall-v0", render_mode="human", viewer_gfx="vulkan")


def _listening(port: int) -> bool:
    """Something accepts on `port` at localhost, over IPv4 or IPv6 (whichever it bound)."""
    for family, addr in ((socket.AF_INET, ("127.0.0.1", port)),
                         (socket.AF_INET6, ("::1", port, 0, 0))):
        with socket.socket(family) as s:
            s.settimeout(0.5)
            if s.connect_ex(addr) == 0:
                return True
    return False


def _wait(cond, timeout=10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


def _web_child() -> None:
    """Child of the web viewer test; one stage per line on stdin, a marker on stdout after
    each, so the parent probes the port while this process holds it."""
    def mark(m):
        print(m, flush=True)
        sys.stdin.readline()

    env = gym.make("GoToBall-v0", render_mode="human", viewer_gfx="web")
    try:
        env.reset(seed=0)
        env.step(env.action_space.sample())  # human mode renders on every step
        mark("OPEN")
        for _ in range(5):
            env.step(env.action_space.sample())
    finally:
        env.close()
    mark("CLOSED")
    env = gym.make("GoToBall-v0", render_mode="human", viewer_gfx="web")
    try:
        env.reset(seed=0)
        env.step(env.action_space.sample())
        print("SECOND OPENED", flush=True)
    except RuntimeError as e:
        print(f"SECOND REFUSED: {e}", flush=True)
    finally:
        env.close()


def test_web_viewer_serves_while_open_stops_on_close_and_opens_once_per_process():
    """In a fresh process: the viewer state is per process, and a viewer left by an
    earlier test in this one would make the first open the refused second."""
    p = subprocess.Popen([sys.executable, __file__, "web"], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

    def until(done):
        seen = ""
        for line in p.stdout:
            seen += line
            if done(seen):
                return seen
        raise AssertionError(f"child ended early: {seen!r}")

    banner = re.compile(r"http://localhost:(\d+)/")

    def go():
        p.stdin.write("\n")
        p.stdin.flush()

    try:
        # The banner comes from the server's thread, before or after the marker.
        seen = until(lambda s: "OPEN\n" in s and banner.search(s))
        port = int(banner.search(seen).group(1))
        assert _wait(lambda: _listening(port))
        go()
        until(lambda s: "CLOSED\n" in s)
        assert _wait(lambda: not _listening(port))
        go()
        # mujoco 4.0.0: a second WebViewer in one process never renders a frame and its
        # server outlives close(). Refuse it up front instead of leaving a port open.
        rest = p.stdout.read()
        assert p.wait(timeout=30) == 0
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
    assert "SECOND REFUSED: mujoco 4.0.0 supports one web viewer per process" in rest, rest
    assert "localhost" not in rest, rest


class _FakeStudio:
    """Stands in for a Studio `ViewerHandle` at the launcher boundary."""

    def __init__(self):
        self.running = True
        self.synced = 0
        self.closed = False

    def is_running(self):
        return self.running

    def sync(self, model, data):
        assert self.running, "synced a closed viewer"
        self.synced += 1
        return None, None  # what the env must ignore

    def close(self):
        self.closed = True
        self.running = False


def test_closing_the_window_mid_episode_keeps_the_episode_and_next_render_reopens(monkeypatch):
    from mjsoccer import viewer

    opened = []

    def launch(gfx):
        opened.append(_FakeStudio())
        return opened[-1]

    monkeypatch.setattr(viewer, "launch", launch)
    env = gym.make("GoToBall-v0", render_mode="human")
    env.reset(seed=0)
    env.step(env.action_space.sample())
    assert len(opened) == 1 and opened[0].synced == 1
    opened[0].running = False  # the user closes the window
    _, _, terminated, truncated, _ = env.step(env.action_space.sample())
    assert not (terminated or truncated)
    assert len(opened) == 2 and opened[1].synced == 1
    env.step(env.action_space.sample())
    assert len(opened) == 2 and opened[1].synced == 2
    env.close()
    assert opened[1].closed


def test_frame_field_sets_the_field_view_once_then_leaves_the_camera_free():
    import mujoco
    from mujoco.studio import messages, plugin, plugin_registry

    from mjsoccer import scene, viewer

    class FakeViewer:
        camera = mujoco.MjvCamera()

    model, *_ = scene.build(1, 0)
    data = mujoco.MjData(model)
    registry = plugin_registry.PluginRegistry([viewer.FrameField()])
    registry.dispatch(plugin.ViewerInitEvent(viewer=FakeViewer))
    registry.dispatch(messages.PostModelEvent(model=model, data=data))
    cam = FakeViewer.camera
    assert cam.type == mujoco.mjtCamera.mjCAMERA_FREE
    assert tuple(cam.lookat) == (0.0, 0.0, 0.0)
    assert cam.distance == pytest.approx(2.2 * scene.FIELD_HALF_X)
    assert (cam.azimuth, cam.elevation) == (90.0, -35.0)

    cam.azimuth, cam.distance = 10.0, 3.0  # the user drags the view
    registry.dispatch(messages.PostModelEvent(model=model, data=data))
    assert (cam.azimuth, cam.distance) == (10.0, 3.0)


if __name__ == "__main__" and sys.argv[1:] == ["web"]:
    _web_child()
