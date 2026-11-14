"""The `render_mode="human"` viewer: MuJoCo Studio, fed one state per `render()`.

Kept apart from `base.py` and imported only when a viewer opens, so a headless run never
loads Studio. No `from __future__ import annotations` here: Studio reads the handlers'
event annotations as real classes.
"""

import mujoco
from mujoco.studio import app, launch_passive, messages, plugin, viewer_protocol

from . import scene

# Where the camera starts, matching the `overview` camera in the MJCF: south side (+x,
# the attacking goal, on the right), both goals in frame, the field roughly filling the
# window. Distance is tied to the field so it survives a differently sized pitch.
FIELD_VIEW = dict(
    lookat=(0.0, 0.0, 0.0),
    distance=2.2 * scene.FIELD_HALF_X,
    azimuth=90.0,
    elevation=-35.0,
)


class FrameField(plugin.ViewerPlugin):
    """Open the viewer on the whole field, once.

    Studio's default free camera is placed off the model's extent statistic, which the
    walls and goal nets inflate well past the playing area -- it lands close in and the
    first thing anyone does is drag it back out. This sets `FIELD_VIEW` when the env's
    model first reaches the viewer and never again: it is a *free* camera, deliberately,
    so the mouse owns it from then on. Pinning the viewer to the fixed `overview` camera
    would take the mouse away.
    """

    done = False

    # Below INTERNAL, so it runs after Studio's App aligns its own camera on this event.
    @messages.handler(priority=messages.Priority.INTERNAL - 1)
    def _on_post_model(self, event: messages.PostModelEvent) -> None:
        if self.done:
            return
        cam = self.viewer.camera
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = FIELD_VIEW["lookat"]
        cam.distance = FIELD_VIEW["distance"]
        cam.azimuth = FIELD_VIEW["azimuth"]
        cam.elevation = FIELD_VIEW["elevation"]
        self.done = True


_web_opened = False


def launch(gfx: str):
    """Open a Studio viewer (native window or web server); return its `ViewerHandle`.

    No `StepControl` plugin: `sync()` neither steps nor paces, the env does both.

    Only one `web` viewer per process: in mujoco 4.0.0 every `WebViewer` after the first
    hangs in its headless ImGui `new_frame()` (it never draws), so its `close()` times out
    and its HTTP server keeps listening until the process exits. A browser tab has no
    close button that ends the viewer, so one per process is all `--view`/`--play` need.
    """
    global _web_opened
    if gfx == "web":
        if _web_opened:
            raise RuntimeError(
                "mujoco 4.0.0 supports one web viewer per process; a second one never "
                "renders and leaves its port open. Restart the process to view again."
            )
        _web_opened = True
    config = viewer_protocol.ViewerConfig(title="mjsoccer", width=1280, height=720, gfx=gfx)
    return launch_passive.launch(config, plugins=[app.App(), FrameField()])
