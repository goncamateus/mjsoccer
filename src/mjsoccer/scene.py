"""Compose N SSL robots and a ball into one MuJoCo model.

The MJCF in `models/` has exactly one robot in it. Rather than template a
multi-robot XML, this attaches the single-robot file N times through `mujoco.MjSpec`,
which namespaces every body, joint, actuator, sensor and geom with a per-robot prefix
(`b0_robot`, `b0_vel_x`, `b0_r_col_barDribbler`, ...).

The parent is `scene.xml` itself with its one robot **deleted**, so the `<option>` block,
the `<visual>` block, the lights, the cameras, the ball and its floor and wall contact
pairs are exactly what the model ships with -- there is no second copy of those numbers here to
drift out of sync. `MjSpec.delete` on the robot body cascades to its actuators, its
sensors and its two dribbler contact pairs.

Those dribbler pairs then get re-emitted once per robot, **copied off the originals before
the delete** rather than retyped. Skipping the step compiles fine and silently discards
the ball/bar contact tuning: MuJoCo would fall back to the elementwise max of the two
geoms' own friction, which is not what the model was tuned against.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

# ponytail: models/ sits next to the package rather than inside it, so a checkout can
# regenerate them in place. Override with MJSOCCER_MODELS to point at another build.
MODELS_DIR = Path(
    os.environ.get("MJSOCCER_MODELS", Path(__file__).resolve().parents[2] / "models")
)

SCENE = "scene.xml"
ROBOT = "robot_kinematic.xml"

# The dribbler bar's speed at full command, rad/s, standing in for its design operating point. Not readable off the model: the
# actuator's ctrlrange is deliberately ~5x this to leave room for the speed feedforward a
# real controller adds on top, so scaling the action by ctrlrange would command a bar
# speed the mechanism was never characterised at. Unsourced (ticket 08): an assumed
# ~3000 rpm, not a team firmware setpoint. Replace it when the real one is known.
DRIBBLER_OMEGA = 314.0

# Possession is read off ball/dribbler *contact*, not off the arm-deflection sensor.
# Hardware uses the arm angle, and so does the model's own dribbler controller, but on
# this build a seated ball deflects the arm the wrong way: measured -0.003 to -0.015 rad
# with the ball parked square in the mouth at (94, 0, 21) mm. Any positive threshold on
# that sensor reports "no ball" forever. Contact is unambiguous and is what the mjssl
# retention benchmarks score against too.

# Kicker and chipper take a *speed*, as the firmware does (ticket 09): a in (0, 1] asks for
# KICK_SPEED_MAX * a, the §8.4.2 cap at a = 1 (ticket 02). The plunger is still a force
# actuator against its return spring; KICK_TABLE_MS maps the asked-for speed back to a
# one-step force on KICK_TABLE_N, per plunger, the way the firmware's own
# calibration table maps speed to pulse width. Measured with a seated, settled ball, robot
# and dribbler stopped, by .scratch/physics-fidelity/candidate-09/calibrate.py -- rerun it
# whenever the plunger, the ball, the contact parameters or the timestep change.
KICK_SPEED_MAX = 6.5
KICK_TABLE_N = tuple(np.arange(0.0, 501.0, 25.0))  # the actuators' ctrlrange is 0..500
# m/s at each force, per plunger (kicker, chipper) -> a prefix of KICK_TABLE_N, cut at the first point
# past the cap: the full curves still saw a little above it. The cap takes ~285-365 N.
KICK_TABLE_MS = (
    (0.0, 0.4, 2.05, 2.96, 3.53, 3.93, 4.44, 4.79, 5.15, 5.62, 5.79, 6.12, 6.4, 6.66),
    (0.0, 0.96, 1.89, 2.75, 3.47, 4.04, 4.52, 4.96, 5.38, 5.78, 6.06, 6.4, 6.67),
)
# A 2 ms step could not resolve the plate hitting the ball: the plate closed ~9 mm per step,
# so exit speed swung +-35 % with force (ticket 09), and 1 ms still saws. The control step a
# plunger fires on is run at timestep / FIRING_SUBSTEPS, ~0.26 ms on the 1/960 s step (ticket
# 13), which the table above was measured at. Delete this if the global step gets that small.
FIRING_SUBSTEPS = 4
assert all(np.all(np.diff(v) > 0) for v in KICK_TABLE_MS), "np.interp needs it rising"

# RoboCup SSL Division A, from the rulebook. Asserted against the CAD in `_check_field`
# rather than measured out of it: the mesh gives the *painted line* extents, and which
# edge of a 20 mm line is the boundary is a rule, not a geometric fact.
FIELD_HALF_X = 6.0
FIELD_HALF_Y = 4.5
GOAL_HALF_Y = 0.9
# Div A penalty (goalkeeper) area, rSim field_type 0's: 1.8 m deep, 3.6 m wide.
PENALTY_DEPTH = 1.8
PENALTY_HALF_Y = 1.8
GOAL_DEPTH = 0.18
GOAL_HEIGHT = 0.155
# The carpet plane is at z = 0.02, not 0 (models/field.xml). Every height that means
# "above the carpet" -- the ball-z observation, the crossbar test -- subtracts it.
FLOOR_Z = 0.02
# Inner faces of the perimeter wall. A robot centre cannot get past them, so they bound
# the robot position observations. The ball can: the wall is 0.2 m tall and a chip clears it.
WALL_HALF_X = 6.3
WALL_HALF_Y = 4.8


@dataclass(frozen=True)
class RobotIds:
    """Every model index one robot needs, resolved once at build time.

    `drive` is the chassis's 3 velocity servos: vel_x, vel_y, vel_yaw.
    """

    prefix: str
    team: str  # "blue" or "yellow"
    body: int
    drive: tuple[int, ...]
    drive_range: np.ndarray  # (len(drive), 2) -- the actuators' own ctrlrange
    kick: int
    chip: int
    dribble: int
    arm_adr: int
    dribbler_body: int
    qpos_adr: int
    qvel_adr: int
    xy0: tuple[float, float]  # where the scene built the body; the slides start there

    def place(self, data: mujoco.MjData, x: float, y: float, theta: float) -> None:
        """Teleport the chassis. Assumes mj_resetData has just run, so everything else --
        dribbler arm, kicker plungers -- is already at its rest pose."""
        a = self.qpos_adr
        # Slide joints are relative to the body's built position, not the origin.
        data.qpos[a : a + 3] = (x - self.xy0[0], y - self.xy0[1], theta)
        data.qvel[self.qvel_adr : self.qvel_adr + 3] = 0.0

    def pose(self, data: mujoco.MjData) -> tuple[float, float, float]:
        """(x, y, yaw) in world coordinates, read off the compiled frame."""
        R = data.xmat[self.body].reshape(3, 3)
        p = data.xpos[self.body]
        return float(p[0]), float(p[1]), float(np.arctan2(R[1, 0], R[0, 0]))

    def velocity(self, data: mujoco.MjData) -> tuple[float, float, float]:
        """(vx, vy, omega), world frame: the chassis's world-axis slides and its yaw hinge."""
        a = self.qvel_adr
        return float(data.qvel[a]), float(data.qvel[a + 1]), float(data.qvel[a + 2])

    def forward(self, data: mujoco.MjData) -> np.ndarray:
        """Unit +x of the robot -- the direction the dribbler and kicker face."""
        return data.xmat[self.body].reshape(3, 3)[:, 0].copy()


@dataclass(frozen=True)
class BallIds:
    body: int
    geom: int
    qpos_adr: int
    qvel_adr: int
    radius: float

    def place(
        self, data: mujoco.MjData, x: float, y: float, vx: float = 0.0, vy: float = 0.0
    ) -> None:
        a, d = self.qpos_adr, self.qvel_adr
        data.qpos[a : a + 2] = (x, y)
        data.qpos[a + 3 : a + 7] = (1.0, 0.0, 0.0, 0.0)
        data.qvel[d : d + 6] = 0.0
        data.qvel[d : d + 2] = (vx, vy)

    def pos(self, data: mujoco.MjData) -> np.ndarray:
        return data.xpos[self.body].copy()

    def vel(self, data: mujoco.MjData) -> np.ndarray:
        return data.qvel[self.qvel_adr : self.qvel_adr + 3].copy()


def _robot_ids(model: mujoco.MjModel, prefix: str, team: str) -> RobotIds:
    def act(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + name)

    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "robot")
    drive = tuple(act(n) for n in ("vel_x", "vel_y", "vel_yaw"))
    assert min(drive) >= 0, f"missing drive actuators for {prefix!r}"
    jnt = model.body_jntadr[body]
    sensor = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SENSOR, prefix + "dribbler_arm_pos"
    )
    return RobotIds(
        prefix=prefix,
        team=team,
        body=body,
        drive=drive,
        drive_range=model.actuator_ctrlrange[list(drive)].copy(),
        kick=act("kicker_force"),
        chip=act("chipper_force"),
        dribble=act("dribbler_vel"),
        arm_adr=int(model.sensor_adr[sensor]),
        dribbler_body=mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, prefix + "dribbler"
        ),
        qpos_adr=int(model.jnt_qposadr[jnt]),
        qvel_adr=int(model.jnt_dofadr[jnt]),
        xy0=(float(model.body_pos[body][0]), float(model.body_pos[body][1])),
    )


@lru_cache(maxsize=8)
def build(
    n_blue: int, n_yellow: int
) -> tuple[mujoco.MjModel, tuple[RobotIds, ...], BallIds]:
    """Compile a field + n_blue + n_yellow robots + one ball.

    ponytail: the MjModel is cached and shared by every env of the same shape, which is
    safe only while nothing mutates it -- each env owns its own MjData. The first caller
    that wants domain randomisation has to drop the cache and pay the ~0.2 s compile.
    """
    if n_blue < 1:
        raise ValueError("need at least one blue robot")

    spec = mujoco.MjSpec.from_file(str(MODELS_DIR / SCENE))
    # Copy the robot's ball contact pairs before deleting the body takes them with it. Only the
    # robot's: the ball's pairs with the carpet and the walls stay, once.
    robot_geoms = {g.name for g in spec.body("robot").find_all(mujoco.mjtObj.mjOBJ_GEOM)}
    ball_pairs = [
        (
            p.geomname2,
            p.condim,
            np.array(p.friction),
            np.array(p.solref),
            np.array(p.solimp),
        )
        for p in spec.pairs
        if p.geomname1 == "ball" and p.geomname2 in robot_geoms
    ]
    assert ball_pairs, f"{SCENE} defines no ball contact pairs to replicate"
    spec.delete(spec.body("robot"))

    teams = [("blue", f"b{i}_") for i in range(n_blue)] + [
        ("yellow", f"y{i}_") for i in range(n_yellow)
    ]
    for i, (_, prefix) in enumerate(teams):
        child = mujoco.MjSpec.from_file(str(MODELS_DIR / ROBOT))
        frame = spec.worldbody.add_frame()
        # Spread the attach frames so nothing starts coincident; reset() overwrites these.
        frame.pos = [0.0, (i - (len(teams) - 1) / 2) * 0.5, 0.0]
        spec.attach(child, prefix=prefix, frame=frame)

    for _, prefix in teams:
        for geom, condim, friction, solref, solimp in ball_pairs:
            pair = spec.add_pair()
            pair.geomname1, pair.geomname2 = "ball", prefix + geom
            pair.condim = condim
            pair.friction, pair.solref, pair.solimp = friction, solref, solimp

    model = spec.compile()
    robots = tuple(_robot_ids(model, prefix, team) for team, prefix in teams)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
    ball = BallIds(
        body=body,
        geom=geom,
        qpos_adr=int(model.jnt_qposadr[model.body_jntadr[body]]),
        qvel_adr=int(model.jnt_dofadr[model.body_jntadr[body]]),
        radius=float(model.geom_size[geom][0]),
    )
    _check_field(model)
    return model, robots, ball


def _check_field(model: mujoco.MjModel) -> None:
    """The rulebook constants above must actually match this CAD.

    Cheap, and it is the one thing that poisons every reward at once: an env that believes
    the goal is at x = 6.0 on a field whose goal is elsewhere trains a policy to shoot at
    nothing, and every check downstream still passes.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    def aabb(match) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for g in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if not match(name) or model.geom_dataid[g] < 0:
                continue
            k = model.geom_dataid[g]
            v = model.mesh_vert[
                model.mesh_vertadr[k] : model.mesh_vertadr[k] + model.mesh_vertnum[k]
            ]
            w = v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            lo, hi = np.minimum(lo, w.min(0)), np.maximum(hi, w.max(0))
        assert np.isfinite(lo).all(), "no field geometry matched"
        return lo, hi

    lo, hi = aabb(lambda n: "f_lines" in n)
    assert abs(max(-lo[0], hi[0]) - FIELD_HALF_X) < 0.025, f"field half-length {hi[0]}"
    assert abs(max(-lo[1], hi[1]) - FIELD_HALF_Y) < 0.025, f"field half-width {hi[1]}"

    lo, hi = aabb(lambda n: "f_goalpost_1" in n and n.startswith("col"))
    assert abs(lo[0] - FIELD_HALF_X) < 0.01, (
        f"goal line at {lo[0]}, expected {FIELD_HALF_X}"
    )
    assert abs((hi[0] - lo[0]) - GOAL_DEPTH) < 0.03, f"goal depth {hi[0] - lo[0]}"
    # The posts sit outside the mouth, so their outer edge must bracket the 1.8 m opening.
    assert GOAL_HALF_Y < hi[1] < GOAL_HALF_Y + 0.06, f"goal mouth outer edge {hi[1]}"

    lo, hi = aabb(lambda n: n.startswith("col_f_walls"))
    # Pins the outer faces (the wall is 20 mm thick), so the inner ones sit within ~5 mm of
    # the constants -- far inside the ~90 mm a robot centre keeps off the wall.
    assert WALL_HALF_X < max(-lo[0], hi[0]) < WALL_HALF_X + 0.025, f"wall at x {hi[0]}"
    assert WALL_HALF_Y < max(-lo[1], hi[1]) < WALL_HALF_Y + 0.025, f"wall at y {hi[1]}"

    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert data.geom_xpos[floor][2] == FLOOR_Z, f"carpet at z = {data.geom_xpos[floor][2]}"


def in_field(x: float, y: float, margin: float = 0.0) -> bool:
    return abs(x) <= FIELD_HALF_X - margin and abs(y) <= FIELD_HALF_Y - margin


def in_goal(pos, sign: float = 1.0) -> bool:
    """Ball centre past the goal line, inside the mouth, under the crossbar."""
    return bool(
        sign * pos[0] > FIELD_HALF_X
        and abs(pos[1]) < GOAL_HALF_Y
        and pos[2] - FLOOR_Z < GOAL_HEIGHT
    )
