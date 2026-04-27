"""Genesis SPH scene: a robot arm pours water from one glass into another.

Runs in the ``genesis-sim`` conda env only. Like the other Genesis variant in
this repo, this module is intentionally not imported from ``npm_sim.__init__``
because Genesis and Newton/Warp are kept in separate processes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

FRAME_RATE = 60
# The moving held glass needs a tighter SPH-rigid coupling step than the
# static cup scene because the cup is kinematically driven through the robot
# hand pose while holding SPH water.
SIM_SUBSTEPS = 84

GLASS_HEIGHT = 0.24
GLASS_BOTTOM_RADIUS = 0.072
GLASS_TOP_RADIUS = 0.098
GLASS_WALL_THICKNESS = 0.010
GLASS_BASE_THICKNESS = 0.050
GLASS_MESH_SEGMENTS = 48

GLASS_OUTER_RADIUS = 0.5 * (GLASS_BOTTOM_RADIUS + GLASS_TOP_RADIUS) + 0.004
GLASS_INNER_RADIUS = GLASS_OUTER_RADIUS - max(GLASS_WALL_THICKNESS * 2.0, 0.020)
GLASS_INNER_FLOOR_Z = -GLASS_HEIGHT * 0.5 + GLASS_BASE_THICKNESS
GLASS_RIM_Z = GLASS_HEIGHT * 0.5

WATER_PARTICLE_SIZE = 0.006
WATER_DENSITY = 1000.0
WATER_VISCOSITY = 1.0e-3
WATER_FILL_FRACTION = 0.80
WATER_BRIM_CLEARANCE = 0.006
WATER_FLOOR_CLEARANCE = 3.0 * WATER_PARTICLE_SIZE
# Genesis's regular particle sampling settles lower than the nominal emission
# cylinder. This factor is calibrated so the settled frame-0 surface is 80% of
# the inner cavity height, not just 80% of the pre-settle emission height.
EMISSION_OVERFILL_FACTOR = 1.405

PICKUP_CENTER = np.array([-0.08, -0.20, GLASS_HEIGHT * 0.5], dtype=np.float64)
POURER_CENTER = np.array([0.18, -0.20, 0.48], dtype=np.float64)
RECEIVER_SCALE = 1.0
RECEIVER_CENTER = np.array([0.29, -0.20, GLASS_HEIGHT * RECEIVER_SCALE * 0.5], dtype=np.float64)
PANDA_BASE_POS = np.array([-0.15, 0.0, 0.0], dtype=np.float64)
PANDA_BASE_EULER = np.array([0.0, 0.0, 0.0], dtype=np.float64)
PANDA_Q_PICKUP_WAYPOINTS = np.array(
    [
        [
            -0.7246374487876892,
            -0.7276675701141357,
            -2.0172975063323975,
            -2.6832516193389893,
            0.22046126425266266,
            1.6461186408996582,
            1.4305617809295654,
            0.026,
            0.026,
        ],
        [
            -0.785836935043335,
            -0.6899892687797546,
            -1.933987021446228,
            -2.725632905960083,
            0.26178979873657227,
            1.6369209289550781,
            1.4312199354171753,
            0.026,
            0.026,
        ],
        [
            -1.0057501792907715,
            -0.635722279548645,
            -1.6544466018676758,
            -2.8286755084991455,
            0.36716127395629883,
            1.603108525276184,
            1.459531545639038,
            0.026,
            0.026,
        ],
        [
            -1.3192821741104126,
            -0.7745277285575867,
            -1.2736576795578003,
            -2.9592037200927734,
            0.47090667486190796,
            1.5834200382232666,
            1.6276137828826904,
            0.026,
            0.026,
        ],
        [
            -1.475845456123352,
            -1.040704607963562,
            -0.9697046875953674,
            -3.068706750869751,
            0.6468808650970459,
            1.5945556163787842,
            1.8681795597076416,
            0.026,
            0.026,
        ],
        [
            -1.707260251045227,
            -1.1022289991378784,
            -0.4545893371105194,
            -3.0717999935150146,
            1.0619280338287354,
            1.4178987741470337,
            1.9398136138916016,
            0.026,
            0.026,
        ],
        [
            -1.66274094581604,
            -1.235813856124878,
            -0.2388017475605011,
            -3.042858839035034,
            1.3112348318099976,
            1.4581555128097534,
            2.1156721115112305,
            0.026,
            0.026,
        ],
        [
            -1.6089760065078735,
            -1.2704020738601685,
            -0.11118859052658081,
            -2.98168683052063,
            1.4556325674057007,
            1.5152837038040161,
            2.2145283222198486,
            0.026,
            0.026,
        ],
        [
            -1.5916905403137207,
            -1.2717534303665161,
            -0.06664533913135529,
            -2.951836109161377,
            1.5030548572540283,
            1.537463665008545,
            2.2464425563812256,
            0.026,
            0.026,
        ],
    ],
    dtype=np.float32,
)
PANDA_Q_UPRIGHT = PANDA_Q_PICKUP_WAYPOINTS[-1].copy()
PANDA_Q_FULL_POUR = np.array(
    [
        -1.6516658067703247,
        -0.798383355140686,
        0.6634261012077332,
        -2.0772507190704346,
        0.5259794592857361,
        1.4164435863494873,
        2.8647570610046387,
        0.026,
        0.026,
    ],
    dtype=np.float32,
)
POUR_POSE_FRACTION = 0.935
PANDA_Q_POUR = PANDA_Q_UPRIGHT + (PANDA_Q_FULL_POUR - PANDA_Q_UPRIGHT) * POUR_POSE_FRACTION
PANDA_HOME_Q = PANDA_Q_PICKUP_WAYPOINTS[0].copy()
PANDA_FINGER_OPENING = 0.026
PANDA_TCP_LOCAL_POINT = np.array([0.0, 0.0, 0.092], dtype=np.float32)
HANDLE_LOCAL_POS = np.array([-GLASS_OUTER_RADIUS - 0.060, 0.0, 0.055], dtype=np.float64)
HANDLE_SIZE = (0.120, 0.050, 0.100)
PANDA_GRASP_TARGET_LOCAL = HANDLE_LOCAL_POS.copy()

SURFACE_HOLD_SECONDS = 0.60
LIFT_SECONDS = 1.40
PRE_POUR_HOLD_SECONDS = 0.35
TILT_SECONDS = 2.00
POUR_HOLD_SECONDS = 0.00
RETURN_SECONDS = 1.10
PLACE_BACK_SECONDS = 1.60
FINAL_HOLD_SECONDS = 0.90
MAX_TILT_DEG = 82.6

VIDEO_NUM_FRAMES = int(round(
    (
        SURFACE_HOLD_SECONDS
        + LIFT_SECONDS
        + PRE_POUR_HOLD_SECONDS
        + TILT_SECONDS
        + POUR_HOLD_SECONDS
        + RETURN_SECONDS
        + PLACE_BACK_SECONDS
        + FINAL_HOLD_SECONDS
    )
    * FRAME_RATE
))
VIDEO_RESOLUTION = (1280, 720)
VIDEO_FPS = 60
CAMERA_POS = (0.95, -1.35, 0.62)
CAMERA_LOOKAT = (0.08, 0.0, 0.22)
CAMERA_FOV = 48.0
SOLID_CHECK_INTERVAL_FRAMES = 6

SETTLED_PARTICLES_CACHE = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "_genesis"
    / "robotic_arm_pickup_pour_base050_p006_fill080_over1405_clear018_settled_water.npy"
)
GLASS_MESH_PATH = Path(__file__).resolve().parents[2] / "outputs" / "_genesis" / "pouring_glass.obj"
SETTLE_BAKE_SECONDS = 0.8
STASHED_PARTICLE_POS = np.array([10.0, 10.0, -10.0], dtype=np.float32)


@dataclass(frozen=True)
class SimulationResult:
    initial_particle_positions: np.ndarray
    final_particle_positions: np.ndarray
    final_pourer_position: np.ndarray
    final_pourer_quat_wxyz: np.ndarray
    final_tilt_degrees: float
    max_tilt_degrees: float
    initial_particle_count: int
    final_particles_in_pourer: int
    final_particles_in_receiver: int
    final_live_particles: int
    max_glass_solid_particles: int
    max_pourer_solid_particles: int
    max_receiver_solid_particles: int
    max_pourer_base_particles: int

    @property
    def receiver_fraction(self) -> float:
        return self.final_particles_in_receiver / max(1, self.initial_particle_count)

    @property
    def pourer_fraction(self) -> float:
        return self.final_particles_in_pourer / max(1, self.initial_particle_count)


def _smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _quat_wxyz_from_axis_angle(axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    half = math.radians(degrees) * 0.5
    s = math.sin(half)
    return np.array([math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)


def _quat_to_matrix_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quat_wxyz_from_matrix(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * s,
                (m[2, 1] - m[1, 2]) / s,
                (m[0, 2] - m[2, 0]) / s,
                (m[1, 0] - m[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1.0e-12)) * 2.0
        quat = np.array(
            [
                (m[2, 1] - m[1, 2]) / s,
                0.25 * s,
                (m[0, 1] + m[1, 0]) / s,
                (m[0, 2] + m[2, 0]) / s,
            ],
            dtype=np.float64,
        )
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1.0e-12)) * 2.0
        quat = np.array(
            [
                (m[0, 2] - m[2, 0]) / s,
                (m[0, 1] + m[1, 0]) / s,
                0.25 * s,
                (m[1, 2] + m[2, 1]) / s,
            ],
            dtype=np.float64,
        )
    else:
        s = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1.0e-12)) * 2.0
        quat = np.array(
            [
                (m[1, 0] - m[0, 1]) / s,
                (m[0, 2] + m[2, 0]) / s,
                (m[1, 2] + m[2, 1]) / s,
                0.25 * s,
            ],
            dtype=np.float64,
        )
    return quat / np.linalg.norm(quat)


def _quat_multiply_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = [float(v) for v in a]
    bw, bx, by, bz = [float(v) for v in b]
    quat = np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )
    return quat / np.linalg.norm(quat)


def _quat_inverse_wxyz(q: np.ndarray) -> np.ndarray:
    quat = np.asarray(q, dtype=np.float64)
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64) / np.dot(quat, quat)


def _angular_velocity_between_wxyz(q0: np.ndarray, q1: np.ndarray, dt: float) -> np.ndarray:
    delta = _quat_multiply_wxyz(np.asarray(q1, dtype=np.float64), _quat_inverse_wxyz(np.asarray(q0, dtype=np.float64)))
    if delta[0] < 0.0:
        delta = -delta
    vector_norm = float(np.linalg.norm(delta[1:]))
    if vector_norm < 1.0e-9:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(vector_norm, float(delta[0]))
    return delta[1:] / vector_norm * (angle / dt)


def _transform_local(pos: np.ndarray, quat_wxyz: np.ndarray, local: np.ndarray) -> np.ndarray:
    return np.asarray(pos, dtype=np.float64) + _quat_to_matrix_wxyz(quat_wxyz) @ np.asarray(local, dtype=np.float64)


def _inverse_transform_points(pos: np.ndarray, quat_wxyz: np.ndarray, points: np.ndarray) -> np.ndarray:
    rotation = _quat_to_matrix_wxyz(quat_wxyz)
    return (np.asarray(points, dtype=np.float64) - np.asarray(pos, dtype=np.float64)) @ rotation


def tilt_degrees_at(time_seconds: float) -> float:
    return MAX_TILT_DEG * pour_motion_fraction_at(time_seconds)


def pour_motion_fraction_at(time_seconds: float) -> float:
    t = float(time_seconds)
    t -= SURFACE_HOLD_SECONDS + LIFT_SECONDS + PRE_POUR_HOLD_SECONDS
    if t < 0.0:
        return 0.0
    if t < TILT_SECONDS:
        return _smoothstep(t / TILT_SECONDS)
    t -= TILT_SECONDS
    if t < POUR_HOLD_SECONDS:
        return 1.0
    t -= POUR_HOLD_SECONDS
    if t < RETURN_SECONDS:
        return 1.0 - _smoothstep(t / RETURN_SECONDS)
    return 0.0


def lift_motion_fraction_at(time_seconds: float) -> float:
    t = float(time_seconds) - SURFACE_HOLD_SECONDS
    if t <= 0.0:
        return 0.0
    if t >= LIFT_SECONDS:
        return 1.0
    return t / LIFT_SECONDS


def _interpolate_q_waypoints(waypoints: np.ndarray, fraction: float) -> np.ndarray:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    if fraction <= 0.0:
        return waypoints[0].copy()
    if fraction >= 1.0:
        return waypoints[-1].copy()
    scaled = fraction * (len(waypoints) - 1)
    segment = min(int(scaled), len(waypoints) - 2)
    local = _smoothstep(scaled - segment)
    return waypoints[segment] + (waypoints[segment + 1] - waypoints[segment]) * local


def standard_robot_q_at(time_seconds: float) -> np.ndarray:
    lift_done_time = SURFACE_HOLD_SECONDS + LIFT_SECONDS
    pour_done_time = lift_done_time + PRE_POUR_HOLD_SECONDS + TILT_SECONDS + POUR_HOLD_SECONDS + RETURN_SECONDS

    if time_seconds < lift_done_time:
        q = _interpolate_q_waypoints(PANDA_Q_PICKUP_WAYPOINTS, lift_motion_fraction_at(time_seconds))
    elif time_seconds < pour_done_time:
        fraction = pour_motion_fraction_at(time_seconds)
        q = PANDA_Q_UPRIGHT + (PANDA_Q_POUR - PANDA_Q_UPRIGHT) * fraction
    elif time_seconds < pour_done_time + PLACE_BACK_SECONDS:
        fraction = (time_seconds - pour_done_time) / PLACE_BACK_SECONDS
        q = _interpolate_q_waypoints(PANDA_Q_PICKUP_WAYPOINTS[::-1], fraction)
    else:
        q = PANDA_Q_PICKUP_WAYPOINTS[0].copy()
    q = q.astype(np.float32, copy=True)
    q[7:] = PANDA_FINGER_OPENING
    return q


def initial_glass_pose() -> tuple[np.ndarray, np.ndarray]:
    return PICKUP_CENTER.copy(), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _panda_asset_path(gs) -> Path:
    return Path(gs.__file__).resolve().parent / "assets" / "xml" / "franka_emika_panda" / "panda.xml"


def _cup_pose_from_grasp_tcp(tcp_pos: np.ndarray, hand_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hand_rotation = _quat_to_matrix_wxyz(hand_quat)
    cup_to_hand = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    cup_rotation = hand_rotation @ cup_to_hand.T
    cup_quat = _quat_wxyz_from_matrix(cup_rotation)
    cup_pos = np.asarray(tcp_pos, dtype=np.float64) - cup_rotation @ PANDA_GRASP_TARGET_LOCAL
    return cup_pos, cup_quat


def _glass_inner_mask(points: np.ndarray, center: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    return _glass_inner_mask_scaled(points, center, quat_wxyz, scale=1.0)


def _glass_inner_mask_scaled(
    points: np.ndarray,
    center: np.ndarray,
    quat_wxyz: np.ndarray,
    *,
    scale: float,
) -> np.ndarray:
    local = _inverse_transform_points(center, quat_wxyz, points) / scale
    r_xy = np.linalg.norm(local[:, :2], axis=1)
    return (
        (r_xy < GLASS_INNER_RADIUS + WATER_PARTICLE_SIZE * 0.75)
        & (local[:, 2] >= GLASS_INNER_FLOOR_Z - 0.001)
        & (local[:, 2] <= GLASS_RIM_Z - WATER_BRIM_CLEARANCE)
    )


def _glass_solid_mask(
    points: np.ndarray,
    center: np.ndarray,
    quat_wxyz: np.ndarray,
    *,
    tolerance: float | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    if tolerance is None:
        tolerance = WATER_PARTICLE_SIZE * 1.25

    local = _inverse_transform_points(center, quat_wxyz, points) / scale
    r_xy = np.linalg.norm(local[:, :2], axis=1)
    base_z = -GLASS_HEIGHT * 0.5
    floor_z = GLASS_INNER_FLOOR_Z
    rim_z = GLASS_RIM_Z

    side_wall = (
        (r_xy > GLASS_INNER_RADIUS + tolerance)
        & (r_xy < GLASS_OUTER_RADIUS - tolerance)
        & (local[:, 2] > floor_z + tolerance)
        & (local[:, 2] < rim_z - tolerance)
    )
    base = (
        (r_xy < GLASS_OUTER_RADIUS - tolerance)
        & (local[:, 2] > base_z + tolerance)
        & (local[:, 2] < floor_z - tolerance)
    )
    return side_wall | base


def _glass_base_solid_mask(
    points: np.ndarray,
    center: np.ndarray,
    quat_wxyz: np.ndarray,
    *,
    tolerance: float | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    if tolerance is None:
        tolerance = WATER_PARTICLE_SIZE

    local = _inverse_transform_points(center, quat_wxyz, points) / scale
    r_xy = np.linalg.norm(local[:, :2], axis=1)
    base_z = -GLASS_HEIGHT * 0.5
    return (
        (r_xy < GLASS_OUTER_RADIUS - tolerance)
        & (local[:, 2] > base_z + tolerance)
        & (local[:, 2] < GLASS_INNER_FLOOR_Z - tolerance)
    )


def _glass_overlap_sample_points() -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, GLASS_MESH_SEGMENTS, endpoint=False)
    zs = np.linspace(-0.5 * GLASS_HEIGHT, 0.5 * GLASS_HEIGHT, 15)
    side = np.array(
        [
            [GLASS_OUTER_RADIUS * math.cos(angle), GLASS_OUTER_RADIUS * math.sin(angle), z]
            for z in zs
            for angle in angles
        ],
        dtype=np.float64,
    )
    rims = np.array(
        [
            [radius * math.cos(angle), radius * math.sin(angle), z]
            for z in (-0.5 * GLASS_HEIGHT, 0.5 * GLASS_HEIGHT)
            for radius in (GLASS_INNER_RADIUS, GLASS_OUTER_RADIUS)
            for angle in angles
        ],
        dtype=np.float64,
    )
    return np.concatenate([side, rims], axis=0)


def _glass_outer_volume_mask_scaled(
    points: np.ndarray,
    center: np.ndarray,
    quat_wxyz: np.ndarray,
    *,
    scale: float,
    tolerance: float = 0.006,
) -> np.ndarray:
    local = _inverse_transform_points(center, quat_wxyz, points) / scale
    r_xy = np.linalg.norm(local[:, :2], axis=1)
    return (
        (r_xy < GLASS_OUTER_RADIUS + tolerance)
        & (local[:, 2] > -0.5 * GLASS_HEIGHT - tolerance)
        & (local[:, 2] < 0.5 * GLASS_HEIGHT + tolerance)
    )


def glass_overlap_sample_count(
    cup_pos: np.ndarray,
    cup_quat: np.ndarray,
    receiver_pos: np.ndarray = RECEIVER_CENTER,
    receiver_quat: np.ndarray | None = None,
    *,
    receiver_scale: float = RECEIVER_SCALE,
) -> int:
    if receiver_quat is None:
        receiver_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    local_points = _glass_overlap_sample_points()
    cup_points = cup_pos + local_points @ _quat_to_matrix_wxyz(cup_quat).T
    receiver_points = receiver_pos + receiver_scale * (local_points @ _quat_to_matrix_wxyz(receiver_quat).T)
    cup_in_receiver = _glass_outer_volume_mask_scaled(
        cup_points,
        receiver_pos,
        receiver_quat,
        scale=receiver_scale,
    )
    receiver_in_cup = _glass_outer_volume_mask_scaled(
        receiver_points,
        cup_pos,
        cup_quat,
        scale=1.0,
    )
    return int(cup_in_receiver.sum() + receiver_in_cup.sum())


def build_glass_mesh(path: Optional[Path] = None) -> Path:
    """Write one watertight open-top glass mesh for Genesis SDF coupling."""
    import trimesh

    if path is None:
        path = GLASS_MESH_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    n = GLASS_MESH_SEGMENTS
    half_h = GLASS_HEIGHT * 0.5
    base_z = -half_h
    rim_z = half_h
    inner_base_z = base_z + GLASS_BASE_THICKNESS
    outer_r = GLASS_OUTER_RADIUS
    inner_r = GLASS_INNER_RADIUS

    def ring(radius: float, z: float) -> np.ndarray:
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        return np.stack([radius * np.cos(angles), radius * np.sin(angles), np.full(n, z)], axis=1)

    outer_bot = ring(outer_r, base_z)
    outer_top = ring(outer_r, rim_z)
    inner_bot = ring(inner_r, inner_base_z)
    inner_top = ring(inner_r, rim_z)
    outer_base = ring(outer_r, base_z)
    inner_floor = ring(inner_r, inner_base_z)

    verts = np.concatenate([outer_bot, outer_top, inner_bot, inner_top, outer_base, inner_floor], axis=0)
    off_ob, off_ot, off_ib, off_it, off_base, off_floor = 0, n, 2 * n, 3 * n, 4 * n, 5 * n
    base_center_idx = verts.shape[0]
    verts = np.concatenate([verts, np.array([[0.0, 0.0, base_z]])], axis=0)
    floor_center_idx = verts.shape[0]
    verts = np.concatenate([verts, np.array([[0.0, 0.0, inner_base_z]])], axis=0)

    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([off_ob + i, off_ob + j, off_ot + j])
        faces.append([off_ob + i, off_ot + j, off_ot + i])
        faces.append([off_ib + i, off_it + j, off_ib + j])
        faces.append([off_ib + i, off_it + i, off_it + j])
        faces.append([off_ot + i, off_it + j, off_it + i])
        faces.append([off_ot + i, off_ot + j, off_it + j])
        faces.append([base_center_idx, off_base + j, off_base + i])
        faces.append([floor_center_idx, off_floor + i, off_floor + j])

    mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces), process=True)
    mesh.fix_normals()
    mesh.export(path)
    return path


class RoboticArmPourGenesisDemo:
    def __init__(
        self,
        *,
        num_frames: int,
        show_viewer: bool = False,
        enable_camera: bool = False,
    ):
        import genesis as gs

        try:
            gs.init(backend=gs.gpu, logging_level="warning")
        except gs.GenesisException as exc:
            if "already initialized" not in str(exc).lower():
                raise

        self.gs = gs
        self.num_frames = num_frames
        self.frame_dt = 1.0 / FRAME_RATE
        self.sim_time = 0.0
        self.frame_index = 0
        self.max_tilt_degrees = 0.0
        self.max_glass_solid_particles = 0
        self.max_pourer_solid_particles = 0
        self.max_receiver_solid_particles = 0
        self.max_pourer_base_particles = 0
        self.standard_robot = None
        self.standard_robot_hand = None
        self._standard_robot_last_q = PANDA_HOME_Q.copy()
        cup_pos, cup_quat = initial_glass_pose()
        self.current_cup_pos = cup_pos
        self.current_cup_quat = cup_quat
        self.current_cup_tilt = 0.0

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.frame_dt,
                substeps=SIM_SUBSTEPS,
                gravity=(0.0, 0.0, -9.81),
            ),
            sph_options=gs.options.SPHOptions(
                particle_size=WATER_PARTICLE_SIZE,
                pressure_solver="DFSPH",
                lower_bound=(-0.85, -0.85, -0.2),
                upper_bound=(1.05, 0.85, 1.15),
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.frame_dt,
                gravity=(0.0, 0.0, -9.81),
            ),
            show_viewer=show_viewer,
        )

        self._add_floor()
        self._add_robot_visuals()
        self._add_glasses()
        self._add_water()

        self.camera = None
        if enable_camera:
            self.camera = self.scene.add_camera(
                res=VIDEO_RESOLUTION,
                pos=CAMERA_POS,
                lookat=CAMERA_LOOKAT,
                fov=CAMERA_FOV,
                GUI=False,
            )

        self.scene.build()
        self._finish_robot_visuals()
        self._apply_kinematic_pose(0.0)

        self.initial_cup_pos = self.current_cup_pos.copy()
        self.initial_cup_quat = self.current_cup_quat.copy()
        self.initial_particles = self._particle_positions()

    def _glass_material(self, *, rho: float = 650.0, gravity_compensation: float = 0.0):
        gs = self.gs
        return gs.materials.Rigid(
            rho=rho,
            coup_softness=0.0,
            coup_friction=1.0,
            coup_restitution=0.0,
            sdf_cell_size=0.003,
            sdf_min_res=48,
            sdf_max_res=192,
            gravity_compensation=gravity_compensation,
        )

    def _add_floor(self) -> None:
        gs = self.gs
        self.floor = self.scene.add_entity(
            material=gs.materials.Rigid(),
            morph=gs.morphs.Box(
                pos=(0.08, 0.0, -0.025),
                size=(1.45, 1.15, 0.05),
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=(0.55, 0.58, 0.56), roughness=0.8),
        )

    def _add_glasses(self) -> None:
        gs = self.gs
        mesh_path = build_glass_mesh()
        glass_surface = gs.surfaces.Default(
            color=(0.76, 0.92, 1.0),
            opacity=0.30,
            roughness=0.08,
        )

        pos, quat = initial_glass_pose()
        self.pouring_glass = self.scene.add_entity(
            material=self._glass_material(rho=50000.0, gravity_compensation=1.0),
            morph=gs.morphs.Mesh(
                file=str(mesh_path),
                pos=tuple(pos),
                quat=tuple(quat),
                fixed=False,
                decimate=False,
                convexify=False,
            ),
            surface=glass_surface,
        )
        handle_pos = _transform_local(pos, quat, HANDLE_LOCAL_POS)
        self.pouring_glass_handle = self.scene.add_entity(
            material=gs.materials.Rigid(needs_coup=False),
            morph=gs.morphs.Box(
                pos=tuple(handle_pos),
                quat=tuple(quat),
                size=HANDLE_SIZE,
                fixed=False,
                collision=False,
            ),
            surface=gs.surfaces.Default(color=(0.04, 0.05, 0.06), roughness=0.45),
        )
        self.receiving_glass = self.scene.add_entity(
            material=self._glass_material(),
            morph=gs.morphs.Mesh(
                file=str(mesh_path),
                pos=tuple(RECEIVER_CENTER),
                scale=RECEIVER_SCALE,
                fixed=True,
                batch_fixed_verts=True,
                decimate=False,
                convexify=False,
            ),
            surface=glass_surface,
        )

    def _add_robot_visuals(self) -> None:
        gs = self.gs
        self.standard_robot = self.scene.add_entity(
            material=gs.materials.Rigid(needs_coup=False),
            morph=gs.morphs.MJCF(
                file=str(_panda_asset_path(gs)),
                pos=tuple(PANDA_BASE_POS),
                euler=tuple(PANDA_BASE_EULER),
                collision=False,
            ),
        )

    def _finish_robot_visuals(self) -> None:
        self.standard_robot_hand = self.standard_robot.get_link(name="hand")
        self.standard_robot.set_dofs_position(PANDA_HOME_Q.copy(), zero_velocity=True)
        self._standard_robot_last_q = PANDA_HOME_Q.copy()

    def _add_water(self) -> None:
        gs = self.gs
        fill_bottom_local_z = GLASS_INNER_FLOOR_Z + WATER_FLOOR_CLEARANCE
        fill_height = (
            WATER_FILL_FRACTION
            * (GLASS_HEIGHT - GLASS_BASE_THICKNESS - WATER_BRIM_CLEARANCE - WATER_FLOOR_CLEARANCE)
        )
        target_volume = math.pi * (GLASS_INNER_RADIUS - WATER_PARTICLE_SIZE) ** 2 * fill_height
        rest_volume_per_particle = 0.8 * WATER_PARTICLE_SIZE ** 3
        target_n_particles = target_volume / rest_volume_per_particle
        emission_volume = target_n_particles * EMISSION_OVERFILL_FACTOR * WATER_PARTICLE_SIZE ** 3
        cylinder_radius = max(GLASS_INNER_RADIUS - WATER_PARTICLE_SIZE, WATER_PARTICLE_SIZE)
        emission_height = emission_volume / (math.pi * cylinder_radius ** 2)
        cylinder_center_local_z = fill_bottom_local_z + emission_height * 0.5
        pickup_pos, _ = initial_glass_pose()
        water_center = pickup_pos + np.array([0.0, 0.0, cylinder_center_local_z], dtype=np.float64)

        self.water = self.scene.add_entity(
            material=gs.materials.SPH.Liquid(rho=WATER_DENSITY, mu=WATER_VISCOSITY),
            morph=gs.morphs.Cylinder(
                pos=tuple(water_center),
                radius=cylinder_radius,
                height=emission_height,
            ),
            surface=gs.surfaces.Default(color=(0.25, 0.55, 0.95, 1.0), vis_mode="particle"),
        )

    def _set_pouring_glass_pose(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        linear_velocity: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> None:
        self.pouring_glass.set_pos(pos.astype(np.float32), zero_velocity=False, relative=False, skip_forward=True)
        self.pouring_glass.set_quat(quat.astype(np.float32), zero_velocity=False, relative=False)
        self.pouring_glass.set_dofs_velocity(
            np.concatenate([linear_velocity, angular_velocity]).astype(np.float32),
            skip_forward=False,
        )

    def _set_handle_pose(self, cup_pos: np.ndarray, cup_quat: np.ndarray) -> None:
        handle_pos = _transform_local(cup_pos, cup_quat, HANDLE_LOCAL_POS)
        self.pouring_glass_handle.set_pos(handle_pos.astype(np.float32), zero_velocity=True, relative=False, skip_forward=True)
        self.pouring_glass_handle.set_quat(cup_quat.astype(np.float32), zero_velocity=True, relative=False)

    def _actual_standard_robot_grasp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        hand_pos = self.standard_robot.get_links_pos([self.standard_robot_hand.idx_local]).cpu().numpy()[0]
        hand_quat = self.standard_robot.get_links_quat([self.standard_robot_hand.idx_local]).cpu().numpy()[0]
        tcp_pos = _transform_local(hand_pos, hand_quat, PANDA_TCP_LOCAL_POINT)
        return tcp_pos, hand_quat

    def _set_standard_robot_pose(
        self,
        time_seconds: float,
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        q_current = standard_robot_q_at(time_seconds)
        q_next = standard_robot_q_at(time_seconds + self.frame_dt)

        self.standard_robot.set_dofs_position(q_current, zero_velocity=False)
        current_grasp_pose = self._actual_standard_robot_grasp_pose()
        self.standard_robot.set_dofs_position(q_next, zero_velocity=False)
        next_grasp_pose = self._actual_standard_robot_grasp_pose()
        self.standard_robot.set_dofs_position(q_current, zero_velocity=False)
        self.standard_robot.set_dofs_velocity((q_next - q_current) / self.frame_dt)
        self._standard_robot_last_q = q_current.copy()
        return current_grasp_pose, next_grasp_pose

    def _apply_kinematic_pose(self, time_seconds: float) -> None:
        (tcp_pos, hand_quat), (next_tcp_pos, next_hand_quat) = self._set_standard_robot_pose(time_seconds)
        cup_pos, cup_quat = _cup_pose_from_grasp_tcp(tcp_pos, hand_quat)
        next_cup_pos, next_cup_quat = _cup_pose_from_grasp_tcp(next_tcp_pos, next_hand_quat)
        # The glass pose is the fixed handle grasp transform from the actual
        # Panda hand FK. The first phase holds that pose on the table, then the
        # lift and pour phases move it only through robot joint commands.
        # Genesis's attach() path made SPH leak through the held mesh, so this
        # keeps the working rigid-SPH coupling while preserving the robot as
        # the source of motion.
        self._set_pouring_glass_pose(
            cup_pos,
            cup_quat,
            (next_cup_pos - cup_pos) / self.frame_dt,
            _angular_velocity_between_wxyz(cup_quat, next_cup_quat, self.frame_dt),
        )
        self._set_handle_pose(cup_pos, cup_quat)
        rotation = _quat_to_matrix_wxyz(cup_quat)
        cup_axis_z = float(np.clip(rotation[2, 2], -1.0, 1.0))
        actual_tilt = math.degrees(math.acos(cup_axis_z))
        self.max_tilt_degrees = max(self.max_tilt_degrees, abs(actual_tilt))
        self.current_cup_pos = cup_pos.copy()
        self.current_cup_quat = cup_quat.copy()
        self.current_cup_tilt = actual_tilt

    def _particle_positions(self) -> np.ndarray:
        return self.water.get_particles_pos().cpu().numpy().copy()

    def _count_glass_solid_particles_by_glass(self, positions: np.ndarray, time_seconds: float) -> tuple[int, int]:
        del time_seconds
        cup_pos = self.current_cup_pos
        cup_quat = self.current_cup_quat
        receiver_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        live = positions[positions[:, 2] > -1.0]
        if live.size:
            pourer_count = int(_glass_solid_mask(live, cup_pos, cup_quat).sum())
            receiver_count = int(_glass_solid_mask(live, RECEIVER_CENTER, receiver_quat, scale=RECEIVER_SCALE).sum())
            return pourer_count, receiver_count
        return 0, 0

    def _count_glass_solid_particles(self, positions: np.ndarray, time_seconds: float) -> int:
        pourer_count, receiver_count = self._count_glass_solid_particles_by_glass(positions, time_seconds)
        return pourer_count + receiver_count

    def _count_pourer_base_particles(self, positions: np.ndarray) -> int:
        live = positions[positions[:, 2] > -1.0]
        if not live.size:
            return 0
        return int(_glass_base_solid_mask(live, self.current_cup_pos, self.current_cup_quat).sum())

    def step(self) -> None:
        self._apply_kinematic_pose(self.sim_time)
        self.scene.step()
        next_time = self.sim_time + self.frame_dt
        self._apply_kinematic_pose(next_time)
        self.sim_time = next_time
        self.frame_index += 1
        positions = self._particle_positions()
        self.max_pourer_base_particles = max(
            self.max_pourer_base_particles,
            self._count_pourer_base_particles(positions),
        )
        if self.frame_index % SOLID_CHECK_INTERVAL_FRAMES == 0:
            pourer_count, receiver_count = self._count_glass_solid_particles_by_glass(
                positions,
                self.sim_time,
            )
            self.max_pourer_solid_particles = max(self.max_pourer_solid_particles, pourer_count)
            self.max_receiver_solid_particles = max(self.max_receiver_solid_particles, receiver_count)
            self.max_glass_solid_particles = max(
                self.max_glass_solid_particles,
                pourer_count + receiver_count,
            )

    def pre_settle(self, seconds: float = SETTLE_BAKE_SECONDS) -> None:
        for _ in range(int(round(seconds * FRAME_RATE))):
            self._apply_kinematic_pose(0.0)
            self.scene.step()
        self._apply_kinematic_pose(0.0)

    def trim_overflow_particles(self) -> None:
        positions = self._particle_positions()
        in_pourer = _glass_inner_mask(positions, self.current_cup_pos, self.current_cup_quat)
        if int((~in_pourer).sum()) > 0:
            positions[~in_pourer] = STASHED_PARTICLE_POS.astype(positions.dtype)
            self.water.set_particles_pos(positions.astype(np.float32))
            self.water.set_particles_vel(np.zeros_like(positions, dtype=np.float32))

    def load_settled_particles(self, path: Path) -> bool:
        if not path.exists():
            return False
        settled = np.load(path).astype(np.float32)
        if settled.shape[0] != self.water.n_particles:
            return False
        in_pourer = _glass_inner_mask(settled, self.current_cup_pos, self.current_cup_quat)
        if int(in_pourer.sum()) < 0.5 * settled.shape[0]:
            return False
        self.water.set_particles_pos(settled)
        self.water.set_particles_vel(np.zeros_like(settled))
        self.initial_particles = settled.copy()
        return True

    def particle_counts(
        self,
        particles: np.ndarray,
        *,
        cup_pos: np.ndarray | None = None,
        cup_quat: np.ndarray | None = None,
    ) -> tuple[int, int, int]:
        if cup_pos is None:
            cup_pos = self.current_cup_pos
        if cup_quat is None:
            cup_quat = self.current_cup_quat
        live = particles[:, 2] > -1.0
        in_pourer = _glass_inner_mask(particles, cup_pos, cup_quat)
        in_receiver = _glass_inner_mask_scaled(
            particles,
            RECEIVER_CENTER,
            np.array([1.0, 0.0, 0.0, 0.0]),
            scale=RECEIVER_SCALE,
        )
        return int(in_pourer.sum()), int(in_receiver.sum()), int(live.sum())

    def run(self) -> SimulationResult:
        for _ in range(self.num_frames):
            self.step()
        return self.result()

    def result(self) -> SimulationResult:
        final_particles = self._particle_positions()
        in_pourer, in_receiver, live = self.particle_counts(final_particles)
        initial_in_pourer, _, _ = self.particle_counts(
            self.initial_particles,
            cup_pos=self.initial_cup_pos,
            cup_quat=self.initial_cup_quat,
        )
        final_pos = self.current_cup_pos.copy()
        final_quat = self.current_cup_quat.copy()
        final_tilt = self.current_cup_tilt
        final_pourer_solid_count, final_receiver_solid_count = self._count_glass_solid_particles_by_glass(
            final_particles,
            self.sim_time,
        )
        final_solid_count = final_pourer_solid_count + final_receiver_solid_count
        final_pourer_base_count = self._count_pourer_base_particles(final_particles)
        return SimulationResult(
            initial_particle_positions=self.initial_particles.copy(),
            final_particle_positions=final_particles,
            final_pourer_position=final_pos,
            final_pourer_quat_wxyz=final_quat,
            final_tilt_degrees=float(final_tilt),
            max_tilt_degrees=float(self.max_tilt_degrees),
            initial_particle_count=initial_in_pourer,
            final_particles_in_pourer=in_pourer,
            final_particles_in_receiver=in_receiver,
            final_live_particles=live,
            max_glass_solid_particles=int(max(self.max_glass_solid_particles, final_solid_count)),
            max_pourer_solid_particles=int(max(self.max_pourer_solid_particles, final_pourer_solid_count)),
            max_receiver_solid_particles=int(max(self.max_receiver_solid_particles, final_receiver_solid_count)),
            max_pourer_base_particles=int(max(self.max_pourer_base_particles, final_pourer_base_count)),
        )


def bake_settled_particles(
    *,
    cache_path: Path = SETTLED_PARTICLES_CACHE,
    settle_seconds: float = SETTLE_BAKE_SECONDS,
) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    demo = RoboticArmPourGenesisDemo(num_frames=0, show_viewer=False, enable_camera=False)
    demo.pre_settle(settle_seconds)
    demo.trim_overflow_particles()
    for _ in range(int(round(0.2 * FRAME_RATE))):
        demo.step()
    demo.trim_overflow_particles()
    settled = demo._particle_positions().astype(np.float32)
    np.save(cache_path, settled)
    return cache_path


def run_simulation(
    *,
    num_frames: int = VIDEO_NUM_FRAMES,
    show_viewer: bool = False,
    settled_cache: Path = SETTLED_PARTICLES_CACHE,
    rebake: bool = False,
) -> SimulationResult:
    if rebake or not settled_cache.exists():
        bake_settled_particles(cache_path=settled_cache)

    demo = RoboticArmPourGenesisDemo(num_frames=num_frames, show_viewer=show_viewer, enable_camera=False)
    if not demo.load_settled_particles(settled_cache):
        bake_settled_particles(cache_path=settled_cache)
        demo.load_settled_particles(settled_cache)
    return demo.run()


def render_video(
    *,
    output_path: str = "outputs/robotic_arm_pour_genesis.mp4",
    num_frames: int = VIDEO_NUM_FRAMES,
    settled_cache: Path = SETTLED_PARTICLES_CACHE,
    rebake: bool = False,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if rebake or not settled_cache.exists():
        bake_settled_particles(cache_path=settled_cache)

    demo = RoboticArmPourGenesisDemo(num_frames=num_frames, show_viewer=False, enable_camera=True)
    assert demo.camera is not None
    if not demo.load_settled_particles(settled_cache):
        bake_settled_particles(cache_path=settled_cache)
        demo.load_settled_particles(settled_cache)

    demo.scene.visualizer.update(force=True)
    demo.camera.render(force_render=True)
    demo.camera.start_recording()
    demo.camera.render(force_render=True)

    for _ in range(num_frames - 1):
        demo.step()
        demo.camera.render()

    demo.camera.stop_recording(save_to_filename=str(output), fps=VIDEO_FPS)
    return output
