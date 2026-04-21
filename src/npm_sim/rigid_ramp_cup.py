from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import newton
import newton.viewer
import numpy as np
import warp as wp

from .materials import MATERIALS
from .rigid_ramp_tower import (
    BALL_CLEARANCE,
    BALL_COLOR,
    BALL_LATERAL_OFFSET,
    BALL_RADIUS,
    BALL_START_MARGIN,
    CAMERA_FOV,
    CAMERA_PITCH,
    CAMERA_POS,
    CAMERA_YAW,
    FLOOR_COLOR,
    FLOOR_LENGTH,
    FLOOR_RAMP_OVERLAP,
    FLOOR_THICKNESS,
    FLOOR_WIDTH,
    FRAME_RATE,
    GROUND_SUPERBALL_BOUNCY,
    RAMP_ANGLE_DEG,
    RAMP_COLOR,
    RAMP_LENGTH,
    RAMP_THICKNESS,
    RAMP_WIDTH,
    RIGID_GAP,
    SIM_SUBSTEPS,
    SOLVER_CONTACT_RELAXATION,
    SOLVER_ITERATIONS,
    VIDEO_CROP_HEIGHT,
    VIDEO_CROP_WIDTH,
    VIDEO_CROP_X,
    VIDEO_CROP_Y,
    VIDEO_HEIGHT,
    VIDEO_NUM_FRAMES,
    VIDEO_WIDTH,
    _build_viewer,
    _quat_x,
    _shape_cfg,
    _static_shape_cfg,
    _superball_shape_cfg,
)

# Fixed rigid cup target constants for one readable scene.
CUP_CENTER_X = BALL_LATERAL_OFFSET
CUP_CENTER_Y = 1.06
CUP_HEIGHT = 0.24
CUP_BOTTOM_RADIUS = 0.072
CUP_TOP_RADIUS = 0.098
CUP_WALL_THICKNESS = 0.010
CUP_BASE_THICKNESS = 0.012
CUP_WALL_ROWS = 4
CUP_WALL_PANEL_COUNT = 12
CUP_BASE_EXTENT_SCALE = 0.78
CUP_WALL_PANEL_OVERLAP = 1.03
CUP_WALL_ROW_OVERLAP = 1.00
CUP_HANDLE_THICKNESS = 0.012
CUP_HANDLE_DEPTH = 0.018
CUP_HANDLE_HEIGHT = 0.12
CUP_HANDLE_CLEARANCE = 0.020
CUP_HANDLE_ATTACH_Z = 0.050

CUP_COLOR = wp.vec3(0.78, 0.18, 0.16)


def cup_wall_base_z(base_thickness: float = CUP_BASE_THICKNESS) -> float:
    return -CUP_HEIGHT * 0.5 + base_thickness


def cup_outer_radius_at_t(t: float) -> float:
    return (1.0 - t) * CUP_BOTTOM_RADIUS + t * CUP_TOP_RADIUS


@dataclass(frozen=True)
class SimulationResult:
    initial_ball_position: np.ndarray
    final_ball_position: np.ndarray
    cup_body_index: int
    initial_cup_transform: np.ndarray
    final_cup_transform: np.ndarray
    initial_body_poses: np.ndarray
    final_body_poses: np.ndarray


def add_cup_body(
    builder: newton.ModelBuilder,
    *,
    cfg: newton.ModelBuilder.ShapeConfig,
    floor_top_z: float,
    center_x: float = CUP_CENTER_X,
    center_y: float = CUP_CENTER_Y,
    color: wp.vec3 = CUP_COLOR,
    label: str = "cup",
    wall_thickness: float = CUP_WALL_THICKNESS,
    base_thickness: float = CUP_BASE_THICKNESS,
    wall_rows: int = CUP_WALL_ROWS,
    wall_panel_count: int = CUP_WALL_PANEL_COUNT,
    base_extent_scale: float = CUP_BASE_EXTENT_SCALE,
    panel_overlap: float = CUP_WALL_PANEL_OVERLAP,
    row_overlap: float = CUP_WALL_ROW_OVERLAP,
) -> int:
    cup_half_z = CUP_HEIGHT * 0.5
    cup_body_index = builder.add_body(
        xform=wp.transform(
            p=wp.vec3(center_x, center_y, floor_top_z + cup_half_z),
            q=wp.quat_identity(),
        ),
        label=label,
    )

    def shape_xform(local_pos: wp.vec3, local_rot: wp.quat) -> wp.transform:
        return wp.transform(p=local_pos, q=local_rot)

    add_cup_shapes_to_body(
        builder,
        body=cup_body_index,
        cfg=cfg,
        color=color,
        label=label,
        wall_thickness=wall_thickness,
        base_thickness=base_thickness,
        wall_rows=wall_rows,
        wall_panel_count=wall_panel_count,
        base_extent_scale=base_extent_scale,
        panel_overlap=panel_overlap,
        row_overlap=row_overlap,
        include_handle=True,
    )

    return cup_body_index


def add_cup_shapes_to_body(
    builder: newton.ModelBuilder,
    *,
    body: int,
    cfg: newton.ModelBuilder.ShapeConfig,
    color: wp.vec3 = CUP_COLOR,
    label: str = "cup",
    wall_thickness: float = CUP_WALL_THICKNESS,
    base_thickness: float = CUP_BASE_THICKNESS,
    wall_rows: int = CUP_WALL_ROWS,
    wall_panel_count: int = CUP_WALL_PANEL_COUNT,
    base_extent_scale: float = CUP_BASE_EXTENT_SCALE,
    panel_overlap: float = CUP_WALL_PANEL_OVERLAP,
    row_overlap: float = CUP_WALL_ROW_OVERLAP,
    include_handle: bool = True,
) -> None:
    def shape_xform(local_pos: wp.vec3, local_rot: wp.quat) -> wp.transform:
        return wp.transform(p=local_pos, q=local_rot)

    _add_cup_shapes(
        builder,
        body=body,
        shape_xform=shape_xform,
        cfg=cfg,
        color=color,
        label=label,
        wall_thickness=wall_thickness,
        base_thickness=base_thickness,
        wall_rows=wall_rows,
        wall_panel_count=wall_panel_count,
        base_extent_scale=base_extent_scale,
        panel_overlap=panel_overlap,
        row_overlap=row_overlap,
        include_handle=include_handle,
    )


def add_static_cup(
    builder: newton.ModelBuilder,
    *,
    cfg: newton.ModelBuilder.ShapeConfig,
    floor_top_z: float,
    center_x: float = CUP_CENTER_X,
    center_y: float = CUP_CENTER_Y,
    color: wp.vec3 = CUP_COLOR,
    label: str = "cup",
    wall_thickness: float = CUP_WALL_THICKNESS,
    base_thickness: float = CUP_BASE_THICKNESS,
    wall_rows: int = CUP_WALL_ROWS,
    wall_panel_count: int = CUP_WALL_PANEL_COUNT,
    base_extent_scale: float = CUP_BASE_EXTENT_SCALE,
    panel_overlap: float = CUP_WALL_PANEL_OVERLAP,
    row_overlap: float = CUP_WALL_ROW_OVERLAP,
) -> None:
    cup_half_z = CUP_HEIGHT * 0.5
    world_center = wp.vec3(center_x, center_y, floor_top_z + cup_half_z)

    def shape_xform(local_pos: wp.vec3, local_rot: wp.quat) -> wp.transform:
        return wp.transform(p=world_center + local_pos, q=local_rot)

    _add_cup_shapes(
        builder,
        body=-1,
        shape_xform=shape_xform,
        cfg=cfg,
        color=color,
        label=label,
        wall_thickness=wall_thickness,
        base_thickness=base_thickness,
        wall_rows=wall_rows,
        wall_panel_count=wall_panel_count,
        base_extent_scale=base_extent_scale,
        panel_overlap=panel_overlap,
        row_overlap=row_overlap,
    )


def _add_cup_shapes(
    builder: newton.ModelBuilder,
    *,
    body: int,
    shape_xform,
    cfg: newton.ModelBuilder.ShapeConfig,
    color: wp.vec3,
    label: str,
    wall_thickness: float,
    base_thickness: float,
    wall_rows: int,
    wall_panel_count: int,
    base_extent_scale: float,
    panel_overlap: float,
    row_overlap: float,
    include_handle: bool,
) -> None:
    cup_half_z = CUP_HEIGHT * 0.5
    wall_half = wall_thickness * 0.5
    base_half = base_thickness * 0.5
    wall_height = CUP_HEIGHT - base_thickness
    wall_row_height = wall_height / wall_rows
    handle_bar_half = CUP_HANDLE_THICKNESS * 0.5
    handle_depth_half = CUP_HANDLE_DEPTH * 0.5
    handle_height_half = CUP_HANDLE_HEIGHT * 0.5
    top_center_radius = CUP_TOP_RADIUS - wall_half
    handle_outer_center_x = CUP_TOP_RADIUS + CUP_HANDLE_CLEARANCE + handle_bar_half
    handle_bridge_half_x = 0.5 * (handle_outer_center_x - top_center_radius)
    handle_bridge_center_x = 0.5 * (handle_outer_center_x + top_center_radius)

    builder.add_shape_box(
        body,
        xform=shape_xform(wp.vec3(0.0, 0.0, -cup_half_z + base_half), wp.quat_identity()),
        hx=CUP_BOTTOM_RADIUS * base_extent_scale,
        hy=CUP_BOTTOM_RADIUS * base_extent_scale,
        hz=base_half,
        cfg=cfg,
        color=color,
        label=f"{label}_base",
    )
    for row_index in range(wall_rows):
        row_t = (row_index + 0.5) / wall_rows
        row_outer_radius = (1.0 - row_t) * CUP_BOTTOM_RADIUS + row_t * CUP_TOP_RADIUS
        row_center_radius = row_outer_radius - wall_half
        row_panel_half_length = row_center_radius * np.tan(np.pi / wall_panel_count) * panel_overlap
        row_center_z = -cup_half_z + base_thickness + (row_index + 0.5) * wall_row_height
        for panel_index in range(wall_panel_count):
            theta = 2.0 * np.pi * panel_index / wall_panel_count
            builder.add_shape_box(
                body,
                xform=shape_xform(
                    wp.vec3(
                        row_center_radius * np.cos(theta),
                        row_center_radius * np.sin(theta),
                        row_center_z,
                    ),
                    wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), theta),
                ),
                hx=wall_half,
                hy=row_panel_half_length,
                hz=wall_row_height * 0.5 * row_overlap,
                cfg=cfg,
                color=color,
                label=f"{label}_wall_{row_index}_{panel_index}",
            )
    if include_handle:
        builder.add_shape_box(
            body,
            xform=shape_xform(wp.vec3(handle_outer_center_x, 0.0, 0.0), wp.quat_identity()),
            hx=handle_bar_half,
            hy=handle_depth_half,
            hz=handle_height_half,
            cfg=cfg,
            color=color,
            label=f"{label}_handle_outer",
        )
        builder.add_shape_box(
            body,
            xform=shape_xform(wp.vec3(handle_bridge_center_x, 0.0, CUP_HANDLE_ATTACH_Z), wp.quat_identity()),
            hx=handle_bridge_half_x,
            hy=handle_depth_half,
            hz=handle_bar_half,
            cfg=cfg,
            color=color,
            label=f"{label}_handle_top",
        )
        builder.add_shape_box(
            body,
            xform=shape_xform(wp.vec3(handle_bridge_center_x, 0.0, -CUP_HANDLE_ATTACH_Z), wp.quat_identity()),
            hx=handle_bridge_half_x,
            hy=handle_depth_half,
            hz=handle_bar_half,
            cfg=cfg,
            color=color,
            label=f"{label}_handle_bottom",
        )


class RampCupDemo:
    def __init__(self, viewer: Any, ball_material: str = "steel", cube_material: str = "wood"):
        self.viewer = viewer
        self.fps = FRAME_RATE
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = SIM_SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps

        ball_preset = MATERIALS[ball_material]
        cup_preset = MATERIALS[cube_material]
        superball_scene = ball_material == "superball" and cube_material == "superball"
        if superball_scene:
            ball_cfg = _superball_shape_cfg(ball_preset, dynamic=True)
            cup_cfg = _superball_shape_cfg(cup_preset, dynamic=True)
            ground_cfg = _superball_shape_cfg(GROUND_SUPERBALL_BOUNCY, dynamic=False)
        else:
            ball_cfg = _shape_cfg(ball_preset)
            cup_cfg = _shape_cfg(cup_preset)
            ground_cfg = _static_shape_cfg(cup_preset)

        builder = newton.ModelBuilder()
        builder.rigid_gap = RIGID_GAP

        ramp_angle = np.deg2rad(RAMP_ANGLE_DEG)
        ramp_half_length = RAMP_LENGTH * 0.5
        ramp_half_thickness = RAMP_THICKNESS * 0.5
        floor_half_length = FLOOR_LENGTH * 0.5
        floor_half_thickness = FLOOR_THICKNESS * 0.5

        floor_top_z = 0.0
        ramp_center = wp.vec3(
            0.0,
            -ramp_half_length * np.cos(ramp_angle) - ramp_half_thickness * np.sin(ramp_angle),
            ramp_half_length * np.sin(ramp_angle) - ramp_half_thickness * np.cos(ramp_angle),
        )
        ramp_xform = wp.transform(p=ramp_center, q=_quat_x(-RAMP_ANGLE_DEG))
        floor_xform = wp.transform(
            p=wp.vec3(0.0, floor_half_length - FLOOR_RAMP_OVERLAP, floor_top_z - floor_half_thickness),
            q=wp.quat_identity(),
        )

        builder.add_shape_box(
            body=-1,
            xform=ramp_xform,
            hx=RAMP_WIDTH * 0.5,
            hy=ramp_half_length,
            hz=ramp_half_thickness,
            cfg=ground_cfg,
            color=RAMP_COLOR,
            label="ramp",
        )
        builder.add_shape_box(
            body=-1,
            xform=floor_xform,
            hx=FLOOR_WIDTH * 0.5,
            hy=floor_half_length,
            hz=floor_half_thickness,
            cfg=ground_cfg,
            color=FLOOR_COLOR,
            label="floor",
        )

        ball_local = wp.vec3(
            BALL_LATERAL_OFFSET,
            -ramp_half_length + BALL_START_MARGIN,
            ramp_half_thickness + BALL_RADIUS + BALL_CLEARANCE,
        )
        ball_start = wp.transform_point(ramp_xform, ball_local)
        self.ball_body_index = builder.add_body(
            xform=wp.transform(p=ball_start, q=wp.quat_identity()),
            label="ball",
        )
        builder.add_shape_sphere(
            self.ball_body_index,
            radius=BALL_RADIUS,
            cfg=ball_cfg,
            color=BALL_COLOR,
            label="ball_shape",
        )

        self.cup_body_index = add_cup_body(
            builder,
            cfg=cup_cfg,
            floor_top_z=floor_top_z,
            color=CUP_COLOR,
            label="cup",
        )

        self.model = builder.finalize()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=SOLVER_ITERATIONS,
            rigid_contact_relaxation=SOLVER_CONTACT_RELAXATION,
            enable_restitution=True,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=CAMERA_POS, pitch=CAMERA_PITCH, yaw=CAMERA_YAW)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = CAMERA_FOV

        self.initial_body_poses = self.body_poses()
        self.capture()

    def body_poses(self) -> np.ndarray:
        return self.state_0.body_q.numpy().copy()

    def capture(self) -> None:
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def result(self) -> SimulationResult:
        final_body_poses = self.body_poses()
        return SimulationResult(
            initial_ball_position=self.initial_body_poses[self.ball_body_index, :3].copy(),
            final_ball_position=final_body_poses[self.ball_body_index, :3].copy(),
            cup_body_index=self.cup_body_index,
            initial_cup_transform=self.initial_body_poses[self.cup_body_index].copy(),
            final_cup_transform=final_body_poses[self.cup_body_index].copy(),
            initial_body_poses=self.initial_body_poses.copy(),
            final_body_poses=final_body_poses,
        )


def build_scene(
    *,
    ball_material: str = "steel",
    cube_material: str = "wood",
    viewer: str | Any = "null",
    num_frames: int = 240,
    output_path: str | None = None,
    device: str | None = None,
) -> RampCupDemo:
    if device:
        wp.set_device(device)

    if isinstance(viewer, str):
        viewer_obj = _build_viewer(viewer, num_frames=num_frames, output_path=output_path)
    else:
        viewer_obj = viewer

    return RampCupDemo(
        viewer=viewer_obj,
        ball_material=ball_material,
        cube_material=cube_material,
    )


def run_simulation(
    *,
    ball_material: str = "steel",
    cube_material: str = "wood",
    viewer: str = "null",
    num_frames: int = 240,
    output_path: str | None = None,
    device: str | None = None,
) -> SimulationResult:
    demo = build_scene(
        ball_material=ball_material,
        cube_material=cube_material,
        viewer=viewer,
        num_frames=num_frames,
        output_path=output_path,
        device=device,
    )

    try:
        while demo.viewer.is_running():
            if not demo.viewer.is_paused():
                demo.step()
            demo.render()
        return demo.result()
    finally:
        demo.viewer.close()


def render_video(
    *,
    output_path: str = "outputs/ramp_cup_rigid.mp4",
    ball_material: str = "steel",
    cube_material: str = "wood",
    num_frames: int = VIDEO_NUM_FRAMES,
    device: str | None = None,
) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to render video output")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PYGLET_HEADLESS", "1")
    viewer = newton.viewer.ViewerGL(width=VIDEO_WIDTH, height=VIDEO_HEIGHT, headless=True)
    demo = build_scene(
        ball_material=ball_material,
        cube_material=cube_material,
        viewer=viewer,
        num_frames=num_frames,
        output_path=None,
        device=device,
    )

    command = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}",
        "-r",
        str(FRAME_RATE),
        "-i",
        "-",
        "-an",
        "-vf",
        (
            f"crop={VIDEO_CROP_WIDTH}:{VIDEO_CROP_HEIGHT}:{VIDEO_CROP_X}:{VIDEO_CROP_Y},"
            f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:flags=lanczos"
        ),
        "-vcodec",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    try:
        for frame_index in range(num_frames):
            if frame_index > 0:
                demo.step()
            demo.render()
            frame = viewer.get_frame()
            frame_np = np.ascontiguousarray(frame.numpy())
            assert process.stdin is not None
            process.stdin.write(frame_np.tobytes())

        assert process.stdin is not None
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr is not None else ""
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {stderr.strip()}")
        return output
    finally:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        if process.stderr is not None:
            process.stderr.close()
        viewer.close()
