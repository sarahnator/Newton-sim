"""Rigid ramp + cup + water scene using Genesis DFSPH.

Runs in the ``genesis-sim`` conda env only. Not wired into ``npm_sim.__init__``
because Genesis cannot coexist with Newton/Warp in one process.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

FRAME_RATE = 120  # 60
# 12 substeps puts effective SPH dt at ≈1.4 ms — inside the DFSPH CFL bound
# for this particle size while keeping SPH-rigid coupling tight enough to
# prevent particles tunnelling through thin cup walls between pressure solves.
SIM_SUBSTEPS = 6  # 12

RAMP_ANGLE_DEG = 20.0
RAMP_LENGTH = 1.25
RAMP_WIDTH = 0.45
RAMP_THICKNESS = 0.08

FLOOR_LENGTH = 2.00
FLOOR_WIDTH = 0.80
FLOOR_THICKNESS = 0.08
FLOOR_RAMP_OVERLAP = 0.03

# Ball radius, start position, and density are tuned together: a ~200 kg steel
# ball contacting near the cup rim after the full ramp runway is what carries
# the cup's CoM past its ~37° tip-over threshold cleanly onto its side.
BALL_RADIUS = 0.18
BALL_START_MARGIN = 0.02
BALL_LATERAL_OFFSET = 0.02
BALL_CLEARANCE = 0.003

CUP_CENTER_X = BALL_LATERAL_OFFSET
CUP_CENTER_Y = 1.06
CUP_HEIGHT = 0.24
CUP_BOTTOM_RADIUS = 0.072
CUP_TOP_RADIUS = 0.098
CUP_WALL_THICKNESS = 0.010
CUP_BASE_THICKNESS = 0.012
CUP_WALL_PANEL_COUNT = 12

WATER_PARTICLE_SIZE = 0.006
WATER_DENSITY = 1000.0
WATER_VISCOSITY = 1.0e-3
WATER_BRIM_CLEARANCE = 0.006

# Genesis SPH uses 0.8 * particle_size**3 as the per-particle rest volume, so
# the emission cylinder sizing below targets
# N_particles * 0.8 * ps**3 = TARGET_FILL_FRACTION * cavity_volume,
# with an overfill multiplier because CoACD decomposition widens the effective
# inner radius a few mm beyond the designed inner_r.
TARGET_FILL_FRACTION = 0.90
EMISSION_OVERFILL_FACTOR = 1.50

# Light (wooden-mug) density — at ceramic density (~1450) the cup resisted
# tipping past the rollover threshold even with the 200 kg ball.
CUP_DENSITY = 650.0
BALL_DENSITY = 7850.0

CAMERA_POS = (3.0, -0.6, 1.4)
CAMERA_LOOKAT = (0.0, 0.6, 0.15)
CAMERA_FOV = 55.0
VIDEO_RESOLUTION = (1280, 720)
VIDEO_FPS = 60

# Bake-and-reload cache for post-settle particle positions. Lets render_video
# open frame 0 already at DFSPH equilibrium density, skipping the visible
# ~0.25 s compression from regular-grid emission to rest density.
SETTLED_PARTICLES_CACHE = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "_genesis"
    / "settled_water.npy"
)
SETTLE_BAKE_SECONDS = 0.8

PRE_SETTLE_SECONDS = 0.0


@dataclass(frozen=True)
class SimulationResult:
    initial_ball_position: np.ndarray
    final_ball_position: np.ndarray
    initial_cup_position: np.ndarray
    final_cup_position: np.ndarray
    initial_cup_quat_wxyz: np.ndarray
    final_cup_quat_wxyz: np.ndarray
    max_cup_tilt_degrees: float
    final_cup_tilt_degrees: float
    initial_particle_positions: np.ndarray
    final_particle_positions: np.ndarray


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion in Genesis/MuJoCo-style wxyz order to a rotation matrix."""
    w, x, y, z = [float(v) for v in q]
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _cup_tilt_degrees_from_quat_wxyz(q: np.ndarray) -> float:
    """
    Tilt angle between cup local +z axis and world +z axis.

    0 deg   = upright
    90 deg  = lying on its side
    180 deg = upside down
    """
    rot = _quat_wxyz_to_matrix(np.asarray(q, dtype=np.float64))
    cup_up_world = rot @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cos_tilt = float(np.clip(np.dot(cup_up_world, world_up), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_tilt)))


def cup_fell_over_from_tilt(max_tilt_degrees: float, threshold_degrees: float = 60.0) -> bool:
    """
    Geometric fall-over label.

    A 60 degree threshold catches cases where the cup has clearly entered
    a tipping/falling regime even if it has not fully settled on its side.
    """
    return float(max_tilt_degrees) >= threshold_degrees


def _cup_mesh_path() -> Path:
    return Path(__file__).resolve().parents[2] / "outputs" / "_genesis" / "cup_hollow_cylinder.obj"


def build_cup_mesh(path: Optional[Path] = None) -> Path:
    """Write a watertight hollow-cylinder mug shell to ``path``.

    Must be one continuous mesh, not a pile of box primitives — Genesis's
    SPH-rigid coupling needs one coherent SDF per geom to contain water
    reliably; CoACD decomposition downstream splits it into a few convex
    pieces. Local frame: centre at origin, base at z = -CUP_HEIGHT/2.
    """
    import trimesh

    if path is None:
        path = _cup_mesh_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    n = 48
    half_h = CUP_HEIGHT * 0.5
    base_z = -half_h
    rim_z = half_h

    outer_r = 0.5 * (CUP_BOTTOM_RADIUS + CUP_TOP_RADIUS) + 0.004
    inner_r = outer_r - max(CUP_WALL_THICKNESS * 2.0, 0.02)
    inner_base_z = base_z + CUP_BASE_THICKNESS

    def _ring(radius: float, z: float) -> np.ndarray:
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        return np.stack(
            [radius * np.cos(angles), radius * np.sin(angles), np.full(n, z)],
            axis=1,
        )

    outer_bot = _ring(outer_r, base_z)
    outer_top = _ring(outer_r, rim_z)
    inner_bot = _ring(inner_r, inner_base_z)
    inner_top = _ring(inner_r, rim_z)
    outer_base_ring = _ring(outer_r, base_z)
    inner_floor_ring = _ring(inner_r, inner_base_z)

    verts = np.concatenate(
        [outer_bot, outer_top, inner_bot, inner_top, outer_base_ring, inner_floor_ring],
        axis=0,
    )
    off_ob, off_ot, off_ib, off_it, off_base, off_floor = 0, n, 2 * n, 3 * n, 4 * n, 5 * n

    base_centre_idx = verts.shape[0]
    verts = np.concatenate([verts, np.array([[0.0, 0.0, base_z]])], axis=0)

    inner_floor_centre_idx = verts.shape[0]
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

        faces.append([base_centre_idx, off_base + j, off_base + i])
        faces.append([inner_floor_centre_idx, off_floor + i, off_floor + j])

    mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces), process=True)
    mesh.fix_normals()
    mesh.export(path)
    return path


class RampCupWaterGenesisDemo:
    def __init__(
        self,
        *,
        num_frames: int,
        show_viewer: bool = False,
        enable_camera: bool = False,
    ):
        import genesis as gs

        # gs.init() can only run once per process; bake_settled_particles() and
        # render_video() both spin up a Demo so we tolerate the second call.
        try:
            gs.init(backend=gs.gpu, logging_level="warning")
        except gs.GenesisException as exc:
            if "already initialized" not in str(exc).lower():
                raise

        self.gs = gs
        self.num_frames = num_frames
        self.frame_dt = 1.0 / FRAME_RATE
        self.sim_time = 0.0

        # upper_bound.y extends well past the cup so spilled water spreads
        # freely instead of piling up against the reflecting domain boundary.
        sph_lower = (-1.2, -1.5, -0.2)
        sph_upper = (1.2, 4.0, 1.5)

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.frame_dt,
                substeps=SIM_SUBSTEPS,
                gravity=(0.0, 0.0, -9.81),
            ),
            sph_options=gs.options.SPHOptions(
                particle_size=WATER_PARTICLE_SIZE,
                pressure_solver="DFSPH",
                lower_bound=sph_lower,
                upper_bound=sph_upper,
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.frame_dt,
                gravity=(0.0, 0.0, -9.81),
            ),
            show_viewer=show_viewer,
        )

        self._add_static_geometry()
        self._add_ball()
        self._add_cup(fixed=False)
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

        self.initial_ball_pos = self._ball_pos()
        self.initial_cup_pos = self._cup_pos()
        self.initial_cup_quat_wxyz = self._cup_quat_wxyz()
        self.initial_particles = self._particle_positions()

        initial_tilt = _cup_tilt_degrees_from_quat_wxyz(self.initial_cup_quat_wxyz)
        self.max_cup_tilt_degrees = float(initial_tilt)

    def _add_static_geometry(self) -> None:
        gs = self.gs
        self.floor = self.scene.add_entity(
            material=gs.materials.Rigid(),
            morph=gs.morphs.Plane(),
        )

        ramp_angle = math.radians(RAMP_ANGLE_DEG)
        ramp_half_length = RAMP_LENGTH * 0.5
        ramp_half_thickness = RAMP_THICKNESS * 0.5
        ramp_centre = (
            0.0,
            -ramp_half_length * math.cos(ramp_angle) - ramp_half_thickness * math.sin(ramp_angle),
            ramp_half_length * math.sin(ramp_angle) - ramp_half_thickness * math.cos(ramp_angle),
        )
        self.ramp = self.scene.add_entity(
            material=gs.materials.Rigid(),
            morph=gs.morphs.Box(
                pos=ramp_centre,
                size=(RAMP_WIDTH, RAMP_LENGTH, RAMP_THICKNESS),
                euler=(-RAMP_ANGLE_DEG, 0.0, 0.0),
                fixed=True,
            ),
        )

    def _add_ball(self) -> None:
        """Place the ball just above the ramp's top surface near the high end.

        Ramp is rotated by R_x(-RAMP_ANGLE_DEG); we map a ramp-local point
        (x, y_local, z_local) to world via (x, y*cos + z*sin, -y*sin + z*cos)
        and offset by the ramp centre.
        """
        gs = self.gs
        angle = math.radians(RAMP_ANGLE_DEG)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        ramp_half_length = RAMP_LENGTH * 0.5
        ramp_half_thickness = RAMP_THICKNESS * 0.5

        local_y = -ramp_half_length + BALL_START_MARGIN
        local_z = ramp_half_thickness + BALL_RADIUS + BALL_CLEARANCE

        ramp_centre_y = -ramp_half_length * cos_a - ramp_half_thickness * sin_a
        ramp_centre_z = ramp_half_length * sin_a - ramp_half_thickness * cos_a

        ball_y = local_y * cos_a + local_z * sin_a + ramp_centre_y
        ball_z = -local_y * sin_a + local_z * cos_a + ramp_centre_z

        self.ball = self.scene.add_entity(
            material=gs.materials.Rigid(rho=BALL_DENSITY),
            morph=gs.morphs.Sphere(
                pos=(BALL_LATERAL_OFFSET, ball_y, ball_z),
                radius=BALL_RADIUS,
            ),
        )

    def _add_cup(self, *, fixed: bool = True) -> None:
        """Single hollow-cylinder mesh cup with hard SPH coupling.

        Must be one mesh (not a pile of box primitives): Genesis SPH-rigid
        coupling leaks through any container split across many small rigid
        geoms because each SDF has limited reach. ``coup_softness=0.0`` and a
        fine SDF are also required to stop SPH tunnelling through thin walls.
        """
        gs = self.gs
        mesh_path = build_cup_mesh()
        cup_material = gs.materials.Rigid(
            rho=CUP_DENSITY,
            coup_softness=0.0,
            coup_friction=1.0,
            coup_restitution=0.0,
            sdf_cell_size=0.004,
            sdf_min_res=32,
            sdf_max_res=128,
        )

        cup_half_h = CUP_HEIGHT * 0.5
        self.cup = self.scene.add_entity(
            material=cup_material,
            morph=gs.morphs.Mesh(
                file=str(mesh_path),
                pos=(CUP_CENTER_X, CUP_CENTER_Y, cup_half_h),
                fixed=fixed,
                decimate=False,
                convexify=False,
            ),
            surface=gs.surfaces.Default(
                color=(0.85, 0.22, 0.18, 0.45),
            ),
        )

    def _add_water(self) -> None:
        gs = self.gs

        # Emission cylinder sized for the settled column after DFSPH compresses
        # regular-grid sampling to rest density.
        outer_r = 0.5 * (CUP_BOTTOM_RADIUS + CUP_TOP_RADIUS) + 0.004
        inner_r = outer_r - max(CUP_WALL_THICKNESS * 2.0, 0.02)

        fill_bottom_z = CUP_BASE_THICKNESS + WATER_PARTICLE_SIZE
        cup_inner_height = CUP_HEIGHT - CUP_BASE_THICKNESS - WATER_BRIM_CLEARANCE - WATER_PARTICLE_SIZE
        target_settled_height = TARGET_FILL_FRACTION * cup_inner_height
        target_settled_volume = math.pi * inner_r**2 * target_settled_height

        rest_volume_per_particle = 0.8 * WATER_PARTICLE_SIZE**3
        target_n_particles = target_settled_volume / rest_volume_per_particle
        emission_volume = target_n_particles * EMISSION_OVERFILL_FACTOR * WATER_PARTICLE_SIZE**3

        cylinder_radius = max(inner_r - WATER_PARTICLE_SIZE, WATER_PARTICLE_SIZE)
        height = emission_volume / (math.pi * cylinder_radius**2)
        cylinder_centre_z = fill_bottom_z + height * 0.5

        self.water = self.scene.add_entity(
            material=gs.materials.SPH.Liquid(rho=WATER_DENSITY, mu=WATER_VISCOSITY),
            morph=gs.morphs.Cylinder(
                pos=(CUP_CENTER_X, CUP_CENTER_Y, cylinder_centre_z),
                radius=cylinder_radius,
                height=height,
            ),
            surface=gs.surfaces.Default(
                color=(0.25, 0.55, 0.95, 1.0),
                # "particle" is ~12× faster than "recon" at this particle
                # count; the frame-0 render-buffer sync issue with particle
                # mode is handled by force_render + visualizer.update in
                # render_video.
                vis_mode="particle",
            ),
        )

    def _ball_pos(self) -> np.ndarray:
        return self.ball.get_pos().cpu().numpy().reshape(-1).copy()

    def _cup_pos(self) -> np.ndarray:
        return self.cup.get_pos().cpu().numpy().reshape(-1).copy()

    def _cup_quat_wxyz(self) -> np.ndarray:
        return self.cup.get_quat().cpu().numpy().reshape(-1).copy()

    def _particle_positions(self) -> np.ndarray:
        return self.water.get_particles_pos().cpu().numpy().copy()

    def step(self) -> None:
        self.scene.step()
        self.sim_time += self.frame_dt

        cup_quat = self._cup_quat_wxyz()
        cup_tilt = _cup_tilt_degrees_from_quat_wxyz(cup_quat)
        self.max_cup_tilt_degrees = max(self.max_cup_tilt_degrees, float(cup_tilt))

    def pre_settle(self, seconds: float = PRE_SETTLE_SECONDS) -> None:
        """Step the sim with the ball held kinematically at its start pose."""
        if self.ball is None or seconds <= 0:
            return

        ball_pos = self.ball.get_pos().cpu().numpy().reshape(-1).copy()
        ball_quat = self.ball.get_quat().cpu().numpy().reshape(-1).copy()

        n = int(round(seconds * FRAME_RATE))
        for _ in range(n):
            self.ball.set_pos(ball_pos)
            self.ball.set_quat(ball_quat)
            self.ball.zero_all_dofs_velocity()
            self.scene.step()

        self.ball.set_pos(ball_pos)
        self.ball.set_quat(ball_quat)
        self.ball.zero_all_dofs_velocity()

    def trim_overflow_particles(self) -> None:
        """Stash particles that aren't in the cup cavity below the floor.

        Genesis has no API to delete particles, so we move overflow ones far
        under the floor plane (out of camera view) with zero velocity.
        """
        positions = self._particle_positions()

        cup_xy = np.array([CUP_CENTER_X, CUP_CENTER_Y])
        outer_r = 0.5 * (CUP_BOTTOM_RADIUS + CUP_TOP_RADIUS) + 0.004
        inner_r = outer_r - max(CUP_WALL_THICKNESS * 2.0, 0.02)

        rim_z = CUP_HEIGHT - WATER_BRIM_CLEARANCE
        floor_z = CUP_BASE_THICKNESS - 0.001

        r_xy = np.linalg.norm(positions[:, :2] - cup_xy, axis=1)
        in_cavity = (
            (r_xy < inner_r)
            & (positions[:, 2] >= floor_z)
            & (positions[:, 2] <= rim_z)
        )

        if int((~in_cavity).sum()) > 0:
            positions[~in_cavity] = np.array([10.0, 10.0, -10.0], dtype=positions.dtype)
            self.water.set_particles_pos(positions.astype(np.float32))
            self.water.set_particles_vel(np.zeros_like(positions, dtype=np.float32))

    def load_settled_particles(self, path: Path) -> bool:
        """Restore baked particle positions; return False on a stale cache.

        Cache is rejected if particle count differs, or if fewer than half of
        the cached particles lie inside the current cup footprint or in the
        stashed zone from trim_overflow_particles.
        """
        if not path.exists():
            return False

        settled = np.load(path).astype(np.float32)
        if settled.shape[0] != self.water.n_particles:
            return False

        cup_xy = np.array([CUP_CENTER_X, CUP_CENTER_Y])
        outer_r = 0.5 * (CUP_BOTTOM_RADIUS + CUP_TOP_RADIUS) + 0.004

        in_footprint = np.linalg.norm(settled[:, :2] - cup_xy, axis=1) < outer_r
        stashed = settled[:, 2] < -0.05

        if int((in_footprint | stashed).sum()) < 0.5 * settled.shape[0]:
            return False

        self.water.set_particles_pos(settled)
        self.water.set_particles_vel(np.zeros_like(settled))
        return True

    def run(self) -> SimulationResult:
        for _ in range(self.num_frames):
            self.step()

        final_cup_quat = self._cup_quat_wxyz()
        final_cup_tilt = _cup_tilt_degrees_from_quat_wxyz(final_cup_quat)

        return SimulationResult(
            initial_ball_position=self.initial_ball_pos,
            final_ball_position=self._ball_pos(),
            initial_cup_position=self.initial_cup_pos,
            final_cup_position=self._cup_pos(),
            initial_cup_quat_wxyz=self.initial_cup_quat_wxyz,
            final_cup_quat_wxyz=final_cup_quat,
            max_cup_tilt_degrees=float(self.max_cup_tilt_degrees),
            final_cup_tilt_degrees=float(final_cup_tilt),
            initial_particle_positions=self.initial_particles,
            final_particle_positions=self._particle_positions(),
        )


def run_simulation(*, num_frames: int = 120, show_viewer: bool = False) -> SimulationResult:
    demo = RampCupWaterGenesisDemo(num_frames=num_frames, show_viewer=show_viewer)
    return demo.run()


def bake_settled_particles(
    *,
    cache_path: Path = SETTLED_PARTICLES_CACHE,
    settle_seconds: float = SETTLE_BAKE_SECONDS,
) -> Path:
    """Run a headless settle pass and save the resulting particle cloud.

    Genesis's ``regular`` sampler places particles at spacing = ``particle_size``,
    but DFSPH rest density corresponds to ~``0.8 * particle_size**3`` per
    particle, so fresh emission is under-compressed and the first ~0.25 s of
    sim is DFSPH compacting the column to correct density. Baking that once
    lets render_video open at the true equilibrium.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    demo = RampCupWaterGenesisDemo(num_frames=0, show_viewer=False, enable_camera=False)
    demo.pre_settle(settle_seconds)
    demo.trim_overflow_particles()

    # Brief re-equilibration so the surface is smooth after stashing overflow.
    for _ in range(int(round(0.2 * FRAME_RATE))):
        demo.step()

    settled = demo._particle_positions().astype(np.float32)
    np.save(cache_path, settled)
    return cache_path


def render_video(
    *,
    output_path: str = "outputs/rigid_ramp_cup_water_genesis.mp4",
    num_frames: int = 240,
    settled_cache: Path = SETTLED_PARTICLES_CACHE,
    rebake: bool = False,
) -> Path:
    """Render the cup-water MP4 starting from a baked settled water state.

    Genesis v0.4.6 gotcha: ``Camera.start_recording()`` only flips a flag — it
    does NOT capture a frame. The explicit ``force_render`` call before the
    first ``scene.step()`` is what makes frame 0 the settled-state image.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if rebake or not settled_cache.exists():
        bake_settled_particles(cache_path=settled_cache)

    demo = RampCupWaterGenesisDemo(num_frames=num_frames, show_viewer=False, enable_camera=True)
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