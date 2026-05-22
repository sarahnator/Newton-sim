#!/usr/bin/env python3
"""
Minimal Genesis pendulum scene for physically-viable benchmark prototyping.

This script:
  - writes a tiny MJCF pendulum model with one hinge joint,
  - loads it in Genesis,
  - runs the simulation,
  - optionally renders a video,
  - logs q(t), qdot(t), and analytic small-angle reference,
  - saves metadata.json and trajectory.npz.

Run without video:
  uv run python src/npm_sim/pendulum_genesis.py \
    --output-dir outputs/pendulum \
    --theta0-deg 30 \
    --damping 0.02 \
    --length 1.0 \
    --mass 1.0 \
    # --viewer

Run with video:
  uv run python src/npm_sim/pendulum_genesis.py \
    --output-dir outputs/pendulum \
    --theta0-deg 30 \
    --damping 0.02 \
    --length 1.0 \
    --mass 1.0 \
    --num-steps 600 \
    --render-video
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class PendulumConfig:
    length: float = 1.0
    mass: float = 1.0
    bob_radius: float = 0.06
    damping: float = 0.02
    theta0_deg: float = 30.0
    omega0: float = 0.0
    gravity: float = 9.81
    dt: float = 1.0 / 240.0
    substeps: int = 4
    num_steps: int = 1200
    seed: int = 0
    backend: str = "cpu"
    show_viewer: bool = False

    # Optional video rendering.
    render_video: bool = False
    video_fps: int = 60
    video_width: int = 1280
    video_height: int = 720


def write_pendulum_mjcf(path: Path, cfg: PendulumConfig) -> None:
    """
    MJCF model:
      world
        └── body pendulum
              └── hinge joint about world y-axis
              └── capsule rod from pivot to bob
              └── sphere bob at distance length

    The hinge angle is approximately the pendulum angle in radians.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # The pendulum hangs along -z at q = 0.
    rod_radius = 0.012

    xml = f"""<mujoco model="minimal_pendulum">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="{cfg.dt}" gravity="0 0 -{cfg.gravity}"/>

  <default>
    <joint damping="{cfg.damping}" armature="0.0"/>
    <geom friction="0.0 0.0 0.0"/>
  </default>

  <worldbody>
    <light name="main_light" pos="0 -3 3" dir="0 1 -1" diffuse="1 1 1"/>
    <geom name="ground"
          type="plane"
          size="2 2 0.02"
          pos="0 0 -1.25"
          rgba="0.75 0.75 0.75 1"/>

    <body name="pendulum" pos="0 0 0">
      <joint name="hinge"
             type="hinge"
             axis="0 1 0"
             pos="0 0 0"
             limited="false"/>

      <geom name="rod"
            type="capsule"
            fromto="0 0 0 0 0 -{cfg.length}"
            size="{rod_radius}"
            density="50"
            rgba="0.2 0.2 0.2 1"/>

      <geom name="bob"
            type="sphere"
            pos="0 0 -{cfg.length}"
            size="{cfg.bob_radius}"
            mass="{cfg.mass}"
            rgba="0.9 0.1 0.1 1"/>
    </body>
  </worldbody>
</mujoco>
"""
    path.write_text(xml)


def maybe_to_numpy(x):
    """
    Genesis may return torch/taichi/numpy-like tensors depending on backend/version.
    This keeps logging tolerant.
    """
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def get_first_dof_position(entity) -> float:
    try:
        return float(maybe_to_numpy(entity.get_dofs_position())[0])
    except Exception:
        return float("nan")


def get_first_dof_velocity(entity) -> float:
    try:
        return float(maybe_to_numpy(entity.get_dofs_velocity())[0])
    except Exception:
        return float("nan")


def set_initial_state(entity, theta0_rad: float, omega0: float) -> None:
    """
    These methods are used by Genesis rigid/articulated entities in common examples,
    but exact availability can depend on Genesis version.
    """
    if hasattr(entity, "set_dofs_position"):
        entity.set_dofs_position(np.array([theta0_rad], dtype=np.float32))
    else:
        print("[warning] entity has no set_dofs_position; initial angle may remain zero.")

    if hasattr(entity, "set_dofs_velocity"):
        entity.set_dofs_velocity(np.array([omega0], dtype=np.float32))
    else:
        print("[warning] entity has no set_dofs_velocity; initial angular velocity may remain zero.")


def stop_camera_recording(camera, video_path: Path, fps: int) -> None:
    """
    Genesis camera APIs can vary slightly across versions.
    Try the fps argument first, then fall back to no fps.
    """
    try:
        camera.stop_recording(save_to_filename=str(video_path), fps=fps)
    except TypeError:
        camera.stop_recording(save_to_filename=str(video_path))


def run_simulation(cfg: PendulumConfig, output_dir: Path) -> dict:
    import genesis as gs

    output_dir.mkdir(parents=True, exist_ok=True)
    mjcf_path = output_dir / "pendulum.xml"
    video_path = output_dir / "pendulum.mp4" if cfg.render_video else None

    write_pendulum_mjcf(mjcf_path, cfg)

    backend = getattr(gs, cfg.backend)
    gs.init(
        backend=backend,
        seed=cfg.seed,
        precision="32",
        logging_level="warning",
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=cfg.dt,
            substeps=cfg.substeps,
            gravity=(0.0, 0.0, -cfg.gravity),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, -3.0, 1.2),
            camera_lookat=(0.0, 0.0, -0.5),
            camera_fov=35,
        ),
        show_viewer=cfg.show_viewer,
    )

    camera = None
    if cfg.render_video:
        camera = scene.add_camera(
            res=(cfg.video_width, cfg.video_height),
            pos=(2.0, -3.0, 1.2),
            lookat=(0.0, 0.0, -0.5),
            fov=35,
            GUI=False,
        )

    # Genesis docs show loading MJCF through gs.morphs.MJCF(file=...).
    pendulum = scene.add_entity(
        gs.morphs.MJCF(file=str(mjcf_path)),
    )

    scene.build()

    theta0_rad = math.radians(cfg.theta0_deg)
    set_initial_state(pendulum, theta0_rad, cfg.omega0)

    ts = np.zeros(cfg.num_steps + 1, dtype=np.float32)
    qs = np.zeros(cfg.num_steps + 1, dtype=np.float32)
    qdots = np.zeros(cfg.num_steps + 1, dtype=np.float32)

    qs[0] = get_first_dof_position(pendulum)
    qdots[0] = get_first_dof_velocity(pendulum)

    if camera is not None:
        camera.start_recording()
        camera.render()

    for i in range(1, cfg.num_steps + 1):
        scene.step()

        if camera is not None:
            camera.render()

        ts[i] = i * cfg.dt
        qs[i] = get_first_dof_position(pendulum)
        qdots[i] = get_first_dof_velocity(pendulum)

    if camera is not None and video_path is not None:
        stop_camera_recording(camera, video_path, cfg.video_fps)

    # Small-angle analytic reference for sanity checks:
    # theta(t) = theta0 cos(sqrt(g/L) t), ignoring damping.
    omega_n = math.sqrt(cfg.gravity / cfg.length)
    theta_ref = theta0_rad * np.cos(omega_n * ts)

    # Approximate bob position implied by logged joint angle.
    # q rotates the rod around y-axis from the -z rest configuration.
    bob_x = -cfg.length * np.sin(qs)
    bob_y = np.zeros_like(qs)
    bob_z = -cfg.length * np.cos(qs)
    bob_pos = np.stack([bob_x, bob_y, bob_z], axis=-1)

    result_path = output_dir / "trajectory.npz"
    np.savez_compressed(
        result_path,
        t=ts,
        q=qs,
        qdot=qdots,
        theta_small_angle_ref=theta_ref.astype(np.float32),
        bob_pos=bob_pos.astype(np.float32),
    )

    metadata = {
        "scene_family": "pendulum",
        "concepts": ["gravity", "length", "mass", "damping", "drag_or_dissipation"],
        "sweepable_parameters": ["length", "mass", "damping", "theta0_deg", "gravity"],
        "config": asdict(cfg),
        "mjcf_path": str(mjcf_path),
        "trajectory_path": str(result_path),
        "video_path": str(video_path) if video_path is not None else None,
        "validation_notes": [
            "For small theta0_deg and low damping, q(t) should approximately match theta0*cos(sqrt(g/L)*t).",
            "Mass should not change the ideal pendulum period, so mass sweeps are a useful invariance test.",
            "Length should change the period as T = 2*pi*sqrt(L/g).",
            "Damping should reduce amplitude over time.",
        ],
    }

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="outputs/pendulum")
    parser.add_argument("--length", type=float, default=1.0)
    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--bob-radius", type=float, default=0.06)
    parser.add_argument("--damping", type=float, default=0.02)
    parser.add_argument("--theta0-deg", type=float, default=30.0)
    parser.add_argument("--omega0", type=float, default=0.0)
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--dt", type=float, default=1.0 / 240.0)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", type=str, default="cpu", choices=["cpu", "gpu", "cuda", "metal"])
    parser.add_argument("--viewer", action="store_true")

    # Video options.
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=60)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = PendulumConfig(
        length=args.length,
        mass=args.mass,
        bob_radius=args.bob_radius,
        damping=args.damping,
        theta0_deg=args.theta0_deg,
        omega0=args.omega0,
        gravity=args.gravity,
        dt=args.dt,
        substeps=args.substeps,
        num_steps=args.num_steps,
        seed=args.seed,
        backend=args.backend,
        show_viewer=args.viewer,
        render_video=args.render_video,
        video_fps=args.video_fps,
        video_width=args.video_width,
        video_height=args.video_height,
    )

    metadata = run_simulation(cfg, Path(args.output_dir))
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()