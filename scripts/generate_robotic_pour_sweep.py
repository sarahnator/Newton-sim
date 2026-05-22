#!/usr/bin/env python3
"""
Generate one-variable-at-a-time benchmark sweeps for src/npm_sim/robotic_arm_pour_genesis.py.

Examples:

  # Dry run
  uv run python scripts/generate_robotic_pour_sweep.py \
    --sweep-param water_viscosity --dry-run

  # Generate no-video entries
  uv run python scripts/generate_robotic_pour_sweep.py \
    --sweep-param water_viscosity --seeds 0 1 2 --num-frames 480 --no-video

  # Generate videos
  uv run python scripts/generate_robotic_pour_sweep.py \
    --sweep-param water_fill_fraction --seeds 0 --num-frames 480
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE_MODULE_PATH = REPO_ROOT / "src" / "npm_sim" / "robotic_arm_pour_genesis.py"


DEFAULT_GRIDS = {
    # Main latent physical concept: fluid viscosity / transfer rate.
    "water_viscosity": [1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 6.0e-3, 1.0e-2, 3.0e-2],

    # Fluid inertia.
    "water_density": [700.0, 850.0, 1000.0, 1150.0, 1300.0, 1450.0, 1600.0],

    # Initial volume / fill-level.
    "water_fill_fraction": [0.25, 0.40, 0.55, 0.70, 0.80, 0.90, 0.97],

    # Surface cohesion / breakup behavior.
    "liquid_surface_tension": [0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08],

    # Rigid-fluid contact coupling against glass.
    "glass_coup_friction": [0.0, 0.01, 0.025, 0.05, 0.08, 0.12, 0.16],

    # Action-side parameter, useful for planning/control queries.
    "pour_hold_seconds": [0.0, 0.25, 0.50, 1.00, 1.50, 2.50, 4.00],

    # Action-side parameter: how close the robot gets to full pour pose.
    "pour_pose_fraction": [0.55, 0.65, 0.72, 0.80, 0.86, 0.92, 1.00],
}


PARAM_TO_MODULE_CONSTANT = {
    "water_viscosity": "WATER_VISCOSITY",
    "water_density": "WATER_DENSITY",
    "water_fill_fraction": "WATER_FILL_FRACTION",
    "liquid_surface_tension": "LIQUID_SURFACE_TENSION",
    "glass_coup_friction": "GLASS_COUP_FRICTION",
    "pour_hold_seconds": "POUR_HOLD_SECONDS",
    "pour_pose_fraction": "POUR_POSE_FRACTION",
}


@dataclass(frozen=True)
class SweepEntry:
    scene_family: str
    scene_id: str
    causal_factor: str
    parameter_name: str
    parameter_value: float
    seed: int
    num_frames: int
    output_dir: str
    video_path: str | None
    result_path: str
    metadata_path: str
    cache_path: str


def stable_id(parts: list[Any]) -> str:
    text = json.dumps(parts, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def load_scene_module():
    spec = importlib.util.spec_from_file_location("robotic_arm_pour_genesis_sweep", SCENE_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load scene module at {SCENE_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_values(args: argparse.Namespace) -> list[float]:
    if args.values:
        return [float(v) for v in args.values.split(",")]
    return list(DEFAULT_GRIDS[args.sweep_param])


def build_entries(args: argparse.Namespace) -> list[SweepEntry]:
    entries: list[SweepEntry] = []

    for value in parse_values(args):
        for seed in args.seeds:
            sid = stable_id(["robotic_arm_pour", args.sweep_param, value, seed, args.num_frames])
            scene_id = f"robotic_pour_{args.sweep_param}_{sid}"
            out_dir = Path(args.output_root) / args.sweep_param / scene_id

            video_path = None if args.no_video else str(out_dir / "video.mp4")
            cache_path = str(out_dir / "settled_water.npy")

            entries.append(
                SweepEntry(
                    scene_family="robotic_arm_pour",
                    scene_id=scene_id,
                    causal_factor=args.sweep_param,
                    parameter_name=args.sweep_param,
                    parameter_value=float(value),
                    seed=int(seed),
                    num_frames=int(args.num_frames),
                    output_dir=str(out_dir),
                    video_path=video_path,
                    result_path=str(out_dir / "result.npz"),
                    metadata_path=str(out_dir / "metadata.json"),
                    cache_path=cache_path,
                )
            )

    return entries


def write_manifest(entries: list[SweepEntry], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_root / "manifest.jsonl"
    csv_path = output_root / "manifest.csv"

    with jsonl_path.open("w") as f:
        for entry in entries:
            f.write(json.dumps(asdict(entry), sort_keys=True) + "\n")

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(entries[0]).keys()))
        writer.writeheader()
        for entry in entries:
            writer.writerow(asdict(entry))

    print(f"Wrote {jsonl_path}")
    print(f"Wrote {csv_path}")


def recompute_duration_dependent_constants(mod) -> None:
    """
    Some action parameters affect VIDEO_NUM_FRAMES and/or PANDA_Q_POUR.
    Keep module-level derived constants consistent after patching.
    """
    if hasattr(mod, "VIDEO_DT") and hasattr(mod, "MICROSTEPS_PER_FRAME"):
        mod.PHYSICS_DT = mod.VIDEO_DT / mod.MICROSTEPS_PER_FRAME

    if hasattr(mod, "PANDA_Q_UPRIGHT") and hasattr(mod, "PANDA_Q_FULL_POUR") and hasattr(mod, "POUR_POSE_FRACTION"):
        mod.PANDA_Q_POUR = mod.PANDA_Q_UPRIGHT + (mod.PANDA_Q_FULL_POUR - mod.PANDA_Q_UPRIGHT) * mod.POUR_POSE_FRACTION

    duration = (
        mod.SURFACE_HOLD_SECONDS
        + mod.LIFT_SECONDS
        + mod.PRE_POUR_HOLD_SECONDS
        + mod.TILT_SECONDS
        + mod.POUR_HOLD_SECONDS
        + mod.RETURN_SECONDS
        + mod.PLACE_BACK_SECONDS
        + mod.FINAL_HOLD_SECONDS
    )
    mod.VIDEO_NUM_FRAMES = int(round(duration * mod.FRAME_RATE))


def configure_module_for_entry(mod, entry: SweepEntry) -> str:
    constant_name = PARAM_TO_MODULE_CONSTANT[entry.parameter_name]
    setattr(mod, constant_name, float(entry.parameter_value))

    # Keep derived constants consistent for action/duration sweeps.
    recompute_duration_dependent_constants(mod)

    # Give every config its own settled cache to prevent accidental state reuse.
    mod.SETTLED_PARTICLES_CACHE = Path(entry.cache_path)

    # Optional: make viscosity visually distinguishable in videos.
    if entry.parameter_name == "water_viscosity":
        mu = float(entry.parameter_value)
        if mu >= 0.02:
            mod.LIQUID_COLOR = (1.0, 0.58, 0.08, 1.0)
        elif mu >= 0.005:
            mod.LIQUID_COLOR = (0.54, 0.45, 0.86, 1.0)
        else:
            mod.LIQUID_COLOR = (0.25, 0.55, 0.95, 1.0)

    return constant_name


def collect_resolved_config(mod, entry: SweepEntry) -> dict[str, Any]:
    keys = [
        "FRAME_RATE",
        "MICROSTEPS_PER_FRAME",
        "VIDEO_DT",
        "PHYSICS_DT",
        "SIM_SUBSTEPS",
        "WATER_PARTICLE_SIZE",
        "WATER_DENSITY",
        "WATER_VISCOSITY",
        "LIQUID_SURFACE_TENSION",
        "LIQUID_COLOR",
        "WATER_FILL_FRACTION",
        "GLASS_COUP_FRICTION",
        "GLASS_COUP_SOFTNESS",
        "GLASS_SDF_CELL_SIZE",
        "EMISSION_OVERFILL_FACTOR",
        "SURFACE_HOLD_SECONDS",
        "LIFT_SECONDS",
        "PRE_POUR_HOLD_SECONDS",
        "TILT_SECONDS",
        "POUR_HOLD_SECONDS",
        "RETURN_SECONDS",
        "PLACE_BACK_SECONDS",
        "FINAL_HOLD_SECONDS",
        "MAX_TILT_DEG",
        "POUR_POSE_FRACTION",
        "VIDEO_NUM_FRAMES",
        "VIDEO_RESOLUTION",
        "VIDEO_FPS",
        "SETTLE_BAKE_SECONDS",
    ]

    cfg = {
        "scene_family": entry.scene_family,
        "scene_id": entry.scene_id,
        "seed": entry.seed,
        "num_frames": entry.num_frames,
        "sweep_parameter": entry.parameter_name,
        "sweep_value": entry.parameter_value,
        "settled_cache": entry.cache_path,
    }

    for key in keys:
        value = getattr(mod, key, None)
        if isinstance(value, Path):
            value = str(value)
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, tuple):
            value = list(value)
        cfg[key] = value

    return cfg


def save_result_npz(result: Any, path: Path) -> None:
    np.savez_compressed(
        path,
        initial_particle_positions=result.initial_particle_positions,
        final_particle_positions=result.final_particle_positions,
        final_pourer_position=result.final_pourer_position,
        final_pourer_quat_wxyz=result.final_pourer_quat_wxyz,
        final_tilt_degrees=np.array(result.final_tilt_degrees, dtype=np.float32),
        max_tilt_degrees=np.array(result.max_tilt_degrees, dtype=np.float32),
        initial_particle_count=np.array(result.initial_particle_count, dtype=np.int32),
        final_particles_in_pourer=np.array(result.final_particles_in_pourer, dtype=np.int32),
        final_particles_in_receiver=np.array(result.final_particles_in_receiver, dtype=np.int32),
        final_live_particles=np.array(result.final_live_particles, dtype=np.int32),
        max_glass_solid_particles=np.array(result.max_glass_solid_particles, dtype=np.int32),
        max_pourer_solid_particles=np.array(result.max_pourer_solid_particles, dtype=np.int32),
        max_receiver_solid_particles=np.array(result.max_receiver_solid_particles, dtype=np.int32),
        max_pourer_base_particles=np.array(result.max_pourer_base_particles, dtype=np.int32),
    )


def result_metrics(result: Any) -> dict[str, Any]:
    return {
        "initial_particle_count": int(result.initial_particle_count),
        "final_live_particles": int(result.final_live_particles),
        "final_particles_in_pourer": int(result.final_particles_in_pourer),
        "final_particles_in_receiver": int(result.final_particles_in_receiver),
        "pourer_fraction": float(result.pourer_fraction),
        "receiver_fraction": float(result.receiver_fraction),
        "lost_or_stashed_particles": int(result.initial_particle_count - result.final_live_particles),
        "final_tilt_degrees": float(result.final_tilt_degrees),
        "max_tilt_degrees": float(result.max_tilt_degrees),
        "max_glass_solid_particles": int(result.max_glass_solid_particles),
        "max_pourer_solid_particles": int(result.max_pourer_solid_particles),
        "max_receiver_solid_particles": int(result.max_receiver_solid_particles),
        "max_pourer_base_particles": int(result.max_pourer_base_particles),
        "successful_transfer_rough": bool(result.receiver_fraction > 0.25),
        "mostly_retained_rough": bool(result.pourer_fraction > 0.50),
    }


def run_single(args: argparse.Namespace) -> int:
    entry = SweepEntry(
        scene_family=args.scene_family,
        scene_id=args.scene_id,
        causal_factor=args.causal_factor,
        parameter_name=args.parameter_name,
        parameter_value=args.parameter_value,
        seed=args.seed,
        num_frames=args.num_frames,
        output_dir=args.output_dir,
        video_path=args.video_path if args.video_path != "None" else None,
        result_path=args.result_path,
        metadata_path=args.metadata_path,
        cache_path=args.cache_path,
    )

    out_dir = Path(entry.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["PYTHONHASHSEED"] = str(entry.seed)

    mod = load_scene_module()
    constant_name = configure_module_for_entry(mod, entry)

    # If num_frames was not explicitly passed by the parent, use the module's duration.
    num_frames = entry.num_frames
    if num_frames <= 0:
        num_frames = int(mod.VIDEO_NUM_FRAMES)

    # For one-variable sweeps, rebaking is safer: fill fraction, viscosity, density,
    # and coupling parameters can all change the settled initial fluid state.
    rebake = not args.no_rebake

    if entry.video_path is not None:
        mod.render_video(
            output_path=entry.video_path,
            num_frames=num_frames,
            settled_cache=Path(entry.cache_path),
            rebake=rebake,
        )
        result = mod.run_simulation(
            num_frames=num_frames,
            settled_cache=Path(entry.cache_path),
            rebake=False,
        )
    else:
        result = mod.run_simulation(
            num_frames=num_frames,
            settled_cache=Path(entry.cache_path),
            rebake=rebake,
        )

    save_result_npz(result, Path(entry.result_path))

    resolved_config = collect_resolved_config(mod, entry)
    resolved_config["num_frames_effective"] = num_frames
    resolved_config_path = out_dir / "resolved_config.json"
    resolved_config_path.write_text(json.dumps(resolved_config, indent=2, sort_keys=True))

    metadata = {
        **asdict(entry),
        "num_frames_effective": num_frames,
        "resolved_config_path": str(resolved_config_path),
        "patched_module_constant": constant_name,
        "metrics": result_metrics(result),
        "notes": [
            "This entry varies exactly one causal/action factor from the default robotic-pour scene.",
            "result.npz contains initial/final particle states and scalar transfer metrics.",
            "Full per-frame fluid trajectories are not logged yet; add that inside robotic_arm_pour_genesis.py for trajectory tracking.",
        ],
    }

    Path(entry.metadata_path).write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


def run_parent(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    entries = build_entries(args)

    if not entries:
        raise RuntimeError("No sweep entries generated.")

    write_manifest(entries, output_root)

    if args.dry_run:
        print(f"Dry run: generated {len(entries)} planned entries.")
        return 0

    for i, entry in enumerate(entries, start=1):
        print(f"[{i}/{len(entries)}] Running {entry.scene_id}")

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_single",
            "--scene-family", entry.scene_family,
            "--scene-id", entry.scene_id,
            "--causal-factor", entry.causal_factor,
            "--parameter-name", entry.parameter_name,
            "--parameter-value", str(entry.parameter_value),
            "--seed", str(entry.seed),
            "--num-frames", str(entry.num_frames),
            "--output-dir", entry.output_dir,
            "--video-path", str(entry.video_path),
            "--result-path", entry.result_path,
            "--metadata-path", entry.metadata_path,
            "--cache-path", entry.cache_path,
        ]

        if args.no_rebake:
            cmd.append("--no-rebake")

        subprocess.run(cmd, check=True)

    print(f"Finished {len(entries)} entries under {output_root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--sweep-param", choices=sorted(DEFAULT_GRIDS), default="water_viscosity")
    parser.add_argument("--values", type=str, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])

    # Use -1 to mean "use module default VIDEO_NUM_FRAMES after patching constants."
    parser.add_argument("--num-frames", type=int, default=-1)
    parser.add_argument("--output-root", type=str, default="outputs/datasets/robotic_pour_sweep")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-rebake",
        action="store_true",
        help="Reuse cache if present. Faster, but less safe for controlled physical sweeps.",
    )

    # Internal single-entry mode.
    parser.add_argument("--_single", action="store_true")
    parser.add_argument("--scene-family", default="")
    parser.add_argument("--scene-id", default="")
    parser.add_argument("--causal-factor", default="")
    parser.add_argument("--parameter-name", default="")
    parser.add_argument("--parameter-value", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--video-path", default="None")
    parser.add_argument("--result-path", default="")
    parser.add_argument("--metadata-path", default="")
    parser.add_argument("--cache-path", default="")

    args = parser.parse_args(argv)

    if args._single:
        return run_single(args)

    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())