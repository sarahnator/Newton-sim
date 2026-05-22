#!/usr/bin/env python3
"""
Generate a one-variable-at-a-time benchmark sweep for the Genesis ramp/ball/cup/water scene.

Usage examples:

  # Dry-run manifest only
  uv run python scripts/generate_ramp_cup_sweep.py \
    --sweep-param ball_density --dry-run

  # Generate no-video benchmark entries
  uv run python scripts/generate_ramp_cup_sweep.py \
    --sweep-param ball_density --no-video --num-frames 240

  # Generate MP4s + per-entry metadata
  uv run python scripts/generate_ramp_cup_sweep.py \
    --sweep-param water_viscosity --num-frames 240

Notes:
  - The parent process launches one subprocess per config because Genesis initialization
    is often not cleanly reusable across many scenes in the same Python process.
  - This is a practical bridge script. The cleaner long-term fix is to refactor the
    scene module to accept a typed config object.
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


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE_MODULE_PATH = REPO_ROOT / "src" / "npm_sim" / "rigid_ramp_cup_water_genesis.py"


DEFAULT_GRIDS = {
    # Rigid impact / momentum transfer
    "ball_density": [300.0, 650.0, 1000.0, 2700.0, 4500.0, 7850.0, 12000.0],

    # Cup inertia / stability
    "cup_density": [250.0, 450.0, 650.0, 1000.0, 1450.0, 2500.0, 5000.0],

    # Fluid transfer / slosh / coupling
    "water_viscosity": [1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1],

    # Cup-fluid center of mass / spillage threshold
    "fill_fraction": [0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95],

    # Impact energy through release geometry
    "ramp_angle_deg": [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0],
}


PARAM_TO_MODULE_CONSTANT = {
    "ball_density": "BALL_DENSITY",
    "cup_density": "CUP_DENSITY",
    "water_viscosity": "WATER_VISCOSITY",
    "fill_fraction": "TARGET_FILL_FRACTION",
    "ramp_angle_deg": "RAMP_ANGLE_DEG",
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


def stable_id(parts: list[Any]) -> str:
    text = json.dumps(parts, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def load_scene_module():
    spec = importlib.util.spec_from_file_location("rigid_ramp_cup_water_genesis", SCENE_MODULE_PATH)
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
    values = parse_values(args)
    entries: list[SweepEntry] = []

    for value in values:
        for seed in args.seeds:
            sid = stable_id(["ramp_ball_cup_water", args.sweep_param, value, seed, args.num_frames])
            scene_id = f"ramp_cup_{args.sweep_param}_{sid}"
            out_dir = Path(args.output_root) / args.sweep_param / scene_id

            video_path = None
            if not args.no_video:
                video_path = str(out_dir / "video.mp4")

            entries.append(
                SweepEntry(
                    scene_family="ramp_ball_cup_water",
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


def _np_scalar(result_npz: dict[str, Any], key: str) -> float | None:
    if key not in result_npz:
        return None
    try:
        return float(np.asarray(result_npz[key]).item())
    except Exception:
        return None


def classify_outcome(result_npz: dict[str, Any]) -> dict[str, Any]:
    initial_cup = result_npz["initial_cup_position"]
    final_cup = result_npz["final_cup_position"]
    initial_ball = result_npz["initial_ball_position"]
    final_ball = result_npz["final_ball_position"]
    final_particles = result_npz["final_particle_positions"]

    cup_displacement = final_cup - initial_cup
    ball_displacement = final_ball - initial_ball

    particle_z = final_particles[:, 2] if final_particles.size else []

    metrics = {
        "cup_displacement_norm": float(np.linalg.norm(cup_displacement)),
        "cup_delta_x": float(cup_displacement[0]),
        "cup_delta_y": float(cup_displacement[1]),
        "cup_delta_z": float(cup_displacement[2]),
        "ball_displacement_norm": float(np.linalg.norm(ball_displacement)),
        "num_fluid_particles": int(final_particles.shape[0]),
        "final_particle_z_max": float(max(particle_z)) if len(particle_z) else None,
        "final_particle_z_min": float(min(particle_z)) if len(particle_z) else None,
    }

    # Displacement labels. These answer "did it move?", not "did it fall?"
    metrics["cup_moved"] = metrics["cup_displacement_norm"] > 0.05
    metrics["large_cup_motion"] = metrics["cup_displacement_norm"] > 0.20

    # Orientation / tilt labels. These are the important labels for:
    #   "Will the cup fall over?"
    #
    # Convention:
    #   0 deg   = upright
    #   90 deg  = on its side
    #   180 deg = upside down
    final_tilt = _np_scalar(result_npz, "final_cup_tilt_degrees")
    max_tilt = _np_scalar(result_npz, "max_cup_tilt_degrees")

    if final_tilt is not None:
        metrics["final_cup_tilt_degrees"] = final_tilt

    if max_tilt is not None:
        metrics["max_cup_tilt_degrees"] = max_tilt

    tilt_for_label = max_tilt if max_tilt is not None else final_tilt

    if tilt_for_label is not None:
        metrics["cup_fell_over"] = bool(tilt_for_label >= 60.0)
        metrics["cup_on_side"] = bool(tilt_for_label >= 75.0)
        metrics["cup_inverted"] = bool(tilt_for_label >= 120.0)
    else:
        # This means the scene result did not expose tilt fields yet.
        # Do not infer fall-over from displacement alone.
        metrics["cup_fell_over"] = None
        metrics["cup_on_side"] = None
        metrics["cup_inverted"] = None

    return metrics


def collect_resolved_config(mod, entry: SweepEntry) -> dict[str, Any]:
    """
    Snapshot all constants that define the experiment.
    Add more fields as the scene becomes more configurable.
    """
    return {
        "scene_family": entry.scene_family,
        "scene_id": entry.scene_id,
        "seed": entry.seed,
        "num_frames": entry.num_frames,

        # Simulation timing
        "FRAME_RATE": getattr(mod, "FRAME_RATE", None),
        "SIM_SUBSTEPS": getattr(mod, "SIM_SUBSTEPS", None),
        "VIDEO_RESOLUTION": getattr(mod, "VIDEO_RESOLUTION", None),
        "VIDEO_FPS": getattr(mod, "VIDEO_FPS", None),

        # Ramp / geometry
        "RAMP_ANGLE_DEG": getattr(mod, "RAMP_ANGLE_DEG", None),
        "RAMP_LENGTH": getattr(mod, "RAMP_LENGTH", None),
        "RAMP_WIDTH": getattr(mod, "RAMP_WIDTH", None),
        "RAMP_THICKNESS": getattr(mod, "RAMP_THICKNESS", None),

        # Ball
        "BALL_RADIUS": getattr(mod, "BALL_RADIUS", None),
        "BALL_DENSITY": getattr(mod, "BALL_DENSITY", None),
        "BALL_START_MARGIN": getattr(mod, "BALL_START_MARGIN", None),
        "BALL_LATERAL_OFFSET": getattr(mod, "BALL_LATERAL_OFFSET", None),
        "BALL_CLEARANCE": getattr(mod, "BALL_CLEARANCE", None),

        # Cup
        "CUP_CENTER_X": getattr(mod, "CUP_CENTER_X", None),
        "CUP_CENTER_Y": getattr(mod, "CUP_CENTER_Y", None),
        "CUP_HEIGHT": getattr(mod, "CUP_HEIGHT", None),
        "CUP_BOTTOM_RADIUS": getattr(mod, "CUP_BOTTOM_RADIUS", None),
        "CUP_TOP_RADIUS": getattr(mod, "CUP_TOP_RADIUS", None),
        "CUP_WALL_THICKNESS": getattr(mod, "CUP_WALL_THICKNESS", None),
        "CUP_BASE_THICKNESS": getattr(mod, "CUP_BASE_THICKNESS", None),
        "CUP_DENSITY": getattr(mod, "CUP_DENSITY", None),

        # Fluid
        "WATER_DENSITY": getattr(mod, "WATER_DENSITY", None),
        "WATER_VISCOSITY": getattr(mod, "WATER_VISCOSITY", None),
        "WATER_PARTICLE_SIZE": getattr(mod, "WATER_PARTICLE_SIZE", None),
        "TARGET_FILL_FRACTION": getattr(mod, "TARGET_FILL_FRACTION", None),
        "EMISSION_OVERFILL_FACTOR": getattr(mod, "EMISSION_OVERFILL_FACTOR", None),

        # Benchmark bookkeeping
        "sweep_parameter": entry.parameter_name,
        "sweep_value": entry.parameter_value,
    }


def simulation_result_to_npz_kwargs(result: Any) -> dict[str, Any]:
    """
    Convert SimulationResult into result.npz fields.

    The hasattr checks make this compatible with older scene outputs, but once
    rigid_ramp_cup_water_genesis.py is updated, the tilt/quaternion fields
    should always be present.
    """
    save_kwargs = {
        "initial_ball_position": result.initial_ball_position,
        "final_ball_position": result.final_ball_position,
        "initial_cup_position": result.initial_cup_position,
        "final_cup_position": result.final_cup_position,
        "initial_particle_positions": result.initial_particle_positions,
        "final_particle_positions": result.final_particle_positions,
    }

    optional_fields = [
        "initial_cup_quat_wxyz",
        "final_cup_quat_wxyz",
        "max_cup_tilt_degrees",
        "final_cup_tilt_degrees",
    ]

    for field in optional_fields:
        if hasattr(result, field):
            value = getattr(result, field)
            if isinstance(value, (float, int, np.floating, np.integer)):
                value = np.array(value, dtype=np.float32)
            save_kwargs[field] = value

    return save_kwargs


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
    )

    out_dir = Path(entry.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility knobs. Genesis may also need its own seed call if exposed.
    os.environ["PYTHONHASHSEED"] = str(entry.seed)

    mod = load_scene_module()

    constant_name = PARAM_TO_MODULE_CONSTANT[entry.parameter_name]
    setattr(mod, constant_name, entry.parameter_value)

    # Avoid reusing settled-water cache across fill/viscosity sweeps unless you are sure
    # it is valid. Give each config its own cache.
    if hasattr(mod, "SETTLED_PARTICLES_CACHE"):
        mod.SETTLED_PARTICLES_CACHE = out_dir / "settled_water.npy"

    if entry.video_path is not None:
        mod.render_video(
            output_path=entry.video_path,
            num_frames=entry.num_frames,
            settled_cache=mod.SETTLED_PARTICLES_CACHE,
            rebake=True,
        )
        result = mod.run_simulation(num_frames=entry.num_frames)
    else:
        result = mod.run_simulation(num_frames=entry.num_frames)

    resolved_config = collect_resolved_config(mod, entry)

    resolved_config_path = Path(entry.output_dir) / "resolved_config.json"
    with resolved_config_path.open("w") as f:
        json.dump(resolved_config, f, indent=2, sort_keys=True)

    np.savez_compressed(
        entry.result_path,
        **simulation_result_to_npz_kwargs(result),
    )

    loaded = dict(np.load(entry.result_path, allow_pickle=True))
    metrics = classify_outcome(loaded)

    metadata = {
        **asdict(entry),
        "resolved_config_path": str(resolved_config_path),
        "patched_module_constant": constant_name,
        "metrics": metrics,
        "notes": [
            "This entry varies exactly one causal factor from the default scene constants.",
            "Cup fall-over is labeled using max_cup_tilt_degrees >= 60 degrees when tilt is available.",
            "result.npz contains initial/final positions, particle positions, and cup orientation/tilt fields when exposed by the scene.",
        ],
    }

    with Path(entry.metadata_path).open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

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
        ]

        subprocess.run(cmd, check=True)

    print(f"Finished {len(entries)} entries under {output_root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--sweep-param",
        choices=sorted(DEFAULT_GRIDS.keys()),
        default="ball_density",
        help="The single causal factor to vary.",
    )
    parser.add_argument(
        "--values",
        type=str,
        default=None,
        help="Comma-separated override values, e.g. '500,1000,2000'.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--num-frames", type=int, default=240)
    parser.add_argument("--output-root", type=str, default="outputs/datasets/ramp_cup_sweep")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

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

    args = parser.parse_args(argv)

    if args.num_frames <= 0:
        parser.error("--num-frames must be positive")

    if args._single:
        return run_single(args)

    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())