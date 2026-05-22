#!/usr/bin/env python3
"""
Generate one-variable-at-a-time benchmark sweeps for src/npm_sim/pendulum_genesis.py.

Examples:

  # Dry run
  uv run python scripts/generate_pendulum_sweep.py \
    --sweep-param length --dry-run

  # No video
  uv run python scripts/generate_pendulum_sweep.py \
    --sweep-param damping --seeds 0 1 2 --num-steps 1200 --no-video

  # With video
  uv run python scripts/generate_pendulum_sweep.py \
    --sweep-param length --seeds 0 --num-steps 600 --render-video
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


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE_MODULE_PATH = REPO_ROOT / "src" / "npm_sim" / "pendulum_genesis.py"


DEFAULT_GRIDS = {
    # Period should scale as sqrt(length).
    "length": [0.35, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00],

    # Ideal pendulum period should be approximately invariant to mass.
    "mass": [0.05, 0.10, 0.25, 0.50, 1.00, 2.00, 5.00],

    # Amplitude should decay faster as damping increases.
    "damping": [0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10],

    # Period should scale as 1 / sqrt(g).
    "gravity": [1.62, 3.71, 5.0, 9.81, 12.0, 15.0, 24.79],

    # Larger angles move away from the small-angle analytic approximation.
    "theta0_deg": [2.0, 5.0, 10.0, 20.0, 35.0, 60.0, 90.0],
}


PARAM_TO_CONFIG_FIELD = {
    "length": "length",
    "mass": "mass",
    "damping": "damping",
    "gravity": "gravity",
    "theta0_deg": "theta0_deg",
}


@dataclass(frozen=True)
class SweepEntry:
    scene_family: str
    scene_id: str
    causal_factor: str
    parameter_name: str
    parameter_value: float
    seed: int
    num_steps: int
    output_dir: str
    result_path: str
    metadata_path: str
    video_path: str | None


def stable_id(parts: list[Any]) -> str:
    text = json.dumps(parts, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def load_scene_module():
    spec = importlib.util.spec_from_file_location("pendulum_genesis", SCENE_MODULE_PATH)
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
            sid = stable_id(["pendulum", args.sweep_param, value, seed, args.num_steps])
            scene_id = f"pendulum_{args.sweep_param}_{sid}"
            out_dir = Path(args.output_root) / args.sweep_param / scene_id

            video_path = str(out_dir / "pendulum.mp4") if args.render_video else None

            entries.append(
                SweepEntry(
                    scene_family="pendulum",
                    scene_id=scene_id,
                    causal_factor=args.sweep_param,
                    parameter_name=args.sweep_param,
                    parameter_value=float(value),
                    seed=int(seed),
                    num_steps=int(args.num_steps),
                    output_dir=str(out_dir),
                    result_path=str(out_dir / "trajectory.npz"),
                    metadata_path=str(out_dir / "metadata.json"),
                    video_path=video_path,
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


def collect_resolved_config(cfg: Any, entry: SweepEntry) -> dict[str, Any]:
    return {
        "scene_family": entry.scene_family,
        "scene_id": entry.scene_id,
        "seed": entry.seed,
        "num_steps": entry.num_steps,
        "sweep_parameter": entry.parameter_name,
        "sweep_value": entry.parameter_value,
        **asdict(cfg),
    }


def run_single(args: argparse.Namespace) -> int:
    entry = SweepEntry(
        scene_family=args.scene_family,
        scene_id=args.scene_id,
        causal_factor=args.causal_factor,
        parameter_name=args.parameter_name,
        parameter_value=args.parameter_value,
        seed=args.seed,
        num_steps=args.num_steps,
        output_dir=args.output_dir,
        result_path=args.result_path,
        metadata_path=args.metadata_path,
        video_path=args.video_path if args.video_path != "None" else None,
    )

    out_dir = Path(entry.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["PYTHONHASHSEED"] = str(entry.seed)

    mod = load_scene_module()
    cfg_kwargs = {
        "length": args.default_length,
        "mass": args.default_mass,
        "bob_radius": args.default_bob_radius,
        "damping": args.default_damping,
        "theta0_deg": args.default_theta0_deg,
        "omega0": args.default_omega0,
        "gravity": args.default_gravity,
        "dt": args.dt,
        "substeps": args.substeps,
        "num_steps": entry.num_steps,
        "seed": entry.seed,
        "backend": args.backend,
        "show_viewer": False,
        "render_video": entry.video_path is not None,
        "video_fps": args.video_fps,
        "video_width": args.video_width,
        "video_height": args.video_height,
    }

    cfg_field = PARAM_TO_CONFIG_FIELD[entry.parameter_name]
    cfg_kwargs[cfg_field] = entry.parameter_value

    cfg = mod.PendulumConfig(**cfg_kwargs)
    metadata = mod.run_simulation(cfg, out_dir)

    resolved_config = collect_resolved_config(cfg, entry)
    resolved_config_path = out_dir / "resolved_config.json"
    resolved_config_path.write_text(json.dumps(resolved_config, indent=2, sort_keys=True))

    # Patch benchmark bookkeeping into the scene metadata produced by pendulum_genesis.py.
    metadata.update(
        {
            **asdict(entry),
            "resolved_config_path": str(resolved_config_path),
            "patched_config_field": cfg_field,
            "notes": [
                "This entry varies exactly one causal factor from the default pendulum config.",
                "trajectory.npz contains t, q, qdot, theta_small_angle_ref, and bob_pos.",
            ],
        }
    )

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
            "--num-steps", str(entry.num_steps),
            "--output-dir", entry.output_dir,
            "--result-path", entry.result_path,
            "--metadata-path", entry.metadata_path,
            "--video-path", str(entry.video_path),

            "--backend", args.backend,
            "--dt", str(args.dt),
            "--substeps", str(args.substeps),
            "--video-fps", str(args.video_fps),
            "--video-width", str(args.video_width),
            "--video-height", str(args.video_height),

            "--default-length", str(args.default_length),
            "--default-mass", str(args.default_mass),
            "--default-bob-radius", str(args.default_bob_radius),
            "--default-damping", str(args.default_damping),
            "--default-theta0-deg", str(args.default_theta0_deg),
            "--default-omega0", str(args.default_omega0),
            "--default-gravity", str(args.default_gravity),
        ]

        subprocess.run(cmd, check=True)

    print(f"Finished {len(entries)} entries under {output_root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--sweep-param", choices=sorted(DEFAULT_GRIDS), default="length")
    parser.add_argument("--values", type=str, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--num-steps", type=int, default=1200)
    parser.add_argument("--output-root", type=str, default="outputs/datasets/pendulum_sweep")
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--backend", type=str, default="cpu", choices=["cpu", "gpu", "cuda", "metal"])
    parser.add_argument("--dt", type=float, default=1.0 / 240.0)
    parser.add_argument("--substeps", type=int, default=4)

    parser.add_argument("--video-fps", type=int, default=60)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)

    # Defaults for non-swept variables.
    parser.add_argument("--default-length", type=float, default=1.0)
    parser.add_argument("--default-mass", type=float, default=1.0)
    parser.add_argument("--default-bob-radius", type=float, default=0.06)
    parser.add_argument("--default-damping", type=float, default=0.02)
    parser.add_argument("--default-theta0-deg", type=float, default=30.0)
    parser.add_argument("--default-omega0", type=float, default=0.0)
    parser.add_argument("--default-gravity", type=float, default=9.81)

    # Internal single-entry mode.
    parser.add_argument("--_single", action="store_true")
    parser.add_argument("--scene-family", default="")
    parser.add_argument("--scene-id", default="")
    parser.add_argument("--causal-factor", default="")
    parser.add_argument("--parameter-name", default="")
    parser.add_argument("--parameter-value", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--result-path", default="")
    parser.add_argument("--metadata-path", default="")
    parser.add_argument("--video-path", default="None")

    args = parser.parse_args(argv)

    if args.num_steps <= 0:
        parser.error("--num-steps must be positive")

    if args._single:
        return run_single(args)

    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())