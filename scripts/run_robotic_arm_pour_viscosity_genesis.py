"""Run the Genesis robot-arm pour with a chosen liquid viscosity.

This is the reusable entry point for calibration demos: Bayesian optimization
can estimate a viscosity, then this script renders the best-match simulation.
Run from the ``genesis-sim`` conda env.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETTLED_CACHE = (
    ROOT
    / "outputs"
    / "_genesis"
    / "robotic_arm_pickup_pour_base050_p006_fill080_over1405_clear006_fric005_soft0015_pose080_slow_fillet012_micro084_sdf0025_align0_corrbase_settled_water.npy"
)


def _load_module():
    path = ROOT / "src" / "npm_sim" / "robotic_arm_pour_genesis.py"
    spec = importlib.util.spec_from_file_location("robotic_arm_pour_genesis_custom_viscosity", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _configure_viscosity(mod, viscosity: float, cache_path: Path) -> None:
    mod.WATER_VISCOSITY = float(viscosity)
    mod.LIQUID_COLOR = (0.35, 0.62, 0.90, 1.0)
    mod.SETTLED_PARTICLES_CACHE = cache_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--viscosity", type=float, required=True)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_SETTLED_CACHE)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip video rendering; run the scene and print particle transfer metrics.",
    )
    parser.add_argument(
        "--rebake",
        action="store_true",
        help="Rebuild the chosen settled starting cache before running.",
    )
    args = parser.parse_args(argv)

    if args.viscosity <= 0.0:
        parser.error("--viscosity must be positive")

    mod = _load_module()
    _configure_viscosity(mod, args.viscosity, args.cache_path)
    num_frames = mod.VIDEO_NUM_FRAMES if args.num_frames is None else args.num_frames
    if num_frames <= 0:
        parser.error("--num-frames must be positive")

    if args.no_video:
        result = mod.run_simulation(
            num_frames=num_frames,
            settled_cache=args.cache_path,
            rebake=args.rebake,
        )
        print("viscosity:        ", f"{args.viscosity:.6g}")
        print("particles initial:", result.initial_particle_count)
        print("particles live:   ", result.final_live_particles)
        print("in pourer:        ", result.final_particles_in_pourer, f"({result.pourer_fraction:.2%})")
        print("in receiver:      ", result.final_particles_in_receiver, f"({result.receiver_fraction:.2%})")
        print("max tilt deg:     ", f"{result.max_tilt_degrees:.1f}")
        print("final tilt deg:   ", f"{result.final_tilt_degrees:.1f}")
        print("solid violations: ", result.max_glass_solid_particles)
        print("  upper glass:    ", result.max_pourer_solid_particles)
        print("  upper base:     ", result.max_pourer_base_particles)
        print("  receiver glass: ", result.max_receiver_solid_particles)
        return 0

    output_path = args.output_path or f"outputs/robotic_arm_pour_mu_{args.viscosity:.4g}.mp4"
    output = mod.render_video(
        output_path=output_path,
        num_frames=num_frames,
        settled_cache=args.cache_path,
        rebake=args.rebake,
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
