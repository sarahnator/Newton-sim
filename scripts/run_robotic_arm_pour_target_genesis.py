"""CLI entry point for the Genesis robot-arm synthetic target-liquid variant.

Run from the ``genesis-sim`` conda env. The target liquid uses a hidden
intermediate viscosity for Bayesian-optimization calibration demos.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")


def _load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "src" / "npm_sim" / "robotic_arm_pour_genesis.py"
    spec = importlib.util.spec_from_file_location("robotic_arm_pour_genesis_target", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.configure_target_variant()
    return module


def main(argv: list[str] | None = None) -> int:
    mod = _load_module()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-frames", type=int, default=mod.VIDEO_NUM_FRAMES)
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/robotic_arm_pour_target_genesis.mp4",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip video rendering; run the scene and print particle transfer metrics.",
    )
    parser.add_argument(
        "--rebake",
        action="store_true",
        help="Rebuild the settled starting target-liquid cache before running.",
    )
    args = parser.parse_args(argv)

    if args.num_frames <= 0:
        parser.error("--num-frames must be positive")

    if args.no_video:
        result = mod.run_simulation(num_frames=args.num_frames, rebake=args.rebake)
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

    output = mod.render_video(
        output_path=args.output_path,
        num_frames=args.num_frames,
        rebake=args.rebake,
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
