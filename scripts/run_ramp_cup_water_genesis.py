"""CLI entry point for the Genesis rigid-cup + SPH-water variant.

Run from the ``genesis-sim`` conda env — the main ``npm_sim`` package imports
Newton/Warp at top level and cannot be imported alongside Genesis. We therefore
import the module directly by path and avoid ``npm_sim.__init__``.

Typical usage:
    uv run python scripts/run_ramp_cup_water_genesis.py \\
        --num-frames 240 --output-path outputs/rigid_ramp_cup_water_genesis.mp4
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

# Pin Genesis to the idle RTX 5090 (CUDA_VISIBLE_DEVICES=1) so runs don't
# compete with other GPU workloads on cuda:0. Must be set BEFORE any import
# that touches CUDA (torch, genesis). Override by exporting
# CUDA_VISIBLE_DEVICES before invoking this script.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")


def _load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "src" / "npm_sim" / "rigid_ramp_cup_water_genesis.py"
    spec = importlib.util.spec_from_file_location("rigid_ramp_cup_water_genesis", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-frames", type=int, default=240)
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/rigid_ramp_cup_water_genesis.mp4",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip video rendering; run the scene and print the SimulationResult only.",
    )
    args = parser.parse_args(argv)

    mod = _load_module()

    if args.no_video:
        result = mod.run_simulation(num_frames=args.num_frames)
        print("initial ball:", result.initial_ball_position)
        print("final ball:  ", result.final_ball_position)
        print("cup move:    ", result.final_cup_position - result.initial_cup_position)
        print(
            "particles:   ",
            result.final_particle_positions.shape[0],
            "final z_top:",
            float(result.final_particle_positions[:, 2].max()),
        )
        return 0

    output = mod.render_video(output_path=args.output_path, num_frames=args.num_frames)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
