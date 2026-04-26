from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from npm_sim.materials import MATERIALS
from npm_sim.ramp_tower import run_simulation as run_jelly_simulation
from npm_sim.rigid_ramp_cup import run_simulation as run_rigid_cup_simulation
from npm_sim.rigid_ramp_tower import run_simulation as run_rigid_simulation
from npm_sim.roboarm_wall import FRICTION_MODE_HIGH, FRICTION_MODE_LOW
from npm_sim.roboarm_wall import PUSH_MODE_CENTER, PUSH_MODE_HIGH, PUSH_MODE_LOW
from npm_sim.roboarm_wall import run_simulation as run_roboarm_wall_simulation

JELLY_VARIANT_WALL_COUNTS = {
    "jelly": 2,
    "jelly-single": 1,
    "jelly-domino": 2,
}


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--variant",
        type=str,
        default="jelly-domino",
        choices=["jelly", "jelly-single", "jelly-domino", "rigid", "rigid-cup", "roboarm-wall"],
        help="Simulation variant.",
    )
    parser.add_argument(
        "--ball-material",
        type=str,
        default="steel",
        choices=sorted(MATERIALS),
        help="Material preset for the moving ball or robot pusher.",
    )
    parser.add_argument(
        "--cube-material",
        type=str,
        default="wood",
        choices=sorted(MATERIALS),
        help="Material preset for the target and static ground surfaces.",
    )
    parser.add_argument(
        "--viewer",
        type=str,
        default="gl",
        choices=["gl", "usd", "null"],
        help="Viewer mode.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=240,
        help="Frame count for null and usd viewers.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="USD output path. Required when --viewer usd is selected.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional Warp device override, for example cpu or cuda:0.",
    )
    parser.add_argument(
        "--roboarm-push-height",
        type=str,
        default=PUSH_MODE_HIGH,
        choices=[PUSH_MODE_HIGH, PUSH_MODE_LOW, PUSH_MODE_CENTER],
        help="Contact height mode for the roboarm-wall variant.",
    )
    parser.add_argument(
        "--roboarm-friction",
        type=str,
        default=None,
        choices=[FRICTION_MODE_HIGH, FRICTION_MODE_LOW],
        help="Required friction mode for the center-push roboarm-wall variant.",
    )
    return parser


def main(argv: list[str] | None = None):
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.num_frames <= 0:
        parser.error("--num-frames must be positive")
    if args.viewer == "usd" and args.output_path is None:
        parser.error("--output-path is required when using --viewer usd")

    if args.variant == "rigid":
        return run_rigid_simulation(
            ball_material=args.ball_material,
            cube_material=args.cube_material,
            viewer=args.viewer,
            num_frames=args.num_frames,
            output_path=args.output_path,
            device=args.device,
        )

    if args.variant == "rigid-cup":
        return run_rigid_cup_simulation(
            ball_material=args.ball_material,
            cube_material=args.cube_material,
            viewer=args.viewer,
            num_frames=args.num_frames,
            output_path=args.output_path,
            device=args.device,
        )

    if args.variant == "roboarm-wall":
        if args.roboarm_push_height == PUSH_MODE_CENTER:
            if args.roboarm_friction is None:
                parser.error("--roboarm-friction is required when --roboarm-push-height center")
        elif args.roboarm_friction is not None:
            parser.error("--roboarm-friction is only valid when --roboarm-push-height center")
        return run_roboarm_wall_simulation(
            arm_material=args.ball_material,
            wall_material=args.cube_material,
            push_mode=args.roboarm_push_height,
            friction_mode=args.roboarm_friction,
            viewer=args.viewer,
            num_frames=args.num_frames,
            output_path=args.output_path,
            device=args.device,
        )

    return run_jelly_simulation(
        ball_material=args.ball_material,
        cube_material=args.cube_material,
        wall_count=JELLY_VARIANT_WALL_COUNTS[args.variant],
        viewer=args.viewer,
        num_frames=args.num_frames,
        output_path=args.output_path,
        device=args.device,
    )


if __name__ == "__main__":
    main()
