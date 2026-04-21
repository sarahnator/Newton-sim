from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from npm_sim.materials import MATERIALS
from npm_sim.ramp_tower import VIDEO_NUM_FRAMES as JELLY_VIDEO_NUM_FRAMES
from npm_sim.ramp_tower import render_video as render_jelly_video
from npm_sim.rigid_ramp_cup import render_video as render_rigid_cup_video
from npm_sim.rigid_ramp_tower import render_video as render_rigid_video

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
        choices=["jelly", "jelly-single", "jelly-domino", "rigid", "rigid-cup"],
        help="Simulation variant.",
    )
    parser.add_argument(
        "--ball-material",
        type=str,
        default="steel",
        choices=sorted(MATERIALS),
        help="Material preset for the moving ball.",
    )
    parser.add_argument(
        "--cube-material",
        type=str,
        default="wood",
        choices=sorted(MATERIALS),
        help="Material preset for the target and static ground surfaces.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Number of simulation frames to render into the video. Uses a variant-specific default when omitted.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="MP4 output path. If omitted, a variant-specific default is used.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional Warp device override, for example cpu or cuda:0.",
    )
    return parser


def main(argv: list[str] | None = None) -> Path:
    parser = create_parser()
    args = parser.parse_args(argv)
    default_num_frames = {
        "jelly": JELLY_VIDEO_NUM_FRAMES,
        "jelly-single": JELLY_VIDEO_NUM_FRAMES,
        "jelly-domino": JELLY_VIDEO_NUM_FRAMES,
        "rigid": JELLY_VIDEO_NUM_FRAMES,
        "rigid-cup": JELLY_VIDEO_NUM_FRAMES,
    }[args.variant]
    num_frames = args.num_frames if args.num_frames is not None else default_num_frames

    if num_frames <= 0:
        parser.error("--num-frames must be positive")

    if args.variant == "rigid":
        return render_rigid_video(
            output_path=args.output_path or "outputs/ramp_tower_rigid.mp4",
            ball_material=args.ball_material,
            cube_material=args.cube_material,
            num_frames=num_frames,
            device=args.device,
        )

    if args.variant == "rigid-cup":
        return render_rigid_cup_video(
            output_path=args.output_path or "outputs/ramp_cup_rigid.mp4",
            ball_material=args.ball_material,
            cube_material=args.cube_material,
            num_frames=num_frames,
            device=args.device,
        )

    default_outputs = {
        "jelly": "outputs/ramp_tower_jelly_domino.mp4",
        "jelly-single": "outputs/ramp_tower_jelly_single.mp4",
        "jelly-domino": "outputs/ramp_tower_jelly_domino.mp4",
    }
    return render_jelly_video(
        output_path=args.output_path or default_outputs[args.variant],
        ball_material=args.ball_material,
        cube_material=args.cube_material,
        wall_count=JELLY_VARIANT_WALL_COUNTS[args.variant],
        num_frames=num_frames,
        device=args.device,
    )


if __name__ == "__main__":
    output = main()
    print(output)
