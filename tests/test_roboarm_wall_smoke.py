from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from npm_sim.roboarm_wall import (
    FRICTION_MODE_HIGH,
    FRICTION_MODE_LOW,
    PUSH_MODE_CENTER,
    PUSH_MODE_HIGH,
    PUSH_MODE_LOW,
    ROBOT_ACCEL_SECONDS,
    ROBOT_HOLD_SECONDS,
    default_video_output_path,
    franka_tool_target,
    run_simulation,
)


class RoboarmWallSmokeTest(unittest.TestCase):
    def test_high_push_tips_wall_over(self) -> None:
        result = run_simulation(push_mode=PUSH_MODE_HIGH, viewer="null", num_frames=220, device="cpu")

        self.assertTrue(np.isfinite(result.final_body_poses).all())

        tool_motion = result.final_tool_position - result.initial_tool_position
        wall_motion_y = result.final_wall_transform[1] - result.initial_wall_transform[1]
        top_motion_y = result.final_wall_top_position[1] - result.initial_wall_top_position[1]
        accel_seconds = ROBOT_ACCEL_SECONDS
        _, early_velocity = franka_tool_target(ROBOT_HOLD_SECONDS + accel_seconds * 0.25, PUSH_MODE_HIGH)
        _, late_velocity = franka_tool_target(ROBOT_HOLD_SECONDS + accel_seconds * 0.75, PUSH_MODE_HIGH)

        self.assertGreater(tool_motion[1], 0.20)
        self.assertLess(abs(tool_motion[0]), 0.02)
        self.assertLess(result.max_tool_path_x_deviation, 0.02)
        self.assertLess(result.max_tool_path_z_deviation, 0.03)
        self.assertGreater(late_velocity[1], early_velocity[1])
        self.assertGreater(wall_motion_y, 0.40)
        self.assertGreater(top_motion_y, 0.70)
        self.assertGreater(result.max_wall_tilt_degrees, 80.0)
        self.assertGreater(result.final_wall_tilt_degrees, 80.0)
        self.assertLess(result.final_wall_transform[2], 0.15)

    def test_low_push_slides_without_tipping(self) -> None:
        result = run_simulation(push_mode=PUSH_MODE_LOW, viewer="null", num_frames=480, device="cpu")

        self.assertTrue(np.isfinite(result.final_body_poses).all())

        tool_motion = result.final_tool_position - result.initial_tool_position
        wall_motion_y = result.final_wall_transform[1] - result.initial_wall_transform[1]

        self.assertGreater(tool_motion[1], 0.20)
        self.assertLess(abs(tool_motion[0]), 0.02)
        self.assertLess(result.max_tool_path_x_deviation, 0.02)
        self.assertLess(result.max_tool_path_z_deviation, 0.03)
        self.assertGreater(wall_motion_y, 0.04)
        self.assertLess(result.max_wall_tilt_degrees, 16.0)
        self.assertLess(result.final_wall_tilt_degrees, 5.0)
        self.assertLess(result.final_wall_velocity_norm, 0.01)

    def test_center_push_high_friction_tips_wall_over(self) -> None:
        result = run_simulation(
            push_mode=PUSH_MODE_CENTER,
            friction_mode=FRICTION_MODE_HIGH,
            viewer="null",
            num_frames=480,
            device="cpu",
        )

        self.assertTrue(np.isfinite(result.final_body_poses).all())
        self.assertGreater(result.max_wall_tilt_degrees, 80.0)
        self.assertGreater(result.final_wall_tilt_degrees, 80.0)
        self.assertLess(result.final_wall_transform[2], 0.15)

    def test_center_push_low_friction_slides_without_tipping(self) -> None:
        result = run_simulation(
            push_mode=PUSH_MODE_CENTER,
            friction_mode=FRICTION_MODE_LOW,
            viewer="null",
            num_frames=480,
            device="cpu",
        )

        self.assertTrue(np.isfinite(result.final_body_poses).all())
        wall_motion_y = result.final_wall_transform[1] - result.initial_wall_transform[1]

        self.assertGreater(wall_motion_y, 0.10)
        self.assertLess(result.max_wall_tilt_degrees, 8.0)
        self.assertLess(result.final_wall_tilt_degrees, 5.0)
        self.assertLess(result.final_wall_velocity_norm, 0.01)

    def test_pushes_use_identical_horizontal_motion(self) -> None:
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            time_seconds = ROBOT_HOLD_SECONDS + ROBOT_ACCEL_SECONDS * fraction
            high_position, high_velocity = franka_tool_target(time_seconds, PUSH_MODE_HIGH)
            low_position, low_velocity = franka_tool_target(time_seconds, PUSH_MODE_LOW)
            center_position, center_velocity = franka_tool_target(time_seconds, PUSH_MODE_CENTER)

            self.assertAlmostEqual(float(high_position[1]), float(low_position[1]), places=6)
            self.assertAlmostEqual(float(high_position[1]), float(center_position[1]), places=6)
            self.assertAlmostEqual(float(high_velocity[1]), float(low_velocity[1]), places=6)
            self.assertAlmostEqual(float(high_velocity[1]), float(center_velocity[1]), places=6)
            self.assertAlmostEqual(float(high_position[0]), float(low_position[0]), places=6)
            self.assertAlmostEqual(float(high_position[0]), float(center_position[0]), places=6)
            self.assertAlmostEqual(float(high_velocity[0]), float(low_velocity[0]), places=6)
            self.assertAlmostEqual(float(high_velocity[0]), float(center_velocity[0]), places=6)

    def test_video_outputs_are_limited_to_the_four_roboarm_wall_cases(self) -> None:
        self.assertEqual(default_video_output_path(PUSH_MODE_HIGH), "outputs/roboarm_wall_high.mp4")
        self.assertEqual(default_video_output_path(PUSH_MODE_LOW), "outputs/roboarm_wall_low.mp4")
        self.assertEqual(
            default_video_output_path(PUSH_MODE_CENTER, FRICTION_MODE_HIGH),
            "outputs/roboarm_wall_center_high_friction.mp4",
        )
        self.assertEqual(
            default_video_output_path(PUSH_MODE_CENTER, FRICTION_MODE_LOW),
            "outputs/roboarm_wall_center_low_friction.mp4",
        )
        with self.assertRaises(ValueError):
            default_video_output_path(PUSH_MODE_CENTER)
        with self.assertRaises(ValueError):
            default_video_output_path(PUSH_MODE_HIGH, FRICTION_MODE_HIGH)


if __name__ == "__main__":
    unittest.main()
