from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from npm_sim.rigid_ramp_cup import run_simulation


def _quat_angle_delta_rad(q0: np.ndarray, q1: np.ndarray) -> float:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.clip(abs(np.dot(q0, q1)), -1.0, 1.0))
    return 2.0 * np.arccos(dot)


class RigidRampCupSmokeTest(unittest.TestCase):
    def test_rigid_ramp_cup_smoke(self) -> None:
        result = run_simulation(viewer="null", num_frames=180, device="cpu")

        self.assertTrue(np.isfinite(result.final_body_poses).all())
        self.assertGreater(result.final_ball_position[1], result.initial_ball_position[1] + 0.20)

        cup_translation = np.linalg.norm(result.final_cup_transform[:3] - result.initial_cup_transform[:3])
        cup_rotation = _quat_angle_delta_rad(result.initial_cup_transform[3:], result.final_cup_transform[3:])

        moved_cup = cup_translation > 0.03 or cup_rotation > np.deg2rad(5.0)
        self.assertTrue(moved_cup)


if __name__ == "__main__":
    unittest.main()
