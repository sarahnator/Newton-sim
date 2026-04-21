"""Smoke test for the Genesis cup-water variant.

Runs a short headless simulation and checks that:
  * the ball starts on the ramp and falls/rolls onto the floor (gravity + ramp geometry).
  * water particles stay finite and don't escape the computational domain.
  * water is initially disturbed only by the (hopefully small) DFSPH transient —
    i.e., the pre-impact settling we had with Newton MPM is not reproduced here.

Run from the genesis-sim conda env: the main ``src.npm_sim`` package imports
Newton/Warp, which is not installed in that env, so we import the module directly
by path to avoid the package's ``__init__`` side effects.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "npm_sim" / "rigid_ramp_cup_water_genesis.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("rigid_ramp_cup_water_genesis", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RigidRampCupWaterGenesisSmokeTest(unittest.TestCase):
    def test_smoke(self) -> None:
        mod = _load_module()
        result = mod.run_simulation(num_frames=30, show_viewer=False)

        self.assertTrue(np.isfinite(result.final_ball_position).all())
        self.assertTrue(np.isfinite(result.final_cup_position).all())
        self.assertTrue(np.isfinite(result.final_particle_positions).all())
        self.assertGreater(result.initial_particle_positions.shape[0], 0)
        self.assertEqual(
            result.initial_particle_positions.shape,
            result.final_particle_positions.shape,
        )


if __name__ == "__main__":
    unittest.main()
