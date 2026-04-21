"""Standalone check: does DFSPH hold water at the brim without ball impact?

This is the critical test for the Genesis variant. With Newton v1.1.0's
``SolverImplicitMPM`` the same cup drops the column from z_max ≈ 0.227 to
0.158 over 1 s of settling (≈30% column loss) because there is no pressure
projection. With DFSPH we expect the top surface to stay within a few particle
diameters of its initial height for the full simulation.

Reports water column top-z over time so we can eyeball any drift.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

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


def main() -> int:
    mod = _load_module()

    # Build the scene but skip the ball so we isolate water settling.
    class _NoBallDemo(mod.RampCupWaterGenesisDemo):
        def _add_ball(self) -> None:
            self.ball = None

        def _ball_pos(self) -> np.ndarray:
            return np.zeros(3)

    demo = _NoBallDemo(num_frames=0, show_viewer=False)
    particles0 = demo._particle_positions()
    z_top0 = float(particles0[:, 2].max())
    z_mean0 = float(particles0[:, 2].mean())
    print(f"t=0.00s  particles={particles0.shape[0]:>5}  z_top={z_top0:.4f}  z_mean={z_mean0:.4f}")

    # Let water settle into cup cavity for t_settle, then measure drift over t_hold.
    t_settle = 0.5
    t_hold = 2.0
    cup_centre_xy = np.array([mod.CUP_CENTER_X, mod.CUP_CENTER_Y])

    def _stats():
        p = demo._particle_positions()
        z_top = float(p[:, 2].max())
        z_mean = float(p[:, 2].mean())
        r = np.linalg.norm(p[:, :2] - cup_centre_xy, axis=1)
        outside = int((r > 0.12).sum())  # outside any reasonable cup radius
        return p.shape[0], z_top, z_mean, outside

    n_settle = int(round(t_settle * mod.FRAME_RATE))
    for _ in range(n_settle):
        demo.step()

    n_particles, z_top_settled, z_mean_settled, outside_settled = _stats()
    print(f"after settle ({t_settle:.2f}s):  particles={n_particles}  z_top={z_top_settled:.4f}  z_mean={z_mean_settled:.4f}  outside={outside_settled}")

    n_hold = int(round(t_hold * mod.FRAME_RATE))
    report_every = max(1, mod.FRAME_RATE // 2)
    start = time.time()
    for frame in range(1, n_hold + 1):
        demo.step()
        if frame % report_every == 0 or frame == n_hold:
            _, z_top, z_mean, outside = _stats()
            drift_top = z_top - z_top_settled
            drift_mean = z_mean - z_mean_settled
            t = (n_settle + frame) / mod.FRAME_RATE
            print(f"t={t:.2f}s  z_top={z_top:.4f} (d={drift_top:+.4f})  z_mean={z_mean:.4f}  outside={outside}")
    wall = time.time() - start
    print(f"hold wall {wall:.2f}s for {t_hold:.2f}s sim ({wall / t_hold:.1f} x real-time)")

    _, z_top_final, z_mean_final, outside_final = _stats()
    top_drift = z_top_final - z_top_settled
    lost_frac = outside_final / n_particles

    print()
    print(f"post-settle top drift: {top_drift:+.4f} m  (target: < ±1 cm)")
    print(f"particles outside cup: {outside_final}/{n_particles} ({lost_frac * 100:.1f}%)  (target: < 2%)")

    if abs(top_drift) < 0.01 and lost_frac < 0.02:
        print("PASS  DFSPH held water at a stable level inside the cup")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
