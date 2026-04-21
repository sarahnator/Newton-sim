"""Measure where Genesis is spending time in the cup-water sim.

Breaks the render pipeline into stages and times each one so we know which
knob actually moves the wall clock: bake, steady-state sim, rendering
(vis_mode='recon' vs 'particle'), SDF resolution, substep count, or particle
count. Output: a single line per stage with elapsed seconds.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

ROOT = Path(__file__).resolve().parents[1]


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("mod", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _time(label: str, fn):
    t0 = time.time()
    result = fn()
    dt = time.time() - t0
    print(f"  [{dt:7.2f}s] {label}")
    return result, dt


def main() -> int:
    mod = _load_module(ROOT / "src" / "npm_sim" / "rigid_ramp_cup_water_genesis.py")

    print(f"SIM_SUBSTEPS = {mod.SIM_SUBSTEPS}")
    print(f"WATER_PARTICLE_SIZE = {mod.WATER_PARTICLE_SIZE}")
    print(f"EMISSION_OVERFILL_FACTOR = {mod.EMISSION_OVERFILL_FACTOR}")
    print()

    print("Stage 1: build scene (no camera)")
    demo = None

    def _build():
        nonlocal demo
        demo = mod.RampCupWaterGenesisDemo(num_frames=0, show_viewer=False, enable_camera=False)

    _, t_build = _time("build+init (includes CoACD of cup)", _build)
    print(f"  particle count: {demo.water.n_particles}")

    print()
    print("Stage 2: 30 pure sim steps (no render)")
    _, t_sim = _time("30 steps, no ball hold", lambda: [demo.step() for _ in range(30)])
    per_step = t_sim / 30 * 1000
    per_particle_substep = per_step / mod.SIM_SUBSTEPS / demo.water.n_particles * 1e6
    print(f"  -> {per_step:.1f} ms/frame, {per_particle_substep:.2f} µs/(particle·substep)")

    print()
    print("Stage 3: 30 sim steps with camera render, vis_mode = recon")
    demo2 = mod.RampCupWaterGenesisDemo(num_frames=0, show_viewer=False, enable_camera=True)
    demo2.camera.render(force_render=True)  # warm up
    _, t_render_recon = _time(
        "30 steps + render (recon)",
        lambda: [(demo2.step(), demo2.camera.render()) for _ in range(30)],
    )
    per_render = t_render_recon / 30 * 1000
    print(f"  -> {per_render:.1f} ms/frame (sim+render)")
    print(f"  -> render-only overhead: {per_render - per_step:.1f} ms/frame")

    print()
    print("Totals")
    print(f"  build (one-time):                 {t_build:.2f} s")
    print(f"  sim-only per 180-frame pass:      {t_sim / 30 * 180:.2f} s")
    print(f"  sim+recon-render per 180-frame:   {t_render_recon / 30 * 180:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
