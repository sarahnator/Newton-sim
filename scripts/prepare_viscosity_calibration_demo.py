"""Prepare Bayesian-optimization viscosity-calibration demo artifacts.

The target is a synthetic "unknown" liquid. The script runs Genesis at a small
number of candidate viscosities, compares simple pour metrics against the
target, and writes plots/tables for a professor-facing presentation.

Run from the ``genesis-sim`` conda env:
    conda run -n genesis-sim python scripts/prepare_viscosity_calibration_demo.py
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "npm_sim" / "robotic_arm_pour_genesis.py"
DEFAULT_OUT_DIR = ROOT / "outputs" / "calibration_demo"
DEFAULT_SETTLED_CACHE = (
    ROOT
    / "outputs"
    / "_genesis"
    / "robotic_arm_pickup_pour_base050_p006_fill080_over1405_clear006_fric005_soft0015_pose080_slow_fillet012_micro084_sdf0025_align0_corrbase_settled_water.npy"
)
INITIAL_VISCOSITIES = (0.003, 0.0055, 0.010)
SEARCH_BOUNDS = (0.003, 0.012)


def _load_module():
    spec = importlib.util.spec_from_file_location("robotic_arm_pour_genesis_calibration", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _configure_viscosity(mod, viscosity: float) -> None:
    mod.WATER_VISCOSITY = float(viscosity)
    mod.WATER_DENSITY = 1000.0
    mod.LIQUID_SURFACE_TENSION = 0.01
    mod.LIQUID_COLOR = (0.35, 0.62, 0.90, 1.0)
    mod.GLASS_COUP_FRICTION = 0.05
    mod.SOURCE_BOUNDARY_CORRECTION_INTERVAL = 1
    mod.POUR_POSE_FRACTION = 0.80
    mod.PANDA_Q_POUR = mod.PANDA_Q_UPRIGHT + (mod.PANDA_Q_FULL_POUR - mod.PANDA_Q_UPRIGHT) * mod.POUR_POSE_FRACTION
    mod.SETTLED_PARTICLES_CACHE = DEFAULT_SETTLED_CACHE


def _ensure_settled_cache(mod, cache_path: Path) -> None:
    if cache_path.exists():
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    mod.bake_settled_particles(cache_path=cache_path)


def _first_receiver_time(history: list[dict[str, float]], threshold_fraction: float) -> float | None:
    for row in history:
        if row["receiver_fraction"] >= threshold_fraction:
            return row["time_seconds"]
    return None


def _spill_fraction(row: dict[str, float]) -> float:
    return max(0.0, row["live_fraction"] - row["pourer_fraction"] - row["receiver_fraction"])


def _record_history_row(mod, demo, frame: int, initial_count: int) -> dict[str, float]:
    particles = demo._particle_positions()
    in_pourer, in_receiver, live = demo.particle_counts(particles)
    return {
        "frame": int(frame),
        "time_seconds": float(frame / mod.FRAME_RATE),
        "pourer_fraction": float(in_pourer / max(1, initial_count)),
        "receiver_fraction": float(in_receiver / max(1, initial_count)),
        "live_fraction": float(live / max(1, initial_count)),
        "spill_fraction": float(max(0, live - in_pourer - in_receiver) / max(1, initial_count)),
    }


def _metrics_cache_path(out_dir: Path, label: str, viscosity: float | None = None) -> Path:
    if viscosity is None:
        return out_dir / f"{label}_metrics.json"
    return out_dir / f"{label}_mu_{viscosity:.6f}.json"


def run_metrics(
    mod,
    *,
    viscosity: float,
    num_frames: int,
    metric_stride: int,
    out_dir: Path,
    label: str,
    force: bool,
) -> dict[str, Any]:
    cache_path = _metrics_cache_path(out_dir, label, None if label == "target" else viscosity)
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text())

    _configure_viscosity(mod, viscosity)
    _ensure_settled_cache(mod, DEFAULT_SETTLED_CACHE)

    start_wall = time.time()
    demo = mod.RoboticArmPourGenesisDemo(num_frames=0, show_viewer=False, enable_camera=False)
    if not demo.load_settled_particles(DEFAULT_SETTLED_CACHE):
        mod.bake_settled_particles(cache_path=DEFAULT_SETTLED_CACHE)
        loaded = demo.load_settled_particles(DEFAULT_SETTLED_CACHE)
        if not loaded:
            raise RuntimeError(f"could not load settled particles from {DEFAULT_SETTLED_CACHE}")

    initial_in_pourer, _, _ = demo.particle_counts(
        demo.initial_particles,
        cup_pos=demo.initial_cup_pos,
        cup_quat=demo.initial_cup_quat,
    )

    history = [_record_history_row(mod, demo, 0, initial_in_pourer)]
    for frame in range(1, num_frames + 1):
        demo.step()
        if frame % metric_stride == 0 or frame == num_frames:
            history.append(_record_history_row(mod, demo, frame, initial_in_pourer))

    result = demo.result()
    final = history[-1]
    metrics = {
        "label": label,
        "viscosity": float(viscosity),
        "log10_viscosity": float(math.log10(viscosity)),
        "num_frames": int(num_frames),
        "metric_stride": int(metric_stride),
        "initial_particle_count": int(result.initial_particle_count),
        "final_particles_in_pourer": int(result.final_particles_in_pourer),
        "final_particles_in_receiver": int(result.final_particles_in_receiver),
        "final_live_particles": int(result.final_live_particles),
        "pourer_fraction": float(result.pourer_fraction),
        "receiver_fraction": float(result.receiver_fraction),
        "live_fraction": float(result.final_live_particles / max(1, result.initial_particle_count)),
        "spill_fraction": float(_spill_fraction(final)),
        "first_receiver_time": _first_receiver_time(history, threshold_fraction=0.01),
        "max_tilt_degrees": float(result.max_tilt_degrees),
        "final_tilt_degrees": float(result.final_tilt_degrees),
        "max_glass_solid_particles": int(result.max_glass_solid_particles),
        "max_pourer_base_particles": int(result.max_pourer_base_particles),
        "history": history,
        "wall_seconds": float(time.time() - start_wall),
    }
    cache_path.write_text(json.dumps(metrics, indent=2, default=_json_default) + "\n")
    return metrics


def _receiver_curve(metrics: dict[str, Any], times: np.ndarray) -> np.ndarray:
    history = metrics["history"]
    src_t = np.array([row["time_seconds"] for row in history], dtype=np.float64)
    src_y = np.array([row["receiver_fraction"] for row in history], dtype=np.float64)
    return np.interp(times, src_t, src_y)


def compute_loss(candidate: dict[str, Any], target: dict[str, Any]) -> float:
    t_end = target["num_frames"] / 60.0
    times = np.linspace(0.0, t_end, 80)
    curve_rmse = float(np.sqrt(np.mean((_receiver_curve(candidate, times) - _receiver_curve(target, times)) ** 2)))
    arrival_candidate = candidate["first_receiver_time"] if candidate["first_receiver_time"] is not None else t_end
    arrival_target = target["first_receiver_time"] if target["first_receiver_time"] is not None else t_end
    arrival_error = (arrival_candidate - arrival_target) / max(t_end, 1.0)

    receiver_error = candidate["receiver_fraction"] - target["receiver_fraction"]
    pourer_error = candidate["pourer_fraction"] - target["pourer_fraction"]
    spill_error = candidate["spill_fraction"] - target["spill_fraction"]
    return float(
        2.0 * receiver_error * receiver_error
        + 2.0 * pourer_error * pourer_error
        + 0.8 * spill_error * spill_error
        + 1.5 * curve_rmse * curve_rmse
        + 0.15 * arrival_error * arrival_error
    )


def _kernel(a: np.ndarray, b: np.ndarray, length_scale: float = 0.18) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)[:, None]
    b = np.asarray(b, dtype=np.float64)[None, :]
    return np.exp(-0.5 * ((a - b) / length_scale) ** 2)


def propose_next_log_mu(
    log_mus: list[float],
    losses: list[float],
    *,
    bounds: tuple[float, float],
    min_gap: float = 0.010,
) -> float:
    xs = np.asarray(log_mus, dtype=np.float64)
    ys = np.asarray(losses, dtype=np.float64)
    grid = np.linspace(bounds[0], bounds[1], 800)

    y_mean = float(ys.mean())
    y_std = float(ys.std())
    if y_std < 1.0e-12:
        y_std = 1.0
    y_norm = (ys - y_mean) / y_std

    K = _kernel(xs, xs) + np.eye(len(xs)) * 1.0e-6
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
    Ks = _kernel(grid, xs)
    mean = Ks @ alpha
    v = np.linalg.solve(L, Ks.T)
    var = np.maximum(1.0 - np.sum(v * v, axis=0), 1.0e-12)
    std = np.sqrt(var)

    best = float(y_norm.min())
    improvement = best - mean
    z = improvement / std
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)
    for x in xs:
        ei[np.abs(grid - x) < min_gap] = -np.inf
    return float(grid[int(np.argmax(ei))])


def run_bo(
    mod,
    *,
    target: dict[str, Any],
    out_dir: Path,
    num_frames: int,
    metric_stride: int,
    initial_viscosities: tuple[float, ...],
    bounds: tuple[float, float],
    bo_iterations: int,
    force: bool,
) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []

    for iteration, viscosity in enumerate(initial_viscosities):
        metrics = run_metrics(
            mod,
            viscosity=viscosity,
            num_frames=num_frames,
            metric_stride=metric_stride,
            out_dir=out_dir,
            label="candidate",
            force=force,
        )
        metrics["iteration"] = iteration
        metrics["phase"] = "initial"
        metrics["loss"] = compute_loss(metrics, target)
        evaluations.append(metrics)
        _write_history(out_dir, target, evaluations)

    log_bounds = (math.log10(bounds[0]), math.log10(bounds[1]))
    for _ in range(bo_iterations):
        log_mus = [row["log10_viscosity"] for row in evaluations]
        losses = [row["loss"] for row in evaluations]
        next_log_mu = propose_next_log_mu(log_mus, losses, bounds=log_bounds)
        next_mu = 10.0 ** next_log_mu
        metrics = run_metrics(
            mod,
            viscosity=next_mu,
            num_frames=num_frames,
            metric_stride=metric_stride,
            out_dir=out_dir,
            label="candidate",
            force=force,
        )
        metrics["iteration"] = len(evaluations)
        metrics["phase"] = "bayes_opt"
        metrics["loss"] = compute_loss(metrics, target)
        evaluations.append(metrics)
        _write_history(out_dir, target, evaluations)

    return evaluations


def _write_history(out_dir: Path, target: dict[str, Any], evaluations: list[dict[str, Any]]) -> None:
    payload = {
        "target": target,
        "evaluations": evaluations,
        "best": min(evaluations, key=lambda row: row["loss"]) if evaluations else None,
    }
    (out_dir / "bo_history.json").write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")

    with (out_dir / "bo_evaluations.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "iteration",
                "phase",
                "viscosity",
                "log10_viscosity",
                "loss",
                "receiver_fraction",
                "pourer_fraction",
                "spill_fraction",
                "first_receiver_time",
                "wall_seconds",
            ],
        )
        writer.writeheader()
        for row in evaluations:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})


def plot_convergence(out_dir: Path, evaluations: list[dict[str, Any]]) -> None:
    iterations = np.array([row["iteration"] for row in evaluations], dtype=int)
    losses = np.array([row["loss"] for row in evaluations], dtype=np.float64)
    best_so_far = np.minimum.accumulate(losses)
    viscosities = np.array([row["viscosity"] for row in evaluations], dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.0), sharex=True)
    axes[0].plot(iterations, losses, "o-", label="evaluated loss")
    axes[0].plot(iterations, best_so_far, "s-", label="best so far")
    axes[0].set_ylabel("objective loss")
    axes[0].set_title("Bayesian Optimization Convergence")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(iterations, viscosities, "o-", color="#3b6fb6")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("simulation evaluation")
    axes[1].set_ylabel("viscosity parameter")
    axes[1].grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "bo_convergence.png", dpi=180)
    plt.close(fig)


def plot_objective(
    out_dir: Path,
    evaluations: list[dict[str, Any]],
    *,
    target_viscosity: float,
    bounds: tuple[float, float],
) -> None:
    xs = np.array([row["log10_viscosity"] for row in evaluations], dtype=np.float64)
    ys = np.array([row["loss"] for row in evaluations], dtype=np.float64)
    grid = np.linspace(math.log10(bounds[0]), math.log10(bounds[1]), 600)

    y_mean = float(ys.mean())
    y_std = float(ys.std())
    if y_std < 1.0e-12:
        y_std = 1.0
    K = _kernel(xs, xs) + np.eye(len(xs)) * 1.0e-6
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, (ys - y_mean) / y_std))
    Ks = _kernel(grid, xs)
    mean = (Ks @ alpha) * y_std + y_mean
    v = np.linalg.solve(L, Ks.T)
    std = np.sqrt(np.maximum(1.0 - np.sum(v * v, axis=0), 1.0e-12)) * y_std

    best = min(evaluations, key=lambda row: row["loss"])
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.fill_between(10.0 ** grid, mean - 1.96 * std, mean + 1.96 * std, alpha=0.18, color="#7aa6d8")
    ax.plot(10.0 ** grid, mean, color="#2f5f9f", label="GP surrogate mean")
    ax.scatter(10.0 ** xs, ys, color="#111111", zorder=3, label="Genesis evaluations")
    ax.axvline(target_viscosity, color="#2f8f4e", linestyle="--", label="synthetic target")
    ax.axvline(best["viscosity"], color="#c4503d", linestyle=":", label=f"best estimate {best['viscosity']:.4g}")
    ax.set_xscale("log")
    ax.set_xlim(bounds)
    ax.set_xlabel("viscosity parameter")
    ax.set_ylabel("objective loss")
    ax.set_title("Narrow Log-Space Viscosity Search")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "viscosity_objective.png", dpi=180)
    plt.close(fig)


def plot_transfer_curves(
    out_dir: Path,
    target: dict[str, Any],
    evaluations: list[dict[str, Any]],
) -> None:
    best = min(evaluations, key=lambda row: row["loss"])
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    def plot_one(metrics: dict[str, Any], label: str, *, linewidth: float = 1.8, alpha: float = 1.0) -> None:
        t = [row["time_seconds"] for row in metrics["history"]]
        r = [row["receiver_fraction"] for row in metrics["history"]]
        ax.plot(t, r, label=label, linewidth=linewidth, alpha=alpha)

    plot_one(target, "target unknown", linewidth=2.6)
    for row in evaluations:
        if row["phase"] == "initial":
            plot_one(row, f"init mu={row['viscosity']:.4g}", linewidth=1.2, alpha=0.65)
    plot_one(best, f"best mu={best['viscosity']:.4g}", linewidth=2.4)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("fraction in receiving glass")
    ax.set_title("Transferred Liquid Over Time")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "receiver_transfer_curves.png", dpi=180)
    plt.close(fig)


def plot_final_metrics(
    out_dir: Path,
    target: dict[str, Any],
    evaluations: list[dict[str, Any]],
) -> None:
    best = min(evaluations, key=lambda row: row["loss"])
    labels = ["receiver", "pourer", "outside"]
    target_values = [target["receiver_fraction"], target["pourer_fraction"], target["spill_fraction"]]
    best_values = [best["receiver_fraction"], best["pourer_fraction"], best["spill_fraction"]]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(x - width / 2, target_values, width, label="target")
    ax.bar(x + width / 2, best_values, width, label=f"best mu={best['viscosity']:.4g}")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, max(1.0, max(target_values + best_values) * 1.15))
    ax.set_ylabel("fraction of initial particles")
    ax.set_title("Final Pour Metrics")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "final_metric_comparison.png", dpi=180)
    plt.close(fig)


def write_summary(
    out_dir: Path,
    *,
    target: dict[str, Any],
    evaluations: list[dict[str, Any]],
    bounds: tuple[float, float],
) -> None:
    best = min(evaluations, key=lambda row: row["loss"])
    lines = [
        "# Viscosity Calibration Demo",
        "",
        "Goal: estimate the Genesis liquid viscosity parameter from pour behavior.",
        "",
        "Setup:",
        f"- Synthetic target viscosity: `{target['viscosity']:.6g}`",
        f"- BO search interval: `[{bounds[0]:.6g}, {bounds[1]:.6g}]`",
        "- Objective compares final receiver/pourer/outside fractions, first receiver-arrival time, and receiver transfer curve.",
        "",
        "Result:",
        f"- Best estimated viscosity: `{best['viscosity']:.6g}`",
        f"- Best objective loss: `{best['loss']:.6g}`",
        f"- Simulations evaluated: `{len(evaluations)}`",
        "",
        "Target metrics:",
        f"- Receiver fraction: `{target['receiver_fraction']:.4f}`",
        f"- Pourer fraction: `{target['pourer_fraction']:.4f}`",
        f"- Outside fraction: `{target['spill_fraction']:.4f}`",
        f"- First receiver arrival: `{target['first_receiver_time']}` s",
        "",
        "Best-match metrics:",
        f"- Receiver fraction: `{best['receiver_fraction']:.4f}`",
        f"- Pourer fraction: `{best['pourer_fraction']:.4f}`",
        f"- Outside fraction: `{best['spill_fraction']:.4f}`",
        f"- First receiver arrival: `{best['first_receiver_time']}` s",
        "",
        "Presentation files:",
        "- `bo_convergence.png`",
        "- `viscosity_objective.png`",
        "- `receiver_transfer_curves.png`",
        "- `final_metric_comparison.png`",
        "- `bo_evaluations.csv`",
        "- `bo_history.json`",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--metric-stride", type=int, default=6)
    parser.add_argument("--bo-iterations", type=int, default=5)
    parser.add_argument("--target-viscosity", type=float, default=None)
    parser.add_argument("--min-viscosity", type=float, default=SEARCH_BOUNDS[0])
    parser.add_argument("--max-viscosity", type=float, default=SEARCH_BOUNDS[1])
    parser.add_argument(
        "--initial-viscosities",
        type=float,
        nargs="+",
        default=list(INITIAL_VISCOSITIES),
    )
    parser.add_argument("--force", action="store_true", help="Re-run simulations even if metrics JSON files exist.")
    args = parser.parse_args(argv)

    if args.min_viscosity <= 0.0 or args.max_viscosity <= args.min_viscosity:
        parser.error("viscosity bounds must be positive and increasing")
    if args.metric_stride <= 0:
        parser.error("--metric-stride must be positive")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mod = _load_module()
    target_viscosity = mod.TARGET_VISCOSITY if args.target_viscosity is None else args.target_viscosity
    num_frames = mod.VIDEO_NUM_FRAMES if args.num_frames is None else args.num_frames
    if num_frames <= 0:
        parser.error("--num-frames must be positive")

    target = run_metrics(
        mod,
        viscosity=target_viscosity,
        num_frames=num_frames,
        metric_stride=args.metric_stride,
        out_dir=out_dir,
        label="target",
        force=args.force,
    )
    evaluations = run_bo(
        mod,
        target=target,
        out_dir=out_dir,
        num_frames=num_frames,
        metric_stride=args.metric_stride,
        initial_viscosities=tuple(args.initial_viscosities),
        bounds=(args.min_viscosity, args.max_viscosity),
        bo_iterations=args.bo_iterations,
        force=args.force,
    )

    _write_history(out_dir, target, evaluations)
    plot_convergence(out_dir, evaluations)
    plot_objective(
        out_dir,
        evaluations,
        target_viscosity=target_viscosity,
        bounds=(args.min_viscosity, args.max_viscosity),
    )
    plot_transfer_curves(out_dir, target, evaluations)
    plot_final_metrics(out_dir, target, evaluations)
    write_summary(out_dir, target=target, evaluations=evaluations, bounds=(args.min_viscosity, args.max_viscosity))

    best = min(evaluations, key=lambda row: row["loss"])
    print(f"wrote calibration demo to {out_dir}")
    print(f"target viscosity: {target_viscosity:.6g}")
    print(f"best estimate:    {best['viscosity']:.6g}")
    print(f"best loss:        {best['loss']:.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
