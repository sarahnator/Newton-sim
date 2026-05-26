#!/usr/bin/env python3
"""
Build a VLM prompt dataset from already-generated sweep datasets.

Expected layout:

datasets/
  ramp_cup_sweep/
    manifest.jsonl
    manifest.csv
    <sweep_param>/<scene_id>/metadata.json
    <sweep_param>/<scene_id>/resolved_config.json
    <sweep_param>/<scene_id>/result.npz
    <sweep_param>/<scene_id>/video.mp4

  robotic_pour_sweep/
    manifest.jsonl
    ...
    <sweep_param>/<scene_id>/metadata.json
    <sweep_param>/<scene_id>/resolved_config.json
    <sweep_param>/<scene_id>/result.npz
    <sweep_param>/<scene_id>/video.mp4

  pendulum_sweep/
    manifest.jsonl
    ...
    <sweep_param>/<scene_id>/metadata.json
    <sweep_param>/<scene_id>/resolved_config.json
    <sweep_param>/<scene_id>/trajectory.npz
    <sweep_param>/<scene_id>/pendulum.mp4

Output:

datasets/vlm_prompt_dataset/
  vlm_entries.jsonl
  dataset_card_summary.json
  frames/
    <scene_key>/<scene_id>/
      frame_000.png
      frame_mid.png
      frame_final.png

This script is scene-aware but simulator-independent:
it consumes saved artifacts and never reruns Genesis.

To run:
  uv run scripts/build_vlm_prompt_dataset.py \
    --datasets-root datasets \
    --output-root datasets/vlm_prompt_dataset
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SWEEP_DIRS = {
    "ramp_cup": "ramp_cup_sweep",
    "robotic_pour": "robotic_pour_sweep",
    "pendulum": "pendulum_sweep",
}

# Ramp-cup thresholds.
CUP_CONTACT_DISPLACEMENT_THRESHOLD_M = 0.02
CUP_MOVED_SIGNIFICANTLY_THRESHOLD_M = 0.05
CUP_LARGE_MOTION_THRESHOLD_M = 0.20
CUP_FELL_OVER_TILT_THRESHOLD_DEG = 60.0
CUP_ON_SIDE_TILT_THRESHOLD_DEG = 75.0

# Robotic-pour thresholds.
# Current result metrics report receiver_fraction as fraction of the initial liquid.
# We use receiver_fraction >= 0.25 as "a quarter of the initial liquid reached
# the receiver" and receiver_fraction >= 0.50 as the proxy for "half-filled".
POUR_QUARTER_INITIAL_FRACTION = 0.25
POUR_HALF_RECEIVER_FRACTION = 0.50
POUR_TARGET_TOLERANCE = 0.075

# Pendulum thresholds.
PENDULUM_SMALL_ANGLE_REL_PERIOD_TOL = 0.10
PENDULUM_SYNC_REL_PERIOD_TOL = 0.05
PENDULUM_SLOWER_PERIOD_TOL = 0.05

def read_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def safe_float(x: Any, default: float | None = None) -> float | None:
    try:
        return float(x)
    except Exception:
        return default


def np_scalar(result_npz: dict[str, np.ndarray], key: str, default: float | None = None) -> float | None:
    if key not in result_npz:
        return default
    try:
        return float(np.asarray(result_npz[key]).item())
    except Exception:
        return default


def find_existing_path(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def resolve_sim_dir(dataset_root: Path, manifest_row: dict[str, Any]) -> Path:
    """
    Prefer output_dir in manifest. Fall back to dataset_root / sweep_param / scene_id.
    """
    output_dir = manifest_row.get("output_dir")
    if output_dir:
        p = Path(output_dir)
        if p.exists():
            return p
        p2 = dataset_root / p
        if p2.exists():
            return p2

    sweep_param = manifest_row.get("parameter_name") or manifest_row.get("causal_factor")
    scene_id = manifest_row.get("scene_id")
    if sweep_param and scene_id:
        return dataset_root / str(sweep_param) / str(scene_id)

    raise FileNotFoundError(f"Could not resolve sim dir for manifest row: {manifest_row}")


def extract_anchor_frames(video_path: Path, frames_dir: Path, overwrite: bool = False) -> list[str]:
    """
    Extract initial/mid/final frames using ffmpeg.

    Returns:
      [frame_000, frame_mid, frame_final]

    If no video exists, returns [].
    """
    if not video_path.exists():
        print(f"[warning] video missing: {video_path}")
        return []

    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_000 = frames_dir / "frame_000.png"
    frame_mid = frames_dir / "frame_mid.png"
    frame_final = frames_dir / "frame_final.png"

    if not overwrite and frame_000.exists() and frame_mid.exists() and frame_final.exists():
        return [str(frame_000), str(frame_mid), str(frame_final)]

    if overwrite:
        for p in [frame_000, frame_mid, frame_final]:
            if p.exists():
                p.unlink()

    def run_ffmpeg(cmd: list[str], label: str) -> bool:
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            return True
        except subprocess.CalledProcessError as exc:
            print(f"[warning] ffmpeg failed while extracting {label} from {video_path}")
            if exc.stderr:
                print(exc.stderr[-1000:])
            return False

    num_frames = None
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        text = probe.stdout.strip()
        if text and text != "N/A":
            num_frames = int(text)
    except Exception as exc:
        print(f"[warning] ffprobe frame count failed for {video_path}: {exc}")

    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            "select=eq(n\\,0)",
            "-vsync",
            "0",
            "-frames:v",
            "1",
            str(frame_000),
        ],
        "frame_000",
    )

    if num_frames is not None and num_frames > 1:
        mid_idx = max(0, num_frames // 2)
        final_idx = max(0, num_frames - 1)

        run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vf",
                f"select=eq(n\\,{mid_idx})",
                "-vsync",
                "0",
                "-frames:v",
                "1",
                str(frame_mid),
            ],
            "frame_mid",
        )

        run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vf",
                f"select=eq(n\\,{final_idx})",
                "-vsync",
                "0",
                "-frames:v",
                "1",
                str(frame_final),
            ],
            "frame_final",
        )
    else:
        duration = None
        try:
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(video_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            duration = float(probe.stdout.strip())
        except Exception:
            duration = None

        if duration is not None and duration > 0:
            run_ffmpeg(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{0.5 * duration:.4f}",
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    str(frame_mid),
                ],
                "frame_mid",
            )
        elif frame_000.exists():
            frame_mid.write_bytes(frame_000.read_bytes())

        run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-sseof",
                "-1",
                "-i",
                str(video_path),
                "-vf",
                "reverse",
                "-frames:v",
                "1",
                str(frame_final),
            ],
            "frame_final",
        )

    existing = [p for p in [frame_000, frame_mid, frame_final] if p.exists()]
    if existing:
        fallback = existing[0]
        for p in [frame_000, frame_mid, frame_final]:
            if not p.exists():
                print(f"[warning] repairing missing {p.name} using {fallback.name}")
                p.write_bytes(fallback.read_bytes())

    return [str(frame_000), str(frame_mid), str(frame_final)]


def frames_without_final(frames: list[str]) -> list[str]:
    """
    For end-result questions, do not include the final frame.
    Usually returns [initial, mid].
    """
    if len(frames) >= 2:
        return frames[:2]
    return frames


def final_frame_only(frames: list[str]) -> list[str]:
    """
    For inverse/initial-value questions, include only the final observed result.
    """
    if len(frames) >= 3:
        return [frames[2]]
    if frames:
        return [frames[-1]]
    return []


def initial_frame_only(frames: list[str]) -> list[str]:
    """
    For pendulum initial-state questions, include only the initial frame.
    """
    if frames:
        return [frames[0]]
    return []

def relative_error(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    denom = max(abs(float(b)), 1.0e-12)
    return abs(float(a) - float(b)) / denom


def small_angle_is_accurate(
    estimated_period: float | None,
    analytic_period: float | None,
    *,
    rel_tol: float = PENDULUM_SMALL_ANGLE_REL_PERIOD_TOL,
) -> tuple[bool | None, dict[str, Any]]:
    """
    Label whether the small-angle approximation is accurate for this rollout.

    This becomes interesting across theta0_deg sweeps:
      small theta0 -> usually accurate
      large theta0 -> nonlinear period deviates
    """
    rel_err = relative_error(estimated_period, analytic_period)
    if rel_err is None:
        return None, {"small_angle_relative_period_error": None}

    return bool(rel_err <= rel_tol), {
        "small_angle_relative_period_error": rel_err,
        "small_angle_relative_period_tolerance": rel_tol,
    }


def pendulum_period_from_cfg(cfg: dict[str, Any]) -> float | None:
    if cfg.get("length") is None or cfg.get("gravity") is None:
        return None
    return 2.0 * math.pi * math.sqrt(float(cfg["length"]) / float(cfg["gravity"]))

def estimate_pendulum_period(t: np.ndarray, q: np.ndarray) -> float | None:
    """
    Estimate period from positive-slope zero crossings.
    Works best for low/moderate damping.
    """
    if len(t) < 3 or len(q) < 3:
        return None

    crossings = []
    for i in range(1, len(q)):
        if q[i - 1] < 0.0 <= q[i]:
            denom = q[i] - q[i - 1]
            if abs(denom) < 1e-12:
                crossings.append(float(t[i]))
            else:
                alpha = -q[i - 1] / denom
                crossings.append(float(t[i - 1] + alpha * (t[i] - t[i - 1])))

    if len(crossings) < 2:
        return None

    periods = np.diff(crossings)
    return float(np.median(periods))


def pendulum_keeps_swinging(
    q: np.ndarray,
    qdot: np.ndarray | None,
    *,
    initial_angle_rad: float | None,
) -> tuple[bool | None, dict[str, Any]]:
    """
    Operational label for "keeps swinging" at the end of the rollout.

    We use the last 20% of the angle trajectory. If the remaining oscillation
    amplitude is still at least 10% of the initial amplitude, or at least 0.02 rad,
    we label it as still swinging.
    """
    if q.size < 10:
        return None, {}

    start = int(0.8 * len(q))
    q_tail = q[start:]
    tail_amp = 0.5 * float(np.max(q_tail) - np.min(q_tail))
    final_abs_angle = float(abs(q[-1]))

    if initial_angle_rad is None:
        initial_amp = max(float(abs(q[0])), 1.0e-6)
    else:
        initial_amp = max(abs(float(initial_angle_rad)), 1.0e-6)

    amp_threshold = max(0.02, 0.10 * initial_amp)
    keeps = bool(tail_amp >= amp_threshold)

    details = {
        "tail_amplitude_rad": tail_amp,
        "initial_amplitude_rad": initial_amp,
        "keeps_swinging_threshold_rad": amp_threshold,
        "final_abs_angle_rad": final_abs_angle,
    }

    if qdot is not None and qdot.size:
        details["final_abs_angular_velocity"] = float(abs(qdot[-1]))

    return keeps, details


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    return dict(np.load(path, allow_pickle=True))


def entry_base(
    *,
    entry_id: str,
    scene_family: str,
    source_simulation_id: str,
    query_type: str,
    query: str,
    anchor_frames: list[str],
    latent_parameters: dict[str, Any],
    answer: str,
    answer_value: Any,
    ground_truth_metrics: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "entry_id": entry_id,
        "scene_family": scene_family,
        "source_simulation_id": source_simulation_id,
        "query_type": query_type,
        "query": query,
        "input_modality": "frames_plus_parameters",
        "anchor_frames": anchor_frames,
        "latent_parameters": latent_parameters,
        "answer": answer,
        "answer_value": answer_value,
        "ground_truth_metrics": ground_truth_metrics,
        "evaluation": evaluation,
    }

def pairwise_entry_base(
    *,
    entry_id: str,
    scene_family: str,
    source_simulation_ids: list[str],
    query_type: str,
    query: str,
    anchor_frames: dict[str, list[str]],
    latent_parameters: dict[str, dict[str, Any]],
    answer: str,
    answer_value: Any,
    ground_truth_metrics: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "entry_id": entry_id,
        "scene_family": scene_family,
        "source_simulation_ids": source_simulation_ids,
        "query_type": query_type,
        "query": query,
        "input_modality": "pairwise_frames_plus_parameters",
        "anchor_frames": anchor_frames,
        "latent_parameters": latent_parameters,
        "answer": answer,
        "answer_value": answer_value,
        "ground_truth_metrics": ground_truth_metrics,
        "evaluation": evaluation,
    }


def choose_pairwise_answer(
    value_a: float | None,
    value_b: float | None,
    *,
    larger_is_answer: bool = True,
    tie_tol: float = 1.0e-6,
) -> str | None:
    if value_a is None or value_b is None:
        return None

    if abs(value_a - value_b) <= tie_tol:
        return "Tie"

    if larger_is_answer:
        return "A" if value_a > value_b else "B"

    return "A" if value_a < value_b else "B"


def sparse_pairs(rows: list[dict[str, Any]], max_pairs_per_group: int = 3) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """
    Create a sparse set of low-vs-high pairs from a list of rollout records
    sorted by manipulated value.

    This avoids O(N^2) pair explosion.
    """
    if len(rows) < 2:
        return []

    rows = sorted(rows, key=lambda r: float(r["manipulated_value"]))

    candidate_pairs = [
        (rows[0], rows[-1]),
    ]

    if len(rows) >= 4:
        candidate_pairs.append((rows[0], rows[len(rows) // 2]))
        candidate_pairs.append((rows[len(rows) // 2], rows[-1]))

    return candidate_pairs[:max_pairs_per_group]

def infer_swept_parameter(manifest_row: dict[str, Any], cfg: dict[str, Any]) -> tuple[str | None, Any]:
    """
    Return the swept parameter name and its value.

    Prefer manifest's parameter_name / parameter_value. Fall back to
    resolved_config's sweep_parameter / sweep_value.
    """
    name = manifest_row.get("parameter_name") or manifest_row.get("causal_factor") or cfg.get("sweep_parameter")
    value = manifest_row.get("parameter_value")
    if value is None:
        value = cfg.get("sweep_value")
    return name, value


def latent_without_manipulated_parameter(
    latent: dict[str, Any],
    manipulated_parameter: str | None,
) -> dict[str, Any]:
    """
    For inverse parameter questions, hide the manipulated parameter from
    latent_parameters, because that is what the VLM should infer.
    """
    out = dict(latent)
    if manipulated_parameter is not None:
        out.pop(manipulated_parameter, None)
    out["parameter_to_infer"] = manipulated_parameter
    return out

def collect_rollout_record(
    *,
    dataset_root: Path,
    scene_key: str,
    manifest_row: dict[str, Any],
    output_root: Path,
    extract_frames_flag: bool,
) -> dict[str, Any] | None:
    try:
        sim_dir = resolve_sim_dir(dataset_root, manifest_row)
    except FileNotFoundError:
        return None

    metadata_path = sim_dir / "metadata.json"
    config_path = sim_dir / "resolved_config.json"

    if not metadata_path.exists() or not config_path.exists():
        return None

    metadata = read_json(metadata_path)
    cfg = read_json(config_path)

    scene_id = metadata.get("scene_id") or manifest_row.get("scene_id") or sim_dir.name
    manipulated_param, manipulated_value = infer_swept_parameter(manifest_row, cfg)

    video_path = find_existing_path(sim_dir / "video.mp4", sim_dir / "pendulum.mp4")
    frames = []
    if extract_frames_flag and video_path:
        frames = extract_anchor_frames(video_path, output_root / "frames" / scene_key / str(scene_id))

    record = {
        "scene_key": scene_key,
        "scene_id": str(scene_id),
        "sim_dir": str(sim_dir),
        "metadata": metadata,
        "cfg": cfg,
        "frames": frames,
        "manipulated_param": manipulated_param,
        "manipulated_value": safe_float(manipulated_value, None),
    }

    if scene_key in {"ramp_cup", "robotic_pour"}:
        result_path = sim_dir / "result.npz"
        result = load_npz(result_path)
        record["result"] = result
    elif scene_key == "pendulum":
        traj_path = sim_dir / "trajectory.npz"
        traj = load_npz(traj_path)
        record["traj"] = traj

    return record

def ramp_cup_pairwise_metrics(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record["metadata"]
    result = record.get("result", {})
    metrics = dict(metadata.get("metrics", {}))

    initial_cup = result.get("initial_cup_position")
    final_cup = result.get("final_cup_position")

    if initial_cup is not None and final_cup is not None:
        d = final_cup - initial_cup
        metrics.setdefault("cup_displacement_norm", float(np.linalg.norm(d)))

    max_tilt = safe_float(metrics.get("max_cup_tilt_degrees"), None)
    if max_tilt is None:
        max_tilt = np_scalar(result, "max_cup_tilt_degrees", None)

    cup_fell = metrics.get("cup_fell_over")
    if cup_fell is None and max_tilt is not None:
        cup_fell = bool(max_tilt >= CUP_FELL_OVER_TILT_THRESHOLD_DEG)

    return {
        "cup_displacement_norm": safe_float(metrics.get("cup_displacement_norm"), None),
        "max_cup_tilt_degrees": max_tilt,
        "cup_fell_over": cup_fell,
    }


def robotic_pour_pairwise_metrics(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record["metadata"]
    result = record.get("result", {})
    metrics = dict(metadata.get("metrics", {}))

    if not metrics and result:
        initial_count = int(np.asarray(result.get("initial_particle_count", 0)).item())
        in_receiver = int(np.asarray(result.get("final_particles_in_receiver", 0)).item())
        in_pourer = int(np.asarray(result.get("final_particles_in_pourer", 0)).item())
        metrics = {
            "receiver_fraction": in_receiver / max(1, initial_count),
            "pourer_fraction": in_pourer / max(1, initial_count),
        }

    return {
        "receiver_fraction": safe_float(metrics.get("receiver_fraction"), None),
        "pourer_fraction": safe_float(metrics.get("pourer_fraction"), None),
        "fills_half_receiver": bool((safe_float(metrics.get("receiver_fraction"), 0.0) or 0.0) >= POUR_HALF_RECEIVER_FRACTION),
    }


def pendulum_pairwise_metrics(record: dict[str, Any]) -> dict[str, Any]:
    cfg = record["cfg"]
    traj = record.get("traj", {})
    t = traj.get("t", np.array([]))
    q = traj.get("q", np.array([]))
    qdot = traj.get("qdot", np.array([]))

    period = estimate_pendulum_period(t, q)
    analytic_period = pendulum_period_from_cfg(cfg)

    theta0_deg = safe_float(cfg.get("theta0_deg"), None)
    theta0_rad = math.radians(theta0_deg) if theta0_deg is not None else None
    keeps_swinging, keeps_details = pendulum_keeps_swinging(q, qdot, initial_angle_rad=theta0_rad)

    return {
        "estimated_period": period,
        "analytic_period": analytic_period,
        "keeps_swinging": keeps_swinging,
        **keeps_details,
    }

def build_ramp_cup_pairwise_entries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries = []

    for a, b in sparse_pairs(records):
        ma = ramp_cup_pairwise_metrics(a)
        mb = ramp_cup_pairwise_metrics(b)

        latent_a = {
            "ball_density": a["cfg"].get("BALL_DENSITY"),
            "cup_density": a["cfg"].get("CUP_DENSITY"),
            "water_density": a["cfg"].get("WATER_DENSITY"),
            "water_viscosity": a["cfg"].get("WATER_VISCOSITY"),
            "fill_fraction": a["cfg"].get("TARGET_FILL_FRACTION"),
            "ramp_angle_deg": a["cfg"].get("RAMP_ANGLE_DEG"),
        }
        latent_b = {
            "ball_density": b["cfg"].get("BALL_DENSITY"),
            "cup_density": b["cfg"].get("CUP_DENSITY"),
            "water_density": b["cfg"].get("WATER_DENSITY"),
            "water_viscosity": b["cfg"].get("WATER_VISCOSITY"),
            "fill_fraction": b["cfg"].get("TARGET_FILL_FRACTION"),
            "ramp_angle_deg": b["cfg"].get("RAMP_ANGLE_DEG"),
        }

        answer = choose_pairwise_answer(
            ma["max_cup_tilt_degrees"],
            mb["max_cup_tilt_degrees"],
            larger_is_answer=True,
            tie_tol=5.0,
        )

        if answer is not None:
            entries.append(
                pairwise_entry_base(
                    entry_id=f"pair_rampcup_{a['scene_id']}_vs_{b['scene_id']}_fall",
                    scene_family="ramp_cup",
                    source_simulation_ids=[a["scene_id"], b["scene_id"]],
                    query_type="pairwise_counterfactual",
                    query="Which scenario is more likely to make the cup fall over, A or B?",
                    anchor_frames={
                        "A": frames_without_final(a["frames"]),
                        "B": frames_without_final(b["frames"]),
                    },
                    latent_parameters={"A": latent_a, "B": latent_b},
                    answer=answer,
                    answer_value=answer,
                    ground_truth_metrics={"A": ma, "B": mb},
                    evaluation={
                        "metric": "pairwise_accuracy",
                        "comparison_target": "max_cup_tilt_degrees",
                        "frame_policy": "no_final_frame_for_end_result_questions",
                    },
                )
            )

        answer = choose_pairwise_answer(
            ma["cup_displacement_norm"],
            mb["cup_displacement_norm"],
            larger_is_answer=True,
            tie_tol=0.02,
        )

        if answer is not None:
            entries.append(
                pairwise_entry_base(
                    entry_id=f"pair_rampcup_{a['scene_id']}_vs_{b['scene_id']}_move",
                    scene_family="ramp_cup",
                    source_simulation_ids=[a["scene_id"], b["scene_id"]],
                    query_type="pairwise_counterfactual",
                    query="Which scenario moves the cup farther after impact, A or B?",
                    anchor_frames={
                        "A": frames_without_final(a["frames"]),
                        "B": frames_without_final(b["frames"]),
                    },
                    latent_parameters={"A": latent_a, "B": latent_b},
                    answer=answer,
                    answer_value=answer,
                    ground_truth_metrics={"A": ma, "B": mb},
                    evaluation={
                        "metric": "pairwise_accuracy",
                        "comparison_target": "cup_displacement_norm",
                        "frame_policy": "no_final_frame_for_end_result_questions",
                    },
                )
            )

    return entries


def build_robotic_pour_pairwise_entries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries = []

    for a, b in sparse_pairs(records):
        ma = robotic_pour_pairwise_metrics(a)
        mb = robotic_pour_pairwise_metrics(b)

        latent_a = {
            "water_viscosity": a["cfg"].get("WATER_VISCOSITY"),
            "water_density": a["cfg"].get("WATER_DENSITY"),
            "water_fill_fraction": a["cfg"].get("WATER_FILL_FRACTION"),
            "liquid_surface_tension": a["cfg"].get("LIQUID_SURFACE_TENSION"),
            "glass_coup_friction": a["cfg"].get("GLASS_COUP_FRICTION"),
            "pour_hold_seconds": a["cfg"].get("POUR_HOLD_SECONDS"),
            "pour_pose_fraction": a["cfg"].get("POUR_POSE_FRACTION"),
            "max_tilt_deg": a["cfg"].get("MAX_TILT_DEG"),
        }
        latent_b = {
            "water_viscosity": b["cfg"].get("WATER_VISCOSITY"),
            "water_density": b["cfg"].get("WATER_DENSITY"),
            "water_fill_fraction": b["cfg"].get("WATER_FILL_FRACTION"),
            "liquid_surface_tension": b["cfg"].get("LIQUID_SURFACE_TENSION"),
            "glass_coup_friction": b["cfg"].get("GLASS_COUP_FRICTION"),
            "pour_hold_seconds": b["cfg"].get("POUR_HOLD_SECONDS"),
            "pour_pose_fraction": b["cfg"].get("POUR_POSE_FRACTION"),
            "max_tilt_deg": b["cfg"].get("MAX_TILT_DEG"),
        }

        answer = choose_pairwise_answer(
            ma["receiver_fraction"],
            mb["receiver_fraction"],
            larger_is_answer=True,
            tie_tol=0.05,
        )

        if answer is not None:
            entries.append(
                pairwise_entry_base(
                    entry_id=f"pair_pour_{a['scene_id']}_vs_{b['scene_id']}_receiver",
                    scene_family="robotic_pour",
                    source_simulation_ids=[a["scene_id"], b["scene_id"]],
                    query_type="pairwise_counterfactual",
                    query="Which scenario transfers more liquid into the receiving glass, A or B?",
                    anchor_frames={
                        "A": frames_without_final(a["frames"]),
                        "B": frames_without_final(b["frames"]),
                    },
                    latent_parameters={"A": latent_a, "B": latent_b},
                    answer=answer,
                    answer_value=answer,
                    ground_truth_metrics={"A": ma, "B": mb},
                    evaluation={
                        "metric": "pairwise_accuracy",
                        "comparison_target": "receiver_fraction",
                        "frame_policy": "no_final_frame_for_end_result_questions",
                    },
                )
            )

        answer = choose_pairwise_answer(
            ma["pourer_fraction"],
            mb["pourer_fraction"],
            larger_is_answer=True,
            tie_tol=0.05,
        )

        if answer is not None:
            entries.append(
                pairwise_entry_base(
                    entry_id=f"pair_pour_{a['scene_id']}_vs_{b['scene_id']}_retained",
                    scene_family="robotic_pour",
                    source_simulation_ids=[a["scene_id"], b["scene_id"]],
                    query_type="pairwise_counterfactual",
                    query="Which scenario leaves more liquid behind in the pouring glass, A or B?",
                    anchor_frames={
                        "A": frames_without_final(a["frames"]),
                        "B": frames_without_final(b["frames"]),
                    },
                    latent_parameters={"A": latent_a, "B": latent_b},
                    answer=answer,
                    answer_value=answer,
                    ground_truth_metrics={"A": ma, "B": mb},
                    evaluation={
                        "metric": "pairwise_accuracy",
                        "comparison_target": "pourer_fraction",
                        "frame_policy": "no_final_frame_for_end_result_questions",
                    },
                )
            )

    return entries


def build_pendulum_pairwise_entries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries = []

    for a, b in sparse_pairs(records):
        ma = pendulum_pairwise_metrics(a)
        mb = pendulum_pairwise_metrics(b)

        latent_a = {
            "length": a["cfg"].get("length"),
            "mass": a["cfg"].get("mass"),
            "bob_radius": a["cfg"].get("bob_radius"),
            "damping": a["cfg"].get("damping"),
            "theta0_deg": a["cfg"].get("theta0_deg"),
            "omega0": a["cfg"].get("omega0"),
            "gravity": a["cfg"].get("gravity"),
        }
        latent_b = {
            "length": b["cfg"].get("length"),
            "mass": b["cfg"].get("mass"),
            "bob_radius": b["cfg"].get("bob_radius"),
            "damping": b["cfg"].get("damping"),
            "theta0_deg": b["cfg"].get("theta0_deg"),
            "omega0": b["cfg"].get("omega0"),
            "gravity": b["cfg"].get("gravity"),
        }

        period_a = ma["estimated_period"] or ma["analytic_period"]
        period_b = mb["estimated_period"] or mb["analytic_period"]

        answer = choose_pairwise_answer(
            period_a,
            period_b,
            larger_is_answer=True,
            tie_tol=PENDULUM_SLOWER_PERIOD_TOL,
        )

        if answer is not None:
            entries.append(
                pairwise_entry_base(
                    entry_id=f"pair_pendulum_{a['scene_id']}_vs_{b['scene_id']}_slower",
                    scene_family="pendulum",
                    source_simulation_ids=[a["scene_id"], b["scene_id"]],
                    query_type="pairwise_counterfactual",
                    query="Which pendulum swings more slowly, A or B?",
                    anchor_frames={
                        "A": initial_frame_only(a["frames"]),
                        "B": initial_frame_only(b["frames"]),
                    },
                    latent_parameters={"A": latent_a, "B": latent_b},
                    answer=answer,
                    answer_value=answer,
                    ground_truth_metrics={"A": ma, "B": mb},
                    evaluation={
                        "metric": "pairwise_accuracy",
                        "comparison_target": "period",
                        "frame_policy": "initial_frame_only",
                    },
                )
            )

        # Mass invariance becomes meaningful as a pairwise synchronization question.
        if a["manipulated_param"] == "mass" and b["manipulated_param"] == "mass":
            if period_a is not None and period_b is not None:
                rel_diff = abs(period_a - period_b) / max(abs(period_a), abs(period_b), 1e-12)
                synchronized = bool(rel_diff <= PENDULUM_SYNC_REL_PERIOD_TOL)

                entries.append(
                    pairwise_entry_base(
                        entry_id=f"pair_pendulum_{a['scene_id']}_vs_{b['scene_id']}_sync_mass",
                        scene_family="pendulum",
                        source_simulation_ids=[a["scene_id"], b["scene_id"]],
                        query_type="pairwise_counterfactual",
                        query=(
                            "These two pendulums differ only in bob mass. "
                            "Will their visible swinging trajectories remain nearly synchronized?"
                        ),
                        anchor_frames={
                            "A": initial_frame_only(a["frames"]),
                            "B": initial_frame_only(b["frames"]),
                        },
                        latent_parameters={"A": latent_a, "B": latent_b},
                        answer="Yes" if synchronized else "No",
                        answer_value=synchronized,
                        ground_truth_metrics={
                            "A": ma,
                            "B": mb,
                            "relative_period_difference": rel_diff,
                        },
                        evaluation={
                            "metric": "binary_accuracy",
                            "positive_condition": f"relative_period_difference <= {PENDULUM_SYNC_REL_PERIOD_TOL}",
                            "frame_policy": "initial_frame_only",
                        },
                    )
                )

    return entries

def build_ramp_cup_entries(
    sim_dir: Path,
    manifest_row: dict[str, Any],
    frames_root: Path,
    extract_frames: bool,
) -> list[dict[str, Any]]:
    metadata_path = sim_dir / "metadata.json"
    config_path = sim_dir / "resolved_config.json"
    result_path = sim_dir / "result.npz"

    if not metadata_path.exists() or not config_path.exists() or not result_path.exists():
        return []

    metadata = read_json(metadata_path)
    cfg = read_json(config_path)
    result = load_npz(result_path)

    scene_id = metadata.get("scene_id") or manifest_row.get("scene_id") or sim_dir.name
    video_path = find_existing_path(sim_dir / "video.mp4", sim_dir / "pendulum.mp4")
    frames = []
    if extract_frames and video_path:
        frames = extract_anchor_frames(video_path, frames_root / str(scene_id))

    end_result_frames = frames_without_final(frames)
    inverse_frames = final_frame_only(frames)

    metrics = metadata.get("metrics", {})
    if not metrics:
        metrics = {}

    initial_cup = result.get("initial_cup_position")
    final_cup = result.get("final_cup_position")
    if initial_cup is not None and final_cup is not None:
        d = final_cup - initial_cup
        metrics.setdefault("cup_displacement_norm", float(np.linalg.norm(d)))
        metrics.setdefault("cup_delta_x", float(d[0]))
        metrics.setdefault("cup_delta_y", float(d[1]))
        metrics.setdefault("cup_delta_z", float(d[2]))

    max_tilt = safe_float(metrics.get("max_cup_tilt_degrees"), None)
    final_tilt = safe_float(metrics.get("final_cup_tilt_degrees"), None)

    if max_tilt is None:
        max_tilt = np_scalar(result, "max_cup_tilt_degrees", None)
        if max_tilt is not None:
            metrics["max_cup_tilt_degrees"] = max_tilt

    if final_tilt is None:
        final_tilt = np_scalar(result, "final_cup_tilt_degrees", None)
        if final_tilt is not None:
            metrics["final_cup_tilt_degrees"] = final_tilt

    cup_displacement = safe_float(metrics.get("cup_displacement_norm"), 0.0) or 0.0
    cup_contacted = bool(cup_displacement > CUP_CONTACT_DISPLACEMENT_THRESHOLD_M)
    cup_moved = bool(cup_displacement > CUP_MOVED_SIGNIFICANTLY_THRESHOLD_M)
    cup_moved_large = bool(cup_displacement > CUP_LARGE_MOTION_THRESHOLD_M)

    if metrics.get("cup_fell_over") is not None:
        cup_fell_over = bool(metrics["cup_fell_over"])
    elif max_tilt is not None:
        cup_fell_over = bool(max_tilt >= CUP_FELL_OVER_TILT_THRESHOLD_DEG)
    elif final_tilt is not None:
        cup_fell_over = bool(final_tilt >= CUP_FELL_OVER_TILT_THRESHOLD_DEG)
    else:
        cup_fell_over = None

    metrics["cup_contacted_rough"] = cup_contacted
    metrics["cup_moved"] = cup_moved
    metrics["large_cup_motion"] = cup_moved_large
    metrics["cup_fell_over"] = cup_fell_over

    if max_tilt is not None:
        metrics["cup_on_side"] = bool(max_tilt >= CUP_ON_SIDE_TILT_THRESHOLD_DEG)

    latent = {
        "ball_density": cfg.get("BALL_DENSITY"),
        "cup_density": cfg.get("CUP_DENSITY"),
        "water_density": cfg.get("WATER_DENSITY"),
        "water_viscosity": cfg.get("WATER_VISCOSITY"),
        "fill_fraction": cfg.get("TARGET_FILL_FRACTION"),
        "ramp_angle_deg": cfg.get("RAMP_ANGLE_DEG"),
    }

    manipulated_param, manipulated_value = infer_swept_parameter(manifest_row, cfg)

    entries = []

    # End-result questions: use initial + mid frames only, no final frame.
    if cup_fell_over is not None:
        entries.append(
            entry_base(
                entry_id=f"{scene_id}_rampcup_q_fall_over",
                scene_family="ramp_cup",
                source_simulation_id=str(scene_id),
                query_type="binary_outcome",
                query="Will the cup fall over after the ball impacts it?",
                anchor_frames=end_result_frames,
                latent_parameters=latent,
                answer="Yes" if cup_fell_over else "No",
                answer_value=bool(cup_fell_over),
                ground_truth_metrics=metrics,
                evaluation={
                    "metric": "binary_accuracy",
                    "positive_condition": f"max_cup_tilt_degrees >= {CUP_FELL_OVER_TILT_THRESHOLD_DEG}",
                    "frame_policy": "no_final_frame_for_end_result_questions",
                },
            )
        )

    entries.append(
        entry_base(
            entry_id=f"{scene_id}_rampcup_q_contact",
            scene_family="ramp_cup",
            source_simulation_id=str(scene_id),
            query_type="binary_outcome",
            query="Will the ball contact or noticeably disturb the cup?",
            anchor_frames=end_result_frames,
            latent_parameters=latent,
            answer="Yes" if cup_contacted else "No",
            answer_value=cup_contacted,
            ground_truth_metrics=metrics,
            evaluation={
                "metric": "binary_accuracy",
                "positive_condition": f"cup_displacement_norm > {CUP_CONTACT_DISPLACEMENT_THRESHOLD_M}",
                "note": "Uses cup displacement as a rough proxy for contact until explicit contact logging is added.",
                "frame_policy": "no_final_frame_for_end_result_questions",
            },
        )
    )

    entries.append(
        entry_base(
            entry_id=f"{scene_id}_rampcup_q_move",
            scene_family="ramp_cup",
            source_simulation_id=str(scene_id),
            query_type="binary_outcome",
            query="Will the ball impact cause the cup to move significantly?",
            anchor_frames=end_result_frames,
            latent_parameters=latent,
            answer="Yes" if cup_moved else "No",
            answer_value=cup_moved,
            ground_truth_metrics=metrics,
            evaluation={
                "metric": "binary_accuracy",
                "positive_condition": f"cup_displacement_norm > {CUP_MOVED_SIGNIFICANTLY_THRESHOLD_M}",
                "frame_policy": "no_final_frame_for_end_result_questions",
            },
        )
    )

    if max_tilt is not None:
        entries.append(
            entry_base(
                entry_id=f"{scene_id}_rampcup_q_max_tilt",
                scene_family="ramp_cup",
                source_simulation_id=str(scene_id),
                query_type="scalar_prediction",
                query="What is the maximum tilt angle of the cup in degrees after impact?",
                anchor_frames=end_result_frames,
                latent_parameters=latent,
                answer=f"{max_tilt:.2f}",
                answer_value=max_tilt,
                ground_truth_metrics=metrics,
                evaluation={
                    "metric": "absolute_error",
                    "tolerance": 10.0,
                    "frame_policy": "no_final_frame_for_end_result_questions",
                },
            )
        )

    # Inverse / initial-value questions:
    # Use final frame only. Hide the manipulated parameter from latent_parameters.
    inverse_latent = latent_without_manipulated_parameter(latent, manipulated_param)

    if manipulated_param is not None and manipulated_value is not None and cup_fell_over is True:
        entries.append(
            entry_base(
                entry_id=f"{scene_id}_rampcup_inv_make_fall_{manipulated_param}",
                scene_family="ramp_cup",
                source_simulation_id=str(scene_id),
                query_type="inverse_parameter_prediction",
                query=(
                    f"The final frame shows a result in which the cup fell over. "
                    f"What plausible value of the manipulated parameter `{manipulated_param}` "
                    f"would produce this outcome?"
                ),
                anchor_frames=inverse_frames,
                latent_parameters=inverse_latent,
                answer=str(manipulated_value),
                answer_value=manipulated_value,
                ground_truth_metrics=metrics,
                evaluation={
                    "metric": "absolute_or_log_error_for_numeric_parameter",
                    "target_parameter": manipulated_param,
                    "target_condition": "cup_fell_over == true",
                    "frame_policy": "final_frame_only_for_inverse_parameter_questions",
                },
            )
        )

    if manipulated_param is not None and manipulated_value is not None and cup_contacted and cup_fell_over is False:
        entries.append(
            entry_base(
                entry_id=f"{scene_id}_rampcup_inv_contact_not_fall_{manipulated_param}",
                scene_family="ramp_cup",
                source_simulation_id=str(scene_id),
                query_type="inverse_parameter_prediction",
                query=(
                    f"The final frame shows a result in which the ball contacted or disturbed "
                    f"the cup, but the cup did not fall over. What plausible value of the "
                    f"manipulated parameter `{manipulated_param}` would produce this outcome?"
                ),
                anchor_frames=inverse_frames,
                latent_parameters=inverse_latent,
                answer=str(manipulated_value),
                answer_value=manipulated_value,
                ground_truth_metrics=metrics,
                evaluation={
                    "metric": "absolute_or_log_error_for_numeric_parameter",
                    "target_parameter": manipulated_param,
                    "target_condition": "cup_contacted_rough == true and cup_fell_over == false",
                    "frame_policy": "final_frame_only_for_inverse_parameter_questions",
                    "note": "Contact is approximated by cup displacement until explicit contact logging is available.",
                },
            )
        )

    return entries


def build_robotic_pour_entries(
    sim_dir: Path,
    manifest_row: dict[str, Any],
    frames_root: Path,
    extract_frames: bool,
) -> list[dict[str, Any]]:
    metadata_path = sim_dir / "metadata.json"
    config_path = sim_dir / "resolved_config.json"
    result_path = sim_dir / "result.npz"

    if not metadata_path.exists() or not config_path.exists() or not result_path.exists():
        return []

    metadata = read_json(metadata_path)
    cfg = read_json(config_path)
    result = load_npz(result_path)

    scene_id = metadata.get("scene_id") or manifest_row.get("scene_id") or sim_dir.name
    video_path = find_existing_path(sim_dir / "video.mp4")
    frames = []
    if extract_frames and video_path:
        frames = extract_anchor_frames(video_path, frames_root / str(scene_id))

    # End-result questions exclude the final frame.
    end_result_frames = frames_without_final(frames)

    # Inverse/control-parameter questions use only the final result frame.
    inverse_frames = final_frame_only(frames)

    metrics = metadata.get("metrics", {})
    if not metrics:
        initial_count = int(np.asarray(result.get("initial_particle_count", 0)).item())
        in_receiver = int(np.asarray(result.get("final_particles_in_receiver", 0)).item())
        in_pourer = int(np.asarray(result.get("final_particles_in_pourer", 0)).item())
        live = int(np.asarray(result.get("final_live_particles", 0)).item())
        metrics = {
            "initial_particle_count": initial_count,
            "final_particles_in_receiver": in_receiver,
            "final_particles_in_pourer": in_pourer,
            "final_live_particles": live,
            "receiver_fraction": in_receiver / max(1, initial_count),
            "pourer_fraction": in_pourer / max(1, initial_count),
        }

    receiver_fraction = safe_float(metrics.get("receiver_fraction"), 0.0) or 0.0
    pourer_fraction = safe_float(metrics.get("pourer_fraction"), 0.0) or 0.0

    transferred_quarter = bool(receiver_fraction >= POUR_QUARTER_INITIAL_FRACTION)
    transferred_half = bool(receiver_fraction >= POUR_HALF_RECEIVER_FRACTION)
    about_quarter = bool(
        abs(receiver_fraction - POUR_QUARTER_INITIAL_FRACTION) <= POUR_TARGET_TOLERANCE
    )

    # Prefer measured max_tilt_degrees from simulation metrics if available.
    # Fall back to the configured max tilt. If the script uses POUR_POSE_FRACTION
    # as an interpolation toward the full-pour pose, this effective value is only
    # an approximate control-side angle.
    measured_max_tilt = safe_float(metrics.get("max_tilt_degrees"), None)
    cfg_max_tilt = safe_float(cfg.get("MAX_TILT_DEG"), None)
    pour_pose_fraction = safe_float(cfg.get("POUR_POSE_FRACTION"), None)

    if measured_max_tilt is not None:
        effective_tilt_deg = measured_max_tilt
    elif cfg_max_tilt is not None and pour_pose_fraction is not None:
        effective_tilt_deg = cfg_max_tilt * pour_pose_fraction
    else:
        effective_tilt_deg = cfg_max_tilt

    if effective_tilt_deg is not None:
        metrics["effective_tilt_degrees_for_label"] = effective_tilt_deg

    latent = {
        "water_viscosity": cfg.get("WATER_VISCOSITY"),
        "water_density": cfg.get("WATER_DENSITY"),
        "water_fill_fraction": cfg.get("WATER_FILL_FRACTION"),
        "liquid_surface_tension": cfg.get("LIQUID_SURFACE_TENSION"),
        "glass_coup_friction": cfg.get("GLASS_COUP_FRICTION"),
        "pour_hold_seconds": cfg.get("POUR_HOLD_SECONDS"),
        "pour_pose_fraction": cfg.get("POUR_POSE_FRACTION"),
        "max_tilt_deg": cfg.get("MAX_TILT_DEG"),
        "effective_tilt_degrees": effective_tilt_deg,
    }

    manipulated_param, manipulated_value = infer_swept_parameter(manifest_row, cfg)

    entries = []

    # -------------------------------------------------------------------------
    # Existing end-result questions, now using named thresholds.
    # -------------------------------------------------------------------------

    entries.append(
        entry_base(
            entry_id=f"{scene_id}_pour_q_half",
            scene_family="robotic_pour",
            source_simulation_id=str(scene_id),
            query_type="binary_outcome",
            query="Will this pour transfer at least half of the initial liquid into the receiving glass?",
            anchor_frames=end_result_frames,
            latent_parameters=latent,
            answer="Yes" if transferred_half else "No",
            answer_value=transferred_half,
            ground_truth_metrics=metrics,
            evaluation={
                "metric": "binary_accuracy",
                "positive_condition": f"receiver_fraction >= {POUR_HALF_RECEIVER_FRACTION}",
                "frame_policy": "no_final_frame_for_end_result_questions",
            },
        )
    )

    entries.append(
        entry_base(
            entry_id=f"{scene_id}_pour_q_fraction",
            scene_family="robotic_pour",
            source_simulation_id=str(scene_id),
            query_type="scalar_prediction",
            query="What fraction of the initial liquid ends in the receiving glass?",
            anchor_frames=end_result_frames,
            latent_parameters=latent,
            answer=f"{receiver_fraction:.4f}",
            answer_value=receiver_fraction,
            ground_truth_metrics=metrics,
            evaluation={
                "metric": "absolute_error",
                "tolerance": 0.05,
                "frame_policy": "no_final_frame_for_end_result_questions",
            },
        )
    )

    entries.append(
        entry_base(
            entry_id=f"{scene_id}_pour_q_retained",
            scene_family="robotic_pour",
            source_simulation_id=str(scene_id),
            query_type="scalar_prediction",
            query="What fraction of the initial liquid remains in the pouring glass?",
            anchor_frames=end_result_frames,
            latent_parameters=latent,
            answer=f"{pourer_fraction:.4f}",
            answer_value=pourer_fraction,
            ground_truth_metrics=metrics,
            evaluation={
                "metric": "absolute_error",
                "tolerance": 0.05,
                "frame_policy": "no_final_frame_for_end_result_questions",
            },
        )
    )

    entries.append(
        entry_base(
            entry_id=f"{scene_id}_pour_q_some_transfer",
            scene_family="robotic_pour",
            source_simulation_id=str(scene_id),
            query_type="binary_outcome",
            query="Will a noticeable amount of liquid, at least one quarter of the initial liquid, reach the receiving glass?",
            anchor_frames=end_result_frames,
            latent_parameters=latent,
            answer="Yes" if transferred_quarter else "No",
            answer_value=transferred_quarter,
            ground_truth_metrics=metrics,
            evaluation={
                "metric": "binary_accuracy",
                "positive_condition": f"receiver_fraction >= {POUR_QUARTER_INITIAL_FRACTION}",
                "frame_policy": "no_final_frame_for_end_result_questions",
            },
        )
    )

    # -------------------------------------------------------------------------
    # New requested end-result / action-condition questions.
    # These exclude the final frame.
    # -------------------------------------------------------------------------

    entries.append(
        entry_base(
            entry_id=f"{scene_id}_pour_q_duration_fills_half_receiver",
            scene_family="robotic_pour",
            source_simulation_id=str(scene_id),
            query_type="binary_outcome",
            query="Will this cup tilt duration transfer enough liquid to fill half of the receiving glass?",
            anchor_frames=end_result_frames,
            latent_parameters=latent,
            answer="Yes" if transferred_half else "No",
            answer_value=transferred_half,
            ground_truth_metrics=metrics,
            evaluation={
                "metric": "binary_accuracy",
                "positive_condition": f"receiver_fraction >= {POUR_HALF_RECEIVER_FRACTION}",
                "frame_policy": "no_final_frame_for_end_result_questions",
                "note": "Uses receiver_fraction as the current proxy for receiving-glass fill fraction.",
            },
        )
    )

    entries.append(
        entry_base(
            entry_id=f"{scene_id}_pour_q_tilt_angle_fills_half_receiver",
            scene_family="robotic_pour",
            source_simulation_id=str(scene_id),
            query_type="binary_outcome",
            query="Will this tilt angle pour enough liquid to fill half of the receiving glass?",
            anchor_frames=end_result_frames,
            latent_parameters=latent,
            answer="Yes" if transferred_half else "No",
            answer_value=transferred_half,
            ground_truth_metrics=metrics,
            evaluation={
                "metric": "binary_accuracy",
                "positive_condition": f"receiver_fraction >= {POUR_HALF_RECEIVER_FRACTION}",
                "frame_policy": "no_final_frame_for_end_result_questions",
                "note": "Uses receiver_fraction as the current proxy for receiving-glass fill fraction.",
            },
        )
    )

    # -------------------------------------------------------------------------
    # Inverse / control-value questions.
    # These use the final frame only and hide the manipulated parameter when
    # that parameter is the thing being inferred.
    # -------------------------------------------------------------------------

    # “How long should the robot hold the cup tilted to pour a quarter of the initial liquid?”
    #
    # This is only well-posed when the sweep actually manipulated pour_hold_seconds.
    # We add the entry for rollouts that reached roughly the target transfer.
    if manipulated_param == "pour_hold_seconds" and manipulated_value is not None and about_quarter:
        inverse_latent = latent_without_manipulated_parameter(latent, manipulated_param)
        entries.append(
            entry_base(
                entry_id=f"{scene_id}_pour_inv_hold_time_quarter_initial",
                scene_family="robotic_pour",
                source_simulation_id=str(scene_id),
                query_type="inverse_parameter_prediction",
                query="How long should the robot hold the cup tilted to pour a quarter of the initial liquid?",
                anchor_frames=inverse_frames,
                latent_parameters=inverse_latent,
                answer=str(manipulated_value),
                answer_value=manipulated_value,
                ground_truth_metrics=metrics,
                evaluation={
                    "metric": "absolute_error",
                    "target_parameter": "pour_hold_seconds",
                    "target_condition": (
                        f"abs(receiver_fraction - {POUR_QUARTER_INITIAL_FRACTION}) "
                        f"<= {POUR_TARGET_TOLERANCE}"
                    ),
                    "tolerance": 0.25,
                    "frame_policy": "final_frame_only_for_inverse_parameter_questions",
                    "note": "Generated only for pour_hold_seconds sweeps close to the quarter-transfer target.",
                },
            )
        )

    # “What tilt angle is required to fill about a quarter of the receiving glass?”
    #
    # This is best posed when the sweep manipulated pour_pose_fraction or a tilt
    # parameter. The answer is reported as an effective tilt angle in degrees.
    # If measured max_tilt_degrees is available, it is preferred.
    tilt_manipulated = manipulated_param in {
        "pour_pose_fraction",
        "max_tilt_deg",
        "max_tilt_degrees",
    }

    if tilt_manipulated and effective_tilt_deg is not None and about_quarter:
        inverse_latent = latent_without_manipulated_parameter(latent, manipulated_param)
        # Hide effective_tilt_degrees too, since that is effectively the answer.
        inverse_latent.pop("effective_tilt_degrees", None)
        inverse_latent["parameter_to_infer"] = "effective_tilt_degrees"

        entries.append(
            entry_base(
                entry_id=f"{scene_id}_pour_inv_tilt_angle_quarter_receiver",
                scene_family="robotic_pour",
                source_simulation_id=str(scene_id),
                query_type="inverse_parameter_prediction",
                query="What tilt angle is required to fill about a quarter of the receiving glass?",
                anchor_frames=inverse_frames,
                latent_parameters=inverse_latent,
                answer=f"{effective_tilt_deg:.4f}",
                answer_value=effective_tilt_deg,
                ground_truth_metrics=metrics,
                evaluation={
                    "metric": "absolute_error",
                    "target_parameter": "effective_tilt_degrees",
                    "target_condition": (
                        f"abs(receiver_fraction - {POUR_QUARTER_INITIAL_FRACTION}) "
                        f"<= {POUR_TARGET_TOLERANCE}"
                    ),
                    "tolerance": 5.0,
                    "frame_policy": "final_frame_only_for_inverse_parameter_questions",
                    "note": (
                        "Generated only for tilt/pose sweeps close to the quarter-transfer target. "
                        "Uses measured max_tilt_degrees if available; otherwise approximates from configuration."
                    ),
                },
            )
        )

    return entries

def build_pendulum_entries(
    sim_dir: Path,
    manifest_row: dict[str, Any],
    frames_root: Path,
    extract_frames: bool,
) -> list[dict[str, Any]]:
    metadata_path = sim_dir / "metadata.json"
    config_path = sim_dir / "resolved_config.json"
    traj_path = sim_dir / "trajectory.npz"

    if not metadata_path.exists() or not config_path.exists() or not traj_path.exists():
        return []

    metadata = read_json(metadata_path)
    cfg = read_json(config_path)
    traj = load_npz(traj_path)

    scene_id = metadata.get("scene_id") or manifest_row.get("scene_id") or sim_dir.name
    video_path = find_existing_path(sim_dir / "pendulum.mp4", sim_dir / "video.mp4")
    frames = []
    if extract_frames and video_path:
        frames = extract_anchor_frames(video_path, frames_root / str(scene_id))

    init_frames = initial_frame_only(frames)
    end_result_frames = frames_without_final(frames)

    t = traj.get("t", np.array([]))
    q = traj.get("q", np.array([]))
    qdot = traj.get("qdot", np.array([]))

    period = estimate_pendulum_period(t, q)
    analytic_period = pendulum_period_from_cfg(cfg)

    q_abs_max = float(np.max(np.abs(q))) if q.size else None
    q_abs_final = float(abs(q[-1])) if q.size else None

    theta0_deg = safe_float(cfg.get("theta0_deg"), None)
    theta0_rad = math.radians(theta0_deg) if theta0_deg is not None else None

    keeps_swinging, keeps_details = pendulum_keeps_swinging(
        q,
        qdot,
        initial_angle_rad=theta0_rad,
    )

    small_angle_ok, small_angle_details = small_angle_is_accurate(
        period,
        analytic_period,
    )

    metrics = {
        "estimated_period": period,
        "analytic_small_angle_period": analytic_period,
        "max_abs_angle_rad": q_abs_max,
        "final_abs_angle_rad": q_abs_final,
        **keeps_details,
        **small_angle_details,
    }

    latent = {
        "length": cfg.get("length"),
        "mass": cfg.get("mass"),
        "bob_radius": cfg.get("bob_radius"),
        "damping": cfg.get("damping"),
        "theta0_deg": cfg.get("theta0_deg"),
        "omega0": cfg.get("omega0"),
        "gravity": cfg.get("gravity"),
    }

    manipulated_param, manipulated_value = infer_swept_parameter(manifest_row, cfg)

    entries = []

    # 1. Damping / dissipation: this is a real rollout prediction.
    if keeps_swinging is not None:
        entries.append(
            entry_base(
                entry_id=f"{scene_id}_pendulum_q_keeps_swinging",
                scene_family="pendulum",
                source_simulation_id=str(scene_id),
                query_type="binary_outcome",
                query=(
                    "Given the initial frame and latent physical parameters, "
                    "will the pendulum still have visible oscillation near the end of the rollout?"
                ),
                anchor_frames=init_frames,
                latent_parameters=latent,
                answer="Yes" if keeps_swinging else "No",
                answer_value=bool(keeps_swinging),
                ground_truth_metrics=metrics,
                evaluation={
                    "metric": "binary_accuracy",
                    "positive_condition": "tail_amplitude_rad >= max(0.02, 0.10 * initial_amplitude_rad)",
                    "frame_policy": "initial_frame_only",
                },
            )
        )

    # 2. Period prediction: trajectory-level, not just conceptual.
    if period is not None:
        entries.append(
            entry_base(
                entry_id=f"{scene_id}_pendulum_q_period",
                scene_family="pendulum",
                source_simulation_id=str(scene_id),
                query_type="scalar_prediction",
                query=(
                    "Given the initial frame and latent physical parameters, "
                    "what is the approximate oscillation period of the pendulum in seconds?"
                ),
                anchor_frames=init_frames,
                latent_parameters=latent,
                answer=f"{period:.4f}",
                answer_value=period,
                ground_truth_metrics=metrics,
                evaluation={
                    "metric": "absolute_error",
                    "tolerance": 0.10,
                    "frame_policy": "initial_frame_only",
                },
            )
        )

    # 3. Length/gravity causal effect as an inverse/single-rollout timing question.
    # This is still single-rollout, but it probes the consequence of L and g.
    if analytic_period is not None:
        entries.append(
            entry_base(
                entry_id=f"{scene_id}_pendulum_q_time_to_next_swing",
                scene_family="pendulum",
                source_simulation_id=str(scene_id),
                query_type="scalar_prediction",
                query=(
                    "Using the given initial state and physical parameters, "
                    "approximately how long should one full swing period take?"
                ),
                anchor_frames=init_frames,
                latent_parameters=latent,
                answer=f"{analytic_period:.4f}",
                answer_value=analytic_period,
                ground_truth_metrics=metrics,
                evaluation={
                    "metric": "absolute_error",
                    "tolerance": 0.10,
                    "frame_policy": "initial_frame_only",
                    "note": "Uses the small-angle formula as a clean physics reference.",
                },
            )
        )

    # 4. Theta0 / regime-crossing question.
    if small_angle_ok is not None:
        entries.append(
            entry_base(
                entry_id=f"{scene_id}_pendulum_q_small_angle_valid",
                scene_family="pendulum",
                source_simulation_id=str(scene_id),
                query_type="binary_outcome",
                query=(
                    "Will the small-angle approximation accurately predict this pendulum's period?"
                ),
                anchor_frames=init_frames,
                latent_parameters=latent,
                answer="Yes" if small_angle_ok else "No",
                answer_value=bool(small_angle_ok),
                ground_truth_metrics=metrics,
                evaluation={
                    "metric": "binary_accuracy",
                    "positive_condition": (
                        f"relative period error <= {PENDULUM_SMALL_ANGLE_REL_PERIOD_TOL}"
                    ),
                    "frame_policy": "initial_frame_only",
                },
            )
        )

    # 5. Inverse parameter question for the manipulated parameter.
    # This asks the model to infer a plausible physical value from the observed final result.
    if manipulated_param in {"length", "gravity", "damping", "theta0_deg"} and manipulated_value is not None:
        inverse_latent = latent_without_manipulated_parameter(latent, manipulated_param)

        entries.append(
            entry_base(
                entry_id=f"{scene_id}_pendulum_inv_{manipulated_param}",
                scene_family="pendulum",
                source_simulation_id=str(scene_id),
                query_type="inverse_parameter_prediction",
                query=(
                    f"The final frame shows the pendulum after evolving from the initial condition. "
                    f"What plausible value of the manipulated parameter `{manipulated_param}` "
                    f"would produce this outcome?"
                ),
                anchor_frames=final_frame_only(frames),
                latent_parameters=inverse_latent,
                answer=str(manipulated_value),
                answer_value=manipulated_value,
                ground_truth_metrics=metrics,
                evaluation={
                    "metric": "absolute_or_log_error_for_numeric_parameter",
                    "target_parameter": manipulated_param,
                    "frame_policy": "final_frame_only_for_inverse_parameter_questions",
                },
            )
        )

    return entries

def build_entries_for_sweep(
    dataset_root: Path,
    scene_key: str,
    output_root: Path,
    extract_frames_flag: bool,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    manifest_path = dataset_root / "manifest.jsonl"
    if not manifest_path.exists():
        print(f"[warning] Missing manifest: {manifest_path}")
        return []

    rows = read_jsonl(manifest_path)
    if limit is not None:
        rows = rows[:limit]

    all_entries: list[dict[str, Any]] = []
    rollout_records: list[dict[str, Any]] = []
    frames_root = output_root / "frames" / scene_key

    for row in rows:
        try:
            sim_dir = resolve_sim_dir(dataset_root, row)
        except FileNotFoundError as exc:
            print(f"[warning] {exc}")
            continue

        if scene_key == "ramp_cup":
            entries = build_ramp_cup_entries(sim_dir, row, frames_root, extract_frames_flag)
        elif scene_key == "robotic_pour":
            entries = build_robotic_pour_entries(sim_dir, row, frames_root, extract_frames_flag)
        elif scene_key == "pendulum":
            entries = build_pendulum_entries(sim_dir, row, frames_root, extract_frames_flag)
        else:
            raise ValueError(f"Unknown scene key: {scene_key}")

        all_entries.extend(entries)

        record = collect_rollout_record(
            dataset_root=dataset_root,
            scene_key=scene_key,
            manifest_row=row,
            output_root=output_root,
            extract_frames_flag=extract_frames_flag,
        )
        if (
            record is not None
            and record["manipulated_param"] is not None
            and record["manipulated_value"] is not None
        ):
            rollout_records.append(record)

    # Group by manipulated parameter and seed so each pair differs along the
    # intended one-variable sweep axis.
    grouped: dict[tuple[str, int | str], list[dict[str, Any]]] = defaultdict(list)
    for rec in rollout_records:
        seed = rec["metadata"].get("seed", rec["cfg"].get("seed", "unknown"))
        grouped[(str(rec["manipulated_param"]), seed)].append(rec)

    pairwise_entries: list[dict[str, Any]] = []

    for (_param, _seed), group in grouped.items():
        # Keep pairwise comparisons sparse:
        #   - low vs high
        #   - low vs middle
        #   - middle vs high
        #
        # The sparse_pairs(...) helper caps each parameter × seed group at 3
        # comparison pairs, avoiding all O(N^2) pair combinations.
        if scene_key == "ramp_cup":
            pairwise_entries.extend(build_ramp_cup_pairwise_entries(group))
        elif scene_key == "robotic_pour":
            pairwise_entries.extend(build_robotic_pour_pairwise_entries(group))
        elif scene_key == "pendulum":
            pairwise_entries.extend(build_pendulum_pairwise_entries(group))

    print(f"{scene_key}: built {len(pairwise_entries)} pairwise entries")
    all_entries.extend(pairwise_entries)

    return all_entries


def summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    by_scene = Counter(e["scene_family"] for e in entries)
    by_query_type = Counter(e["query_type"] for e in entries)

    by_scene_query = defaultdict(Counter)
    for e in entries:
        by_scene_query[e["scene_family"]][e["query_type"]] += 1

    return {
        "num_vlm_entries": len(entries),
        "num_entries_by_scene": dict(by_scene),
        "num_entries_by_query_type": dict(by_query_type),
        "num_entries_by_scene_and_query_type": {
            scene: dict(counter) for scene, counter in by_scene_query.items()
        },
        "notes": [
            "This dataset is derived from completed simulation sweeps.",
            "End-result questions exclude the final frame.",
            "Ramp-cup inverse parameter questions use the final frame only.",
            "Pendulum keeps-swinging questions use the initial frame only.",
            "Each entry contains a query, latent parameters, anchor frame paths if available, ground-truth answer, and evaluation metadata.",
            "Pairwise comparisons are sparse: at most low-vs-high, low-vs-mid, and mid-vs-high per scene x manipulated parameter x seed group.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output-root", type=Path, default=Path("datasets/vlm_prompt_dataset"))
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=["ramp_cup", "robotic_pour", "pendulum"],
        choices=sorted(SWEEP_DIRS),
    )
    parser.add_argument("--no-frame-extraction", action="store_true")
    parser.add_argument(
        "--limit-per-sweep",
        type=int,
        default=None,
        help="Optional debugging limit on manifest rows per sweep folder.",
    )
    args = parser.parse_args()

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []

    for scene_key in args.scenes:
        sweep_dir = args.datasets_root / SWEEP_DIRS[scene_key]
        if not sweep_dir.exists():
            print(f"[warning] Missing sweep directory: {sweep_dir}")
            continue

        scene_entries = build_entries_for_sweep(
            dataset_root=sweep_dir,
            scene_key=scene_key,
            output_root=output_root,
            extract_frames_flag=not args.no_frame_extraction,
            limit=args.limit_per_sweep,
        )
        print(f"{scene_key}: built {len(scene_entries)} VLM entries")
        entries.extend(scene_entries)

    output_jsonl = output_root / "vlm_entries.jsonl"
    write_jsonl(entries, output_jsonl)

    summary = summarize(entries)
    summary_path = output_root / "dataset_card_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(f"\nWrote {output_jsonl}")
    print(f"Wrote {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()