#!/usr/bin/env python3
"""
Evaluate VLM benchmark entries with:
  1. Gemini directly on anchor frames.
  2. Exported diffusion jobs for Colab.
  3. Gemini judging of diffusion-generated videos after Colab runs.

This script does NOT require Colab for Gemini evaluation.
Colab is only used for running the diffusion model.

Typical workflow:

Local/HPC:
  uv run scripts/evaluate_vlm_dataset.py \
    --mode gemini \
    --vlm-jsonl datasets/vlm_prompt_dataset/vlm_entries.jsonl \
    --output-dir evaluation_outputs \
    --gemini-model gemini-2.5-flash \
    --limit 20

Local/HPC, export diffusion jobs:
  uv run scripts/evaluate_vlm_dataset.py \
    --mode export_diffusion_jobs \
    --vlm-jsonl datasets/vlm_prompt_dataset/vlm_entries.jsonl \
    --output-dir evaluation_outputs \
    --limit 20

Upload this folder to Colab:
  evaluation_outputs/colab_diffusion_assets/

Colab:
  python colab_run_ltx_diffusion_jobs.py \
    --assets-dir /content/colab_diffusion_assets \
    --ltx-root /content/LTX-2

Local/HPC, after downloading Colab outputs:
  uv run scripts/evaluate_vlm_dataset.py \
    --mode judge_diffusion \
    --vlm-jsonl datasets/vlm_prompt_dataset/vlm_entries.jsonl \
    --output-dir evaluation_outputs \
    --colab-results-jsonl evaluation_outputs/colab_diffusion_assets/diffusion_raw_results.jsonl \
    --generated-video-root evaluation_outputs/colab_diffusion_assets/generated_videos \
    --gemini-model gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# -----------------------------
# IO helpers
# -----------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def load_existing_ids(path: Path) -> set[str]:
    return {str(row["entry_id"]) for row in read_jsonl(path) if "entry_id" in row}


def read_image_bytes(path: Path) -> bytes:
    with path.open("rb") as f:
        return f.read()


def ensure_abs_path(path_like: str | Path, base: Path | None = None) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    if base is not None:
        candidate = base / p
        if candidate.exists():
            return candidate
    return p.resolve()


def compact_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


# -----------------------------
# Prompt construction
# -----------------------------


def build_gemini_prompt(entry: dict[str, Any], *, judging_diffusion: bool = False) -> str:
    latent = entry.get("latent_parameters", {})
    evaluation = entry.get("evaluation", {})
    answer_value_type = type(entry.get("answer_value")).__name__

    context_line = (
        "You are evaluating a generated video from a world model."
        if judging_diffusion
        else "You are evaluating a simulated physical scenario from anchor frame(s)."
    )

    return f"""
{context_line}

Task:
Answer the question using only the provided visual input and latent physical parameters.

Question:
{entry.get("query")}

Latent physical parameters:
{json.dumps(latent, indent=2, sort_keys=True)}

Query type:
{entry.get("query_type")}

Evaluation metadata:
{json.dumps(evaluation, indent=2, sort_keys=True)}

Return ONLY valid JSON in this schema:
{{
  "answer": string,
  "answer_value": boolean | number | string,
  "confidence": number,
  "rationale": string
}}

Rules:
- For binary questions, answer_value must be true or false.
- For scalar or inverse-parameter questions, answer_value must be a number when possible.
- Keep rationale short.
- Do not include markdown fences.
- Expected answer_value type for scoring is approximately: {answer_value_type}.
""".strip()


def build_diffusion_prompt(entry: dict[str, Any]) -> str:
    latent = entry.get("latent_parameters", {})
    scene = entry.get("scene_family", "")
    query = entry.get("query", "")

    if scene == "ramp_cup":
        return (
            "Continue the physical scene from the provided frame. "
            "A ball rolls down the ramp under gravity and interacts with a cup containing liquid. "
            "Respect rigid body contact, momentum transfer, cup stability, and fluid sloshing. "
            f"Latent parameters: {compact_json(latent)}. "
            f"The downstream question is: {query}"
        )

    if scene == "robotic_pour":
        return (
            "Continue the physical scene from the provided frame. "
            "A robot arm pours liquid from one glass into a receiving glass. "
            "Respect viscosity, density, gravity, fluid transfer, and spillage. "
            f"Latent parameters: {compact_json(latent)}. "
            f"The downstream question is: {query}"
        )

    if scene == "pendulum":
        return (
            "Continue the physical scene from the provided frame. "
            "A pendulum swings under gravity with the specified length, damping, mass, and initial angle. "
            "Respect pendulum dynamics and damping. "
            f"Latent parameters: {compact_json(latent)}. "
            f"The downstream question is: {query}"
        )

    return (
        "Continue the physical scene from the provided frame while respecting the latent physical parameters. "
        f"Latent parameters: {compact_json(latent)}. "
        f"The downstream question is: {query}"
    )


# -----------------------------
# Parsing and scoring
# -----------------------------


def parse_jsonish(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {
        "answer": text,
        "answer_value": text,
        "confidence": None,
        "rationale": "Could not parse strict JSON.",
    }


def normalize_bool(x: Any) -> bool | None:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)) and x in (0, 1):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"yes", "true", "y", "1"}:
            return True
        if s in {"no", "false", "n", "0"}:
            return False
    return None


def normalize_float(x: Any) -> float | None:
    if isinstance(x, (int, float, np.number)):
        return float(x)
    if isinstance(x, str):
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", x)
        if match:
            return float(match.group(0))
    return None


def score_prediction(entry: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    gt = entry.get("answer_value")
    pred_value = pred.get("answer_value", pred.get("answer"))
    evaluation = entry.get("evaluation", {})
    metric = evaluation.get("metric", "")
    query_type = entry.get("query_type", "")

    if query_type in {"binary_outcome", "conceptual_binary"} or metric == "binary_accuracy":
        gt_bool = normalize_bool(gt)
        pred_bool = normalize_bool(pred_value)
        correct = gt_bool is not None and pred_bool is not None and gt_bool == pred_bool
        return {
            "score_type": "binary_accuracy",
            "correct": bool(correct),
            "gt_normalized": gt_bool,
            "pred_normalized": pred_bool,
        }

    if query_type in {"scalar_prediction", "inverse_parameter_prediction"} or "absolute" in metric:
        gt_float = normalize_float(gt)
        pred_float = normalize_float(pred_value)

        if gt_float is None or pred_float is None:
            return {
                "score_type": "numeric",
                "correct": False,
                "absolute_error": None,
                "relative_error": None,
                "gt_normalized": gt_float,
                "pred_normalized": pred_float,
            }

        abs_err = abs(pred_float - gt_float)
        rel_err = abs_err / max(abs(gt_float), 1e-12)

        if "log" in metric and gt_float > 0 and pred_float > 0:
            log_err = abs(math.log10(pred_float) - math.log10(gt_float))
            correct = log_err <= 0.5
            return {
                "score_type": "numeric_log",
                "correct": bool(correct),
                "absolute_error": abs_err,
                "relative_error": rel_err,
                "log10_error": log_err,
                "gt_normalized": gt_float,
                "pred_normalized": pred_float,
            }

        tolerance = evaluation.get("tolerance")
        correct = abs_err <= float(tolerance) if tolerance is not None else None
        return {
            "score_type": "numeric",
            "correct": correct,
            "absolute_error": abs_err,
            "relative_error": rel_err,
            "gt_normalized": gt_float,
            "pred_normalized": pred_float,
        }

    return {
        "score_type": "unscored",
        "correct": None,
        "gt_normalized": gt,
        "pred_normalized": pred_value,
    }


# -----------------------------
# Gemini
# -----------------------------


@dataclass
class GeminiConfig:
    model: str
    api_key_env: str
    sleep_seconds: float
    max_retries: int


def init_gemini_client(cfg: GeminiConfig):
    from google import genai

    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing Gemini API key. Set {cfg.api_key_env}=...")
    return genai.Client(api_key=api_key)


def gemini_generate_once(
    client: Any,
    *,
    model: str,
    prompt: str,
    image_paths: list[Path],
) -> dict[str, Any]:
    from google.genai import types

    contents: list[Any] = [prompt]
    for path in image_paths:
        if path.exists():
            contents.append(
                types.Part.from_bytes(
                    data=read_image_bytes(path),
                    mime_type="image/png",
                )
            )

    response = client.models.generate_content(model=model, contents=contents)
    text = getattr(response, "text", "") or ""
    return {
        "model_used": model,
        "raw_text": text,
        "parsed": parse_jsonish(text),
    }


def gemini_generate_with_retries(
    client: Any,
    *,
    cfg: GeminiConfig,
    prompt: str,
    image_paths: list[Path],
) -> dict[str, Any]:
    last_exc: Exception | None = None

    for attempt in range(cfg.max_retries + 1):
        try:
            return gemini_generate_once(
                client,
                model=cfg.model,
                prompt=prompt,
                image_paths=image_paths,
            )
        except Exception as exc:
            last_exc = exc
            msg = repr(exc)
            retry_sleep = cfg.sleep_seconds

            # Try to respect "retry in N seconds" if present.
            match = re.search(r"retry in ([0-9.]+)s", msg, flags=re.IGNORECASE)
            if match:
                retry_sleep = max(retry_sleep, float(match.group(1)) + 2.0)

            if attempt >= cfg.max_retries:
                break

            print(f"[gemini] retry {attempt + 1}/{cfg.max_retries} after error: {exc}")
            time.sleep(retry_sleep)

    raise RuntimeError(f"Gemini failed after retries: {last_exc}")


def evaluate_gemini_direct(
    entries: list[dict[str, Any]],
    *,
    output_path: Path,
    cfg: GeminiConfig,
    vlm_jsonl_base: Path,
    limit: int | None,
    resume: bool,
) -> None:
    client = init_gemini_client(cfg)
    done = load_existing_ids(output_path) if resume else set()
    selected = entries if limit is None else entries[:limit]

    for i, entry in enumerate(selected, start=1):
        entry_id = str(entry["entry_id"])
        if entry_id in done:
            print(f"[gemini] skip existing {entry_id}")
            continue

        image_paths = [
            ensure_abs_path(p, base=vlm_jsonl_base.parent)
            for p in entry.get("anchor_frames", [])
        ]
        prompt = build_gemini_prompt(entry, judging_diffusion=False)

        print(f"[gemini] {i}/{len(selected)} {entry_id}")
        try:
            result = gemini_generate_with_retries(
                client,
                cfg=cfg,
                prompt=prompt,
                image_paths=image_paths,
            )
            pred = result["parsed"]
            score = score_prediction(entry, pred)

            row = {
                "entry_id": entry_id,
                "model_family": "gemini",
                "model": result.get("model_used", cfg.model),
                "query": entry.get("query"),
                "query_type": entry.get("query_type"),
                "ground_truth_answer": entry.get("answer"),
                "ground_truth_answer_value": entry.get("answer_value"),
                "prediction": pred,
                "raw_response": result["raw_text"],
                "score": score,
                "anchor_frames": [str(p) for p in image_paths],
            }
        except Exception as exc:
            row = {
                "entry_id": entry_id,
                "model_family": "gemini",
                "model": cfg.model,
                "error": repr(exc),
            }

        append_jsonl(output_path, row)
        if cfg.sleep_seconds > 0:
            time.sleep(cfg.sleep_seconds)


# -----------------------------
# Colab diffusion export
# -----------------------------


def first_frame_for_diffusion(entry: dict[str, Any], base: Path) -> Path | None:
    frames = entry.get("anchor_frames", [])
    if not frames:
        return None
    return ensure_abs_path(frames[0], base=base.parent)


def write_diffusion_jobs_for_colab(
    entries: list[dict[str, Any]],
    *,
    output_job_path: Path,
    assets_dir: Path,
    vlm_jsonl_base: Path,
    limit: int | None,
) -> None:
    selected = entries if limit is None else entries[:limit]
    images_dir = assets_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    output_job_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for entry in selected:
        entry_id = str(entry["entry_id"])
        input_image = first_frame_for_diffusion(entry, vlm_jsonl_base)

        if input_image is None or not input_image.exists():
            print(f"[warning] skipping {entry_id}: no input image")
            continue

        dst_image = images_dir / f"{entry_id}.png"
        shutil.copy2(input_image, dst_image)

        rows.append(
            {
                "entry_id": entry_id,
                "scene_family": entry.get("scene_family"),
                "query": entry.get("query"),
                "query_type": entry.get("query_type"),
                "latent_parameters": entry.get("latent_parameters", {}),
                "ground_truth_answer": entry.get("answer"),
                "ground_truth_answer_value": entry.get("answer_value"),
                "evaluation": entry.get("evaluation", {}),
                "input_image": str(dst_image.relative_to(assets_dir)),
                "prompt": build_diffusion_prompt(entry),
            }
        )

    write_jsonl(output_job_path, rows)
    print(f"Wrote {len(rows)} Colab diffusion jobs to {output_job_path}")
    print(f"Copied input images to {images_dir}")


# -----------------------------
# Judge Colab diffusion outputs
# -----------------------------


def extract_video_summary_frames(video_path: Path, frames_dir: Path) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_000 = frames_dir / "generated_frame_000.png"
    frame_mid = frames_dir / "generated_frame_mid.png"
    frame_final = frames_dir / "generated_frame_final.png"

    if frame_000.exists() and frame_mid.exists() and frame_final.exists():
        return [frame_000, frame_mid, frame_final]

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

    targets = [(frame_000, 0.0), (frame_mid, 0.5), (frame_final, 0.999)]
    for out, frac in targets:
        if duration is not None and duration > 0:
            t = max(0.0, min(duration - 1e-3, frac * duration))
            cmd = ["ffmpeg", "-y", "-ss", f"{t:.4f}", "-i", str(video_path), "-frames:v", "1", str(out)]
        else:
            cmd = ["ffmpeg", "-y", "-i", str(video_path), "-frames:v", "1", str(out)]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return [frame_000, frame_mid, frame_final]


def judge_diffusion_outputs(
    entries: list[dict[str, Any]],
    *,
    colab_results_jsonl: Path,
    generated_video_root: Path,
    output_path: Path,
    output_dir: Path,
    cfg: GeminiConfig,
    limit: int | None,
    resume: bool,
) -> None:
    client = init_gemini_client(cfg)
    done = load_existing_ids(output_path) if resume else set()

    entry_by_id = {str(e["entry_id"]): e for e in entries}
    colab_rows = read_jsonl(colab_results_jsonl)
    if limit is not None:
        colab_rows = colab_rows[:limit]

    frames_root = output_dir / "diffusion_frames"

    for i, row0 in enumerate(colab_rows, start=1):
        entry_id = str(row0["entry_id"])
        if entry_id in done:
            print(f"[judge] skip existing {entry_id}")
            continue

        entry = entry_by_id.get(entry_id)
        if entry is None:
            append_jsonl(output_path, {"entry_id": entry_id, "error": "Missing original VLM entry."})
            continue

        video_path = Path(row0.get("generated_video", ""))
        if not video_path.is_absolute():
            candidate = generated_video_root / video_path.name
            if candidate.exists():
                video_path = candidate
            else:
                video_path = generated_video_root / f"{entry_id}.mp4"

        print(f"[judge] {i}/{len(colab_rows)} {entry_id}")

        try:
            if not video_path.exists():
                raise FileNotFoundError(f"Missing generated video: {video_path}")

            generated_frames = extract_video_summary_frames(video_path, frames_root / entry_id)
            prompt = build_gemini_prompt(entry, judging_diffusion=True)

            result = gemini_generate_with_retries(
                client,
                cfg=cfg,
                prompt=prompt,
                image_paths=generated_frames,
            )
            pred = result["parsed"]
            score = score_prediction(entry, pred)

            out = {
                "entry_id": entry_id,
                "model_family": "diffusion_ltx_colab",
                "judge_model_family": "gemini",
                "judge_model": result.get("model_used", cfg.model),
                "query": entry.get("query"),
                "query_type": entry.get("query_type"),
                "ground_truth_answer": entry.get("answer"),
                "ground_truth_answer_value": entry.get("answer_value"),
                "generated_video": str(video_path),
                "generated_video_frames_for_judge": [str(p) for p in generated_frames],
                "prediction": pred,
                "raw_judge_response": result["raw_text"],
                "score": score,
                "colab_diffusion_row": row0,
            }
        except Exception as exc:
            out = {
                "entry_id": entry_id,
                "model_family": "diffusion_ltx_colab",
                "judge_model_family": "gemini",
                "judge_model": cfg.model,
                "error": repr(exc),
                "colab_diffusion_row": row0,
            }

        append_jsonl(output_path, out)
        if cfg.sleep_seconds > 0:
            time.sleep(cfg.sleep_seconds)


# -----------------------------
# Summary
# -----------------------------


def summarize_results(result_paths: list[Path], summary_path: Path) -> None:
    summary: dict[str, Any] = {"files": {}}

    for path in result_paths:
        rows = read_jsonl(path)
        correct_values = []
        num_errors = 0
        by_query_type: dict[str, dict[str, int]] = {}

        for row in rows:
            if "error" in row:
                num_errors += 1
                continue

            qtype = row.get("query_type", "unknown")
            by_query_type.setdefault(qtype, {"n": 0, "correct": 0, "scored": 0})
            by_query_type[qtype]["n"] += 1

            correct = row.get("score", {}).get("correct")
            if correct is not None:
                by_query_type[qtype]["scored"] += 1
                by_query_type[qtype]["correct"] += int(bool(correct))
                correct_values.append(bool(correct))

        summary["files"][str(path)] = {
            "num_rows": len(rows),
            "num_errors": num_errors,
            "num_scored": len(correct_values),
            "accuracy_on_scored": sum(correct_values) / len(correct_values) if correct_values else None,
            "by_query_type": by_query_type,
        }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--mode",
        choices=["gemini", "export_diffusion_jobs", "judge_diffusion"],
        required=True,
    )
    parser.add_argument("--vlm-jsonl", type=Path, default=Path("datasets/vlm_prompt_dataset/vlm_entries.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_outputs"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")

    # Gemini.
    parser.add_argument("--gemini-model", type=str, default="gemini-2.0-flash")
    parser.add_argument("--gemini-api-key-env", type=str, default="GEMINI_API_KEY")
    parser.add_argument("--sleep-seconds", type=float, default=4.0)
    parser.add_argument("--max-retries", type=int, default=2)

    # Colab diffusion export.
    parser.add_argument(
        "--colab-assets-dir",
        type=Path,
        default=Path("evaluation_outputs/colab_diffusion_assets"),
    )

    # Colab diffusion judge.
    parser.add_argument(
        "--colab-results-jsonl",
        type=Path,
        default=Path("evaluation_outputs/colab_diffusion_assets/diffusion_raw_results.jsonl"),
    )
    parser.add_argument(
        "--generated-video-root",
        type=Path,
        default=Path("evaluation_outputs/colab_diffusion_assets/generated_videos"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    entries = read_jsonl(args.vlm_jsonl)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gemini_cfg = GeminiConfig(
        model=args.gemini_model,
        api_key_env=args.gemini_api_key_env,
        sleep_seconds=args.sleep_seconds,
        max_retries=args.max_retries,
    )

    if args.mode == "gemini":
        out = args.output_dir / "gemini_results.jsonl"
        evaluate_gemini_direct(
            entries,
            output_path=out,
            cfg=gemini_cfg,
            vlm_jsonl_base=args.vlm_jsonl,
            limit=args.limit,
            resume=not args.no_resume,
        )
        summarize_results([out], args.output_dir / "summary_gemini.json")
        return

    if args.mode == "export_diffusion_jobs":
        write_diffusion_jobs_for_colab(
            entries,
            output_job_path=args.colab_assets_dir / "diffusion_jobs.jsonl",
            assets_dir=args.colab_assets_dir,
            vlm_jsonl_base=args.vlm_jsonl,
            limit=args.limit,
        )
        return

    if args.mode == "judge_diffusion":
        out = args.output_dir / "diffusion_colab_judged_results.jsonl"
        judge_diffusion_outputs(
            entries,
            colab_results_jsonl=args.colab_results_jsonl,
            generated_video_root=args.generated_video_root,
            output_path=out,
            output_dir=args.output_dir,
            cfg=gemini_cfg,
            limit=args.limit,
            resume=not args.no_resume,
        )
        summarize_results([out], args.output_dir / "summary_diffusion_colab_judged.json")
        return

    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()