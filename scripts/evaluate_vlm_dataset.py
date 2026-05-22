#!/usr/bin/env python3
"""
Evaluate Gemini and an image-to-video diffusion model on vlm_entries.jsonl.

Inputs:
  datasets/vlm_prompt_dataset/vlm_entries.jsonl

Outputs:
  evaluation_outputs/
    gemini_results.jsonl
    diffusion_results.jsonl
    diffusion_videos/
    diffusion_frames/
    summary.json

Required for Gemini:
  pip install -U google-genai
  export GEMINI_API_KEY="..."

Required for diffusion:
  LTX-2 repo set up locally, with checkpoints downloaded, e.g.
    models/ltx-2-19b-dev.safetensors
    models/ltx-2-19b-distilled-lora-384.safetensors
    models/ltx-2-spatial-upscaler-x2-1.0.safetensors
    models/gemma/

Example Gemini-only:
uv run scripts/evaluate_vlm_dataset.py \
      --vlm-jsonl datasets/vlm_prompt_dataset/vlm_entries.jsonl \
            --output-dir evaluation_outputs  \
                      --models gemini \
                         --limit 20 \
                         --gemini-model gemini-2.5-flash

Example diffusion + Gemini judge:
  uv run scripts/evaluate_vlm_dataset.py \
    --vlm-jsonl datasets/vlm_prompt_dataset/vlm_entries.jsonl \
    --output-dir evaluation_outputs \
    --models diffusion \
    --ltx-root /path/to/LTX-2 \
    --limit 10

Example both:
  uv run scripts/evaluate_vlm_dataset.py \
    --vlm-jsonl datasets/vlm_prompt_dataset/vlm_entries.jsonl \
    --output-dir evaluation_outputs \
    --models gemini diffusion \
    --ltx-root /path/to/LTX-2
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# -----------------------------
# IO helpers
# -----------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for row in read_jsonl(path):
        if "entry_id" in row:
            ids.add(str(row["entry_id"]))
    return ids


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


# -----------------------------
# Prompt construction
# -----------------------------


def compact_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def build_gemini_prompt(entry: dict[str, Any], *, judging_diffusion: bool = False) -> str:
    """
    Ask for strict JSON so parsing/scoring is easy.
    """
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
    """
    Convert benchmark query into an image-to-video continuation prompt.

    The diffusion model is not directly answering the question; it produces a
    video continuation. Gemini is then used as a judge/answer extractor from
    the generated video/frames.
    """
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
# Answer parsing and scoring
# -----------------------------


def parse_jsonish(text: str) -> dict[str, Any]:
    text = text.strip()

    # Remove markdown fences if the model ignored instructions.
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract the first JSON object.
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
            try:
                return float(match.group(0))
            except ValueError:
                return None
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

        tolerance = evaluation.get("tolerance", None)
        if tolerance is not None:
            correct = abs_err <= float(tolerance)
        elif "log" in metric and gt_float > 0 and pred_float > 0:
            # Useful for density/viscosity parameters spanning orders of magnitude.
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
        else:
            correct = None

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
# Gemini direct / judge
# -----------------------------


@dataclass
class GeminiConfig:
    model: str
    api_key_env: str
    sleep_seconds: float


def init_gemini_client(cfg: GeminiConfig):
    from google import genai

    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing Gemini API key. Set {cfg.api_key_env}=...")
    return genai.Client(api_key=api_key)


def gemini_generate(
    client: Any,
    *,
    model: str,
    prompt: str,
    image_paths: list[Path],
) -> dict[str, Any]:
    from google.genai import types

    contents: list[Any] = [prompt]
    for path in image_paths:
        if not path.exists():
            continue
        contents.append(
            types.Part.from_bytes(
                data=read_image_bytes(path),
                mime_type="image/png",
            )
        )

    response = client.models.generate_content(
        model=model,
        contents=contents,
    )
    text = getattr(response, "text", "") or ""
    parsed = parse_jsonish(text)

    return {
        "raw_text": text,
        "parsed": parsed,
    }


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

        image_paths = [ensure_abs_path(p, base=vlm_jsonl_base.parent) for p in entry.get("anchor_frames", [])]
        prompt = build_gemini_prompt(entry, judging_diffusion=False)

        print(f"[gemini] {i}/{len(selected)} {entry_id}")
        try:
            result = gemini_generate(
                client,
                model=cfg.model,
                prompt=prompt,
                image_paths=image_paths,
            )
            pred = result["parsed"]
            score = score_prediction(entry, pred)

            row = {
                "entry_id": entry_id,
                "model_family": "gemini",
                "model": cfg.model,
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
# Diffusion generation + Gemini judge
# -----------------------------


@dataclass
class DiffusionConfig:
    ltx_root: Path
    checkpoint_path: Path
    distilled_lora_path: Path
    distilled_lora_scale: float
    spatial_upsampler_path: Path
    gemma_root: Path
    seed: int
    max_entries: int | None
    timeout_seconds: int
    diffusion_num_frames: int | None
    judge_with_gemini: bool


def first_frame_for_diffusion(entry: dict[str, Any], base: Path) -> Path | None:
    frames = entry.get("anchor_frames", [])
    if not frames:
        return None
    return ensure_abs_path(frames[0], base=base.parent)


def run_ltx_diffusion(
    entry: dict[str, Any],
    *,
    input_image: Path,
    output_video: Path,
    cfg: DiffusionConfig,
) -> dict[str, Any]:
    output_video.parent.mkdir(parents=True, exist_ok=True)

    prompt = build_diffusion_prompt(entry)

    cmd = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "ltx_pipelines.ti2vid_two_stages",
        "--checkpoint-path",
        str(cfg.checkpoint_path),
        "--distilled-lora",
        str(cfg.distilled_lora_path),
        str(cfg.distilled_lora_scale),
        "--spatial-upsampler-path",
        str(cfg.spatial_upsampler_path),
        "--gemma-root",
        str(cfg.gemma_root),
        "--image",
        str(input_image),
        "0",
        "1.0",
        "--seed",
        str(cfg.seed),
        "--prompt",
        prompt,
        "--output-path",
        str(output_video),
    ]

    if cfg.diffusion_num_frames is not None:
        # Only works if the installed LTX pipeline exposes this flag.
        # If unsupported, remove this from the command or leave unset.
        cmd.extend(["--num-frames", str(cfg.diffusion_num_frames)])

    completed = subprocess.run(
        cmd,
        cwd=str(cfg.ltx_root),
        text=True,
        capture_output=True,
        timeout=cfg.timeout_seconds,
    )

    return {
        "cmd": cmd,
        "prompt": prompt,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "output_video": str(output_video),
    }


def extract_video_summary_frames(video_path: Path, frames_dir: Path) -> list[Path]:
    """
    Extract initial/mid/final frames from a generated diffusion video so Gemini
    can judge/answer questions from the generated rollout.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_000 = frames_dir / "generated_frame_000.png"
    frame_mid = frames_dir / "generated_frame_mid.png"
    frame_final = frames_dir / "generated_frame_final.png"

    if frame_000.exists() and frame_mid.exists() and frame_final.exists():
        return [frame_000, frame_mid, frame_final]

    # Reuse a simple robust approach.
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

    times = [0.0, 0.5, 0.999]
    outputs = [frame_000, frame_mid, frame_final]

    for out, frac in zip(outputs, times, strict=True):
        if duration is not None and duration > 0:
            t = max(0.0, min(duration - 1e-3, frac * duration))
            cmd = ["ffmpeg", "-y", "-ss", f"{t:.4f}", "-i", str(video_path), "-frames:v", "1", str(out)]
        else:
            cmd = ["ffmpeg", "-y", "-i", str(video_path), "-frames:v", "1", str(out)]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return outputs


def evaluate_diffusion(
    entries: list[dict[str, Any]],
    *,
    output_path: Path,
    output_dir: Path,
    diffusion_cfg: DiffusionConfig,
    gemini_cfg: GeminiConfig,
    vlm_jsonl_base: Path,
    limit: int | None,
    resume: bool,
) -> None:
    if shutil.which("uv") is None:
        raise RuntimeError("uv not found on PATH; required for the LTX command.")

    done = load_existing_ids(output_path) if resume else set()
    selected = entries if limit is None else entries[:limit]
    if diffusion_cfg.max_entries is not None:
        selected = selected[: diffusion_cfg.max_entries]

    gemini_client = init_gemini_client(gemini_cfg) if diffusion_cfg.judge_with_gemini else None

    videos_root = output_dir / "diffusion_videos"
    frames_root = output_dir / "diffusion_frames"

    for i, entry in enumerate(selected, start=1):
        entry_id = str(entry["entry_id"])
        if entry_id in done:
            print(f"[diffusion] skip existing {entry_id}")
            continue

        input_image = first_frame_for_diffusion(entry, vlm_jsonl_base)
        if input_image is None or not input_image.exists():
            append_jsonl(
                output_path,
                {
                    "entry_id": entry_id,
                    "model_family": "diffusion",
                    "error": "No anchor frame available for diffusion input.",
                },
            )
            continue

        output_video = videos_root / f"{entry_id}.mp4"

        print(f"[diffusion] {i}/{len(selected)} {entry_id}")
        row: dict[str, Any] = {
            "entry_id": entry_id,
            "model_family": "diffusion_ltx",
            "query": entry.get("query"),
            "query_type": entry.get("query_type"),
            "ground_truth_answer": entry.get("answer"),
            "ground_truth_answer_value": entry.get("answer_value"),
            "input_image": str(input_image),
            "generated_video": str(output_video),
        }

        try:
            diffusion_result = run_ltx_diffusion(
                entry,
                input_image=input_image,
                output_video=output_video,
                cfg=diffusion_cfg,
            )
            row["diffusion"] = diffusion_result

            if diffusion_result["returncode"] != 0:
                row["error"] = "Diffusion subprocess failed."
                append_jsonl(output_path, row)
                continue

            if diffusion_cfg.judge_with_gemini:
                assert gemini_client is not None
                generated_frames = extract_video_summary_frames(output_video, frames_root / entry_id)
                judge_prompt = build_gemini_prompt(entry, judging_diffusion=True)

                judge_result = gemini_generate(
                    gemini_client,
                    model=gemini_cfg.model,
                    prompt=judge_prompt,
                    image_paths=generated_frames,
                )

                pred = judge_result["parsed"]
                score = score_prediction(entry, pred)

                row["judge_model_family"] = "gemini"
                row["judge_model"] = gemini_cfg.model
                row["generated_video_frames_for_judge"] = [str(p) for p in generated_frames]
                row["prediction"] = pred
                row["raw_judge_response"] = judge_result["raw_text"]
                row["score"] = score

        except Exception as exc:
            row["error"] = repr(exc)

        append_jsonl(output_path, row)
        if gemini_cfg.sleep_seconds > 0:
            time.sleep(gemini_cfg.sleep_seconds)


# -----------------------------
# Summary
# -----------------------------


def summarize_results(result_paths: list[Path], summary_path: Path) -> None:
    summary: dict[str, Any] = {"files": {}}

    for path in result_paths:
        if not path.exists():
            continue

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

            score = row.get("score", {})
            correct = score.get("correct")
            if correct is not None:
                by_query_type[qtype]["scored"] += 1
                by_query_type[qtype]["correct"] += int(bool(correct))
                correct_values.append(bool(correct))

        summary["files"][str(path)] = {
            "num_rows": len(rows),
            "num_errors": num_errors,
            "num_scored": len(correct_values),
            "accuracy_on_scored": (
                sum(correct_values) / len(correct_values) if correct_values else None
            ),
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

    parser.add_argument("--vlm-jsonl", type=Path, default=Path("datasets/vlm_prompt_dataset/vlm_entries.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_outputs"))
    parser.add_argument("--models", nargs="+", choices=["gemini", "diffusion"], default=["gemini"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")

    # Gemini.
    # parser.add_argument("--gemini-model", type=str, default="gemini-2.5-pro")
    parser.add_argument("--gemini-model", type=str, default="gemini-2.0-flash")
    # parser.add_argument("--gemini-model", type=str, default="gemini-2.0-pro-exp-02-05")
    parser.add_argument("--gemini-api-key-env", type=str, default="GEMINI_API_KEY")
    parser.add_argument("--sleep-seconds", type=float, default=4.0)

    # Diffusion / LTX.
    parser.add_argument("--ltx-root", type=Path, default=Path("."))
    parser.add_argument("--checkpoint-path", type=Path, default=Path("models/ltx-2-19b-dev.safetensors"))
    parser.add_argument("--distilled-lora-path", type=Path, default=Path("models/ltx-2-19b-distilled-lora-384.safetensors"))
    parser.add_argument("--distilled-lora-scale", type=float, default=0.8)
    parser.add_argument("--spatial-upsampler-path", type=Path, default=Path("models/ltx-2-spatial-upscaler-x2-1.0.safetensors"))
    parser.add_argument("--gemma-root", type=Path, default=Path("models/gemma"))
    parser.add_argument("--diffusion-seed", type=int, default=0)
    parser.add_argument("--diffusion-timeout-seconds", type=int, default=1800)
    parser.add_argument("--diffusion-max-entries", type=int, default=None)
    parser.add_argument("--diffusion-num-frames", type=int, default=None)
    parser.add_argument("--no-gemini-judge", action="store_true")

    return parser.parse_args()


def resolve_ltx_paths(args: argparse.Namespace) -> DiffusionConfig:
    root = args.ltx_root.resolve()

    def resolve_under_root(p: Path) -> Path:
        if p.is_absolute():
            return p
        return root / p

    return DiffusionConfig(
        ltx_root=root,
        checkpoint_path=resolve_under_root(args.checkpoint_path),
        distilled_lora_path=resolve_under_root(args.distilled_lora_path),
        distilled_lora_scale=args.distilled_lora_scale,
        spatial_upsampler_path=resolve_under_root(args.spatial_upsampler_path),
        gemma_root=resolve_under_root(args.gemma_root),
        seed=args.diffusion_seed,
        max_entries=args.diffusion_max_entries,
        timeout_seconds=args.diffusion_timeout_seconds,
        diffusion_num_frames=args.diffusion_num_frames,
        judge_with_gemini=not args.no_gemini_judge,
    )


def main() -> None:
    args = parse_args()

    entries = read_jsonl(args.vlm_jsonl)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    gemini_cfg = GeminiConfig(
        model=args.gemini_model,
        api_key_env=args.gemini_api_key_env,
        sleep_seconds=args.sleep_seconds,
    )

    result_paths: list[Path] = []

    if "gemini" in args.models:
        gemini_out = output_dir / "gemini_results.jsonl"
        evaluate_gemini_direct(
            entries,
            output_path=gemini_out,
            cfg=gemini_cfg,
            vlm_jsonl_base=args.vlm_jsonl,
            limit=args.limit,
            resume=not args.no_resume,
        )
        result_paths.append(gemini_out)

    if "diffusion" in args.models:
        diffusion_out = output_dir / "diffusion_results.jsonl"
        diffusion_cfg = resolve_ltx_paths(args)
        evaluate_diffusion(
            entries,
            output_path=diffusion_out,
            output_dir=output_dir,
            diffusion_cfg=diffusion_cfg,
            gemini_cfg=gemini_cfg,
            vlm_jsonl_base=args.vlm_jsonl,
            limit=args.limit,
            resume=not args.no_resume,
        )
        result_paths.append(diffusion_out)

    summarize_results(result_paths, output_dir / "summary.json")


if __name__ == "__main__":
    main()