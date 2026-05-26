#!/usr/bin/env python3
"""
Build human-readable evaluation artifacts from evaluate_vlm_dataset.py outputs.

Inputs:
  evaluation_outputs/
    gemini_results.jsonl              optional
    diffusion_results.jsonl           optional
    diffusion_videos/                 optional
    diffusion_frames/                 optional

Outputs:
  evaluation_outputs/
    results_table.csv
    results_table.html
    gallery.html

Example:
  uv run python scripts/make_eval_artifacts.py \
    --output-dir evaluation_outputs
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import pandas as pd


# -----------------------------
# JSONL helpers
# -----------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_get(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def relpath_or_none(path_like: Any, base_dir: Path) -> str | None:
    if not path_like:
        return None

    p = Path(str(path_like))
    try:
        if p.exists():
            return str(p.relative_to(base_dir) if p.is_relative_to(base_dir) else p)
    except Exception:
        pass

    return str(path_like)


def file_url(path_like: Any, base_dir: Path) -> str | None:
    """
    Convert a local path into a relative URL suitable for HTML.

    If the path is inside output_dir, make it relative to output_dir.
    Otherwise leave it as-is.
    """
    if not path_like:
        return None

    p = Path(str(path_like))

    try:
        if p.exists() and p.is_absolute():
            return p.as_uri()
    except Exception:
        pass

    try:
        if p.exists() and p.is_relative_to(base_dir):
            return p.relative_to(base_dir).as_posix()
    except Exception:
        pass

    # Common case: stored path is already something like
    # evaluation_outputs/diffusion_videos/foo.mp4.
    try:
        if str(p).startswith(str(base_dir)):
            return p.relative_to(base_dir).as_posix()
    except Exception:
        pass

    return str(path_like)


# -----------------------------
# Row flattening
# -----------------------------


def flatten_result_row(row: dict[str, Any], source_file: Path, output_dir: Path) -> dict[str, Any]:
    pred = row.get("prediction", {}) or {}
    score = row.get("score", {}) or {}
    diffusion = row.get("diffusion", {}) or {}

    generated_video = row.get("generated_video") or diffusion.get("output_video")
    input_image = row.get("input_image")

    generated_frames = row.get("generated_video_frames_for_judge") or []

    return {
        "source_file": source_file.name,
        "entry_id": row.get("entry_id"),
        "model_family": row.get("model_family"),
        "model": row.get("model"),
        "judge_model_family": row.get("judge_model_family"),
        "judge_model": row.get("judge_model"),
        "query_type": row.get("query_type"),
        "query": row.get("query"),
        "ground_truth_answer": row.get("ground_truth_answer"),
        "ground_truth_answer_value": row.get("ground_truth_answer_value"),
        "prediction_answer": pred.get("answer"),
        "prediction_answer_value": pred.get("answer_value"),
        "prediction_confidence": pred.get("confidence"),
        "prediction_rationale": pred.get("rationale"),
        "score_type": score.get("score_type"),
        "correct": score.get("correct"),
        "absolute_error": score.get("absolute_error"),
        "relative_error": score.get("relative_error"),
        "log10_error": score.get("log10_error"),
        "gt_normalized": score.get("gt_normalized"),
        "pred_normalized": score.get("pred_normalized"),
        "anchor_frames": "; ".join(row.get("anchor_frames", []) or []),
        "input_image": input_image,
        "generated_video": generated_video,
        "generated_frames_for_judge": "; ".join(str(p) for p in generated_frames),
        "error": row.get("error"),
        "diffusion_returncode": diffusion.get("returncode"),
        "diffusion_prompt": diffusion.get("prompt"),
        "diffusion_stdout_tail": diffusion.get("stdout_tail"),
        "diffusion_stderr_tail": diffusion.get("stderr_tail"),
    }


def collect_results(output_dir: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    result_paths = [
        output_dir / "gemini_results.jsonl",
        output_dir / "diffusion_results.jsonl",
    ]

    flat_rows: list[dict[str, Any]] = []
    existing_paths: list[Path] = []

    for path in result_paths:
        if not path.exists():
            continue

        existing_paths.append(path)
        rows = read_jsonl(path)
        for row in rows:
            flat_rows.append(flat_result := flatten_result_row(row, path, output_dir))

    return flat_rows, existing_paths


# -----------------------------
# Table generation
# -----------------------------


def make_results_table(output_dir: Path) -> pd.DataFrame:
    rows, existing_paths = collect_results(output_dir)

    if not rows:
        raise RuntimeError(
            f"No result files found in {output_dir}. "
            "Expected gemini_results.jsonl and/or diffusion_results.jsonl."
        )

    df = pd.DataFrame(rows)

    preferred_cols = [
        "entry_id",
        "model_family",
        "model",
        "judge_model",
        "query_type",
        "query",
        "ground_truth_answer_value",
        "prediction_answer_value",
        "correct",
        "absolute_error",
        "relative_error",
        "log10_error",
        "prediction_confidence",
        "prediction_rationale",
        "input_image",
        "generated_video",
        "generated_frames_for_judge",
        "error",
        "source_file",
    ]

    remaining_cols = [c for c in df.columns if c not in preferred_cols]
    df = df[[c for c in preferred_cols if c in df.columns] + remaining_cols]

    csv_path = output_dir / "results_table.csv"
    html_path = output_dir / "results_table.html"

    df.to_csv(csv_path, index=False)

    styled_html = dataframe_to_html_page(
        df,
        title="Evaluation Results Table",
        subtitle=f"Loaded: {', '.join(p.name for p in existing_paths)}",
    )
    html_path.write_text(styled_html)

    return df


def dataframe_to_html_page(df: pd.DataFrame, *, title: str, subtitle: str) -> str:
    table_html = df.to_html(
        index=False,
        escape=True,
        classes="results-table",
        border=0,
    )

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 2rem;
      background: #fafafa;
      color: #222;
    }}
    h1 {{
      margin-bottom: 0.25rem;
    }}
    .subtitle {{
      color: #666;
      margin-bottom: 1.5rem;
    }}
    .table-wrap {{
      overflow-x: auto;
      background: white;
      border: 1px solid #ddd;
      border-radius: 12px;
      padding: 1rem;
    }}
    table.results-table {{
      border-collapse: collapse;
      font-size: 0.85rem;
      width: max-content;
      min-width: 100%;
    }}
    table.results-table th {{
      position: sticky;
      top: 0;
      background: #f0f0f0;
      border-bottom: 2px solid #ccc;
      padding: 0.5rem;
      text-align: left;
      white-space: nowrap;
    }}
    table.results-table td {{
      border-bottom: 1px solid #eee;
      padding: 0.5rem;
      max-width: 360px;
      overflow-wrap: anywhere;
      vertical-align: top;
    }}
    table.results-table tr:hover {{
      background: #f8f8f8;
    }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="subtitle">{html.escape(subtitle)}</div>
  <div class="table-wrap">
    {table_html}
  </div>
</body>
</html>
"""


# -----------------------------
# Gallery generation
# -----------------------------


def correctness_label(value: Any) -> tuple[str, str]:
    if value is True:
        return "Correct", "correct"
    if value is False:
        return "Incorrect", "incorrect"
    if value is None:
        return "Unscored", "unscored"
    return str(value), "unscored"


def html_escape(x: Any) -> str:
    if x is None:
        return ""
    return html.escape(str(x))


def render_image(path_like: Any, output_dir: Path, *, caption: str) -> str:
    url = file_url(path_like, output_dir)
    if not url:
        return ""

    return f"""
    <figure>
      <img src="{html_escape(url)}" alt="{html_escape(caption)}">
      <figcaption>{html_escape(caption)}</figcaption>
    </figure>
    """


def render_video(path_like: Any, output_dir: Path) -> str:
    url = file_url(path_like, output_dir)
    if not url:
        return ""

    return f"""
    <video controls muted loop preload="metadata">
      <source src="{html_escape(url)}" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    """


def render_gallery_card(row: dict[str, Any], output_dir: Path) -> str:
    pred = row.get("prediction", {}) or {}
    score = row.get("score", {}) or {}
    diffusion = row.get("diffusion", {}) or {}

    entry_id = row.get("entry_id")
    correct_text, correct_class = correctness_label(score.get("correct"))

    input_image = row.get("input_image")
    generated_video = row.get("generated_video") or diffusion.get("output_video")
    generated_frames = row.get("generated_video_frames_for_judge") or []

    # Fall back to known diffusion_frames/<entry_id>/ layout if explicit frame list is absent.
    if not generated_frames and entry_id:
        frame_dir = output_dir / "diffusion_frames" / str(entry_id)
        candidates = [
            frame_dir / "generated_frame_000.png",
            frame_dir / "generated_frame_mid.png",
            frame_dir / "generated_frame_final.png",
        ]
        generated_frames = [str(p) for p in candidates if p.exists()]

    frame_html = ""
    captions = ["generated start", "generated middle", "generated final"]
    for p, cap in zip(generated_frames, captions, strict=False):
        frame_html += render_image(p, output_dir, caption=cap)

    video_html = render_video(generated_video, output_dir)

    if not video_html and not frame_html:
        media_html = """
        <div class="empty-media">
          No generated video or frames found for this row.
        </div>
        """
    else:
        media_html = f"""
        <div class="video-wrap">
          {video_html}
        </div>
        <div class="frames-grid">
          {frame_html}
        </div>
        """

    error = row.get("error")
    error_html = ""
    if error:
        error_html = f"""
        <div class="error-box">
          <strong>Error:</strong> {html_escape(error)}
        </div>
        """

    return f"""
    <section class="card {correct_class}">
      <div class="card-header">
        <div>
          <h2>{html_escape(entry_id)}</h2>
          <div class="meta">
            {html_escape(row.get("model_family"))}
            {(" · judge: " + html_escape(row.get("judge_model"))) if row.get("judge_model") else ""}
            {(" · " + html_escape(row.get("query_type"))) if row.get("query_type") else ""}
          </div>
        </div>
        <div class="badge {correct_class}">{html_escape(correct_text)}</div>
      </div>

      <div class="qa-grid">
        <div>
          <h3>Question</h3>
          <p>{html_escape(row.get("query"))}</p>
        </div>
        <div>
          <h3>Ground truth</h3>
          <p>{html_escape(row.get("ground_truth_answer_value"))}</p>
        </div>
        <div>
          <h3>Prediction</h3>
          <p>{html_escape(pred.get("answer_value"))}</p>
        </div>
        <div>
          <h3>Confidence</h3>
          <p>{html_escape(pred.get("confidence"))}</p>
        </div>
      </div>

      <div class="rationale">
        <h3>Rationale</h3>
        <p>{html_escape(pred.get("rationale"))}</p>
      </div>

      {error_html}

      <div class="media-section">
        <div class="input-image">
          {render_image(input_image, output_dir, caption="input image")}
        </div>
        <div class="generated-media">
          {media_html}
        </div>
      </div>
    </section>
    """


def make_gallery(output_dir: Path) -> None:
    diffusion_path = output_dir / "diffusion_results.jsonl"
    gemini_path = output_dir / "gemini_results.jsonl"

    if diffusion_path.exists():
        rows = read_jsonl(diffusion_path)
        gallery_source = diffusion_path.name
        title = "Diffusion Evaluation Gallery"
    elif gemini_path.exists():
        rows = read_jsonl(gemini_path)
        gallery_source = gemini_path.name
        title = "Gemini Evaluation Gallery"
    else:
        raise RuntimeError(
            f"No result files found in {output_dir}. "
            "Expected diffusion_results.jsonl or gemini_results.jsonl."
        )

    cards = "\n".join(render_gallery_card(row, output_dir) for row in rows)

    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html_escape(title)}</title>
  <style>
    body {{
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 2rem;
      background: #f7f7f7;
      color: #222;
    }}
    h1 {{
      margin-bottom: 0.25rem;
    }}
    .subtitle {{
      color: #666;
      margin-bottom: 1.5rem;
    }}
    .card {{
      background: white;
      border: 1px solid #ddd;
      border-left: 8px solid #aaa;
      border-radius: 14px;
      padding: 1.25rem;
      margin-bottom: 1.5rem;
      box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    }}
    .card.correct {{
      border-left-color: #2e7d32;
    }}
    .card.incorrect {{
      border-left-color: #c62828;
    }}
    .card.unscored {{
      border-left-color: #777;
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      gap: 1rem;
      align-items: flex-start;
      margin-bottom: 1rem;
    }}
    .card h2 {{
      margin: 0;
      font-size: 1.2rem;
    }}
    .meta {{
      color: #666;
      font-size: 0.9rem;
      margin-top: 0.25rem;
    }}
    .badge {{
      padding: 0.35rem 0.65rem;
      border-radius: 999px;
      font-weight: 700;
      font-size: 0.85rem;
      white-space: nowrap;
      background: #eee;
    }}
    .badge.correct {{
      background: #e8f5e9;
      color: #1b5e20;
    }}
    .badge.incorrect {{
      background: #ffebee;
      color: #b71c1c;
    }}
    .badge.unscored {{
      background: #eeeeee;
      color: #444;
    }}
    .qa-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 1rem;
      margin-bottom: 1rem;
    }}
    .qa-grid div,
    .rationale {{
      background: #fafafa;
      border: 1px solid #eee;
      border-radius: 10px;
      padding: 0.75rem;
    }}
    h3 {{
      margin: 0 0 0.4rem 0;
      font-size: 0.9rem;
      color: #555;
    }}
    p {{
      margin: 0;
      line-height: 1.4;
    }}
    .error-box {{
      background: #fff3e0;
      border: 1px solid #ffcc80;
      border-radius: 10px;
      padding: 0.75rem;
      margin: 1rem 0;
      color: #5d3b00;
      overflow-wrap: anywhere;
    }}
    .media-section {{
      display: grid;
      grid-template-columns: 280px 1fr;
      gap: 1rem;
      margin-top: 1rem;
      align-items: start;
    }}
    figure {{
      margin: 0;
    }}
    figcaption {{
      color: #666;
      font-size: 0.8rem;
      margin-top: 0.25rem;
    }}
    img {{
      max-width: 100%;
      border-radius: 10px;
      border: 1px solid #ddd;
      background: #eee;
    }}
    video {{
      width: 100%;
      max-height: 520px;
      border-radius: 10px;
      border: 1px solid #ddd;
      background: black;
    }}
    .video-wrap {{
      margin-bottom: 1rem;
    }}
    .frames-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 0.75rem;
    }}
    .empty-media {{
      background: #fafafa;
      border: 1px dashed #ccc;
      border-radius: 10px;
      padding: 1rem;
      color: #666;
    }}
    @media (max-width: 1000px) {{
      .qa-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      .media-section {{
        grid-template-columns: 1fr;
      }}
    }}
    @media (max-width: 650px) {{
      .qa-grid {{
        grid-template-columns: 1fr;
      }}
      .frames-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <h1>{html_escape(title)}</h1>
  <div class="subtitle">
    Source: {html_escape(gallery_source)} · Rows: {len(rows)}
  </div>
  {cards}
</body>
</html>
"""

    (output_dir / "gallery.html").write_text(html_doc)


# -----------------------------
# Optional text summary
# -----------------------------


def print_summary(df: pd.DataFrame) -> None:
    print("\nWrote results artifacts.")
    print(f"Rows: {len(df)}")

    if "correct" in df.columns:
        scored = df[df["correct"].notna()]
        if len(scored) > 0:
            print("\nAccuracy by model_family:")
            print(scored.groupby("model_family")["correct"].mean())

            if "query_type" in scored.columns:
                print("\nAccuracy by model_family/query_type:")
                print(scored.groupby(["model_family", "query_type"])["correct"].mean())


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_outputs"),
        help="Directory containing evaluation result JSONL files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = make_results_table(output_dir)
    make_gallery(output_dir)

    print_summary(df)

    print(f"\nCreated:")
    print(f"  {output_dir / 'results_table.csv'}")
    print(f"  {output_dir / 'results_table.html'}")
    print(f"  {output_dir / 'gallery.html'}")


if __name__ == "__main__":
    main()

"""
example usage:
uv run python scripts/make_eval_artifacts.py \
    --output-dir evaluation_outputs

To see results table:
open evaluation_outputs/results_table.html

on a remote server, you can run a simple HTTP server to view the HTML files:
cd evaluation_outputs
python -m http.server 8000
Then open http://localhost:8000/results_table.html in your browser.
"""