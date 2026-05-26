#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


SCENES = {
    "ramp_cup": {
        "sweep_dir": "ramp_cup_sweep",
        "result_names": ["result.npz"],
        "video_names": ["video.mp4"],
    },
    "robotic_pour": {
        "sweep_dir": "robotic_pour_sweep",
        "result_names": ["result.npz"],
        "video_names": ["video.mp4"],
    },
    "pendulum": {
        "sweep_dir": "pendulum_sweep",
        "result_names": ["trajectory.npz"],
        "video_names": ["pendulum.mp4", "video.mp4"],
    },
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
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


def copy_file(src: Path, dst: Path, *, overwrite: bool) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return True
    shutil.copy2(src, dst)
    return True


def copy_tree(src: Path, dst: Path, *, overwrite: bool) -> bool:
    if not src.exists():
        return False
    if dst.exists() and overwrite:
        shutil.rmtree(dst)
    if dst.exists():
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    return True


def rewrite_path_string(path: str) -> str:
    p = path.replace("\\", "/")
    replacements = [
        ("datasets/vlm_prompt_dataset/frames/", "artifacts/frames/"),
        ("./datasets/vlm_prompt_dataset/frames/", "artifacts/frames/"),
        ("vlm_prompt_dataset/frames/", "artifacts/frames/"),
        ("datasets/vlm_prompt_dataset/", ""),
        ("./datasets/vlm_prompt_dataset/", ""),
    ]
    for old, new in replacements:
        if old in p:
            p = p.replace(old, new)

    if p.startswith("/") and "/frames/" in p:
        tail = p.split("/frames/", 1)[1]
        p = "artifacts/frames/" + tail

    return p


def rewrite_vlm_entry_paths(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    frames = row.get("anchor_frames")

    if isinstance(frames, list):
        row["anchor_frames"] = [rewrite_path_string(str(p)) for p in frames]
    elif isinstance(frames, dict):
        row["anchor_frames"] = {
            key: [rewrite_path_string(str(p)) for p in value]
            for key, value in frames.items()
        }

    return row


def resolve_sim_dir(sweep_root: Path, row: dict[str, Any]) -> Path | None:
    output_dir = row.get("output_dir")
    if output_dir:
        p = Path(output_dir)
        for candidate in [p, Path.cwd() / p, sweep_root / p]:
            if candidate.exists():
                return candidate

    param = row.get("parameter_name") or row.get("causal_factor")
    scene_id = row.get("scene_id")
    if param and scene_id:
        candidate = sweep_root / str(param) / str(scene_id)
        if candidate.exists():
            return candidate

    return None


def find_first_existing(sim_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        p = sim_dir / name
        if p.exists():
            return p
    return None


def prepare_vlm_entries(datasets_root: Path, hf_dir: Path) -> dict[str, Any]:
    src = datasets_root / "vlm_prompt_dataset" / "vlm_entries.jsonl"
    if not src.exists():
        raise FileNotFoundError(f"Missing VLM entries file: {src}")

    rows = read_jsonl(src)
    rows = [rewrite_vlm_entry_paths(row) for row in rows]

    write_jsonl(hf_dir / "data" / "vlm_entries.jsonl", rows)

    return {
        "num_vlm_entries": len(rows),
        "num_entries_by_scene": dict(Counter(row.get("scene_family", "unknown") for row in rows)),
        "num_entries_by_query_type": dict(Counter(row.get("query_type", "unknown") for row in rows)),
    }


def copy_summary(datasets_root: Path, hf_dir: Path, computed_summary: dict[str, Any]) -> dict[str, Any]:
    src = datasets_root / "vlm_prompt_dataset" / "dataset_card_summary.json"
    summary = read_json(src) if src.exists() else {}
    summary.update(computed_summary)
    (hf_dir / "dataset_card_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def copy_manifests(datasets_root: Path, hf_dir: Path, *, overwrite: bool) -> None:
    all_rows = []

    for scene_key, info in SCENES.items():
        src_scene_dir = datasets_root / info["sweep_dir"]
        dst_scene_dir = hf_dir / "scenes" / scene_key
        dst_scene_dir.mkdir(parents=True, exist_ok=True)

        copy_file(src_scene_dir / "manifest.jsonl", dst_scene_dir / "manifest.jsonl", overwrite=overwrite)
        copy_file(src_scene_dir / "manifest.csv", dst_scene_dir / "manifest.csv", overwrite=overwrite)

        for row in read_jsonl(src_scene_dir / "manifest.jsonl"):
            row = dict(row)
            row["scene_key"] = scene_key
            all_rows.append(row)

    write_jsonl(hf_dir / "data" / "manifest_all.jsonl", all_rows)


def copy_frames(datasets_root: Path, hf_dir: Path, *, overwrite: bool) -> None:
    src_frames = datasets_root / "vlm_prompt_dataset" / "frames"
    dst_frames = hf_dir / "artifacts" / "frames"

    if not src_frames.exists():
        print(f"[warning] no frames folder found at {src_frames}")
        return

    copy_tree(src_frames, dst_frames, overwrite=overwrite)


def copy_videos_and_trajectories(
    datasets_root: Path,
    hf_dir: Path,
    *,
    overwrite: bool,
    copy_videos: bool,
    copy_trajectories: bool,
) -> dict[str, int]:
    stats = {
        "videos_copied": 0,
        "trajectories_copied": 0,
        "metadata_or_config_copied": 0,
        "missing_videos": 0,
        "missing_trajectories": 0,
    }

    for scene_key, info in SCENES.items():
        sweep_root = datasets_root / info["sweep_dir"]
        rows = read_jsonl(sweep_root / "manifest.jsonl")

        if not rows:
            print(f"[warning] no manifest rows found for {scene_key} at {sweep_root}")
            continue

        for row in rows:
            sim_dir = resolve_sim_dir(sweep_root, row)
            if sim_dir is None:
                continue

            scene_id = str(row.get("scene_id") or sim_dir.name)
            param = str(row.get("parameter_name") or row.get("causal_factor") or "unknown_param")

            if copy_videos:
                src_video = find_first_existing(sim_dir, info["video_names"])
                if src_video is None:
                    stats["missing_videos"] += 1
                else:
                    dst_video = hf_dir / "artifacts" / "videos" / scene_key / param / f"{scene_id}{src_video.suffix}"
                    if copy_file(src_video, dst_video, overwrite=overwrite):
                        stats["videos_copied"] += 1

            if copy_trajectories:
                src_result = find_first_existing(sim_dir, info["result_names"])
                if src_result is None:
                    stats["missing_trajectories"] += 1
                else:
                    dst_result = hf_dir / "artifacts" / "trajectories" / scene_key / param / f"{scene_id}{src_result.suffix}"
                    if copy_file(src_result, dst_result, overwrite=overwrite):
                        stats["trajectories_copied"] += 1

            for small_name in ["metadata.json", "resolved_config.json"]:
                src_small = sim_dir / small_name
                if src_small.exists():
                    dst_small = hf_dir / "artifacts" / "trajectories" / scene_key / param / scene_id / small_name
                    if copy_file(src_small, dst_small, overwrite=overwrite):
                        stats["metadata_or_config_copied"] += 1

    return stats


def prepare_readme(hf_dir: Path, repo_id: str, summary: dict[str, Any], private: bool) -> None:
    by_scene = json.dumps(summary.get("num_entries_by_scene", {}), indent=2, sort_keys=True)
    by_query_type = json.dumps(summary.get("num_entries_by_query_type", {}), indent=2, sort_keys=True)
    num_entries = summary.get("num_vlm_entries", "unknown")

    lines = [
        "---",
        "license: mit",
        "task_categories:",
        "- visual-question-answering",
        "- video-classification",
        "- robotics",
        "language:",
        "- en",
        "tags:",
        "- physics",
        "- physical-reasoning",
        "- world-models",
        "- embodied-ai",
        "- genesis",
        "- simulation",
        "- counterfactual",
        "pretty_name: Genesis Physical Intervention Benchmark",
        "---",
        "",
        "# Genesis Physical Intervention Benchmark",
        "",
        "This dataset contains controlled Genesis simulations for evaluating physically viable world models in embodied AI settings.",
        "",
        f"Repository: `{repo_id}`",
        f"Visibility at upload time: `{'private' if private else 'public'}`",
        "",
        "## Scenes",
        "",
        "- Ramp-cup-water",
        "- Robotic pour",
        "- Pendulum",
        "",
        "## Task Types",
        "",
        "- Single-rollout outcome prediction",
        "- Scalar physical prediction",
        "- Inverse parameter prediction",
        "- Pairwise counterfactual comparison",
        "",
        "## Files",
        "",
        "```text",
        "data/vlm_entries.jsonl",
        "data/manifest_all.jsonl",
        "scenes/",
        "artifacts/frames/",
        "artifacts/videos/",
        "artifacts/trajectories/",
        "```",
        "",
        "## Dataset Size Summary",
        "",
        f"Number of VLM entries: `{num_entries}`",
        "",
        "Entries by scene:",
        "",
        "```json",
        by_scene,
        "```",
        "",
        "Entries by query type:",
        "",
        "```json",
        by_query_type,
        "```",
        "",
        "## Loading Example",
        "",
        "```python",
        "from datasets import load_dataset",
        f'ds = load_dataset("{repo_id}", data_files="data/vlm_entries.jsonl", split="train")',
        "print(ds[0])",
        "```",
        "",
        "## Limitations",
        "",
        "- Labels are simulator-derived, not real-world measurements.",
        "- Some labels use proxies, such as receiver_fraction for robotic pouring.",
        "- This is a research benchmark for physical reasoning, not a complete real-world robotics benchmark.",
        "",
    ]

    (hf_dir / "README.md").write_text("\n".join(lines))


def prepare_hf_dataset(args: argparse.Namespace) -> dict[str, Any]:
    datasets_root = args.datasets_root
    hf_dir = args.hf_dataset_dir

    if hf_dir.exists() and args.clean:
        shutil.rmtree(hf_dir)

    hf_dir.mkdir(parents=True, exist_ok=True)
    (hf_dir / "data").mkdir(parents=True, exist_ok=True)
    (hf_dir / "scenes").mkdir(parents=True, exist_ok=True)
    (hf_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    computed_summary = prepare_vlm_entries(datasets_root, hf_dir)
    summary = copy_summary(datasets_root, hf_dir, computed_summary)

    copy_manifests(datasets_root, hf_dir, overwrite=args.overwrite)
    copy_frames(datasets_root, hf_dir, overwrite=args.overwrite)

    artifact_stats = copy_videos_and_trajectories(
        datasets_root,
        hf_dir,
        overwrite=args.overwrite,
        copy_videos=not args.no_videos,
        copy_trajectories=not args.no_trajectories,
    )

    prepare_readme(hf_dir, args.repo_id, summary, args.private)

    upload_summary = {
        "hf_dataset_dir": str(hf_dir),
        "repo_id": args.repo_id,
        "summary": summary,
        "artifact_stats": artifact_stats,
        "included_videos": not args.no_videos,
        "included_trajectories": not args.no_trajectories,
    }

    (hf_dir / "prepare_summary.json").write_text(json.dumps(upload_summary, indent=2, sort_keys=True))
    return upload_summary


def infer_repo_id_from_username(repo_name: str) -> str:
    from huggingface_hub import HfApi
    api = HfApi()
    who = api.whoami()
    username = who.get("name")
    if not username:
        raise RuntimeError("Could not infer Hugging Face username. Pass --repo-id USERNAME/REPO.")
    return f"{username}/{repo_name}"


def upload_to_hub(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(args.hf_dataset_dir),
        path_in_repo=".",
        commit_message=args.commit_message,
        ignore_patterns=args.ignore_patterns,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    parser.add_argument("--hf-dataset-dir", type=Path, default=Path("hf_dataset"))

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--repo-id", type=str, default=None)
    group.add_argument("--repo-name", type=str, default=None)

    parser.add_argument("--private", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--no-trajectories", action="store_true")
    parser.add_argument("--commit-message", type=str, default="Upload Genesis physical intervention benchmark dataset")
    parser.add_argument("--ignore-patterns", nargs="*", default=["**/.DS_Store", "**/__pycache__/**"])

    args = parser.parse_args()

    if args.repo_id is None:
        if args.repo_name is None:
            args.repo_name = "genesis-physical-interventions"
        if args.prepare_only:
            args.repo_id = f"sarahnator/{args.repo_name}"
        else:
            args.repo_id = infer_repo_id_from_username(args.repo_name)

    return args


def main() -> None:
    args = parse_args()

    print(f"[prepare] staging dataset in {args.hf_dataset_dir}")
    summary = prepare_hf_dataset(args)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.prepare_only:
        print("[done] prepare-only mode; not uploading.")
        print(f"Inspect staged folder: {args.hf_dataset_dir}")
        return

    print(f"[upload] uploading to dataset repo {args.repo_id}")
    upload_to_hub(args)
    print(f"[done] uploaded to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()


"""
Example usage:
python -u scripts/prepare_and_upload_hf_dataset.py   --datasets-root datasets   --hf-dataset-dir hf_dataset   --repo-name genesis-physical-interventions   --prepare-only   --clean   --overwrite
"""