#!/usr/bin/env python3
"""
Validate that a sweep changes only the intended parameter.

Usage:
  python scripts/validate_one_variable_sweep.py \
    outputs/datasets/ramp_cup_sweep/ball_density \
    --allowed-changing BALL_DENSITY seed scene_id sweep_value

Example: For a sweep that varies only the ball density, we expect only the swept variable plus bookkeeping fields to change:
python tests/validate_one_variable_sweep.py \
  outputs/datasets/ramp_cup_sweep/ball_density \
  --allowed-changing \
  BALL_DENSITY \
  sweep_value \
  seed \
  scene_id

  If you generated multiple seeds, the checker should report something like:
  
    Changed keys:
    BALL_DENSITY: 7 unique values
    scene_id: 21 unique values
    seed: 3 unique values
    sweep_value: 7 unique values

    PASSED: only allowed keys changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict


def load_configs(root: Path):
    paths = sorted(root.glob("**/resolved_config.json"))
    if not paths:
        raise FileNotFoundError(f"No resolved_config.json files found under {root}")

    configs = []
    for path in paths:
        with path.open("r") as f:
            cfg = json.load(f)
        configs.append((path, cfg))

    return configs


def normalize(value):
    """
    Convert lists/tuples/dicts into stable JSON strings so they can be compared.
    """
    return json.dumps(value, sort_keys=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_root", type=str)
    parser.add_argument(
        "--allowed-changing",
        nargs="+",
        required=True,
        help="Config keys allowed to differ across this sweep.",
    )
    args = parser.parse_args()

    sweep_root = Path(args.sweep_root)
    allowed = set(args.allowed_changing)

    configs = load_configs(sweep_root)

    all_keys = set()
    for _, cfg in configs:
        all_keys.update(cfg.keys())

    changed = defaultdict(set)

    for key in sorted(all_keys):
        for _, cfg in configs:
            changed[key].add(normalize(cfg.get(key, None)))

    actually_changed = {
        key: values for key, values in changed.items()
        if len(values) > 1
    }

    unexpected = {
        key: values for key, values in actually_changed.items()
        if key not in allowed
    }

    print(f"Checked {len(configs)} configs under {sweep_root}")
    print("\nChanged keys:")
    for key, values in actually_changed.items():
        print(f"  {key}: {len(values)} unique values")

    if unexpected:
        print("\nFAILED: unexpected changing keys:")
        for key, values in unexpected.items():
            preview = list(values)[:5]
            print(f"  {key}: {preview}")
        raise SystemExit(1)

    print("\nPASSED: only allowed keys changed.")


if __name__ == "__main__":
    main()

