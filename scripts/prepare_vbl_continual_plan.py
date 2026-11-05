#!/usr/bin/env python
"""Build a unit-level continual-learning plan for VBL-VA001.

The packed dataset stores one record per selected file, ``group_size = 1``, so a
record group is one file.  Files of the same *unit* are different recordings of
the same physical condition and must not straddle partitions, so this script
allocates **units** (not files) and expands them to their record groups.

Shares per class: holdout 20%, validation 10%, round 0 25%, round 1 20%, round 2
25%, assigned with the largest-remainder method so they always sum to the unit
count exactly.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from entropy_agent.continual import BatchPlan, save_plan  # noqa: E402
from entropy_agent.continual.plan import Batch  # noqa: E402

SHARES = (
    ("holdout", 0.20),
    ("validation", 0.10),
    ("round0", 0.25),
    ("round1", 0.20),
    ("round2", 0.25),
)


def _largest_remainder(total: int, shares: tuple[tuple[str, float], ...]) -> dict[str, int]:
    raw = {name: share * total for name, share in shares}
    counts = {name: int(value) for name, value in raw.items()}
    remaining = total - sum(counts.values())
    for name, _ in sorted(shares, key=lambda item: -(raw[item[0]] - int(raw[item[0]]))):
        if remaining <= 0:
            break
        counts[name] += 1
        remaining -= 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.005)
    parser.add_argument("--dataset-id", default=None, help="override the dataset id from the manifest")
    parser.add_argument("--seed", type=int, default=20261104, help="shuffle units before allocating")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    dataset_id = args.dataset_id or manifest["dataset_id"]

    holdout: list[str] = []
    validation: list[str] = []
    rounds: dict[str, list[str]] = {"round0": [], "round1": [], "round2": []}

    for entry in manifest["classes"]:
        class_id = int(entry["class_id"])
        by_unit: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(entry["files"]):
            by_unit[row["unit"]].append(index)
        units = sorted(by_unit)
        random.Random(args.seed + class_id).shuffle(units)
        counts = _largest_remainder(len(units), SHARES)
        cursor = 0
        for name, _ in SHARES:
            chosen = units[cursor : cursor + counts[name]]
            cursor += counts[name]
            groups = [
                f"class{class_id + 1}.npy#group-{index}"
                for unit in chosen
                for index in by_unit[unit]
            ]
            if name == "holdout":
                holdout.extend(groups)
            elif name == "validation":
                validation.extend(groups)
            else:
                rounds[name].extend(groups)

    plan = BatchPlan(
        dataset_id=dataset_id,
        batches=(
            Batch(batch_id=0, groups=tuple(rounds["round0"]), note="round 0 (initial)"),
            Batch(batch_id=1, groups=tuple(rounds["round1"]), note="round 1"),
            Batch(batch_id=2, groups=tuple(rounds["round2"]), note="round 2"),
        ),
        validation_groups=tuple(validation),
        holdout_groups=tuple(holdout),
        margin=args.margin,
        notes="unit-isolated split; 20% holdout, 10% validation, 25/20/25% rounds",
    )
    save_plan(args.output, plan)

    print(f"plan -> {args.output}")
    print(f"records: holdout {len(holdout)}, validation {len(validation)}, "
          f"round0 {len(rounds['round0'])}, round1 {len(rounds['round1'])}, "
          f"round2 {len(rounds['round2'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
