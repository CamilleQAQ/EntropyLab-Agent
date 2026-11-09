#!/usr/bin/env python
"""Build the Paderborn continual-learning plan (3 classes, 3 operating conditions).

Allocation, per (class, condition), from the 20 measurements:

* holdout   4 measurements - frozen, never trained, scored once at the end;
* probe     3 measurements - held out from training to measure retention and
            new-domain gain;
* train     13 measurements - the round's training data.

Rounds are the three N15 operating conditions, ordered from the reference
condition to the more distant ones. ``--seed`` reshuffles which measurements take
each role, so several independent splits can be aggregated.
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

ROUNDS = ("N15_M07_F10", "N15_M01_F10", "N15_M07_F04")
HOLDOUT_PER_CELL = 4
PROBE_PER_CELL = 3
VALIDATION_PER_CELL = 3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--margin", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--extra-retention-per-cell", type=int, choices=range(10), default=0,
                        help="additional measurements per class and prior condition, reserved from training")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    dataset_id = args.dataset_id or manifest["dataset_id"]
    windows = int(manifest["windows_per_measurement"])
    rng = random.Random(args.seed)

    holdout: list[str] = []
    validation: list[str] = []
    round_groups: dict[str, list[str]] = defaultdict(list)
    round_probes: dict[str, list[str]] = defaultdict(list)
    retention_probes: dict[str, list[str]] = defaultdict(list)

    for entry in manifest["classes"]:
        class_id = int(entry["class_id"])
        # measurement index -> record indices, per condition
        by_cell: dict[tuple[str, int], list[int]] = defaultdict(list)
        for index, row in enumerate(entry["files"]):
            by_cell[(row["condition"], int(row["measurement"]))].append(index)

        for condition in ROUNDS:
            measurements = sorted(
                key[1] for key in by_cell if key[0] == condition
            )
            if len(measurements) != 20:
                raise SystemExit(
                    f"{entry['label']} {condition}: expected 20 measurements, got {len(measurements)}"
                )
            rng.shuffle(measurements)
            def groups_for(chosen: list[int]) -> list[str]:
                collected: list[str] = []
                seen: set[str] = set()
                for measurement in chosen:
                    for index in by_cell[(condition, measurement)]:
                        name = f"class{class_id + 1}.npy#group-{index // windows}"
                        if name not in seen:
                            seen.add(name)
                            collected.append(name)
                return collected

            holdout_measurements = measurements[:HOLDOUT_PER_CELL]
            probe_measurements = measurements[HOLDOUT_PER_CELL : HOLDOUT_PER_CELL + PROBE_PER_CELL]
            used = HOLDOUT_PER_CELL + PROBE_PER_CELL
            if condition == ROUNDS[0]:
                validation_measurements = measurements[used : used + VALIDATION_PER_CELL]
                used += VALIDATION_PER_CELL
                validation.extend(groups_for(validation_measurements))
            extra = args.extra_retention_per_cell if condition != ROUNDS[-1] else 0
            retention_probes[condition].extend(groups_for(measurements[used : used + extra]))
            train_measurements = measurements[used + extra:]

            holdout.extend(groups_for(holdout_measurements))
            round_probes[condition].extend(groups_for(probe_measurements))
            round_groups[condition].extend(groups_for(train_measurements))

    batches = []
    for batch_id, condition in enumerate(ROUNDS):
        batches.append(
            Batch(
                batch_id=batch_id,
                groups=tuple(round_groups[condition]),
                note=condition,
                probe_groups=tuple(round_probes.get(condition, ())),
            retention_probe_groups=tuple(retention_probes.get(condition, ())),
            )
        )

    plan = BatchPlan(
        dataset_id=dataset_id,
        batches=tuple(batches),
        validation_groups=tuple(validation),
        holdout_groups=tuple(holdout),
        margin=args.margin,
        notes=(
            "per (class, condition): 4 holdout + 3 probe + rest train "
            "(reference condition also reserves 3 validation measurements); "
            f"earlier conditions reserve {args.extra_retention_per_cell} additional retention measurements per class; "
            "rounds = N15_M07_F10 -> N15_M01_F10 -> N15_M07_F04"
        ),
    )
    save_plan(args.output, plan)

    print(f"plan -> {args.output}")
    for batch in plan.batches:
        print(
            f"  round {batch.batch_id} ({batch.note}): "
            f"train {len(batch.groups)} groups, probe {len(batch.probe_groups)} groups, "
            f"extra retention {len(batch.retention_probe_groups)} groups"
        )
    print(
        f"  validation {len(plan.validation_groups)} groups, "
        f"holdout {len(plan.holdout_groups)} groups"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
