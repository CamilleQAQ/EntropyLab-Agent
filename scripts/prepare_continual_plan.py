#!/usr/bin/env python
"""Build an explicit continual-learning batch plan for a dataset.

Two dataset-level strategies are offered, both file/group level and auditable:

``chronological``  groups of each class are ordered as stored; the last block per
                   class becomes the holdout, the block before it the validation,
                   and the rest is split into equal consecutive batches.
``by-load``        groups are grouped by the load recorded for their source file
                   (``--load-manifest``); one holdout and one validation group per
                   class, the rest form one batch per load tier.

The strategy is dataset knowledge, so it lives in this adapter script, not in the
core: the core only ever sees a plan of group names.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from entropy_agent.continual import BatchPlan, save_plan  # noqa: E402
from entropy_agent.continual.plan import Batch  # noqa: E402
from entropy_agent.io.dataset import load_dataset_records, load_dataset_spec  # noqa: E402


def _group_index(group: str) -> int:
    return int(group.rsplit("-", 1)[-1])


def _order(records) -> dict[int, list[str]]:
    """label -> group names in their stored order."""
    grouped: dict[int, set[str]] = defaultdict(set)
    for record in records:
        grouped[int(record.label)].add(record.group)
    return {
        label: sorted(groups, key=_group_index) for label, groups in sorted(grouped.items())
    }


def _take_per_class(
    ordered: dict[int, list[str]], count: int, *, from_end: bool
) -> tuple[list[str], dict[int, list[str]]]:
    picked: list[str] = []
    remaining: dict[int, list[str]] = {}
    for label, groups in ordered.items():
        if len(groups) <= count:
            raise SystemExit(
                f"class {label} has only {len(groups)} groups; cannot reserve {count}"
            )
        chosen = groups[-count:] if from_end else groups[:count]
        rest = groups[:-count] if from_end else groups[count:]
        picked.extend(chosen)
        remaining[label] = rest
    return picked, remaining


def _chronological(ordered, batches: int, reserve: int) -> tuple[list[list[str]], list[str], list[str]]:
    holdout, remaining = _take_per_class(ordered, reserve, from_end=True)
    validation, remaining = _take_per_class(remaining, reserve, from_end=True)
    per_class = min(len(groups) for groups in remaining.values())
    if per_class < batches:
        raise SystemExit(
            f"only {per_class} groups per class left for {batches} batches"
        )
    size = per_class // batches
    batch_groups: list[list[str]] = [[] for _ in range(batches)]
    for groups in remaining.values():
        for index in range(batches):
            start = index * size
            end = len(groups) if index == batches - 1 else (index + 1) * size
            batch_groups[index].extend(groups[start:end])
    return batch_groups, validation, holdout


def _by_load(ordered, load_by_group: dict[str, float], *, stratified_validation: bool):
    holdout, remaining = _take_per_class(ordered, 1, from_end=True)
    if stratified_validation:
        # One validation group per (class, load) keeps every load tier visible to
        # the promotion decision from the first batch on.
        validation: list[str] = []
        for label, groups in list(remaining.items()):
            by_tier: dict[float, list[str]] = defaultdict(list)
            for group in groups:
                by_tier[load_by_group[group]].append(group)
            validation.extend(tier_groups[0] for tier_groups in by_tier.values())
        chosen = set(validation)
        remaining = {label: [g for g in groups if g not in chosen] for label, groups in remaining.items()}
    else:
        validation, remaining = _take_per_class(remaining, 1, from_end=True)
    tiers: dict[float, list[str]] = defaultdict(list)
    for groups in remaining.values():
        for group in groups:
            if group not in load_by_group:
                raise SystemExit(f"group {group} has no recorded load")
            tiers[load_by_group[group]].append(group)
    batch_groups = [sorted(tiers[load], key=_group_index) for load in sorted(tiers)]
    return batch_groups, validation, holdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strategy", choices=("chronological", "by-load"), required=True)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--reserve", type=int, default=2, help="chronological: groups per class")
    parser.add_argument("--load-manifest", type=Path, default=None)
    parser.add_argument("--margin", type=float, default=0.005)
    parser.add_argument(
        "--stratified-validation",
        action="store_true",
        help="by-load plans: validate on one group per (class, load) instead of per class",
    )
    args = parser.parse_args()

    spec = load_dataset_spec(args.spec)
    records = load_dataset_records(spec, args.data_root)
    ordered = _order(records)

    if args.strategy == "chronological":
        batch_groups, validation, holdout = _chronological(ordered, args.batches, args.reserve)
        notes = f"chronological blocks; {args.reserve} holdout + {args.reserve} validation groups per class"
    else:
        if args.load_manifest is None:
            raise SystemExit("--load-manifest is required for --strategy by-load")
        manifest = json.loads(args.load_manifest.read_text(encoding="utf-8"))
        load_by_group: dict[str, float] = {}
        for entry in manifest["classes"]:
            for index, source in enumerate(entry["sources"]):
                load_by_group[f"class{entry['class_id'] + 1}.npy#group-{index}"] = float(
                    source["load_nm"]
                )
        batch_groups, validation, holdout = _by_load(
            ordered, load_by_group, stratified_validation=args.stratified_validation
        )
        notes = (
            "load tiers; 1 holdout per class; validation = one group per (class, load)"
            if args.stratified_validation
            else "load tiers; 1 holdout + 1 validation group per class"
        )

    name_by_label = dict(enumerate(spec.labels.names))
    plan = BatchPlan(
        dataset_id=spec.dataset_id,
        batches=tuple(
            Batch(batch_id=index, groups=tuple(groups), note=f"batch {index}")
            for index, groups in enumerate(batch_groups)
        ),
        validation_groups=tuple(validation),
        holdout_groups=tuple(holdout),
        margin=args.margin,
        notes=notes,
    )
    # Label each batch with the load tiers it actually contains, when known.
    if args.strategy == "by-load" and "load_by_group" in locals():
        batches = []
        for index, groups in enumerate(batch_groups):
            loads = sorted({load_by_group[group] for group in groups})
            batches.append(
                Batch(
                    batch_id=index,
                    groups=tuple(groups),
                    note=" + ".join(f"{load:g} Nm" for load in loads),
                )
            )
        plan = BatchPlan(
            dataset_id=plan.dataset_id,
            batches=tuple(batches),
            validation_groups=plan.validation_groups,
            holdout_groups=plan.holdout_groups,
            margin=plan.margin,
            notes=plan.notes,
        )

    info = plan.validate(records)
    save_plan(args.output, plan)

    print(f"plan -> {args.output}")
    print(f"batches: {info['groups']['batches']}")
    print(f"validation groups: {info['groups']['validation']}  holdout groups: {info['groups']['holdout']}")
    for batch in plan.batches:
        classes = sorted(
            {
                name_by_label[int(record.label)]
                for record in records
                if record.group in set(batch.groups)
            }
        )
        print(f"  batch {batch.batch_id} ({batch.note}): {len(batch.groups)} groups, "
              f"classes {classes}")
    if not info["initial_batch_covers_all_classes"]:
        print(f"  note: initial batch misses {info['initial_batch_missing_classes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
