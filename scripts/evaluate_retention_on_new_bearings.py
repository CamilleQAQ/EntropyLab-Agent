"""Evaluate 27 versus 54 retention measurements on disjoint Paderborn bearings."""

from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path

import numpy as np

from compare_continual_gates import (
    DATA, HISTORICAL_SEEDS, PLANS, ROOT, assessment_scores, assess, fit_rounds,
    load, measurement_cells, probe_results, replay, save,
)
from compare_retention_probe_sizes import expand
from entropy_agent.continual.plan import BatchPlan
from entropy_agent.io.dataset import DatasetRecord, load_dataset_records, load_dataset_spec
from entropy_agent.io.input_signal import SignalInput
from entropy_agent.training.pipeline import extract_features
from prepare_paderborn_3class import CONDITIONS, OFFSET, WINDOW, WINDOWS_PER_MEASUREMENT, _load_vibration

# Same three conditions as the development stream; two pure-label bearings per class.
HOLDOUT_BEARINGS = (("K004", "K006"), ("KA22", "KA30"), ("KI17", "KI18"))


def holdout_inventory(raw_root, bearing_classes=HOLDOUT_BEARINGS):
    inventory = []
    for label, bearings in enumerate(bearing_classes):
        for bearing in bearings:
            for condition in CONDITIONS:
                folder = raw_root / bearing
                files = sorted(folder.glob(f"{condition}_{bearing}_*.mat"))
                expected = {f"{condition}_{bearing}_{i}.mat" for i in range(1, 21)}
                if {path.name for path in files} != expected:
                    raise ValueError(f"incomplete holdout inventory for {bearing}, {condition}")
                inventory.extend({"label": label, "bearing": bearing, "condition": condition,
                                  "source": str(path.resolve())} for path in files)
    return inventory


def holdout_records(inventory, packed_path):
    rows, windows, conditions = [], [], {}
    for item in inventory:
        source = Path(item["source"])
        signal = _load_vibration(source)
        group = source.stem
        conditions[group] = item["condition"]
        for index in range(WINDOWS_PER_MEASUREMENT):
            start = OFFSET + index * WINDOW
            samples = signal[start : start + WINDOW]
            if len(samples) != WINDOW or not np.isfinite(samples).all():
                raise ValueError(f"invalid holdout signal in {source}")
            rows.append(DatasetRecord(f"{group}#{index}", item["label"], group,
                                      SignalInput(path=packed_path, indices=(len(windows),))))
            windows.append(samples)
    np.save(packed_path, np.asarray(windows, dtype=np.float32), allow_pickle=False)
    return rows, conditions


def annotate_errors(steps):
    for step in steps[1:]:
        gains = list(step["candidate_gain_by_condition"].values())
        exceeds_loss_limit = any(item["gain"] < -0.02 for item in gains[:-1])
        qualifies = not exceeds_loss_limit and gains[-1]["gain"] > 0
        step["false_rejection"] = not step["promoted"] and qualifies
        step["promotion_with_forgetting"] = step["promoted"] and exceeds_loss_limit
    return steps


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data-sources/paderborn-raw")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    spec = load_dataset_spec(ROOT / "configs/paderborn_3class_fusion_time_extratrees.json")
    manifest = load(DATA / "manifest.json")
    development_bearings = {bearing for entry in manifest["classes"] for bearing in entry["bearings"]}
    inventory = holdout_inventory(args.raw_root)
    if development_bearings.intersection(row["bearing"] for row in inventory):
        raise ValueError("holdout bearing overlaps the development manifest")
    save(args.output / "holdout_inventory.json", inventory)
    records = load_dataset_records(spec, DATA)
    cells = measurement_cells(manifest)
    locked = {}
    for seed in HISTORICAL_SEEDS:
        original = BatchPlan.load(PLANS / f"paderborn-plan-{seed}.json")
        large = expand(original, cells)
        small = replace(large, batches=tuple(replace(b, retention_probe_groups=()) for b in large.batches))
        large.validate(records)
        test = set(large.holdout_groups)
        bundle = extract_features(spec, DATA, records=[row for row in records if row.group not in test])
        models = fit_rounds(spec, large, bundle)
        decisions = {name: replay("current_gate", probe_results(spec, plan, bundle, models), {})
                     for name, plan in (("matched_27", small), ("expanded_54", large))}
        locked[seed] = (large, models, decisions)
        print(f"Locked decisions for {seed}", flush=True)
    save(args.output / "locked_decisions.json", {seed: decisions for seed, (_, _, decisions) in locked.items()})

    rows, conditions = holdout_records(inventory, (args.output / "holdout_windows.npy").resolve())
    bundle = extract_features(spec, args.output, records=rows)
    results = {}
    for seed, (plan, models, decisions) in locked.items():
        external = replace(plan, holdout_groups=tuple(sorted(conditions)))
        scores = assessment_scores(spec, external, bundle, models, conditions)
        results[seed] = {name: annotate_errors(assess(steps, scores)) for name, steps in decisions.items()}
    save(args.output / "holdout_results.json", results)


if __name__ == "__main__":
    main()
