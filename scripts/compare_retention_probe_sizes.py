"""Replay the Paderborn guardrail with 27 versus 54 old-condition measurements."""

from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path

from compare_continual_gates import (
    DATA, HISTORICAL_SEEDS, PLANS, ROOT, assessment_scores, assess,
    fit_rounds, load, measurement_cells, probe_results, replay, save,
)
from entropy_agent.continual.plan import BatchPlan
from entropy_agent.io.dataset import load_dataset_records, load_dataset_spec
from entropy_agent.training.pipeline import extract_features


def expand(plan, cells):
    batches = []
    for batch in plan.batches[:-1]:
        available = set(batch.groups)
        extras = []
        for (_, condition), measurements in cells.items():
            if condition != batch.note:
                continue
            # Plan order follows the original seeded shuffle. Keep each measurement
            # index together across the bearings represented in this class.
            positions = {group: i for i, group in enumerate(batch.groups)}
            eligible = [groups for groups in measurements.values() if available.issuperset(groups)]
            eligible.sort(key=lambda groups: min(positions[group] for group in groups))
            if len(eligible) < 3:
                raise ValueError("each class and prior condition needs three extra measurement indices")
            extras.extend(group for groups in eligible[:3] for group in groups)
        extra_set = set(extras)
        batches.append(replace(batch, groups=tuple(g for g in batch.groups if g not in extra_set),
                               retention_probe_groups=tuple(extras)))
    return replace(plan, batches=(*batches, plan.batches[-1]))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=False)
    spec = load_dataset_spec(ROOT / "configs/paderborn_3class_fusion_time_extratrees.json")
    records = load_dataset_records(spec, DATA)
    cells = measurement_cells(load(DATA / "manifest.json"))
    conditions = {g: condition for (_, condition), measurements in cells.items()
                  for groups in measurements.values() for g in groups}
    plans = {}
    for seed in HISTORICAL_SEEDS:
        original = BatchPlan.load(PLANS / f"paderborn-plan-{seed}.json")
        large = expand(original, cells)
        small = replace(large, batches=tuple(replace(batch, retention_probe_groups=()) for batch in large.batches))
        original.validate(records)
        large.validate(records)
        plans[seed] = (original, small, large)
    save(output / "plans.json", {seed: large.to_dict() for seed, (_, _, large) in plans.items()})

    # Keep each plan's test records out of its fit and decision bundle.
    locked = {}
    for seed, (original, small, large) in plans.items():
        test = set(original.holdout_groups)
        bundle = extract_features(spec, DATA, records=[row for row in records if row.group not in test])
        original_models = fit_rounds(spec, original, bundle)
        shared_models = fit_rounds(spec, large, bundle)
        arms = {
            "original_27": (original, original_models),
            "matched_27": (small, shared_models),
            "expanded_54": (large, shared_models),
        }
        locked[seed] = {
            name: (models, replay("current_gate", probe_results(spec, plan, bundle, models), {}))
            for name, (plan, models) in arms.items()
        }
        print(f"Locked {seed}: {[(name, [step['promoted'] for step in steps]) for name, (_, steps) in locked[seed].items()]}", flush=True)
    save(output / "locked_decisions.json", {s: {n: steps for n, (_, steps) in a.items()} for s, a in locked.items()})
    results = {}
    for seed, (plan, _, _) in plans.items():
        test = set(plan.holdout_groups)
        bundle = extract_features(spec, DATA, records=[row for row in records if row.group in test])
        results[seed] = {
            name: assess(steps, assessment_scores(spec, plan, bundle, models, conditions))
            for name, (models, steps) in locked[seed].items()
        }
    save(output / "holdout_replay.json", results)


if __name__ == "__main__":
    main()
