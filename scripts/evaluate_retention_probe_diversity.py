"""Compare equally sized retention probes covering six or nine bearings."""

from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from random import Random

from compare_continual_gates import (
    DATA, HISTORICAL_SEEDS, PLANS, ROOT, assessment_scores, assess, correct_counts,
    fit_rounds, load, measurement_cells, replay, save,
)
from compare_retention_probe_sizes import expand
from entropy_agent.continual.plan import BatchPlan
from entropy_agent.io.dataset import load_dataset_records, load_dataset_spec
from entropy_agent.training.pipeline import extract_features
from evaluate_retention_on_new_bearings import annotate_errors, holdout_inventory, holdout_records

HOLDOUT = (("K005",), ("KA07", "KA09"), ("KI07", "KI21"))


def probe_members(plan, metadata, seed):
    rng = Random(seed)
    concentrated, distributed = {}, {}
    for batch in plan.batches[:-1]:
        cells = defaultdict(lambda: defaultdict(list))
        for group in (*batch.probe_groups, *batch.retention_probe_groups):
            label, bearing = metadata[group]
            cells[label][bearing].append(group)
        name = f"batch-{batch.batch_id}"
        concentrated[name], distributed[name] = [], []
        for cell in cells.values():
            bearings = sorted(cell)
            if len(bearings) != 3 or any(len(groups) != 6 for groups in cell.values()):
                raise ValueError("expected three bearings per class with six reserved measurements each")
            rng.shuffle(bearings)
            shuffled = [rng.sample(cell[b], len(cell[b])) for b in bearings]
            # Share six of nine measurements per class. Only three are moved to
            # the third bearing when coverage increases.
            concentrated[name].extend(shuffled[0][:5] + shuffled[1][:4])
            distributed[name].extend(shuffled[0][:3] + shuffled[1][:3] + shuffled[2][:3])
    return {"six_bearings": concentrated, "nine_bearings": distributed}


def score_probes(spec, plan, bundle, models, old_probes):
    rounds = []
    for index in range(1, len(models)):
        visible = plan.visible_probes(index)
        for previous in range(index):
            name = f"batch-{previous}"
            visible[name] = old_probes[name]
        rounds.append({
            name: [correct_counts(spec, bundle, model, set(groups)) for model in models[:index + 1]]
            for name, groups in visible.items()
        })
    return rounds


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data-sources/paderborn-raw")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    spec = load_dataset_spec(ROOT / "configs/paderborn_3class_fusion_time_extratrees.json")
    manifest = load(DATA / "manifest.json")
    metadata = {
        f'{entry["packed"]}#group-{i}': (entry["class_id"], row["bearing"])
        for entry in manifest["classes"] for i, row in enumerate(entry["files"][::2])
    }
    inventory = holdout_inventory(args.raw_root, HOLDOUT)
    if {bearing for _, bearing in metadata.values()}.intersection(row["bearing"] for row in inventory):
        raise ValueError("holdout bearing overlaps the continual development manifest")
    save(args.output / "holdout_inventory.json", inventory)
    records = load_dataset_records(spec, DATA)
    cells = measurement_cells(manifest)
    members, locked = {}, {}
    for seed in HISTORICAL_SEEDS:
        plan = expand(BatchPlan.load(PLANS / f"paderborn-plan-{seed}.json"), cells)
        plan.validate(records)
        prior_holdout = set(plan.holdout_groups)
        bundle = extract_features(spec, DATA, records=[row for row in records if row.group not in prior_holdout])
        models = fit_rounds(spec, plan, bundle)
        members[seed] = probe_members(plan, metadata, seed)
        decisions = {
            arm: replay("current_gate", score_probes(spec, plan, bundle, models, probes), {})
            for arm, probes in members[seed].items()
        }
        locked[seed] = (plan, models, decisions)
        print(f"Locked decisions for {seed}", flush=True)
    save(args.output / "probe_members.json", members)
    save(args.output / "locked_decisions.json", {seed: decisions for seed, (_, _, decisions) in locked.items()})

    rows, conditions = holdout_records(inventory, (args.output / "holdout_windows.npy").resolve())
    bundle = extract_features(spec, args.output, records=rows)
    assessments = {
        "fresh_fault_bearings": {row.group for row in rows if row.label != spec.labels.healthy_class_id},
        "three_class_with_historical_healthy_bearing": set(conditions),
    }
    results = {}
    for name, groups in assessments.items():
        results[name] = {}
        for seed, (plan, models, decisions) in locked.items():
            external = replace(plan, holdout_groups=tuple(sorted(groups)))
            scores = assessment_scores(spec, external, bundle, models, conditions)
            results[name][seed] = {
                arm: annotate_errors(assess([step.copy() for step in steps], scores))
                for arm, steps in decisions.items()
            }
    save(args.output / "holdout_results.json", results)


if __name__ == "__main__":
    main()
