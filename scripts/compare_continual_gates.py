"""Compare four fixed-model policies on the Paderborn measurement stream."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import product
import json
from pathlib import Path
import random
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from entropy_agent.continual.decision import decide  # noqa: E402
from entropy_agent.continual.loop import _prepared  # noqa: E402
from entropy_agent.continual.plan import Batch, BatchPlan  # noqa: E402
from entropy_agent.io.dataset import load_dataset_records, load_dataset_spec  # noqa: E402
from entropy_agent.training.pipeline import (  # noqa: E402
    evaluate_partition,
    extract_features,
    fit_prepared,
)
from prepare_paderborn_continual_plan import ROUNDS  # noqa: E402

DATA = ROOT / "data/paper/paderborn_3class"
PLANS = ROOT / "results/plans"
HISTORICAL_SEEDS = range(20261104, 20261109)
DEVELOPMENT_SEEDS = range(110700, 110705)
EVALUATION_SEEDS = range(110710, 110715)
ARMS = ("frozen", "always_publish", "current_gate", "paired_gate")
TOLERANCE = 0.02


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def measurement_cells(manifest):
    cells = defaultdict(lambda: defaultdict(list))
    for entry in manifest["classes"]:
        for index, row in enumerate(entry["files"][::2]):
            group = f'{entry["packed"]}#group-{index}'
            cells[entry["class_id"], row["condition"]][row["measurement"]].append(group)
    return cells


def reserve_holdout(cells, historical):
    previously_tested = {group for plan in historical.values() for group in plan.holdout_groups}
    rng = random.Random(110707)
    reserved = set()
    for measurements in cells.values():
        eligible = [key for key, groups in sorted(measurements.items()) if previously_tested.isdisjoint(groups)]
        for key in rng.sample(eligible, 4):
            reserved.update(measurements[key])
    return reserved


def make_plan(spec, cells, reserved, seed, *, development):
    rng = random.Random(seed)
    train, probes = defaultdict(list), defaultdict(list)
    reference, assessment = [], []
    for (_, condition), measurements in cells.items():
        available = [key for key, groups in sorted(measurements.items()) if reserved.isdisjoint(groups)]
        rng.shuffle(available)
        for index, key in enumerate(available):
            groups = measurements[key]
            if index < 3:
                probes[condition].extend(groups)
            elif index < 6 and development:
                assessment.extend(groups)
            elif index < (9 if development else 6) and condition == ROUNDS[0]:
                reference.extend(groups)
            else:
                train[condition].extend(groups)
    return BatchPlan(
        dataset_id=spec.dataset_id,
        batches=tuple(Batch(i, tuple(train[c]), c, tuple(probes[c])) for i, c in enumerate(ROUNDS)),
        validation_groups=tuple(reference),
        holdout_groups=tuple(sorted(assessment if development else reserved)),
    )


def sign_flip_tail(differences):
    """One-sided paired sign randomization on integer correct-window deltas."""
    distribution = Counter({0: 1})
    for delta in differences:
        if delta == 0:
            continue
        next_distribution = Counter()
        for total, count in distribution.items():
            next_distribution[total + int(delta)] += count
            next_distribution[total - int(delta)] += count
        distribution = next_distribution
    observed = sum(differences)
    return sum(count for total, count in distribution.items() if total >= observed) / sum(distribution.values())


def comparison(challenger, champion):
    deltas = challenger - champion
    return {
        "gain": float(deltas.mean() / 2),
        "improvement_p": sign_flip_tail(deltas),
        "harm_p": sign_flip_tail(-deltas),
        "measurements": len(deltas),
        "improved": int(np.count_nonzero(deltas > 0)),
        "worsened": int(np.count_nonzero(deltas < 0)),
    }


def fit_rounds(spec, plan, bundle):
    models = []
    records = bundle[0]
    for index in range(plan.batch_count):
        visible = set(plan.history_groups(index))
        mask = np.array([record.group in visible for record in records])
        model, _ = fit_prepared(_prepared(spec, *bundle, {"train": mask}), spec)
        models.append(model)
    return models


def correct_counts(spec, bundle, model, groups):
    records, _, labels, _ = bundle
    mask = np.array([record.group in groups for record in records])
    prepared = _prepared(spec, *bundle, {"assessment": mask})
    _, predicted, _, _ = evaluate_partition(
        prepared, model, "assessment", healthy_class_id=spec.labels.healthy_class_id
    )
    correct = defaultdict(int)
    counts = Counter()
    for record, passed in zip((row for row in records if row.group in groups), predicted == labels[mask]):
        correct[record.group] += int(passed)
        counts[record.group] += 1
    if set(counts) != groups or set(counts.values()) != {2}:
        raise ValueError("assessment requires two windows per measurement")
    return np.array([correct[group] for group in sorted(groups)])


def probe_results(spec, plan, bundle, models):
    results = []
    for index in range(1, len(models)):
        results.append({
            name: [correct_counts(spec, bundle, model, set(groups)) for model in models[:index + 1]]
            for name, groups in plan.visible_probes(index).items()
        })
    return results


def replay(arm, probes, rule):
    incumbent = 0
    steps = [{"batch": 0, "champion": 0, "promoted": True}]
    for index, visible in enumerate(probes, 1):
        current = f"batch-{index}"
        pairs = {name: comparison(scores[index], scores[incumbent]) for name, scores in visible.items()}
        new = pairs[current]
        previous = {name: pair for name, pair in pairs.items() if name != current}
        worst = min(previous, key=lambda name: previous[name]["gain"])
        blocked = []
        if arm == "current_gate":
            decision = decide(
                criterion="guardrail", champion_validation_macro_f1=0, challenger_validation_macro_f1=0,
                margin=0, retention_drop=-previous[worst]["gain"], gain_new=new["gain"],
                guard_tolerance=TOLERANCE,
            )
            promoted = decision.promoted
            blocked = [worst] if -previous[worst]["gain"] > TOLERANCE else []
        elif arm == "paired_gate":
            blocked = [name for name, pair in previous.items()
                       if pair["gain"] < -TOLERANCE and pair["harm_p"] <= rule["harm_alpha"]]
            promoted = not blocked and new["gain"] > 0 and new["improvement_p"] <= rule["gain_alpha"]
        else:
            promoted = arm == "always_publish"
        before = incumbent
        if promoted:
            incumbent = index
        steps.append({"batch": index, "before": before, "champion": incumbent, "promoted": promoted,
                      "blocked_by": blocked, "probes": pairs})
    return steps


def assessment_scores(spec, plan, bundle, models, group_conditions):
    return {
        condition: [correct_counts(spec, bundle, model, {g for g in plan.holdout_groups
                    if group_conditions[g] == condition}) for model in models]
        for condition in ROUNDS
    }


def assess(steps, scores):
    for step in steps:
        index = step["batch"]
        active = ROUNDS[:index + 1]
        step["accuracy_by_condition"] = {
            condition: float(scores[condition][step["champion"]].mean() / 2) for condition in active
        }
        step["mean_accuracy"] = statistics.mean(step["accuracy_by_condition"].values())
        if index:
            step["candidate_gain_by_condition"] = {
                condition: comparison(scores[condition][index], scores[condition][step["before"]])
                for condition in active
            }
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=False)
    spec = load_dataset_spec(ROOT / "configs/paderborn_3class_fusion_time_extratrees.json")
    records = load_dataset_records(spec, DATA)
    historical = {seed: BatchPlan.from_dict(load(PLANS / f"paderborn-plan-{seed}.json"))
                  for seed in HISTORICAL_SEEDS}
    cells = measurement_cells(load(DATA / "manifest.json"))
    conditions = {group: condition for (_, condition), measurements in cells.items()
                  for groups in measurements.values() for group in groups}
    reserved = reserve_holdout(cells, historical)
    development = {seed: make_plan(spec, cells, reserved, seed, development=True) for seed in DEVELOPMENT_SEEDS}
    evaluation = {seed: make_plan(spec, cells, reserved, seed, development=False) for seed in EVALUATION_SEEDS}
    development_records = [row for row in records if row.group not in reserved]
    for plan in development.values():
        plan.validate(development_records)
    for plan in evaluation.values():
        plan.validate(records)
    save(output / "splits.json", {"reserved": sorted(reserved), "development": {s: p.to_dict() for s, p in development.items()},
                                  "evaluation": {s: p.to_dict() for s, p in evaluation.items()}})
    dev_bundle = extract_features(spec, DATA, records=development_records)
    dev_runs = []
    for plan in development.values():
        models = fit_rounds(spec, plan, dev_bundle)
        dev_runs.append((probe_results(spec, plan, dev_bundle, models),
                         assessment_scores(spec, plan, dev_bundle, models, conditions)))
    grid = []
    for harm, gain in product((0.20, 0.10, 0.05), (0.10, 0.20)):
        rule = {"harm_alpha": harm, "gain_alpha": gain}
        runs = [assess(replay("paired_gate", probes, rule), scores) for probes, scores in dev_runs]
        mean = statistics.mean(step["mean_accuracy"] for run in runs for step in run[1:])
        grid.append({**rule, "development_mean_post_initial_accuracy": mean,
                     "promotions": sum(step["promoted"] for run in runs for step in run[1:])})
    rule = max(grid, key=lambda row: (row["development_mean_post_initial_accuracy"], -row["promotions"]))
    save(output / "locked_rule.json", {"selected": rule, "grid": grid, "guard_tolerance": TOLERANCE})
    print(f"Development complete, locked {rule}", flush=True)

    # Fit and lock all policy decisions using only non-reserved measurements.
    locked_runs = {}
    for seed, plan in evaluation.items():
        models = fit_rounds(spec, plan, dev_bundle)
        probes = probe_results(spec, plan, dev_bundle, models)
        locked_runs[seed] = (models, {arm: replay(arm, probes, rule) for arm in ARMS})
    save(output / "locked_decisions.json", {seed: decisions for seed, (_, decisions) in locked_runs.items()})
    holdout_bundle = extract_features(spec, DATA, records=[row for row in records if row.group in reserved])
    results = {}
    for seed, (models, decisions) in locked_runs.items():
        scores = assessment_scores(spec, evaluation[seed], holdout_bundle, models, conditions)
        results[seed] = {arm: assess(steps, scores) for arm, steps in decisions.items()}
    save(output / "holdout.json", results)

    # Historical replays are forensic comparisons on previously inspected holdouts.
    full_bundle = (records, *combine_features(records, dev_bundle, holdout_bundle))
    replays = {}
    for seed in (20261105, 20261106):
        plan = historical[seed]
        models = fit_rounds(spec, plan, full_bundle)
        probes = probe_results(spec, plan, full_bundle, models)
        scores = assessment_scores(spec, plan, full_bundle, models, conditions)
        replays[seed] = {arm: assess(replay(arm, probes, rule), scores) for arm in ARMS}
    save(output / "historical_replays.json", replays)


def combine_features(records, *bundles):
    by_record = {row.record_id: features[i] for rows, features, _, _ in bundles for i, row in enumerate(rows)}
    return np.stack([by_record[row.record_id] for row in records]), np.array([row.label for row in records]), bundles[0][3]


if __name__ == "__main__":
    main()
