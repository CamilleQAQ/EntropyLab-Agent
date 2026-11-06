#!/usr/bin/env python
"""Sensitivity sweep for the guardrail criterion on Paderborn.

For every (guard_tolerance, min_gain) pair and every plan seed, run the continual
loop and record the frozen/continual/joint holdout accuracy plus how many updates
were accepted.  Features are extracted once and reused, so the sweep is cheap.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from entropy_agent.continual import BatchPlan, run_continual  # noqa: E402
from entropy_agent.io.dataset import load_dataset_records, load_dataset_spec  # noqa: E402
from entropy_agent.training.pipeline import extract_features  # noqa: E402

SEEDS = (20261104, 1, 2, 3, 4)
TOLERANCES = (0.0, 0.01, 0.02, 0.05, 0.10)
MIN_GAINS = (0.0, 0.01, 0.03)


def main() -> int:
    spec = load_dataset_spec(ROOT / "configs/paderborn_3class_fusion_time_extratrees.json")
    data_root = ROOT / "data/paper/paderborn_3class"
    records = load_dataset_records(spec, data_root)
    records, features, labels, dimension = extract_features(spec, data_root, records=records)
    bundle = (records, features, labels, dimension)
    scratch = ROOT / "tmp/sensitivity"
    shutil.rmtree(scratch, ignore_errors=True)

    results = []
    for tolerance in TOLERANCES:
        for min_gain in MIN_GAINS:
            scores = {"frozen": [], "continual": [], "joint": []}
            commits = []
            for seed in SEEDS:
                plan = BatchPlan.load(data_root / f"continual_plan_seed{seed}.json")
                output = scratch / f"gt{tolerance}-mg{min_gain}-seed{seed}"
                summary = run_continual(
                    spec,
                    data_root,
                    plan,
                    output,
                    model_prefix=f"sens-gt{tolerance}-mg{min_gain}-s{seed}",
                    criterion="guardrail",
                    guard_tolerance=tolerance,
                    min_gain=min_gain,
                    features_bundle=bundle,
                )
                holdout = summary["holdout"]
                for arm in scores:
                    scores[arm].append(holdout[arm]["accuracy"])
                commits.append(sum(1 for d in summary["decisions"] if d["promoted"]))
                shutil.rmtree(output, ignore_errors=True)
            entry = {
                "guard_tolerance": tolerance,
                "min_gain": min_gain,
                "frozen_mean": float(np.mean(scores["frozen"])),
                "continual_mean": float(np.mean(scores["continual"])),
                "continual_std": float(np.std(scores["continual"], ddof=1)),
                "joint_mean": float(np.mean(scores["joint"])),
                "gain_over_frozen": float(
                    np.mean(np.subtract(scores["continual"], scores["frozen"]))
                ),
                "per_seed": [round(float(v), 4) for v in scores["continual"]],
                "commits_per_seed": commits,
            }
            results.append(entry)
            print(
                f"tolerance={tolerance:.2f} min_gain={min_gain:.2f}: "
                f"continual {entry['continual_mean']:.4f} ± {entry['continual_std']:.4f}, "
                f"gain {entry['gain_over_frozen']:+.4f}, commits {sum(commits)}/10"
            )

    output_path = ROOT / "results/continual-paderborn-sensitivity-20261106/summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "dataset": "PADERBORN_3CLASS",
                "seeds": list(SEEDS),
                "tolerances": list(TOLERANCES),
                "min_gains": list(MIN_GAINS),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"summary -> {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
