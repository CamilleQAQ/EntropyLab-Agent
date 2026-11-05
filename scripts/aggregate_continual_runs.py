#!/usr/bin/env python
"""Aggregate continual-learning runs across independent unit allocations.

Each run is one independent split of units into holdout/validation/rounds, so the
spread across runs is the honest uncertainty of the comparison.  Reports mean and
standard deviation per arm, the paired per-run difference (continual - frozen),
and a pooled paired bootstrap over records plus an exact McNemar test.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _bootstrap(deltas: np.ndarray, iterations: int, seed: int) -> dict:
    generator = np.random.default_rng(seed)
    means = np.empty(iterations)
    n = len(deltas)
    for index in range(iterations):
        sample = generator.integers(0, n, n)
        means[index] = deltas[sample].mean()
    return {
        "mean": float(deltas.mean()),
        "low": float(np.percentile(means, 2.5)),
        "high": float(np.percentile(means, 97.5)),
        "iterations": iterations,
    }


def _mcnemar(frozen_correct: np.ndarray, continual_correct: np.ndarray) -> dict:
    from scipy.stats import binomtest

    only_continual = int(np.sum(continual_correct & ~frozen_correct))
    only_frozen = int(np.sum(frozen_correct & ~continual_correct))
    discordant = only_continual + only_frozen
    p_value = (
        1.0
        if discordant == 0
        else float(binomtest(only_continual, discordant, 0.5).pvalue)
    )
    return {
        "only_continual_correct": only_continual,
        "only_frozen_correct": only_frozen,
        "discordant": discordant,
        "p_value": p_value,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in args.runs]
    arms = ("frozen", "continual", "joint")
    accuracies = {arm: [] for arm in arms}
    f1s = {arm: [] for arm in arms}
    for summary in summaries:
        for arm in arms:
            entry = summary["holdout"].get(arm)
            if entry is None:
                continue
            accuracies[arm].append(entry["accuracy"])
            f1s[arm].append(entry["macro_f1"])

    table = {}
    for arm in arms:
        values = np.asarray(accuracies[arm])
        table[arm] = {
            "runs": len(values),
            "accuracy_mean": float(values.mean()),
            "accuracy_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "macro_f1_mean": float(np.mean(f1s[arm])),
            "per_run_accuracy": [round(float(v), 4) for v in values],
        }

    paired = np.asarray(accuracies["continual"]) - np.asarray(accuracies["frozen"])
    paired_vs_joint = np.asarray(accuracies["continual"]) - np.asarray(accuracies["joint"])

    # Pool per-record outcomes across runs for a paired bootstrap + McNemar.
    deltas = []
    only_cont = 0
    only_frozen = 0
    for summary in summaries:
        frozen = summary["holdout"].get("frozen")
        continual = summary["holdout"].get("continual")
        if not frozen or not continual:
            continue
        by_id = {row["record_id"]: row for row in frozen["predictions"]}
        for row in continual["predictions"]:
            other = by_id[row["record_id"]]
            truth = row["truth"]
            deltas.append(float(int(row["predicted"] == truth) - int(other["predicted"] == truth)))
        frozen_ok = np.asarray([r["predicted"] == r["truth"] for r in frozen["predictions"]])
        continual_ok = np.asarray([r["predicted"] == r["truth"] for r in continual["predictions"]])
        only_cont += int(np.sum(continual_ok & ~frozen_ok))
        only_frozen += int(np.sum(frozen_ok & ~continual_ok))

    deltas_array = np.asarray(deltas)
    from scipy.stats import binomtest

    discordant = only_cont + only_frozen
    mcnemar_p = (
        1.0 if discordant == 0 else float(binomtest(only_cont, discordant, 0.5).pvalue)
    )

    result = {
        "runs": len(summaries),
        "arms": table,
        "paired_continual_minus_frozen": {
            "mean": float(paired.mean()),
            "std": float(paired.std(ddof=1)) if len(paired) > 1 else 0.0,
            "per_run": [round(float(v), 4) for v in paired],
            "wins": int(np.sum(paired > 0)),
            "ties": int(np.sum(paired == 0)),
            "losses": int(np.sum(paired < 0)),
        },
        "paired_continual_minus_joint": {
            "mean": float(paired_vs_joint.mean()),
            "std": float(paired_vs_joint.std(ddof=1)) if len(paired_vs_joint) > 1 else 0.0,
            "per_run": [round(float(v), 4) for v in paired_vs_joint],
        },
        "pooled_records": int(len(deltas_array)),
        "bootstrap_accuracy_delta": _bootstrap(deltas_array, 4000, 20261104),
        "mcnemar": {
            "only_continual_correct": only_cont,
            "only_frozen_correct": only_frozen,
            "discordant": discordant,
            "p_value": mcnemar_p,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"runs: {len(summaries)}")
    for arm in arms:
        entry = table[arm]
        print(
            f'  {arm:9s} accuracy {entry["accuracy_mean"]:.4f} '
            f'± {entry["accuracy_std"]:.4f}   macro-F1 {entry["macro_f1_mean"]:.4f}'
        )
    print(
        f'  paired continual-frozen: {paired.mean():+.4f} ± '
        f'{paired.std(ddof=1) if len(paired) > 1 else 0.0:.4f} '
        f'(wins {int(np.sum(paired > 0))}/{len(paired)})'
    )
    print(
        f'  paired continual-joint:  {paired_vs_joint.mean():+.4f} ± '
        f'{paired_vs_joint.std(ddof=1) if len(paired_vs_joint) > 1 else 0.0:.4f}'
    )
    print(
        f'  pooled bootstrap delta accuracy: {result["bootstrap_accuracy_delta"]["mean"]:+.4f} '
        f'[{result["bootstrap_accuracy_delta"]["low"]:+.4f}, '
        f'{result["bootstrap_accuracy_delta"]["high"]:+.4f}]'
    )
    print(
        f'  McNemar: only-continual {only_cont}, only-frozen {only_frozen}, '
        f'p = {mcnemar_p:.4g}'
    )
    print(f"summary -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
