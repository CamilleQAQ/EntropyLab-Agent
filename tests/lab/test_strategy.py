from __future__ import annotations

import numpy as np
import pytest

from entropy_agent.lab.strategy import (
    CLASSIFIERS,
    FEATURES,
    DatasetProfile,
    aggregate,
    evaluate,
    regret_at_k,
)

CANDIDATES = [
    ("FUSION_TIME", "EXTRA-TREES"),
    ("FUSION_TIME_SPEC", "EXTRA-TREES"),
    ("MDE", "EXTRA-TREES"),
    ("MPE", "EXTRA-TREES"),
]


def _profiles() -> dict[str, DatasetProfile]:
    profiles = {}
    for index in range(6):
        high_rate = index < 3
        profiles[f"d{index}"] = DatasetProfile(
            records=300 + 100 * index,
            classes=4,
            sampling_rate=48000.0 if high_rate else 12000.0,
            window=2048,
            imbalance=1.0,
        )
    return profiles


def _rows() -> list[dict]:
    rows = []
    for index in range(6):
        high_rate = index < 3
        for feature, classifier in CANDIDATES:
            # Rule the learner should discover: high-rate datasets favour MDE,
            # low-rate datasets favour the time-domain fusion.
            if high_rate:
                score = 0.95 if feature == "MDE" else 0.80
            else:
                score = 0.95 if feature == "FUSION_TIME" else 0.82
            rows.append(
                {
                    "dataset": f"d{index}",
                    "feature": feature,
                    "classifier": classifier,
                    "seed": 1,
                    "macro_f1": score,
                    "seconds": 2.0 if feature.startswith("FUSION") else 1.0,
                }
            )
    return rows


def test_aggregate_and_seconds() -> None:
    table = aggregate(_rows())
    assert set(table) == {f"d{i}" for i in range(6)}
    assert table["d0"][("MDE", "EXTRA-TREES")] == 0.95


def test_regret_at_k_counts_only_the_budget() -> None:
    truth = {("FUSION_TIME", "EXTRA-TREES"): 0.8, ("MDE", "EXTRA-TREES"): 0.95}
    order = [("FUSION_TIME", "EXTRA-TREES"), ("MDE", "EXTRA-TREES")]
    assert regret_at_k(order, truth, 1) == pytest.approx(0.15)
    assert regret_at_k(order, truth, 2) == 0.0


def test_learned_ranking_beats_the_fixed_order_across_datasets() -> None:
    rows = _rows()
    table = aggregate(rows)
    profiles = _profiles()
    seconds = {
        (feature, classifier): 2.0 if feature.startswith("FUSION") else 1.0
        for feature, classifier in CANDIDATES
    }
    report = evaluate(table, profiles, seconds, random_trials=50)

    learned = np.mean([entry["regret_learned@1"] for entry in report.values()])
    fixed = np.mean([entry["regret_fixed@1"] for entry in report.values()])
    assert learned < fixed, (learned, fixed)
    # The rule is learnable, so the learned policy should be near-perfect here.
    assert learned < 0.02
