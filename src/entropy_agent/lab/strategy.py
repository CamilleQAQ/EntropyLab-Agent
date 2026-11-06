"""Cross-dataset strategy learning: which candidate to try first.

The grid script records ``(dataset, feature, classifier) -> macro-F1``.  This
module turns that into a ranking policy:

* dataset features (records, classes, sampling rate, window, imbalance);
* candidate features (feature one-hot, classifier one-hot, extraction seconds);
* a small ridge model predicts macro-F1, and candidates are ranked by prediction.

Evaluation is **leave-one-dataset-out**: the policy is fit on the other datasets
and scored on the held-out one, then compared with the fixed order and with
random guessing at the same trial budget.
"""

from __future__ import annotations

import csv
import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FEATURES = ("FUSION_TIME", "FUSION_TIME_SPEC", "FUSION_TIME_ENVELOPE", "MDE", "MPE", "RCMDE")
CLASSIFIERS = ("EXTRA-TREES", "SVM-RBF", "RANDOM-FOREST")

# The order the project used before any learning: anchor first, then the rest.
FIXED_ORDER = [
    ("FUSION_TIME", "EXTRA-TREES"),
    ("FUSION_TIME", "SVM-RBF"),
    ("FUSION_TIME", "RANDOM-FOREST"),
    ("FUSION_TIME_SPEC", "EXTRA-TREES"),
    ("FUSION_TIME_SPEC", "SVM-RBF"),
    ("FUSION_TIME_SPEC", "RANDOM-FOREST"),
    ("FUSION_TIME_ENVELOPE", "EXTRA-TREES"),
    ("FUSION_TIME_ENVELOPE", "SVM-RBF"),
    ("FUSION_TIME_ENVELOPE", "RANDOM-FOREST"),
    ("MDE", "EXTRA-TREES"),
    ("MDE", "SVM-RBF"),
    ("MDE", "RANDOM-FOREST"),
    ("MPE", "EXTRA-TREES"),
    ("MPE", "SVM-RBF"),
    ("MPE", "RANDOM-FOREST"),
    ("RCMDE", "EXTRA-TREES"),
    ("RCMDE", "SVM-RBF"),
    ("RCMDE", "RANDOM-FOREST"),
]


@dataclass(frozen=True)
class DatasetProfile:
    records: int
    classes: int
    sampling_rate: float
    window: int
    imbalance: float


def load_grid(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["macro_f1"] = float(row["macro_f1"])
        row["seconds"] = float(row["seconds"])
    return rows


def aggregate(rows: list[dict]) -> dict[str, dict[tuple[str, str], float]]:
    """dataset -> (feature, classifier) -> mean macro-F1."""
    grouped: dict[str, dict[tuple[str, str], list[float]]] = {}
    for row in rows:
        key = (row["feature"], row["classifier"])
        grouped.setdefault(row["dataset"], {}).setdefault(key, []).append(row["macro_f1"])
    return {
        dataset: {key: float(np.mean(values)) for key, values in table.items()}
        for dataset, table in grouped.items()
    }


def seconds_of(rows: list[dict]) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((row["feature"], row["classifier"]), []).append(row["seconds"])
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def build_vectors(
    profiles: dict[str, DatasetProfile],
    seconds: dict[tuple[str, str], float],
    dataset: str,
    candidates: list[tuple[str, str]],
) -> np.ndarray:
    profile = profiles[dataset]
    rows = []
    for feature, classifier in candidates:
        numeric = [
            math.log1p(profile.records),
            profile.classes,
            math.log1p(profile.sampling_rate),
            math.log1p(profile.window),
            profile.imbalance,
        ]
        feature_hot = [1.0 if feature == name else 0.0 for name in FEATURES]
        classifier_hot = [1.0 if classifier == name else 0.0 for name in CLASSIFIERS]
        # Interactions let a linear model learn rules such as
        # "high sampling rate favours this feature family".
        interactions = [
            value * flag for value in numeric for flag in feature_hot + classifier_hot
        ]
        rows.append(
            [
                *numeric,
                *feature_hot,
                *classifier_hot,
                math.log1p(seconds.get((feature, classifier), 1.0)),
                *interactions,
            ]
        )
    return np.asarray(rows, dtype=float)


def fit_predict(
    table: dict[str, dict[tuple[str, str], float]],
    profiles: dict[str, DatasetProfile],
    seconds: dict[tuple[str, str], float],
    holdout: str,
) -> dict[tuple[str, str], float]:
    from sklearn.linear_model import Ridge

    train_datasets = [name for name in table if name != holdout]
    xs, ys = [], []
    for name in train_datasets:
        candidates = list(table[name])
        xs.append(build_vectors(profiles, seconds, name, candidates))
        values = np.asarray([table[name][candidate] for candidate in candidates])
        # Normalise inside each dataset: the model learns which candidate is
        # relatively better, not how hard the dataset is.
        spread = values.std(ddof=0) or 1.0
        ys.extend((values - values.mean()) / spread)
    x_train = np.vstack(xs)
    y_train = np.asarray(ys)
    model = Ridge(alpha=1.0)
    model.fit(x_train, y_train)
    candidates = list(table[holdout])
    scores = model.predict(build_vectors(profiles, seconds, holdout, candidates))
    return {candidate: float(score) for candidate, score in zip(candidates, scores)}


def regret_at_k(
    order: list[tuple[str, str]], truth: dict[tuple[str, str], float], k: int
) -> float:
    best = max(truth.values())
    achieved = max(truth[candidate] for candidate in order[:k])
    return float(best - achieved)


def evaluate(
    table: dict[str, dict[tuple[str, str], float]],
    profiles: dict[str, DatasetProfile],
    seconds: dict[tuple[str, str], float],
    *,
    ks: tuple[int, ...] = (1, 2, 3),
    random_trials: int = 500,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    report: dict[str, dict] = {}
    for holdout in table:
        predicted = fit_predict(table, profiles, seconds, holdout)
        learned = sorted(predicted, key=lambda candidate: predicted[candidate], reverse=True)
        truth = table[holdout]
        available = [candidate for candidate in FIXED_ORDER if candidate in truth]
        fixed = available + [candidate for candidate in truth if candidate not in available]
        random_regret = {k: [] for k in ks}
        for _ in range(random_trials):
            shuffled = [truth[candidate] for candidate in truth]
            order = list(rng.permutation(len(shuffled)))
            for k in ks:
                top = max(shuffled[index] for index in order[:k])
                random_regret[k].append(float(max(truth.values()) - top))
        entry = {
            "best_possible": round(max(truth.values()), 4),
            "fixed@1": round(truth[fixed[0]], 4),
            "learned@1": round(truth[learned[0]], 4),
            "random@1": round(float(np.mean(random_regret[1])), 4),
            "learned_top1": learned[0],
        }
        for k in ks:
            entry[f"regret_learned@{k}"] = round(regret_at_k(learned, truth, k), 4)
            entry[f"regret_fixed@{k}"] = round(regret_at_k(fixed, truth, k), 4)
            entry[f"regret_random@{k}"] = round(float(np.mean(random_regret[k])), 4)
        report[holdout] = entry
    return report
