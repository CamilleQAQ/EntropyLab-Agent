"""Per-round candidate selection for continual learning (V2 step 2).

The Agent's action at each round is no longer "retrain the fixed configuration":
it may train any candidate from the pool (feature family x classifier) and commit
the best one that passes the guardrail.  Candidates are paid for with the same
seconds-based cost model used by the feature-search Agent (``agent/cost.py``): a
candidate costs its estimated feature-extraction seconds, and the round stops when
the budget cannot afford the next candidate.

Feature spaces differ per candidate, so features are extracted per
(feature method, parameters) and cached for the whole run.  Training wall-clock is
reported but not charged; only extraction is priced, which matches the semantics
of ``--budget-seconds`` elsewhere in the project.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..agent.candidates import Candidate
from ..agent.cost import BudgetLedger, CostModel
from ..io.dataset import DatasetSpec
from ..training.pipeline import (
    PreparedDataset,
    evaluate_partition,
    extract_features,
    train_dataset,
)
from .decision import decide
from .search_policy import SearchPolicy, SearchState, should_stop


@dataclass
class SelectionOutcome:
    candidate_id: str
    method: str
    classifier: str
    estimated_seconds: float
    feasible: bool
    gain_new: float | None = None
    retention_drop: float | None = None
    validation_macro_f1: float | None = None
    reason: str = ""


@dataclass
class SelectionResult:
    spec: DatasetSpec | None = None
    model: object | None = None
    model_directory: Path | None = None
    model_id: str | None = None
    validation_macro_f1: float | None = None
    validation_accuracy: float | None = None
    probe_scores: dict[str, float] = field(default_factory=dict)
    outcomes: list[SelectionOutcome] = field(default_factory=list)
    selected: SelectionOutcome | None = None
    skipped: list[dict] = field(default_factory=list)
    spent_seconds: float = 0.0
    evaluated: int = 0
    stop_reason: str = "budget exhausted"


def candidate_spec(spec: DatasetSpec, candidate: Candidate) -> DatasetSpec:
    """A DatasetSpec that differs only in feature method/parameters and classifier."""
    return spec.model_copy(
        update={
            "feature": spec.feature.model_copy(
                update={
                    "method": candidate.method.method,
                    "parameters": dict(candidate.method.parameters),
                }
            ),
            "classifier": spec.classifier.model_copy(
                update={
                    "algorithm": candidate.classifier.algorithm,
                    "backend": candidate.classifier.backend,
                    "parameters": dict(candidate.classifier.parameters),
                }
            ),
        }
    )


def _feature_key(spec: DatasetSpec) -> str:
    return spec.feature.method + "|" + json.dumps(spec.feature.parameters, sort_keys=True)


def features_for(
    spec: DatasetSpec,
    data_root: Path,
    records,
    cache: dict[str, tuple[np.ndarray, np.ndarray, int]],
) -> tuple[np.ndarray, np.ndarray, int]:
    key = _feature_key(spec)
    if key not in cache:
        _, features, labels, dimension = extract_features(spec, data_root, records=records)
        cache[key] = (features, np.asarray(labels, dtype=np.int64), int(dimension))
    return cache[key]


def _prepared(
    spec: DatasetSpec,
    records,
    features: np.ndarray,
    labels: np.ndarray,
    dimension: int,
    masks: dict[str, np.ndarray],
) -> PreparedDataset:
    split_by_group: dict[str, str] = {}
    for name, mask in masks.items():
        for index in np.flatnonzero(mask):
            split_by_group.setdefault(records[int(index)].group, name)
    return PreparedDataset(
        spec=spec,
        records=list(records),
        features=features,
        labels=labels,
        groups=tuple(record.group for record in records),
        feature_dimension=dimension,
        split_by_group=split_by_group,
        masks=masks,
        feature_seconds=0.0,
    )


def probe_scores(
    spec: DatasetSpec,
    model: object,
    records,
    features: np.ndarray,
    labels: np.ndarray,
    dimension: int,
    masks: dict[str, np.ndarray],
    probe_names: tuple[str, ...],
    healthy: int | None,
) -> dict[str, float]:
    prepared = _prepared(spec, records, features, labels, dimension, masks)
    return {
        name: float(
            evaluate_partition(prepared, model, name, healthy_class_id=healthy)[0][
                "multiclass_accuracy"
            ]
        )
        for name in probe_names
    }


def select_and_train(
    *,
    spec: DatasetSpec,
    data_root: Path,
    records,
    labels: np.ndarray,
    masks: dict[str, np.ndarray],
    probe_names: tuple[str, ...],
    champion,
    pool,
    budget_seconds: float,
    cost_model: CostModel,
    windows: int,
    model_directory: Path,
    model_prefix: str,
    batch_index: int,
    healthy: int | None,
    guard_tolerance: float,
    min_gain: float,
    margin: float,
    feature_cache: dict,
    policy: SearchPolicy | None = None,
    method_memory: dict[str, float] | None = None,
) -> SelectionResult:
    """Try candidates cheapest-first until the budget runs out; return the best commit.

    ``champion is None`` means this is the initial round: candidates are ranked by
    validation macro-F1 and the best one becomes the champion.  Otherwise a
    candidate must pass the guardrail, and the feasible one with the largest
    new-batch probe gain wins (ties broken by validation, then by lower cost).
    """
    ledger = BudgetLedger(budget_seconds)
    result = SelectionResult()

    memory = method_memory if method_memory is not None else {}
    # Methods that paid off in earlier rounds are tried first; the rest stay in
    # cost order so the cheap exploration still happens.
    ordered = sorted(
        enumerate(pool),
        key=lambda item: (
            -memory.get(item[1].method.method, 0.0),
            cost_model.estimate(item[1].method.method, windows),
            item[0],
        ),
    )
    costs = [cost_model.estimate(candidate.method.method, windows) for _i, candidate in ordered]

    champion_scores = None
    if champion is not None:
        champion_features, champion_labels, champion_dimension = features_for(
            champion.spec, data_root, records, feature_cache
        )
        champion_scores = probe_scores(
            champion.spec,
            champion.model,
            records,
            champion_features,
            champion_labels,
            champion_dimension,
            masks,
            probe_names,
            healthy,
        )

    evaluated: list[dict] = []
    gains: list[float] = []
    no_improve = 0
    search_policy = policy if policy is not None else SearchPolicy()
    for position, (_index, candidate) in enumerate(ordered):
        spec_c = candidate_spec(spec, candidate)
        # Feature extraction is charged once per (method, parameters); a second
        # classifier on an already-extracted feature space is free.
        cached = _feature_key(spec_c) in feature_cache
        estimate = 0.0 if cached else costs[position]
        if not ledger.can_afford(estimate):
            ledger.skip(candidate.method.method, estimate, "does not fit the remaining budget")
            result.skipped.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "estimated_seconds": round(estimate, 3),
                    "reason": "does not fit the remaining budget",
                }
            )
            continue

        features, labels_c, dimension = features_for(spec_c, data_root, records, feature_cache)
        prepared = _prepared(spec_c, records, features, labels_c, dimension, masks)
        model_dir = model_directory / f"batch-{batch_index}" / candidate.candidate_id
        model_id = f"{model_prefix}-{spec.dataset_id.lower()}-b{batch_index}-{candidate.candidate_id}"
        train_result = train_dataset(
            spec_c,
            Path(data_root),
            model_dir,
            model_id=model_id,
            prepared=prepared,
            evaluate_test=False,
        )
        model = train_result.artifact.model
        validation = train_result.test_result["validation"]
        scores = probe_scores(
            spec_c, model, records, features, labels_c, dimension, masks, probe_names, healthy
        )
        ledger.charge(candidate.method.method, estimate)

        entry = {
            "candidate": candidate,
            "spec": spec_c,
            "model": model,
            "directory": model_dir,
            "model_id": model_id,
            "validation_macro_f1": float(validation["multiclass_macro_f1"]),
            "validation_accuracy": float(validation["multiclass_accuracy"]),
            "scores": scores,
            "estimate": estimate,
        }

        if champion is None:
            outcome = SelectionOutcome(
                candidate_id=candidate.candidate_id,
                method=candidate.method.method,
                classifier=candidate.classifier.algorithm,
                estimated_seconds=estimate,
                feasible=True,
                validation_macro_f1=entry["validation_macro_f1"],
                reason="initial round: candidates ranked by validation",
            )
        else:
            current = f"batch-{batch_index}"
            gain_new = (
                scores[current] - champion_scores[current]
                if current in scores
                else entry["validation_macro_f1"] - champion.validation_macro_f1
            )
            previous = [name for name in probe_names if name != current]
            retention_drop = (
                max(champion_scores[name] - scores[name] for name in previous)
                if previous
                else 0.0
            )
            decision = decide(
                criterion="guardrail",
                champion_validation_macro_f1=champion.validation_macro_f1,
                challenger_validation_macro_f1=entry["validation_macro_f1"],
                margin=margin,
                retention_drop=retention_drop,
                gain_new=gain_new,
                guard_tolerance=guard_tolerance,
                min_gain=min_gain,
            )
            entry["gain_new"] = float(gain_new)
            entry["retention_drop"] = float(retention_drop)
            entry["feasible"] = decision.promoted
            outcome = SelectionOutcome(
                candidate_id=candidate.candidate_id,
                method=candidate.method.method,
                classifier=candidate.classifier.algorithm,
                estimated_seconds=estimate,
                feasible=decision.promoted,
                gain_new=float(gain_new),
                retention_drop=float(retention_drop),
                validation_macro_f1=entry["validation_macro_f1"],
                reason=decision.reason,
            )
        result.outcomes.append(outcome)
        entry["outcome"] = outcome
        evaluated.append(entry)

        if champion is not None and entry.get("feasible") is True:
            gain = entry["gain_new"]
            if not gains or gain > max(gains):
                no_improve = 0
            else:
                no_improve += 1
            gains.append(float(gain))
        elif champion is not None:
            no_improve += 1

        if champion is not None and search_policy.enabled:
            ordered_gains = sorted(gains, reverse=True)
            best_gain = ordered_gains[0] if ordered_gains else None
            runner_up = ordered_gains[1] if len(ordered_gains) > 1 else None
            decision = should_stop(
                SearchState(
                    evaluated=len(evaluated),
                    best_gain=best_gain,
                    runner_up_gain=runner_up,
                    no_improve=no_improve,
                    remaining_seconds=ledger.remaining,
                ),
                search_policy,
            )
            if decision.stop:
                result.stop_reason = decision.reason
                break

        remaining = []
        for rest_position in range(position + 1, len(ordered)):
            rest = ordered[rest_position][1]
            rest_cached = _feature_key(candidate_spec(spec, rest)) in feature_cache
            remaining.append((rest, 0.0 if rest_cached else costs[rest_position]))
        if not any(ledger.can_afford(cost) for _rest, cost in remaining):
            for rest, cost in remaining:
                result.skipped.append(
                    {
                        "candidate_id": rest.candidate_id,
                        "estimated_seconds": round(cost, 3),
                        "reason": "budget exhausted before reaching this candidate",
                    }
                )
            break

    if not evaluated:
        result.spent_seconds = ledger.spent_seconds
        return result

    if champion is None:
        best = max(
            evaluated,
            key=lambda item: (item["validation_macro_f1"], -item["estimate"]),
        )
    else:
        feasible = [
            item
            for item in evaluated
            if item.get("feasible") is True and item.get("gain_new") is not None
        ]
        if not feasible:
            result.spent_seconds = ledger.spent_seconds
            return result
        best = max(
            feasible,
            key=lambda item: (
                item["gain_new"],
                item["validation_macro_f1"],
                -item["estimate"],
            ),
        )

    result.spec = best["spec"]
    result.model = best["model"]
    result.model_directory = best["directory"]
    result.model_id = best["model_id"]
    result.validation_macro_f1 = best["validation_macro_f1"]
    result.validation_accuracy = best["validation_accuracy"]
    result.probe_scores = best["scores"]
    result.selected = best.get("outcome")
    result.spent_seconds = ledger.spent_seconds
    result.evaluated = len(evaluated)
    if champion is None:
        # The initial round has no champion, so there is no gain to remember.
        # Rank methods by how far they fell short of the round's best validation
        # score (<= 0), which makes the best method sort first next round.
        scored = [
            outcome.validation_macro_f1
            for outcome in result.outcomes
            if outcome.validation_macro_f1 is not None
        ]
        best_validation = max(scored) if scored else None
        if best_validation is not None:
            for outcome in result.outcomes:
                if outcome.validation_macro_f1 is None:
                    continue
                score = float(outcome.validation_macro_f1) - float(best_validation)
                previous = memory.get(outcome.method, -1.0)
                memory[outcome.method] = max(previous, score)
    else:
        for outcome in result.outcomes:
            if outcome.gain_new is None:
                continue
            previous = memory.get(outcome.method, 0.0)
            memory[outcome.method] = max(previous, float(outcome.gain_new))
    return result
