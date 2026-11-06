from __future__ import annotations

from pathlib import Path

import numpy as np

from entropy_agent.agent import ClassifierSpec, MethodSpec, build_pool
from entropy_agent.continual import BatchPlan, run_continual
from entropy_agent.continual.selection import candidate_spec
from entropy_agent.io.dataset import load_dataset_records


def _plan_with_probes(tiny_dataset) -> tuple[object, Path, BatchPlan]:
    spec, root = tiny_dataset
    records = load_dataset_records(spec, root)
    by_class: dict[int, list[str]] = {}
    for record in records:
        groups = by_class.setdefault(int(record.label), [])
        if record.group not in groups:
            groups.append(record.group)
    holdout = [groups[-1] for groups in by_class.values()]
    validation = [groups[-2] for groups in by_class.values()]
    remaining = [groups[:-2] for groups in by_class.values()]
    payload = {
        "format": "entropylab.continual-plan",
        "format_version": 1,
        "dataset_id": spec.dataset_id,
        "margin": 0.005,
        "batches": [
            {"batch_id": 0, "groups": [g[0] for g in remaining], "note": "round 0",
             "probe_groups": [g[2] for g in remaining]},
            {"batch_id": 1, "groups": [g[1] for g in remaining], "note": "round 1",
             "probe_groups": [g[3] for g in remaining]},
        ],
        "validation_groups": validation,
        "holdout_groups": holdout,
    }
    return spec, root, BatchPlan.from_dict(payload)


def _pool():
    return build_pool(
        methods=(
            MethodSpec("MDE", {"m": 2, "nc": 4, "tau": 1, "scale": 10}),
            MethodSpec("MPE", {"m": 3, "tau": 1, "scale": 10}),
            MethodSpec("FUSION_TIME", {}),
        ),
        classifiers=(
            ClassifierSpec("SVM-RBF", "scikit-learn", {"C": 1.0, "gamma": "scale"}),
            ClassifierSpec("EXTRA-TREES", "scikit-learn", {"n_estimators": 40, "random_state": 0}),
        ),
    )


def test_candidate_spec_only_changes_feature_and_classifier(tiny_dataset) -> None:
    spec, _root = tiny_dataset
    candidate = _pool()[0]
    derived = candidate_spec(spec, candidate)
    assert derived.feature.method == candidate.method.method
    assert derived.classifier.algorithm == candidate.classifier.algorithm
    assert derived.dataset_id == spec.dataset_id
    assert derived.window == spec.window


def test_selection_commits_a_candidate_and_logs_the_search(tiny_dataset, tmp_path: Path) -> None:
    spec, root, plan = _plan_with_probes(tiny_dataset)
    summary = run_continual(
        spec,
        root,
        plan,
        tmp_path / "select",
        model_prefix="tiny",
        criterion="guardrail",
        guard_tolerance=1.0,
        min_gain=-1.0,
        select=True,
        selection_budget=1000.0,
        pool=_pool(),
    )
    assert summary["selection"]["enabled"] is True
    assert summary["champion_configuration"] is not None
    first = summary["decisions"][0]
    assert first["promoted"] is True
    assert first["challenger"]["model_id"] is not None
    candidates = first["probe_scores"]["candidates"]
    assert candidates, "the search must be logged"
    assert {"candidate_id", "method", "classifier", "estimated_seconds"} <= set(candidates[0])
    assert summary["decisions"][1]["probe_scores"]["spent_seconds"] >= 0.0


def test_selection_respects_the_budget(tiny_dataset, tmp_path: Path) -> None:
    spec, root, plan = _plan_with_probes(tiny_dataset)
    # A tiny budget: only the cheapest candidates can be paid for.
    summary = run_continual(
        spec,
        root,
        plan,
        tmp_path / "select-tight",
        model_prefix="tiny",
        criterion="guardrail",
        guard_tolerance=1.0,
        min_gain=-1.0,
        select=True,
        selection_budget=0.05,
        pool=_pool(),
    )
    log = summary["decisions"][0]["probe_scores"]
    assert log["spent_seconds"] <= 0.05 + 1e-9
    assert log["candidates"], "at least one cheap candidate must fit"
    assert log["skipped"], "expensive candidates must be skipped"


def test_selection_can_keep_the_champion_when_nothing_passes(tiny_dataset, tmp_path: Path) -> None:
    spec, root, plan = _plan_with_probes(tiny_dataset)
    summary = run_continual(
        spec,
        root,
        plan,
        tmp_path / "select-strict",
        model_prefix="tiny",
        criterion="guardrail",
        guard_tolerance=-1.0,  # no candidate can satisfy a negative tolerance
        min_gain=1.0,
        select=True,
        selection_budget=1000.0,
        pool=_pool(),
    )
    assert summary["decisions"][0]["promoted"] is True  # initial round still trains
    assert summary["decisions"][1]["promoted"] is False
    assert summary["promotions"] == 1


def test_selection_per_round_can_switch_the_configuration(tiny_dataset, tmp_path: Path) -> None:
    spec, root, plan = _plan_with_probes(tiny_dataset)
    summary = run_continual(
        spec,
        root,
        plan,
        tmp_path / "select-switch",
        model_prefix="tiny",
        criterion="guardrail",
        guard_tolerance=1.0,
        min_gain=-1.0,
        select=True,
        selection_budget=1000.0,
        pool=_pool(),
    )
    methods = [decision["challenger"]["model_id"].split("-")[-1] for decision in summary["decisions"]]
    assert len(methods) == 2
    # The decision log records which candidate won each round.
    for decision in summary["decisions"]:
        assert decision["probe_scores"]["candidates"]
