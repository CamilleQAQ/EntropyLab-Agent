from __future__ import annotations

from pathlib import Path

import pytest

from entropy_agent.agent import ClassifierSpec, MethodSpec, build_pool
from entropy_agent.continual import BatchPlan, run_continual
from entropy_agent.continual.search_policy import SearchPolicy, SearchState, should_stop
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


def test_policy_waits_for_enough_candidates() -> None:
    policy = SearchPolicy.adaptive(min_lead=0.02, patience=3, min_candidates=3)
    decision = should_stop(SearchState(1, 0.05, None, 0, 5.0), policy)
    assert decision.stop is False


def test_policy_stops_on_a_clear_lead() -> None:
    policy = SearchPolicy.adaptive(min_lead=0.02, patience=3, min_candidates=3)
    decision = should_stop(SearchState(3, 0.05, 0.0, 0, 2.0), policy)
    assert decision.stop is True
    assert "clear lead" in decision.reason


def test_policy_does_not_stop_without_a_runner_up() -> None:
    policy = SearchPolicy.adaptive(min_lead=0.02, patience=3, min_candidates=3)
    assert should_stop(SearchState(3, 0.05, None, 0, 2.0), policy).stop is False


def test_policy_stops_after_patience() -> None:
    policy = SearchPolicy.adaptive(min_lead=0.0, patience=3, min_candidates=3)
    decision = should_stop(SearchState(4, 0.05, 0.05, 3, 1.0), policy)
    assert decision.stop is True
    assert "no improvement" in decision.reason


def test_policy_stops_when_nothing_is_feasible() -> None:
    policy = SearchPolicy.adaptive(min_lead=0.02, patience=3, min_candidates=3)
    decision = should_stop(SearchState(3, None, None, 3, 1.0), policy)
    assert decision.stop is True
    assert "no feasible" in decision.reason


def test_disabled_policy_never_stops() -> None:
    state = SearchState(100, 0.5, 0.0, 50, 0.0)
    assert should_stop(state, SearchPolicy()).stop is False


def test_adaptive_search_evaluates_fewer_candidates(tiny_dataset, tmp_path: Path) -> None:
    spec, root, plan = _plan_with_probes(tiny_dataset)
    common = dict(
        model_prefix="tiny",
        criterion="guardrail",
        guard_tolerance=1.0,
        min_gain=-1.0,
        select=True,
        selection_budget=1000.0,
        pool=_pool(),
    )
    full = run_continual(spec, root, plan, tmp_path / "full", search_policy=None, **common)
    adaptive = run_continual(
        spec,
        root,
        plan,
        tmp_path / "adaptive",
        search_policy=SearchPolicy.adaptive(min_lead=0.0, patience=1, min_candidates=3),
        **common,
    )
    full_evaluated = full["decisions"][1]["probe_scores"]["evaluated"]
    adaptive_evaluated = adaptive["decisions"][1]["probe_scores"]["evaluated"]
    assert adaptive_evaluated < full_evaluated
    assert adaptive["decisions"][1]["probe_scores"]["stop_reason"]
    assert adaptive["selection"]["search_policy"]["enabled"] is True
    assert full["decisions"][1]["probe_scores"]["stop_reason"] == "budget exhausted"
