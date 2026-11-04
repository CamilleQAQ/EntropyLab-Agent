from __future__ import annotations

from entropy_agent.agent import ClassifierSpec, MethodSpec, build_pool, propose
from entropy_agent.agent.candidates import Candidate


def _pool() -> tuple[Candidate, ...]:
    return build_pool(
        methods=(
            MethodSpec("MDE", {"m": 2, "nc": 4, "tau": 1, "scale": 10}),
            MethodSpec("MPE", {"m": 3, "tau": 1, "scale": 10}),
            MethodSpec("TSMEFuDE", {"m": 2, "nc": 4, "tau": 1, "kmax": 10}),
        ),
        classifiers=(
            ClassifierSpec("SVM-RBF", "scikit-learn", {"C": 1.0, "gamma": "scale"}),
            ClassifierSpec("EXTRA-TREES", "scikit-learn", {"n_estimators": 40, "random_state": 0}),
            ClassifierSpec("BT-LSTSVM", "torch", {"c1": 1.0, "c2": 1.0, "bandwidth": 1.0, "epsilon": 1e-4}),
        ),
    )


def _history(entries: list[tuple[Candidate, float]]) -> list[dict]:
    return [
        {
            "candidate_id": candidate.candidate_id,
            "method": candidate.method.method,
            "family": candidate.family,
            "classifier": candidate.classifier.algorithm,
            "status": "OK",
            "val_accuracy": score,
            "val_macro_f1": score,
        }
        for candidate, score in entries
    ]


def test_round_zero_starts_from_anchor_and_covers_families() -> None:
    pool = _pool()
    anchor = next(
        candidate
        for candidate in pool
        if candidate.method.method == "TSMEFuDE"
        and candidate.classifier.algorithm == "BT-LSTSVM"
    )
    batch = propose(0, [], pool, per_round=3, anchor_id=anchor.candidate_id)

    assert len(batch) == 3
    assert batch[0].candidate_id == anchor.candidate_id
    assert len({candidate.family for candidate in batch}) == 3


def test_proposals_are_deterministic_and_never_repeat() -> None:
    pool = _pool()
    history = _history([(pool[0], 0.5)])
    first = propose(2, history, pool, per_round=2)
    second = propose(2, history, pool, per_round=2)

    assert [candidate.candidate_id for candidate in first] == [
        candidate.candidate_id for candidate in second
    ]
    assert all(
        candidate.candidate_id != pool[0].candidate_id for candidate in first
    )


def test_later_rounds_exploit_the_best_method() -> None:
    pool = _pool()
    best = next(
        candidate
        for candidate in pool
        if candidate.method.method == "MDE"
        and candidate.classifier.algorithm == "SVM-RBF"
    )
    history = _history([(best, 0.9), (pool[-1], 0.2)])
    batch = propose(3, history, pool, per_round=2)

    assert len(batch) == 2
    assert any(candidate.method.method == "MDE" for candidate in batch)


def test_exhausted_pool_returns_nothing() -> None:
    pool = _pool()
    history = _history([(candidate, 0.5) for candidate in pool])

    assert propose(4, history, pool, per_round=3) == ()


def test_per_round_must_be_positive() -> None:
    try:
        propose(0, [], _pool(), per_round=0)
    except ValueError as error:
        assert "per_round" in str(error)
    else:  # pragma: no cover - the raise above is expected
        raise AssertionError("per_round=0 must be rejected")
