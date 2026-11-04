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
    assert batch[0].candidate.candidate_id == anchor.candidate_id
    assert batch[0].reason == "anchor combination"
    assert len({proposal.candidate.family for proposal in batch}) == 3


def test_proposals_are_deterministic_and_never_repeat() -> None:
    pool = _pool()
    history = _history([(pool[0], 0.5)])
    first = propose(2, history, pool, per_round=2)
    second = propose(2, history, pool, per_round=2)

    assert [proposal.candidate.candidate_id for proposal in first] == [
        proposal.candidate.candidate_id for proposal in second
    ]
    assert all(
        proposal.candidate.candidate_id != pool[0].candidate_id for proposal in first
    )
    assert all(proposal.reason for proposal in first)


def test_exploitation_prefers_methods_whose_features_are_cached() -> None:
    pool = _pool()
    # MDE is the best method and its features are already extracted; MPE scores
    # lower but would need a new extraction.
    best = next(
        candidate
        for candidate in pool
        if candidate.method.method == "MDE"
        and candidate.classifier.algorithm == "SVM-RBF"
    )
    history = _history([(best, 0.9)])
    batch = propose(2, history, pool, per_round=2, extracted_methods={"MDE"})

    assert batch[0].candidate.method.method == "MDE"
    assert "free features" in batch[0].reason


def test_untried_classifier_beats_retrying_a_known_one() -> None:
    pool = _pool()
    best = next(
        candidate
        for candidate in pool
        if candidate.method.method == "MDE"
        and candidate.classifier.algorithm == "SVM-RBF"
    )
    history = _history([(best, 0.9)])
    batch = propose(3, history, pool, per_round=1, extracted_methods={"MDE"})

    # MDE + SVM-RBF already scored 0.9; a different MDE classifier has never run
    # and costs nothing, so it must be preferred over repeating the known pair.
    assert batch[0].candidate.method.method == "MDE"
    assert batch[0].candidate.candidate_id != best.candidate_id


def test_untried_family_is_reserved_when_exploiting() -> None:
    pool = _pool()
    mde = next(candidate for candidate in pool if candidate.method.method == "MDE")
    mpe = next(candidate for candidate in pool if candidate.method.method == "MPE")
    history = _history([(mde, 0.9), (mpe, 0.8)])
    batch = propose(3, history, pool, per_round=3, extracted_methods={"MDE", "MPE"})

    families = {proposal.candidate.family for proposal in batch}
    assert "time_shift_ensemble" in families


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
