from __future__ import annotations

from pathlib import Path

from entropy_agent.agent import (
    Candidate,
    ClassifierSpec,
    FeatureCache,
    MethodSpec,
    build_pool,
)
from entropy_agent.io.dataset import load_dataset_records


def _pool() -> tuple[Candidate, ...]:
    return build_pool(
        methods=(
            MethodSpec("MDE", {"m": 2, "nc": 4, "tau": 1, "scale": 10}),
            MethodSpec("MPE", {"m": 3, "tau": 1, "scale": 10}),
        ),
        classifiers=(
            ClassifierSpec("SVM-RBF", "scikit-learn", {"C": 1.0, "gamma": "scale"}),
            ClassifierSpec("EXTRA-TREES", "scikit-learn", {"n_estimators": 40, "random_state": 0}),
        ),
    )


def test_pool_is_method_major_and_ids_are_stable() -> None:
    pool = _pool()
    assert [candidate.method.method for candidate in pool] == ["MDE", "MDE", "MPE", "MPE"]
    assert len({candidate.candidate_id for candidate in pool}) == 4
    assert pool[0].candidate_id == _pool()[0].candidate_id
    assert pool[0].family == "dispersion"


def test_cache_returns_identical_features_and_reuses_them(tiny_dataset) -> None:
    spec, root = tiny_dataset
    records = load_dataset_records(spec, root)
    cache = FeatureCache(root / "cache")

    first = cache.get_or_extract(spec, root, records)
    second = cache.get_or_extract(spec, root, records)

    assert first.hit is False
    assert second.hit is True
    assert second.key == first.key
    assert first.seconds > 0
    assert second.seconds == 0
    assert (second.features == first.features).all()


def test_cache_key_changes_with_feature_parameters(tiny_dataset) -> None:
    spec, root = tiny_dataset
    records = load_dataset_records(spec, root)
    cache = FeatureCache(root / "cache")
    other = spec.model_copy(
        update={"feature": spec.feature.model_copy(update={"method": "MPE", "parameters": {"m": 3, "tau": 1, "scale": 10}})}
    )

    base = cache.get_or_extract(spec, root, records)
    changed = cache.get_or_extract(other, root, records)

    assert base.key != changed.key
    assert changed.hit is False


def test_cache_disabled_still_extracts(tiny_dataset) -> None:
    spec, root = tiny_dataset
    records = load_dataset_records(spec, root)
    cache = FeatureCache(None)

    first = cache.get_or_extract(spec, root, records)
    second = cache.get_or_extract(spec, root, records)

    assert first.hit is False
    assert second.hit is False
    assert (first.features == second.features).all()
    assert not (Path(root) / "features").exists()
