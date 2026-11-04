from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("sklearn")

from entropy_agent.classifiers.sklearn_classifiers import (
    RBFSVMClassifier,
    RandomForestClassifier,
)


@pytest.fixture
def multiclass_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(12)
    x = np.vstack(
        [
            rng.normal((-3, -3), 0.25, size=(12, 2)),
            rng.normal((0, 3), 0.25, size=(12, 2)),
            rng.normal((3, -2), 0.25, size=(12, 2)),
        ]
    )
    return x, np.repeat([10, 20, 30], 12)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RBFSVMClassifier(C=2.0, gamma="scale"),
        lambda: RandomForestClassifier(n_estimators=40, random_state=7),
    ],
)
def test_multiclass_fit_predict_and_trusted_round_trip(
    factory, multiclass_data, tmp_path
) -> None:
    x, y = multiclass_data
    model = factory().fit(x, y)
    expected = model.predict(x)
    assert np.mean(expected == y) >= 0.95

    path = tmp_path / "model.joblib"
    manifest = model.save(path, metadata={"purpose": "unit-test"})
    assert manifest["feature_dimension"] == 2
    assert manifest["classes"] == [10, 20, 30]
    assert type(model).load(path, trusted=True).predict(x).tolist() == expected.tolist()


def test_joblib_load_requires_explicit_trust(multiclass_data, tmp_path) -> None:
    x, y = multiclass_data
    path = tmp_path / "svm.joblib"
    RBFSVMClassifier().fit(x, y).save(path)
    with pytest.raises(ValueError, match="trusted=True"):
        RBFSVMClassifier.load(path)


def test_loader_rejects_algorithm_mismatch_before_joblib_load(
    multiclass_data, tmp_path
) -> None:
    x, y = multiclass_data
    path = tmp_path / "forest.joblib"
    RandomForestClassifier(n_estimators=10).fit(x, y).save(path)
    manifest_path = path.with_suffix(".joblib.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["algorithm"] == "RANDOM-FOREST"
    with pytest.raises(ValueError, match="compatible SVM-RBF"):
        RBFSVMClassifier.load(path, trusted=True)


def test_nonfinite_features_and_wrong_width_are_rejected(
    multiclass_data,
) -> None:
    x, y = multiclass_data
    model = RBFSVMClassifier().fit(x, y)
    with pytest.raises(ValueError, match="nonfinite"):
        model.predict([[np.nan, 0]])
    with pytest.raises(ValueError, match="dimension"):
        model.predict([[0, 1, 2]])


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RBFSVMClassifier(),
        lambda: RandomForestClassifier(n_estimators=10, random_state=0),
    ],
)
def test_float_labels_are_rejected_before_sklearn(factory) -> None:
    features = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    with pytest.raises(ValueError, match="integers"):
        factory().fit(features, np.asarray([0.0, 1.0, 0.0, 1.0]))
