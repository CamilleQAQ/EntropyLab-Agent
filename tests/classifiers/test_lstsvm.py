from itertools import combinations

import numpy as np
import pytest

from entropy_agent.classifiers.bt_lstsvm import BinaryLSTSVM
from entropy_agent.classifiers.lstsvm import LSTSVM
from entropy_agent.classifiers.lstsvm_artifact import load_lstsvm_artifact, save_lstsvm_artifact


def data():
    rng = np.random.default_rng(42)
    centers = np.array([[-3., -2.], [3., -2.], [0., 3.]])
    return np.vstack([rng.normal(c, .15, (n, 2)) for c, n in zip(centers, (6, 7, 8))]), np.repeat([-7, 42, 900], (6, 7, 8))


@pytest.mark.parametrize("kernel", ["linear", "rbf"])
def test_ovo_matches_binary_models_and_roundtrips(tmp_path, kernel):
    x, y = data()
    model = LSTSVM(kernel=kernel, c1=2., c2=3.).fit(x, y)
    assert model.pairs_ == tuple(combinations(range(3), 2))
    np.testing.assert_array_equal(model.predict(x), y)
    np.testing.assert_array_equal(model.predict_votes(x).sum(axis=1), 3)
    for (left, right), node in zip(model.pairs_, model.models_):
        mask = (y == model.classes_[left]) | (y == model.classes_[right])
        expected = BinaryLSTSVM(**model.parameters).fit(x[mask], np.where(y[mask] == model.classes_[left], 1, -1))
        np.testing.assert_allclose(node._plane1, expected._plane1)
        np.testing.assert_allclose(node._plane2, expected._plane2)
    path = tmp_path / "model.npz"
    save_lstsvm_artifact(path, model, metadata={"test": "roundtrip"})
    restored, manifest = load_lstsvm_artifact(path)
    assert manifest["metadata"] == {"test": "roundtrip"}
    np.testing.assert_array_equal(restored.predict(x), model.predict(x))
    np.testing.assert_array_equal(restored.predict_votes(x), model.predict_votes(x))
    x[:] = 100  # The fitted state must not alias caller storage.
    np.testing.assert_array_equal(restored.predict(x), model.predict(x))


def test_binary_case_equals_existing_binary():
    x, y = data()
    mask = y != 900
    model = LSTSVM().fit(x[mask], y[mask])
    binary = BinaryLSTSVM().fit(x[mask], np.where(y[mask] == -7, 1, -1))
    np.testing.assert_array_equal(model.predict(x), np.where(binary.predict(x) == 1, -7, 42))


def test_vote_tie_is_lowest_class_id():
    x, y = data()
    model = LSTSVM().fit(x, y)
    class Fixed:
        def __init__(self, sign):
            self.sign = sign
        def predict(self, x):
            return np.full(len(x), self.sign)
    model.models_ = [Fixed(1), Fixed(-1), Fixed(1)]
    np.testing.assert_array_equal(model.predict_votes(x), np.ones((len(x), 3)))
    assert (model.predict(x) == -7).all()


@pytest.mark.parametrize("mutation", ["nonfinite", "plane_shape", "missing", "pairs"])
def test_corrupt_artifact_rejected(tmp_path, mutation):
    import json
    x, y = data()
    path = tmp_path / "bad.npz"
    save_lstsvm_artifact(path, LSTSVM().fit(x, y))
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if mutation == "nonfinite":
        arrays["plane1_0"][0] = np.nan
    elif mutation == "plane_shape":
        arrays["plane1_0"] = arrays["plane1_0"][:-1]
    elif mutation == "missing":
        del arrays["anchors_0"]
    else:
        manifest = json.loads(str(arrays["manifest"]))
        manifest["pairs"] = [[0, 1]]
        arrays["manifest"] = np.asarray(json.dumps(manifest))
    np.savez(path, **arrays)
    with pytest.raises(ValueError):
        load_lstsvm_artifact(path)


def test_input_validation_and_refit():
    x, y = data()
    model = LSTSVM()
    with pytest.raises(ValueError, match="fitted"):
        model.predict(x)
    for labels in (np.ones(len(x), dtype=int), y.astype(float), y[:-1]):
        with pytest.raises(ValueError):
            model.fit(x, labels)
    with pytest.raises(ValueError):
        LSTSVM(epsilon=0)
    with pytest.raises(ValueError, match="nonfinite"):
        model.fit(np.full_like(x, np.nan), y)
    model.fit(x, y)
    with pytest.raises(ValueError, match="dimension"):
        model.predict(np.ones((2, 3)))
    model.fit(x[y != 900], y[y != 900])
    assert len(model.models_) == 1
    assert model.classes_.tolist() == [-7, 42]
