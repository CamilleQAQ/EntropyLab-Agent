from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from entropy_agent.classifiers.torch_artifact import (
    FORMAT, load_torch_bt_artifact, save_torch_bt_artifact,
)
from entropy_agent.classifiers.torch_bt_lstsvm import (
    MatlabCompatTorchBTLSTSVM,
    TorchBTLSTSVM,
)


def _model() -> TorchBTLSTSVM:
    x = np.array([[-4.0], [-3.0], [0.0], [1.0], [4.0], [5.0]])
    y = np.array([1, 1, 2, 2, 3, 3])
    return TorchBTLSTSVM().fit(x, y, class_order=(1, 2, 3))


def test_numeric_artifact_round_trip_preserves_predictions_and_metadata(tmp_path: Path) -> None:
    model = _model()
    path = tmp_path / "torch_bt.npz"
    manifest = save_torch_bt_artifact(
        path, model,
        metadata={"diagnosis_profile_id": "TEST_PROFILE", "validation_run_id": "fixture-1"},
    )
    restored, loaded_manifest = load_torch_bt_artifact(path)
    test = np.array([[-3.5], [0.5], [4.5]])
    np.testing.assert_array_equal(restored.predict(test), model.predict(test))
    assert manifest == loaded_manifest
    assert loaded_manifest["format"] == FORMAT
    assert loaded_manifest["metadata"]["diagnosis_profile_id"] == "TEST_PROFILE"


def test_artifact_rejects_non_json_metadata(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="finite JSON object"):
        save_torch_bt_artifact(tmp_path / "bad.npz", _model(), metadata={"bad": np.nan})


def test_artifact_refuses_matlab_compatibility_model(tmp_path: Path) -> None:
    x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([1, 1, 2, 2])
    model = MatlabCompatTorchBTLSTSVM().fit(x, y, class_order=(1, 2))
    with pytest.raises(ValueError, match="only TorchBTLSTSVM"):
        save_torch_bt_artifact(tmp_path / "compat.npz", model)  # type: ignore[arg-type]


def test_loader_rejects_incompatible_manifest(tmp_path: Path) -> None:
    path = tmp_path / "wrong.npz"
    manifest = np.asarray(json.dumps({"format": "unknown"}))
    with path.open("wb") as handle:
        np.savez_compressed(handle, manifest=manifest)
    with pytest.raises(ValueError, match="incompatible"):
        load_torch_bt_artifact(path)
