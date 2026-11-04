"""Artifact contract tests for OOD numerics and the v4-only loader."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from entropy_agent.classifiers.lstsvm import LSTSVM
from entropy_agent.diagnosis.artifact import (
    _algorithm_digest,
    load_model_artifact,
    save_model_artifact,
)
from entropy_agent.diagnosis.runner import diagnose_artifact_signal
from entropy_agent.io.dataset import (
    ClassifierConfig,
    DatasetLayout,
    DatasetSpec,
    DiagnosisConfig,
    FeatureConfig,
    InputConfig,
    LabelConfig,
    OODConfig,
    SplitConfig,
    WindowConfig,
)
from entropy_agent.training.pipeline import _build_ood_contract, train_dataset


NAMES = ("normal", "fault")


def _signal(label: int, group: int, *, samples: int = 256) -> np.ndarray:
    rng = np.random.default_rng(3000 + label * 100 + group)
    time = np.arange(samples, dtype=np.float64) / 1024
    frequency = 35 if label == 0 else 170
    return np.sin(2 * np.pi * frequency * time) + 0.03 * rng.normal(size=samples)


def _spec() -> DatasetSpec:
    return DatasetSpec(
        dataset_id="ARTIFACT_CONTRACT",
        sampling_rate_hz=1024,
        input=InputConfig(channel="signal"),
        layout=DatasetLayout(file_pattern="*/*.csv"),
        labels=LabelConfig(names=NAMES, mapping={name: index for index, name in enumerate(NAMES)}),
        window=WindowConfig(size=128, stride=128, windows_per_record=2),
        feature=FeatureConfig(
            method="MDE", parameters={"m": 2, "nc": 5, "tau": 1, "scale": 3}
        ),
        classifier=ClassifierConfig(algorithm="LSTSVM", backend="numpy", parameters={}),
        split=SplitConfig(seed=31),
        diagnosis=DiagnosisConfig(
            profile_id="ARTIFACT_CONTRACT_PROFILE",
            display_name="Artifact contract test",
            signal_type="VIBRATION",
            signal_role="SYNTHETIC_SENSOR",
            minimum_window_count=1,
            evidence_window_count=2,
            validation_scope="synthetic group-isolated holdout",
        ),
    )


@pytest.fixture(scope="module")
def trained_artifact(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("artifact-contract")
    data = root / "data"
    for label, name in enumerate(NAMES):
        folder = data / name
        folder.mkdir(parents=True)
        for group in range(5):
            values = _signal(label, group)
            (folder / f"{group}.csv").write_text(
                "signal\n" + "\n".join(f"{value:.17g}" for value in values),
                encoding="utf-8",
            )
    directory = root / "model"
    train_dataset(_spec(), data, directory, model_id="artifact-contract")
    return directory


@pytest.fixture
def artifact_copy(trained_artifact: Path, tmp_path: Path) -> Path:
    destination = tmp_path / "model"
    shutil.copytree(trained_artifact, destination)
    return destination


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _enable_ood(config: dict) -> dict:
    dimension = config["feature"]["dimension"]
    config["ood"] = {
        "enabled": True,
        "distance": "NEAREST_TRAINING_STANDARDIZED_EUCLIDEAN",
        "series_policy": "STRICT_MAJORITY_WINDOWS",
        "threshold": 1.0,
        "center": [0.0] * dimension,
        "scale": [1.0] * dimension,
        "anchors": [[0.0] * dimension],
        "calibration": "test fixture",
        "margin": 1.05,
        "calibration_rows": 2,
    }
    return config


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda ood: ood.__setitem__("threshold", float("nan")), "ood.threshold"),
        (lambda ood: ood.__setitem__("threshold", float("inf")), "ood.threshold"),
        (lambda ood: ood.__setitem__("threshold", -1.0), "ood.threshold"),
        (lambda ood: ood["center"].__setitem__(0, float("nan")), "ood.center"),
        (lambda ood: ood["scale"].__setitem__(0, 0.0), "ood.scale"),
        (lambda ood: ood["scale"].__setitem__(0, float("inf")), "ood.scale"),
        (lambda ood: ood["anchors"][0].__setitem__(0, float("nan")), "ood.anchors"),
        (lambda ood: ood.__setitem__("margin", float("nan")), "ood.margin"),
        (lambda ood: ood["center"].append(0.0), "ood.center"),
        (lambda ood: ood.pop("calibration"), "OOD contract"),
    ],
)
def test_invalid_ood_numeric_values_are_rejected(
    artifact_copy: Path, mutate, match: str
) -> None:
    config = _enable_ood(_read_json(artifact_copy / "config.json"))
    mutate(config["ood"])
    _write_json(artifact_copy / "config.json", config)

    with pytest.raises(ValueError, match=match):
        load_model_artifact(artifact_copy)


def test_runtime_nonfinite_ood_standardization_abstains(
    trained_artifact: Path, tmp_path: Path
) -> None:
    artifact = load_model_artifact(trained_artifact)
    config = _enable_ood(_read_json(trained_artifact / "config.json"))
    dimension = config["feature"]["dimension"]
    config["ood"].update(
        {
            "center": [1e308] * dimension,
            "scale": [1e-12] * dimension,
            "anchors": [[1e308] * dimension],
        }
    )
    destination = tmp_path / "extreme-ood"
    save_model_artifact(
        destination,
        artifact.model,
        config=config,
        labels=artifact.labels,
        test_result=artifact.test_result,
    )

    report = diagnose_artifact_signal(
        _signal(1, 99), destination, sampling_rate_hz=1024
    )
    assert report["status"] == "ABSTAINED_INVALID_INPUT"
    assert report["diagnosis"] is None
    assert "nonfinite" in report["reason"]


def test_integral_float_predictions_are_cast_before_bincount(
    monkeypatch, trained_artifact: Path
) -> None:
    monkeypatch.setattr(
        LSTSVM, "predict", lambda self, x: np.zeros(len(x), dtype=np.float64)
    )
    report = diagnose_artifact_signal(
        _signal(1, 99), trained_artifact, sampling_rate_hz=1024
    )
    assert report["status"] == "DIAGNOSED"


def test_fractional_predictions_abstain_instead_of_raising(
    monkeypatch, trained_artifact: Path
) -> None:
    monkeypatch.setattr(
        LSTSVM, "predict", lambda self, x: np.full(len(x), 0.5, dtype=np.float64)
    )
    report = diagnose_artifact_signal(
        _signal(1, 99), trained_artifact, sampling_rate_hz=1024
    )
    assert report["status"] == "ABSTAINED_INVALID_INPUT"
    assert report["diagnosis"] is None
    assert "class ID" in report["reason"]


def test_ood_builder_rejects_nonfinite_margin(
    trained_artifact: Path,
) -> None:
    artifact = load_model_artifact(trained_artifact)
    features = np.zeros((4, 3), dtype=np.float64)
    mask = np.asarray([True, True, False, False])
    spec = _spec().model_copy(
        update={
            "ood": OODConfig.model_construct(
                calibration="VALIDATION_MAX_NEAREST_TRAINING",
                margin=float("nan"),
            )
        }
    )
    with pytest.raises(ValueError, match="margin"):
        _build_ood_contract(spec, features, features, mask, 3)
    assert artifact.config["format_version"] == 4


def test_ood_builder_rejects_nonfinite_features() -> None:
    features = np.zeros((4, 3), dtype=np.float64)
    features[0, 0] = np.nan
    mask = np.asarray([True, True, False, False])
    spec = _spec().model_copy(
        update={
            "ood": OODConfig(
                calibration="VALIDATION_MAX_NEAREST_TRAINING", margin=1.05
            )
        }
    )
    with pytest.raises(ValueError, match="OOD"):
        _build_ood_contract(spec, features, features, mask, 3)


def test_version_4_digest_binds_acquisition_diagnosis_and_ood(
    trained_artifact: Path,
) -> None:
    config = _read_json(trained_artifact / "config.json")
    base = _algorithm_digest(config)

    mutations = [
        lambda item: item.__setitem__("sample_rate_hz", 2048),
        lambda item: item["diagnosis"].__setitem__("minimum_window_count", 2),
        lambda item: _enable_ood(item)["ood"].__setitem__("threshold", 2.0),
    ]
    for mutate in mutations:
        variant = copy.deepcopy(config)
        mutate(variant)
        assert _algorithm_digest(variant) != base


@pytest.mark.parametrize("field,value", [
    ("channel", 1), ("key", "other_signal"), ("sample_axis", 0),
    ("indices", [1]), ("has_header", False), ("delimiter", ";"),
    ("skip_rows", 3),
])
def test_version_4_rejects_modified_input_contract(artifact_copy, field, value):
    config = _read_json(artifact_copy / "config.json")
    assert config["format_version"] == 4
    config["input"][field] = value
    _write_json(artifact_copy / "config.json", config)
    with pytest.raises(ValueError, match="identity"):
        load_model_artifact(artifact_copy)


def test_version_4_input_dictionary_order_does_not_change_digest(trained_artifact):
    config = _read_json(trained_artifact / "config.json")
    variant = copy.deepcopy(config)
    variant["input"] = dict(reversed(list(variant["input"].items())))
    assert _algorithm_digest(variant) == _algorithm_digest(config)


def test_sampling_rate_tamper_fails_identity_check(artifact_copy: Path) -> None:
    config = _read_json(artifact_copy / "config.json")
    config["sample_rate_hz"] = 2048
    _write_json(artifact_copy / "config.json", config)
    with pytest.raises(ValueError, match="identity"):
        load_model_artifact(artifact_copy)


def test_unsupported_preprocessing_is_rejected(artifact_copy: Path) -> None:
    config = _read_json(artifact_copy / "config.json")
    config["preprocessing"] = {"method": "ZSCORE"}
    _write_json(artifact_copy / "config.json", config)
    with pytest.raises(ValueError, match="preprocessing"):
        load_model_artifact(artifact_copy)


def test_disabled_ood_requires_a_reason(artifact_copy: Path) -> None:
    config = _read_json(artifact_copy / "config.json")
    config["ood"] = {"enabled": False}
    _write_json(artifact_copy / "config.json", config)
    with pytest.raises(ValueError, match="reason"):
        load_model_artifact(artifact_copy)


@pytest.mark.parametrize("version", [1, 2, 3, True, 3.0, "3"])
def test_legacy_or_non_integer_format_version_is_rejected(
    artifact_copy: Path, version
) -> None:
    config = _read_json(artifact_copy / "config.json")
    config["format_version"] = version
    _write_json(artifact_copy / "config.json", config)
    with pytest.raises(ValueError, match="incompatible"):
        load_model_artifact(artifact_copy)


def test_malformed_diagnosis_contract_is_rejected(artifact_copy: Path) -> None:
    config = _read_json(artifact_copy / "config.json")
    config["diagnosis"].pop("minimum_window_count")
    _write_json(artifact_copy / "config.json", config)
    with pytest.raises(ValueError, match="diagnosis"):
        load_model_artifact(artifact_copy)


def test_malformed_input_contract_is_rejected(artifact_copy: Path) -> None:
    config = _read_json(artifact_copy / "config.json")
    config["input"] = []
    _write_json(artifact_copy / "config.json", config)
    with pytest.raises(ValueError, match="input"):
        load_model_artifact(artifact_copy)


@pytest.mark.parametrize("scale", [0, 1.5, float("nan"), float("inf")])
def test_invalid_entropy_scale_is_rejected(
    artifact_copy: Path, scale: float
) -> None:
    config = _read_json(artifact_copy / "config.json")
    config["feature"]["parameters"]["scale"] = scale
    _write_json(artifact_copy / "config.json", config)
    with pytest.raises(ValueError):
        load_model_artifact(artifact_copy)
