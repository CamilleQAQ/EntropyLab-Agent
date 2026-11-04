"""Regression tests for the model label contract.

The class ID -> class name mapping shown to users is part of the artifact
contract.  Renaming, reordering or reassigning labels must never silently change
what a stored prediction means.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from entropy_agent.diagnosis.artifact import (
    label_contract_digest,
    load_model_artifact,
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
    SplitConfig,
    WindowConfig,
)
from entropy_agent.training.pipeline import train_dataset


NAMES = ("normal", "fault")


def _signal(label: int, group: int, *, samples: int = 256) -> np.ndarray:
    rng = np.random.default_rng(1000 + label * 100 + group)
    time = np.arange(samples, dtype=np.float64) / 1024
    frequency = 35 if label == 0 else 170
    return np.sin(2 * np.pi * frequency * time) + 0.03 * rng.normal(size=samples)


def _spec() -> DatasetSpec:
    return DatasetSpec(
        dataset_id="LABEL_CONTRACT",
        sampling_rate_hz=1024,
        input=InputConfig(channel="signal"),
        layout=DatasetLayout(file_pattern="*/*.csv"),
        labels=LabelConfig(names=NAMES, mapping={name: index for index, name in enumerate(NAMES)}),
        window=WindowConfig(size=128, stride=128, windows_per_record=2),
        feature=FeatureConfig(
            method="MDE", parameters={"m": 2, "nc": 5, "tau": 1, "scale": 3}
        ),
        classifier=ClassifierConfig(algorithm="LSTSVM", backend="numpy", parameters={}),
        split=SplitConfig(seed=17),
        diagnosis=DiagnosisConfig(
            profile_id="LABEL_CONTRACT_PROFILE",
            display_name="Label contract test",
            signal_type="VIBRATION",
            signal_role="SYNTHETIC_SENSOR",
            minimum_window_count=1,
            evidence_window_count=2,
            validation_scope="synthetic group-isolated holdout",
        ),
    )


@pytest.fixture(scope="module")
def trained_artifact(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("label-contract")
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
    train_dataset(_spec(), data, directory, model_id="label-contract")
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


def _labels(directory: Path) -> dict:
    return _read_json(directory / "labels.json")


def _set_labels(directory: Path, document: dict) -> None:
    _write_json(directory / "labels.json", document)


def _config(directory: Path) -> dict:
    return _read_json(directory / "config.json")


def _set_config(directory: Path, document: dict) -> None:
    _write_json(directory / "config.json", document)


def test_valid_mapping_round_trips_and_names_predictions(artifact_copy: Path) -> None:
    artifact = load_model_artifact(artifact_copy)
    assert artifact.labels == {0: "normal", 1: "fault"}
    assert artifact.profile.output_labels == {0: "normal", 1: "fault"}
    report = diagnose_artifact_signal(
        _signal(1, 99), artifact_copy, sampling_rate_hz=1024
    )
    assert report["status"] == "DIAGNOSED"
    assert report["diagnosis"] in NAMES
    for window in report["window_predictions"]:
        assert window["class_label"] == artifact.labels[window["class_id"]]


def test_swapped_class_names_are_rejected(artifact_copy: Path) -> None:
    document = _labels(artifact_copy)
    document["labels"][0]["name"], document["labels"][1]["name"] = (
        document["labels"][1]["name"],
        document["labels"][0]["name"],
    )
    _set_labels(artifact_copy, document)

    with pytest.raises(ValueError, match="label contract"):
        load_model_artifact(artifact_copy)
    with pytest.raises(ValueError, match="label contract"):
        diagnose_artifact_signal(
            _signal(1, 99), artifact_copy, sampling_rate_hz=1024
        )


def test_reassigned_ids_with_same_names_are_rejected(artifact_copy: Path) -> None:
    document = _labels(artifact_copy)
    first_id = document["labels"][0]["id"]
    document["labels"][0]["id"] = document["labels"][1]["id"]
    document["labels"][1]["id"] = first_id
    _set_labels(artifact_copy, document)

    with pytest.raises(ValueError, match="label contract"):
        load_model_artifact(artifact_copy)
    with pytest.raises(ValueError):
        diagnose_artifact_signal(
            _signal(0, 99), artifact_copy, sampling_rate_hz=1024
        )


def test_label_list_order_alone_is_allowed(artifact_copy: Path) -> None:
    document = _labels(artifact_copy)
    document["labels"].reverse()
    _set_labels(artifact_copy, document)

    artifact = load_model_artifact(artifact_copy)
    assert artifact.labels == {0: "normal", 1: "fault"}


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda labels: labels.__setitem__(slice(None), labels[:1]), "contiguous"),
        (
            lambda labels: labels.append({"id": labels[0]["id"], "name": "extra"}),
            "duplicate IDs",
        ),
        (
            lambda labels: labels[1].__setitem__("name", labels[0]["name"]),
            "duplicate class names",
        ),
        (lambda labels: labels[1].__setitem__("name", "   "), "invalid label"),
        (lambda labels: labels[1].__setitem__("id", True), "invalid label"),
    ],
)
def test_missing_duplicate_or_invalid_labels_are_rejected(
    artifact_copy: Path, mutate, match: str
) -> None:
    document = _labels(artifact_copy)
    mutate(document["labels"])
    _set_labels(artifact_copy, document)

    with pytest.raises(ValueError, match=match):
        load_model_artifact(artifact_copy)


def test_label_digest_mismatch_between_config_and_manifest_is_rejected(
    artifact_copy: Path,
) -> None:
    document = _labels(artifact_copy)
    swapped = {0: "fault", 1: "normal"}
    document["labels"] = [{"id": key, "name": value} for key, value in swapped.items()]
    _set_labels(artifact_copy, document)
    config = _config(artifact_copy)
    config["labels_sha256"] = label_contract_digest(swapped)
    _set_config(artifact_copy, config)

    # config.json alone now agrees with labels.json, but the model manifest does not.
    with pytest.raises(ValueError, match="not bound to labels.json"):
        load_model_artifact(artifact_copy)


def test_reordered_class_order_is_rejected(artifact_copy: Path) -> None:
    model_path = artifact_copy / "model.npz"
    with np.load(model_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    manifest = json.loads(str(arrays["manifest"]))
    manifest["class_order"] = list(reversed(manifest["class_order"]))
    np.savez(model_path, **dict(arrays, manifest=np.asarray(json.dumps(manifest))))

    # The LSTSVM codec requires ascending classes, so the reorder never reaches
    # inference; the artifact layer additionally enforces ascending order for
    # every backend whose tree or estimator depends on class position.
    with pytest.raises(ValueError, match="classes|class order"):
        load_model_artifact(artifact_copy)
