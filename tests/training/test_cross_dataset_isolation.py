"""Different datasets must never share models, predictions, or metrics."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from entropy_agent.diagnosis.artifact import load_model_artifact
from entropy_agent.diagnosis.model_registry import ModelRegistry
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


def _signal(label: int, group: int, *, samples: int = 256) -> np.ndarray:
    rng = np.random.default_rng(5000 + label * 100 + group)
    time = np.arange(samples, dtype=np.float64) / 1024
    frequencies = (25, 120, 300)
    return np.sin(2 * np.pi * frequencies[label] * time) + 0.03 * rng.normal(size=samples)


def _write_dataset(root: Path, names: tuple[str, ...]) -> None:
    for label, name in enumerate(names):
        folder = root / name
        folder.mkdir(parents=True)
        for group in range(5):
            values = _signal(label, group)
            (folder / f"{group}.csv").write_text(
                "signal\n" + "\n".join(f"{value:.17g}" for value in values),
                encoding="utf-8",
            )


def _spec(dataset_id: str, names: tuple[str, ...], profile_id: str) -> DatasetSpec:
    return DatasetSpec(
        dataset_id=dataset_id,
        sampling_rate_hz=1024,
        input=InputConfig(channel="signal"),
        layout=DatasetLayout(file_pattern="*/*.csv"),
        labels=LabelConfig(
            names=names,
            mapping={name: index for index, name in enumerate(names)},
        ),
        window=WindowConfig(size=128, stride=128, windows_per_record=1),
        feature=FeatureConfig(
            method="MDE", parameters={"m": 2, "nc": 5, "tau": 1, "scale": 3}
        ),
        classifier=ClassifierConfig(algorithm="LSTSVM", backend="numpy", parameters={}),
        split=SplitConfig(seed=43),
        diagnosis=DiagnosisConfig(
            profile_id=profile_id,
            display_name=f"{dataset_id} profile",
            signal_type="VIBRATION",
            signal_role="SYNTHETIC_SENSOR",
            minimum_window_count=1,
            evidence_window_count=1,
            validation_scope="synthetic group-isolated holdout",
        ),
    )


@pytest.fixture(scope="module")
def datasets(tmp_path_factory) -> tuple[Path, Path, DatasetSpec, DatasetSpec]:
    root = tmp_path_factory.mktemp("cross-dataset")
    data_a = root / "data-a"
    data_b = root / "data-b"
    names_a = ("normal", "fault")
    names_b = ("healthy", "inner_race", "outer_race")
    _write_dataset(data_a, names_a)
    _write_dataset(data_b, names_b)
    return (
        data_a,
        data_b,
        _spec("DATASET_A", names_a, "DATASET_A_PROFILE"),
        _spec("DATASET_B", names_b, "DATASET_B_PROFILE"),
    )


def test_models_and_metrics_are_isolated_between_datasets(
    datasets, tmp_path: Path
) -> None:
    data_a, data_b, spec_a, spec_b = datasets
    training_a = train_dataset(
        spec_a, data_a, tmp_path / "models" / "a", model_id="dataset-a"
    )
    training_b = train_dataset(
        spec_b, data_b, tmp_path / "models" / "b", model_id="dataset-b"
    )
    artifact_a = training_a.artifact
    artifact_b = training_b.artifact

    # Each artifact keeps its own labels and source identity.
    assert artifact_a.labels == {0: "normal", 1: "fault"}
    assert artifact_b.labels == {0: "healthy", 1: "inner_race", 2: "outer_race"}
    assert artifact_a.config["provenance"]["dataset_id"] == "DATASET_A"
    assert artifact_b.config["provenance"]["dataset_id"] == "DATASET_B"
    assert artifact_a.config["model_id"] != artifact_b.config["model_id"]

    # Metrics come from each artifact's own split, not from a shared pool.
    assert artifact_a.test_result["split_records"] == {
        "train": 6, "validation": 2, "test": 2,
    }
    assert artifact_b.test_result["split_records"] == {
        "train": 9, "validation": 3, "test": 3,
    }
    assert artifact_a.test_result == training_a.test_result
    assert artifact_b.test_result == training_b.test_result
    assert artifact_b.test_result != artifact_a.test_result

    # The stored metrics are exactly the metrics of the stored held-out predictions.
    for artifact in (artifact_a, artifact_b):
        rows = artifact.test_result["test_predictions"]
        truth = np.asarray([row["truth"] for row in rows], dtype=np.int64)
        predicted = np.asarray([row["predicted"] for row in rows], dtype=np.int64)
        class_count = len(artifact.labels)
        metrics = artifact.test_result["test"]
        assert len(rows) == artifact.test_result["split_records"]["test"]
        assert metrics["multiclass_accuracy"] == pytest.approx(
            accuracy_score(truth, predicted)
        )
        assert metrics["multiclass_macro_f1"] == pytest.approx(
            f1_score(truth, predicted, labels=range(class_count), average="macro", zero_division=0)
        )
        assert metrics["multiclass_confusion_matrix"] == confusion_matrix(
            truth, predicted, labels=range(class_count)
        ).tolist()

    # Re-loading and diagnosing each model must use its own labels.
    loaded_a = load_model_artifact(tmp_path / "models" / "a")
    loaded_b = load_model_artifact(tmp_path / "models" / "b")
    assert loaded_a.labels == artifact_a.labels
    assert loaded_b.labels == artifact_b.labels

    report_a = diagnose_artifact_signal(
        _signal(1, 99), loaded_a.directory, sampling_rate_hz=1024
    )
    report_b = diagnose_artifact_signal(
        _signal(2, 99), loaded_b.directory, sampling_rate_hz=1024
    )
    assert report_a["status"] == "DIAGNOSED"
    assert report_b["status"] == "DIAGNOSED"
    assert report_a["diagnosis"] in set(artifact_a.labels.values())
    assert report_b["diagnosis"] in set(artifact_b.labels.values())
    assert report_a["model_id"] == "dataset-a"
    assert report_b["model_id"] == "dataset-b"

    # The registry exposes both profiles without merging their labels.
    registry = ModelRegistry(tmp_path / "registry")
    registry.register(tmp_path / "models" / "a")
    registry.register(tmp_path / "models" / "b")
    assert registry.model_ids() == ("dataset-a", "dataset-b")
    profiles = {profile.profile_id: profile for profile in registry.as_diagnosis_registry().profiles()}
    assert profiles["DATASET_A_PROFILE"].output_labels == artifact_a.labels
    assert profiles["DATASET_B_PROFILE"].output_labels == artifact_b.labels

    # Training or diagnosing B cannot mutate A's recorded evaluation.
    before = copy.deepcopy(artifact_a.test_result)
    _ = diagnose_artifact_signal(
        _signal(0, 99), loaded_b.directory, sampling_rate_hz=1024
    )
    assert artifact_a.test_result == before
