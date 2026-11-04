import gc
import json
from pathlib import Path

import numpy as np
import pytest

from entropy_agent.diagnosis.artifact import load_model_artifact, save_model_artifact
from entropy_agent.diagnosis.runner import diagnose_artifact_signal
from entropy_agent.diagnosis.composition import load_diagnosis_registry
from entropy_agent.diagnosis.model_registry import ModelRegistry
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
from entropy_agent.classifiers.registry import classifier_catalog


def test_unknown_classifier_fails_before_reading_dataset(tmp_path):
    assert classifier_catalog()["LSTSVM"]["available"] is True
    spec = _spec().model_copy(update={"classifier": ClassifierConfig(
        algorithm="UNKNOWN", backend="numpy", parameters={})})
    with pytest.raises(ValueError, match="unsupported classifier"):
        train_dataset(spec, tmp_path / "absent", tmp_path / "model")


@pytest.mark.parametrize("method,parameters,algorithm,backend,classifier_parameters", [
    ("MDE", {"m": 2, "nc": 5, "tau": 1, "scale": 3},
     "LSTSVM", "numpy", {}),
    ("MDE", {"m": 2, "nc": 5, "tau": 1, "scale": 3},
     "SVM-RBF", "scikit-learn", {}),
    ("TSMEFuDE", {"m": 2, "nc": 5, "tau": 1, "kmax": 3},
     "BT-LSTSVM", "torch", {}),
    ("MDE", {"m": 2, "nc": 5, "tau": 1, "scale": 3},
     "RANDOM-FOREST", "scikit-learn", {"n_estimators": 10}),
])
def test_configured_algorithm_roundtrip(tmp_path, method, parameters, algorithm,
                                      backend, classifier_parameters):
    for label_id, label_name in enumerate(("normal", "fault")):
        folder = tmp_path / "data" / label_name
        folder.mkdir(parents=True)
        for group in range(5):
            _write_signal(folder / f"{group}.csv", _signal(label_id, group))
    spec = _spec().model_copy(update={
        "feature": FeatureConfig(method=method, parameters=parameters),
        "classifier": ClassifierConfig(algorithm=algorithm, backend=backend,
                                        parameters=classifier_parameters),
    })
    directory = tmp_path / "model"
    result = train_dataset(spec, tmp_path / "data", directory)
    trusted = backend == "scikit-learn"
    if trusted:
        with pytest.raises(ValueError, match="trusted=True"):
            load_model_artifact(directory)
    restored = load_model_artifact(directory, trusted=trusted)
    assert restored.config["feature"]["method"] == method
    assert restored.config["classifier"]["parameters"] == restored.model.parameters
    assert result.test_result["test"]["multiclass_accuracy"] >= 0.5
    registry = ModelRegistry(tmp_path / "registry", trusted=trusted)
    registry.register(directory)
    registration = registry.as_diagnosis_registry().get(spec.diagnosis.profile_id)
    report = registration.runner(_signal(1, 99), directory, sampling_rate_hz=1024)
    assert report["status"] == "DIAGNOSED"
    assert len(report["window_predictions"][0]["feature_values"]) == 3
    if backend == "torch":
        model_path = directory / "model.npz"
        with np.load(model_path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}

        # A reordered tree manifest would silently assign branches to the wrong names.
        reordered = json.loads(str(arrays["manifest"]))
        reordered["class_order"] = list(reversed(reordered["class_order"]))
        np.savez(
            model_path,
            **dict(arrays, manifest=np.asarray(json.dumps(reordered))),
        )
        with pytest.raises(ValueError, match="class order"):
            load_model_artifact(directory)
        np.savez(model_path, **arrays)
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["feature"]["parameters"]["nc"] = 6
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        load_model_artifact(directory, trusted=trusted)


def test_three_class_lstsvm_training_and_routed_diagnosis(tmp_path):
    names = ("low", "medium", "high")
    rng = np.random.default_rng(918)
    for label, name in enumerate(names):
        folder = tmp_path / "data" / name
        folder.mkdir(parents=True)
        for group in range(5):
            t = np.arange(256) / 1024
            signal = np.sin(2 * np.pi * (35, 170, 350)[label] * t)
            _write_signal(folder / f"{group}.csv", signal + rng.normal(0, .02, 256))
    spec = _spec().model_copy(update={
        "labels": LabelConfig(names=names, mapping={name: i for i, name in enumerate(names)}),
        "classifier": ClassifierConfig(algorithm="LSTSVM", backend="numpy", parameters={}),
    })
    directory = tmp_path / "model"
    result = train_dataset(spec, tmp_path / "data", directory)
    assert len(result.artifact.model.models_) == 3
    assert result.artifact.labels == dict(enumerate(names))
    registry = ModelRegistry(tmp_path / "registry")
    registry.register(directory)
    registration = registry.as_diagnosis_registry().get(spec.diagnosis.profile_id)
    report = registration.runner(signal, directory, sampling_rate_hz=1024)
    assert report["status"] == "DIAGNOSED"
    assert report["diagnosis"] in names
    assert sum(report["class_counts"].values()) == 2


def _signal(label: int, group: int, *, samples: int = 256) -> np.ndarray:
    rng = np.random.default_rng(1000 + label * 100 + group)
    time = np.arange(samples, dtype=np.float64) / 1024
    frequency = 35 if label == 0 else 170
    return np.sin(2 * np.pi * frequency * time) + 0.03 * rng.normal(size=samples)


def _write_signal(path: Path, values: np.ndarray) -> None:
    path.write_text(
        "signal\n" + "\n".join(f"{value:.17g}" for value in values),
        encoding="utf-8",
    )


def _spec() -> DatasetSpec:
    return DatasetSpec(
        dataset_id="SYNTHETIC_E2E",
        sampling_rate_hz=1024,
        input=InputConfig(channel="signal"),
        layout=DatasetLayout(file_pattern="*/*.csv"),
        labels=LabelConfig(
            names=("normal", "fault"), mapping={"normal": 0, "fault": 1},
        ),
        window=WindowConfig(size=128, stride=128, windows_per_record=2),
        feature=FeatureConfig(
            method="MDE", parameters={"m": 2, "nc": 5, "tau": 1, "scale": 3},
        ),
        classifier=ClassifierConfig(parameters={
            "c1": 0.25, "c2": 0.25, "bandwidth": 4.0, "epsilon": 1e-4,
        }),
        split=SplitConfig(seed=17),
        diagnosis=DiagnosisConfig(
            profile_id="SYNTHETIC_E2E_PROFILE",
            display_name="Synthetic end-to-end diagnosis",
            signal_type="VIBRATION",
            signal_role="SYNTHETIC_SENSOR",
            minimum_window_count=1,
            evidence_window_count=2,
            validation_scope="synthetic group-isolated holdout",
        ),
    )


def test_synthetic_dataset_trains_saves_reloads_and_diagnoses_new_raw_signal(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    for label_id, label_name in enumerate(("normal", "fault")):
        folder = dataset / label_name
        folder.mkdir(parents=True)
        for group in range(5):
            _write_signal(folder / f"group-{group}.csv", _signal(label_id, group))

    model_directory = tmp_path / "models" / "synthetic-v1"
    training = train_dataset(
        _spec(), dataset, model_directory, model_id="synthetic-v1",
    )
    assert {path.name for path in model_directory.iterdir()} == {
        "model.npz", "config.json", "labels.json", "test_result.json",
    }
    assert training.test_result["split_records"] == {
        "train": 6, "validation": 2, "test": 2,
    }

    del training
    gc.collect()
    restored = load_model_artifact(model_directory)
    assert restored.config["provenance"]["dataset_id"] == "SYNTHETIC_E2E"
    del restored
    gc.collect()

    new_raw_signal = _signal(1, 99)
    diagnosis = diagnose_artifact_signal(
        new_raw_signal, model_directory, sampling_rate_hz=1024,
    )
    assert diagnosis["status"] == "DIAGNOSED"
    assert diagnosis["diagnosis"] in {"normal", "fault"}
    assert diagnosis["window_count"] == 2
    assert len(diagnosis["window_predictions"]) == 2
    assert sum(diagnosis["class_counts"].values()) == 2
    assert diagnosis["probability"] is None

    registry = ModelRegistry(tmp_path / "registry")
    registry.register(model_directory)
    assert registry.model_ids() == ("synthetic-v1",)
    assert registry.show("synthetic-v1")["profile"]["profile_id"] == "SYNTHETIC_E2E_PROFILE"
    assert registry.load("synthetic-v1").labels == {0: "normal", 1: "fault"}

    diagnosis_registry = load_diagnosis_registry(
        None, model_registry_root=tmp_path / "registry",
    )
    registration = diagnosis_registry.get("SYNTHETIC_E2E_PROFILE")
    assert registration is not None
    routed = registration.runner(
        new_raw_signal, registration.model_path, sampling_rate_hz=1024,
    )
    assert routed["status"] == "DIAGNOSED"

    artifact_path = registry.delete("synthetic-v1")
    assert artifact_path == model_directory.resolve()
    assert registry.model_ids() == ()
    assert model_directory.is_dir()  # Registry deletion does not delete the model by default.

    artifact = load_model_artifact(model_directory)
    config = artifact.config
    dimension = config["feature"]["dimension"]
    config["ood"] = {
        "enabled": True,
        "distance": "NEAREST_TRAINING_STANDARDIZED_EUCLIDEAN",
        "series_policy": "STRICT_MAJORITY_WINDOWS",
        "threshold": 1e-12,
        "center": [0.0] * dimension,
        "scale": [1.0] * dimension,
        "anchors": [[0.0] * dimension],
        "calibration": "synthetic explicit rejection fixture",
    }
    save_model_artifact(
        model_directory,
        artifact.model,
        config=config,
        labels=artifact.labels,
        test_result=artifact.test_result,
        overwrite=True,
    )
    rejected = diagnose_artifact_signal(
        new_raw_signal, model_directory, sampling_rate_hz=1024,
    )
    assert rejected["status"] == "ABSTAINED_OUT_OF_DISTRIBUTION"
    assert rejected["diagnosis"] is None
    assert rejected["ood"]["rejected"] is True


def test_obt_dataset_trains_saves_reloads_registers_and_diagnoses(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    for label_id, label_name in enumerate(("normal", "fault")):
        folder = dataset / label_name
        folder.mkdir(parents=True)
        for group in range(5):
            _write_signal(folder / f"group-{group}.csv", _signal(label_id, group))

    base = _spec()
    spec = base.model_copy(
        update={
            "classifier": ClassifierConfig(
                algorithm="OBT-LSTSVM",
                backend="numpy",
                parameters={
                    "c1": 0.25,
                    "c2": 0.25,
                    "kernel": "rbf",
                    "kernel_parameter": 4.0,
                    "regularization": 1e-4,
                    "matlab_compat": True,
                },
            ),
            "diagnosis": base.diagnosis.model_copy(
                update={
                    "profile_id": "SYNTHETIC_OBT_PROFILE",
                    "display_name": "Synthetic OBT end-to-end diagnosis",
                }
            ),
        }
    )
    model_directory = tmp_path / "models" / "synthetic-obt-v1"
    training = train_dataset(
        spec, dataset, model_directory, model_id="synthetic-obt-v1"
    )
    assert training.artifact.profile.classifier == "OBT-LSTSVM"
    assert training.artifact.config["classifier"]["backend"] == "numpy"
    assert training.artifact.model.class_order_.tolist() == [0, 1]

    restored = load_model_artifact(model_directory)
    assert type(restored.model).__name__ == "OBTLSTSVM"
    diagnosis = diagnose_artifact_signal(
        _signal(1, 99), model_directory, sampling_rate_hz=1024
    )
    assert diagnosis["status"] == "DIAGNOSED"
    assert diagnosis["diagnosis"] in {"normal", "fault"}

    registry = ModelRegistry(tmp_path / "registry")
    registry.register(model_directory)
    shown = registry.show("synthetic-obt-v1")
    assert shown["classifier"]["algorithm"] == "OBT-LSTSVM"
    registration = registry.as_diagnosis_registry().get("SYNTHETIC_OBT_PROFILE")
    assert registration is not None
    routed = registration.runner(
        _signal(0, 101), registration.model_path, sampling_rate_hz=1024
    )
    assert routed["status"] == "DIAGNOSED"
