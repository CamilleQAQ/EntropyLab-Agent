"""End-to-end, dataset-independent diagnosis training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from ..analysis import entropy_backend
from ..analysis.feature_pipeline import (
    DEFAULT_FUSION_ENTROPY_CONFIGS,
    FUSION_METHODS,
    WindowSpec,
    extract_record_vector,
    families_for_fusion,
)
from ..analysis.schemas import WindowPlan
from ..analysis.tools import inspect_signal
from ..classifiers.registry import get_classifier_registration
from ..diagnosis.artifact import (
    FORMAT,
    FORMAT_VERSION,
    ModelArtifact,
    load_model_artifact,
    save_model_artifact,
)
from ..io.csv_windows import window_signal
from ..io.dataset import (
    DatasetRecord,
    DatasetSpec,
    load_dataset_records,
    load_record_signal,
    resolve_dataset_split,
)
from .grouped import classification_metrics


@dataclass(frozen=True)
class TrainingResult:
    artifact: ModelArtifact
    test_result: dict


def _majority(predictions: np.ndarray, class_count: int) -> np.ndarray:
    return np.asarray(
        [
            int(np.argmax(np.bincount(row, minlength=class_count)))
            for row in predictions
        ],
        dtype=np.int64,
    )


def _effective_windows_per_record(spec: DatasetSpec) -> int:
    """Fusion recipes aggregate every source window into one record row."""
    return 1 if spec.feature.method in FUSION_METHODS else spec.window.windows_per_record


def _extract_fusion_features(spec: DatasetSpec, records):
    families = families_for_fusion(spec.feature.method)
    max_windows = int(
        spec.feature.parameters.get("max_windows", spec.window.windows_per_record)
    )
    sample_stride = int(
        spec.feature.parameters.get("sample_stride", spec.window.stride)
    )
    if max_windows < 1:
        raise ValueError("fusion max_windows must be positive")
    window = WindowSpec(spec.window.size, sample_stride, max_windows)
    entropy_configs = DEFAULT_FUSION_ENTROPY_CONFIGS if "entropy" in families else ()
    rows: list[np.ndarray] = []
    names: list[str] | None = None
    dimension: int | None = None
    for record in records:
        loaded = load_record_signal(record)
        vector, vector_names, _ = extract_record_vector(
            loaded.values,
            spec.sampling_rate_hz,
            window=window,
            families=families,
            entropy_configs=entropy_configs,
            include_full_signal=True,
        )
        if names is None:
            names = vector_names
            dimension = int(vector.size)
        elif names != vector_names or vector.size != dimension:
            raise ValueError("fusion feature schema changes between records")
        rows.append(vector[None, :])
    if not rows or dimension is None:
        raise ValueError("fusion feature extraction produced no records")
    return (
        records,
        np.asarray(rows, dtype=np.float64),
        np.asarray([record.label for record in records], dtype=np.int64),
        dimension,
    )


def extract_features(spec: DatasetSpec, data_root: Path, *, records=None):
    """Featurize the given records under the dataset's feature contract.

    Public because the Agent's feature cache featurizes an arbitrary record
    subset (for example train+validation only) without opening the test
    partition.
    """
    if records is None:
        records = load_dataset_records(spec, data_root)
    if spec.feature.method in FUSION_METHODS:
        return _extract_fusion_features(spec, records)
    plan = WindowPlan(
        window_size_samples=spec.window.size,
        step_size_samples=spec.window.stride,
    )
    feature_rows = []
    feature_dimension = None

    for record in records:
        loaded = load_record_signal(record)
        windows = window_signal(loaded.values, plan)
        if len(windows) < spec.window.windows_per_record:
            raise ValueError(
                f"record {record.record_id} has {len(windows)} complete windows; "
                f"requires {spec.window.windows_per_record}"
            )

        record_feature_rows = []
        for segment in windows[:spec.window.windows_per_record]:
            inspection = inspect_signal(segment.values)
            if not inspection.valid_for_entropy:
                codes = ", ".join(issue.code.value for issue in inspection.issues)
                raise ValueError(
                    f"record {record.record_id} window {segment.index} failed inspection: {codes}"
                )

            feature_curve = np.asarray(
                entropy_backend.compute(
                    spec.feature.method,
                    segment.values,
                    spec.feature.parameters,
                ),
                dtype=np.float64,
            )
            if (
                feature_curve.ndim != 1
                or feature_curve.size == 0
                or not np.isfinite(feature_curve).all()
            ):
                raise ValueError(f"record {record.record_id} produced invalid features")
            feature_dimension = (
                int(feature_curve.size)
                if feature_dimension is None
                else feature_dimension
            )
            if feature_curve.shape != (feature_dimension,):
                raise ValueError("feature output dimension changes between windows")
            record_feature_rows.append(feature_curve)

        feature_rows.append(record_feature_rows)

    return (
        records,
        np.asarray(feature_rows, dtype=np.float64),
        np.asarray([record.label for record in records], dtype=np.int64),
        int(feature_dimension),
    )


@dataclass(frozen=True)
class PreparedDataset:
    """Features and group-isolated partitions reusable across classifier trials."""

    spec: DatasetSpec
    records: list[DatasetRecord]
    features: np.ndarray
    labels: np.ndarray
    groups: tuple[str, ...]
    feature_dimension: int
    split_by_group: dict[str, str]
    masks: dict[str, np.ndarray]
    feature_seconds: float


def prepare_dataset(
    spec: DatasetSpec, data_root: Path, *, include_test: bool = True,
) -> PreparedDataset:
    """Split metadata first; optionally leave held-out signals unopened."""
    started = perf_counter()
    records = load_dataset_records(spec, data_root)
    split_by_group = resolve_dataset_split(spec, records)
    if not include_test:
        records = [record for record in records if split_by_group[record.group] != "test"]
    records, features, labels, feature_dimension = extract_features(
        spec, data_root, records=records,
    )
    groups = tuple(record.group for record in records)
    split = np.asarray([split_by_group[group] for group in groups])
    masks = {name: split == name for name in ("train", "validation", "test")}
    return PreparedDataset(
        spec=spec,
        records=records,
        features=features,
        labels=labels,
        groups=groups,
        feature_dimension=feature_dimension,
        split_by_group=split_by_group,
        masks=masks,
        feature_seconds=perf_counter() - started,
    )


def fit_prepared(
    prepared: PreparedDataset,
    spec: DatasetSpec,
    *,
    classifier_parameters: dict | None = None,
) -> tuple[object, float]:
    """Fit the registered classifier on a prepared split; return model and seconds."""
    if (
        prepared.spec.dataset_id != spec.dataset_id
        or prepared.spec.sampling_rate_hz != spec.sampling_rate_hz
        or prepared.spec.input != spec.input
        or prepared.spec.layout != spec.layout
        or prepared.spec.preprocessing != spec.preprocessing
        or prepared.spec.feature != spec.feature
        or prepared.spec.window != spec.window
        or prepared.spec.labels != spec.labels
        or prepared.spec.split != spec.split
    ):
        raise ValueError(
            "prepared dataset does not match the requested "
            "data/feature/split contract"
        )
    registration = get_classifier_registration(
        spec.classifier.algorithm, spec.classifier.backend
    )
    if not registration.available:
        raise ValueError(registration.unavailable_reason)
    parameters = (
        dict(spec.classifier.parameters)
        if classifier_parameters is None
        else dict(classifier_parameters)
    )
    model = registration.factory(parameters)
    training_features = prepared.features[prepared.masks["train"]].reshape(
        -1, prepared.feature_dimension
    )
    training_labels = np.repeat(
        prepared.labels[prepared.masks["train"]],
        _effective_windows_per_record(spec),
    )
    started = perf_counter()
    model = registration.fit_model(
        model,
        training_features,
        training_labels,
        tuple(range(len(spec.labels.names))),
    )
    return model, perf_counter() - started


def evaluate_partition(
    prepared: PreparedDataset,
    model: object,
    split_name: str,
    *,
    healthy_class_id: int | None,
) -> tuple[dict, np.ndarray, np.ndarray, float]:
    """Evaluate one prepared partition with the same aggregation as training."""
    if split_name not in prepared.masks:
        raise ValueError(f"unknown split {split_name!r}")
    mask = prepared.masks[split_name]
    class_count = len(prepared.spec.labels.names)
    partition_features = prepared.features[mask]
    started = perf_counter()
    window_predictions = model.predict(
        partition_features.reshape(-1, prepared.feature_dimension)
    ).reshape(partition_features.shape[0], partition_features.shape[1])
    inference_seconds = perf_counter() - started
    record_predictions = _majority(window_predictions, class_count)
    result = classification_metrics(
        prepared.labels[mask],
        record_predictions,
        class_count=class_count,
        healthy_class_id=healthy_class_id,
    )
    result["window_accuracy"] = float(
        np.mean(window_predictions == prepared.labels[mask, None])
    )
    return result, record_predictions, window_predictions, inference_seconds


def _build_ood_contract(
    spec: DatasetSpec,
    training_features: np.ndarray,
    features: np.ndarray,
    validation_mask: np.ndarray,
    feature_dimension: int,
) -> dict:
    if spec.ood.calibration == "DISABLED":
        return {
            "enabled": False,
            "reason": "no OOD calibration was requested by the dataset specification",
        }

    margin = float(spec.ood.margin)
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError("OOD margin must be a finite positive number")
    if training_features.size == 0 or feature_dimension < 1:
        raise ValueError("OOD calibration requires at least one training feature row")

    center = np.median(training_features, axis=0)
    scale = np.std(training_features, axis=0)
    scale = np.where(scale > np.finfo(np.float64).eps, scale, 1.0)
    if (
        not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
    ):
        raise ValueError("training features cannot produce finite OOD statistics")
    standardized_anchors = (training_features - center) / scale
    if not np.isfinite(standardized_anchors).all():
        raise ValueError("training features produced nonfinite OOD anchors")

    validation_features = features[validation_mask].reshape(-1, feature_dimension)
    if len(validation_features) == 0:
        raise ValueError("OOD calibration requires at least one validation row")
    standardized_validation = (validation_features - center) / scale
    if not np.isfinite(standardized_validation).all():
        raise ValueError("validation features produced nonfinite OOD values")
    distances = np.min(
        np.linalg.norm(
            standardized_validation[:, None, :] - standardized_anchors[None, :, :],
            axis=2,
        ),
        axis=1,
    )
    if not np.isfinite(distances).all():
        raise ValueError("validation features produced a nonfinite OOD distance")
    threshold = float(np.max(distances) * margin)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("validation features cannot produce a positive OOD threshold")

    return {
        "enabled": True,
        "distance": "NEAREST_TRAINING_STANDARDIZED_EUCLIDEAN",
        "series_policy": "STRICT_MAJORITY_WINDOWS",
        "threshold": threshold,
        "center": center.tolist(),
        "scale": scale.tolist(),
        "anchors": standardized_anchors.tolist(),
        "calibration": "maximum validation-to-training distance times explicit margin",
        "margin": margin,
        "calibration_rows": int(len(validation_features)),
    }


def train_dataset(
    spec: DatasetSpec,
    data_root: Path,
    model_directory: Path,
    *,
    model_id: str | None = None,
    overwrite: bool = False,
    prepared: PreparedDataset | None = None,
    evaluate_test: bool = True,
) -> TrainingResult:
    """Read, window, featurize, split, train, evaluate, save and reload.

    ``prepared`` lets callers reuse features and the fitted split from the same
    experiment instead of reading the data again for final training. The
    prepared contract is checked by ``fit_prepared`` before anything is fit.
    """
    if spec.feature.method in FUSION_METHODS:
        feature_parameters = dict(spec.feature.parameters)
    else:
        feature_parameters = entropy_backend.validate_config(
            spec.feature.method, spec.feature.parameters
        )
    classifier_registration = get_classifier_registration(
        spec.classifier.algorithm, spec.classifier.backend
    )
    if not classifier_registration.available:
        raise ValueError(classifier_registration.unavailable_reason)
    # Resolve defaults before fitting so the artifact is a complete contract.
    classifier_registration.factory(dict(spec.classifier.parameters))

    if prepared is None:
        prepared = prepare_dataset(spec, data_root, include_test=evaluate_test)
    model, fit_seconds = fit_prepared(prepared, spec)

    validation_metrics, _, _, validation_inference_seconds = evaluate_partition(
        prepared,
        model,
        "validation",
        healthy_class_id=spec.labels.healthy_class_id,
    )
    test_metrics = None
    test_record_predictions = test_window_predictions = np.asarray([])
    test_inference_seconds = 0.0
    if evaluate_test:
        (
            test_metrics, test_record_predictions,
            test_window_predictions, test_inference_seconds,
        ) = evaluate_partition(
            prepared, model, "test", healthy_class_id=spec.labels.healthy_class_id,
        )

    training_features = prepared.features[prepared.masks["train"]].reshape(
        -1, prepared.feature_dimension
    )
    ood_contract = _build_ood_contract(
        spec,
        training_features,
        prepared.features,
        prepared.masks["validation"],
        prepared.feature_dimension,
    )

    resolved_model_id = model_id or Path(model_directory).name
    artifact_config = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "model_id": resolved_model_id,
        "sample_rate_hz": spec.sampling_rate_hz,
        "input": spec.input.model_dump(mode="json"),
        "window": {"size": spec.window.size, "stride": spec.window.stride},
        "preprocessing": {"method": spec.preprocessing},
        "feature": {
            "method": spec.feature.method,
            "parameters": feature_parameters,
            "dimension": prepared.feature_dimension,
            "order": [
                f"{spec.feature.method}[{index}]"
                for index in range(prepared.feature_dimension)
            ],
        },
        "classifier": {
            "algorithm": spec.classifier.algorithm,
            "backend": spec.classifier.backend,
            "parameters": dict(model.parameters),
        },
        "aggregation": {"method": "MAJORITY_VOTE", "tie_break": "LOWEST_CLASS_ID"},
        "diagnosis": spec.diagnosis.model_dump(mode="json"),
        "ood": ood_contract,
        "provenance": {
            "dataset_id": spec.dataset_id,
            "split_seed": spec.split.seed,
            "split_contract": spec.split.model_dump(mode="json"),
            "group_isolated": True,
        },
    }
    test_result = {
        "split_records": {
            name: (None if name == "test" and not evaluate_test else int(mask.sum()))
            for name, mask in prepared.masks.items()
        },
        "split_groups": {
            name: sorted(
                group
                for group, value in prepared.split_by_group.items()
                if value == name
            )
            for name in prepared.masks
        },
        "windows_per_record": _effective_windows_per_record(spec),
        "training_rows": int(len(training_features)),
        "validation": validation_metrics,
        "test": test_metrics,
        "test_evaluation_status": "COMPLETED" if evaluate_test else "DEFERRED",
        # Auditable source of the test metrics: one row per held-out record,
        # produced by this model on this split. No dataset-specific fallback.
        "test_predictions": [
            {
                "record_id": prepared.records[index].record_id,
                "group": prepared.records[index].group,
                "truth": int(prepared.labels[index]),
                "predicted": int(test_record_predictions[position]),
            }
            for position, index in enumerate(np.flatnonzero(prepared.masks["test"]))
            if evaluate_test
        ],
        "timings": {
            "feature_extraction_seconds": prepared.feature_seconds,
            "validation_inference_seconds": validation_inference_seconds,
            "classifier_fit_seconds": fit_seconds,
            "test_inference_seconds": test_inference_seconds,
        },
    }

    artifact = save_model_artifact(
        model_directory,
        model,
        config=artifact_config,
        labels={index: name for index, name in enumerate(spec.labels.names)},
        test_result=test_result,
        overwrite=overwrite,
    )

    restored_artifact = load_model_artifact(model_directory, trusted=True)
    verification_partition = "test" if evaluate_test else "validation"
    verification_features = prepared.features[prepared.masks[verification_partition]].reshape(
        -1, prepared.feature_dimension,
    )
    # Verify serialization on development data when the test is still sealed.
    expected = model.predict(verification_features)
    restored_predictions = restored_artifact.model.predict(verification_features)
    if not np.array_equal(restored_predictions, expected):
        raise ValueError("saved artifact predictions differ from the trained pipeline")

    return TrainingResult(artifact=artifact, test_result=test_result)
