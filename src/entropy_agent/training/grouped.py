"""Dataset-agnostic grouped training and evaluation utilities.

Dataset adapters supply features, labels, group identifiers and channel names.
This module deliberately contains no dataset names, paths, sampling rates or
label prose.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..classifiers.bt_lstsvm import BTLSTSVM


def _sklearn_metrics():
    """Import scikit-learn only when evaluation metrics are requested."""
    try:
        from sklearn import metrics
    except ImportError as exc:  # optional dependency, not a programming error
        raise RuntimeError(
            "grouped evaluation metrics require the optional 'scikit-learn' "
            "dependency; install entropylab-agent[ml]"
        ) from exc
    return metrics


DEFAULT_MODEL_PARAMETERS = {
    "c1": 1.0,
    "c2": 1.0,
    "bandwidth": 1.0,
    "kernel": "rbf",
    "epsilon": 1e-4,
}

CLASSIFIER_BACKENDS = ("numpy", "torch")


def _build_classifier(backend: str, parameters: dict):
    """Construct one BT-LSTSVM backend without coupling datasets to it."""
    if backend == "numpy":
        return BTLSTSVM(**parameters)
    if backend == "torch":
        kernel = parameters.get("kernel", "rbf")
        if kernel != "rbf":
            raise ValueError("the PyTorch BT-LSTSVM backend currently supports only RBF")
        torch_parameters = {
            key: value for key, value in parameters.items() if key != "kernel"
        }
        try:
            from ..classifiers.torch_bt_lstsvm import TorchBTLSTSVM
        except ImportError as exc:
            raise ValueError("the optional PyTorch backend is not installed") from exc
        return TorchBTLSTSVM(**torch_parameters)
    raise ValueError(f"classifier_backend must be one of {CLASSIFIER_BACKENDS}")


def stratified_group_split(
    labels: np.ndarray,
    groups: Sequence[str],
    *,
    seed: int,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> dict[str, str]:
    """Assign complete, single-label groups to train/validation/test splits."""
    labels = np.asarray(labels, dtype=np.int64)
    groups = tuple(str(group) for group in groups)
    if labels.ndim != 1 or len(labels) != len(groups):
        raise ValueError("labels and groups must be parallel one-dimensional sequences")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a test partition")

    label_by_group: dict[str, int] = {}
    for label, group in zip(labels, groups, strict=True):
        previous = label_by_group.setdefault(group, int(label))
        if previous != int(label):
            raise ValueError(f"group {group!r} contains multiple labels")

    rng = np.random.default_rng(seed)
    split_by_group: dict[str, str] = {}
    for label in sorted(set(label_by_group.values())):
        class_groups = np.asarray(
            sorted(
                group
                for group, group_label in label_by_group.items()
                if group_label == label
            ),
            dtype=object,
        )
        if len(class_groups) < 3:
            raise ValueError(f"class {label} has fewer than three independent groups")

        class_groups = class_groups[rng.permutation(len(class_groups))]
        train_count = max(1, int(train_fraction * len(class_groups)))
        validation_count = max(1, int(validation_fraction * len(class_groups)))
        if train_count + validation_count >= len(class_groups):
            train_count = len(class_groups) - 2
            validation_count = 1

        for position, group in enumerate(class_groups):
            split = (
                "train"
                if position < train_count
                else "validation"
                if position < train_count + validation_count
                else "test"
            )
            split_by_group[str(group)] = split

    return split_by_group


def _majority_vote(window_predictions: np.ndarray, class_count: int) -> np.ndarray:
    return np.asarray(
        [
            int(np.argmax(np.bincount(row, minlength=class_count)))
            for row in np.asarray(window_predictions, dtype=np.int64)
        ],
        dtype=np.int64,
    )


def classification_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
    *,
    class_count: int,
    healthy_class_id: int | None,
) -> dict:
    """Return multiclass metrics and optional healthy/fault collapsed metrics."""
    truth = np.asarray(truth, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    metrics = _sklearn_metrics()
    result = {
        "rows": int(len(truth)),
        "multiclass_accuracy": float(metrics.accuracy_score(truth, predicted)),
        "multiclass_macro_f1": float(
            metrics.f1_score(
                truth,
                predicted,
                labels=range(class_count),
                average="macro",
                zero_division=0,
            )
        ),
        "multiclass_confusion_matrix": metrics.confusion_matrix(
            truth,
            predicted,
            labels=range(class_count),
        ).tolist(),
    }
    if healthy_class_id is not None:
        true_binary = (truth != healthy_class_id).astype(np.int64)
        predicted_binary = (predicted != healthy_class_id).astype(np.int64)
        matrix = metrics.confusion_matrix(true_binary, predicted_binary, labels=(0, 1))
        result.update({
            "binary_accuracy": float(metrics.accuracy_score(true_binary, predicted_binary)),
            "binary_confusion_matrix": matrix.tolist(),
        })
        # A partition that contains only one side of the collapse has no recall
        # to report; emitting NaN here would make the artifact unsaveable.
        if matrix[0].sum() > 0:
            result["healthy_recall"] = float(matrix[0, 0] / matrix[0].sum())
        if matrix[1].sum() > 0:
            result["fault_recall"] = float(matrix[1, 1] / matrix[1].sum())

    return result


def bootstrap_metric_ci(
    truth: np.ndarray,
    predicted: np.ndarray,
    *,
    class_count: int,
    iterations: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict:
    """Percentile bootstrap for accuracy and macro-F1 on one fixed prediction set.

    A held-out partition of a few dozen records cannot resolve a difference of one
    or two records; this reports the interval those records actually support.
    """
    truth = np.asarray(truth, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    if truth.shape != predicted.shape or truth.ndim != 1:
        raise ValueError("truth and predicted must be parallel 1-D arrays")
    if truth.size == 0:
        raise ValueError("bootstrap requires at least one prediction")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    metrics = _sklearn_metrics()
    generator = np.random.default_rng(seed)
    accuracies: list[float] = []
    macro_f1: list[float] = []
    for _ in range(iterations):
        sample = generator.integers(0, truth.size, truth.size)
        resampled_truth = truth[sample]
        resampled_predicted = predicted[sample]
        accuracies.append(
            float(metrics.accuracy_score(resampled_truth, resampled_predicted))
        )
        macro_f1.append(
            float(
                metrics.f1_score(
                    resampled_truth,
                    resampled_predicted,
                    labels=range(class_count),
                    average="macro",
                    zero_division=0,
                )
            )
        )
    tail = (1.0 - confidence) / 2.0
    return {
        "iterations": int(iterations),
        "confidence": float(confidence),
        "rows": int(truth.size),
        "accuracy": {
            "low": float(np.quantile(accuracies, tail)),
            "high": float(np.quantile(accuracies, 1.0 - tail)),
        },
        "macro_f1": {
            "low": float(np.quantile(macro_f1, tail)),
            "high": float(np.quantile(macro_f1, 1.0 - tail)),
        },
    }


def evaluate_channel_candidates(
    features: np.ndarray,
    labels: np.ndarray,
    groups: Sequence[str],
    split_by_group: dict[str, str],
    *,
    channel_names: Sequence[str],
    class_names: Sequence[str],
    healthy_class_id: int | None = 0,
    model_parameters: dict | None = None,
    balance_training_classes: bool = False,
    classifier_backend: str = "numpy",
) -> dict:
    """Train one BT-LSTSVM per channel and select it using validation groups only.

    This is a deliberately specialized evaluation protocol for the BT-LSTSVM
    channel-selection experiments.  It is not the generic algorithm entry point:
    new datasets and arbitrary algorithms use ``DatasetSpec`` +
    ``training.pipeline.train_dataset`` + ``classifiers.registry``.  Keeping the
    two backends explicit here preserves the historical channel protocol instead
    of silently changing its defaults or model family.

    ``features`` has shape ``(records, windows, channels, feature_dimension)``.
    All windows from a record/group remain in the same partition. Metrics and
    channel selection use one majority-voted prediction per record.
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    groups = tuple(str(group) for group in groups)
    channel_names = tuple(str(name) for name in channel_names)
    class_names = tuple(str(name) for name in class_names)
    if features.ndim != 4:
        raise ValueError("features must have record/window/channel/feature dimensions")
    if features.shape[0] != len(labels) or len(labels) != len(groups):
        raise ValueError("feature records, labels and groups have different lengths")
    if features.shape[2] != len(channel_names):
        raise ValueError("feature channel dimension does not match channel names")
    if not np.isfinite(features).all():
        raise ValueError("features contain nonfinite values")
    if not class_names:
        raise ValueError("class names cannot be empty")

    split_names = np.asarray([split_by_group[group] for group in groups])
    masks = {
        name: split_names == name for name in ("train", "validation", "test")
    }
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("group split contains an empty partition")
    for group in set(groups):
        group_splits = set(
            split_names[index]
            for index, value in enumerate(groups)
            if value == group
        )
        if len(group_splits) != 1:
            raise ValueError(f"group leakage detected for {group!r}")

    parameters = dict(DEFAULT_MODEL_PARAMETERS)
    if model_parameters is not None:
        parameters.update(model_parameters)
    class_order = tuple(range(len(class_names)))
    windows_per_record = features.shape[1]
    channel_results = []

    for channel_index, channel_name in enumerate(channel_names):
        training_features = features[masks["train"], :, channel_index, :].reshape(
            -1, features.shape[-1]
        )
        training_labels = np.repeat(labels[masks["train"]], windows_per_record)
        training_rows_before_balance = len(training_labels)

        if balance_training_classes:
            indices_by_class = [
                np.flatnonzero(training_labels == label) for label in class_order
            ]
            target_count = max(len(indices) for indices in indices_by_class)
            balanced_indices = np.concatenate(
                [
                    np.resize(indices, target_count)
                    for indices in indices_by_class
                ]
            )
            training_features = training_features[balanced_indices]
            training_labels = training_labels[balanced_indices]

        model = _build_classifier(classifier_backend, parameters).fit(
            training_features,
            training_labels,
            class_order=class_order,
        )
        channel_result = {"channel": channel_name}

        for split_name in ("validation", "test"):
            split_features = features[masks[split_name], :, channel_index, :]
            window_predictions = model.predict(
                split_features.reshape(-1, features.shape[-1])
            ).reshape(split_features.shape[0], windows_per_record)
            record_predictions = _majority_vote(window_predictions, len(class_names))
            channel_result[split_name] = classification_metrics(
                labels[masks[split_name]],
                record_predictions,
                class_count=len(class_names),
                healthy_class_id=healthy_class_id,
            )
            channel_result[f"{split_name}_window_accuracy"] = float(
                _sklearn_metrics().accuracy_score(
                    np.repeat(labels[masks[split_name]], windows_per_record),
                    window_predictions.reshape(-1),
                )
            )
        channel_results.append(channel_result)

    selected_index = max(
        range(len(channel_results)),
        key=lambda index: (
            channel_results[index]["validation"]["multiclass_macro_f1"],
            channel_results[index]["validation"]["multiclass_accuracy"],
            -index,
        ),
    )
    return {
        "selected_channel": channel_names[selected_index],
        "classifier": "BT-LSTSVM",
        "classifier_backend": classifier_backend,
        "classifier_parameters": parameters,
        "selection_rule": "highest_validation_record_macro_f1_then_accuracy",
        "class_names": list(class_names),
        "split_records": {name: int(mask.sum()) for name, mask in masks.items()},
        "windows_per_record": int(windows_per_record),
        "training_class_balance": (
            "deterministic_minority_oversampling" if balance_training_classes
            else "none"
        ),
        "training_rows_before_balance": int(training_rows_before_balance),
        "training_rows_after_balance": int(len(training_labels)),
        "channels": channel_results,
        "selected_channel_test": channel_results[selected_index]["test"],
    }
