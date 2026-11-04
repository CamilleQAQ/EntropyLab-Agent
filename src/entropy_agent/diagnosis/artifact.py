"""Self-contained, dataset-independent diagnosis pipeline artifacts.

An artifact binds four reviewable files into one contract:

``config.json``      the algorithm, acquisition, feature and OOD contract;
``labels.json``      the class ID -> class name mapping shown to users;
``model.npz`` or ``model.joblib``
                     the fitted classifier state and its manifest;
``test_result.json`` the recorded held-out evaluation.

Only format version 4 is readable by this build.  Class names, the input
selection, acquisition, diagnosis and OOD contracts are all part of the hash
contract, so editing any of them after training fails closed instead of silently
relabelling predictions or changing what the model accepts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from ..analysis import entropy_backend
from ..analysis.feature_pipeline import FUSION_METHODS
from ..classifiers.registry import get_classifier_registration
from .registry import DiagnosisProfile


FORMAT = "entropylab.diagnosis-pipeline"
FORMAT_VERSION = 4
LABEL_CONTRACT_FIELD = "labels_sha256"
METADATA_FILES = ("config.json", "labels.json", "test_result.json")
#: Backwards-compatible alias for callers that imported the old constant.
REQUIRED_FILES = METADATA_FILES


def label_contract_digest(labels: Mapping[int, str]) -> str:
    """Hash the canonical class ID -> name mapping that users see.

    The digest is order-insensitive with respect to the JSON list order (IDs are
    sorted) but sensitive to any ID/name reassignment or rename.
    """
    document = [[int(class_id), str(labels[class_id])] for class_id in sorted(labels)]
    encoded = json.dumps(
        document, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _algorithm_digest(config: dict) -> str:
    """Hash the complete saved inference contract."""
    keys = (
        "feature",
        "classifier",
        "preprocessing",
        "window",
        "sample_rate_hz",
        "aggregation",
        "diagnosis",
        "ood",
        "input",
    )
    contract = {key: config[key] for key in keys}
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: object, *, name: str, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if positive and number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _finite_vector(
    value: object, *, name: str, length: int, positive: bool = False
) -> None:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    for item in value:
        _finite_number(item, name=name, positive=positive)


def _validate_diagnosis_contract(value: object) -> None:
    """Validate the profile fields that loading will consume."""
    if not isinstance(value, dict):
        raise ValueError("invalid diagnosis contract")
    text_fields = (
        "profile_id",
        "display_name",
        "workflow",
        "signal_type",
        "signal_role",
        "validation_scope",
    )
    if any(
        not isinstance(value.get(name), str) or not value[name].strip()
        for name in text_fields
    ):
        raise ValueError("diagnosis contract is missing required text fields")
    for name in ("minimum_window_count", "evidence_window_count"):
        if not _is_integer(value.get(name)) or value[name] < 1:
            raise ValueError(f"diagnosis.{name} must be a positive integer")
    tolerance = _finite_number(
        value.get("sampling_rate_tolerance_hz"),
        name="diagnosis.sampling_rate_tolerance_hz",
    )
    if tolerance < 0:
        raise ValueError("diagnosis.sampling_rate_tolerance_hz cannot be negative")
    if not isinstance(value.get("experimental"), bool):
        raise ValueError("diagnosis.experimental must be a boolean")
    if value["minimum_window_count"] > value["evidence_window_count"]:
        raise ValueError("diagnosis minimum window count exceeds evidence count")


def _json_value(value: Any, *, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite JSON data") from exc


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_config(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("config.json must contain a JSON object")
    format_version = value.get("format_version")
    if (
        value.get("format") != FORMAT
        or not _is_integer(format_version)
        or format_version != FORMAT_VERSION
    ):
        raise ValueError(
            "diagnosis artifact format is incompatible; this build reads "
            f"{FORMAT} format version {FORMAT_VERSION} only "
            "(legacy v1-v3 artifacts are archived and must be retrained)"
        )

    required = {
        "model_id",
        "sample_rate_hz",
        "input",
        "window",
        "preprocessing",
        "feature",
        "classifier",
        "aggregation",
        "diagnosis",
        "ood",
        "provenance",
    }
    if not required.issubset(value):
        raise ValueError(f"config.json is missing fields: {sorted(required - set(value))}")
    _finite_number(value["sample_rate_hz"], name="sample_rate_hz", positive=True)
    if not isinstance(value["input"], dict):
        raise ValueError("invalid input contract")
    if not isinstance(value["model_id"], str) or not value["model_id"].strip():
        raise ValueError("model_id must be a nonempty string")
    _validate_diagnosis_contract(value["diagnosis"])
    if value["preprocessing"] != {"method": "NONE"}:
        raise ValueError("unsupported preprocessing contract")

    labels_digest = value.get(LABEL_CONTRACT_FIELD)
    if not _is_sha256(labels_digest):
        raise ValueError(
            f"config.json is missing a valid {LABEL_CONTRACT_FIELD} label contract"
        )

    window_contract = value["window"]
    if (
        not isinstance(window_contract, dict)
        or not _is_integer(window_contract.get("size"))
        or window_contract["size"] < 1
        or not _is_integer(window_contract.get("stride"))
        or window_contract["stride"] < 1
    ):
        # The step is independent of the window length: sparse windows are a
        # valid contract.  Only the window size bounds the window itself.
        raise ValueError("invalid window contract")

    feature_contract = value["feature"]
    if (
        not isinstance(feature_contract, dict)
        or not isinstance(feature_contract.get("method"), str)
        or not isinstance(feature_contract.get("parameters"), dict)
        or not _is_integer(feature_contract.get("dimension"))
        or feature_contract["dimension"] < 1
        or not isinstance(feature_contract.get("order"), list)
        or len(feature_contract["order"]) != feature_contract["dimension"]
    ):
        raise ValueError("invalid feature contract")

    classifier_contract = value["classifier"]
    if feature_contract["method"] in FUSION_METHODS:
        if "max_windows" in feature_contract["parameters"] and not _is_integer(
            feature_contract["parameters"]["max_windows"]
        ):
            raise ValueError("fusion feature contract has invalid max_windows")
    else:
        entropy_backend.validate_config(
            feature_contract["method"], feature_contract["parameters"]
        )
    if (
        not isinstance(classifier_contract, dict)
        or not isinstance(classifier_contract.get("algorithm"), str)
        or not isinstance(classifier_contract.get("backend"), str)
        or not isinstance(classifier_contract.get("parameters"), dict)
    ):
        raise ValueError("invalid classifier contract")
    try:
        registration = get_classifier_registration(
            classifier_contract["algorithm"], classifier_contract["backend"]
        )
        if not registration.available:
            raise ValueError(registration.unavailable_reason)
        registration.factory(dict(classifier_contract["parameters"]))
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"unsupported classifier contract: {exc}") from exc
    if value["aggregation"] != {
        "method": "MAJORITY_VOTE",
        "tie_break": "LOWEST_CLASS_ID",
    }:
        raise ValueError("unsupported aggregation contract")

    ood_contract = value["ood"]
    if not isinstance(ood_contract, dict) or not isinstance(
        ood_contract.get("enabled"), bool
    ):
        raise ValueError("invalid OOD contract")
    if ood_contract["enabled"]:
        feature_dimension = feature_contract["dimension"]
        if (
            ood_contract.get("distance")
            != "NEAREST_TRAINING_STANDARDIZED_EUCLIDEAN"
            or ood_contract.get("series_policy") != "STRICT_MAJORITY_WINDOWS"
            or not isinstance(ood_contract.get("calibration"), str)
            or not ood_contract["calibration"].strip()
        ):
            raise ValueError("unsupported OOD contract")
        _finite_number(
            ood_contract.get("threshold"),
            name="ood.threshold",
            positive=True,
        )
        _finite_vector(
            ood_contract.get("center"),
            name="ood.center",
            length=feature_dimension,
        )
        _finite_vector(
            ood_contract.get("scale"),
            name="ood.scale",
            length=feature_dimension,
            positive=True,
        )
        anchors = ood_contract.get("anchors")
        if not isinstance(anchors, list) or not anchors:
            raise ValueError("ood.anchors must contain at least one vector")
        for anchor in anchors:
            _finite_vector(
                anchor, name="ood.anchors", length=feature_dimension
            )
        if "margin" in ood_contract:
            _finite_number(
                ood_contract["margin"], name="ood.margin", positive=True
            )
    elif not isinstance(ood_contract.get("reason"), str) or not ood_contract[
        "reason"
    ].strip():
        raise ValueError("disabled OOD contract requires a reason")

    return value


def _validate_labels(value: object) -> dict[int, str]:
    if not isinstance(value, dict) or not isinstance(value.get("labels"), list):
        raise ValueError("labels.json must contain a labels list")

    labels: dict[int, str] = {}
    names: set[str] = set()
    for label_entry in value["labels"]:
        if (
            not isinstance(label_entry, dict)
            or not isinstance(label_entry.get("id"), int)
            or isinstance(label_entry.get("id"), bool)
            or not isinstance(label_entry.get("name"), str)
            or not label_entry["name"].strip()
        ):
            raise ValueError("labels.json contains an invalid label")
        name = label_entry["name"]
        if label_entry["id"] in labels:
            raise ValueError("labels.json contains duplicate IDs")
        if name in names:
            raise ValueError("labels.json contains duplicate class names")
        labels[label_entry["id"]] = name
        names.add(name)

    if set(labels) != set(range(len(labels))) or len(labels) < 2:
        raise ValueError("label IDs must be contiguous from zero")

    return labels


@dataclass(frozen=True)
class ModelArtifact:
    directory: Path
    model: Any
    config: dict[str, Any]
    labels: dict[int, str]
    test_result: dict[str, Any]

    @property
    def profile(self) -> DiagnosisProfile:
        diagnosis_contract = self.config["diagnosis"]
        feature_contract = self.config["feature"]
        return DiagnosisProfile(
            profile_id=diagnosis_contract["profile_id"],
            display_name=diagnosis_contract["display_name"],
            workflow=diagnosis_contract["workflow"],
            signal_type=diagnosis_contract["signal_type"],
            signal_role=diagnosis_contract["signal_role"],
            sampling_rate_hz=float(self.config["sample_rate_hz"]),
            sampling_rate_tolerance_hz=float(
                diagnosis_contract["sampling_rate_tolerance_hz"]
            ),
            window_size_samples=int(self.config["window"]["size"]),
            window_stride_samples=int(
                feature_contract["parameters"].get(
                    "sample_stride", self.config["window"]["stride"]
                )
                if feature_contract.get("method") in FUSION_METHODS
                else self.config["window"]["stride"]
            ),
            minimum_window_count=int(diagnosis_contract["minimum_window_count"]),
            evidence_window_count=int(diagnosis_contract["evidence_window_count"]),
            feature_method=feature_contract["method"],
            feature_dimension=int(feature_contract["dimension"]),
            classifier=self.config["classifier"]["algorithm"],
            output_labels=self.labels,
            model_contract_profile=self.config["model_id"],
            validation_scope=diagnosis_contract["validation_scope"],
            experimental=bool(diagnosis_contract["experimental"]),
        )


def save_model_artifact(
    directory: Path,
    model: Any,
    *,
    config: Mapping[str, Any],
    labels: Mapping[int, str],
    test_result: Mapping[str, Any],
    overwrite: bool = False,
) -> ModelArtifact:
    """Save the classifier and every inference dependency as four reviewable files."""
    directory = Path(directory).expanduser().resolve()

    labels_document = {
        "labels": [
            {"id": int(class_id), "name": str(name)}
            for class_id, name in sorted(labels.items())
        ]
    }
    validated_labels = _validate_labels(_json_value(labels_document, name="labels"))
    labels_digest = label_contract_digest(validated_labels)

    prepared_config = dict(config)
    prepared_config["format"] = FORMAT
    prepared_config["format_version"] = FORMAT_VERSION
    prepared_config[LABEL_CONTRACT_FIELD] = labels_digest
    validated_config = _validate_config(
        _json_value(prepared_config, name="config")
    )
    validated_test_result = _json_value(dict(test_result), name="test_result")

    classifier_contract = validated_config["classifier"]
    registration = get_classifier_registration(
        classifier_contract["algorithm"], classifier_contract["backend"]
    )
    if not registration.matches(model):
        raise ValueError(
            "classifier model type does not match config.json algorithm/backend"
        )

    existing = [
        name
        for name in (*METADATA_FILES, registration.filename)
        if (directory / name).exists()
    ]
    if existing and not overwrite:
        raise ValueError(f"model artifact already exists: {directory}")
    directory.mkdir(parents=True, exist_ok=True)

    registration.save_model(
        directory / registration.filename,
        model,
        {
            "model_id": validated_config["model_id"],
            "pipeline_format": FORMAT,
            "algorithm_contract_sha256": _algorithm_digest(validated_config),
            LABEL_CONTRACT_FIELD: labels_digest,
        },
    )
    _write_json(directory / "config.json", validated_config)
    _write_json(directory / "labels.json", labels_document)
    _write_json(directory / "test_result.json", validated_test_result)

    return load_model_artifact(directory, trusted=True)


def load_model_artifact(
    directory: Path,
    *,
    trusted: bool = False,
) -> ModelArtifact:
    """Load a complete v4 pipeline without importing its training dataset.

    Loading fails closed when ``labels.json`` or any other contract field no
    longer matches the model. Legacy format versions are intentionally rejected;
    their archived files are retained but must be retrained.
    """
    directory = Path(directory).expanduser().resolve()
    missing = [name for name in METADATA_FILES if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"model artifact is incomplete; missing {missing}")

    try:
        config = _validate_config(
            json.loads((directory / "config.json").read_text(encoding="utf-8"))
        )
        labels = _validate_labels(
            json.loads((directory / "labels.json").read_text(encoding="utf-8"))
        )
        test_result = json.loads(
            (directory / "test_result.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read model artifact: {exc}") from exc
    if not isinstance(test_result, dict):
        raise ValueError("test_result.json must contain a JSON object")

    labels_digest = label_contract_digest(labels)
    if config[LABEL_CONTRACT_FIELD] != labels_digest:
        raise ValueError(
            "labels.json does not match the label contract recorded in config.json; "
            "the class ID/name mapping was modified after training"
        )

    feature_contract = config["feature"]
    classifier_contract = config["classifier"]
    registration = get_classifier_registration(
        classifier_contract["algorithm"], classifier_contract["backend"]
    )
    if not registration.available:
        raise ValueError(registration.unavailable_reason)
    if registration.requires_trust and not trusted:
        raise ValueError("trusted=True is required for scikit-learn joblib pipeline artifacts")
    model_path = directory / registration.filename
    if not model_path.is_file():
        raise ValueError(f"model artifact is incomplete; missing {registration.filename}")
    model, manifest = registration.load_model(model_path)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("metadata"), dict):
        raise ValueError("classifier manifest is malformed")

    if manifest.get("feature_dimension") != feature_contract["dimension"]:
        raise ValueError("classifier feature dimension does not match config.json")
    class_order = manifest.get("class_order")
    if (
        not isinstance(class_order, list)
        or len(class_order) != len(labels)
        or set(class_order) != set(labels)
        or any(isinstance(item, bool) for item in class_order)
    ):
        raise ValueError("classifier classes do not match labels.json")
    if manifest.get("parameters") != classifier_contract["parameters"]:
        raise ValueError("classifier parameters do not match config.json")
    if registration.algorithm != "OBT-LSTSVM" and list(class_order) != sorted(
        class_order
    ):
        # BT-LSTSVM, OvO LSTSVM and scikit-learn store ascending classes; a
        # reordered manifest would silently change which branch owns which label.
        raise ValueError(
            "classifier class order is not ascending for this backend"
        )

    expected_metadata = {
        "model_id": config["model_id"],
        "pipeline_format": FORMAT,
        "algorithm_contract_sha256": _algorithm_digest(config),
        LABEL_CONTRACT_FIELD: labels_digest,
    }
    if manifest["metadata"].get(LABEL_CONTRACT_FIELD) != labels_digest:
        raise ValueError(
            "classifier manifest is not bound to labels.json; "
            "the class mapping was modified after training"
        )
    if manifest["metadata"] != expected_metadata:
        raise ValueError("classifier identity does not match config.json")

    artifact = ModelArtifact(directory, model, config, labels, test_result)
    # Build the profile during loading so invalid diagnosis metadata fails early.
    artifact.profile
    return artifact
