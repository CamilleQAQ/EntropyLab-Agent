"""Safe numeric NPZ artifacts for OBT-LSTSVM models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .obt_lstsvm import OBTBinaryLSTSVM, OBTLSTSVM


FORMAT = "entropylab.obt-lstsvm"
FORMAT_VERSION = 1
ALGORITHM = "OBT-LSTSVM"
BACKEND = "numpy.float64"
MODE = "training_only_separability_optimized_binary_tree"


def _json_object(value: Mapping[str, Any] | None) -> dict[str, Any]:
    try:
        restored = json.loads(
            json.dumps(dict(value or {}), ensure_ascii=False, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must be a finite JSON object") from exc
    return restored


def _manifest(model: OBTLSTSVM, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if not hasattr(model, "nodes_"):
        raise ValueError("model has not been fitted")
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "algorithm": ALGORITHM,
        "backend": BACKEND,
        "mode": MODE,
        "dtype": "float64",
        "feature_dimension": int(model.n_features_in_),
        "classes": model.classes_.tolist(),
        "class_order": model.class_order_.tolist(),
        "parameters": model.parameters,
        "metadata": _json_object(metadata),
    }


def save_obt_artifact(
    path: Path,
    model: OBTLSTSVM,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically save JSON metadata and finite numeric model state, no pickle."""
    if type(model) is not OBTLSTSVM:
        raise ValueError("only OBTLSTSVM models may use this artifact format")
    manifest = _manifest(model, metadata)
    arrays: dict[str, np.ndarray] = {
        "manifest": np.asarray(json.dumps(manifest, ensure_ascii=False, allow_nan=False)),
        "separability_scores": np.asarray(model.separability_scores_, dtype=np.float64),
        "intra_class_distances": np.asarray(
            model.intra_class_distances_, dtype=np.float64
        ),
        "inter_class_distances": np.asarray(
            model.inter_class_distances_, dtype=np.float64
        ),
        "separability_matrix": np.asarray(model.separability_matrix_, dtype=np.float64),
    }
    for index, node in enumerate(model.nodes_):
        arrays[f"w1_{index}"] = np.asarray(node.w1_, dtype=np.float64)
        arrays[f"w2_{index}"] = np.asarray(node.w2_, dtype=np.float64)
        arrays[f"b1_{index}"] = np.asarray(node.b1_, dtype=np.float64)
        arrays[f"b2_{index}"] = np.asarray(node.b2_, dtype=np.float64)
        if node.reference_ is not None:
            arrays[f"reference_{index}"] = np.asarray(
                node.reference_, dtype=np.float64
            )
    if not all(np.isfinite(value).all() for key, value in arrays.items() if key != "manifest"):
        raise ValueError("OBT-LSTSVM model contains nonfinite numeric state")

    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact_path.with_name(artifact_path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(artifact_path)
    return manifest


def _validate_manifest(value: object) -> dict[str, Any]:
    required = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "algorithm": ALGORITHM,
        "backend": BACKEND,
        "mode": MODE,
        "dtype": "float64",
    }
    if not isinstance(value, dict) or any(
        value.get(key) != expected for key, expected in required.items()
    ):
        raise ValueError("artifact manifest is incompatible with OBTLSTSVM")
    width = value.get("feature_dimension")
    classes = value.get("classes")
    order = value.get("class_order")
    parameters = value.get("parameters")
    if not isinstance(width, int) or width < 1:
        raise ValueError("artifact feature dimension is invalid")
    if (
        not isinstance(classes, list)
        or len(classes) < 2
        or not isinstance(order, list)
        or len(order) != len(classes)
        or len({json.dumps(item, sort_keys=True) for item in classes}) != len(classes)
        or {json.dumps(item, sort_keys=True) for item in order}
        != {json.dumps(item, sort_keys=True) for item in classes}
    ):
        raise ValueError("artifact classes or optimized order are invalid")
    if not isinstance(parameters, dict) or not isinstance(value.get("metadata"), dict):
        raise ValueError("artifact parameters or metadata are invalid")
    OBTLSTSVM(**parameters)
    return value


def load_obt_artifact(path: Path) -> tuple[OBTLSTSVM, dict[str, Any]]:
    """Load and structurally validate a numeric OBT-LSTSVM artifact."""
    try:
        archive_context = np.load(Path(path), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot open OBT-LSTSVM artifact: {exc}") from exc
    with archive_context as archive:
        try:
            manifest = _validate_manifest(json.loads(str(archive["manifest"].item())))
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("artifact manifest is missing or malformed") from exc
        model = OBTLSTSVM(**manifest["parameters"])
        model.n_features_in_ = manifest["feature_dimension"]
        model.classes_ = np.asarray(manifest["classes"])
        model.class_order_ = np.asarray(manifest["class_order"])
        class_count = len(model.classes_)
        expected_vector = (class_count,)
        expected_matrix = (class_count, class_count)
        try:
            model.separability_scores_ = np.asarray(
                archive["separability_scores"], dtype=np.float64
            )
            model.intra_class_distances_ = np.asarray(
                archive["intra_class_distances"], dtype=np.float64
            )
            model.inter_class_distances_ = np.asarray(
                archive["inter_class_distances"], dtype=np.float64
            )
            model.separability_matrix_ = np.asarray(
                archive["separability_matrix"], dtype=np.float64
            )
        except KeyError as exc:
            raise ValueError("artifact separability state is incomplete") from exc
        if (
            model.separability_scores_.shape != expected_vector
            or model.intra_class_distances_.shape != expected_vector
            or model.inter_class_distances_.shape != expected_matrix
            or model.separability_matrix_.shape != expected_matrix
            or not all(
                np.isfinite(value).all()
                for value in (
                    model.separability_scores_,
                    model.intra_class_distances_,
                    model.inter_class_distances_,
                    model.separability_matrix_,
                )
            )
        ):
            raise ValueError("artifact separability state is invalid")

        model.nodes_ = []
        for index in range(class_count - 1):
            try:
                w1 = np.asarray(archive[f"w1_{index}"], dtype=np.float64)
                w2 = np.asarray(archive[f"w2_{index}"], dtype=np.float64)
                b1 = float(archive[f"b1_{index}"].item())
                b2 = float(archive[f"b2_{index}"].item())
            except KeyError as exc:
                raise ValueError(f"artifact node {index} is incomplete") from exc
            reference = (
                np.asarray(archive[f"reference_{index}"], dtype=np.float64)
                if f"reference_{index}" in archive
                else None
            )
            expected_width = (
                model.n_features_in_
                if model.parameters["kernel"] == "linear"
                else (len(reference) if reference is not None else -1)
            )
            if (
                w1.shape != (expected_width,)
                or w2.shape != (expected_width,)
                or not np.isfinite([*w1, *w2, b1, b2]).all()
                or (
                    model.parameters["kernel"] == "rbf"
                    and (
                        reference is None
                        or reference.ndim != 2
                        or reference.shape[1] != model.n_features_in_
                        or not np.isfinite(reference).all()
                    )
                )
            ):
                raise ValueError(f"artifact node {index} has invalid numeric state")
            node = OBTBinaryLSTSVM(**model.parameters)
            node.w1_, node.w2_ = w1, w2
            node.b1_, node.b2_ = b1, b2
            node.reference_ = reference
            node.n_features_in_ = model.n_features_in_
            model.nodes_.append(node)
    return model, manifest


__all__ = ["load_obt_artifact", "save_obt_artifact"]
