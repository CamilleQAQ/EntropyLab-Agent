"""Safe numeric artifacts for the deployable PyTorch BT-LSTSVM backend.

Only classifier state is handled here. Dataset labels, DiagnosisProfile
contracts, validation evidence, and OOD thresholds are caller-supplied metadata
and remain outside the classifier algorithm.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .torch_bt_lstsvm import TorchBTLSTSVM, TorchBinaryLSTSVM


FORMAT = "entropylab.torch-bt-lstsvm"
FORMAT_VERSION = 1
ALGORITHM = "BT-LSTSVM"
BACKEND = "torch.float64.cpu"
MODE = "label_free_sequential_binary_tree"


def _json_object(value: Mapping[str, Any] | None, *, name: str) -> dict[str, Any]:
    json_object = {} if value is None else dict(value)
    try:
        encoded_json = json.dumps(
            json_object, ensure_ascii=False, allow_nan=False, sort_keys=True
        )
        decoded_json = json.loads(encoded_json)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite JSON object") from exc
    if not isinstance(decoded_json, dict):
        raise ValueError(f"{name} must be a JSON object")
    return decoded_json


def _manifest(
    model: TorchBTLSTSVM, metadata: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Build the portable contract stored beside the numeric model state."""
    if not hasattr(model, "models_"):
        raise ValueError("model has not been fitted")
    if len(model.classes_) < 2 or len(model.models_) != len(model.classes_) - 1:
        raise ValueError("fitted model has an invalid binary-tree structure")
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "algorithm": ALGORITHM,
        "backend": BACKEND,
        "mode": MODE,
        "dtype": "float64",
        "device": "cpu",
        "feature_dimension": int(model._width),
        "class_order": [int(item) for item in model.classes_],
        "parameters": {
            key: float(value) for key, value in model.parameters.items()
        },
        "metadata": _json_object(metadata, name="metadata"),
    }


def save_torch_bt_artifact(
    path: Path,
    model: TorchBTLSTSVM,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically save a fitted deployable-mode tree without pickle."""
    if type(model) is not TorchBTLSTSVM:
        raise ValueError("only TorchBTLSTSVM deployable-mode models may be saved")
    manifest = _manifest(model, metadata)
    artifact_arrays: dict[str, np.ndarray] = {
        "manifest": np.asarray(
            json.dumps(manifest, ensure_ascii=False, allow_nan=False)
        ),
    }
    for index, node in enumerate(model.models_):
        for name in ("_anchors", "_plane1", "_plane2"):
            tensor = getattr(node, name, None)
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"node {index} has no fitted {name[1:]}")
            if tensor.dtype != torch.float64 or tensor.device.type != "cpu":
                raise ValueError(f"node {index} is not a float64 CPU model")
            array = tensor.detach().numpy().copy()
            if not np.isfinite(array).all():
                raise ValueError(f"node {index} contains nonfinite {name[1:]}")
            artifact_arrays[f"{name[1:]}_{index}"] = array

    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = artifact_path.with_name(artifact_path.name + ".tmp")
    with temporary_path.open("wb") as artifact_file:
        np.savez_compressed(artifact_file, **artifact_arrays)
    temporary_path.replace(artifact_path)
    return manifest


def _validate_manifest(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("artifact manifest must be a JSON object")
    required_fields = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "algorithm": ALGORITHM,
        "backend": BACKEND,
        "mode": MODE,
        "dtype": "float64",
        "device": "cpu",
    }
    if any(
        value.get(key) != expected for key, expected in required_fields.items()
    ):
        raise ValueError("artifact manifest is incompatible with TorchBTLSTSVM")

    feature_dimension = value.get("feature_dimension")
    class_order = value.get("class_order")
    parameters = value.get("parameters")
    if not isinstance(feature_dimension, int) or feature_dimension < 1:
        raise ValueError("artifact feature dimension is invalid")
    if (
        not isinstance(class_order, list)
        or len(class_order) < 2
        or not all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in class_order
        )
        or len(set(class_order)) != len(class_order)
    ):
        raise ValueError("artifact class order is invalid")
    if not isinstance(parameters, dict) or set(parameters) != {
        "c1",
        "c2",
        "bandwidth",
        "epsilon",
    }:
        raise ValueError("artifact parameters are invalid")

    # Constructor performs finite-positive validation.
    TorchBTLSTSVM(**parameters)
    if not isinstance(value.get("metadata"), dict):
        raise ValueError("artifact metadata must be a JSON object")
    return value


def load_torch_bt_artifact(path: Path) -> tuple[TorchBTLSTSVM, dict[str, Any]]:
    """Load and structurally validate a numeric PyTorch BT-LSTSVM artifact."""
    try:
        archive_context = np.load(Path(path), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot open BT-LSTSVM artifact: {exc}") from exc
    with archive_context as archive:
        try:
            manifest = _validate_manifest(json.loads(str(archive["manifest"].item())))
        except (KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
            if isinstance(exc, ValueError) and not isinstance(exc, json.JSONDecodeError):
                raise
            raise ValueError("artifact manifest is missing or malformed") from exc
        model = TorchBTLSTSVM(**manifest["parameters"])
        model._width = manifest["feature_dimension"]
        model.classes_ = tuple(manifest["class_order"])
        model.models_ = []
        for index in range(len(model.classes_) - 1):
            try:
                anchors = np.asarray(archive[f"anchors_{index}"], dtype=np.float64)
                plane1 = np.asarray(archive[f"plane1_{index}"], dtype=np.float64)
                plane2 = np.asarray(archive[f"plane2_{index}"], dtype=np.float64)
            except KeyError as exc:
                raise ValueError(f"artifact node {index} is incomplete") from exc

            expected_plane = anchors.shape[0] + 1 if anchors.ndim == 2 else -1
            arrays_are_finite = all(
                np.isfinite(item).all() for item in (anchors, plane1, plane2)
            )
            if (
                anchors.ndim != 2
                or anchors.shape[1] != model._width
                or plane1.shape != (expected_plane,)
                or plane2.shape != (expected_plane,)
                or not arrays_are_finite
            ):
                raise ValueError(f"artifact node {index} has invalid numeric arrays")

            node = TorchBinaryLSTSVM(**model.parameters)
            node._width = model._width
            node._anchors = torch.tensor(
                anchors.copy(), dtype=torch.float64, device="cpu"
            )
            node._plane1 = torch.tensor(plane1.copy(), dtype=torch.float64, device="cpu")
            node._plane2 = torch.tensor(plane2.copy(), dtype=torch.float64, device="cpu")
            model.models_.append(node)

    return model, manifest
