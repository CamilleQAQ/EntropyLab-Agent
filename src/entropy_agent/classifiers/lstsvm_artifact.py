"""Versioned JSON + numeric NPZ state for the independent OvO LSTSVM."""

import json
from itertools import combinations
from pathlib import Path

import numpy as np

from .bt_lstsvm import BinaryLSTSVM
from .lstsvm import LSTSVM


FORMAT = "entropylab.ovo-lstsvm"
MODE = "ovo_existing_binary_raw_residual_v1"


def save_lstsvm_artifact(path, model, *, metadata=None):
    if type(model) is not LSTSVM or not hasattr(model, "models_"):
        raise ValueError("only fitted LSTSVM models may be saved")
    manifest = {
        "format": FORMAT, "format_version": 1, "algorithm": "LSTSVM",
        "backend": "numpy", "mode": MODE,
        "feature_dimension": model.n_features_in_,
        "class_order": model.classes_.tolist(), "parameters": model.parameters,
        "pairs": [list(pair) for pair in model.pairs_], "metadata": dict(metadata or {}),
    }
    arrays = {"manifest": np.asarray(json.dumps(manifest, allow_nan=False))}
    for index, node in enumerate(model.models_):
        for name in ("plane1", "plane2"):
            arrays[f"{name}_{index}"] = getattr(node, "_" + name)
        if node._anchors is not None:
            arrays[f"anchors_{index}"] = node._anchors
    if any(not np.isfinite(value).all() for key, value in arrays.items() if key != "manifest"):
        raise ValueError("model contains nonfinite state")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
    return manifest


def load_lstsvm_artifact(path):
    try:
        with np.load(path, allow_pickle=False) as archive:
            manifest = json.loads(str(archive["manifest"].item()))
            if not isinstance(manifest, dict):
                raise ValueError("invalid LSTSVM manifest")
            if any(manifest.get(key) != value for key, value in {
                "format": FORMAT, "format_version": 1, "algorithm": "LSTSVM",
                "backend": "numpy", "mode": MODE,
            }.items()):
                raise ValueError("incompatible LSTSVM artifact")
            model = LSTSVM(**manifest["parameters"])
            width = manifest["feature_dimension"]
            classes = np.asarray(manifest["class_order"])
            if (type(width) is not int or width < 1 or classes.ndim != 1 or
                    len(classes) < 2 or not np.issubdtype(classes.dtype, np.integer) or
                    not np.array_equal(classes, np.unique(classes)) or
                    not isinstance(manifest["metadata"], dict)):
                raise ValueError("invalid LSTSVM dimensions/classes/metadata")
            pairs = tuple(combinations(range(len(classes)), 2))
            if manifest["pairs"] != [list(pair) for pair in pairs]:
                raise ValueError("invalid LSTSVM pair structure")
            expected_keys = {"manifest"}
            nodes = []
            for index in range(len(pairs)):
                node = BinaryLSTSVM(**model.parameters)
                node._width = width
                node._anchors = None
                plane_width = width + 1
                if node.kernel == "rbf":
                    key = f"anchors_{index}"
                    expected_keys.add(key)
                    anchors = np.asarray(archive[key], dtype=np.float64)
                    if (anchors.ndim != 2 or anchors.shape[0] < 2 or
                            anchors.shape[1] != width or not np.isfinite(anchors).all()):
                        raise ValueError("invalid LSTSVM anchors")
                    node._anchors = anchors.copy()
                    plane_width = len(anchors) + 1
                for name in ("plane1", "plane2"):
                    key = f"{name}_{index}"
                    expected_keys.add(key)
                    plane = np.asarray(archive[key], dtype=np.float64)
                    if plane.shape != (plane_width,) or not np.isfinite(plane).all():
                        raise ValueError("invalid LSTSVM plane")
                    setattr(node, "_" + name, plane.copy())
                nodes.append(node)
            if set(archive.files) != expected_keys:
                raise ValueError("unexpected LSTSVM artifact arrays")
            model.classes_, model.n_features_in_ = classes.copy(), width
            model.pairs_, model.models_ = pairs, nodes
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load LSTSVM artifact: {exc}") from exc
    return model, manifest
