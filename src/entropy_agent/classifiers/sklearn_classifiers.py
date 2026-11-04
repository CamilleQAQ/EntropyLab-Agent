"""Reusable scikit-learn baselines with explicit multiclass persistence.

These dataset-independent adapters are registered in the diagnosis pipeline.

The joblib payload uses pickle internally.  Loading is therefore restricted to
artifacts from a trusted source; the adjacent JSON manifest is validated before
the payload is opened, but it cannot make an untrusted pickle safe.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Mapping

import numpy as np


FORMAT = "entropylab.sklearn-classifier"
FORMAT_VERSION = 1


def _features(value: object, *, width: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError("features must be a nonempty two-dimensional matrix")
    if width is not None and array.shape[1] != width:
        raise ValueError("feature dimension does not match training data")
    if not np.isfinite(array).all():
        raise ValueError("nonfinite features are not accepted or imputed")
    return array


def _labels(value: object, row_count: int) -> np.ndarray:
    labels = np.asarray(value)
    if labels.shape != (row_count,):
        raise ValueError("labels must contain one value per feature row")
    if labels.dtype.kind not in "iu":
        raise ValueError("training labels must be integers, not floats or strings")
    if len(np.unique(labels)) < 2:
        raise ValueError("training requires at least two classes")
    return labels


def _json_object(value: Mapping[str, Any] | None) -> dict[str, Any]:
    try:
        restored = json.loads(
            json.dumps(dict(value or {}), ensure_ascii=False, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must be a finite JSON object") from exc
    if not isinstance(restored, dict):
        raise ValueError("metadata must be a JSON object")
    return restored


class _SklearnClassifier:
    """Shared validation and trusted-artifact persistence."""

    algorithm: ClassVar[str]

    def fit(self, x: object, y: object) -> "_SklearnClassifier":
        features = _features(x)
        labels = _labels(y, len(features))
        self.estimator_ = self._build_estimator()
        self.estimator_.fit(features, labels)
        self.feature_dimension_ = int(features.shape[1])
        self.classes_ = np.asarray(self.estimator_.classes_).copy()
        return self

    def predict(self, x: object) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            raise ValueError("model has not been fitted")
        return np.asarray(
            self.estimator_.predict(_features(x, width=self.feature_dimension_))
        )

    def save(
        self, path: Path, *, metadata: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Save a fitted model and an inspectable sidecar manifest.

        The returned path is a trusted-only joblib artifact.  The sidecar is
        written first and checked before loading, so accidental algorithm or
        feature-schema mismatches fail before deserialization.
        """
        if not hasattr(self, "estimator_"):
            raise ValueError("model has not been fitted")
        try:
            import joblib
            import sklearn
        except ImportError as exc:  # pragma: no cover - dependency message
            raise RuntimeError("scikit-learn and joblib are required") from exc

        artifact_path = Path(path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "algorithm": self.algorithm,
            "backend": "scikit-learn",
            "sklearn_version": sklearn.__version__,
            "feature_dimension": self.feature_dimension_,
            "classes": self.classes_.tolist(),
            "parameters": self.parameters,
            "metadata": _json_object(metadata),
            "trusted_source_required": True,
        }
        manifest_path = artifact_path.with_suffix(artifact_path.suffix + ".json")
        temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary_manifest.replace(manifest_path)
        temporary_artifact = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
        joblib.dump(self.estimator_, temporary_artifact)
        temporary_artifact.replace(artifact_path)
        return manifest

    @classmethod
    def load(cls, path: Path, *, trusted: bool = False) -> "_SklearnClassifier":
        """Load an artifact only after the caller acknowledges pickle trust."""
        if not trusted:
            raise ValueError("trusted=True is required to load a joblib artifact")
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover - dependency message
            raise RuntimeError("scikit-learn and joblib are required") from exc

        artifact_path = Path(path)
        manifest_path = artifact_path.with_suffix(artifact_path.suffix + ".json")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read classifier manifest: {exc}") from exc
        required = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "algorithm": cls.algorithm,
            "backend": "scikit-learn",
            "trusted_source_required": True,
        }
        if not isinstance(manifest, dict) or any(
            manifest.get(name) != expected for name, expected in required.items()
        ):
            raise ValueError(f"artifact is not a compatible {cls.algorithm} model")
        width = manifest.get("feature_dimension")
        parameters = manifest.get("parameters")
        if not isinstance(width, int) or width < 1 or not isinstance(parameters, dict):
            raise ValueError("classifier manifest has an invalid schema")

        model = cls(**parameters)
        model.estimator_ = joblib.load(artifact_path)
        model.feature_dimension_ = width
        model.classes_ = np.asarray(manifest.get("classes"))
        if not np.array_equal(model.classes_, np.asarray(model.estimator_.classes_)):
            raise ValueError("classifier classes do not match the manifest")
        if getattr(model.estimator_, "n_features_in_", None) != width:
            raise ValueError("classifier feature dimension does not match the manifest")
        return model


class RBFSVMClassifier(_SklearnClassifier):
    """RBF-kernel SVM using scikit-learn's native multiclass strategy."""

    algorithm = "SVM-RBF"

    def __init__(
        self,
        *,
        C: float = 1.0,
        gamma: str | float = "scale",
        class_weight: str | dict | None = None,
    ) -> None:
        if not np.isfinite(C) or C <= 0:
            raise ValueError("C must be finite and positive")
        if not (
            gamma in {"scale", "auto"}
            if isinstance(gamma, str)
            else np.isfinite(gamma) and gamma > 0
        ):
            raise ValueError("gamma must be 'scale', 'auto', or a positive number")
        self.parameters = {"C": float(C), "gamma": gamma, "class_weight": class_weight}

    def _build_estimator(self):
        try:
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC
        except ImportError as exc:  # pragma: no cover - dependency message
            raise RuntimeError("scikit-learn is required for RBFSVMClassifier") from exc
        return make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", decision_function_shape="ovr", **self.parameters),
        )


class RandomForestClassifier(_SklearnClassifier):
    """Dataset-independent scikit-learn random-forest classifier."""

    algorithm = "RANDOM-FOREST"

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        class_weight: str | dict | None = None,
        random_state: int | None = 0,
    ) -> None:
        if (
            isinstance(n_estimators, bool)
            or not isinstance(n_estimators, (int, np.integer))
            or n_estimators < 1
        ):
            raise ValueError("n_estimators must be a positive integer")
        if max_depth is not None and (
            isinstance(max_depth, bool)
            or not isinstance(max_depth, (int, np.integer))
            or max_depth < 1
        ):
            raise ValueError("max_depth must be None or a positive integer")
        if (
            isinstance(min_samples_leaf, bool)
            or not isinstance(min_samples_leaf, (int, np.integer))
            or min_samples_leaf < 1
        ):
            raise ValueError("min_samples_leaf must be a positive integer")
        self.parameters = {
            "n_estimators": int(n_estimators),
            "max_depth": max_depth,
            "min_samples_leaf": int(min_samples_leaf),
            "class_weight": class_weight,
            "random_state": random_state,
        }

    def _build_estimator(self):
        try:
            from sklearn.ensemble import RandomForestClassifier as SklearnRandomForest
        except ImportError as exc:  # pragma: no cover - dependency message
            raise RuntimeError("scikit-learn is required for RandomForestClassifier") from exc
        return SklearnRandomForest(**self.parameters)


class ExtraTreesClassifier(_SklearnClassifier):
    """Dataset-independent extremely-randomized trees baseline."""

    algorithm = "EXTRA-TREES"

    def __init__(
        self,
        *,
        n_estimators: int = 600,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        class_weight: str | dict | None = None,
        random_state: int | None = 0,
    ) -> None:
        if (
            isinstance(n_estimators, bool)
            or not isinstance(n_estimators, (int, np.integer))
            or n_estimators < 1
        ):
            raise ValueError("n_estimators must be a positive integer")
        if max_depth is not None and (
            isinstance(max_depth, bool)
            or not isinstance(max_depth, (int, np.integer))
            or max_depth < 1
        ):
            raise ValueError("max_depth must be None or a positive integer")
        if (
            isinstance(min_samples_leaf, bool)
            or not isinstance(min_samples_leaf, (int, np.integer))
            or min_samples_leaf < 1
        ):
            raise ValueError("min_samples_leaf must be a positive integer")
        self.parameters = {
            "n_estimators": int(n_estimators),
            "max_depth": max_depth,
            "min_samples_leaf": int(min_samples_leaf),
            "class_weight": class_weight,
            "random_state": random_state,
        }

    def _build_estimator(self):
        try:
            from sklearn.ensemble import ExtraTreesClassifier as SklearnExtraTrees
        except ImportError as exc:  # pragma: no cover - dependency message
            raise RuntimeError("scikit-learn is required for ExtraTreesClassifier") from exc
        return SklearnExtraTrees(**self.parameters)


__all__ = [
    "ExtraTreesClassifier",
    "RBFSVMClassifier",
    "RandomForestClassifier",
]
