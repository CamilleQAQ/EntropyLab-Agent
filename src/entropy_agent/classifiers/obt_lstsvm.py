"""Optimized binary-tree least-squares twin SVM.

This is the project-integrated form of the user-supplied Python migration of
the original MATLAB OBT-LSTWSVM code.  The source's ``LSTWSVM_DAG.m`` contains
no sample-weight matrix: mathematically each node is an LSTSVM/LS-TWSVM, not a
weighted LSTWSVM.

``matlab_compat=True`` preserves two historical numerical details needed by
the frozen MATLAB reference: the label-code contribution to inter-class
distances and the leading zero row inserted into each binary class.  Unlike
the original driver, class ordering is deliberately computed inside ``fit``
from training rows only; prediction cannot change the learned order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.linalg import LinAlgError, solve
from scipy.spatial.distance import cdist, pdist


def _features(value: object, *, width: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError("features must be a nonempty two-dimensional matrix")
    if width is not None and array.shape[1] != width:
        raise ValueError("feature dimension does not match training data")
    if not np.isfinite(array).all():
        raise ValueError("nonfinite features are not accepted or imputed")
    return array


def _validated_xy(x: object, y: object) -> tuple[np.ndarray, np.ndarray]:
    features = _features(x)
    labels = np.asarray(y)
    if labels.ndim != 1 or len(labels) != len(features):
        raise ValueError("labels must contain one value per feature row")
    if labels.dtype.kind not in "iu":
        raise ValueError("training labels must be integers, not floats or strings")
    if len(np.unique(labels)) < 2:
        raise ValueError("training requires at least two classes")
    return features, labels


def _linear_solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve the MATLAB-backslash-like systems with a least-squares fallback."""
    try:
        return solve(matrix, rhs, assume_a="sym")
    except (LinAlgError, ValueError):
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]


def _kernel(
    left: np.ndarray, right: np.ndarray, kind: str, parameter: float
) -> np.ndarray:
    if kind == "linear":
        return left @ right.T
    if kind == "rbf":
        # MATLAB kernelfun.m: exp(-||x-y||^2 / (2 * pars(1))).
        return np.exp(-cdist(left, right, metric="sqeuclidean") / (2.0 * parameter))
    raise ValueError("kernel must be 'linear' or 'rbf'")


def class_distance_matrices(
    x: object,
    y: object,
    *,
    classes: object | None = None,
    matlab_compat: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean intra/inter-class distances from the supplied rows only."""
    features, labels = _validated_xy(x, y)
    class_array = np.unique(labels) if classes is None else np.asarray(classes)
    if class_array.ndim != 1 or len(class_array) < 2:
        raise ValueError("at least two classes are required")
    if len(np.unique(class_array)) != len(class_array):
        raise ValueError("classes must be unique")

    groups: list[np.ndarray] = []
    intra = np.empty(len(class_array), dtype=np.float64)
    for index, label in enumerate(class_array):
        group = features[labels == label]
        if len(group) < 2:
            raise ValueError(f"class {label!r} needs at least two training samples")
        groups.append(
            np.column_stack((group, np.full(len(group), index + 1.0)))
            if matlab_compat
            else group
        )
        # Equal to MATLAB sum(D)/(n*(n-1)); the diagonal contributes zero.
        intra[index] = float(np.mean(pdist(group, metric="euclidean")))

    inter = np.zeros((len(class_array), len(class_array)), dtype=np.float64)
    for left in range(len(class_array) - 1):
        for right in range(left + 1, len(class_array)):
            value = float(
                np.mean(cdist(groups[left], groups[right], metric="euclidean"))
            )
            inter[left, right] = value
            inter[right, left] = value
    return intra, inter, class_array


def class_separability(intra: object, inter: object) -> np.ndarray:
    """Implement ``fun_DivisibilityMeasure_Sum.m`` without alteration."""
    intra_array = np.asarray(intra, dtype=np.float64)
    inter_array = np.asarray(inter, dtype=np.float64)
    if intra_array.ndim != 1 or inter_array.shape != (
        len(intra_array), len(intra_array)
    ):
        raise ValueError("intra and inter have incompatible shapes")
    denominator = intra_array[:, None] + intra_array[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        result = inter_array + 1.0 / denominator
    np.fill_diagonal(result, 0.0)
    return result


def optimized_class_order(
    x: object, y: object, *, matlab_compat: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort classes by descending summed separability, using these rows only."""
    intra, inter, classes = class_distance_matrices(
        x, y, matlab_compat=matlab_compat
    )
    scores = class_separability(intra, inter).sum(axis=1)
    indices = np.argsort(-scores, kind="stable")
    return classes[indices], scores, intra, inter


@dataclass
class OBTBinaryLSTSVM:
    """One binary LSTSVM node migrated from ``LSTWSVM_DAG.m``."""

    c1: float = 1.0
    c2: float = 1.0
    kernel: str = "rbf"
    kernel_parameter: float = 1.0
    regularization: float = 1e-4
    matlab_compat: bool = True

    def fit(self, positive: object, negative: object) -> "OBTBinaryLSTSVM":
        a = _features(positive)
        b = _features(negative, width=a.shape[1])
        if not all(
            np.isfinite(value) and value > 0
            for value in (
                self.c1,
                self.c2,
                self.kernel_parameter,
                self.regularization,
            )
        ):
            raise ValueError("c1, c2, kernel_parameter and regularization must be positive")
        if self.kernel not in {"linear", "rbf"}:
            raise ValueError("kernel must be 'linear' or 'rbf'")

        if self.matlab_compat:
            zero = np.zeros((1, a.shape[1]), dtype=np.float64)
            a = np.vstack((zero, a))
            b = np.vstack((zero, b))

        ones_a = np.ones((len(a), 1))
        ones_b = np.ones((len(b), 1))
        epsilon = float(self.regularization)
        self.reference_ = None

        if self.kernel == "linear":
            g = np.hstack((a, ones_a))
            h = np.hstack((b, ones_b))
            gtg, hth = g.T @ g, h.T @ h
            identity = np.eye(g.shape[1])
            plane1 = -_linear_solve(
                gtg / self.c1 + hth + epsilon * identity, h.T @ ones_b
            )
            plane2 = _linear_solve(
                hth / self.c2 + gtg + epsilon * identity, g.T @ ones_a
            )
        else:
            reference = np.vstack((a, b))
            g = np.hstack(
                ((_kernel(a, reference, self.kernel, self.kernel_parameter)), ones_a)
            )
            h = np.hstack(
                ((_kernel(b, reference, self.kernel, self.kernel_parameter)), ones_b)
            )
            self.reference_ = reference
            if len(a) == len(b):
                gtg, hth = g.T @ g, h.T @ h
                identity = np.eye(g.shape[1])
                plane1 = -_linear_solve(
                    gtg / self.c1 + hth + epsilon * identity, h.T @ ones_b
                )
                plane2 = _linear_solve(
                    hth / self.c2 + gtg + epsilon * identity, g.T @ ones_a
                )
            elif len(a) < len(b):
                y_matrix = (
                    np.eye(h.shape[1])
                    - h.T
                    @ _linear_solve(epsilon * np.eye(len(h)) + h @ h.T, h)
                ) / epsilon
                y11 = g @ y_matrix @ g.T
                correction1 = y_matrix - y_matrix @ g.T @ _linear_solve(
                    self.c1 * np.eye(len(g)) + y11, g @ y_matrix
                )
                correction2 = y_matrix - y_matrix @ g.T @ _linear_solve(
                    np.eye(len(g)) / self.c2 + y11, g @ y_matrix
                )
                plane1 = -correction1 @ h.T @ ones_b
                plane2 = self.c2 * correction2 @ g.T @ ones_a
            else:
                z_matrix = (
                    np.eye(g.shape[1])
                    - g.T
                    @ _linear_solve(epsilon * np.eye(len(g)) + g @ g.T, g)
                ) / epsilon
                z11 = h @ z_matrix @ h.T
                correction1 = z_matrix - z_matrix @ h.T @ _linear_solve(
                    np.eye(len(h)) / self.c1 + z11, h @ z_matrix
                )
                correction2 = z_matrix - z_matrix @ h.T @ _linear_solve(
                    self.c2 * np.eye(len(h)) + z11, h @ z_matrix
                )
                plane1 = -self.c1 * correction1 @ h.T @ ones_b
                plane2 = correction2 @ g.T @ ones_a

        self.w1_ = plane1[:-1, 0]
        self.b1_ = float(plane1[-1, 0])
        self.w2_ = plane2[:-1, 0]
        self.b2_ = float(plane2[-1, 0])
        self.n_features_in_ = a.shape[1]
        return self

    def decision_distances(self, x: object) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "w1_"):
            raise ValueError("model has not been fitted")
        features = _features(x, width=self.n_features_in_)
        transformed = (
            features
            if self.kernel == "linear"
            else _kernel(
                features, self.reference_, self.kernel, self.kernel_parameter
            )
        )
        return (
            np.abs(transformed @ self.w1_ + self.b1_),
            np.abs(transformed @ self.w2_ + self.b2_),
        )

    def predict(self, x: object) -> np.ndarray:
        positive, negative = self.decision_distances(x)
        return np.where(positive <= negative, 1, -1).astype(np.int8)


class OBTLSTSVM:
    """Multiclass optimized binary tree whose order is learned during fit."""

    def __init__(
        self,
        *,
        c1: float = 1.0,
        c2: float = 1.0,
        kernel: str = "rbf",
        kernel_parameter: float = 1.0,
        regularization: float = 1e-4,
        matlab_compat: bool = True,
    ) -> None:
        self.parameters: dict[str, Any] = {
            "c1": float(c1),
            "c2": float(c2),
            "kernel": kernel,
            "kernel_parameter": float(kernel_parameter),
            "regularization": float(regularization),
            "matlab_compat": bool(matlab_compat),
        }
        if kernel not in {"linear", "rbf"}:
            raise ValueError("kernel must be 'linear' or 'rbf'")
        if not all(
            np.isfinite(value) and value > 0
            for value in (c1, c2, kernel_parameter, regularization)
        ):
            raise ValueError(
                "c1, c2, kernel_parameter and regularization must be finite and positive"
            )

    def fit(self, x: object, y: object) -> "OBTLSTSVM":
        features, labels = _validated_xy(x, y)
        order, scores, intra, inter = optimized_class_order(
            features, labels, matlab_compat=self.parameters["matlab_compat"]
        )
        self.classes_ = np.unique(labels)
        self.class_order_ = order
        self.separability_scores_ = scores
        self.intra_class_distances_ = intra
        self.inter_class_distances_ = inter
        self.separability_matrix_ = class_separability(intra, inter)
        self.n_features_in_ = int(features.shape[1])
        self.nodes_: list[OBTBinaryLSTSVM] = []

        for index, label in enumerate(order[:-1]):
            positive = features[labels == label]
            negative = features[np.isin(labels, order[index + 1 :])]
            self.nodes_.append(
                OBTBinaryLSTSVM(**self.parameters).fit(positive, negative)
            )
        return self

    def predict(self, x: object) -> np.ndarray:
        if not hasattr(self, "nodes_"):
            raise ValueError("model has not been fitted")
        features = _features(x, width=self.n_features_in_)
        result = np.empty(len(features), dtype=self.class_order_.dtype)
        active = np.arange(len(features))
        for label, node in zip(self.class_order_[:-1], self.nodes_):
            if not len(active):
                break
            decisions = node.predict(features[active])
            result[active[decisions == 1]] = label
            active = active[decisions == -1]
        result[active] = self.class_order_[-1]
        return result

    def save(
        self, path: str | Path, *, metadata: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        from .obt_artifact import save_obt_artifact

        return save_obt_artifact(Path(path), self, metadata=metadata)

    @classmethod
    def load(cls, path: str | Path) -> "OBTLSTSVM":
        from .obt_artifact import load_obt_artifact

        return load_obt_artifact(Path(path))[0]


__all__ = [
    "OBTLSTSVM",
    "OBTBinaryLSTSVM",
    "class_distance_matrices",
    "class_separability",
    "optimized_class_order",
]
