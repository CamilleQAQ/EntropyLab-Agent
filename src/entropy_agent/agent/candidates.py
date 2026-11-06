"""Deterministic, dataset-agnostic candidate pool for the v1 selector.

The pool lists (feature method, its parameters) x (classifier, its parameters).
Feature methods are the eight entropy methods plus the deterministic
FUSION_* time/frequency/envelope families: a dataset whose class signal is
energy-based (see the gear dataset) cannot be served by entropy features alone.
Nothing here knows about a dataset: the same pool serves every injected
``DatasetSpec``.  Parameter values are explicit rather than implicit defaults so
that the pool is reproducible across library versions and so that the paper
anchor (TSMEFuDE + BT-LSTSVM with ``c1=c2=bandwidth=1``) is inside the pool.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from ..analysis.methods import FEATURE_FAMILIES


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


@dataclass(frozen=True)
class MethodSpec:
    method: str
    parameters: dict

    @property
    def family(self) -> str:
        return FEATURE_FAMILIES.get(self.method, "unknown")


@dataclass(frozen=True)
class ClassifierSpec:
    algorithm: str
    backend: str
    parameters: dict


@dataclass(frozen=True)
class Candidate:
    """One concrete (method, classifier) trial the Agent can run."""

    method: MethodSpec
    classifier: ClassifierSpec

    @property
    def candidate_id(self) -> str:
        payload = {
            "method": self.method.method,
            "parameters": self.method.parameters,
            "classifier": self.classifier.algorithm,
            "backend": self.classifier.backend,
            "classifier_parameters": self.classifier.parameters,
        }
        return f"{self.method.method}+{self.classifier.algorithm}@{_digest(payload)}"

    @property
    def family(self) -> str:
        return self.method.family

    @property
    def label(self) -> str:
        return f"{self.method.method} + {self.classifier.algorithm}"

    def describe(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "method": self.method.method,
            "family": self.family,
            "method_parameters": dict(self.method.parameters),
            "classifier": self.classifier.algorithm,
            "classifier_backend": self.classifier.backend,
            "classifier_parameters": dict(self.classifier.parameters),
        }


# Method defaults follow the authors' MATLAB drivers and the two anchor configs.
# Every method has its own parameter names; a shared dictionary would be invalid.
DEFAULT_METHODS: tuple[MethodSpec, ...] = (
    MethodSpec("TSMEFuDE", {"m": 2, "nc": 5, "tau": 1, "kmax": 20}),
    MethodSpec("RCMFuDE", {"m": 2, "nc": 5, "tau": 1, "scale": 20, "type_": 0}),
    MethodSpec("MFE", {"m": 2, "r": 0.15, "n": 2, "tau": 1, "scale": 20}),
    MethodSpec("MFuDE", {"m": 2, "nc": 5, "tau": 1, "scale": 20, "type_": 0}),
    MethodSpec("IMFRDE", {"m": 2, "nc": 5, "tau": 1, "scale": 20, "type_": 0, "q": 2}),
    MethodSpec("RCMDE", {"m": 2, "nc": 5, "tau": 1, "scale": 20}),
    MethodSpec("MDE", {"m": 2, "nc": 5, "tau": 1, "scale": 20}),
    MethodSpec("MPE", {"m": 3, "tau": 1, "scale": 20}),
    # Deterministic time/frequency/envelope descriptors.  They are orders of
    # magnitude cheaper than an entropy extraction (milliseconds per window) and
    # they keep the amplitude/energy information that entropy normalises away.
    MethodSpec("FUSION_TIME", {}),
    MethodSpec("FUSION_TIME_SPEC", {}),
    MethodSpec("FUSION_TIME_ENVELOPE", {}),
    MethodSpec("FUSION_TIME_ENVELOPE_ENTROPY", {}),
    MethodSpec("FUSION_ALL", {}),
)

DEFAULT_CLASSIFIERS: tuple[ClassifierSpec, ...] = (
    # Anchor parameters: keeps the paper baseline inside the candidate pool.
    ClassifierSpec("BT-LSTSVM", "torch", {"c1": 1.0, "c2": 1.0, "bandwidth": 1.0, "epsilon": 1e-4}),
    ClassifierSpec("SVM-RBF", "scikit-learn", {"C": 1.0, "gamma": "scale"}),
    ClassifierSpec("EXTRA-TREES", "scikit-learn", {
        "n_estimators": 600, "max_depth": None, "min_samples_leaf": 1,
        "class_weight": None, "random_state": 0,
    }),
    ClassifierSpec("LSTSVM", "numpy", {
        "c1": 1.0, "c2": 1.0, "kernel": "rbf", "bandwidth": 1.0, "epsilon": 1e-4,
    }),
    ClassifierSpec("OBT-LSTSVM", "numpy", {
        "c1": 1.0, "c2": 1.0, "kernel": "rbf", "kernel_parameter": 1.0,
        "regularization": 1e-4, "matlab_compat": True,
    }),
    ClassifierSpec("RANDOM-FOREST", "scikit-learn", {
        "n_estimators": 200, "max_depth": None, "min_samples_leaf": 1,
        "class_weight": None, "random_state": 0,
    }),
)


def build_pool(
    methods: tuple[MethodSpec, ...] = DEFAULT_METHODS,
    classifiers: tuple[ClassifierSpec, ...] = DEFAULT_CLASSIFIERS,
) -> tuple[Candidate, ...]:
    """Cartesian product in a stable order (method-major, then classifier)."""
    return tuple(
        Candidate(method=method, classifier=classifier)
        for method in methods
        for classifier in classifiers
    )
