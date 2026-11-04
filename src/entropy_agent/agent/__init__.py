"""Deterministic 5-round Agent: candidate ranking, feature cache, memory.

The package is dataset-agnostic.  A caller supplies a ``DatasetSpec`` and its
data root at the application boundary; nothing here imports a concrete dataset,
and no selection rule reads the test partition.
"""

from .cache import FeatureCache, FeatureSet
from .candidates import (
    DEFAULT_CLASSIFIERS,
    DEFAULT_METHODS,
    Candidate,
    ClassifierSpec,
    MethodSpec,
    build_pool,
)
from .loop import TrialResult, run_agent
from .memory import ExperimentMemory
from .profile import DataProfile, build_profile
from .selector import anchor_candidate, propose

__all__ = [
    "ClassifierSpec",
    "Candidate",
    "DEFAULT_CLASSIFIERS",
    "DEFAULT_METHODS",
    "DataProfile",
    "ExperimentMemory",
    "FeatureCache",
    "FeatureSet",
    "MethodSpec",
    "TrialResult",
    "anchor_candidate",
    "build_pool",
    "build_profile",
    "propose",
    "run_agent",
]
