"""Model-governed diagnosis contracts and application composition."""

from .composition import load_diagnosis_registry
from .artifact import ModelArtifact, load_model_artifact, save_model_artifact
from .registry import DiagnosisProfile, DiagnosisRegistry, MatchStatus
from .runner import diagnose_artifact_signal
from .model_registry import ModelRegistry

__all__ = [
    "DiagnosisProfile", "DiagnosisRegistry", "MatchStatus", "ModelArtifact", "ModelRegistry",
    "diagnose_artifact_signal", "load_diagnosis_registry", "load_model_artifact",
    "save_model_artifact",
]
