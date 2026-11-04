"""Validated diagnosis profiles and their deterministic runtime registry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isclose
from pathlib import Path
from typing import Any, Callable, Mapping

from pydantic import Field, model_validator

from ..analysis.schemas import StrictModel


DiagnosisRunner = Callable[..., dict]


def _value(metadata: object, name: str) -> Any:
    if isinstance(metadata, Mapping):
        return metadata.get(name)
    return getattr(metadata, name, None)


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


class DiagnosisProfile(StrictModel):
    """The complete, reviewable input and model contract for one classifier."""

    profile_id: str = Field(min_length=1, pattern=r"^[A-Z0-9_]+$")
    display_name: str = Field(min_length=1)
    workflow: str = Field(min_length=1)
    signal_type: str = Field(min_length=1)
    signal_role: str = Field(min_length=1)
    sampling_rate_hz: float = Field(gt=0, allow_inf_nan=False)
    sampling_rate_tolerance_hz: float = Field(default=0, ge=0, allow_inf_nan=False)
    window_size_samples: int = Field(ge=1)
    window_stride_samples: int | None = Field(default=None, ge=1)
    minimum_window_count: int = Field(default=1, ge=1)
    evidence_window_count: int = Field(ge=1)
    feature_method: str = Field(min_length=1)
    feature_dimension: int = Field(ge=1)
    classifier: str = Field(min_length=1)
    output_labels: dict[int, str]
    model_contract_profile: str = Field(min_length=1)
    validation_scope: str = Field(min_length=1)
    experimental: bool = False

    @model_validator(mode="after")
    def validate_labels(self) -> "DiagnosisProfile":
        if len(self.output_labels) < 2:
            raise ValueError("output_labels must contain at least two classes")
        if any(not label.strip() for label in self.output_labels.values()):
            raise ValueError("output label names must be nonempty")
        if self.minimum_window_count > self.evidence_window_count:
            raise ValueError("minimum_window_count cannot exceed evidence_window_count")
        # The step is independent of the window length.  A larger stride is a
        # sparse-window contract, not an invalid one.
        return self

    @property
    def minimum_samples(self) -> int:
        stride = self.window_stride_samples or self.window_size_samples
        return self.window_size_samples + stride * (self.minimum_window_count - 1)

    @property
    def preferred_samples(self) -> int:
        stride = self.window_stride_samples or self.window_size_samples
        return self.window_size_samples + stride * (self.evidence_window_count - 1)

    def missing_requirements(self, metadata: object) -> list[str]:
        missing_requirements: list[str] = []
        if _value(metadata, "sampling_rate_hz") is None:
            missing_requirements.append(
                f"sampling_rate_hz={_number(self.sampling_rate_hz)}"
            )
        signal_type = _value(metadata, "signal_type")
        if signal_type in {None, "UNKNOWN"}:
            missing_requirements.append(f"signal_type={self.signal_type}")
        if _value(metadata, "signal_role") is None:
            missing_requirements.append(f"signal_role={self.signal_role}")
        return missing_requirements

    def mismatch_reasons(self, metadata: object) -> list[str]:
        mismatch_reasons: list[str] = []
        sampling_rate_hz = _value(metadata, "sampling_rate_hz")
        if sampling_rate_hz is not None and not isclose(
            float(sampling_rate_hz),
            self.sampling_rate_hz,
            rel_tol=0,
            abs_tol=self.sampling_rate_tolerance_hz,
        ):
            mismatch_reasons.append(
                f"sampling_rate_hz={_number(float(sampling_rate_hz))} does not match "
                f"{_number(self.sampling_rate_hz)}"
            )
        signal_type = _value(metadata, "signal_type")
        if signal_type not in {None, "UNKNOWN", self.signal_type}:
            mismatch_reasons.append(
                f"signal_type={signal_type} does not match {self.signal_type}"
            )
        signal_role = _value(metadata, "signal_role")
        if signal_role is not None and signal_role != self.signal_role:
            mismatch_reasons.append(
                f"signal_role={signal_role} does not match {self.signal_role}"
            )
        return mismatch_reasons


@dataclass(frozen=True)
class DiagnosisRegistration:
    profile: DiagnosisProfile
    model_path: Path | None
    runner: DiagnosisRunner


class MatchStatus(str, Enum):
    MATCHED = "MATCHED"
    INCOMPLETE = "INCOMPLETE"
    INCOMPATIBLE = "INCOMPATIBLE"
    NOT_FOUND = "NOT_FOUND"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"


@dataclass(frozen=True)
class DiagnosisMatch:
    status: MatchStatus
    registration: DiagnosisRegistration | None = None
    required_metadata: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


class DiagnosisRegistry:
    """In-memory allowlist of validated profiles, model artifacts and runners."""

    def __init__(self) -> None:
        self._registrations: dict[str, DiagnosisRegistration] = {}

    def register(
        self,
        profile: DiagnosisProfile,
        model_path: Path | None,
        runner: DiagnosisRunner,
    ) -> None:
        if profile.profile_id in self._registrations:
            raise ValueError(
                f"diagnosis profile {profile.profile_id!r} is already registered"
            )
        resolved_model_path = (
            Path(model_path).expanduser().resolve()
            if model_path is not None
            else None
        )
        self._registrations[profile.profile_id] = DiagnosisRegistration(
            profile, resolved_model_path, runner
        )

    def profile_ids(self) -> tuple[str, ...]:
        return tuple(self._registrations)

    def profiles(self) -> tuple[DiagnosisProfile, ...]:
        return tuple(item.profile for item in self._registrations.values())

    def get(self, profile_id: str) -> DiagnosisRegistration | None:
        return self._registrations.get(profile_id)

    def match(
        self, metadata: object, *, sample_count: int | None = None
    ) -> DiagnosisMatch:
        requested_profile_id = _value(metadata, "diagnostic_profile")
        if requested_profile_id is None:
            # A profile is requested explicitly or presented as missing metadata;
            # it is never inferred from sampling rate or signal characteristics.
            if not self._registrations:
                return DiagnosisMatch(
                    MatchStatus.NOT_FOUND,
                    reasons=("no diagnosis profiles are registered",),
                )
            if len(self._registrations) == 1:
                registration = next(iter(self._registrations.values()))
                required = (
                    f"diagnostic_profile={registration.profile.profile_id}",
                    *registration.profile.missing_requirements(metadata),
                )
            else:
                choices = "|".join(self.profile_ids())
                registration = None
                required = (f"diagnostic_profile={choices}",)
            return DiagnosisMatch(
                MatchStatus.INCOMPLETE,
                registration=registration,
                required_metadata=tuple(required),
            )

        registration = self.get(str(requested_profile_id))
        if registration is None:
            return DiagnosisMatch(
                MatchStatus.NOT_FOUND,
                reasons=(
                    f"diagnostic profile {requested_profile_id!r} is not registered",
                ),
            )

        mismatch_reasons = registration.profile.mismatch_reasons(metadata)
        if (
            sample_count is not None
            and sample_count < registration.profile.minimum_samples
        ):
            mismatch_reasons.append(
                f"sample_count={sample_count} is below minimum "
                f"{registration.profile.minimum_samples}"
            )
        if mismatch_reasons:
            return DiagnosisMatch(
                MatchStatus.INCOMPATIBLE,
                registration=registration,
                reasons=tuple(mismatch_reasons),
            )

        missing_requirements = registration.profile.missing_requirements(metadata)
        if missing_requirements:
            return DiagnosisMatch(
                MatchStatus.INCOMPLETE,
                registration=registration,
                required_metadata=tuple(missing_requirements),
            )

        if registration.model_path is None:
            return DiagnosisMatch(
                MatchStatus.MODEL_UNAVAILABLE,
                registration=registration,
                reasons=(
                    f"model artifact for {requested_profile_id!r} is not configured",
                ),
            )

        return DiagnosisMatch(MatchStatus.MATCHED, registration=registration)

    def describe(self) -> list[dict[str, Any]]:
        """Return a serializable catalog without loading model artifacts."""
        return [
            {
                **registration.profile.model_dump(mode="json"),
                "model_configured": registration.model_path is not None,
                "model_path": (
                    str(registration.model_path)
                    if registration.model_path
                    else None
                ),
            }
            for registration in self._registrations.values()
        ]
