from pathlib import Path

import pytest
from pydantic import ValidationError

from entropy_agent.diagnosis.registry import DiagnosisProfile, DiagnosisRegistry, MatchStatus


def _profile(profile_id: str = "TEST_PROFILE") -> DiagnosisProfile:
    return DiagnosisProfile(
        profile_id=profile_id,
        display_name="Test diagnosis",
        workflow="TEST_DIAGNOSIS",
        signal_type="ACCELERATION",
        signal_role="SENSOR_A",
        sampling_rate_hz=10_000,
        window_size_samples=1_000,
        evidence_window_count=3,
        feature_method="MDE",
        feature_dimension=20,
        classifier="TEST_CLASSIFIER",
        output_labels={0: "HEALTHY", 1: "FAULT"},
        model_contract_profile="test_contract_v1",
        validation_scope="synthetic_test_only",
    )


def _metadata(**updates):
    values = {
        "diagnostic_profile": "TEST_PROFILE",
        "sampling_rate_hz": 10_000,
        "signal_type": "ACCELERATION",
        "signal_role": "SENSOR_A",
    }
    values.update(updates)
    return values


def test_profile_validates_contract_and_computes_minimum_samples() -> None:
    profile = _profile()
    assert profile.minimum_samples == 1_000
    assert profile.preferred_samples == 3_000
    with pytest.raises(ValidationError):
        _profile("contains-lowercase")
    with pytest.raises(ValidationError, match="at least two"):
        DiagnosisProfile.model_validate(
            {**_profile().model_dump(), "output_labels": {0: "ONLY"}}
        )


def test_registry_matches_complete_metadata_and_describes_catalog(tmp_path: Path) -> None:
    runner = lambda *args, **kwargs: {"status": "ok"}
    registry = DiagnosisRegistry()
    registry.register(_profile(), tmp_path / "model.npz", runner)
    match = registry.match(_metadata())
    assert match.status == MatchStatus.MATCHED
    assert match.registration is not None
    assert match.registration.runner is runner
    assert match.registration.model_path == (tmp_path / "model.npz").resolve()
    assert registry.profile_ids() == ("TEST_PROFILE",)
    assert registry.describe()[0]["model_configured"] is True


def test_registry_reports_missing_mismatch_unknown_and_unavailable() -> None:
    registry = DiagnosisRegistry()
    registry.register(_profile(), None, lambda *args, **kwargs: {})

    missing = registry.match({})
    assert missing.status == MatchStatus.INCOMPLETE
    assert missing.required_metadata == (
        "diagnostic_profile=TEST_PROFILE",
        "sampling_rate_hz=10000",
        "signal_type=ACCELERATION",
        "signal_role=SENSOR_A",
    )

    incompatible = registry.match(_metadata(sampling_rate_hz=20_000))
    assert incompatible.status == MatchStatus.INCOMPATIBLE
    assert "does not match 10000" in incompatible.reasons[0]

    too_short = registry.match(_metadata(), sample_count=999)
    assert too_short.status == MatchStatus.INCOMPATIBLE
    assert "below minimum 1000" in too_short.reasons[0]

    unknown = registry.match(_metadata(diagnostic_profile="UNKNOWN_PROFILE"))
    assert unknown.status == MatchStatus.NOT_FOUND

    unavailable = registry.match(_metadata())
    assert unavailable.status == MatchStatus.MODEL_UNAVAILABLE


def test_registry_rejects_duplicate_profile() -> None:
    registry = DiagnosisRegistry()
    registry.register(_profile(), None, lambda *args, **kwargs: {})
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_profile(), None, lambda *args, **kwargs: {})


def test_multiple_profiles_require_explicit_selection() -> None:
    registry = DiagnosisRegistry()
    registry.register(_profile("PROFILE_A"), None, lambda *args, **kwargs: {})
    registry.register(_profile("PROFILE_B"), None, lambda *args, **kwargs: {})
    match = registry.match({})
    assert match.status == MatchStatus.INCOMPLETE
    assert match.required_metadata == ("diagnostic_profile=PROFILE_A|PROFILE_B",)
