"""Strict, serializable contracts for the first deterministic tool slice."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .methods import AgentMethod, METHOD_FAMILIES


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class IssueCode(str, Enum):
    CSV_PARSE_FAILED = "CSV_PARSE_FAILED"
    CHANNEL_NOT_FOUND = "CHANNEL_NOT_FOUND"
    WINDOW_EXCEEDS_SIGNAL = "WINDOW_EXCEEDS_SIGNAL"
    EMPTY_SIGNAL = "EMPTY_SIGNAL"
    SIGNAL_NOT_1D = "SIGNAL_NOT_1D"
    SIGNAL_NOT_NUMERIC = "SIGNAL_NOT_NUMERIC"
    NONFINITE_INPUT = "NONFINITE_INPUT"
    CONSTANT_SIGNAL = "CONSTANT_SIGNAL"
    PARAMETER_UNSAFE = "PARAMETER_UNSAFE"
    INSUFFICIENT_EFFECTIVE_WINDOWS = "INSUFFICIENT_EFFECTIVE_WINDOWS"
    INSUFFICIENT_EXTREMA = "INSUFFICIENT_EXTREMA"
    ENTROPY_EXECUTION_FAILED = "ENTROPY_EXECUTION_FAILED"
    OUTPUT_LENGTH_MISMATCH = "OUTPUT_LENGTH_MISMATCH"
    RESULT_MASK_MISMATCH = "RESULT_MASK_MISMATCH"
    RESULT_METADATA_MISMATCH = "RESULT_METADATA_MISMATCH"
    NONFINITE_OUTPUT = "NONFINITE_OUTPUT"
    RETRY_NOT_JUSTIFIED = "RETRY_NOT_JUSTIFIED"
    RETRY_BUDGET_EXCEEDED = "RETRY_BUDGET_EXCEEDED"
    STALE_JOB_REVISION = "STALE_JOB_REVISION"
    DUPLICATE_ATTEMPT = "DUPLICATE_ATTEMPT"
    SIGNAL_MISMATCH = "SIGNAL_MISMATCH"
    PLANNER_FAILED = "PLANNER_FAILED"


class ValidationIssue(StrictModel):
    code: IssueCode
    message: str
    segment_index: int | None = None
    scale: int | None = None
    retryable: bool = False


class SignalInspection(StrictModel):
    sample_count: int
    nan_count: int
    pos_inf_count: int
    neg_inf_count: int
    mean: float | None
    std: float | None
    minimum: float | None
    maximum: float | None
    is_constant: bool
    valid_for_entropy: bool
    issues: list[ValidationIssue] = Field(default_factory=list)


class BaseParameters(StrictModel):
    m: int = Field(ge=2)
    tau: int = Field(ge=1)


class DispersionParameters(BaseParameters):
    nc: int = Field(ge=2)
    scale: int = Field(ge=1)


class FuzzyDispersionParameters(DispersionParameters):
    type_: Literal[0, 1] = 0


class RenyiFuzzyDispersionParameters(FuzzyDispersionParameters):
    q: float = Field(default=1.3, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def reject_undefined_q_one(self) -> "RenyiFuzzyDispersionParameters":
        if self.q == 1:
            raise ValueError("q=1 is undefined in the validated MATLAB formula")
        return self


class PermutationParameters(BaseParameters):
    scale: int = Field(ge=1)


class FuzzyParameters(BaseParameters):
    r: float = Field(gt=0)
    n: float = Field(gt=0)
    scale: int = Field(ge=1)


class TimeShiftEnsembleParameters(BaseParameters):
    nc: int = Field(ge=2)
    kmax: int = Field(ge=1)


Parameters = (
    DispersionParameters
    | FuzzyDispersionParameters
    | RenyiFuzzyDispersionParameters
    | PermutationParameters
    | FuzzyParameters
    | TimeShiftEnsembleParameters
)


class EntropyJob(StrictModel):
    job_id: str = Field(min_length=1)
    revision: int = Field(default=0, ge=0)
    method: AgentMethod
    parameters: Parameters

    @model_validator(mode="after")
    def check_method_parameters(self) -> "EntropyJob":
        expected_by_family = {
            "dispersion": DispersionParameters,
            "permutation": PermutationParameters,
            "fuzzy": FuzzyParameters,
            "fuzzy_dispersion": FuzzyDispersionParameters,
            "time_shift_ensemble": TimeShiftEnsembleParameters,
            "renyi_fuzzy_dispersion": RenyiFuzzyDispersionParameters,
        }
        expected = expected_by_family[METHOD_FAMILIES[self.method]]
        if type(self.parameters) is not expected:
            raise ValueError(f"{self.method} requires {expected.__name__}")
        return self


class EntropyResult(StrictModel):
    job_id: str
    method: AgentMethod
    signal_length: int
    parameters: dict[str, int | float]
    values: list[float]
    finite_mask: list[bool]
    # Conservative geometric count J - (m - 1) * tau, not an estimator's
    # internal match count or probability-vector size.
    effective_window_counts: list[int]
    issues: list[ValidationIssue] = Field(default_factory=list)


class CSVLoadRequest(StrictModel):
    path: str = Field(min_length=1)
    channel: str | int
    has_header: bool
    delimiter: str = ","
    encoding: str = "utf-8-sig"
    skip_rows: int = Field(default=0, ge=0)
    start_sample: int = Field(default=0, ge=0)
    max_samples: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_csv_selection(self) -> "CSVLoadRequest":
        if len(self.delimiter) != 1:
            raise ValueError("delimiter must be exactly one character")
        if isinstance(self.channel, int) and self.channel < 0:
            raise ValueError("channel index must be nonnegative")
        if isinstance(self.channel, str) and not self.has_header:
            raise ValueError("a named channel requires has_header=True")
        return self


class WindowPlan(StrictModel):
    """A window length and an independent step.

    ``step_size_samples`` may be smaller than the window (overlap), equal
    (non-overlap), or larger (sparse sampling).  It is not a contract that the
    step must fit inside the window.
    """

    window_size_samples: int = Field(ge=1)
    step_size_samples: int = Field(ge=1)
    incomplete_window_policy: Literal["DROP"] = "DROP"


class SegmentOutcome(StrictModel):
    segment_index: int
    start_sample: int
    stop_sample: int
    signal_length: int
    method: AgentMethod
    parameters: dict[str, int | float]
    result: EntropyResult | None = None
    issue: ValidationIssue | None = None

    @model_validator(mode="after")
    def require_exclusive_result(self) -> "SegmentOutcome":
        if (self.result is None) == (self.issue is None):
            raise ValueError("exactly one of result or issue is required")
        return self


class EntropyBatchResult(StrictModel):
    job_id: str
    revision: int = Field(ge=0)
    signal_sha256: str = Field(min_length=64, max_length=64)
    method: AgentMethod
    parameters: dict[str, int | float]
    window_plan: WindowPlan
    segments: list[SegmentOutcome]
