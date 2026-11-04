"""Deterministic signal inspection and safe bundled-entropy integration."""

from __future__ import annotations

import numpy as np

from . import entropy_backend as entropy_toolbox
from .schemas import EntropyJob, EntropyResult, IssueCode, SignalInspection, ValidationIssue


class ScientificToolError(ValueError):
    def __init__(self, issue: ValidationIssue):
        self.issue = issue
        super().__init__(f"{issue.code.value}: {issue.message}")


def _as_numeric_vector(signal: object) -> np.ndarray:
    try:
        signal_values = np.asarray(signal)
    except (TypeError, ValueError) as exc:
        raise ScientificToolError(
            ValidationIssue(code=IssueCode.SIGNAL_NOT_NUMERIC, message=str(exc))
        ) from exc
    if signal_values.ndim != 1:
        raise ScientificToolError(
            ValidationIssue(
                code=IssueCode.SIGNAL_NOT_1D,
                message=f"Expected 1-D signal, got shape {signal_values.shape}",
            )
        )
    if signal_values.dtype.kind not in "iuf":
        raise ScientificToolError(
            ValidationIssue(
                code=IssueCode.SIGNAL_NOT_NUMERIC,
                message=f"Expected real numeric samples, got {signal_values.dtype}",
            )
        )
    return signal_values.astype(np.float64, copy=False)


def inspect_signal(signal: object) -> SignalInspection:
    """Inspect one channel without changing any samples."""
    signal_values = _as_numeric_vector(signal)
    issues: list[ValidationIssue] = []
    if signal_values.size == 0:
        issues.append(ValidationIssue(code=IssueCode.EMPTY_SIGNAL, message="Signal is empty"))
    nan_count = int(np.isnan(signal_values).sum())
    pos_inf_count = int(np.isposinf(signal_values).sum())
    neg_inf_count = int(np.isneginf(signal_values).sum())
    if nan_count or pos_inf_count or neg_inf_count:
        issues.append(
            ValidationIssue(
                code=IssueCode.NONFINITE_INPUT,
                message=(
                    "Signal contains NaN or infinity; no automatic imputation "
                    "is performed"
                ),
            )
        )
    finite_values = signal_values[np.isfinite(signal_values)]
    is_constant = bool(
        finite_values.size > 0 and np.all(finite_values == finite_values[0])
    )
    if is_constant:
        issues.append(
            ValidationIssue(
                code=IssueCode.CONSTANT_SIGNAL,
                message="Constant channel cannot enter entropy analysis",
            )
        )
    return SignalInspection(
        sample_count=int(signal_values.size),
        nan_count=nan_count,
        pos_inf_count=pos_inf_count,
        neg_inf_count=neg_inf_count,
        mean=float(np.mean(finite_values)) if finite_values.size else None,
        std=float(np.std(finite_values)) if finite_values.size else None,
        minimum=float(np.min(finite_values)) if finite_values.size else None,
        maximum=float(np.max(finite_values)) if finite_values.size else None,
        is_constant=is_constant,
        valid_for_entropy=not issues,
        issues=issues,
    )


def _effective_counts(n: int, job: EntropyJob) -> list[int]:
    params = job.parameters
    max_scale = getattr(params, "scale", None) or getattr(params, "kmax")
    m, tau = params.m, params.tau
    counts = []
    for scale in range(1, max_scale + 1):
        # Composite offsets can be one point shorter than the first offset.
        j = (
            (n - scale + 1) // scale
            if job.method in {"RCMDE", "RCMFuDE", "IMFRDE"}
            else n // scale
        )
        counts.append(max(0, j - (m - 1) * tau))
    return counts


def _validate_scales(arr: np.ndarray, job: EntropyJob, counts: list[int]) -> None:
    del arr, job
    for scale, count in enumerate(counts, start=1):
        if count < 1:
            raise ScientificToolError(ValidationIssue(
                code=IssueCode.INSUFFICIENT_EFFECTIVE_WINDOWS,
                message=f"No valid embedding windows at scale {scale}",
                scale=scale, retryable=True,
            ))


def compute_entropy(signal: object, job: EntropyJob) -> EntropyResult:
    """Validate one signal/job, then preserve toolbox output exactly, including nonfinite values."""
    signal_values = _as_numeric_vector(signal)
    inspection = inspect_signal(signal_values)
    if not inspection.valid_for_entropy:
        raise ScientificToolError(inspection.issues[0])
    effective_window_counts = _effective_counts(signal_values.size, job)
    _validate_scales(signal_values, job, effective_window_counts)
    parameters = job.parameters.model_dump()
    try:
        output = entropy_toolbox.compute(job.method, signal_values, parameters)
    except Exception as exc:
        raise ScientificToolError(ValidationIssue(
            code=IssueCode.ENTROPY_EXECUTION_FAILED,
            message=f"{type(exc).__name__}: {exc}", retryable=False,
        )) from exc
    if output.ndim != 1 or output.size != len(effective_window_counts):
        raise ScientificToolError(ValidationIssue(
            code=IssueCode.OUTPUT_LENGTH_MISMATCH,
            message=(
                f"Expected {len(effective_window_counts)} values, "
                f"got shape {output.shape}"
            ),
        ))
    values = output.astype(np.float64, copy=False)
    finite_mask = np.isfinite(values)
    issues = []
    for scale in np.flatnonzero(~finite_mask) + 1:
        issues.append(
            ValidationIssue(
                code=IssueCode.NONFINITE_OUTPUT,
                message="Toolbox returned a nonfinite value; original output retained",
                scale=int(scale),
                retryable=True,
            )
        )
    return EntropyResult(
        job_id=job.job_id,
        method=job.method,
        signal_length=signal_values.size,
        parameters=parameters,
        values=values.tolist(),
        finite_mask=finite_mask.tolist(),
        effective_window_counts=effective_window_counts,
        issues=issues,
    )
