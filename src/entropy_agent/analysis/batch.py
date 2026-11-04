"""Window-level isolation: one failing segment does not discard its peers."""

from __future__ import annotations

import hashlib

import numpy as np

from ..io.csv_windows import window_signal
from .schemas import (
    EntropyBatchResult,
    EntropyJob,
    IssueCode,
    SegmentOutcome,
    ValidationIssue,
    WindowPlan,
)
from .tools import ScientificToolError, _as_numeric_vector, compute_entropy


def signal_sha256(signal: object) -> str:
    """Hash the exact float64 samples passed to window-level tools."""
    signal_values = _as_numeric_vector(signal)
    contiguous_values = np.ascontiguousarray(signal_values)
    return hashlib.sha256(memoryview(contiguous_values)).hexdigest()


def compute_entropy_batch(
    signal: object, job: EntropyJob, window_plan: WindowPlan
) -> EntropyBatchResult:
    signal_values = _as_numeric_vector(signal)
    segments = window_signal(signal_values, window_plan)
    parameters = job.parameters.model_dump()
    outcomes: list[SegmentOutcome] = []

    for segment in segments:
        try:
            result = compute_entropy(segment.values, job)
            issue = None
        except ScientificToolError as exc:
            result = None
            issue = exc.issue
        except Exception as exc:
            result = None
            issue = ValidationIssue(
                code=IssueCode.ENTROPY_EXECUTION_FAILED,
                message=f"{type(exc).__name__}: {exc}",
            )
        outcomes.append(
            SegmentOutcome(
                segment_index=segment.index,
                start_sample=segment.start,
                stop_sample=segment.stop,
                signal_length=segment.values.size,
                method=job.method,
                parameters=parameters,
                result=result,
                issue=issue,
            )
        )

    return EntropyBatchResult(
        job_id=job.job_id,
        revision=job.revision,
        signal_sha256=signal_sha256(signal_values),
        method=job.method,
        parameters=parameters,
        window_plan=window_plan,
        segments=outcomes,
    )
