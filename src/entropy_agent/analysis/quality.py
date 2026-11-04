"""Deterministic quality assessment; recommendations never execute a retry."""

from __future__ import annotations

from enum import Enum

import numpy as np
from pydantic import Field

from .schemas import EntropyBatchResult, IssueCode, StrictModel, ValidationIssue


class QualityStatus(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    UNUSABLE = "UNUSABLE"


class QualityAction(str, Enum):
    KEEP = "KEEP"
    REVIEW_FAILED_WINDOWS = "REVIEW_FAILED_WINDOWS"
    CONSIDER_SCALE_REDUCTION = "CONSIDER_SCALE_REDUCTION"
    REVIEW_WINDOW_LENGTH = "REVIEW_WINDOW_LENGTH"
    REVIEW_NONFINITE_OUTPUT = "REVIEW_NONFINITE_OUTPUT"
    STOP = "STOP"


class QualityPolicy(StrictModel):
    # Defaults are computational acceptance limits, NOT universal guarantees
    # of statistical reliability. Domain experiments may set stricter values.
    minimum_geometric_windows: int = Field(default=1, ge=1)
    minimum_usable_fraction: float = Field(default=1.0, gt=0, le=1)


class ScaleQuality(StrictModel):
    scale: int
    evaluated_segments: int
    finite_segments: int
    usable_segments: int
    usable_fraction: float | None
    minimum_geometric_windows: int | None
    meets_policy: bool


class BatchQuality(StrictModel):
    job_id: str
    policy: QualityPolicy
    status: QualityStatus
    requested_scale_max: int
    contiguous_usable_scale_max: int
    total_segments: int
    successful_segments: int
    failed_segments: int
    scale_quality: list[ScaleQuality]
    actions: list[QualityAction]
    issues: list[ValidationIssue]


def _expected_scale_count(batch: EntropyBatchResult) -> int:
    value = batch.parameters.get("scale", batch.parameters.get("kmax"))
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("batch has no positive integer scale or kmax")
    return value


def _structural_issue(batch: EntropyBatchResult, segment_index: int, expected: int):
    segment = batch.segments[segment_index]
    result = segment.result
    if result is None:
        return None
    if (
        segment.method != batch.method
        or segment.parameters != batch.parameters
        or result.job_id != batch.job_id
        or result.method != batch.method
        or result.parameters != batch.parameters
        or result.signal_length != segment.signal_length
    ):
        return ValidationIssue(
            code=IssueCode.RESULT_METADATA_MISMATCH,
            message=f"segment {segment.segment_index} result metadata differs from the batch",
        )
    if not (
        len(result.values) == len(result.finite_mask)
        == len(result.effective_window_counts) == expected
    ):
        return ValidationIssue(
            code=IssueCode.OUTPUT_LENGTH_MISMATCH,
            message=f"segment {segment.segment_index} result length differs from {expected}",
        )
    actual_mask = np.isfinite(np.asarray(result.values, dtype=np.float64)).tolist()
    if actual_mask != result.finite_mask:
        return ValidationIssue(
            code=IssueCode.RESULT_MASK_MISMATCH,
            message=f"segment {segment.segment_index} finite mask differs from raw values",
        )
    return None


def evaluate_entropy_batch(
    batch: EntropyBatchResult, policy: QualityPolicy | None = None
) -> BatchQuality:
    """Summarize finite coverage and geometric support, without altering results."""
    policy = policy or QualityPolicy()
    expected_scale_count = _expected_scale_count(batch)
    issues: list[ValidationIssue] = []
    valid_results = []

    for index, segment in enumerate(batch.segments):
        if segment.issue is not None:
            issues.append(
                segment.issue.model_copy(
                    update={"segment_index": segment.segment_index}
                )
            )
            continue
        structural_issue = _structural_issue(batch, index, expected_scale_count)
        if structural_issue is not None:
            issues.append(
                structural_issue.model_copy(
                    update={"segment_index": segment.segment_index}
                )
            )
            continue
        valid_results.append(segment.result)
        issues.extend(
            issue.model_copy(update={"segment_index": segment.segment_index})
            for issue in segment.result.issues
        )

    scale_quality = []
    saw_nonfinite = False
    saw_low_support = False
    for scale_index in range(expected_scale_count):
        finite_count = sum(result.finite_mask[scale_index] for result in valid_results)
        supported = [
            result
            for result in valid_results
            if result.effective_window_counts[scale_index] is None
            or result.effective_window_counts[scale_index]
            >= policy.minimum_geometric_windows
        ]
        usable_count = sum(result.finite_mask[scale_index] for result in supported)
        window_counts = [
            result.effective_window_counts[scale_index]
            for result in valid_results
            if result.effective_window_counts[scale_index] is not None
        ]
        minimum_count = min(window_counts) if window_counts else None
        denominator = len(valid_results)
        fraction = usable_count / denominator if denominator else None
        meets_policy = bool(
            denominator and fraction >= policy.minimum_usable_fraction
        )
        saw_nonfinite |= finite_count < denominator
        saw_low_support |= len(supported) < denominator
        scale_quality.append(
            ScaleQuality(
                scale=scale_index + 1,
                evaluated_segments=denominator,
                finite_segments=finite_count,
                usable_segments=usable_count,
                usable_fraction=fraction,
                minimum_geometric_windows=minimum_count,
                meets_policy=meets_policy,
            )
        )

    contiguous_usable_scale_max = 0
    for scale in scale_quality:
        if not scale.meets_policy:
            break
        contiguous_usable_scale_max = scale.scale

    failed_segment_count = len(batch.segments) - len(valid_results)
    if not valid_results or not any(scale.meets_policy for scale in scale_quality):
        status = QualityStatus.UNUSABLE
    elif (
        failed_segment_count == 0
        and contiguous_usable_scale_max == expected_scale_count
    ):
        status = QualityStatus.COMPLETE
    else:
        status = QualityStatus.PARTIAL

    actions: list[QualityAction] = []
    if status == QualityStatus.UNUSABLE:
        actions.append(QualityAction.STOP)
    elif status == QualityStatus.COMPLETE:
        actions.append(QualityAction.KEEP)
    if failed_segment_count:
        actions.append(QualityAction.REVIEW_FAILED_WINDOWS)
    if 0 < contiguous_usable_scale_max < expected_scale_count:
        actions.append(QualityAction.CONSIDER_SCALE_REDUCTION)
    if saw_low_support:
        actions.append(QualityAction.REVIEW_WINDOW_LENGTH)
    if saw_nonfinite:
        actions.append(QualityAction.REVIEW_NONFINITE_OUTPUT)

    return BatchQuality(
        job_id=batch.job_id,
        policy=policy,
        status=status,
        requested_scale_max=expected_scale_count,
        contiguous_usable_scale_max=contiguous_usable_scale_max,
        total_segments=len(batch.segments),
        successful_segments=len(valid_results),
        failed_segments=failed_segment_count,
        scale_quality=scale_quality,
        actions=actions,
        issues=issues,
    )
