"""Explicit, bounded re-execution of one entropy job after quality review."""

from __future__ import annotations

import json

from pydantic import Field

from .batch import compute_entropy_batch, signal_sha256
from .quality import BatchQuality, QualityAction, evaluate_entropy_batch
from .schemas import EntropyBatchResult, EntropyJob, IssueCode, StrictModel, ValidationIssue
from .tools import ScientificToolError


class RetryPolicy(StrictModel):
    maximum_revisions_per_job: int = Field(default=2, ge=0)
    maximum_revisions_per_run: int = Field(default=3, ge=0)


class RetryState(StrictModel):
    total_revisions: int = Field(default=0, ge=0)
    revisions_by_job: dict[str, int] = Field(default_factory=dict)
    attempted_fingerprints: list[str] = Field(default_factory=list)


class ReduceScalePatch(StrictModel):
    target_job_id: str = Field(min_length=1)
    new_max_scale: int = Field(ge=1)
    reasoning_summary: str = Field(min_length=1)


class RetryAttempt(StrictModel):
    patch: ReduceScalePatch
    previous_job: EntropyJob
    revised_job: EntropyJob
    previous_quality: BatchQuality
    revised_batch: EntropyBatchResult
    revised_quality: BatchQuality
    state: RetryState


def _reject(code: IssueCode, message: str) -> None:
    raise ScientificToolError(ValidationIssue(code=code, message=message))


def _fingerprint(job: EntropyJob) -> str:
    return json.dumps(
        {
            "job_id": job.job_id,
            "method": job.method,
            "parameters": job.parameters.model_dump(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def retry_reduced_scale(
    signal: object,
    current_job: EntropyJob,
    previous_batch: EntropyBatchResult,
    previous_quality: BatchQuality,
    patch: ReduceScalePatch,
    state: RetryState | None = None,
    policy: RetryPolicy | None = None,
) -> RetryAttempt:
    """Apply an explicitly supplied scale reduction to this job only.

    No patch is generated here. The caller retains the previous batch, and the
    returned attempt records both job revisions plus the revised batch/quality.
    """
    state = state or RetryState()
    policy = policy or RetryPolicy()
    if patch.target_job_id != current_job.job_id:
        _reject(IssueCode.RETRY_NOT_JUSTIFIED, "patch target does not match current job")
    current_max = getattr(current_job.parameters, "scale", None)
    if current_max is None:
        current_max = current_job.parameters.kmax
    if (
        previous_batch.job_id != current_job.job_id
        or previous_batch.revision != current_job.revision
        or previous_batch.method != current_job.method
        or previous_batch.parameters != current_job.parameters.model_dump()
        or previous_quality.job_id != current_job.job_id
        or previous_quality.requested_scale_max != current_max
        or evaluate_entropy_batch(previous_batch, previous_quality.policy) != previous_quality
    ):
        _reject(IssueCode.RETRY_NOT_JUSTIFIED, "previous result does not match current job")
    if previous_batch.signal_sha256 != signal_sha256(signal):
        _reject(
            IssueCode.SIGNAL_MISMATCH,
            "retry input differs from the previous batch input",
        )
    if state.revisions_by_job.get(current_job.job_id, 0) != current_job.revision:
        _reject(IssueCode.STALE_JOB_REVISION, "job revision does not match retry state")
    if (
        state.total_revisions >= policy.maximum_revisions_per_run
        or state.revisions_by_job.get(current_job.job_id, 0)
        >= policy.maximum_revisions_per_job
    ):
        _reject(IssueCode.RETRY_BUDGET_EXCEEDED, "retry budget exhausted")
    if QualityAction.CONSIDER_SCALE_REDUCTION not in previous_quality.actions:
        _reject(IssueCode.RETRY_NOT_JUSTIFIED, "quality review did not advise scale reduction")

    old_scale = previous_quality.requested_scale_max
    if not (1 <= patch.new_max_scale < old_scale):
        _reject(
            IssueCode.PARAMETER_UNSAFE,
            "new scale must be positive and lower than the old scale",
        )
    if patch.new_max_scale > previous_quality.contiguous_usable_scale_max:
        _reject(IssueCode.PARAMETER_UNSAFE, "new scale exceeds the contiguous usable prefix")
    scale_field = "kmax" if hasattr(current_job.parameters, "kmax") else "scale"
    revised_parameters = type(current_job.parameters).model_validate(
        {
            **current_job.parameters.model_dump(),
            scale_field: patch.new_max_scale,
        }
    )
    revised_job = EntropyJob(
        job_id=current_job.job_id,
        revision=current_job.revision + 1,
        method=current_job.method,
        parameters=revised_parameters,
    )
    fingerprint = _fingerprint(revised_job)
    if fingerprint in state.attempted_fingerprints:
        _reject(IssueCode.DUPLICATE_ATTEMPT, "identical method and parameters were already retried")

    revised_batch = compute_entropy_batch(signal, revised_job, previous_batch.window_plan)
    revised_quality = evaluate_entropy_batch(revised_batch, previous_quality.policy)
    revised_state = RetryState(
        total_revisions=state.total_revisions + 1,
        revisions_by_job={
            **state.revisions_by_job,
            current_job.job_id: state.revisions_by_job.get(current_job.job_id, 0) + 1,
        },
        attempted_fingerprints=[*state.attempted_fingerprints, fingerprint],
    )
    return RetryAttempt(
        patch=patch,
        previous_job=current_job,
        revised_job=revised_job,
        previous_quality=previous_quality,
        revised_batch=revised_batch,
        revised_quality=revised_quality,
        state=revised_state,
    )
