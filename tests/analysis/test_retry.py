import unittest
from unittest.mock import patch

import numpy as np

from entropy_agent.analysis import entropy_backend as entropy_toolbox
from entropy_agent.analysis.batch import compute_entropy_batch
from entropy_agent.analysis.quality import QualityStatus, evaluate_entropy_batch
from entropy_agent.analysis.retry import (
    ReduceScalePatch,
    RetryPolicy,
    RetryState,
    retry_reduced_scale,
)
from entropy_agent.analysis.schemas import (
    EntropyJob,
    IssueCode,
    PermutationParameters,
    TimeShiftEnsembleParameters,
    WindowPlan,
)
from entropy_agent.analysis.tools import ScientificToolError


def fake_entropy(_method, _signal, parameters):
    if parameters["scale"] == 3:
        return np.array([1.0, 2.0, np.nan])
    return np.array([1.0, 2.0])


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.signal = np.random.default_rng(42).normal(size=128)
        self.job = EntropyJob(
            job_id="mpe", method="MPE",
            parameters=PermutationParameters(m=2, tau=1, scale=3),
        )
        with patch.object(entropy_toolbox, "compute", side_effect=fake_entropy):
            self.previous_batch = compute_entropy_batch(
                self.signal, self.job,
                WindowPlan(window_size_samples=64, step_size_samples=64),
            )
        self.previous_quality = evaluate_entropy_batch(self.previous_batch)
        self.patch = ReduceScalePatch(
            target_job_id="mpe", new_max_scale=2,
            reasoning_summary="Scale 3 is nonfinite in both windows",
        )

    def _run(self, **kwargs):
        with patch.object(entropy_toolbox, "compute", side_effect=fake_entropy):
            return retry_reduced_scale(
                self.signal, self.job, self.previous_batch,
                self.previous_quality, self.patch, **kwargs,
            )

    def test_retries_only_target_job_and_retains_prior_result(self):
        attempt = self._run()
        self.assertEqual(attempt.revised_job.revision, 1)
        self.assertEqual(attempt.revised_job.parameters.scale, 2)
        self.assertEqual(attempt.revised_quality.status, QualityStatus.COMPLETE)
        self.assertEqual(attempt.state.total_revisions, 1)
        self.assertEqual(attempt.revised_batch.signal_sha256, self.previous_batch.signal_sha256)
        self.assertTrue(np.isnan(self.previous_batch.segments[0].result.values[2]))
        self.assertEqual(len(attempt.revised_batch.segments), 2)

    def test_rejects_changed_input(self):
        changed = self.signal.copy()
        changed[0] += 1
        with self.assertRaises(ScientificToolError) as context:
            retry_reduced_scale(
                changed, self.job, self.previous_batch,
                self.previous_quality, self.patch,
            )
        self.assertEqual(context.exception.issue.code, IssueCode.SIGNAL_MISMATCH)

    def test_rejects_scale_above_usable_prefix(self):
        bad_patch = ReduceScalePatch(
            target_job_id="mpe", new_max_scale=3, reasoning_summary="try same scale",
        )
        with self.assertRaises(ScientificToolError) as context:
            retry_reduced_scale(
                self.signal, self.job, self.previous_batch,
                self.previous_quality, bad_patch,
            )
        self.assertEqual(context.exception.issue.code, IssueCode.PARAMETER_UNSAFE)

    def test_budget_is_enforced_before_execution(self):
        with self.assertRaises(ScientificToolError) as context:
            self._run(policy=RetryPolicy(maximum_revisions_per_job=0))
        self.assertEqual(context.exception.issue.code, IssueCode.RETRY_BUDGET_EXCEEDED)

    def test_duplicate_deterministic_attempt_is_rejected(self):
        attempt = self._run()
        # An already-attempted parameter set cannot be submitted as a fresh try.
        replay_state = RetryState(
            total_revisions=1, revisions_by_job={"mpe": 0},
            attempted_fingerprints=attempt.state.attempted_fingerprints,
        )
        with self.assertRaises(ScientificToolError) as context:
            self._run(state=replay_state)
        self.assertEqual(context.exception.issue.code, IssueCode.DUPLICATE_ATTEMPT)

    def test_stale_job_revision_is_rejected(self):
        state = RetryState(total_revisions=1, revisions_by_job={"mpe": 1})
        with self.assertRaises(ScientificToolError) as context:
            self._run(state=state)
        self.assertEqual(context.exception.issue.code, IssueCode.STALE_JOB_REVISION)

    def test_time_shift_patch_updates_kmax_not_scale(self):
        job = EntropyJob(
            job_id="time-shift", method="TSMEFuDE",
            parameters=TimeShiftEnsembleParameters(m=2, nc=3, tau=1, kmax=3),
        )

        def fake_time_shift(_method, _signal, parameters):
            return np.array([1.0, 2.0, np.nan])[:parameters["kmax"]]

        with patch.object(entropy_toolbox, "compute", side_effect=fake_time_shift):
            original = compute_entropy_batch(
                self.signal, job,
                WindowPlan(window_size_samples=64, step_size_samples=64),
            )
            quality = evaluate_entropy_batch(original)
            attempt = retry_reduced_scale(
                self.signal, job, original, quality,
                ReduceScalePatch(
                    target_job_id="time-shift", new_max_scale=2,
                    reasoning_summary="Highest kmax is nonfinite",
                ),
            )
        self.assertEqual(attempt.revised_job.parameters.kmax, 2)
        self.assertEqual(attempt.revised_quality.status, QualityStatus.COMPLETE)

    def test_complete_result_does_not_allow_unjustified_retry(self):
        with patch.object(
            entropy_toolbox, "compute", return_value=np.array([1.0, 2.0, 3.0]),
        ):
            complete = compute_entropy_batch(
                self.signal, self.job,
                WindowPlan(window_size_samples=64, step_size_samples=64),
            )
        with self.assertRaises(ScientificToolError) as context:
            retry_reduced_scale(
                self.signal, self.job, complete,
                evaluate_entropy_batch(complete), self.patch,
            )
        self.assertEqual(context.exception.issue.code, IssueCode.RETRY_NOT_JUSTIFIED)


if __name__ == "__main__":
    unittest.main()
