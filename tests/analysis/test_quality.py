import unittest

from entropy_agent.analysis.quality import (
    QualityAction,
    QualityPolicy,
    QualityStatus,
    evaluate_entropy_batch,
)
from entropy_agent.analysis.schemas import (
    EntropyBatchResult,
    EntropyResult,
    IssueCode,
    SegmentOutcome,
    ValidationIssue,
    WindowPlan,
)


PARAMS = {"m": 2, "nc": 3, "tau": 1, "scale": 3}


def make_result(index, values, counts=(127, 63, 41), mask=None, issues=None):
    if mask is None:
        mask = [value == value and value != float("inf") for value in values]
    return EntropyResult(
        job_id="job", method="MDE", signal_length=128,
        parameters=PARAMS, values=values, finite_mask=mask,
        effective_window_counts=list(counts), issues=issues or [],
    )


def make_batch(*outcomes):
    segments = []
    for index, outcome in enumerate(outcomes):
        segments.append(SegmentOutcome(
            segment_index=index, start_sample=128 * index,
            stop_sample=128 * (index + 1), signal_length=128,
            method="MDE", parameters=PARAMS,
            result=outcome if isinstance(outcome, EntropyResult) else None,
            issue=outcome if isinstance(outcome, ValidationIssue) else None,
        ))
    return EntropyBatchResult(
        job_id="job", revision=0, signal_sha256="0" * 64,
        method="MDE", parameters=PARAMS,
        window_plan=WindowPlan(window_size_samples=128, step_size_samples=128),
        segments=segments,
    )


class QualityTests(unittest.TestCase):
    def test_all_finite_is_computationally_complete(self):
        quality = evaluate_entropy_batch(make_batch(
            make_result(0, [1.0, -0.0, 3.0]),
            make_result(1, [1.1, 2.1, 3.1]),
        ))
        self.assertEqual(quality.status, QualityStatus.COMPLETE)
        self.assertEqual(quality.contiguous_usable_scale_max, 3)
        self.assertEqual(quality.actions, [QualityAction.KEEP])

    def test_nonfinite_high_scale_suggests_but_does_not_apply_reduction(self):
        issue = ValidationIssue(
            code=IssueCode.NONFINITE_OUTPUT, message="raw NaN retained", scale=3,
        )
        batch = make_batch(
            make_result(0, [1.0, 2.0, float("nan")], issues=[issue]),
            make_result(1, [1.1, 2.1, 3.1]),
        )
        quality = evaluate_entropy_batch(batch)
        self.assertEqual(quality.status, QualityStatus.PARTIAL)
        self.assertEqual(quality.contiguous_usable_scale_max, 2)
        self.assertIn(QualityAction.CONSIDER_SCALE_REDUCTION, quality.actions)
        self.assertIn(QualityAction.REVIEW_NONFINITE_OUTPUT, quality.actions)
        self.assertEqual(quality.scale_quality[2].usable_fraction, 0.5)
        self.assertEqual(quality.issues[0].segment_index, 0)
        self.assertTrue(batch.segments[0].result.values[2] != batch.segments[0].result.values[2])

    def test_failed_window_does_not_poison_successful_window(self):
        failure = ValidationIssue(code=IssueCode.NONFINITE_INPUT, message="NaN input")
        quality = evaluate_entropy_batch(make_batch(
            make_result(0, [1.0, 2.0, 3.0]), failure,
        ))
        self.assertEqual(quality.status, QualityStatus.PARTIAL)
        self.assertEqual(quality.successful_segments, 1)
        self.assertEqual(quality.failed_segments, 1)
        self.assertEqual(quality.contiguous_usable_scale_max, 3)
        self.assertIn(QualityAction.REVIEW_FAILED_WINDOWS, quality.actions)
        self.assertEqual(quality.issues[0].segment_index, 1)

    def test_stricter_geometric_threshold_is_explicit(self):
        quality = evaluate_entropy_batch(
            make_batch(make_result(0, [1.0, 2.0, 3.0], counts=(127, 63, 4))),
            QualityPolicy(minimum_geometric_windows=20),
        )
        self.assertEqual(quality.status, QualityStatus.PARTIAL)
        self.assertEqual(quality.contiguous_usable_scale_max, 2)
        self.assertIn(QualityAction.REVIEW_WINDOW_LENGTH, quality.actions)

    def test_mask_mismatch_makes_result_unusable(self):
        quality = evaluate_entropy_batch(make_batch(
            make_result(0, [1.0, float("nan"), 3.0], mask=[True, True, True]),
        ))
        self.assertEqual(quality.status, QualityStatus.UNUSABLE)
        self.assertEqual(quality.issues[0].code, IssueCode.RESULT_MASK_MISMATCH)

    def test_all_windows_failed(self):
        failure = ValidationIssue(code=IssueCode.CONSTANT_SIGNAL, message="dead channel")
        quality = evaluate_entropy_batch(make_batch(failure, failure))
        self.assertEqual(quality.status, QualityStatus.UNUSABLE)
        self.assertEqual(quality.actions[0], QualityAction.STOP)


if __name__ == "__main__":
    unittest.main()
