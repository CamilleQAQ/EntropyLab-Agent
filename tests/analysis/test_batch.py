import unittest
from pathlib import Path

import numpy as np

from entropy_agent.analysis.batch import compute_entropy_batch
from entropy_agent.io.csv_windows import load_csv_signal, window_signal
from entropy_agent.analysis.schemas import (
    CSVLoadRequest,
    DispersionParameters,
    EntropyJob,
    IssueCode,
    WindowPlan,
)
from entropy_agent.analysis.tools import ScientificToolError


class CSVTests(unittest.TestCase):
    FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

    def test_explicit_named_channel(self):
        loaded = load_csv_signal(CSVLoadRequest(
            path=str(self.FIXTURES / "three_channels.csv"),
            channel="y", has_header=True,
        ))
        np.testing.assert_array_equal(loaded.values, [10, 20])

    def test_header_outer_whitespace_is_ignored(self):
        loaded = load_csv_signal(CSVLoadRequest(
            path=str(self.FIXTURES / "three_channels.csv"),
            channel="X-axis", has_header=True,
        ))
        np.testing.assert_array_equal(loaded.values, [1, 2])

    def test_missing_channel_is_not_guessed(self):
        with self.assertRaises(ScientificToolError) as context:
            load_csv_signal(CSVLoadRequest(
                path=str(self.FIXTURES / "three_channels.csv"),
                channel="z", has_header=True,
            ))
        self.assertEqual(context.exception.issue.code, IssueCode.CHANNEL_NOT_FOUND)

    def test_empty_cell_is_not_imputed(self):
        with self.assertRaises(ScientificToolError) as context:
            load_csv_signal(CSVLoadRequest(
                path=str(self.FIXTURES / "missing_cell.csv"),
                channel="y", has_header=True,
            ))
        self.assertEqual(context.exception.issue.code, IssueCode.CSV_PARSE_FAILED)


class WindowAndBatchTests(unittest.TestCase):
    def test_window_offsets_and_drop_tail(self):
        plan = WindowPlan(window_size_samples=4, step_size_samples=3)
        segments = window_signal(np.arange(11), plan)
        self.assertEqual([(s.start, s.stop) for s in segments], [(0, 4), (3, 7), (6, 10)])

    def test_step_can_be_larger_than_window(self):
        plan = WindowPlan(window_size_samples=4, step_size_samples=5)
        segments = window_signal(np.arange(11), plan)
        self.assertEqual([(s.start, s.stop) for s in segments], [(0, 4), (5, 9)])

    def test_real_toolbox_call_and_window_failure_isolation(self):
        rng = np.random.default_rng(10)
        signal = rng.normal(size=384)
        signal[150] = np.nan  # only the second complete window is invalid
        job = EntropyJob(
            job_id="mde-windows", method="MDE",
            parameters=DispersionParameters(m=2, nc=3, tau=1, scale=3),
        )
        batch = compute_entropy_batch(
            signal, job, WindowPlan(window_size_samples=128, step_size_samples=128)
        )
        self.assertEqual(len(batch.segments), 3)
        self.assertIsNotNone(batch.segments[0].result)
        self.assertEqual(batch.segments[1].issue.code, IssueCode.NONFINITE_INPUT)
        self.assertIsNotNone(batch.segments[2].result)
        self.assertEqual(len(batch.segments[0].result.values), 3)
        self.assertEqual(batch.segments[0].result.effective_window_counts, [127, 63, 41])


if __name__ == "__main__":
    unittest.main()
