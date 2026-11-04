import unittest
from typing import get_args
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

from entropy_agent.analysis.schemas import (
    DispersionParameters,
    EntropyJob,
    RenyiFuzzyDispersionParameters,
    IssueCode,
    PermutationParameters,
)
from entropy_agent.analysis import entropy_backend as entropy_toolbox
from entropy_agent.analysis.methods import METHOD_FAMILIES, V1_AUTO_METHODS, AgentMethod
from entropy_agent.analysis.tools import ScientificToolError, compute_entropy, inspect_signal


class SchemaTests(unittest.TestCase):
    def test_reject_method_parameter_mismatch(self):
        with self.assertRaises(ValidationError):
            EntropyJob(
                job_id="a", method="MPE",
                parameters=DispersionParameters(m=2, tau=1, nc=5, scale=2),
            )

    def test_reject_nonpositive_scale(self):
        with self.assertRaises(ValidationError):
            PermutationParameters(m=2, tau=1, scale=0)


class ToolTests(unittest.TestCase):
    def test_inspection_rejects_constant_and_nonfinite(self):
        result = inspect_signal([1.0, 1.0, float("nan")])
        self.assertFalse(result.valid_for_entropy)
        self.assertEqual(result.nan_count, 1)
        self.assertEqual(
            {issue.code for issue in result.issues},
            {IssueCode.CONSTANT_SIGNAL, IssueCode.NONFINITE_INPUT},
        )

    def test_composite_last_offset_is_checked(self):
        job = EntropyJob(
            job_id="a", method="RCMDE",
            parameters=DispersionParameters(m=3, nc=5, tau=2, scale=3),
        )
        with self.assertRaises(ScientificToolError) as context:
            compute_entropy(np.arange(15, dtype=float), job)
        self.assertEqual(context.exception.issue.code, IssueCode.INSUFFICIENT_EFFECTIVE_WINDOWS)
        self.assertEqual(context.exception.issue.scale, 3)

    def test_nonfinite_and_negative_zero_are_preserved(self):
        job = EntropyJob(
            job_id="a", method="MPE",
            parameters=PermutationParameters(m=2, tau=1, scale=3),
        )
        with patch.object(
            entropy_toolbox, "compute",
            return_value=np.array([-0.0, np.nan, np.inf]),
        ):
            result = compute_entropy(np.arange(12, dtype=float), job)
        self.assertTrue(np.signbit(result.values[0]))
        self.assertTrue(np.isnan(result.values[1]))
        self.assertTrue(np.isposinf(result.values[2]))
        self.assertEqual(result.finite_mask, [True, False, False])
        self.assertEqual(result.effective_window_counts, [11, 5, 3])

    def test_agent_methods_exist_in_toolbox_registry(self):
        # The planner literal, its family table and the toolbox registry are three
        # views of one list; exact equality catches silent drift in any of them.
        registry = set(entropy_toolbox.available_methods())
        self.assertEqual(set(V1_AUTO_METHODS), registry)
        self.assertEqual(set(get_args(AgentMethod)), registry)
        self.assertEqual(set(METHOD_FAMILIES), registry)

    def test_imfrde_uses_shortest_shifted_offset_for_safety(self):
        job = EntropyJob(
            job_id="a",
            method="IMFRDE",
            parameters=RenyiFuzzyDispersionParameters(
                m=3, nc=5, tau=2, scale=6, type_=0, q=1.3
            ),
        )
        with self.assertRaises(ScientificToolError) as context:
            compute_entropy(np.arange(30, dtype=float), job)
        self.assertEqual(
            context.exception.issue.code, IssueCode.INSUFFICIENT_EFFECTIVE_WINDOWS
        )


if __name__ == "__main__":
    unittest.main()
