from __future__ import annotations

import numpy as np

from entropy_agent.lab import LabContext
from entropy_agent.lab.factory import (
    Candidate,
    build,
    evaluate_candidate,
    run_in_sandbox,
    validate_code,
)

GOOD_CODE = """
import numpy as np


def extract(windows):
    centered = windows - windows.mean(axis=1, keepdims=True)
    energy = np.sqrt((centered ** 2).mean(axis=1))
    peak = np.abs(centered).max(axis=1)
    crest = peak / (energy + 1e-12)
    spectrum = np.abs(np.fft.rfft(centered, axis=1))
    centroid = (spectrum * np.arange(spectrum.shape[1])).sum(axis=1) / (spectrum.sum(axis=1) + 1e-12)
    return np.stack([energy, crest, centroid], axis=1)
"""

CONSTANT_CODE = """
import numpy as np


def extract(windows):
    return np.ones((windows.shape[0], 2))
"""


def test_validate_rejects_imports_and_labels() -> None:
    assert any("不允许导入 os" in item for item in validate_code("import os\n\ndef extract(w):\n    return w\n"))
    bad = "def extract(windows, y):\n    return windows\n"
    assert any("标签" in item for item in validate_code(bad))
    assert validate_code("def not_extract(w):\n    return w\n")


def test_validate_accepts_a_good_candidate() -> None:
    assert validate_code(GOOD_CODE) == []


def test_sandbox_runs_code_and_returns_features() -> None:
    windows = np.random.default_rng(0).normal(size=(6, 256))
    result = run_in_sandbox(GOOD_CODE, windows)
    assert result.ok, result.error
    assert result.report["shape"] == [6, 3]
    assert result.report["finite"] and result.report["deterministic"]
    assert result.features.shape == (6, 3)


def test_sandbox_rejects_a_constant_feature() -> None:
    windows = np.random.default_rng(0).normal(size=(6, 256))
    result = run_in_sandbox(CONSTANT_CODE, windows)
    assert result.ok
    assert result.report["nontrivial"] is False


def test_build_rejects_a_constant_candidate(tiny_dataset, tmp_path) -> None:
    spec, root = tiny_dataset
    ctx = LabContext(spec=spec, data_root=root, workspace=tmp_path / "lab")
    result = build(ctx, Candidate(name="constant-feature", rationale="returns a constant on purpose", code=CONSTANT_CODE))
    assert result["accepted"] is False
    assert result["stage"] == "contract"


def test_evaluate_candidate_returns_split_statistics(tiny_dataset, tmp_path) -> None:
    spec, root = tiny_dataset
    ctx = LabContext(spec=spec, data_root=root, workspace=tmp_path / "lab")
    windows = np.random.default_rng(0).normal(size=(36, spec.window.size))
    features = np.stack(
        [windows.std(axis=1), np.abs(windows).max(axis=1)], axis=1
    )
    metrics = evaluate_candidate(ctx, features, seeds=3)
    assert metrics["seeds"] == 3
    assert 0.0 <= metrics["macro_f1_mean"] <= 1.0
