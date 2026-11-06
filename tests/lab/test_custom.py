from __future__ import annotations

import json
from pathlib import Path

from entropy_agent.lab import LabContext
from entropy_agent.lab.custom import catalogue, load_code, registered_tools
from entropy_agent.lab.tools import CustomExperimentArgs, run_custom_experiment

TOOL_CODE = """
import numpy as np


def extract(windows):
    centered = windows - windows.mean(axis=1, keepdims=True)
    rms = np.sqrt((centered ** 2).mean(axis=1))
    return np.stack([rms, np.abs(centered).max(axis=1)], axis=1)
"""


def _register(workspace: Path, name: str = "rms_peak") -> None:
    tools = workspace / "lab_tools"
    tools.mkdir(parents=True, exist_ok=True)
    (tools / f"{name}.py").write_text(TOOL_CODE, encoding="utf-8")
    registry = tools / "registry.json"
    entries = json.loads(registry.read_text(encoding="utf-8")) if registry.is_file() else []
    entries.append({"name": name, "code": f"{name}.py", "macro_f1_mean": 0.5, "delta": 0.01})
    registry.write_text(json.dumps(entries), encoding="utf-8")


def test_registry_helpers(tmp_path: Path) -> None:
    assert registered_tools(tmp_path) == []
    _register(tmp_path)
    assert registered_tools(tmp_path)[0]["name"] == "rms_peak"
    assert "def extract" in load_code(tmp_path, "rms_peak")
    assert catalogue(tmp_path)[0]["name"] == "rms_peak"


def test_custom_experiment_uses_registered_code(tiny_dataset, tmp_path: Path) -> None:
    spec, root = tiny_dataset
    workspace = tmp_path / "lab"
    _register(workspace)
    ctx = LabContext(spec=spec, data_root=root, workspace=workspace)
    result = run_custom_experiment(ctx, CustomExperimentArgs(name="rms_peak", seeds=3))
    assert result["tool"] == "run_custom_experiment"
    assert result["feature"] == "rms_peak"
    assert result["seeds"] == 3
    assert 0.0 <= result["macro_f1_mean"] <= 1.0


def test_custom_experiment_reports_a_missing_tool(tiny_dataset, tmp_path: Path) -> None:
    spec, root = tiny_dataset
    ctx = LabContext(spec=spec, data_root=root, workspace=tmp_path / "lab")
    try:
        run_custom_experiment(ctx, CustomExperimentArgs(name="nope", seeds=1))
    except KeyError as error:
        assert "nope" in str(error)
    else:  # pragma: no cover - the call must raise
        raise AssertionError("missing tool must raise")
