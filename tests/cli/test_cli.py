from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from entropy_agent.cli import main
from entropy_agent.io.dataset import load_dataset_spec


def _write_signal(spec, root: Path, path: Path, label: int, record: int = 0) -> None:
    """One raw window of the tiny dataset, saved as a bare NPY vector."""
    from entropy_agent.io.dataset import load_dataset_records, load_record_signal

    records = [item for item in load_dataset_records(spec, root) if item.label == label]
    np.save(path, load_record_signal(records[record]).values)


def _train_model(tiny_dataset, tmp_path: Path) -> tuple[Path, Path]:
    spec, root = tiny_dataset
    config = tmp_path / "spec.json"
    config.write_text(json.dumps(spec.model_dump(mode="json"), indent=2), encoding="utf-8")
    output = tmp_path / "run"
    code = main(
        [
            "train",
            "--config", str(config),
            "--data-root", str(root),
            "--output", str(output),
            "--model-id", "tiny-cli-model",
        ]
    )
    assert code == 0
    return output / "model", config


def test_train_writes_metrics_and_artifact(tiny_dataset, tmp_path: Path, capsys) -> None:
    model_dir, _ = _train_model(tiny_dataset, tmp_path)
    assert (model_dir / "config.json").is_file()
    metrics = json.loads((tmp_path / "run" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["model_id"] == "tiny-cli-model"
    assert metrics["validation"]["multiclass_accuracy"] is not None
    out = capsys.readouterr().out
    assert "validation" in out and "test" in out


def test_models_register_list_and_show(tiny_dataset, tmp_path: Path, capsys) -> None:
    model_dir, _ = _train_model(tiny_dataset, tmp_path)
    registry = tmp_path / "registry"

    assert main(["models", "--registry", str(registry), "--trusted", "register",
                 "--artifact", str(model_dir)]) == 0
    capsys.readouterr()

    assert main(["models", "--registry", str(registry), "--trusted", "list"]) == 0
    listing = capsys.readouterr().out
    assert "tiny-cli-model" in listing
    assert "TINY" in listing

    assert main(["models", "--registry", str(registry), "--trusted", "show", "tiny-cli-model"]) == 0
    detail = json.loads(capsys.readouterr().out)
    assert detail["model_id"] == "tiny-cli-model"
    assert detail["profile"]["profile_id"] == "TINY"


def test_diagnose_one_signal_through_the_registry(tiny_dataset, tmp_path: Path) -> None:
    model_dir, _ = _train_model(tiny_dataset, tmp_path)
    registry = tmp_path / "registry"
    assert main(["models", "--registry", str(registry), "--trusted", "register",
                 "--artifact", str(model_dir)]) == 0

    signal = tmp_path / "sig" / "sample.npy"
    signal.parent.mkdir(parents=True, exist_ok=True)
    spec, root = tiny_dataset
    _write_signal(spec, root, signal, 0, record=0)

    code = main(
        [
            "diagnose",
            "--registry", str(registry),
            "--trusted",
            "--profile", "TINY",
            "--signal", str(signal),
            "--sampling-rate", "1000",
            "--signal-type", "ACCELERATION",
            "--signal-role", "TEST_CHANNEL",
            "--json",
        ]
    )
    assert code == 0


def test_diagnose_rejects_a_missing_sampling_rate(tiny_dataset, tmp_path: Path) -> None:
    spec, root = tiny_dataset
    model_dir, _ = _train_model(tiny_dataset, tmp_path)
    signal = tmp_path / "sig" / "sample.npy"
    signal.parent.mkdir(parents=True, exist_ok=True)
    _write_signal(spec, root, signal, 0)

    code = main(
        ["diagnose", "--artifact", str(model_dir), "--trusted", "--signal", str(signal)]
    )
    assert code == 1  # abstained, and the CLI reports it as a non-zero exit


def test_diagnose_batch_writes_a_csv(tiny_dataset, tmp_path: Path) -> None:
    spec, root = tiny_dataset
    model_dir, _ = _train_model(tiny_dataset, tmp_path)
    signals = tmp_path / "signals"
    signals.mkdir()
    for index in range(3):
        _write_signal(spec, root, signals / f"w{index}.npy", index % 3, record=index)

    output = tmp_path / "batch.csv"
    code = main(
        [
            "diagnose-batch",
            "--artifact", str(model_dir),
            "--trusted",
            "--signal", "ignored",
            "--input-dir", str(signals),
            "--pattern", "*.npy",
            "--sampling-rate", "1000",
            "--output", str(output),
        ]
    )
    assert code == 0
    rows = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 4  # header + 3 signals
    assert "DIAGNOSED" in output.read_text(encoding="utf-8")


def test_agent_run_via_cli_uses_a_seconds_budget(tiny_dataset, tmp_path: Path) -> None:
    spec, root = tiny_dataset
    config = tmp_path / "spec.json"
    config.write_text(json.dumps(spec.model_dump(mode="json"), indent=2), encoding="utf-8")
    output = tmp_path / "agent-run"

    code = main(
        [
            "agent", "run",
            "--spec", str(config),
            "--data-root", str(root),
            "--output", str(output),
            "--budget-seconds", "2",
            "--seeds", "7",
            "--validation-repeats", "1",
            "--arms", "agent",
            "--no-test",
        ]
    )
    assert code == 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["budget"]["mode"] == "seconds"
    assert (output / "report.md").is_file()


def test_inspect_describes_a_csv(tmp_path: Path, capsys) -> None:
    path = tmp_path / "two.csv"
    path.write_text("time,vibration\n0,1\n1,2\n", encoding="utf-8")
    assert main(["inspect", "--signal", str(path)]) == 0
    assert "vibration" in capsys.readouterr().out


def test_untrusted_artifact_is_refused(tiny_dataset, tmp_path: Path, capsys) -> None:
    model_dir, _ = _train_model(tiny_dataset, tmp_path)
    signal = tmp_path / "sig" / "sample.npy"
    signal.parent.mkdir(parents=True, exist_ok=True)
    spec, root = tiny_dataset
    _write_signal(spec, root, signal, 0)

    code = main(
        ["diagnose", "--artifact", str(model_dir), "--signal", str(signal),
         "--sampling-rate", "1000"]
    )
    assert code == 2
    assert "trusted" in capsys.readouterr().err
