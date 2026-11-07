from __future__ import annotations

import json
from pathlib import Path

from entropy_agent.diagnosis.model_registry import (
    REGISTRY_FORMAT,
    REGISTRY_VERSION,
    ModelRegistry,
)


def test_registry_accepts_the_index_file(tmp_path: Path) -> None:
    index = tmp_path / "reg" / "registry.json"
    registry = ModelRegistry(index)
    assert registry.index_path == index
    assert registry.root == index.parent
    # writing must create the directory, not fail on the file itself
    registry._write({"format": REGISTRY_FORMAT, "format_version": REGISTRY_VERSION, "models": {}})
    assert index.is_file()


def test_registry_accepts_the_directory(tmp_path: Path) -> None:
    directory = tmp_path / "reg"
    registry = ModelRegistry(directory)
    assert registry.index_path == directory / "registry.json"
    assert registry.root == directory


def test_entries_exposes_paths_without_loading_artifacts(tmp_path: Path) -> None:
    index = tmp_path / "reg" / "registry.json"
    index.parent.mkdir(parents=True)
    index.write_text(
        json.dumps(
            {
                "format": REGISTRY_FORMAT,
                "format_version": REGISTRY_VERSION,
                "models": {"demo": {"path": "demo-seed-1"}},
            }
        ),
        encoding="utf-8",
    )
    registry = ModelRegistry(index)
    entries = registry.entries()
    assert entries == [{"model_id": "demo", "path": str((index.parent / "demo-seed-1").resolve())}]
