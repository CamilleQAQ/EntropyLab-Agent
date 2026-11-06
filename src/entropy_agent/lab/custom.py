"""Access to feature extractors registered by the tool factory."""

from __future__ import annotations

import json
from pathlib import Path


def registry_path(workspace: Path) -> Path:
    return Path(workspace) / "lab_tools" / "registry.json"


def registered_tools(workspace: Path) -> list[dict]:
    path = registry_path(workspace)
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_code(workspace: Path, name: str) -> str:
    """Return the source of a registered tool."""
    for entry in registered_tools(workspace):
        if entry.get("name") == name:
            code_path = Path(workspace) / "lab_tools" / entry["code"]
            if not code_path.is_file():
                raise FileNotFoundError(f"registered code missing: {code_path}")
            return code_path.read_text(encoding="utf-8")
    raise KeyError(f"no registered tool named {name!r}")


def catalogue(workspace: Path) -> list[dict]:
    return [
        {
            "name": entry["name"],
            "macro_f1_mean": entry.get("macro_f1_mean"),
            "delta": entry.get("delta"),
        }
        for entry in registered_tools(workspace)
    ]
