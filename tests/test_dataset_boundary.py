"""Architecture boundary checks, not a banned-word scan.

The core must not import concrete dataset adapters or the MATLAB compatibility
layer.  This stays meaningful as more core modules are migrated.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE_PACKAGES = [ROOT / "src" / "entropy_agent", ROOT / "src" / "msentropy"]
FORBIDDEN_PREFIXES = (
    "entropy_agent.datasets",
    "entropy_agent.compatibility",
)


def _python_files() -> list[Path]:
    files: list[Path] = []
    for package in CORE_PACKAGES:
        files.extend(path for path in package.rglob("*.py") if "__pycache__" not in path.parts)
    return sorted(files)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_core_does_not_import_dataset_adapters() -> None:
    offenders: list[str] = []
    for path in _python_files():
        for name in _imports(path):
            if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_PREFIXES):
                offenders.append(f"{path.relative_to(ROOT)}: {name}")
    assert not offenders, "core imported a dataset/compatibility adapter:\n" + "\n".join(offenders)


def test_concrete_dataset_packages_are_not_in_the_core_tree() -> None:
    assert not (ROOT / "src" / "entropy_agent" / "datasets").exists()
    assert not (ROOT / "src" / "entropy_agent" / "compatibility").exists()
