#!/usr/bin/env python
"""Pack the VBL-VA001 lab vibration dataset into the repository's npy-tensor form.

Source layout (public dataset, 20 kHz, ~100k samples per file):

    7数据源/../VBL-VA001/VBL-VA001/{normal,bearing,misalignment,unbalance}/*.csv

Each CSV holds ``time, a1, a2, a3``.  The archive documents no axis order, so this
adapter always takes the **last** acceleration column and records that choice in
the manifest instead of pretending to know it is "Z".

Records are chosen by *unit* (the 3-digit id in the file name): files of the same
unit are different recordings of the same physical condition, so they must never
be split across partitions.  We spread the sample across units (round-robin) so a
partition never ends up with only one unit of a class.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CLASSES = (
    (0, "NORMAL", "normal"),
    (1, "BEARING", "bearing"),
    (2, "MISALIGNMENT", "misalignment"),
    (3, "UNBALANCE", "unbalance"),
)

UNIT_PATTERN = re.compile(r"_(?P<unit>\d{3})(?:-\d+)?_", re.IGNORECASE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unit_of(path: Path) -> str:
    match = UNIT_PATTERN.search(path.name)
    if match is None:
        raise SystemExit(f"cannot parse a unit id from {path.name}")
    return match.group("unit")


def _select(paths: list[Path], limit: int) -> list[Path]:
    """Round-robin over units so the sample covers as many units as possible."""
    by_unit: "OrderedDict[str, list[Path]]" = OrderedDict()
    for path in sorted(paths, key=lambda item: (_unit_of(item), item.name)):
        by_unit.setdefault(_unit_of(path), []).append(path)
    chosen: list[Path] = []
    depth = 0
    while len(chosen) < limit:
        added = False
        for unit_paths in by_unit.values():
            if depth < len(unit_paths):
                chosen.append(unit_paths[depth])
                added = True
                if len(chosen) >= limit:
                    break
        if not added:
            break
        depth += 1
    return chosen


def _read_window(path: Path, window: int, offset: int = 0) -> np.ndarray:
    import pandas as pd

    frame = pd.read_csv(
        path,
        header=None,
        skiprows=offset,
        nrows=window,
        engine="c",
        on_bad_lines="skip",
    )
    values = frame.to_numpy(dtype=np.float64)
    if values.shape[0] < window:
        raise SystemExit(f"{path.name} has only {values.shape[0]} rows, need {window}")
    signal = values[:, -1]
    if not np.isfinite(signal).all():
        raise SystemExit(f"{path.name} contains non-finite samples")
    return signal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=2048)
    parser.add_argument("--per-class", type=int, default=250)
    parser.add_argument("--sampling-rate", type=float, default=20000.0)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format": "entropylab.dataset-source",
        "format_version": 1,
        "dataset_id": "VBL_4CLASS",
        "sampling_rate_hz": args.sampling_rate,
        "window": args.window,
        "source": str(args.source),
        "channel": "last acceleration column (axis order undocumented in the archive)",
        "classes": [],
    }

    for class_id, label, folder in CLASSES:
        files = sorted((Path(args.source) / folder).rglob("*.csv"))
        selected = _select(files, args.per_class)
        rows = []
        signals = []
        for path in selected:
            signals.append(_read_window(path, args.window))
            rows.append(
                {
                    "file": path.name,
                    "unit": _unit_of(path),
                    "sha256": _sha256(path),
                }
            )
        packed = np.stack(signals)[:, None, :]
        target = output / f"class{class_id + 1}.npy"
        with target.open("wb") as handle:
            np.save(handle, packed.astype(np.float32))
        manifest["classes"].append(
            {
                "class_id": class_id,
                "label": label,
                "source_folder": folder,
                "available_files": len(files),
                "selected_files": len(selected),
                "units": sorted({row["unit"] for row in rows}),
                "files": rows,
                "packed": target.name,
                "shape": list(packed.shape),
            }
        )
        print(
            f"{label:13s} available {len(files):4d}  selected {len(selected):4d}  "
            f"units {len(manifest['classes'][-1]['units']):3d}  -> {target.name} {packed.shape}"
        )

    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"manifest -> {output / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
