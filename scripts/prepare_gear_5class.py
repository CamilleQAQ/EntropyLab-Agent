#!/usr/bin/env python
"""Prepare the herringbone-gear 5-state dataset.

Source layout (one folder per health state, one CSV per run):

    7数据源/GearDatasets/GearDatasets/人字齿轮/{健康,裂纹,磨损,断齿,剥落}/*.csv

Each CSV starts with four metadata rows followed by the samples:

    line 1  ``15360,460800,10,0,3``   sampling rate, sample count, load, ?, segments
    line 2  channel maxima
    line 3  channel enable flags (last column = channel count)
    line 4  zero row
    line 5+ samples, one column per channel

Columns 1-9 are vibration channels and column 10 is a tachometer-like speed
signal.  No documentation in the archive identifies the sensor placement, so this
protocol fixes **the first vibration channel** and records that choice here
instead of implying a physical meaning it does not have.

Protocol: **every** file of each class and the first six non-overlapping
2000-point windows of each file.  Using every file matters: the archive mixes
10/15/20 Nm loads inside each class, so taking only the first few files would turn
the load setting into a class label.  Records per class therefore follow the file
counts (5/10/8/6/8), and the manifest records the load of every source file so the
confound stays visible.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SOURCE = Path(
    r"E:\Research\EntropyLab Agent\7数据源\GearDatasets\GearDatasets\人字齿轮"
)
OUT = Path(r"E:\Research\EntropyLab Agent\renovation\data\paper\gear_5state")
CHANNEL_COLUMN = 0  # first vibration channel; column 10 is the speed signal
WINDOW = 2000
WINDOWS_PER_FILE = 6
METADATA_ROWS = 4

# class id -> (label, source folder)
CLASSES = (
    (0, "HEALTHY", "健康"),
    (1, "CRACK", "裂纹"),
    (2, "WEAR", "磨损"),
    (3, "MISSING_TOOTH", "断齿"),
    (4, "SPALLING", "剥落"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_header(path: Path) -> dict:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        first = handle.readline().strip()
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 3:
        raise ValueError(f"{path}: unexpected header {first!r}")
    return {
        "sampling_rate": int(parts[0]),
        "samples": int(parts[1]),
        "header_field_3": int(parts[2]),
    }


def load_from_name(path: Path) -> int | None:
    """The archive puts the load in the file name (`-10Nm-`), not in the header."""
    match = re.search(r"(\d+)Nm", path.name)
    return int(match.group(1)) if match else None


def read_channel(path: Path, rows: int) -> np.ndarray:
    """Read ``rows`` samples of one column, skipping the metadata block."""
    values: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        for _ in range(METADATA_ROWS):
            next(reader, None)
        for row in reader:
            if len(values) >= rows:
                break
            if not row:
                continue
            values.append(float(row[CHANNEL_COLUMN]))
    if len(values) < rows:
        raise ValueError(f"{path}: only {len(values)} samples after the metadata block")
    return np.asarray(values, dtype=np.float64)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    for class_id, label, folder in CLASSES:
        directory = SOURCE / folder
        chosen = sorted(path for path in directory.glob("*.csv"))
        if len(chosen) < 3:
            raise ValueError(f"{directory}: {len(chosen)} files, need at least 3 groups")
        segments: list[np.ndarray] = []
        sources = []
        for path in chosen:
            header = read_header(path)
            needed = WINDOWS_PER_FILE * WINDOW
            signal = read_channel(path, needed)
            for index in range(WINDOWS_PER_FILE):
                segments.append(signal[index * WINDOW : (index + 1) * WINDOW])
            sources.append(
                {
                    "path": path.relative_to(SOURCE).as_posix(),
                    "sampling_rate": header["sampling_rate"],
                    "samples": header["samples"],
                    "load_nm": load_from_name(path),
                    "header_field_3": header["header_field_3"],
                    "windows_used": WINDOWS_PER_FILE,
                    "sha256": sha256(path),
                }
            )
        packed = np.stack(segments, axis=0)[:, None, :]
        target = OUT / f"class{class_id + 1}.npy"
        np.save(target, packed)
        entries.append(
            {
                "class_id": class_id,
                "label": label,
                "folder": folder,
                "records": int(packed.shape[0]),
                "windows_per_file": WINDOWS_PER_FILE,
                "sources": sources,
                "output": target.name,
                "output_shape": list(packed.shape),
            }
        )
        print(f"class{class_id + 1}.npy <= {folder} ({packed.shape[0]} x {WINDOW})")

    manifest = {
        "format": "entropylab.gear-prep",
        "format_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(SOURCE),
        "gear_type": "人字齿轮 (herringbone)",
        "channel_column": CHANNEL_COLUMN,
        "channel_note": "first vibration column; column 11 is a speed signal, placement undocumented",
        "sampling_rate_hz": 15360,
        "window_size": WINDOW,
        "windows_per_file": WINDOWS_PER_FILE,
        "files_per_class": "all files in the class folder",
        "records_per_class": "files x WINDOWS_PER_FILE (unbalanced by design)",
        "window_rule": (
            "first 6 non-overlapping 2000-point windows of every file of each "
            "class; metadata rows 1-4 skipped"
        ),
        "classes": entries,
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"manifest -> {OUT / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
