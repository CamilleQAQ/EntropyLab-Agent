#!/usr/bin/env python
"""Prepare the triaxial 7-class paper protocol.

The published protocol uses seven states of one motor test rig:

* six inner-race defects at 300 W load: 0.7, 0.9, 1.1, 1.3, 1.5, 1.7 mm;
* one healthy record (`healthy without pulley`).

The author-supplied loader `DataloaderOfTraixialBearing.m` fixes the file list
and reads the X axis (CSV column 2 / 0-based index 1).  The author-supplied
feature driver `fun_WindTurbineBearingFeatureExtraction.m` uses
`number=60, test_positon=31, N=1892`, so each class is cut into 60
non-overlapping windows of 1892 points starting at sample 0, exactly matching
`f = B(1 + N*(i-1) : i*N)`.

The shortest source record (0.7 mm inner race) holds 113,520 = 60 x 1892
samples, which is why 1892 rather than the 2000 points named in the paper text
is the window length that yields 60 windows per class.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SOURCE = Path(
    r"E:\Research\EntropyLab Agent\7数据源"
    r"\TriaxialBearingVibrationDatasetofInductionMotorunderVaryingLoad\Dataset"
)
OUT = Path(
    str(Path(__file__).resolve().parents[1] / "data/paper/triaxial_7class")
)
CHANNEL_COLUMN = 1  # X axis: column 1 is the time stamp, 2..4 are X/Y/Z
WINDOW = 1892
WINDOWS_PER_CLASS = 60

# class_id -> (label, source file relative to SOURCE)
CLASSES = (
    (0, "INNER_RACE_0.7MM", "0.7mm-bearing-faults/0.7inner-300watt-Zo8w7U.csv"),
    (1, "INNER_RACE_0.9MM", "0.9mm-bearing-faults/0.9inner-300watt.csv"),
    (2, "INNER_RACE_1.1MM", "1.1mm-bearing-faults/1.1inner-300watt.csv"),
    (3, "INNER_RACE_1.3MM", "1.3mm-bearing-faults/1.3inner-300watt.csv"),
    (4, "INNER_RACE_1.5MM", "1.5mm-bearing-faults/1.5inner-300watt.csv"),
    (5, "INNER_RACE_1.7MM", "1.7mm-bearing-faults/1.7inner-300watt.csv"),
    (6, "HEALTHY", "Healthy bearing data/healthy without pulley.csv"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_axis(path: Path) -> np.ndarray:
    """Read one CSV axis column, skipping the header row."""
    values: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if len(header) <= CHANNEL_COLUMN:
            raise ValueError(f"{path}: expected at least four columns, got {header}")
        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            token = row[CHANNEL_COLUMN].strip()
            if not token:
                raise ValueError(f"{path}: empty value at line {line_number}")
            values.append(float(token))
    return np.asarray(values, dtype=np.float64)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    for class_id, label, relative in CLASSES:
        source = SOURCE / relative
        signal = read_axis(source)
        available = int(signal.size // WINDOW)
        if available < WINDOWS_PER_CLASS:
            raise ValueError(
                f"{source}: {signal.size} samples yield {available} complete "
                f"windows, need {WINDOWS_PER_CLASS}"
            )
        segments = [
            signal[index * WINDOW : (index + 1) * WINDOW]
            for index in range(WINDOWS_PER_CLASS)
        ]
        if any(segment.size != WINDOW for segment in segments):
            raise ValueError(f"{source}: a window is shorter than {WINDOW} samples")
        packed = np.stack(segments, axis=0)[:, None, :]
        target = OUT / f"class{class_id + 1}.npy"
        np.save(target, packed)
        entries.append(
            {
                "class_id": class_id,
                "label": label,
                "source": relative,
                "source_rows": int(signal.size),
                "source_sha256": sha256(source),
                "complete_windows_available": available,
                "windows_used": WINDOWS_PER_CLASS,
                "window_size": WINDOW,
                "output": target.name,
                "output_shape": list(packed.shape),
            }
        )
        print(
            f"class{class_id + 1}.npy <= {relative} "
            f"({packed.shape[0]} x {WINDOW}, {available} windows available)"
        )

    manifest = {
        "format": "entropylab.triaxial-prep",
        "format_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(SOURCE),
        "axis": "X",
        "csv_column": CHANNEL_COLUMN,
        "window_size": WINDOW,
        "windows_per_class": WINDOWS_PER_CLASS,
        "window_rule": (
            "first 60 non-overlapping windows from sample 0, matching "
            "MATLAB f = B(1 + N*(i-1) : i*N)"
        ),
        "loader_reference": "DataloaderOfTraixialBearing.m",
        "driver_reference": "fun_WindTurbineBearingFeatureExtraction.m",
        "classes": entries,
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"manifest -> {(OUT / 'manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
