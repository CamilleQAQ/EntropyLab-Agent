#!/usr/bin/env python
"""Pack the Paderborn bearing dataset into the repository's npy-tensor form.

Source: ``Paderborn-source-verification-20261102/archives/<bearing>/<bearing>/*.mat``
(one .mat per measurement, named ``{condition}_{bearing}_{index}.mat``).

Signal choice: ``Y.vibration_1`` (the 64 kHz accelerometer, 256001 samples = 4 s).
Two non-overlapping 2048-point windows are taken per measurement, starting after a
20k-sample run-in, so a group (``group_size = 2``) is exactly one measurement.

Class labels come from the official factsheets (``Component`` field), not from the
K/KA/KI file prefix: the prefix does not encode "healthy / artificial / real".
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# class_id -> label -> bearings (all damaged bearings below were produced by
# lifetime tests, so "engraver vs lifetime" cannot become a shortcut feature).
CLASSES = (
    (0, "HEALTHY", ("K001", "K002", "K003")),
    (1, "OUTER_RING", ("KA04", "KA15", "KA16")),
    (2, "INNER_RING", ("KI04", "KI14", "KI16")),
)

CONDITIONS = ("N15_M07_F10", "N15_M01_F10", "N15_M07_F04")

WINDOW = 2048
WINDOWS_PER_MEASUREMENT = 2
OFFSET = 20000  # skip the run-in transient of the 4 s recording


def _load_vibration(path: Path) -> np.ndarray:
    from scipy.io import loadmat

    key = path.stem
    mat = loadmat(path, squeeze_me=False, struct_as_record=False)
    if key not in mat:
        raise SystemExit(f"{path.name}: variable {key!r} missing")
    record = mat[key][0, 0]
    for entry in record.Y[0]:
        if str(entry.Name[0]) == "vibration_1":
            return np.asarray(entry.Data, dtype=np.float64).ravel()
    raise SystemExit(f"{path.name}: vibration_1 missing")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="archives directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--windows-per-measurement", type=int, default=WINDOWS_PER_MEASUREMENT)
    parser.add_argument("--offset", type=int, default=OFFSET)
    parser.add_argument("--sampling-rate", type=float, default=64000.0)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "entropylab.dataset-source",
        "format_version": 1,
        "dataset_id": "PADERBORN_3CLASS",
        "sampling_rate_hz": args.sampling_rate,
        "window": args.window,
        "windows_per_measurement": args.windows_per_measurement,
        "offset": args.offset,
        "signal": "Y.vibration_1 (64 kHz accelerometer)",
        "conditions": list(CONDITIONS),
        "classes": [],
    }

    for class_id, label, bearings in CLASSES:
        rows: list[dict] = []
        signals: list[np.ndarray] = []
        for bearing in bearings:
            folder = Path(args.source) / bearing / bearing
            if not folder.is_dir():
                raise SystemExit(f"missing extracted folder: {folder}")
            for condition in CONDITIONS:
                files = sorted(folder.glob(f"{condition}_{bearing}_*.mat"))
                if not files:
                    raise SystemExit(f"no measurements for {bearing} {condition}")
                for path in files:
                    signal = _load_vibration(path)
                    for index in range(args.windows_per_measurement):
                        start = args.offset + index * args.window
                        window = signal[start : start + args.window]
                        if window.size < args.window:
                            raise SystemExit(f"{path.name}: window out of range")
                        signals.append(window)
                        rows.append(
                            {
                                "file": path.name,
                                "bearing": bearing,
                                "condition": condition,
                                "measurement": int(path.stem.rsplit("_", 1)[-1]),
                                "window_index": index,
                            }
                        )
        packed = np.stack(signals)[:, None, :].astype(np.float32)
        target = output / f"class{class_id + 1}.npy"
        with target.open("wb") as handle:
            np.save(handle, packed)
        manifest["classes"].append(
            {
                "class_id": class_id,
                "label": label,
                "bearings": list(bearings),
                "packed": target.name,
                "shape": list(packed.shape),
                "files": rows,
            }
        )
        print(f"{label:11s} bearings {bearings} -> {target.name} {packed.shape}")

    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"manifest -> {output / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
