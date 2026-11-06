#!/usr/bin/env python
"""Pack the CWRU 12k drive-end bearing data into the repository's npy form.

Source: ``data-sources/cwru`` (github.com/s-whynot/CWRU-dataset).

* classes: NORMAL / BALL / INNER_RACE / OUTER_RACE;
* signal: the ``X*_DE_time`` channel (drive-end accelerometer, 12 kHz);
* windows: ``--windows-per-file`` non-overlapping 2048-point windows per recording;
* group: one recording (``group_size`` equals the window count), so a file never
  straddles a split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CLASSES = (
    (0, "NORMAL", "Normal"),
    (1, "BALL", "12k_Drive_End_Bearing_Fault_Data/B"),
    (2, "INNER_RACE", "12k_Drive_End_Bearing_Fault_Data/IR"),
    (3, "OUTER_RACE", "12k_Drive_End_Bearing_Fault_Data/OR"),
)

WINDOW = 2048
WINDOWS_PER_FILE = 6


def _load(channel: Path) -> tuple[np.ndarray, str]:
    from scipy.io import loadmat

    mat = loadmat(channel)
    for key, value in mat.items():
        if key.endswith("_DE_time"):
            return np.asarray(value, dtype=np.float64).ravel(), key
    # Normal recordings use a plain ``X097_DE_time``-style key as well; if the
    # drive-end channel is missing, fail loudly instead of guessing.
    raise SystemExit(f"{channel.name}: no *_DE_time channel")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--windows-per-file", type=int, default=WINDOWS_PER_FILE)
    parser.add_argument("--sampling-rate", type=float, default=12000.0)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "entropylab.dataset-source",
        "format_version": 1,
        "dataset_id": "CWRU_4CLASS",
        "sampling_rate_hz": args.sampling_rate,
        "window": args.window,
        "windows_per_file": args.windows_per_file,
        "signal": "X*_DE_time (12 kHz drive-end accelerometer)",
        "classes": [],
    }

    for class_id, label, folder in CLASSES:
        files = sorted((Path(args.source) / folder).rglob("*.mat"))
        if not files:
            raise SystemExit(f"no mat files under {folder}")
        windows: list[np.ndarray] = []
        rows: list[dict] = []
        for path in files:
            signal, key = _load(path)
            for index in range(args.windows_per_file):
                start = index * args.window
                if start + args.window > signal.size:
                    break
                windows.append(signal[start : start + args.window])
                rows.append(
                    {
                        "file": str(path.relative_to(Path(args.source))).replace("\\", "/"),
                        "group": path.stem,
                        "channel": key,
                        "load": int(path.stem.rsplit("_", 1)[-1]) if "_" in path.stem else None,
                        "severity": path.parent.name,
                    }
                )
        packed = np.stack(windows)[:, None, :].astype(np.float32)
        target = output / f"class{class_id + 1}.npy"
        with target.open("wb") as handle:
            np.save(handle, packed)
        manifest["classes"].append(
            {
                "class_id": class_id,
                "label": label,
                "folder": folder,
                "files": len(files),
                "records": len(rows),
                "groups": len({row["group"] for row in rows}),
                "packed": target.name,
                "shape": list(packed.shape),
                "entries": rows,
            }
        )
        print(f"{label:11s} files={len(files):3d} records={len(rows):4d} -> {target.name} {packed.shape}")

    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"manifest -> {output / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
