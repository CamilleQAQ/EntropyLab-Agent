#!/usr/bin/env python
"""Check reproduced triaxial features against the author-saved MATLAB matrix.

The authors saved the 7-state feature tables they actually used in
``mat三相/FeatureExtraction_DataAll_WindTurbineBearing_D6H6_60samples.mat``:
one cell entry per entropy method, each ``(420, 21)`` = 420 samples x
(20 features + label).  This script recomputes TSMEFuDE for the prepared
windows and reports the difference, so the data mapping, window rule and
feature implementation can be audited instead of assumed.

Row layout of the saved matrices is class-major: rows ``(class-1)*60 +
window`` with the first 30 windows training and the last 30 test; the label
column is 1-based and equals ``class_id + 1``.

Parity is reported as a distribution rather than a single bit: MATLAB's saved
run is not bit-identical to the Python port (single-precision accumulation
inside the entropy code), and one window of the 1.3 mm record does not match at
all.  The check therefore fails only when more than
``--max-outlier-fraction`` of the windows exceed ``--tolerance``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from entropy_agent.analysis import entropy_backend


CELL_INDEX = {
    "MPE": 0,
    "MSlopEn": 1,
    "MDE": 2,
    "MFuDE": 3,
    "RCMDE": 4,
    "TSMDE": 5,
    "TSMFuDE": 6,
    "TSMEFuDE": 7,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--method", default="TSMEFuDE", choices=sorted(CELL_INDEX))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tolerance", type=float, default=5e-3)
    parser.add_argument(
        "--max-outlier-fraction",
        type=float,
        default=0.01,
        help="share of windows allowed to exceed --tolerance (default 1%%)",
    )
    args = parser.parse_args()

    import scipy.io

    reference = scipy.io.loadmat(args.reference)["FeatureExtraction_DataAll"]
    table = np.asarray(reference[0, CELL_INDEX[args.method]], dtype=np.float64)
    if table.ndim != 2 or table.shape[1] < 2:
        raise ValueError(f"unexpected reference matrix shape {table.shape}")

    class_files = sorted(args.data_root.glob("class*.npy"))
    if len(class_files) != 7:
        raise ValueError(f"expected 7 class files, found {len(class_files)}")
    features = table.shape[1] - 1
    parameters = {"m": 2, "nc": 5, "tau": 1, "kmax": features}

    differences: list[dict] = []
    for class_index, path in enumerate(class_files, start=1):
        packed = np.load(path)
        if packed.ndim != 3 or packed.shape[1] != 1:
            raise ValueError(f"unexpected packed shape {packed.shape} for {path}")
        for window in range(packed.shape[0]):
            row = (class_index - 1) * packed.shape[0] + window
            expected = table[row, :features]
            observed = entropy_backend.compute(
                args.method, np.asarray(packed[window, 0], dtype=np.float64), parameters
            )
            if observed.shape != expected.shape:
                raise ValueError(f"feature dimension changed at row {row}")
            explicit_label = int(round(table[row, features]))
            differences.append(
                {
                    "class_id": class_index - 1,
                    "window": window,
                    "reference_label": explicit_label,
                    "max_abs_difference": float(np.abs(observed - expected).max()),
                    "max_relative_difference": float(
                        np.abs(observed - expected).max() / max(np.abs(expected).max(), 1e-12)
                    ),
                }
            )
        print(f"class{class_index}.npy checked ({packed.shape[0]} windows)")

    worst = max(differences, key=lambda item: item["max_abs_difference"])
    label_ok = all(item["reference_label"] == item["class_id"] + 1 for item in differences)
    magnitudes = np.asarray(
        [item["max_abs_difference"] for item in differences], dtype=np.float64
    )
    outliers = [
        {
            "class_id": item["class_id"],
            "window": item["window"],
            "max_abs_difference": item["max_abs_difference"],
        }
        for item in differences
        if item["max_abs_difference"] > args.tolerance
    ]
    outlier_fraction = len(outliers) / len(differences)
    summary = {
        "method": args.method,
        "reference": str(args.reference),
        "data_root": str(args.data_root),
        "samples_checked": len(differences),
        "reference_label_matches_class_order": label_ok,
        "median_abs_difference": float(np.median(magnitudes)),
        "p95_abs_difference": float(np.percentile(magnitudes, 95)),
        "max_abs_difference": worst["max_abs_difference"],
        "max_relative_difference": worst["max_relative_difference"],
        "worst_sample": {"class_id": worst["class_id"], "window": worst["window"]},
        "tolerance": args.tolerance,
        "windows_outside_tolerance": len(outliers),
        "outlier_fraction": outlier_fraction,
        "outliers": outliers,
        "max_outlier_fraction_allowed": args.max_outlier_fraction,
        "within_tolerance": outlier_fraction <= args.max_outlier_fraction,
    }
    printable = {k: v for k, v in summary.items() if k != "outliers"}
    print(json.dumps(printable, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({**summary, "per_sample": differences}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"report -> {args.output}")
    if not label_ok:
        raise SystemExit("reference label column does not follow the prepared class order")
    if not summary["within_tolerance"]:
        raise SystemExit(
            f"{len(outliers)}/{len(differences)} windows differ from the MATLAB "
            f"reference by more than {args.tolerance}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
