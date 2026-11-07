#!/usr/bin/env python
"""Prepare the aero-engine 5-state paper protocol.

Paper protocol: 5 states (2 healthy + 3 fault), 60 non-overlapping samples of
2000 points per state, TSMEFuDE kmax=20, BT-LSTSVM.  The source files are
(records, 8 channels, 20480 samples).  This script selects one channel and the
first 60 complete 2000-point windows per state.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


SOURCE = Path(r"E:\Research\EntropyLab Agent\7数据源\Aero engine bearing fault diagnosis\dataset")
OUT = Path(__file__).resolve().parents[1] / "data/paper/aero_engine_5state"
CHANNEL = 2
WINDOW = 2000
SAMPLES_PER_STATE = 60


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for index in range(1, 6):
        source = SOURCE / f"data{index}.npy"
        array = np.load(source, mmap_mode="r")
        if array.ndim != 3 or array.shape[1] <= CHANNEL:
            raise ValueError(f"unexpected shape {array.shape} for {source}")
        segments: list[np.ndarray] = []
        for record in array[:, CHANNEL, :]:
            for start in range(0, record.size - WINDOW + 1, WINDOW):
                segments.append(np.asarray(record[start : start + WINDOW], dtype=np.float64))
                if len(segments) == SAMPLES_PER_STATE:
                    break
            if len(segments) == SAMPLES_PER_STATE:
                break
        if len(segments) != SAMPLES_PER_STATE:
            raise ValueError(f"{source}: only {len(segments)} complete windows")
        packed = np.stack(segments, axis=0)[:, None, :]
        target = OUT / f"state{index}.npy"
        np.save(target, packed)
        print(f"{target.name}: {packed.shape} {packed.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
