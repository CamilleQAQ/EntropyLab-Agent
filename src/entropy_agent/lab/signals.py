"""Signal statistics per dataset: the evidence a strategy actually needs.

Dataset size and sampling rate say little about which feature family fits; the
signal itself does.  These cheap statistics (kurtosis, RMS, spectral centroid,
band ratio, Fisher separability of RMS across classes) are computed once per
dataset and fed to the strategy model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..io.dataset import load_dataset_records, load_record_signal


def signal_features(spec, data_root: Path, *, limit: int | None = None) -> dict[str, float]:
    records = load_dataset_records(spec, data_root)
    if limit is not None:
        records = records[:limit]
    kurtosis, rms, centroid, band_ratio = [], [], [], []
    labels = []
    for record in records:
        values = np.asarray(load_record_signal(record).values, dtype=float)
        centered = values - values.mean()
        std = centered.std() + 1e-12
        rms.append(float(np.sqrt((centered**2).mean())))
        kurtosis.append(float((centered**4).mean() / std**4))
        spectrum = np.abs(np.fft.rfft(centered)) ** 2
        total = spectrum.sum() + 1e-12
        freqs = np.fft.rfftfreq(centered.size, d=1.0 / float(spec.sampling_rate_hz))
        centroid.append(float((spectrum * freqs).sum() / total))
        half = spectrum.size // 2
        band_ratio.append(float(spectrum[:half].sum() / total))
        labels.append(int(record.label))

    labels_array = np.asarray(labels)
    rms_array = np.asarray(rms)
    means = [rms_array[labels_array == label].mean() for label in sorted(set(labels))]
    within = np.mean(
        [rms_array[labels_array == label].std() + 1e-12 for label in sorted(set(labels))]
    )
    between = np.std(means) + 1e-12
    return {
        "kurtosis": float(np.mean(kurtosis)),
        "rms": float(np.mean(rms)),
        "centroid": float(np.mean(centroid)),
        "band_ratio": float(np.mean(band_ratio)),
        "fisher": float(between / within),
    }


def feature_vector(stats: dict[str, float]) -> tuple[float, ...]:
    return (
        float(np.log1p(stats["kurtosis"])),
        float(np.log1p(stats["rms"])),
        float(np.log1p(stats["centroid"])),
        float(stats["band_ratio"]),
        float(np.log1p(stats["fisher"])),
    )
