"""Deterministic windowed feature engineering for research experiments.

This module is deliberately independent of the entropy/classifier math: it only
computes observable time, spectral, envelope, wavelet-packet and (optional)
entropy descriptors, then aggregates them per record.  No labels or test data
are used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.signal import hilbert

from . import entropy_backend

# Named, immutable feature recipes selectable through DatasetSpec.feature.method.
# The method string is a feature-pipeline identifier, not an entropy method; the
# training/diagnosis code dispatches on it before touching the entropy backend.
FUSION_METHODS: dict[str, tuple[str, ...]] = {
    "FUSION_TIME": ("time",),
    "FUSION_TIME_ENVELOPE": ("time", "envelope"),
    "FUSION_TIME_SPEC": ("time", "spectral"),
    "FUSION_TIME_ENVELOPE_ENTROPY": ("time", "envelope", "entropy"),
    "FUSION_ALL": ("time", "spectral", "envelope", "wavelet", "entropy"),
}


def families_for_fusion(method: str) -> tuple[str, ...]:
    try:
        return FUSION_METHODS[method]
    except KeyError as exc:
        raise ValueError(f"unknown fusion feature method {method!r}") from exc


DEFAULT_FUSION_ENTROPY_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "mde3",
        "method": "MDE",
        "parameters": {"m": 2, "nc": 3, "tau": 1, "scale": 3},
    },
    {
        "name": "mde5",
        "method": "MDE",
        "parameters": {"m": 2, "nc": 5, "tau": 1, "scale": 5},
    },
    {
        "name": "rcmde3",
        "method": "RCMDE",
        "parameters": {"m": 2, "nc": 3, "tau": 1, "scale": 3},
    },
)


@dataclass(frozen=True)
class WindowSpec:
    size: int
    stride: int
    max_windows: int | None = None

    def __post_init__(self) -> None:
        if self.size < 2 or self.stride < 1:
            raise ValueError("window size must be >=2 and stride >=1")
        if self.max_windows is not None and self.max_windows < 1:
            raise ValueError("max_windows must be positive or None")


def window_starts(length: int, spec: WindowSpec) -> list[int]:
    """Evenly sub-sample a full record while keeping the last window."""
    if length < spec.size:
        return []
    starts = list(range(0, length - spec.size + 1, spec.stride))
    if spec.max_windows is None or len(starts) <= spec.max_windows:
        return starts
    positions = np.linspace(0, len(starts) - 1, spec.max_windows).round().astype(int)
    selected = [starts[index] for index in sorted(set(positions.tolist()))]
    if selected[-1] != starts[-1]:
        selected[-1] = starts[-1]
    return selected


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _stats(values: np.ndarray, prefix: str) -> tuple[list[str], list[float]]:
    vector = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(vector))
    std = float(np.std(vector))
    centered = vector - mean
    skew = _safe_div(float(np.mean(centered**3)), (std**3) if std > 0 else 0.0)
    kurtosis = _safe_div(float(np.mean(centered**4)), (std**4) if std > 0 else 0.0)
    peak = float(np.max(np.abs(vector)))
    rms = float(np.sqrt(np.mean(vector**2)))
    abs_mean = float(np.mean(np.abs(vector)))
    sqrt_mean = float(np.mean(np.sqrt(np.abs(vector))))
    diff = np.diff(vector)
    mobility = _safe_div(float(np.std(diff)), std)
    complexity = _safe_div(
        _safe_div(float(np.std(np.diff(diff))), float(np.std(diff))), mobility
    )
    pairs = [
        (f"{prefix}_mean", mean),
        (f"{prefix}_std", std),
        (f"{prefix}_rms", rms),
        (f"{prefix}_skew", skew),
        (f"{prefix}_kurtosis", kurtosis),
        (f"{prefix}_peak", peak),
        (f"{prefix}_peak_to_peak", float(np.max(vector) - np.min(vector))),
        (f"{prefix}_crest_factor", _safe_div(peak, rms)),
        (f"{prefix}_shape_factor", _safe_div(rms, abs_mean)),
        (f"{prefix}_impulse_factor", _safe_div(peak, abs_mean)),
        (f"{prefix}_clearance_factor", _safe_div(peak, sqrt_mean**2)),
        (
            f"{prefix}_zero_crossing_rate",
            float(np.mean(np.diff(np.signbit(vector)))),
        ),
        (f"{prefix}_line_length", float(np.mean(np.abs(diff)))),
        (f"{prefix}_hjorth_mobility", mobility),
        (f"{prefix}_hjorth_complexity", complexity),
    ]
    return [name for name, _ in pairs], [value for _, value in pairs]


def _band_ratios(power: np.ndarray, freqs: np.ndarray, count: int, prefix: str):
    total = float(np.sum(power))
    edges = np.geomspace(max(float(freqs[0]), 1.0), float(freqs[-1]), count + 1)
    names: list[str] = []
    values: list[float] = []
    for index in range(count):
        low, high = edges[index], edges[index + 1]
        mask = (freqs >= low) & (freqs < high) if index < count - 1 else (freqs >= low)
        energy = float(np.sum(power[mask]))
        names.append(f"{prefix}_band_{index}_ratio")
        values.append(_safe_div(energy, total))
    return names, values


def _spectral_features(values: np.ndarray, sampling_rate: float, prefix: str):
    vector = np.asarray(values, dtype=np.float64)
    vector = vector - np.mean(vector)
    taper = np.hanning(len(vector)) if len(vector) > 2 else np.ones(len(vector))
    spectrum = np.fft.rfft(vector * taper)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(len(vector), d=1.0 / sampling_rate)
    total = float(np.sum(power))
    if total <= 0 or len(freqs) < 4:
        names = [
            f"{prefix}_total_log",
            f"{prefix}_centroid",
            f"{prefix}_spread",
            f"{prefix}_skew",
            f"{prefix}_kurtosis",
            f"{prefix}_entropy",
            f"{prefix}_flatness",
            f"{prefix}_rolloff85",
            f"{prefix}_dominant_freq",
            f"{prefix}_dominant_ratio",
            f"{prefix}_high_low_ratio",
        ]
        return names, [0.0] * len(names)
    probability = power / total
    centroid = float(np.sum(freqs * probability))
    spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * probability)))
    skew = _safe_div(
        float(np.sum(((freqs - centroid) ** 3) * probability)),
        spread**3,
    )
    kurt = _safe_div(
        float(np.sum(((freqs - centroid) ** 4) * probability)),
        spread**4,
    )
    positive = probability[probability > 0]
    entropy = _safe_div(
        float(-np.sum(positive * np.log(positive))), float(np.log(len(probability)))
    )
    geometric = float(np.exp(np.mean(np.log(np.maximum(power, 1e-300)))))
    flatness = _safe_div(geometric, float(np.mean(power)))
    cumulative = np.cumsum(power)
    rolloff = float(freqs[int(np.searchsorted(cumulative, 0.85 * total))])
    dominant_index = int(np.argmax(power[1:]) + 1) if len(power) > 1 else 0
    band_names, band_values = _band_ratios(power[1:], freqs[1:], 8, prefix)
    nyquist = float(freqs[-1])
    mid = float(np.median(freqs))
    high_low = _safe_div(
        float(np.sum(power[freqs >= mid])), float(np.sum(power[freqs < mid]))
    )
    names = [
        f"{prefix}_total_log",
        f"{prefix}_centroid",
        f"{prefix}_spread",
        f"{prefix}_skew",
        f"{prefix}_kurtosis",
        f"{prefix}_entropy",
        f"{prefix}_flatness",
        f"{prefix}_rolloff85",
        f"{prefix}_dominant_freq",
        f"{prefix}_dominant_ratio",
        f"{prefix}_high_low_ratio",
        *band_names,
    ]
    values = [
        float(np.log1p(total)),
        centroid,
        spread,
        skew,
        kurt,
        entropy,
        flatness,
        rolloff,
        _safe_div(float(freqs[dominant_index]), nyquist),
        _safe_div(float(power[dominant_index]), total),
        high_low,
        *band_values,
    ]
    return names, values


def _envelope_features(values: np.ndarray, sampling_rate: float, prefix: str):
    envelope = np.abs(hilbert(np.asarray(values, dtype=np.float64)))
    names, stats = _stats(envelope, f"{prefix}_env")
    spectral_names, spectral_values = _spectral_features(
        envelope, sampling_rate, f"{prefix}_env_spec"
    )
    return names + spectral_names, stats + spectral_values


def _haar_packet_energies(values: np.ndarray, level: int) -> np.ndarray:
    coefficients = [np.asarray(values, dtype=np.float64)]
    for _ in range(level):
        next_level: list[np.ndarray] = []
        for coefficient in coefficients:
            usable = coefficient[: len(coefficient) // 2 * 2]
            if usable.size < 2:
                next_level.append(np.zeros(1))
                continue
            even = usable[0::2]
            odd = usable[1::2]
            next_level.append((even + odd) / np.sqrt(2.0))
            next_level.append((even - odd) / np.sqrt(2.0))
        coefficients = next_level
    return np.asarray([float(np.sum(item**2)) for item in coefficients])


def _wavelet_features(values: np.ndarray, prefix: str):
    energies = _haar_packet_energies(values, level=3)
    total = float(np.sum(energies))
    ratios = [0.0] * len(energies) if total <= 0 else (energies / total).tolist()
    positive = [value for value in ratios if value > 0]
    entropy = (
        -sum(value * np.log(value) for value in positive) / np.log(len(ratios))
        if positive
        else 0.0
    )
    names = [f"{prefix}_band_{index}_ratio" for index in range(len(ratios))]
    names.append(f"{prefix}_entropy")
    names.append(f"{prefix}_total_log")
    return names, [*ratios, float(entropy), float(np.log1p(total))]


def _entropy_features(values: np.ndarray, configs: Sequence[Mapping[str, Any]]):
    names: list[str] = []
    values_out: list[float] = []
    for config in configs:
        method = str(config["method"])
        parameters = dict(config["parameters"])
        name = str(config.get("name", method))
        curve = np.asarray(
            entropy_backend.compute(method, np.asarray(values, dtype=np.float64), parameters),
            dtype=np.float64,
        )
        if curve.ndim != 1 or not np.isfinite(curve).all():
            raise ValueError(f"entropy config {name!r} produced invalid features")
        names.extend(f"entropy_{name}_{index}" for index in range(curve.size))
        values_out.extend(float(item) for item in curve)
    return names, values_out


def _feature_vector(
    values: np.ndarray,
    sampling_rate: float,
    *,
    families: Iterable[str],
    entropy_configs: Sequence[Mapping[str, Any]],
    prefix: str,
):
    families = set(families)
    names: list[str] = []
    vector: list[float] = []
    if "time" in families:
        block_names, block_values = _stats(values, f"{prefix}_time")
        names.extend(block_names)
        vector.extend(block_values)
    if "spectral" in families:
        block_names, block_values = _spectral_features(
            values, sampling_rate, f"{prefix}_spec"
        )
        names.extend(block_names)
        vector.extend(block_values)
    if "envelope" in families:
        block_names, block_values = _envelope_features(
            values, sampling_rate, f"{prefix}_env"
        )
        names.extend(block_names)
        vector.extend(block_values)
    if "wavelet" in families:
        block_names, block_values = _wavelet_features(values, f"{prefix}_wp")
        names.extend(block_names)
        vector.extend(block_values)
    if "entropy" in families:
        block_names, block_values = _entropy_features(values, entropy_configs)
        names.extend(f"{prefix}_{name}" for name in block_names)
        vector.extend(block_values)
    return names, vector


def extract_record_vector(
    signal: np.ndarray,
    sampling_rate: float,
    *,
    window: WindowSpec,
    families: Iterable[str],
    entropy_configs: Sequence[Mapping[str, Any]] = (),
    include_full_signal: bool = True,
) -> tuple[np.ndarray, list[str], int]:
    """Return one aggregated feature vector per record."""
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("record signal must be a finite one-dimensional array")
    starts = window_starts(len(values), window)
    if not starts:
        raise ValueError("record has no complete window")
    window_names: list[str] = []
    window_rows: list[list[float]] = []
    for start in starts:
        names, vector = _feature_vector(
            values[start : start + window.size],
            sampling_rate,
            families=families,
            entropy_configs=entropy_configs,
            prefix="win",
        )
        if not window_names:
            window_names = names
        elif names != window_names:
            raise ValueError("window feature names changed between windows")
        window_rows.append(vector)
    matrix = np.asarray(window_rows, dtype=np.float64)
    aggregated = np.concatenate(
        [
            np.mean(matrix, axis=0),
            np.std(matrix, axis=0),
            np.min(matrix, axis=0),
            np.max(matrix, axis=0),
            np.median(matrix, axis=0),
        ]
    )
    aggregate_names = [
        f"{name}_{suffix}"
        for suffix in ("mean", "std", "min", "max", "median")
        for name in window_names
    ]
    if include_full_signal:
        full_names, full_values = _feature_vector(
            values,
            sampling_rate,
            families={
                family
                for family in set(families)
                if family in {"time", "spectral", "envelope", "wavelet"}
            },
            entropy_configs=(),
            prefix="full",
        )
        aggregate_names.extend(full_names)
        aggregated = np.concatenate([aggregated, np.asarray(full_values)])
    if not np.isfinite(aggregated).all():
        raise ValueError("record feature vector contains non-finite values")
    return aggregated, aggregate_names, len(starts)


__all__ = [
    "DEFAULT_FUSION_ENTROPY_CONFIGS",
    "FUSION_METHODS",
    "WindowSpec",
    "extract_record_vector",
    "families_for_fusion",
    "window_starts",
]
