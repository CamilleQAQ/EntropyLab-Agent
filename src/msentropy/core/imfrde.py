"""Improved multiscale fuzzy Rényi dispersion entropy (IMFRDE).

This is the integrated form of the user-supplied Python migration of
``IMFuRDE.m``.  It reproduces the NCDF fuzzy cumulative Rényi implementation;
the GGD extension is intentionally not included.  The numerical path is pinned
by four independently generated MATLAB fixtures in the test suite.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr


def _positive_integer(value: object, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"IMFRDE: {name} must be a positive integer")
    return int(value)


def _pattern_pdf(
    x: np.ndarray,
    m: int,
    nc: int,
    tau: int,
    mu: float,
    sigma: float,
) -> np.ndarray:
    """Return fuzzy pattern memberships in the MATLAB lexicographic order."""
    y = ndtr((x - mu) / sigma)
    y[y == 0] = 1e-10
    y[y == 1] = 1 - 1e-10
    z = y * nc + 0.5
    membership = np.maximum(
        1 - np.abs(np.arange(1, nc + 1)[:, None] - z), 0
    )
    membership[0] = np.clip(2 - z, 0, 1)
    membership[-1] = np.clip(z - nc + 1, 0, 1)
    vector_count = len(x) - (m - 1) * tau
    pattern_count = nc**m
    pdf = np.empty(pattern_count, dtype=np.float64)
    batch_size = max(1, min(256, 1_000_000 // vector_count))
    for start in range(0, pattern_count, batch_size):
        indices = np.arange(start, min(start + batch_size, pattern_count))
        products = np.ones((len(indices), vector_count), dtype=np.float64)
        for axis in range(m):
            symbols = (indices // nc ** (m - axis - 1)) % nc
            first = axis * tau
            products *= membership[symbols, first : first + vector_count]
        pdf[start : start + len(indices)] = products.mean(axis=1)
    return pdf


def _renyi_entropy(pdf: np.ndarray, type_: int, q: float) -> float:
    values = pdf if type_ == 0 else 1 - np.cumsum(pdf)
    # MATLAB can leave a tiny negative final residual through roundoff.  It is
    # not a negative probability and is clipped before the fractional power.
    values = np.maximum(values, 0)
    values = values[values != 0]
    return float(np.log(np.sum(values**q)) / (1 - q))


def imfrde(
    x: np.ndarray,
    m: int,
    nc: int,
    tau: int,
    scale: int,
    type_: int = 0,
    q: float = 1.3,
) -> np.ndarray:
    """Compute the NCDF fuzzy cumulative Rényi IMFRDE curve.

    At scale ``s`` the method builds ``s`` shifted non-overlapping block-mean
    series, computes fuzzy Rényi dispersion entropy for each, and averages the
    results.  Every scale uses the original signal's mean and sample standard
    deviation, matching ``IMFuRDE.m``.
    """
    m = _positive_integer(m, "m")
    nc = _positive_integer(nc, "nc")
    tau = _positive_integer(tau, "tau")
    scale = _positive_integer(scale, "scale")
    if nc < 2:
        raise ValueError("IMFRDE: nc must be at least 2")
    signal = np.asarray(x, dtype=np.float64)
    if signal.ndim != 1 or len(signal) < 2 or not np.isfinite(signal).all():
        raise ValueError("IMFRDE: x must be a finite 1-D signal with at least two samples")
    if type_ not in (0, 1):
        raise ValueError("IMFRDE: type_ must be 0 or 1")
    if not np.isfinite(q) or q <= 0 or q == 1:
        raise ValueError("IMFRDE: q must be finite, positive, and different from 1")
    if (len(signal) - scale + 1) // scale <= (m - 1) * tau:
        raise ValueError("IMFRDE: signal is too short for scale, m and tau")
    sigma = float(np.std(signal, ddof=1))
    if sigma == 0 or not np.isfinite(sigma):
        raise ValueError("IMFRDE: x must have a finite, nonzero sample standard deviation")
    mu = float(np.mean(signal))
    result = np.empty(scale, dtype=np.float64)
    for current_scale in range(1, scale + 1):
        entropies = []
        for offset in range(current_scale):
            length = (len(signal) - offset) // current_scale
            coarse = signal[
                offset : offset + length * current_scale
            ].reshape(length, current_scale).mean(axis=1)
            entropies.append(
                _renyi_entropy(
                    _pattern_pdf(coarse, m, nc, tau, mu, sigma), type_, q
                )
            )
        result[current_scale - 1] = np.mean(entropies)
    return result


__all__ = ["imfrde"]
