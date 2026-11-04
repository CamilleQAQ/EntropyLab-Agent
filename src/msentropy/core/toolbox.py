"""Numerical helpers shared by entropy implementations.

The module contains amplitude mappings, activation functions, downsampling,
NaN-aware aggregation, and shared pattern-counting operations."""

import numpy as np
from scipy import special


def mapminmax(
    x: np.ndarray, ymin: float = -1.0, ymax: float = 1.0
) -> np.ndarray:
    """Map the minimum and maximum values of a 1-D array to ``[ymin, ymax]``."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        rows = x.reshape(1, -1)
    elif x.ndim == 2:
        rows = x
    else:
        raise ValueError(
            "mapminmax: only 1-D and 2-D inputs are supported "
            "(MATLAB processes each row of a matrix as one signal)"
        )
    if rows.size == 0:
        return x.copy()

    # Per-row min/max IGNORING NaN: NaN is replaced by +/-inf so that an
    # all-NaN row gets an infinite range and falls into the fix-up
    # branch, exactly like the historical reference scalar-path behavior.
    rows_min = np.min(np.where(np.isnan(rows), np.inf, rows), axis=1)
    rows_max = np.max(np.where(np.isnan(rows), -np.inf, rows), axis=1)

    xrange_ = rows_max - rows_min
    yrange = ymax - ymin
    # A constant row gives 0/0 here; MATLAB computes the same Inf/NaN
    # silently (it does not warn on double division). The fix-up branch
    # below replaces the result anyway, so silence the numpy warning.
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = yrange / xrange_
    # MATLAB's fix-up ("no change") per row.
    fix = (np.abs(gain) > 1e14) | ~np.isfinite(xrange_) | (xrange_ == 0)
    gain = np.where(fix, 1.0, gain)
    xoffset = np.where(fix, ymin, rows_min)
    y = (rows - xoffset[:, np.newaxis]) * gain[:, np.newaxis] + ymin
    return y.reshape(x.shape)


def normcdf(x: np.ndarray, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    """Normal cumulative distribution function."""
    x = np.asarray(x, dtype=np.float64)
    if sigma < 0:
        return np.full(x.shape, np.nan)
    if sigma == 0:
        return (x >= mu).astype(np.float64)
    z = (x - mu) / sigma
    return 0.5 * special.erfc(-z / np.sqrt(2.0))


def logsig(n: np.ndarray) -> np.ndarray:
    """Log-sigmoid transfer function."""
    n = np.asarray(n, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-n))


def tansig(n: np.ndarray) -> np.ndarray:
    """Hyperbolic-tangent sigmoid transfer function."""
    n = np.asarray(n, dtype=np.float64)
    return 2.0 / (1.0 + np.exp(-2.0 * n)) - 1.0


def nanmean_columns(a: np.ndarray) -> np.ndarray:
    """Column-wise mean ignoring NaN, i.e. MATLAB's ``nanmean(a)`` on a matrix."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(
            f"nanmean_columns: expected a 2-D matrix, got shape {a.shape}"
        )
    total = np.zeros(a.shape[1])
    seeded = np.zeros(a.shape[1], dtype=bool)
    count = np.zeros(a.shape[1], dtype=np.intp)
    for row in a:
        finite = ~np.isnan(row)
        first = finite & ~seeded
        seeded |= finite
        # Seed the accumulator with the first non-NaN value of each
        # column, then add the remaining ones.
        total = np.where(first, row, total)
        total = np.where(finite & ~first, total + row, total)
        count += finite
    # An all-NaN column gives 0/0 = NaN, silently, as in MATLAB.
    with np.errstate(divide="ignore", invalid="ignore"):
        return total / count


def downsample(x: np.ndarray, n: int, phase: int = 0) -> np.ndarray:
    """Keep every ``n``-th sample of ``x``, starting with sample ``phase + 1``."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        raise ValueError(
            "downsample: empty input is not allowed "
            "(MATLAB: signal:downsample:Nonempty)"
        )
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(
            f"downsample: n must be a positive integer, got {n!r} "
            "(MATLAB: signal:downsample:expectedInteger)"
        )
    if not isinstance(phase, (int, np.integer)) or not 0 <= phase <= n - 1:
        raise ValueError(
            f"downsample: phase must be an integer in [0, {n - 1}], "
            f"got {phase!r}"
        )

    if x.ndim == 0:
        # A MATLAB scalar is 1x1; the all-singleton fallback below then
        # leaves it untouched, matching the reference behavior.
        x = x.reshape(1, 1)

    size_x = list(x.shape)
    dim = 0
    for d, s in enumerate(size_x):
        if s != 1:
            dim = d
            break

    # circshift(sizeX, -(dim-1)): rotate the reduced axis to the front.
    shifted = np.roll(size_x, -dim)
    ytemp = x.reshape(tuple(int(v) for v in shifted), order="F")
    ytemp1 = ytemp[phase::n]

    size_x[dim] = ytemp1.shape[0]
    return ytemp1.reshape(tuple(size_x), order="F")
