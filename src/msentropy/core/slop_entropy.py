"""Single-scale Slope Entropy.

Consecutive differences are encoded into five symbols using delta and gama
thresholds. Shannon entropy is calculated from length-(m - 1) symbol patterns;
the compatibility implementation supports 2 <= m <= 5."""

from itertools import product

import numpy as np

_SYMBOLS = [-2, -1, 0, 1, 2]

#: Largest validated embedding dimension for the compatibility implementation.
_MAX_SUPPORTED_M = 5


def slop_entropy(
    x: np.ndarray,
    m: int,
    delta: float,
    gama: float,
) -> tuple[float, np.ndarray]:
    """Single-scale Slope Entropy.

    Consecutive differences are encoded into five symbols using delta and gama
    thresholds. Shannon entropy is calculated from length-(m - 1) symbol patterns;
    the compatibility implementation supports 2 <= m <= 5."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 2 or (x.ndim == 2 and 1 not in x.shape):
        raise ValueError(
            "slop_entropy: x must be 1-D or a vector-shaped 2-D array; "
            "general matrices are not supported"
        )
    if m < 2:
        raise ValueError(
            f"slop_entropy: m must be >= 2, got {m}"
        )
    if m > _MAX_SUPPORTED_M:
        raise ValueError(
            f"slop_entropy: m must be <= {_MAX_SUPPORTED_M}, got {m}"
        )
    if gama < delta:
        print("gama should be greater than delta ")

    x = x.ravel()
    diffs = np.diff(x)
    symbols = _symbol_map(diffs, delta, gama)
    patterns = _all_patterns(m)

    w = x.size - (m - 1)  # window count and probability normalizer
    if w <= 0:
        # Preserve the direct function's non-finite short-signal behavior.
        npat = patterns.shape[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            npdf = np.zeros(npat, dtype=np.float64) / w
        p = npdf[npdf != 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            slop_en = -np.sum(p * np.log(p))
        return float(slop_en), npdf

    windows = _symbol_windows(symbols, m, w)
    counts = _count_patterns(patterns, windows)
    npdf = counts / w
    p = npdf[npdf != 0]
    slop_en = -np.sum(p * np.log(p))
    return float(slop_en), npdf


def _symbol_map(diffs: np.ndarray, delta: float, gama: float) -> np.ndarray:
    """Encode consecutive differences into five slope symbols."""
    c0 = (-delta <= diffs) & (diffs <= delta)
    c1 = (delta < diffs) & (diffs <= gama)
    c2 = gama < diffs
    c3 = (-gama <= diffs) & (diffs < -delta)
    c4 = diffs < -gama
    return np.select([c0, c1, c2, c3, c4], [0, 1, 2, -1, -2], default=np.nan)


def _all_patterns(m: int) -> np.ndarray:
    """All ``5^(m-1)`` symbol patterns in ascending row order."""
    return np.array(
        list(product(_SYMBOLS, repeat=m - 1)), dtype=np.int64
    ).reshape(-1, m - 1)


def _symbol_windows(symbols: np.ndarray, m: int, w: int) -> np.ndarray:
    """Build per-window symbol rows with the established NaN semantics."""
    win = np.empty((w, m - 1), dtype=np.float64)
    buf = np.full(m - 1, np.nan)
    for j in range(w):
        for p in range(m - 1):
            s = symbols[j + p]
            if not np.isnan(s):
                buf[p] = s
        if j == 0:
            if np.isnan(buf[-1]):
                if np.all(np.isnan(buf)):
                    raise UnboundLocalError(
                        "slop_entropy: NaN differences left every symbol "
                        "of the first window unassigned"
                    )
                raise ValueError(
                    "slop_entropy: a NaN difference left the last symbol "
                    "of the first window unassigned"
                )
            # Holes below the last assigned position fill with implicit 0.
            buf[np.isnan(buf)] = 0.0
        win[j] = buf
    return win


def _count_patterns(patterns: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """Count symbol-window rows against the complete pattern set."""
    npat = patterns.shape[0]
    length = patterns.shape[1]
    weights = np.power(5, np.arange(length, dtype=np.int64))
    pattern_keys = (patterns + 2) @ weights
    window_keys = (windows + 2).astype(np.int64) @ weights
    order = np.argsort(pattern_keys)
    pos = np.searchsorted(pattern_keys[order], window_keys)
    return np.bincount(order[pos], minlength=npat)
