"""Single-scale Permutation Entropy.

The signal is embedded into delayed windows, each window is replaced by its
ordinal pattern, and Shannon entropy is calculated from pattern frequencies."""

import numpy as np


def permutation_entropy(data: np.ndarray, m: int, t: int) -> float:
    """Single-scale Permutation Entropy.

    The signal is embedded into delayed windows, each window is replaced by its
    ordinal pattern, and Shannon entropy is calculated from pattern frequencies."""
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 1 and not (x.ndim == 2 and x.shape[0] == 1):
        raise ValueError(
            "permutation_entropy: x must be 1-D or a (1, N) row; "
            "general matrices and (N, 1) columns are not supported "
            "(MATLAB column input always returns 0 through an "
            "implicit-broadcasting behavior)"
        )
    if m < 1 or t < 1:
        raise ValueError(
            f"permutation_entropy: m and t must be >= 1 (got m={m}, t={t}); "
            "MATLAB silently returns 0 for m <= 0, which has "
            "no scientific meaning and is not reproduced"
        )

    x = x.ravel()
    perms_rows, iv_all, c_full = _pattern_counts(x, m, t)

    # MATLAB: hist = c; c = hist(find(hist ~= 0)); p = c/sum(c);
    # pe = -sum(p .* log(p)); the normalization line is commented out.
    nz = c_full[c_full != 0]
    p = nz / nz.sum()
    pe = -np.sum(p * np.log(p))
    return float(pe)


def _matlab_perms(n: int) -> np.ndarray:
    """Rows of MATLAB ``perms(1:n)`` in the R2023b (23.2.0) row order."""
    rows = np.array([[1]], dtype=np.int64)  # permsr starting P = 1
    for nn in range(2, n + 1):
        m_old = rows.shape[0]
        new = np.empty((nn * m_old, nn), dtype=np.int64)
        # Block 1: column 1 = nn, remaining columns = old rows.
        new[:m_old, 0] = nn
        new[:m_old, 1:] = rows
        row0 = m_old
        # Blocks i = nn-1 .. 1: column 1 = i, rest = reorder(Psmall).
        for i in range(nn - 1, 0, -1):
            reorder = np.concatenate(
                [np.arange(1, i), np.arange(i + 1, nn + 1)]
            )
            block = new[row0 : row0 + m_old]
            block[:, 0] = i
            block[:, 1:] = reorder[rows - 1]  # rows hold 1-based indices
            row0 += m_old
        rows = new
    return rows


def _pattern_counts(x: np.ndarray, m: int, t: int):
    """Per-window rank patterns and the MATLAB-ordered pattern histogram."""
    perms_rows = _matlab_perms(m)
    npat = perms_rows.shape[0]
    n = x.size
    w = n - t * (m - 1)

    if w <= 0:
        return perms_rows, np.empty((0, m), dtype=np.int64), np.zeros(npat)

    # Window matrix: row i holds x[i : i + t*(m-1)] with i = 0-based
    # start, delay t between columns (strided windowing, sample order
    # preserved).
    starts = np.arange(w, dtype=np.int64)[:, None]
    offsets = (t * np.arange(m, dtype=np.int64))[None, :]
    windows = x[starts + offsets]
    # Stable ascending argsort = MATLAB sort, 1-based index pattern.
    iv_all = np.argsort(windows, axis=1, kind="stable") + 1

    # Encode each pattern (and each perms row) as an integer key with a
    # base of m+1 per column so keys are unique; locate each window's
    # key among the sorted perms keys, then scatter the positions back
    # through the perms row order.
    weights = np.power(m + 1, np.arange(m, dtype=np.int64))
    perm_keys = perms_rows @ weights
    order = np.argsort(perm_keys)
    pos = np.searchsorted(perm_keys[order], iv_all @ weights)
    c_full = np.bincount(order[pos], minlength=npat)
    return perms_rows, iv_all, c_full
