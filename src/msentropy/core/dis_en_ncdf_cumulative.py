# Adapted from MATLAB code by Hamed Azami and Javier Escudero.
# MATLAB-to-Python translation and modifications by CamilleQAQ.
# SPDX-License-Identifier: CC-BY-4.0

"""Dispersion Entropy with a fixed NCDF mapping.

Samples are mapped to discrete classes using normal-CDF coordinates, embedded
into length-m patterns, and summarized with Shannon entropy."""

import numpy as np

from msentropy.core import toolbox


def _enumerate_patterns(nc: int, m: int) -> np.ndarray:
    """Pattern enumeration matrix ``v`` (nc**m x m), MATLAB order."""
    idx0 = np.arange(nc**m)[:, np.newaxis]
    powers = nc ** np.arange(m)[np.newaxis, :]
    return (idx0 // powers) % nc + 1


def _pattern_keys(nc: int, m: int, base: float) -> np.ndarray:
    """Positional keys of every pattern, mirroring MATLAB's ``key`` loop."""
    v_mat = _enumerate_patterns(nc, m).astype(np.float64)
    weights = base ** np.arange(m - 1, -1, -1)
    return v_mat @ weights


def _window_codes(z: np.ndarray, m: int, tau: int, base: float = 10.0) -> np.ndarray:
    """Base-``base`` embedding codes of every window, mirroring ``embd2``."""
    n = z.size
    w = n - (m - 1) * tau
    if w <= 0:
        return np.zeros(0)
    codes = np.zeros(w)
    for i in range(m):
        start = i * tau
        codes = z[start : start + w] * (base ** (m - 1 - i)) + codes
    return codes


def _count_patterns(
    z: np.ndarray, nc: int, m: int, tau: int, base: float = 10.0
) -> np.ndarray:
    """Pattern distribution ``npdf`` from the integer class vector ``z``."""
    key = _pattern_keys(nc, m, base)
    codes = _window_codes(z, m, tau, base)

    w = z.size - (m - 1) * tau
    # MATLAB counts with `find(embd2 == key(id))`, so a code that is not
    # exactly one of the keys contributes to NO pattern. Two kinds of code
    # can miss: NaN/Inf (from NaN/Inf mapped values), and — for ``DispEn``
    # but not for the NCDF_Cumulative family — an integer OUTSIDE the class
    # grid. ``DispEn`` reaches ``z == nc + 1`` through a TIE, not through a
    # ``mapminmax`` overshoot: its maximum mapped value is ``1 - 2**-53``
    # (never 1, so the ``y == 1`` protection does not fire), and for a power
    # of two ``nc`` the sum ``y*nc + 0.5`` lands exactly on a half integer,
    # which MATLAB's ``round`` takes away from zero. Measured per case in
    # Compatibility tests cover codes that fall outside the class grid.
    # Every window containing such a sample is silently lost. Blind
    # ``np.searchsorted`` would park these codes on the largest key, so the
    # membership is tested explicitly and only real hits are counted.
    codes = codes[np.isfinite(codes)]
    order = np.argsort(key)
    positions = np.searchsorted(key[order], codes)
    positions_clipped = np.clip(positions, 0, nc**m - 1)
    hit = (positions < nc**m) & (key[order[positions_clipped]] == codes)
    count = np.bincount(
        order[positions_clipped[hit]].astype(np.intp), minlength=nc**m
    )
    pdf = count.astype(np.float64)
    # W == 0 gives 0/0 = NaN; MATLAB raises no warning for this division,
    # so neither should Python.
    with np.errstate(divide="ignore", invalid="ignore"):
        return pdf / w


def _entropy_from_y(
    y: np.ndarray, nc: int, m: int, tau: int, type_: int
) -> tuple[float, np.ndarray]:
    """Class assignment, pattern counting and entropy from the mapped ``y``."""
    # Boundary protection with 1e-10 (NOT machine epsilon).
    y = np.where(y == 1.0, 1.0 - 1e-10, y)
    y = np.where(y == 0.0, 1e-10, y)

    # Class assignment. MATLAB's round() halves away from zero; for the
    # always-positive value v = y*nc + 0.5 that is equivalent to comparing
    # the exact fractional part against 0.5 (a plain np.floor(v + 0.5)
    # would mis-round values a fraction of an ulp below a tie, where the
    # double v + 0.5 already rounds up).
    v = y * nc + 0.5
    frac = v - np.floor(v)
    z = np.floor(v) + (frac >= 0.5)

    # Pattern enumeration, decadic window codes, counting.
    npdf = _count_patterns(z, nc, m, tau, 10.0)

    # Entropy. MATLAB branches on sign(type).
    if np.sign(type_) == 0:
        p = npdf[npdf != 0.0]  # NaN != 0 is True here, as in MATLAB
        out = -np.sum(p * np.log(p))
    elif np.sign(type_) > 0:
        raise NotImplementedError(
            "type > 0 requires cumulativeFunc, which is missing from the "
            "historical references and dies in MATLAB with an undefined-function "
            "error; only type = 0 "
            "is implemented"
        )
    else:
        raise ValueError(
            "type < 0 hits the 'fprint' typo in the MATLAB original "
            "(the invalid-branch policy); Python raises "
            "ValueError instead of an undefined-name error"
        )

    return float(out), npdf


def dis_en_ncdf_cumulative(
    x: np.ndarray,
    m: int,
    nc: int,
    tau: int,
    type_: int,
) -> tuple[float, np.ndarray]:
    """Dispersion Entropy with a fixed NCDF mapping.

    Samples are mapped to discrete classes using normal-CDF coordinates, embedded
    into length-m patterns, and summarized with Shannon entropy."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 and not (x.ndim == 2 and x.shape[0] == 1):
        raise ValueError(
            "dis_en_ncdf_cumulative: x must be 1-D or a (1, N) row; "
            "general matrices and (N, 1) columns are not supported "
            "(MATLAB column input hits an implicit-broadcasting quirk "
            "inside the embedding loop)"
        )
    z_all = x.ravel()
    n = z_all.size

    # MATLAB's mean([]) / std([]) are NaN, produced silently; numpy
    # warns on empty input, so handle the empty case explicitly.
    if n == 0:
        mu_x = np.nan
        sigma_x = np.nan
    elif n == 1:
        # MATLAB's std of a single sample is exactly 0, while
        # np.std(ddof=1) gives NaN and warns. With sigma = 0,
        # the NCDF mapping takes its Heaviside path exactly like MATLAB.
        mu_x = np.mean(z_all)
        sigma_x = 0.0
    else:
        mu_x = np.mean(z_all)
        sigma_x = np.std(z_all, ddof=1)

    y = toolbox.normcdf(z_all, mu_x, sigma_x)
    return _entropy_from_y(y, nc, m, tau, type_)
