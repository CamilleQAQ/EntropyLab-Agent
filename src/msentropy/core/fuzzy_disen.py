
"""Single-scale Fuzzy Dispersion Entropy.

The signal is mapped to fuzzy class memberships, embedded into patterns, and
summarized with Shannon entropy."""

import numpy as np

from msentropy.core import toolbox
from msentropy.core.membership import trapezoidal_mf, triangle_mf


def fuzzy_disen(
    x: np.ndarray,
    m: int,
    nc: int,
    ma: str,
    tau: int,
    type_: int,
) -> tuple[float, np.ndarray]:
    """Single-scale Fuzzy Dispersion Entropy.

    The signal is mapped to fuzzy class memberships, embedded into patterns, and
    summarized with Shannon entropy."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 2 or (x.ndim == 2 and 1 not in x.shape):
        raise ValueError(
            "fuzzy_disen: x must be 1-D (MATLAB row convention) or a "
            "vector-shaped 2-D array (1xN or Nx1); general matrices are "
            "not supported"
        )
    n = x.size

    # Empty-input moments are NaN; handle them explicitly to avoid warnings.
    if n == 0:
        mu_x = np.nan
        sigma_x = np.nan
    elif n == 1:
        # MATLAB's std of a single sample is exactly 0 (hex
        # 0000000000000000), while np.std(ddof=1) gives NaN and warns.
        # With sigma = 0 the NCDF mapping takes its Heaviside path
        # exactly like MATLAB, and only LOGSIG/TANSIG fail on 0/0.
        mu_x = np.mean(x)
        sigma_x = 0.0
    else:
        sigma_x = np.std(x, ddof=1)
        mu_x = np.mean(x)

    # Step 1: mapping to class coordinates (z is NOT rounded here).
    if ma == "LM":
        y = toolbox.mapminmax(x, 0.0, 1.0)
    elif ma == "NCDF":
        y = toolbox.mapminmax(toolbox.normcdf(x, mu_x, sigma_x), 0.0, 1.0)
    elif ma == "LOGSIG":
        # A constant signal gives (x-mu)/sigma = 0/0 = NaN here; MATLAB
        # produces it silently, so silence the numpy warning. The NaN
        # then reproduces MATLAB's failure inside trapezoidal_MF.
        with np.errstate(invalid="ignore"):
            y = toolbox.mapminmax(toolbox.logsig((x - mu_x) / sigma_x), 0.0, 1.0)
    elif ma == "TANSIG":
        with np.errstate(invalid="ignore"):
            y = toolbox.mapminmax(
                toolbox.tansig((x - mu_x) / sigma_x) + 1.0, 0.0, 1.0
            )
    elif ma == "SORT":
        raise NotImplementedError(
            "fuzzy_disen: the SORT mapping is not needed by the TSMEFuDE "
            "main chain and its MATLAB code has a known bug "
            "(the supported mapping policy)"
        )
    else:
        raise ValueError(
            f"fuzzy_disen: unknown mapping method {ma!r}; MATLAB dies "
            "with an unrecognized variable 'z' in this case"
        )

    # Boundary protection: machine epsilon, exactly like the historical reference.
    eps_value = np.finfo(np.float64).eps
    y = np.where(y == 1.0, 1.0 - eps_value, y)
    y = np.where(y == 0.0, eps_value, y)
    z = y * nc + 0.5
    return _fuzzy_pdf_entropy_from_z(z, nc, m, tau, type_)


def _fuzzy_pdf_entropy_from_z(
    z: np.ndarray, nc: int, m: int, tau: int, type_: int
) -> tuple[float, np.ndarray]:
    """Fuzzy memberships, pattern enumeration, u_pi and entropy from z."""
    n = z.size

    # Step 2: fuzzy membership matrix u_M (nc x N). Edge classes use the
    # trapezoidal shape, interior classes the triangular one. The flat
    # index reproduces MATLAB's scalar z(r) on both row and column z.
    u_m = np.full((nc, n), np.nan)
    for k in range(1, nc + 1):
        if k == 1 or k == nc:
            for r in range(n):
                u_m[k - 1, r] = trapezoidal_mf(k, z.flat[r], nc)
        else:
            for r in range(n):
                u_m[k - 1, r] = triangle_mf(k, z.flat[r])

    # Step 3: pattern enumeration matrix v (nc**m x m). Column m-k+1 (in
    # 1-based terms) holds class digits with period nc**k.
    #
    # MATLAB's repmat(1:nc, nc^(k-1), 1) stacks the row 1:nc VERTICALLY
    # nc^(k-1) times, and reshape(..., [], 1) then flattens column-major,
    # which repeats each value nc^(k-1) times consecutively. That is
    # np.repeat, NOT np.tile (np.tile concatenates whole copies).
    v = np.full((nc**m, m), np.nan)
    for k in range(1, m + 1):
        a = np.repeat(np.arange(1, nc + 1), nc ** (k - 1))
        temp_a = a.reshape(-1, 1)
        v[:, m - k] = np.tile(temp_a, (nc ** (m - k), 1)).ravel()

    # Steps 4-5: pattern memberships and the pdf. A negative window count
    # produces an empty array in MATLAB (ones(n, -1) -> n x 0), so the
    # same clamping is applied here; the normalization still divides by
    # the true W below.
    n_windows = n - (m - 1) * tau
    u_pi = np.ones((nc**m, max(n_windows, 0)))
    v_idx = v.astype(np.intp) - 1  # 1-based MATLAB class numbers -> rows
    for i in range(m):
        u_pi *= u_m[v_idx[:, i], i * tau : i * tau + u_pi.shape[1]]

    u_pi_sum = np.sum(u_pi, axis=1)
    # W == 0 gives 0/0 = NaN; MATLAB raises no warning for this division,
    # so neither should Python.
    with np.errstate(divide="ignore", invalid="ignore"):
        npdf = u_pi_sum / n_windows

    # Step 6: entropy. MATLAB branches on sign(type).
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
            "type < 0 hits the 'fprint' typo in the historical MATLAB reference "
            "(the invalid-branch policy); Python raises "
            "ValueError instead of an undefined-name error"
        )

    return float(out), npdf
