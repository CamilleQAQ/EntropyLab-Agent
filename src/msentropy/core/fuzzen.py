"""Single-scale Fuzzy Entropy.

Fuzzy Entropy compares embedded templates with an exponential membership of
their Chebyshev distance and returns the logarithmic ratio for dimensions
m and m + 1."""

import numpy as np

from msentropy.core.sampen import _as_integral, _matlab_divide, _matlab_max
from msentropy.core.toolbox import downsample


def fuzzen(
    x: np.ndarray,
    m: int,
    r: float,
    n: float = 2,
    tau: int = 1,
) -> tuple[float, np.ndarray]:
    """Single-scale Fuzzy Entropy.

    Fuzzy Entropy compares embedded templates with an exponential membership of
    their Chebyshev distance and returns the logarithmic ratio for dimensions
    m and m + 1."""
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim > 2 or (x_arr.ndim == 2 and 1 not in x_arr.shape):
        raise ValueError(
            "fuzzen: x must be 1-D or a vector-shaped 2-D array; "
            "general matrices are not supported"
        )
    m = _as_integral(m, "fuzzen: m")
    if m < 1:
        raise ValueError(
            f"fuzzen: m must be >= 1, got {m}"
        )

    if tau > 1:
        x_arr = downsample(x_arr, _as_integral(tau, "fuzzen: tau"))

    x_flat = x_arr.ravel()
    size = x_flat.size
    nm = size - m
    width = max(nm, 0)

    x_mat = np.zeros((m + 1, width), dtype=np.float64)
    for i in range(1, m + 2):
        # Explicit width avoids negative-stop slice semantics for short input.
        start = i - 1
        x_mat[i - 1, :] = x_flat[start : start + width]

    p = np.zeros(2, dtype=np.float64)
    for k in (m, m + 1):
        count = np.zeros(width, dtype=np.float64)
        temp_mat = x_mat[:k, :]
        for i in range(1, max(size - k, 0) + 1):
            n_cols = max(nm - i, 0)
            window = temp_mat[:, i : i + n_cols]
            anchor = temp_mat[:, i - 1 : i]
            dist = _matlab_max(np.abs(window - anchor))
            # Direct calls retain non-finite results for degenerate parameters.
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                df = np.exp(-(dist**n) / r)
            count[i - 1] = _matlab_divide(np.sum(df), nm)
        p[k - m] = _matlab_divide(float(np.sum(count)), nm)

    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = float(np.log(_matlab_divide(p[0], p[1])))
    return entropy, p
