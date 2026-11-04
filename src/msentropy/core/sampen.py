"""Single-scale Sample Entropy.

Sample Entropy compares delayed templates with a Chebyshev-distance threshold
and returns the logarithmic ratio of matches for dimensions m and m + 1."""

import numpy as np

from msentropy.core.toolbox import downsample


def _matlab_max(a: np.ndarray) -> np.ndarray:
    """Maximum over the leading non-singleton dimension, ignoring NaN."""
    if a.size == 0:
        return np.empty(0, dtype=np.float64)
    values = a.ravel() if a.shape[0] == 1 else a
    masked = np.where(np.isnan(values), -np.inf, values)
    reduced = np.max(masked, axis=0)
    return np.where(np.all(np.isnan(values), axis=0), np.nan, reduced)


def _matlab_divide(numerator: float, denominator: float) -> float:
    """Floating division that preserves NaN and signed-zero edge behavior."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.float64(numerator) / np.float64(denominator))


def _as_integral(value, name: str) -> int:
    """Coerce an integer-valued scalar to ``int``, or raise."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got a bool")
    try:
        as_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if not np.isfinite(as_float) or as_float != int(as_float):
        raise ValueError(
            f"{name} must be an integer-valued scalar, got {value!r}"
        )
    return int(as_float)


def sampen(
    x: np.ndarray,
    m: int,
    r: float,
    tau: int = 1,
) -> tuple[float, np.ndarray]:
    """Single-scale Sample Entropy.

    Sample Entropy compares delayed templates with a Chebyshev-distance threshold
    and returns the logarithmic ratio of matches for dimensions m and m + 1."""
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim > 2 or (x_arr.ndim == 2 and 1 not in x_arr.shape):
        raise ValueError(
            "sampen: x must be 1-D or a vector-shaped 2-D array; "
            "general matrices are not supported"
        )
    m = _as_integral(m, "sampen: m")
    if m < 1:
        raise ValueError(
            f"sampen: m must be >= 1, got {m}"
        )

    if tau > 1:
        x_arr = downsample(x_arr, _as_integral(tau, "sampen: tau"))

    x_flat = x_arr.ravel()
    n = x_flat.size
    nm = n - m
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
        for i in range(1, max(n - k, 0) + 1):
            n_cols = max(nm - i, 0)
            window = temp_mat[:, i : i + n_cols]
            anchor = temp_mat[:, i - 1 : i]
            dist = _matlab_max(np.abs(window - anchor))
            count[i - 1] = _matlab_divide(np.count_nonzero(dist < r), nm)
        p[k - m] = _matlab_divide(float(np.sum(count)), nm)

    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = float(np.log(_matlab_divide(p[0], p[1])))
    return entropy, p
