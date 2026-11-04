"""Single-scale Dispersion Entropy with selectable mappings.

The signal is mapped into discrete classes, embedded into patterns, and
summarized with Shannon entropy."""

import numpy as np

from msentropy.core import toolbox
from msentropy.core.dis_en_ncdf_cumulative import _count_patterns

#: Implemented mappings; the unused SORT branch is not supported.
MAPPINGS = ("LM", "NCDF", "LOGSIG", "TANSIG")


def _sample_moments(x: np.ndarray) -> tuple[float, float]:
    """MATLAB's ``mean(x)`` and ``std(x)`` for the NCDF/sigmoid mappings."""
    n = x.size
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(np.mean(x)), 0.0
    return float(np.mean(x)), float(np.std(x, ddof=1))


def _map_to_z(
    x: np.ndarray, nc: int, ma: str
) -> tuple[np.ndarray, np.ndarray]:
    """Map the signal to class coordinates ``z`` (steps 1-3 of the historical reference)."""
    mu_x, sigma_x = _sample_moments(x)

    if ma == "LM":
        y = toolbox.mapminmax(x, 0.0, 1.0)
    elif ma == "NCDF":
        y = toolbox.mapminmax(toolbox.normcdf(x, mu_x, sigma_x), 0.0, 1.0)
    elif ma == "LOGSIG":
        # A constant signal gives (x - mu)/sigma = 0/0 = NaN here; MATLAB
        # produces it silently, so silence the numpy warning.
        with np.errstate(divide="ignore", invalid="ignore"):
            y = toolbox.mapminmax(toolbox.logsig((x - mu_x) / sigma_x), 0.0, 1.0)
    elif ma == "TANSIG":
        with np.errstate(divide="ignore", invalid="ignore"):
            y = toolbox.mapminmax(
                toolbox.tansig((x - mu_x) / sigma_x) + 1.0, 0.0, 1.0
            )
    elif ma == "SORT":
        raise NotImplementedError(
            "disp_en: the SORT mapping is not needed by the EnsDE main "
            "chain, which uses only LM, NCDF, TANSIG and LOGSIG"
        )
    else:
        raise ValueError(
            f"disp_en: unknown mapping method {ma!r}; MATLAB falls through "
            "the switch and dies with an unrecognized variable 'z'"
        )

    # Boundary protection: machine epsilon, exactly like the historical reference.
    eps_value = np.finfo(np.float64).eps
    y = np.where(y == 1.0, 1.0 - eps_value, y)
    y = np.where(y == 0.0, eps_value, y)

    # Class assignment. MATLAB's round() halves away from zero; comparing
    # the exact fractional part against 0.5 reproduces that for the
    # non-negative value v = y*nc + 0.5 (a plain np.floor(v + 0.5) would
    # mis-round values a fraction of an ulp below a tie, where the double
    # v + 0.5 already rounds up).
    v = y * nc + 0.5
    frac = v - np.floor(v)
    z = np.floor(v) + (frac >= 0.5)
    return y, z


def disp_en(
    x: np.ndarray,
    m: int,
    nc: int,
    ma: str,
    tau: int,
) -> tuple[float, np.ndarray]:
    """Single-scale Dispersion Entropy with selectable mappings.

    The signal is mapped into discrete classes, embedded into patterns, and
    summarized with Shannon entropy."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 and not (x.ndim == 2 and x.shape[0] == 1):
        raise ValueError(
            "disp_en: x must be 1-D or a (1, N) row; general matrices and "
            "(N, 1) columns are not supported (MATLAB column input passes "
            "unchanged through mapminmax and then hits an "
            "implicit-broadcasting behavior inside the embedding loop)"
        )
    x = x.ravel()

    _, z = _map_to_z(x, nc, ma)

    # Pattern enumeration, base-100 window codes, counting and the
    # normalization by W = N - (m-1)*tau.
    npdf = _count_patterns(z, nc, m, tau, 100.0)

    p = npdf[npdf != 0.0]  # NaN != 0 is True here, as in MATLAB
    out_disp_en = -np.sum(p * np.log(p))
    return float(out_disp_en), npdf
