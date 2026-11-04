"""Ensemble Fuzzy Dispersion Entropy.

The method combines fuzzy pattern memberships from several amplitude mappings
and returns entropy together with the aggregated pattern distribution."""

import numpy as np

from msentropy.core import toolbox
from msentropy.core.fuzzy_disen import fuzzy_disen


def ensfude(
    x: np.ndarray,
    m: int,
    nc: int,
    tau: int,
) -> tuple[float, np.ndarray]:
    """Ensemble Fuzzy Dispersion Entropy.

    The method combines fuzzy pattern memberships from several amplitude mappings
    and returns entropy together with the aggregated pattern distribution."""
    n_patterns = nc**m

    # MA is hard-coded inside the historical reference function.
    ma = [1, 2, 3, 4]

    npdf_lm = np.full(n_patterns, np.nan)
    npdf_ncdf = np.full(n_patterns, np.nan)
    npdf_logsig = np.full(n_patterns, np.nan)
    npdf_tansig = np.full(n_patterns, np.nan)
    npdf_sort = np.full(n_patterns, np.nan)

    if 1 in ma:
        _, npdf_lm = fuzzy_disen(x, m, nc, "LM", tau, 0)
    if 2 in ma:
        _, npdf_ncdf = fuzzy_disen(x, m, nc, "NCDF", tau, 0)
    if 3 in ma:
        _, npdf_tansig = fuzzy_disen(x, m, nc, "TANSIG", tau, 0)
    if 4 in ma:
        _, npdf_logsig = fuzzy_disen(x, m, nc, "LOGSIG", tau, 0)
    if 5 in ma:
        # Dead code, exactly as in MATLAB: 5 is not in MA. Kept for
        # structural fidelity; fuzzy_disen raises NotImplementedError for
        # SORT if this unreachable branch were enabled.
        _, npdf_sort = fuzzy_disen(x, m, nc, "SORT", tau, 0)

    # Column-wise nanmean of the 5-row stack (MATLAB nanmean, default
    # dim = 1), shared with EnsDE since it became the second consumer.
    stack = np.vstack([npdf_lm, npdf_ncdf, npdf_tansig, npdf_logsig, npdf_sort])
    npdf = toolbox.nanmean_columns(stack)

    # Entropy. NaN != 0 is True here, as in MATLAB (npdf ~= 0).
    p = npdf[npdf != 0.0]
    out_disp_en = -np.sum(p * np.log(p))

    return float(out_disp_en), npdf
