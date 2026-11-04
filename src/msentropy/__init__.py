"""Python toolbox for single-scale and multiscale entropy algorithms.

The numerical implementations have been validated against independently
generated MATLAB reference outputs. See ``README.md`` for the usage guide
and ``docs/validation.md`` for validation scope.

Two layers are available:

- ``msentropy.compute(method, x, **params)`` is the **high-level API**:
  supported multiscale methods behind one keyword-parameter entry point, with vector
  normalization and checks for common input mistakes.
- ``msentropy.<method>(x, ...)`` and ``msentropy.core.*`` are the
  **compatibility API**: direct numerical functions retaining documented
  MATLAB-compatible behavior for scientific reproduction.

For valid inputs, both interfaces use the same numerical functions. Prefer the
high-level API for new analysis code.

Example
-------
>>> import numpy as np, msentropy
>>> x = np.random.default_rng(0).standard_normal(2048)
>>> msentropy.compute("MDE", x, m=3, nc=6, tau=1, scale=5)   # doctest: +SKIP
array([2.4870, 2.2520, 2.1787, 2.1018, 1.9973])
>>> msentropy.list_methods()[:3]
('MPE', 'MDE', 'MFuDE')
"""

from msentropy.api import (
    PARAMETERS,
    Method,
    as_signal,
    compute,
    get_method,
    list_methods,
)
from msentropy.core.imfrde import imfrde
from msentropy.core.mde import mde
from msentropy.core.mfude import mfude
from msentropy.core.mfe_mu import mfe_mu
from msentropy.core.mpe import mpe
from msentropy.core.rcmde import rcmde
from msentropy.core.rcmfude import rcmfude
from msentropy.core.tsmefude import tsmefude

__all__ = [
    # high-level API
    "PARAMETERS",
    "Method",
    "as_signal",
    "compute",
    "get_method",
    "list_methods",
    # compatibility API: direct numerical methods by Python function name
    "imfrde",
    "mde",
    "mfude",
    "mfe_mu",
    "mpe",
    "rcmde",
    "rcmfude",
    "tsmefude",
]
