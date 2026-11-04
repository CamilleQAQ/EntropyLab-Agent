"""The only Agent module that imports the bundled ``msentropy`` package."""

from __future__ import annotations

import msentropy
import numpy as np


def available_methods() -> tuple[str, ...]:
    """Read the authoritative high-level toolbox registry."""
    return msentropy.list_methods()


def validate_config(method: str, parameters: dict) -> dict:
    """Validate method and parameter schema without executing a signal."""
    from .schemas import EntropyJob
    job = EntropyJob.model_validate({"job_id": "config", "method": method,
                                    "parameters": parameters})
    return job.parameters.model_dump()


def compute(method: str, signal: np.ndarray, parameters: dict[str, int | float]) -> np.ndarray:
    """Call the high-level API without modifying its numerical output."""
    return np.asarray(msentropy.compute(method, signal, **parameters))
