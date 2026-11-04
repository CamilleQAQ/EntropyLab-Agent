"""Read explicitly named channels from MATLAB X/Y measurement exports.

No dataset or channel is inferred. SciPy handles MAT v4/v5 through v7.2 and
unwraps MATLAB structs/cell arrays; HDF5 v7.3 is rejected explicitly.
"""
from pathlib import Path
import numpy as np
from scipy.io import loadmat


def load_named_mat_signal(path: Path, *, key: str | None, signal_name: str,
                          nominal_sampling_rate_hz: float | None) -> tuple[np.ndarray, dict]:
    if not signal_name or nominal_sampling_rate_hz is None or not np.isfinite(nominal_sampling_rate_hz) or nominal_sampling_rate_hz <= 0:
        raise ValueError("named signal requires an explicit name and finite positive nominal sampling rate")
    try:
        values = loadmat(path, simplify_cells=True)
    except NotImplementedError as exc:
        raise ValueError("MATLAB HDF5 v7.3 is not supported by this measurement reader") from exc
    keys = [name for name in values if not name.startswith("__")]
    if key is None:
        if len(keys) != 1:
            raise ValueError("measurement MAT requires an explicit top-level variable key")
        key = keys[0]
    if key not in values or not isinstance(values[key], dict):
        raise ValueError("selected MATLAB variable must be a measurement struct")
    measurement = values[key]
    if "Y" not in measurement or "X" not in measurement:
        raise ValueError("measurement requires explicit X and Y fields")
    channels = measurement["Y"]
    channels = [channels] if isinstance(channels, dict) else list(np.asarray(channels, dtype=object).flat)
    selected = [c for c in channels if isinstance(c, dict) and c.get("Name") == signal_name]
    if len(selected) != 1:
        raise ValueError(f"expected exactly one channel named {signal_name!r}, found {len(selected)}")
    channel = selected[0]
    signal = np.asarray(channel.get("Data"))
    if signal.ndim != 1 or not np.issubdtype(signal.dtype, np.number) or np.iscomplexobj(signal) or len(signal) < 2 or not np.isfinite(signal).all():
        raise ValueError("measurement channel must be a finite real one-dimensional signal")
    index = channel.get("XIndex")
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise ValueError("measurement XIndex must be an integer MATLAB index")
    axes = measurement["X"]
    axes = [axes] if isinstance(axes, dict) else list(np.asarray(axes, dtype=object).flat)
    if not 1 <= index <= len(axes) or not isinstance(axes[index - 1], dict):
        raise ValueError("measurement XIndex is outside the time axes")
    time = np.asarray(axes[index - 1].get("Data"))
    if time.ndim != 1 or time.shape != signal.shape or not np.isfinite(time).all() or not (np.diff(time) > 0).all():
        raise ValueError("measurement time axis must match the signal and be finite and strictly increasing")
    return np.asarray(signal, dtype=np.float64), {
        "key": key, "mat_signal_name": signal_name, "data_field": "Y[Name].Data",
        "time_index_matlab": int(index), "nominal_sampling_rate_hz": nominal_sampling_rate_hz,
        "sampling_rate_evidence": "EXPLICIT_CONFIG_NOMINAL", "resampled": False,
        "source_samples": len(signal), "time_start": float(time[0]), "time_end": float(time[-1]),
        "time_step_min": float(np.diff(time).min()), "time_step_max": float(np.diff(time).max()),
    }
