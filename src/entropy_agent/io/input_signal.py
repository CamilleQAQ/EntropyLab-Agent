"""Load one selected 1-D signal from common research file formats.

File structure is resolved here; the Agent receives only a numeric vector.
No dataset-specific row counts, labels, or grouping rules enter this module.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat, whosmat

from ..analysis.schemas import CSVLoadRequest
from .csv_windows import load_csv_signal


SUPPORTED_SUFFIXES = frozenset({".csv", ".npy", ".npz", ".mat"})


@dataclass(frozen=True)
class SignalInput:
    path: Path
    channel: str | int | None = None  # CSV column only
    key: str | None = None  # NPZ or MAT variable
    sample_axis: int = -1
    indices: tuple[int, ...] = ()  # One index per non-sample axis
    start_sample: int = 0
    max_samples: int | None = None
    has_header: bool = True
    delimiter: str = ","
    skip_rows: int = 0
    mat_signal_name: str | None = None
    nominal_sampling_rate_hz: float | None = None


@dataclass(frozen=True)
class LoadedSignal:
    values: np.ndarray
    path: str
    format: str
    selection: dict
    source_samples: int | None
    analyzed_samples: int


def describe_input(
    path: Path,
    *,
    has_header: bool = True,
    delimiter: str = ",",
) -> str:
    """Show choices without guessing a multichannel or multiarray selection."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            first = next(csv.reader(handle, delimiter=delimiter), [])
        if has_header:
            return "CSV 列：" + "，".join(
                f"{index}: {name.strip()}" for index, name in enumerate(first)
            )
        return f"无表头 CSV：首行有 {len(first)} 列；列序号从 0 开始"
    if suffix == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        return f"NPY 数组形状：{array.shape}；默认最后一轴是样本轴"
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return "NPZ 数组：" + "，".join(
                f"{name}: {archive[name].shape}" for name in archive.files
            )
    if suffix == ".mat":
        return "MAT 数组：" + "，".join(
            f"{name}: {shape}" for name, shape, _ in whosmat(path)
            if not name.startswith("__")
        )
    raise ValueError(f"不支持 {suffix or '无扩展名'} 文件；支持 CSV、NPY、NPZ、MAT")


def _single_array_key(names: list[str], requested: str | None, kind: str) -> str:
    if requested is not None:
        if requested not in names:
            raise ValueError(f"{kind} 中没有数组 {requested!r}；可选：{', '.join(names)}")
        return requested
    if len(names) == 1:
        return names[0]
    if not names:
        raise ValueError(f"{kind} 中没有可选数组")
    raise ValueError(f"{kind} 有多个数组，请用 --key 指定：{', '.join(names)}")


def _select_array(array: np.ndarray, spec: SignalInput) -> tuple[np.ndarray, int]:
    if array.ndim == 0:
        raise ValueError("所选变量是标量，至少需要一维样本")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise ValueError(f"所选数组必须是实数数值数组，当前类型为 {array.dtype}")
    axis = spec.sample_axis % array.ndim
    if spec.sample_axis < -array.ndim or spec.sample_axis >= array.ndim:
        raise ValueError(f"样本轴 {spec.sample_axis} 超出 {array.ndim} 维数组范围")
    other_axes = [i for i in range(array.ndim) if i != axis]
    indices = spec.indices
    if not indices and all(array.shape[i] == 1 for i in other_axes):
        indices = (0,) * len(other_axes)  # MATLAB row vectors are stored as 1 x N.
    if len(indices) != len(other_axes):
        raise ValueError(
            f"数组形状 {array.shape}，样本轴 {axis}；请按其余轴顺序提供"
            f" {len(other_axes)} 个 --index，当前提供 {len(spec.indices)} 个"
        )
    selection: list[int | slice] = [slice(None)] * array.ndim
    for dim, index in zip(other_axes, indices):
        if index < 0 or index >= array.shape[dim]:
            raise ValueError(f"第 {dim} 轴索引 {index} 超出 0..{array.shape[dim] - 1}")
        selection[dim] = index
    selected = array[tuple(selection)]
    count = int(selected.shape[0])
    stop = None if spec.max_samples is None else spec.start_sample + spec.max_samples
    # Detach from NPY memory mapping so the source can be closed or replaced on Windows.
    return np.array(selected[spec.start_sample:stop], dtype=np.float64, copy=True), count


def load_input_signal(spec: SignalInput) -> LoadedSignal:
    """Normalize a selected file series to a float64 vector for the Agent."""
    path = Path(spec.path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"文件不存在：{path}")
    if spec.start_sample < 0 or (spec.max_samples is not None and spec.max_samples < 1):
        raise ValueError("--start-sample 必须非负，--max-samples 必须为正")
    suffix = path.suffix.lower()
    if spec.mat_signal_name is not None:
        if suffix != ".mat":
            raise ValueError("named MATLAB signal selection requires a MAT file")
        from .mat_signal import load_named_mat_signal
        values, selection = load_named_mat_signal(
            path, key=spec.key, signal_name=spec.mat_signal_name,
            nominal_sampling_rate_hz=spec.nominal_sampling_rate_hz,
        )
        count = len(values)
        if spec.start_sample < 0 or (spec.max_samples is not None and spec.max_samples < 1):
            raise ValueError("invalid MATLAB sample range")
        stop = None if spec.max_samples is None else spec.start_sample + spec.max_samples
        values = values[spec.start_sample:stop].copy()
        if not len(values):
            raise ValueError("selected MATLAB signal is empty")
        return LoadedSignal(values, str(path), "mat", selection, count, len(values))
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"不支持 {suffix or '无扩展名'} 文件；支持 CSV、NPY、NPZ、MAT")
    if suffix == ".csv":
        if spec.key is not None or spec.indices or spec.sample_axis != -1:
            raise ValueError("CSV 使用 --channel 选列；--key、--index、--sample-axis 用于数组文件")
        channel = spec.channel
        if channel is None:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                first = next(csv.reader(handle, delimiter=spec.delimiter), [])
            if len(first) != 1:
                raise ValueError("CSV 有多列，请用 --channel 指定信号列")
            channel = 0
        loaded = load_csv_signal(
            CSVLoadRequest(
                path=str(path),
                channel=channel,
                has_header=spec.has_header,
                delimiter=spec.delimiter,
                start_sample=spec.start_sample,
                max_samples=spec.max_samples,
                skip_rows=spec.skip_rows,
            )
        )
        values = loaded.values
        selection = {
            "channel": channel,
            "has_header": spec.has_header,
            "skip_rows": spec.skip_rows,
        }
        source_samples = None  # Streaming CSV does not scan the unselected tail.
    else:
        if spec.channel is not None:
            raise ValueError("数组文件使用 --index 选序列；--channel 仅用于 CSV")
        if suffix == ".npy":
            if spec.key is not None:
                raise ValueError("NPY 只有一个数组，不需要 --key")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            key = None
        elif suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                key = _single_array_key(archive.files, spec.key, "NPZ")
                array = archive[key]
        else:
            names = [name for name, _, _ in whosmat(path) if not name.startswith("__")]
            key = _single_array_key(names, spec.key, "MAT")
            array = loadmat(path, variable_names=[key])[key]
        values, source_samples = _select_array(array, spec)
        selection = {
            "key": key,
            "sample_axis": spec.sample_axis,
            "indices": list(spec.indices),
        }
    if values.size == 0:
        raise ValueError("所选样本范围为空，请检查 --start-sample 和文件长度")
    return LoadedSignal(
        values=values,
        path=str(path),
        format=suffix[1:],
        selection=selection,
        source_samples=source_samples,
        analyzed_samples=int(values.size),
    )
