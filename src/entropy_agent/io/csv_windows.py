"""Explicit CSV channel loading and deterministic, complete-window segmentation."""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np

from ..analysis.schemas import CSVLoadRequest, IssueCode, ValidationIssue, WindowPlan
from ..analysis.tools import ScientificToolError, _as_numeric_vector


@dataclass(frozen=True)
class LoadedCSV:
    path: str
    channel: str | int
    values: np.ndarray


@dataclass(frozen=True)
class WindowSegment:
    index: int
    start: int
    stop: int
    values: np.ndarray


def load_csv_signal(request: CSVLoadRequest) -> LoadedCSV:
    """Read exactly one explicitly selected numeric channel; never guess a column."""
    values: list[float] = []
    try:
        with open(request.path, "r", encoding=request.encoding, newline="") as handle:
            reader = csv.reader(handle, delimiter=request.delimiter, strict=True)
            for skipped in range(request.skip_rows):
                if next(reader, None) is None:
                    raise ValueError(
                        f"CSV ended before skip_rows={request.skip_rows} at row {skipped + 1}"
                    )
            header = next(reader, None) if request.has_header else None
            if header is not None and not header:
                raise ValueError("CSV header is empty")
            if isinstance(request.channel, str):
                matches = [
                    index
                    for index, name in enumerate(header or [])
                    if name.strip() == request.channel.strip()
                ]
                if len(matches) != 1:
                    raise ScientificToolError(
                        ValidationIssue(
                            code=IssueCode.CHANNEL_NOT_FOUND,
                            message=(
                                f"Named channel {request.channel!r} occurs "
                                f"{len(matches)} times"
                            ),
                        )
                    )
                channel_index = matches[0]
            else:
                channel_index = request.channel
                if header is not None and channel_index >= len(header):
                    raise ScientificToolError(
                        ValidationIssue(
                            code=IssueCode.CHANNEL_NOT_FOUND,
                            message=(
                                f"Channel index {channel_index} exceeds header width "
                                f"{len(header)}"
                            ),
                        )
                    )
            expected_width = len(header) if header is not None else None
            for sample_index, row in enumerate(reader):
                if sample_index < request.start_sample:
                    continue
                if (
                    request.max_samples is not None
                    and len(values) >= request.max_samples
                ):
                    break
                row_number = (
                    request.skip_rows
                    + sample_index
                    + (2 if request.has_header else 1)
                )
                if expected_width is None:
                    expected_width = len(row)
                if len(row) != expected_width:
                    raise ValueError(
                        f"row {row_number} has {len(row)} columns; expected {expected_width}"
                    )
                if channel_index >= len(row):
                    raise ValueError(f"row {row_number} has no column {channel_index}")
                cell = row[channel_index].strip()
                if not cell:
                    raise ValueError(f"row {row_number}, column {channel_index} is empty")
                try:
                    values.append(float(cell))
                except ValueError as exc:
                    raise ValueError(
                        f"row {row_number}, column {channel_index} is not numeric: {cell!r}"
                    ) from exc
    except ScientificToolError:
        raise
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        raise ScientificToolError(
            ValidationIssue(
                code=IssueCode.CSV_PARSE_FAILED,
                message=f"{type(exc).__name__}: {exc}",
            )
        ) from exc
    return LoadedCSV(request.path, request.channel, np.asarray(values, dtype=np.float64))


def window_signal(signal: object, plan: WindowPlan) -> list[WindowSegment]:
    """Return complete windows only; record exact zero-based, half-open offsets."""
    signal_values = _as_numeric_vector(signal)
    size = plan.window_size_samples
    if size > signal_values.size:
        raise ScientificToolError(
            ValidationIssue(
                code=IssueCode.WINDOW_EXCEEDS_SIGNAL,
                message=(
                    f"window_size_samples={size} exceeds signal length "
                    f"{signal_values.size}"
                ),
            )
        )
    return [
        WindowSegment(
            index=index,
            start=start,
            stop=start + size,
            values=signal_values[start : start + size],
        )
        for index, start in enumerate(
            range(0, signal_values.size - size + 1, plan.step_size_samples)
        )
    ]
