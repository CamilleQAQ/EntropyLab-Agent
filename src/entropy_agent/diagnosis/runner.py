"""Raw-signal inference restored entirely from a ModelArtifact directory."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..analysis import entropy_backend
from ..analysis.feature_pipeline import (
    DEFAULT_FUSION_ENTROPY_CONFIGS,
    FUSION_METHODS,
    WindowSpec,
    extract_record_vector,
    families_for_fusion,
)
from ..analysis.schemas import WindowPlan
from ..analysis.tools import ScientificToolError, inspect_signal
from ..io.csv_windows import window_signal
from .artifact import ModelArtifact, load_model_artifact


def _abstained(artifact: ModelArtifact, reason: str, *, status: str = "ABSTAINED") -> dict:
    return {
        "status": status,
        "diagnosis": None,
        "diagnostic_profile_id": artifact.profile.profile_id,
        "workflow": artifact.profile.workflow,
        "model_id": artifact.config["model_id"],
        "model_path": str(artifact.directory),
        "reason": reason,
        "probability": None,
    }


def diagnose_artifact_signal(
    signal: object,
    model_path: Path,
    *,
    sampling_rate_hz: float | None,
    trusted: bool = False,
    artifact: ModelArtifact | None = None,
) -> dict:
    """Apply saved preprocessing, features, classifier and aggregation.

    ``artifact`` lets a caller that already loaded and validated the directory
    (for example the model CLI) avoid a second deserialization.  It must belong
    to ``model_path``.
    """
    if artifact is None:
        artifact = load_model_artifact(
            model_path,
            trusted=trusted,
        )
    elif Path(model_path).expanduser().resolve() != artifact.directory:
        raise ValueError("provided artifact does not belong to model_path")
    profile = artifact.profile

    if sampling_rate_hz is None or not np.isfinite(sampling_rate_hz):
        return _abstained(
            artifact,
            "sampling_rate_hz must be supplied from verified metadata",
        )
    if not np.isclose(
        float(sampling_rate_hz),
        profile.sampling_rate_hz,
        rtol=0,
        atol=profile.sampling_rate_tolerance_hz,
    ):
        return _abstained(
            artifact,
            f"sampling_rate_hz={sampling_rate_hz} does not match {profile.sampling_rate_hz}",
            status="ABSTAINED_INCOMPATIBLE",
        )

    try:
        signal_values = np.asarray(signal, dtype=np.float64)
        if signal_values.ndim != 1 or not np.isfinite(signal_values).all():
            raise ValueError("input must be one finite real numeric signal")

        feature_contract = artifact.config["feature"]
        ood_contract = artifact.config["ood"]
        fusion = feature_contract["method"] in FUSION_METHODS
        if ood_contract["enabled"]:
            dimension = feature_contract["dimension"]
            center = np.asarray(ood_contract["center"], dtype=np.float64)
            scale = np.asarray(ood_contract["scale"], dtype=np.float64)
            training_anchors = np.asarray(
                ood_contract["anchors"], dtype=np.float64
            )
            threshold = float(ood_contract["threshold"])
            if (
                center.shape != (dimension,)
                or scale.shape != (dimension,)
                or not np.isfinite(center).all()
                or not np.isfinite(scale).all()
                or (scale <= 0).any()
                or training_anchors.ndim != 2
                or training_anchors.shape[1] != dimension
                or not np.isfinite(training_anchors).all()
                or not np.isfinite(threshold)
                or threshold <= 0
            ):
                raise ValueError("saved OOD contract contains invalid numeric values")

        feature_rows = []
        window_reports = []
        if fusion:
            max_windows = int(feature_contract["parameters"].get("max_windows", 1))
            sample_stride = int(
                feature_contract["parameters"].get(
                    "sample_stride", artifact.config["window"]["stride"]
                )
            )
            families = families_for_fusion(feature_contract["method"])
            feature_values, _, used_windows = extract_record_vector(
                signal_values,
                profile.sampling_rate_hz,
                window=WindowSpec(
                    int(artifact.config["window"]["size"]),
                    sample_stride,
                    max_windows,
                ),
                families=families,
                entropy_configs=(
                    DEFAULT_FUSION_ENTROPY_CONFIGS
                    if "entropy" in families
                    else ()
                ),
                include_full_signal=True,
            )
            if feature_values.shape != (feature_contract["dimension"],) or not np.isfinite(
                feature_values
            ).all():
                raise ValueError("aggregated feature violates the saved feature schema")
            feature_rows.append(feature_values)
            window_report = {
                "window_index": 0,
                "start_sample": 0,
                "stop_sample": int(len(signal_values)),
                "aggregated_windows": int(used_windows),
                "feature_values": feature_values.tolist(),
            }
            if ood_contract["enabled"]:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    standardized_features = (feature_values - center) / scale
                if not np.isfinite(standardized_features).all():
                    raise ValueError(
                        "OOD standardization produced a nonfinite feature value"
                    )
                distance = float(
                    np.min(
                        np.linalg.norm(
                            training_anchors - standardized_features, axis=1
                        )
                    )
                )
                if not np.isfinite(distance):
                    raise ValueError("OOD distance is not finite")
                window_report.update(
                    {
                        "ood_distance": distance,
                        "ood_threshold": threshold,
                        "out_of_distribution": distance > threshold,
                    }
                )
            window_reports.append(window_report)
        else:
            window_plan = WindowPlan(
                window_size_samples=artifact.config["window"]["size"],
                step_size_samples=artifact.config["window"]["stride"],
            )
            windows = window_signal(signal_values, window_plan)
            if len(windows) < profile.minimum_window_count:
                raise ValueError(
                    f"signal provides {len(windows)} complete windows; "
                    f"profile requires {profile.minimum_window_count}"
                )
            for segment in windows:
                inspection = inspect_signal(segment.values)
                if not inspection.valid_for_entropy:
                    codes = ", ".join(issue.code.value for issue in inspection.issues)
                    raise ValueError(
                        f"window {segment.index} failed signal inspection: {codes}"
                    )

                feature_values = np.asarray(
                    entropy_backend.compute(
                        feature_contract["method"],
                        segment.values,
                        feature_contract["parameters"],
                    ),
                    dtype=np.float64,
                )
                if feature_values.shape != (
                    feature_contract["dimension"],
                ) or not np.isfinite(feature_values).all():
                    raise ValueError(
                        f"window {segment.index} violates the saved feature schema"
                    )

                feature_rows.append(feature_values)
                window_report = {
                    "window_index": segment.index,
                    "start_sample": segment.start,
                    "stop_sample": segment.stop,
                    "feature_values": feature_values.tolist(),
                }
                if ood_contract["enabled"]:
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        standardized_features = (feature_values - center) / scale
                    if not np.isfinite(standardized_features).all():
                        raise ValueError(
                            "OOD standardization produced a nonfinite feature value"
                        )
                    distance = float(
                        np.min(
                            np.linalg.norm(
                                training_anchors - standardized_features,
                                axis=1,
                            )
                        )
                    )
                    if not np.isfinite(distance):
                        raise ValueError("OOD distance is not finite")
                    window_report.update(
                        {
                            "ood_distance": distance,
                            "ood_threshold": threshold,
                            "out_of_distribution": distance > threshold,
                        }
                    )
                window_reports.append(window_report)

        if ood_contract["enabled"]:
            outlier_count = sum(
                report["out_of_distribution"] for report in window_reports
            )
            # The saved policy rejects only a strict majority; a tie proceeds.
            if outlier_count > len(window_reports) // 2:
                report = _abstained(
                    artifact,
                    f"{outlier_count}/{len(window_reports)} windows exceed "
                    "the calibrated OOD threshold",
                    status="ABSTAINED_OUT_OF_DISTRIBUTION",
                )
                report["window_predictions"] = window_reports
                report["ood"] = {
                    "rejected": True,
                    "outlier_windows": outlier_count,
                    "window_count": len(window_reports),
                    "distance": ood_contract["distance"],
                    "threshold": threshold,
                    "calibration": ood_contract["calibration"],
                }
                return report

        window_predictions = artifact.model.predict(
            np.asarray(feature_rows, dtype=np.float64)
        )
        window_predictions = np.asarray(window_predictions)
        if window_predictions.shape != (len(window_reports),):
            raise ValueError("classifier must return exactly one prediction per window")
        for value in window_predictions:
            if (
                isinstance(value, (bool, np.bool_))
                or not float(value).is_integer()
                or int(value) not in artifact.labels
            ):
                raise ValueError("classifier returned a class ID outside labels.json")
        # np.bincount requires integers; fractional labels must never reach it.
        window_predictions = window_predictions.astype(np.int64, copy=False)
    except (TypeError, ValueError, ScientificToolError) as exc:
        return _abstained(
            artifact,
            f"{type(exc).__name__}: {exc}",
            status="ABSTAINED_INVALID_INPUT",
        )

    class_counts = np.bincount(
        window_predictions, minlength=len(artifact.labels)
    )
    # np.argmax implements the saved LOWEST_CLASS_ID tie-break rule.
    final_class_id = int(np.argmax(class_counts))
    for window_report, class_id in zip(
        window_reports, window_predictions, strict=True
    ):
        window_report.update(
            {
                "class_id": int(class_id),
                "class_label": artifact.labels[int(class_id)],
            }
        )

    return {
        "status": "DIAGNOSED",
        "diagnosis": artifact.labels[final_class_id],
        "class_id": final_class_id,
        "diagnostic_profile_id": profile.profile_id,
        "workflow": profile.workflow,
        "model_id": artifact.config["model_id"],
        "model_path": str(artifact.directory),
        "sampling_rate_hz": float(sampling_rate_hz),
        "window_count": len(window_reports),
        "window_predictions": window_reports,
        "class_counts": {
            artifact.labels[index]: int(class_counts[index])
            for index in range(len(class_counts))
        },
        "aggregation": artifact.config["aggregation"],
        "vote_share": float(
            class_counts[final_class_id] / len(window_predictions)
        ),
        "probability": None,
        "ood": (
            {
                "enabled": True,
                "rejected": False,
                "outlier_windows": sum(
                    report["out_of_distribution"] for report in window_reports
                ),
                "window_count": len(window_reports),
                "distance": ood_contract["distance"],
                "threshold": ood_contract["threshold"],
                "calibration": ood_contract["calibration"],
            }
            if ood_contract["enabled"]
            else {"enabled": False, "reason": ood_contract["reason"]}
        ),
        "validation_scope": profile.validation_scope,
    }
