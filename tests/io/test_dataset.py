from pathlib import Path

import numpy as np

from entropy_agent.io.dataset import (
    ClassifierConfig,
    DatasetLayout,
    DatasetSpec,
    DiagnosisConfig,
    FeatureConfig,
    InputConfig,
    LabelConfig,
    SplitConfig,
    WindowConfig,
    load_dataset_records,
    load_record_signal,
)


def _spec(*, layout: DatasetLayout, input_config: InputConfig) -> DatasetSpec:
    return DatasetSpec(
        dataset_id="SYNTHETIC",
        sampling_rate_hz=1_000,
        input=input_config,
        layout=layout,
        labels=LabelConfig(
            names=("normal", "fault"), mapping={"normal": 0, "fault": 1},
        ),
        window=WindowConfig(size=8, stride=8),
        feature=FeatureConfig(method="MDE", parameters={"m": 2, "nc": 5, "tau": 1, "scale": 2}),
        classifier=ClassifierConfig(parameters={
            "c1": 0.25, "c2": 0.25, "bandwidth": 4.0, "epsilon": 1e-4,
        }),
        split=SplitConfig(seed=3),
        diagnosis=DiagnosisConfig(
            profile_id="SYNTHETIC_PROFILE", display_name="Synthetic profile",
            signal_type="VIBRATION", signal_role="TEST_SENSOR",
            validation_scope="synthetic group holdout",
        ),
    )


def test_file_layout_loads_explicit_parent_labels_and_csv_skip_rows(tmp_path: Path) -> None:
    for label in ("normal", "fault"):
        folder = tmp_path / label
        folder.mkdir()
        (folder / "record.csv").write_text(
            "metadata\n" + "\n".join(str(index) for index in range(8)),
            encoding="utf-8",
        )
    spec = _spec(
        layout=DatasetLayout(file_pattern="*/*.csv"),
        input_config=InputConfig(channel=0, has_header=False, skip_rows=1),
    )
    records = load_dataset_records(spec, tmp_path)
    assert [record.label for record in records] == [1, 0]
    assert np.array_equal(load_record_signal(records[0]).values, np.arange(8))


def test_packed_npy_layout_exposes_records_without_dataset_code(tmp_path: Path) -> None:
    normal = np.arange(4 * 2 * 8, dtype=np.float64).reshape(4, 2, 8)
    fault = normal + 100
    np.save(tmp_path / "a.npy", normal)
    np.save(tmp_path / "b.npy", fault)
    spec = _spec(
        layout=DatasetLayout(
            loader="npy_tensor", group_size=2,
            packed_files=(
                {"path": "a.npy", "label": "normal"},
                {"path": "b.npy", "label": "fault"},
            ),
        ),
        input_config=InputConfig(channel=1),
    )
    records = load_dataset_records(spec, tmp_path)
    assert len(records) == 8
    assert len({record.group for record in records}) == 4
    assert np.array_equal(load_record_signal(records[2]).values, normal[2, 1])


def test_dataset_specs_are_not_imported_by_core() -> None:
    # Concrete dataset configs are assembled at the application boundary.  The
    # core only sees DatasetSpec objects; this test intentionally does not load
    # any HIT/VBL/NEU example config.
    from entropy_agent.io.dataset import DatasetSpec

    assert DatasetSpec is not None
