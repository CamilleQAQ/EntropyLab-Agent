from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from entropy_agent.io.dataset import (
    ClassifierConfig,
    DatasetLayout,
    DatasetSpec,
    DiagnosisConfig,
    FeatureConfig,
    InputConfig,
    LabelConfig,
    PackedFile,
    SplitConfig,
    WindowConfig,
)


LABELS = ("BENIGN", "DEFECT_A", "DEFECT_B")


def _signal(label: int, record: int, size: int = 200) -> np.ndarray:
    generator = np.random.default_rng(label * 1000 + record)
    base = 0.2 * (label + 1)
    return base * np.sin(np.linspace(0, 12.0 * (label + 1), size)) + generator.normal(
        0.0, 0.01, size
    )


def write_tiny_dataset(root: Path, *, records_per_class: int = 12) -> DatasetSpec:
    root.mkdir(parents=True, exist_ok=True)
    packed = []
    for label in LABELS:
        index = LABELS.index(label)
        array = np.stack(
            [_signal(index, record)[None, :] for record in range(records_per_class)]
        )
        np.save(root / f"{label}.npy", array)
        packed.append(PackedFile(path=f"{label}.npy", label=label))
    return DatasetSpec(
        dataset_id="TINY_TEST_SET",
        sampling_rate_hz=1000,
        input=InputConfig(channel=0, has_header=False),
        layout=DatasetLayout(
            loader="npy_tensor",
            file_pattern="**/*.npy",
            packed_files=tuple(packed),
            group_size=2,
        ),
        labels=LabelConfig(
            names=LABELS,
            mapping={label: index for index, label in enumerate(LABELS)},
            healthy_class_id=0,
        ),
        window=WindowConfig(size=200, stride=200, windows_per_record=1),
        feature=FeatureConfig(method="MDE", parameters={"m": 2, "nc": 4, "tau": 1, "scale": 10}),
        classifier=ClassifierConfig(
            algorithm="SVM-RBF",
            backend="scikit-learn",
            parameters={"C": 1.0, "gamma": "scale"},
        ),
        split=SplitConfig(mode="random", seed=7, train_fraction=0.6, validation_fraction=0.2),
        diagnosis=DiagnosisConfig(
            profile_id="TINY",
            display_name="tiny",
            signal_type="ACCELERATION",
            signal_role="TEST_CHANNEL",
            validation_scope="unit test fixture",
        ),
    )


@pytest.fixture()
def tiny_dataset(tmp_path: Path) -> tuple[DatasetSpec, Path]:
    root = tmp_path / "tiny"
    spec = write_tiny_dataset(root)
    return spec, root
