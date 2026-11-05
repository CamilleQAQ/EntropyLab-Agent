#!/usr/bin/env python
"""Generic supervised experiment runner (thin wrapper around the shared runner).

The implementation lives in ``entropy_agent.training.experiment`` so the CLI and
this script share one code path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from entropy_agent.io.dataset import load_dataset_spec  # noqa: E402
from entropy_agent.training.experiment import run_supervised_experiment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()

    spec = load_dataset_spec(args.config)
    try:
        summary = run_supervised_experiment(
            spec,
            args.data_root,
            args.output,
            model_id=args.model_id,
            config_path=args.config,
        )
    except FileExistsError as error:
        print(error)
        return 2

    print("experiment complete")
    print(f"feature_dimension={summary['feature']['dimension']}")
    print(f"records={summary['split']['records']}")
    print(f"validation_accuracy={summary['validation']['multiclass_accuracy']:.6f}")
    print(f"validation_macro_f1={summary['validation']['multiclass_macro_f1']:.6f}")
    print(f"test_accuracy={summary['test']['multiclass_accuracy']:.6f}")
    print(f"test_macro_f1={summary['test']['multiclass_macro_f1']:.6f}")
    print(f"model={summary['model_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
