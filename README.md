# EntropyLab Agent v2 — clean-room renovation

Status: **paper reproduction started: Aero-engine 5-state TSMEFuDE + BT-LSTSVM = 98.33% test accuracy**

This repository is a clean-room rebuild of the old `entropy-agent` tree.  The old
tree remains a read-only reference.  HIT-specific code, MATLAB compatibility,
contaminated results and legacy reports are **not** copied here.

**New conversation? Read [HANDOFF.md](HANDOFF.md) first.**
It contains the current state, verified results, next tasks and hard constraints.

## What is in this repository

```text
src/msentropy/                 8 entropy methods + public registry
src/entropy_agent/
  analysis/                    entropy backend, windows, quality, feature pipeline
  classifiers/                 6 classifiers + registry + artifact codecs
  io/                          DatasetSpec, CSV/NPY/NPZ/MAT, windowing
  training/                    grouped split + generic training pipeline
  diagnosis/                   v4 artifact, runner, profile registry
tests/                         migrated core tests + boundary test
configs/                       Paderborn baseline DatasetSpecs
manifests/paderborn/           frozen Phase 1 manifests + development.csv
docs/                          Phase 0/1/2 documents + migration log
scripts/                       Phase 1 audit, manifest verifier, Phase 2 runner
data/paderborn/raw/            extracted official MAT payload, ignored by Git
```

Not migrated yet:

- `agent/` — the old LangGraph signal graph is not the new 5-round selection loop;
- `experiments/` — kept in the old tree until the new Agent loop is designed;
- `continual/` — replay/training must be rewritten before copying;
- `cli/` — commands will be re-exposed after the core stabilises;
- `datasets/` — only the Paderborn adapter/config will be added at the boundary;
- `compatibility/` — MATLAB parity is intentionally dropped;
- old `results/`, `mlruns/`, old reports and old HIT artifacts.

## Current test status

```text
231 passed
```

Covered:

- 8 entropy methods and API contracts;
- 6 classifier backends and artifact round-trips;
- DatasetSpec and windowing;
- group-isolated split;
- generic training → artifact → reload → diagnosis;
- dataset-boundary architecture test.

The suite was run with:

```powershell
$env:PYTHONPATH="$PWD\src"
python -m pytest tests -q --basetemp="$PWD\.pytest-tmp"
```

On Windows, pytest's temporary directories can hit ACL restrictions.  Use a
writable `--basetemp` (and, inside the DSH sandbox, full access for the test
process).  The tests themselves use only synthetic data.

## Phase 0 / Phase 1 data foundation

Paderborn is the primary dataset for this renovation:

- 32 official archives, 2,560 MAT files, 20.83 GB extracted payload;
- 26 pure bearings / 2,080 records selected;
- bearing-level split: initial train 8 / validation 4 / batch 01 4 / batch 02 4
  / final holdout 6;
- all selected MAT files SHA-256 and `vibration_1` verified;
- `final_holdout` is frozen and has not been read by any model.

Documents:

- [Phase 0 scope](docs/00_scope.md)
- [Phase 1 protocol](docs/01_protocol.md)
- [Aero-engine 5-state reproduction](docs/05_aero_engine_reproduction.md)
- [Phase 1 result](docs/02_phase1_result.md)
- [Phase 2 result](docs/04_phase2_result.md) — Paderborn cross-bearing stress test, not the paper main path
- [Migration log](docs/03_migration_log.md)
- [Manifest README](manifests/paderborn/README.md)

## Next

Phase 2 starts with:

1. canonical `window` / `effective_sample_stride` contract;
2. a Paderborn `DatasetSpec` at the application boundary;
3. one feature method and one classifier baseline;
4. contract tests before any full training run.

`final_holdout` is not passed to Phase 2.
