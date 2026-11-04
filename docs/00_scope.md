# Phase 0 — Scope and dataset decision

Date: 2026-11-04  
Project: EntropyLab renovation (clean-room track)  
Status: **superseded for the paper main path; see the paper reproduction document**

> **Update after reading the two paper DOIs:** the paper main datasets are the
> Aero-engine 5-state benchmark and the Triaxial 7-class benchmark.  Paderborn
> is retained only as an optional cross-bearing stress test.  See
> [Aero-engine reproduction](05_aero_engine_reproduction.md).

## 1. Task

Three-class bearing fault diagnosis:

| ID | Class | Meaning |
|---:|---|---|
| 0 | `HEALTHY` | undamaged bearing |
| 1 | `OUTER_RACE` | outer-race damage |
| 2 | `INNER_RACE` | inner-race damage |

The evaluation unit is a **complete bearing**, not a record and not a window. All
operating conditions and all recordings of one bearing stay in the same
partition.

The renovation target is one complete path:

```text
raw archives
  -> audited records + provenance
  -> bearing-level split
  -> one feature method
  -> one classifier
  -> saved model artifact
  -> diagnosis on a new signal
  -> sequential batch updates with historical replay
  -> final unseen-bearing holdout
```

## 2. Dataset decision

### Primary development dataset: Paderborn

Reasons:

- official bearing identifiers (`K001` ... `KI21`);
- 4 documented operating conditions per bearing;
- 20 recordings per bearing/condition;
- 64 kHz nominal vibration channel `vibration_1`;
- 6 healthy, 11 pure outer-race and 9 pure inner-race bearings after exclusions;
- enough bearings for bearing-level train/validation/update/final partitions.

Source: <https://groups.uni-paderborn.de/kat/BearingDataCenter/>  
Local archives: `E:\Research\EntropyLab Agent\Paderborn-source-verification-20261102\archives`

License boundary: the official Paderborn data page states **CC BY-NC 4.0**.
This track is non-commercial research. The license must be rechecked before any
redistribution.

### External blind test: CWRU

CWRU is reserved for a later external validation on a different test rig. It is
**not** used for development or model selection. Its bearing/specimen identity is
weaker than Paderborn's, so it cannot replace Paderborn for bearing-level
continual-learning evidence.

### Smoke test only: VBL

VBL is useful for a quick 20 kHz pipeline smoke test. It has 5 motors and 4
operating conditions with confounded device/class identity, so it is not the
final generalization evidence.

### Not a primary dataset: HIT

HIT is binary (healthy/fault), and its acquisition time and physical device
identity are still `UNKNOWN`. It may be used later as a robustness/transfer
check, not as the main continual-learning dataset.

## 3. Closed-set selection rule

The official archive has 32 bearings and 2,560 MAT files. This track uses the
26 bearings with a single supported label in the three-class contract.

Excluded bearings:

| Bearing | Reason |
|---|---|
| `KB23`, `KB24`, `KB27` | combined inner+outer damage; not a pure class |
| `KI04`, `KI14` | combined inner+outer damage recorded in the factsheet |
| `KA08` | outer-race label ambiguity; one unreadable MAT record |

Expected closed set: **26 bearings = 6 healthy + 11 outer + 9 inner**.

Each selected bearing contributes 4 conditions x 20 recordings = 80 records.
Expected total: **26 x 80 = 2,080 records**.

The old exploratory manifest had 2,076 records. The Phase 1 audit must explain
the four-record difference; it must not silently drop records.

## 4. Split design

The split is assigned at bearing level. The target design is:

| Partition | Healthy | Outer | Inner | Bearings | Purpose |
|---|---:|---:|---:|---:|---|
| `initial_train` | 2 | 3 | 3 | 8 | fit the first model |
| `initial_validation` | 1 | 2 | 1 | 4 | feature/classifier selection |
| `batch_01` | 1 | 2 | 1 | 4 | first update batch |
| `batch_02` | 1 | 1 | 2 | 4 | second update batch |
| `final_holdout` | 1 | 3 | 2 | 6 | final test, opened once |
| **total** | **6** | **11** | **9** | **26** | |

Design constraints:

- every partition covers all three classes;
- no bearing crosses partitions;
- damage origin (artificial vs accelerated lifetime) is balanced where the class
  allows it;
- assignment is made **before** any model score is computed;
- the assignment is not changed after seeing validation or test results.

The concrete bearing list is frozen by
`manifests/paderborn/assignments.csv` after the Phase 1 audit.

## 5. Batch protocol

```text
initial_train + initial_validation
  -> model v0
batch_01 (new bearings, labels)
  -> update v1, candidate vs v0 on the fixed evaluation holdout
batch_02 (new bearings, labels)
  -> update v2, candidate vs v1 on the fixed evaluation holdout
final_holdout (never used before)
  -> one-time final test of the final champion
```

Rules:

- `final_holdout` is never read during training, feature selection, drift
  calibration, or update decisions.
- Each batch update must preserve historical training coverage. The candidate
  replay must not re-split historical train records into validation/test.
- Every update is compared with a frozen champion on the same fixed holdout.
- No test result may be used to change a feature, classifier, threshold, replay
  ratio, or promotion rule.

## 6. Metrics

Primary:

- Macro-F1;
- per-class Recall (especially `INNER_RACE`);
- group-level Macro-F1 (bearing groups);
- history coverage: fraction of champion training records present in the
  candidate training set.

Secondary:

- Accuracy, balanced accuracy, confusion matrix;
- inference time per record;
- model size;
- number of trials / wall-clock time;
- provenance completeness.

## 7. Non-goals for this phase

- MATLAB bit-level parity;
- Agent/AutoML optimization;
- RUL or run-to-failure prediction;
- cross-device claims;
- online `partial_fit` learning;
- deployment on an industrial machine.

## 8. Known design limitations

These are accepted for the renovation baseline and must stay visible in every
report:

1. Only **6 healthy bearings** exist. The initial training partition has only
   2 healthy bearings, so healthy-class variability is weakly represented.
2. The final holdout has **6 bearings / 480 records**. It is a bearing-level
   test, but the number of independent bearing groups is small.
3. The four operating conditions are repeated measurements of the same bearing;
   they are not independent devices.
4. The update batches are **retrospective bearing partitions**, not an
   acquisition-time stream. This track tests batch model evolution, not real
   temporal drift.
5. Artificial and accelerated-lifetime damage origins are balanced where
   possible, but damage origin is not a causal variable in this design.
6. The dataset does not expose a verified physical machine ID or a shared
   lifetime-test cohort. Those fields remain `UNKNOWN`.
7. The Paderborn data page states CC BY-NC 4.0. This is non-commercial research;
   redistribution requires a separate license review.
8. The project must report per-class recall and group-level metrics, not only
   overall accuracy.

## 8. Phase 0 exit criteria

- dataset and license recorded;
- 3-class closed-set rule frozen;
- 26-bearing selection and 2,080-record expectation frozen;
- split counts frozen;
- batch order frozen;
- no signal training has been run.

Approval: pending human sign-off after the Phase 1 audit.
