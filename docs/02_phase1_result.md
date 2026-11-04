# Phase 1 result — Paderborn audit and frozen bearing split

Date: 2026-11-04  
Status: **PASS**  
Training performed: **none**

## 1. Result

The official Paderborn source was independently re-extracted and re-audited.

| Item | Result |
|---|---:|
| Official RAR archives | 32 |
| Extracted MAT files | 2,560 |
| Raw payload size | 20.83 GB |
| Selected pure bearings | 26 |
| Selected records | 2,080 |
| Missing selected MAT files | 0 |
| SHA-256 mismatches | 0 |
| Deep `vibration_1` failures | 0 |
| Cross-partition duplicate content | 0 |
| Partitions | 5 |
| Final holdout | 6 bearings / 480 records |

The audit script and the independent manifest verifier both pass.

This is an **internal provenance and split freeze**, not a formal dataset or
training approval. The independent inventory marks every selected record as
`formal_approval=false` and `status=STRUCTURE_AUDITED_NOT_TRAINING_APPROVED`.
That status is preserved; this track does not relabel it as approved.

## 2. Closed set

```text
32 official bearings
  - 3 combined inner+outer (KB23, KB24, KB27)
  - 2 combined inner+outer (KI04, KI14)
  - 1 ambiguous outer (KA08)
= 26 pure bearings
```

Label mapping:

```text
HEALTHY     0
OUTER_RACE  1
INNER_RACE  2
```

Class totals: 6 healthy bearings / 11 outer-race / 9 inner-race.

## 3. Frozen split

| Partition | Bearings | Records | Healthy | Outer | Inner |
|---|---:|---:|---:|---:|---:|
| `initial_train` | 8 | 640 | 2 | 3 | 3 |
| `initial_validation` | 4 | 320 | 1 | 2 | 1 |
| `batch_01` | 4 | 320 | 1 | 2 | 1 |
| `batch_02` | 4 | 320 | 1 | 1 | 2 |
| `final_holdout` | 6 | 480 | 1 | 3 | 2 |
| **total** | **26** | **2,080** | **6** | **11** | **9** |

The exact bearing lists are in `assignments.csv`. The assignment is
bearing-level: all four conditions and all 20 recordings of a bearing stay in one
partition.

## 4. Provenance

- official source: <https://groups.uni-paderborn.de/kat/BearingDataCenter/>;
- local archives: `Paderborn-source-verification-20261102\archives`;
- archive bytes: 5,357,708,539;
- nominal sampling rate: 64,000 Hz;
- signal: `vibration_1`;
- every selected record has archive SHA-256 and raw MAT SHA-256;
- every selected MAT was loaded and `vibration_1` was checked for shape, length,
  finiteness and monotonic `XIndex`;
- per-bearing conditions: 4;
- per-bearing/condition recordings: 20.

Unknown fields remain explicit:

`physical_machine_id`, `machine_serial_number`, `lifetime_test_id`,
`lifetime_test_batch_id`, `datetime_timezone`, `signal_unit`,
`shifted_or_partial_cross_record_overlap`.

## 5. The four restored KA04 records

The old exploratory manifest had 2,076 rows; the new manifest has 2,080. The
difference is four KA04 records omitted by the old manifest without a recorded
reason:

```text
PU:KA04:N09_M07_F10:r17   KA04/N09_M07_F10_KA04_17.mat
PU:KA04:N15_M07_F04:r15   KA04/N15_M07_F04_KA04_15.mat
PU:KA04:N15_M07_F04:r19   KA04/N15_M07_F04_KA04_19.mat
PU:KA04:N15_M07_F04:r20   KA04/N15_M07_F04_KA04_20.mat
```

All four pass byte-hash and deep signal checks. They are kept in the new manifest.
If a future source review finds a real exclusion reason, it must be recorded
explicitly; the old silent omission is not repeated.

## 6. Frozen manifest hashes

| Manifest | SHA-256 |
|---|---|
| `all_records.csv` | `05d5a6c7ba6ce4650e6a5a40dda7caee74a0bf615e9e4804a971247775d70e84` |
| `initial_train.csv` | `6345ada41d92d12c7f115316f1948ef8724b7a01497f3bf75a24178049a438fd` |
| `initial_validation.csv` | `3f1ff325de39cfb2892c802099e86b558c624d540b98f841546f0e8f2ff3dab5` |
| `batch_01.csv` | `64d05dcafd32a7e6dce79ac28d50e02bad2c0dd1ef8ab796dc04840fd1759918` |
| `batch_02.csv` | `15ba5a8008eb63e3abb6b526dc4dd71dccadd88211eb0706548364202e04e3b7` |
| `final_holdout.csv` | `dbd704911b465e9f6e436922ae1d2f4e17404fcc1814c5be8bb86f26938ceffe` |

`final_holdout.csv` is frozen. It must not be passed to training, feature
selection, drift calibration, or promotion decisions until the final evaluation.

## 7. Commands and evidence

```powershell
python scripts/phase1_audit.py --hash --deep
python scripts/verify_manifests.py
```

Evidence files:

- `manifests/paderborn/audit_report.md`;
- `manifests/paderborn/provenance.json`;
- `manifests/paderborn/split_summary.json`;
- `manifests/paderborn/audit_run.log`;
- `manifests/paderborn/verify_manifests.log`.

## 8. Accepted limitations

- only 6 healthy bearings; initial training has 2;
- final holdout has 6 bearing groups / 480 records;
- operating conditions are repeated measurements, not independent devices;
- batches are retrospective bearing partitions, not acquisition-time drift;
- physical machine identity and timezone are unknown;
- Paderborn license is CC BY-NC 4.0; redistribution needs review.

These limitations must be repeated in later reports.

## 9. Exit decision

Phase 1 is accepted as the data foundation for Phase 2. Phase 2 starts with the
canonical window/stride contract and a single-method, single-classifier baseline
configuration; it must not open `final_holdout`.
