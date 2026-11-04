# Paderborn Phase 1 audit

Generated: 2026-11-04T09:11:11.281871+00:00

**Status: PASS**

## Counts

| Partition | Bearings | Records | Classes |
|---|---:|---:|---|
| `initial_train` | 8 | 640 | {'HEALTHY': 160, 'OUTER_RACE': 240, 'INNER_RACE': 240} |
| `initial_validation` | 4 | 320 | {'HEALTHY': 80, 'OUTER_RACE': 160, 'INNER_RACE': 80} |
| `batch_01` | 4 | 320 | {'HEALTHY': 80, 'OUTER_RACE': 160, 'INNER_RACE': 80} |
| `batch_02` | 4 | 320 | {'HEALTHY': 80, 'OUTER_RACE': 80, 'INNER_RACE': 160} |
| `final_holdout` | 6 | 480 | {'HEALTHY': 80, 'OUTER_RACE': 240, 'INNER_RACE': 160} |

Total: **2080 records / 26 bearings**.

## Checks

- archives: 32
- selected MAT missing: 0
- SHA-256 mismatches: 0
- deep signal failures: 0
- cross-partition duplicate hashes: 0
- errors: 0
- warnings: 1

## Inventory status

The independent inventory marks every selected record as `formal_approval=false` and `status=STRUCTURE_AUDITED_NOT_TRAINING_APPROVED`. This Phase 1 result is an internal provenance/split freeze, not a formal dataset or training approval.

- inventory status: {'STRUCTURE_AUDITED_NOT_TRAINING_APPROVED': 2080}
- formal approval: {'False': 2080}

## Excluded bearings

- `KB23`: combined inner+outer damage; not a pure class
- `KB24`: combined inner+outer damage; not a pure class
- `KB27`: combined inner+outer damage; not a pure class
- `KI04`: factsheet records combined inner+outer damage
- `KI14`: factsheet records combined inner+outer damage
- `KA08`: outer-race label ambiguity; one MAT record is unreadable

## Unknown fields

`physical_machine_id`, `machine_serial_number`, `lifetime_test_id`, `lifetime_test_batch_id`, `datetime_timezone`, `signal_unit`, `shifted_or_partial_cross_record_overlap`

## Old exploratory manifest

- old records: 2076
- new-only records: 4
- old-only records: 0

New-only records restored by this audit:

- `PU:KA04:N09_M07_F10:r17` `KA04/N09_M07_F10_KA04_17.mat` (OUTER_RACE, N09_M07_F10)
- `PU:KA04:N15_M07_F04:r15` `KA04/N15_M07_F04_KA04_15.mat` (OUTER_RACE, N15_M07_F04)
- `PU:KA04:N15_M07_F04:r19` `KA04/N15_M07_F04_KA04_19.mat` (OUTER_RACE, N15_M07_F04)
- `PU:KA04:N15_M07_F04:r20` `KA04/N15_M07_F04_KA04_20.mat` (OUTER_RACE, N15_M07_F04)

These four KA04 records were omitted by the old manifest without a recorded reason. They pass byte-hash and deep signal checks and are kept in the new manifest. Any later source-level exclusion must be documented explicitly.

## Warnings

- old manifest differs: new-only=4 old-only=0
