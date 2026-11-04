# Phase 1 — Data audit, provenance and frozen split

Date: 2026-11-04  
Status: **audit in progress; no model training**

## 1. Source

- Official page: <https://groups.uni-paderborn.de/kat/BearingDataCenter/>
- Local archives: `Paderborn-source-verification-20261102\archives`
- 32 RAR archives, 2,560 MAT files expected.
- Extraction target for this audit: `renovation\data\paderborn\raw`
- Old exploratory manifest: `entropy-agent\results\paderborn-e2e-20261102T133445Z-9eee8c\data\records.csv`
- Old full inventory: `_archive\entropy-agent-20261104\docs\paderborn_full_acceptance_20261102\records.json`
- Bearing factsheets: `entropy-agent\docs\paderborn_full_acceptance_20261102\bearings.csv`

The old files are used only as cross-checks. The new manifest is generated from
the official archives and the extracted MAT files, not copied from the old
record-level split.

## 2. Unit of identity

```text
record  = one MAT file / one measurement
group   = bearing_id
partition = complete bearing
```

A bearing's 4 operating conditions and 20 recordings per condition stay together.
No record-level random split is used.

## 3. Audit checks

### 3.1 Archive and file integrity

- every archive has a SHA-256 receipt;
- every selected MAT exists on disk;
- every selected MAT byte hash equals the hash recorded in the independent
  inventory;
- no missing, extra, zero-length or unreadable MAT in the selected set;
- no duplicate record ID;
- no duplicate content hash across different partitions.

### 3.2 Label contract

- only `HEALTHY`, `OUTER_RACE`, `INNER_RACE` are mapped;
- ID mapping is 0/1/2;
- `COMBINED_INNER_OUTER` is excluded, not mapped to a pure class;
- excluded bearings are listed with a reason;
- label source is the official factsheet / paper table, not a filename prefix.

### 3.3 Group and condition integrity

- each selected bearing has 4 conditions:
  `N09_M07_F10`, `N15_M01_F10`, `N15_M07_F04`, `N15_M07_F10`;
- each bearing/condition has 20 recording numbers;
- no group crosses a partition;
- final holdout is class-covered.

### 3.4 Signal contract

- nominal sampling rate: 64,000 Hz;
- signal: `vibration_1`;
- time axis `XIndex` finite and strictly increasing;
- vibration values finite;
- `vibration_samples` matches the actual array length;
- sample count is not assumed to be exactly 256,000 (records vary).

Unknown and not invented:

- physical machine ID;
- machine serial number;
- lifetime-test cohort;
- absolute timezone;
- vibration unit;
- shifted/partial overlap between records.

### 3.5 Provenance fields

Every selected record carries:

```text
record_id
path
label
label_id
bearing_id
condition_id
recording_number
group_id
source_archive
archive_sha256
raw_member
raw_sha256
acquisition_datetime_local_unzoned
timezone = UNKNOWN
sampling_rate_hz = 64000
signal_name = vibration_1
vibration_samples
split
```

## 4. Frozen split rule

| Partition | Healthy | Outer | Inner | Bearings |
|---|---:|---:|---:|---:|
| initial_train | 2 | 3 | 3 | 8 |
| initial_validation | 1 | 2 | 1 | 4 |
| batch_01 | 1 | 2 | 1 | 4 |
| batch_02 | 1 | 1 | 2 | 4 |
| final_holdout | 1 | 3 | 2 | 6 |

The concrete bearing assignment is written to `assignments.csv` by the audit
script. The script uses a fixed, documented assignment; it does not read model
scores or test metrics.

## 5. Manifest files

```text
manifests/paderborn/
  all_records.csv          # all selected records, with split label
  initial_train.csv
  initial_validation.csv
  batch_01.csv
  batch_02.csv
  final_holdout.csv
  assignments.csv          # bearing -> partition
  split_summary.json       # counts, class/origin/condition coverage, SHA-256
  provenance.json          # source receipts, exclusions, unknown fields
  audit_report.md          # human-readable result
```

Each manifest must be self-contained: `path,label,record_id,group,partition` plus
the provenance fields. The final holdout manifest is frozen by SHA-256 and is not
passed to training until the final evaluation step.

## 6. Acceptance gate

Phase 1 passes only if:

- all 2,080 expected records are present and hash-verified;
- every discrepancy from 2,080 is explained in `audit_report.md`;
- no group crosses partitions;
- no cross-partition duplicate content;
- all 4 conditions and 20 recordings per bearing/condition are accounted for;
- each partition covers all 3 classes;
- `split_summary.json` records the SHA-256 of every manifest;
- a second independent audit run produces identical manifests.

If a check fails, the audit stops and does not create a "best effort" split.
