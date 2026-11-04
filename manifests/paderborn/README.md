# Paderborn manifests

Frozen: 2026-11-04  
Audit: **PASS** (internal provenance/split freeze, not formal training approval)  
Source: official Paderborn Bearing Data Center archives

## Files

| File | Contents |
|---|---|
| `all_records.csv` | all 2,080 selected records |
| `initial_train.csv` | 8 bearings / 640 records |
| `initial_validation.csv` | 4 bearings / 320 records |
| `batch_01.csv` | 4 bearings / 320 records |
| `batch_02.csv` | 4 bearings / 320 records |
| `final_holdout.csv` | 6 bearings / 480 records; **locked** |
| `development.csv` | Phase 2 derived manifest: initial_train + initial_validation, 960 records |
| `assignments.csv` | bearing -> partition |
| `split_summary.json` | counts, class/origin/condition coverage, manifest hashes |
| `provenance.json` | archive hashes, exclusions, audit evidence |
| `audit_report.md` | human-readable audit result |

## Identity contract

```text
record_id  = PU:<bearing>:<condition>:r<NN>
group_id   = PU:<bearing>
partition  = complete bearing
```

All four conditions and all 20 recordings of one bearing stay in one partition.
No record-level random split is used.

## Data contract

```text
sampling_rate_hz = 64000
signal_name      = vibration_1
channels         = 4 conditions x 20 recordings x 26 bearings
```

Every row carries `archive_sha256`, `raw_sha256`, `raw_bytes`,
`acquisition_datetime_local_unzoned`, and `vibration_samples`.

## Frozen hashes

```text
all_records.csv          05d5a6c7ba6ce4650e6a5a40dda7caee74a0bf615e9e4804a971247775d70e84
initial_train.csv        6345ada41d92d12c7f115316f1948ef8724b7a01497f3bf75a24178049a438fd
initial_validation.csv   3f1ff325de39cfb2892c802099e86b558c624d540b98f841546f0e8f2ff3dab5
batch_01.csv             64d05dcafd32a7e6dce79ac28d50e02bad2c0dd1ef8ab796dc04840fd1759918
batch_02.csv             15ba5a8008eb63e3abb6b526dc4dd71dccadd88211eb0706548364202e04e3b7
final_holdout.csv        dbd704911b465e9f6e436922ae1d2f4e17404fcc1814c5be8bb86f26938ceffe
development.csv          711ba00cae16a3192f9365220fb5b30354a2144520b25c28e981275ea4218e3e
```

`development.csv` is derived from `initial_train.csv` and
`initial_validation.csv`.  It changes the `path` prefix to
`data/paderborn/raw/...` and maps partitions to `train`/`validation`; it does not
add records.  It is the manifest used by Phase 2.  Its hash is frozen.

## Rules

1. `final_holdout.csv` is never passed to training, feature selection, drift
   calibration, or update decisions before the final evaluation.
2. Do not edit a manifest by hand. Regenerate with
   `python scripts/phase1_audit.py --hash --deep` and re-run
   `python scripts/verify_manifests.py`.
3. A failed audit must not produce a "best effort" split.
4. Keep the old exploratory manifest out of this directory; it is superseded and
   record-level.

## Known issues

- The old exploratory manifest omitted four KA04 records. The new audit restores
  them after byte-hash and deep signal checks.
- `PYTHONHASHSEED` is irrelevant to the split because the assignment is
  pre-registered, not randomized at runtime.
- Paderborn is CC BY-NC 4.0; do not redistribute the raw MAT payload.
