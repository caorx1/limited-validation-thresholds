# Limited-validation threshold selection: minimal verification package

Companion code for “Choosing classification thresholds with limited validation labels,” by Ruixia Cao.

## What is provided

- The unchanged S1 empirical, S2 bootstrap-median and S3 grid threshold implementation; S0 is the fixed threshold 0.5.
- Reduced caches for all 45 original source/model/outer-partition tasks: test probabilities, labels, final thresholds and test Macro-F1 values for the original eligible budgets. Redundant internal-bootstrap and validation records are omitted.
- 27,000 primary paired gains and two frozen comparison tables. The code reconstructs mean gains and loss frequencies and checks source-level decision disagreement.
- Saved V2 task and reference summaries for the two feature-inclusion comparisons and their equal-source combinations.

## Run

Tested with Python 3.12.14, NumPy 2.3.5 and pandas 3.0.1. These are observed working versions, not a certified fresh installation.

```text
python -m pip install -r requirements.txt
python code/verify_saved_predictions.py
python code/rebuild_summaries.py
python code/rebuild_v2.py
```

The scripts also work from another current directory when invoked by their full paths. The first checks 82,800 Macro-F1 values, 27,000 paired gains and 92 disagreement summaries. The second checks 108 primary summary values. The third starts with 288 saved V2 task summaries, checks 288 source summary values and 160 equal-source configuration values, and writes the two rebuilt tables. Each script writes a record in `reports/`.

Input checks and scientific comparisons use Python assertions: do not run with python -O. Comparing to provided reference tables checks consistency within the archived results; it does not independently establish the accuracy of source labels or model fitting.

## Reproduction boundary

The scripts verify saved results. End-to-end fitting, validation sampling, bootstrap searches, V1 and figure generation are outside this package. V2 starts from saved summaries. The selector code is included for inspection.

## Licensing and attribution

Project code/documentation: MIT, see LICENSE. Result CSV files and derived caches: CC BY 4.0, see DATA_LICENSE.md. Preserve original source attributions in DATA_SOURCES.md. No raw predictors, paper drafts, author replies, machine paths, credentials or internal review logs are included. Dependencies are installed separately, see THIRD_PARTY_NOTICES.md.

The source hashes in `cache_manifest.json` refer to the full experiment files. The reduced files have separate hashes. The `input_manifest.json` lists the files in this package before the verification scripts produce their reports.
