# Lab 4 — Data Pipeline and Quality

## Goal

Build a data pipeline for UCI Adult that is reproducible and free of
leakage. It ingests the raw data without changing it, validates it
against a written schema, cleans it using statistics learned only from
the training set, adds derived features, audits label quality, and
outputs a processed dataset with a SHA256 hash. The full problem
definition is in `problem_definition.md`, and the data card is in
`data_card.md`.

## Method

The pipeline is split into one module per stage in `lab04/src/`:

| Stage | File | What it does |
|---|---|---|
| Config | `config.py` | Paths, schema, allowed categories, ranges, seeds |
| Ingestion | `ingest.py` | Downloads OpenML Adult v2 once, saves an untouched snapshot, logs the shape and file hash |
| Validation | `validate.py` | 42 checks for schema, type, category, range, missing values, duplicates and cross-field rules. Reports only, drops nothing |
| Cleaning and features | `transform.py` | Stratified split, then a scikit-learn `Pipeline` fitted on train only |
| Label audit | `label_audit.py` | Duplicate-conflict audit and a check of a 50-row sample |
| Orchestration | `pipeline.py` | Runs everything, saves results, hashes and `config.yaml` |

**One command:** `python lab04/src/pipeline.py`. All output goes to
`lab04/results/`, and a full log is written to `results/pipeline.log`.

### Ingestion

OpenML already converts the original `'?'` markers to NaN. To follow
the lab's null policy exactly, the ingestion step writes the snapshot
with `'?'` for missing values (the original UCI format). Every later
step reads that file as plain strings (`dtype=str`,
`keep_default_na=False`), so nothing is converted or dropped on the
way in.

- Logged: **rows = 48,842, columns = 15**. This matches the lab sheet.

### Leakage controls

1. A 70/30 split stratified on `income` (seed 42) happens **before**
   any cleaning.
2. Exact duplicate rows are dropped from **train only**. 24 rows were
   dropped and the test set was left alone.
3. The imputers (median for numeric, mode for categorical), the 1st and
   99th percentile clip limits, the scaler and the one-hot categories
   are all fitted inside one scikit-learn `Pipeline` on train only.
   Because they live inside the pipeline, cross-validation re-fits them
   on each fold's training part, so even the CV scores have no leakage.
4. The derived features use only values from the same row, so they
   cannot leak anything between rows.

The clip limits were written by a custom `PercentileClipper`
transformer so they are stored inside the saved preprocessor. For
example, age is clipped to 17–74, hours to 8–80 and capital-gain to
0–15,024. All learned values are in `results/fitted_statistics.json`.

## Results

### Validation report

There were 42 checks: 35 passed, 3 were informational (missing values,
which the null policy allows) and 4 failed. No rows were dropped. The
full table is in `results/validation_report.md`.

| Check | Result |
|---|---|
| Schema: 15 expected columns | PASS |
| Types: 6 numeric columns are integers | PASS (all 6) |
| Categories: values in the data dictionary or `'?'` | PASS (all 9 categorical columns) |
| Range: age 17–90, hours 1–99, capital-gain 0–99,999, capital-loss 0–4,356 | PASS (all 4) |
| Missing `'?'` | workclass 2,799 · occupation 2,809 · native-country 857 (3,620 rows in total) |
| Exact duplicate rows | **FAIL: 52 rows** |
| Same features, different label | **FAIL: 10 rows** |
| education-num matches education | PASS |
| Husband/Wife ⇒ married status | PASS |
| Husband coded as Female | **FAIL: 1 row** |
| Wife coded as Male | **FAIL: 3 rows** |
| Never-worked has no occupation | PASS |

The range checks pass, but they hide one issue: capital-gain has
**244 rows at exactly 99,999**, which is the top-code (placeholder) in
the census data, not a real amount. Clipping to the 99th percentile
(15,024) caps these rows along with every other large gain.

### Derived features vs baseline (LogisticRegression)

| Setting | Features after encoding | Test accuracy | Test F1 | Train 5-fold CV F1 |
|---|---:|---:|---:|---:|
| Baseline (14 features) | 105 | 0.8540 | 0.6601 | 0.6552 ± 0.0118 |
| + capital_net, hours_per_age | 107 | **0.8578** | **0.6696** | 0.6627 ± 0.0132 |

On the test set the derived features improve accuracy by 0.4 points
and F1 by about 1 point. The CV gain (+0.0075 F1) is smaller than one
standard deviation across folds, so the improvement is real on this
split but small. A likely reason it helps: logistic regression is
linear, and `capital_net` puts gains and losses on a single axis.
`hours_per_age` gives the model a ratio it cannot build from a
weighted sum of age and hours.

### Label-quality audit

**Duplicate-conflict audit** (`results/duplicate_conflicts.csv`):

| Feature set used for matching | Rows in duplicate groups | Rows with conflicting labels | Conflicting groups |
|---|---:|---:|---:|
| All 14 features | 111 | **10** | **5** |
| 13 features (fnlwgt removed) | 11,174 | 3,873 | 1,029 |

With all 14 features, 5 pairs of rows are identical in every feature,
including `fnlwgt`, but have opposite income labels. At least one label
in each pair has to be wrong, or the features don't contain enough
information to decide. That is 0.02% of the data, so the label noise is
small. When `fnlwgt` (a census sampling weight) is removed, thousands
of people share the same profile with different incomes. This shows
that the features cannot fully determine income: some of the error is
noise that no model can remove (irreducible error).

**50-row manual check** (`results/label_audit_sample.csv`, seed 42): I
compared each sampled row with the data dictionary. All 50 have
in-range ages, hours and capital values, valid category codes, a
matching education and education-num, and consistent relationship and
marital status. No impossible values were found. 3 of the 50 rows have
`'?'` values: two retired-age men (62 and 67) are missing both
workclass and occupation, and one person is missing native-country.
This fits the idea that the missing workclass and occupation values
mostly belong to people outside the workforce. I did not change any
labels, because Adult labels come from census answers and cannot be
relabelled by hand.

### Processed-data hash and reproducibility

Hashes are stored in `results/config.yaml` together with every setting
used in the run:

| File | SHA256 |
|---|---|
| Raw snapshot | `cb49122a…66d11` |
| Processed train (34,165 × 108) | `253cc302…82b862` |
| Processed test (14,653 × 108) | `ee20c623…03b7` |
| Combined | `374b540d…9e1034e` |

To test reproducibility I deleted the whole `data/` folder (raw,
processed and the OpenML cache) and ran the single command again. It
downloaded the data again and produced a **byte-identical
`config.yaml`**, which means every hash matched. The processed CSVs are
written with a fixed float format and line ending so that the hash
depends only on the data.

## Conclusions

1. Validation should report problems, not fix them. The data passes
   every type and range check, but the useful findings were in the
   checks that failed or needed thought: 52 exact duplicates, 10 rows
   with conflicting labels, 4 relationship/sex inconsistencies, and a
   99,999 sentinel value that looks valid. Silently dropping rows would
   have hidden all of this.
2. Putting every fitted step inside one scikit-learn `Pipeline` that is
   fitted on train only is what keeps the pipeline free of leakage. It
   also makes cross-validation honest, and the whole thing can be saved
   as a single `preprocessor.joblib` file.
3. The derived features help a linear model a little (+1 F1 point on
   test), but the gain is about the size of the variation between CV
   folds. The duplicate audit points to a bigger limit: thousands of
   identical profiles have different incomes, so the features alone
   cannot reach perfect accuracy.

## Outputs

- `problem_definition.md`, `data_card.md`
- `results/pipeline.log`: the full run log, including the
  48,842 × 15 ingestion count
- `results/validation_report.md` and `.csv`
- `results/preprocessor.joblib`, `results/fitted_statistics.json`
- `results/feature_comparison.csv`
- `results/label_audit.json`, `results/duplicate_conflicts.csv`,
  `results/label_audit_sample.csv`
- `results/config.yaml`: configuration and SHA256 hashes
- `results/versions.txt`
