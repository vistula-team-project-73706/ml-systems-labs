"""
Lab 04: Data pipeline and quality
Dataset: UCI Adult (OpenML, version 2)

One command runs everything:
    python lab04/src/pipeline.py

Steps:
    1. Ingest raw data (unchanged snapshot, row/column count logged)
    2. Validate schema, types, ranges, missing values, duplicates
    3. Split 70/30 stratified, then clean using train-only statistics
    4. Compare LogisticRegression with and without derived features
    5. Label-quality audit (duplicate conflicts + 50-row sample)
    6. Save processed data, SHA256 hash and config.yaml

Outputs (lab04/results):
    pipeline.log, validation_report.md/.csv, preprocessor.joblib,
    feature_comparison.csv, label_audit.json, label_audit_sample.csv,
    duplicate_conflicts.csv, fitted_statistics.json, config.yaml
"""

import hashlib
import json
import logging
import random
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

import config
import label_audit
import transform
import validate
from ingest import ingest, sha256_file

log = logging.getLogger("lab04")


def setup_logging():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.RESULTS_DIR / "pipeline.log", mode="w"),
        ],
    )


def set_seeds():
    random.seed(config.SEED)
    np.random.seed(config.SEED)


# ============================================================
# FEATURE EXPERIMENT
# ============================================================

def evaluate_features(X_train, X_test, y_train, y_test):
    rows = []
    fitted = {}
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)

    for with_features in (False, True):
        name = "with_derived_features" if with_features else "baseline_features"
        model = Pipeline([
            ("prep", transform.build_preprocessor(with_features)),
            ("clf", LogisticRegression(max_iter=config.LOGREG_MAX_ITER)),
        ])

        # Cross-validation on train only gives a spread, not just one number.
        cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "setting": name,
            "n_features_after_encoding": len(model.named_steps["prep"].get_feature_names_out()),
            "test_accuracy": round(accuracy_score(y_test, pred), 4),
            "test_f1": round(f1_score(y_test, pred), 4),
            "cv_f1_mean": round(cv_f1.mean(), 4),
            "cv_f1_std": round(cv_f1.std(), 4),
        })
        fitted[name] = model
        log.info("%s: test acc=%.4f f1=%.4f | train %d-fold CV f1=%.4f +- %.4f",
                 name, rows[-1]["test_accuracy"], rows[-1]["test_f1"],
                 config.CV_FOLDS, cv_f1.mean(), cv_f1.std())

    return pd.DataFrame(rows), fitted


# ============================================================
# PROCESSED DATA + HASH
# ============================================================

def save_processed(preprocessor, X, y, path):
    names = preprocessor.get_feature_names_out()
    data = pd.DataFrame(preprocessor.transform(X), columns=names)
    data[config.TARGET] = y.to_numpy()
    # Fixed float format and line endings so the bytes (and hash) are reproducible.
    data.to_csv(path, index=False, float_format="%.6f", lineterminator="\n")
    return sha256_file(path)


def combined_hash(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(p.read_bytes())
    return h.hexdigest()


def to_yaml(value, indent=0):
    """Minimal YAML writer for nested dicts/lists of plain values."""
    pad = "  " * indent
    lines = []
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{pad}{k}:")
                lines.append(to_yaml(v, indent + 1))
            else:
                lines.append(f"{pad}{k}: {json.dumps(v)}")
    else:
        for v in value:
            lines.append(f"{pad}- {json.dumps(v)}")
    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    setup_logging()
    set_seeds()
    out = config.RESULTS_DIR

    log.info("=== 1. Ingestion ===")
    raw = ingest()
    raw_hash = sha256_file(config.RAW_FILE)

    log.info("=== 2. Validation ===")
    report = validate.validate(raw)
    report.to_csv(out / "validation_report.csv", index=False)
    validate.write_markdown(report, out / "validation_report.md")

    log.info("=== 3. Split and cleaning ===")
    X_train, X_test, y_train, y_test = transform.split(raw)
    n_train_before = len(X_train)
    X_train, y_train, n_dups = transform.drop_train_duplicates(X_train, y_train)

    log.info("=== 4. Derived features vs baseline ===")
    comparison, fitted = evaluate_features(X_train, X_test, y_train, y_test)
    comparison.to_csv(out / "feature_comparison.csv", index=False)

    preprocessor = fitted["with_derived_features"].named_steps["prep"]
    joblib.dump(preprocessor, out / "preprocessor.joblib")
    log.info("Saved fitted preprocessor to %s", (out / "preprocessor.joblib").relative_to(config.PROJECT_ROOT))

    fitted_stats = transform.describe_fitted(preprocessor)
    (out / "fitted_statistics.json").write_text(json.dumps(fitted_stats, indent=2))

    log.info("=== 5. Label-quality audit ===")
    audit_results, conflict_rows, sample = label_audit.audit(raw)
    (out / "label_audit.json").write_text(json.dumps(audit_results, indent=2))
    conflict_rows.sort_values(config.FEATURE_COLUMNS).to_csv(out / "duplicate_conflicts.csv")
    sample.to_csv(out / "label_audit_sample.csv")

    log.info("=== 6. Processed data hash and config ===")
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_path = config.PROCESSED_DIR / "adult_train_processed.csv"
    test_path = config.PROCESSED_DIR / "adult_test_processed.csv"
    train_hash = save_processed(preprocessor, X_train, y_train, train_path)
    test_hash = save_processed(preprocessor, X_test, y_test, test_path)
    all_hash = combined_hash([train_path, test_path])
    log.info("Processed train sha256=%s", train_hash)
    log.info("Processed test  sha256=%s", test_hash)
    log.info("Processed combined sha256=%s", all_hash)

    run_config = {
        "dataset": config.DATASET,
        "target": {"column": config.TARGET, "positive_label": config.POSITIVE_LABEL},
        "null_policy": {"missing_token": config.MISSING_TOKEN, "action": "report in validation, impute after split"},
        "schema": {"types": config.SCHEMA_TYPES, "ranges": {k: list(v) for k, v in config.RANGES.items()}},
        "split": {"test_size": config.TEST_SIZE, "stratify": config.TARGET, "random_state": config.SEED},
        "cleaning": {
            "numeric_imputer": "median (train only)",
            "categorical_imputer": "most_frequent (train only)",
            "clip_percentiles": list(config.CLIP_PERCENTILES),
            "duplicates": "exact duplicates dropped from train only",
            "scaler": "StandardScaler (train only)",
            "encoder": "OneHotEncoder(handle_unknown='ignore')",
        },
        "derived_features": {
            "capital_net": "capital-gain - capital-loss",
            "hours_per_age": "hours-per-week / (age - 16)",
        },
        "model": {"type": "LogisticRegression", "max_iter": config.LOGREG_MAX_ITER, "cv_folds": config.CV_FOLDS},
        "counts": {
            "raw_rows": len(raw),
            "raw_columns": raw.shape[1],
            "train_rows_before_dedup": n_train_before,
            "train_duplicates_dropped": n_dups,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "processed_columns": len(preprocessor.get_feature_names_out()) + 1,
        },
        "hashes_sha256": {
            "raw_file": raw_hash,
            "processed_train": train_hash,
            "processed_test": test_hash,
            "processed_combined": all_hash,
        },
    }
    (out / "config.yaml").write_text(
        "# Generated by lab04/src/pipeline.py. Do not edit by hand.\n" + to_yaml(run_config) + "\n"
    )
    log.info("Wrote %s", (out / "config.yaml").relative_to(config.PROJECT_ROOT))
    log.info("Pipeline finished")


if __name__ == "__main__":
    main()
