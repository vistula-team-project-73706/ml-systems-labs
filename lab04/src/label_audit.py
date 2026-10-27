"""
Step 6: Label-quality audit.

Adult labels come from census answers, so they cannot be relabelled by
hand. Instead the lab asks for:
    1. A duplicate-conflict audit: rows whose 14 features are identical
       but whose income labels differ.
    2. A random sample of 50 rows checked against the data dictionary
       for impossible values.
"""

import logging

import pandas as pd

import config

log = logging.getLogger(__name__)


def duplicate_conflicts(raw, feature_columns):
    groups = raw.groupby(feature_columns, sort=False)[config.TARGET]
    n_labels = groups.transform("nunique")
    group_size = groups.transform("size")

    duplicated = group_size > 1
    conflicting = n_labels > 1
    conflict_rows = raw[conflicting]
    n_groups = conflict_rows.groupby(feature_columns, sort=False).ngroups

    return {
        "duplicate_feature_rows": int(duplicated.sum()),
        "conflicting_rows": int(conflicting.sum()),
        "conflicting_groups": int(n_groups),
    }, conflict_rows


def sample_check(raw):
    """Check 50 random rows against the data dictionary."""
    sample = raw.sample(n=config.LABEL_AUDIT_SAMPLE_SIZE, random_state=config.SEED).copy()

    problems = []
    for idx, row in sample.iterrows():
        issues = []
        for col, (low, high) in config.RANGES.items():
            value = int(row[col])
            if not low <= value <= high:
                issues.append(f"{col}={value} outside {low}-{high}")
        for col in config.CATEGORICAL_COLUMNS + [config.TARGET]:
            allowed = set(config.ALLOWED_CATEGORIES[col]) | {config.MISSING_TOKEN}
            if row[col] not in allowed:
                issues.append(f"{col}='{row[col]}' not in dictionary")
        if config.EDUCATION_TO_NUM[row["education"]] != int(row["education-num"]):
            issues.append("education-num does not match education")
        if row["relationship"] in ("Husband", "Wife") and not row["marital-status"].startswith("Married"):
            issues.append("spouse relationship but not married")
        if int(row["age"]) < 18 and row["marital-status"] != "Never-married":
            issues.append("age < 18 but has been married")
        if row["workclass"] == "Never-worked" and int(row["hours-per-week"]) > 0 and row["occupation"] != "?":
            issues.append("Never-worked but has an occupation")
        missing = [c for c in config.FEATURE_COLUMNS if row[c] == config.MISSING_TOKEN]
        if missing:
            issues.append("missing: " + ",".join(missing))
        problems.append("; ".join(issues) if issues else "OK")

    sample["audit_result"] = problems
    return sample


def audit(raw):
    results = {}

    full, conflict_rows = duplicate_conflicts(raw, config.FEATURE_COLUMNS)
    results["all_14_features"] = full
    log.info("Duplicate-conflict audit (14 features): %s", full)

    # fnlwgt is a census sampling weight, not a property of the person.
    # Without it many more "same person profile" duplicates appear.
    no_weight = [c for c in config.FEATURE_COLUMNS if c != "fnlwgt"]
    loose, _ = duplicate_conflicts(raw, no_weight)
    results["13_features_without_fnlwgt"] = loose
    log.info("Duplicate-conflict audit (13 features, no fnlwgt): %s", loose)

    sample = sample_check(raw)
    n_ok = int((sample["audit_result"] == "OK").sum())
    log.info("50-row sample: %d OK, %d with notes", n_ok, len(sample) - n_ok)

    return results, conflict_rows, sample
