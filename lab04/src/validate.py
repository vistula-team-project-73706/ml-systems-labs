"""
Step 2: Validation.

Runs schema, type, category, range, missing-value, duplicate and
cross-field checks on the raw data. Every check reports how many rows
violate it and gives a few example row indices. Nothing is dropped or
changed here: the job of this step is to report, not to fix.
"""

import logging

import pandas as pd

import config

log = logging.getLogger(__name__)


def _examples(mask, limit=5):
    return ";".join(str(i) for i in mask[mask].index[:limit])


def _result(check, column, mask, rule):
    n = int(mask.sum())
    return {
        "check": check,
        "column": column,
        "rule": rule,
        "violations": n,
        "status": "PASS" if n == 0 else "FAIL",
        "example_rows": _examples(mask),
    }


def check_schema(raw):
    expected = list(config.SCHEMA_TYPES)
    missing = [c for c in expected if c not in raw.columns]
    extra = [c for c in raw.columns if c not in expected]
    return {
        "check": "schema_columns",
        "column": "*",
        "rule": f"exactly {len(expected)} expected columns",
        "violations": len(missing) + len(extra),
        "status": "PASS" if not missing and not extra else "FAIL",
        "example_rows": f"missing={missing} extra={extra}",
    }


def check_types(raw):
    rows = []
    for col, kind in config.SCHEMA_TYPES.items():
        values = raw[col]
        if kind == "int":
            # '?' in a numeric column would also be a type violation.
            bad = ~values.str.fullmatch(r"-?\d+")
            rows.append(_result("type", col, bad, "integer"))
        else:
            allowed = set(config.ALLOWED_CATEGORIES[col]) | {config.MISSING_TOKEN}
            bad = ~values.isin(allowed)
            rows.append(_result("allowed_categories", col, bad, "value in data dictionary or '?'"))
    return rows


def check_ranges(raw):
    rows = []
    for col, (low, high) in config.RANGES.items():
        numbers = pd.to_numeric(raw[col], errors="coerce")
        bad = (numbers < low) | (numbers > high)
        rows.append(_result("range", col, bad, f"{low} <= value <= {high}"))
    return rows


def check_missing(raw):
    rows = []
    for col in raw.columns:
        is_missing = (raw[col] == config.MISSING_TOKEN) | (raw[col].str.strip() == "")
        n = int(is_missing.sum())
        rows.append({
            "check": "missing",
            "column": col,
            "rule": "count of '?' (informational, allowed by null policy)",
            "violations": n,
            "status": "INFO" if n else "PASS",
            "example_rows": _examples(is_missing),
        })
    return rows


def check_duplicates(raw):
    exact = raw.duplicated(keep="first")
    features_only = raw.duplicated(subset=config.FEATURE_COLUMNS, keep=False)
    conflicting = features_only & (
        raw.groupby(config.FEATURE_COLUMNS, sort=False)[config.TARGET].transform("nunique") > 1
    )
    return [
        _result("duplicates_exact", "*", exact, "no repeated rows (all 15 columns)"),
        _result("duplicates_conflicting_label", "*", conflicting,
                "identical 14 features must have the same income label"),
    ]


def check_cross_field(raw):
    """Consistency rules taken from the data dictionary."""
    rows = []

    expected_num = raw["education"].map(config.EDUCATION_TO_NUM)
    actual_num = pd.to_numeric(raw["education-num"], errors="coerce")
    rows.append(_result("cross_field", "education/education-num",
                        expected_num != actual_num, "education-num matches education level"))

    married = raw["marital-status"].str.startswith("Married")
    spouse = raw["relationship"].isin(["Husband", "Wife"])
    rows.append(_result("cross_field", "relationship/marital-status",
                        spouse & ~married, "Husband/Wife implies a Married-* status"))

    rows.append(_result("cross_field", "relationship/sex",
                        (raw["relationship"] == "Husband") & (raw["sex"] == "Female"),
                        "Husband is Male (as coded in the 1994 census)"))
    rows.append(_result("cross_field", "relationship/sex",
                        (raw["relationship"] == "Wife") & (raw["sex"] == "Male"),
                        "Wife is Female (as coded in the 1994 census)"))

    never_worked = raw["workclass"] == "Never-worked"
    rows.append(_result("cross_field", "workclass/occupation",
                        never_worked & (raw["occupation"] != config.MISSING_TOKEN),
                        "Never-worked has no occupation"))
    return rows


def validate(raw):
    """Run all checks and return a DataFrame report."""
    results = [check_schema(raw)]
    results += check_types(raw)
    results += check_ranges(raw)
    results += check_missing(raw)
    results += check_duplicates(raw)
    results += check_cross_field(raw)

    report = pd.DataFrame(results)
    n_fail = int((report["status"] == "FAIL").sum())
    log.info("Validation: %d checks, %d failed, %d informational",
             len(report), n_fail, int((report["status"] == "INFO").sum()))
    for _, r in report[report["status"] == "FAIL"].iterrows():
        log.info("  FAIL %-28s %-28s %d rows", r["check"], r["column"], r["violations"])

    # Validation never drops rows.
    log.info("Rows after validation: %d (unchanged)", len(raw))
    return report


def write_markdown(report, path):
    lines = [
        "# Validation report",
        "",
        "Generated by `lab04/src/pipeline.py`. Rows are reported, never dropped.",
        "",
        "| Check | Column | Rule | Violations | Status | Example rows |",
        "|---|---|---|---:|---|---|",
    ]
    for _, r in report.iterrows():
        lines.append(
            f"| {r['check']} | {r['column']} | {r['rule']} | {r['violations']} "
            f"| {r['status']} | {r['example_rows']} |"
        )
    path.write_text("\n".join(lines) + "\n")
