"""
Steps 3-4: Split, cleaning, transformation and derived features.

Leakage rules followed here:
    - The 70/30 stratified split happens before any cleaning.
    - Every fitted statistic (imputer mode/median, clip percentiles,
      scaler mean/std, one-hot categories) is learned from train only.
    - Exact duplicates are removed from train only; test is untouched.
    - Derived features only use values from the same row, so they
      cannot leak information between rows.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

import config

log = logging.getLogger(__name__)

DERIVED_COLUMNS = ["capital_net", "hours_per_age"]


# ============================================================
# SPLIT AND DUPLICATE REMOVAL
# ============================================================

def to_typed_frame(raw):
    """Turn the raw string table into typed columns. '?' becomes NaN."""
    df = raw.replace(config.MISSING_TOKEN, np.nan)
    for col in config.NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col])
    return df


def split(raw):
    df = to_typed_frame(raw)
    X = df[config.FEATURE_COLUMNS]
    y = (df[config.TARGET] == config.POSITIVE_LABEL).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.SEED,
    )
    log.info("Split: train=%d test=%d (positive rate train=%.4f test=%.4f)",
             len(X_train), len(X_test), y_train.mean(), y_test.mean())
    return X_train, X_test, y_train, y_test


def drop_train_duplicates(X_train, y_train):
    combined = X_train.assign(**{config.TARGET: y_train})
    dup = combined.duplicated(keep="first")
    log.info("Dropping %d exact duplicate rows from train only (test untouched)", int(dup.sum()))
    return X_train[~dup], y_train[~dup], int(dup.sum())


# ============================================================
# TRANSFORMERS
# ============================================================

def add_derived_features(X):
    X = X.copy()
    X["capital_net"] = X["capital-gain"] - X["capital-loss"]
    # Schema guarantees age >= 17, so (age - 16) >= 1 and never divides by zero.
    X["hours_per_age"] = X["hours-per-week"] / (X["age"] - 16)
    return X


def derived_feature_names(transformer, input_features):
    return np.asarray(list(input_features) + DERIVED_COLUMNS, dtype=object)


class PercentileClipper(BaseEstimator, TransformerMixin):
    """Clip each column to percentiles learned during fit (train only)."""

    def __init__(self, lower=1, upper=99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.low_ = np.nanpercentile(X, self.lower, axis=0)
        self.high_ = np.nanpercentile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_preprocessor(with_features):
    numeric = list(config.NUMERIC_COLUMNS)
    if with_features:
        numeric += DERIVED_COLUMNS

    low, high = config.CLIP_PERCENTILES
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clip", PercentileClipper(lower=low, upper=high)),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    columns = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, config.CATEGORICAL_COLUMNS),
    ])

    steps = []
    if with_features:
        steps.append(("derive", FunctionTransformer(add_derived_features,
                                                    feature_names_out=derived_feature_names)))
    steps.append(("columns", columns))
    return Pipeline(steps)


def describe_fitted(preprocessor):
    """Return train-learned statistics so they can be written to config.yaml."""
    columns = preprocessor.named_steps["columns"]
    num = columns.named_transformers_["num"]
    cat = columns.named_transformers_["cat"]
    num_cols = columns.transformers_[0][2]

    return {
        "numeric_median": {c: float(v) for c, v in zip(num_cols, num.named_steps["impute"].statistics_)},
        "clip_low_p1": {c: float(v) for c, v in zip(num_cols, num.named_steps["clip"].low_)},
        "clip_high_p99": {c: float(v) for c, v in zip(num_cols, num.named_steps["clip"].high_)},
        "categorical_mode": {c: str(v) for c, v in
                             zip(config.CATEGORICAL_COLUMNS, cat.named_steps["impute"].statistics_)},
    }
