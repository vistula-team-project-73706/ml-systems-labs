"""
Step 1: Ingestion.

Downloads UCI Adult from OpenML once and stores an untouched raw snapshot
in data/raw/adult_raw.csv. OpenML already decodes the original '?'
markers as NaN, so the snapshot writes them back as '?' to match the
original UCI file. No rows or values are changed.

Later steps always read the snapshot as plain strings, so ingestion
never modifies the data.
"""

import hashlib
import logging

import pandas as pd
from sklearn.datasets import fetch_openml

import config

log = logging.getLogger(__name__)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_raw():
    """Fetch from OpenML and write the raw snapshot if it does not exist."""
    if config.RAW_FILE.exists():
        log.info("Raw snapshot already exists: %s", config.RAW_FILE.relative_to(config.PROJECT_ROOT))
        return

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    bunch = fetch_openml(
        name=config.DATASET["name"],
        version=config.DATASET["version"],
        as_frame=True,
        data_home=config.DATA_DIR / "openml_cache",
        parser="auto",
    )
    frame = bunch.frame.rename(columns={"class": config.TARGET})
    frame.to_csv(config.RAW_FILE, index=False, na_rep=config.MISSING_TOKEN)
    log.info("Downloaded raw data from OpenML to %s", config.RAW_FILE.relative_to(config.PROJECT_ROOT))


def ingest():
    """Return the raw data exactly as stored (all columns as strings)."""
    download_raw()

    # dtype=str and keep_default_na=False: read every value literally,
    # so '?' stays '?' and nothing is converted or dropped here.
    raw = pd.read_csv(config.RAW_FILE, dtype=str, keep_default_na=False)

    n_rows, n_cols = raw.shape
    log.info("Ingested rows=%d columns=%d", n_rows, n_cols)
    log.info("Raw file sha256=%s", sha256_file(config.RAW_FILE))

    if n_rows != config.DATASET["expected_rows"] or n_cols != config.DATASET["expected_columns"]:
        log.warning(
            "Shape %s differs from expected (%d, %d)",
            raw.shape, config.DATASET["expected_rows"], config.DATASET["expected_columns"],
        )

    return raw
