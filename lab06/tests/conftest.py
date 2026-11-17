import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import load_config


@pytest.fixture
def fake_cfg(tmp_path):
    """A tiny run on random images, with MLflow and checkpoints in tmp_path."""

    return load_config(overrides=[
        "data.dataset=fake",
        "data.train_size=200",
        "data.val_size=40",
        "data.batch_size=32",
        "data.eval_batch_size=64",
        "train.epochs=2",
        f"checkpoint.dir={tmp_path / 'checkpoints'}",
        "mlflow.enabled=false",
        f"mlflow.tracking_uri={tmp_path / 'mlruns'}",
    ])
