"""Tests for the Lab 06 pipeline. All runs use random images and finish in seconds."""

import copy

import mlflow
import pytest
import torch

from config import apply_override, flatten, load_config, resolve_path
from model import CNN
from profile_epoch import PHASES, phase_breakdown, profile_one_epoch
from train import run


# ============================================================
# TASK 1: CONFIG, SEED, CHECKPOINTS, MLFLOW
# ============================================================

def test_base_config_loads_every_section():
    cfg = load_config()

    for section in ["experiment", "data", "model", "optimizer", "scheduler",
                    "train", "checkpoint", "mlflow"]:
        assert section in cfg
    assert cfg["experiment"]["seed"] == 42


def test_override_parses_value_type():
    cfg = load_config(overrides=["train.epochs=10", "optimizer.lr=0.05", "checkpoint.resume=null"])

    assert cfg["train"]["epochs"] == 10
    assert cfg["optimizer"]["lr"] == 0.05
    assert cfg["checkpoint"]["resume"] is None


def test_override_rejects_unknown_key():
    with pytest.raises(KeyError):
        apply_override(load_config(), "train.epoch=10")


def test_flatten_for_mlflow_params():
    assert flatten({"a": {"b": 1, "c": {"d": 2}}}) == {"a.b": 1, "a.c.d": 2}


def test_model_is_the_lab03_cnn():
    model = CNN()

    assert model(torch.zeros(2, 1, 28, 28)).shape == (2, 10)
    assert sum(p.numel() for p in model.parameters()) == 20_490


def test_same_seed_gives_same_run(fake_cfg):
    first = run(copy.deepcopy(fake_cfg))
    second = run(copy.deepcopy(fake_cfg))

    # Timings differ between runs, the numbers computed from the data must not.
    for a, b in zip(first["history"], second["history"]):
        assert a["train_loss"] == b["train_loss"]
        assert a["val_accuracy"] == b["val_accuracy"]


def test_checkpoint_saved_every_epoch(fake_cfg):
    run(fake_cfg)
    folder = resolve_path(fake_cfg["checkpoint"]["dir"]) / fake_cfg["experiment"]["run_name"]

    assert sorted(p.name for p in folder.glob("*.pt")) == ["epoch_001.pt", "epoch_002.pt", "last.pt"]


def test_mlflow_logs_params_metrics_and_artifacts(fake_cfg):
    apply_override(fake_cfg, "mlflow.enabled=true")
    summary = run(fake_cfg)

    mlflow.set_tracking_uri(resolve_path(fake_cfg["mlflow"]["tracking_uri"]).as_uri())
    logged = mlflow.get_run(summary["mlflow_run_id"])
    client = mlflow.MlflowClient()
    artifacts = {a.path for a in client.list_artifacts(summary["mlflow_run_id"])}

    assert logged.data.params["optimizer.name"] == "adam"
    assert "test_accuracy" in logged.data.metrics
    assert len(client.get_metric_history(summary["mlflow_run_id"], "val_accuracy")) == 2
    assert {"config.yaml", "history.csv", "last.pt"} <= artifacts


# ============================================================
# TASK 2: PROFILER BREAKDOWN
# ============================================================

def test_profile_breakdown_covers_the_four_phases(fake_cfg):
    prof, wall, epoch_seconds, batches = profile_one_epoch(fake_cfg)
    rows = phase_breakdown(prof.key_averages(), wall, epoch_seconds)
    shares = {row["phase"]: row["profiler_share"] for row in rows if row["phase"] in PHASES}

    assert set(shares) == set(PHASES)
    assert all(share > 0 for share in shares.values())
    assert sum(shares.values()) == pytest.approx(1.0)


# ============================================================
# TASK 5: SCHEDULERS AND OPTIMIZERS
# ============================================================

@pytest.mark.parametrize("optimizer", ["adam", "sgd"])
@pytest.mark.parametrize("scheduler", ["none", "step", "cosine", "plateau"])
def test_every_optimizer_and_scheduler_runs(fake_cfg, optimizer, scheduler):
    apply_override(fake_cfg, f"optimizer.name={optimizer}")
    apply_override(fake_cfg, f"scheduler.name={scheduler}")
    apply_override(fake_cfg, "scheduler.step_size=1")

    summary = run(fake_cfg)

    assert len(summary["history"]) == 2


def test_step_scheduler_lowers_lr(fake_cfg):
    for item in ["scheduler.name=step", "scheduler.step_size=1", "scheduler.gamma=0.1"]:
        apply_override(fake_cfg, item)

    history = run(fake_cfg)["history"]

    assert history[1]["lr"] == pytest.approx(history[0]["lr"] * 0.1)


# ============================================================
# TASK 6: GRADIENT ACCUMULATION
# ============================================================

def test_accumulation_matches_one_big_batch(fake_cfg):
    # 200 samples do not divide into groups of 64, so the smaller last group
    # is tested too. Same shuffle seed means the same samples in each group.
    big = copy.deepcopy(fake_cfg)
    apply_override(big, "data.batch_size=64")

    small = copy.deepcopy(fake_cfg)
    apply_override(small, "data.batch_size=16")
    apply_override(small, "train.accumulation_steps=4")

    big_history = run(big)["history"]
    small_history = run(small)["history"]

    assert big_history[0]["optimizer_steps"] == small_history[0]["optimizer_steps"] == 4
    for a, b in zip(big_history, small_history):
        assert a["train_loss"] == pytest.approx(b["train_loss"], abs=1e-5)
        assert a["val_accuracy"] == pytest.approx(b["val_accuracy"])


# ============================================================
# TASK 7: RESUME
# ============================================================

def test_resume_continues_exactly(fake_cfg):
    apply_override(fake_cfg, "train.epochs=4")
    straight = run(copy.deepcopy(fake_cfg))["history"]

    interrupted = copy.deepcopy(fake_cfg)
    apply_override(interrupted, "experiment.run_name=interrupted")
    apply_override(interrupted, "train.epochs=2")
    run(interrupted)

    checkpoint = resolve_path(fake_cfg["checkpoint"]["dir"]) / "interrupted" / "epoch_002.pt"
    resumed_cfg = copy.deepcopy(fake_cfg)
    apply_override(resumed_cfg, "experiment.run_name=interrupted")
    apply_override(resumed_cfg, f"checkpoint.resume={checkpoint}")
    resumed = run(resumed_cfg)["history"]

    assert [row["epoch"] for row in resumed] == [1, 2, 3, 4]
    for a, b in zip(straight, resumed):
        assert a["train_loss"] == pytest.approx(b["train_loss"], abs=1e-6)
