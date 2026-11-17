"""
Lab 06: YAML-driven training pipeline (Task 1)

Usage:
    python lab06/src/train.py
    python lab06/src/train.py --set train.epochs=10 optimizer.name=sgd optimizer.lr=0.05
    python lab06/src/train.py --set checkpoint.resume=lab06/checkpoints/baseline/epoch_003.pt

Every run:
    - is described by one YAML config object (lab06/configs/base.yaml + overrides)
    - seeds random, numpy and torch from the config
    - saves a checkpoint after every epoch (model, optimizer, scheduler, RNG states)
    - can resume from any checkpoint and continues the same MLflow run
    - logs params, per-epoch metrics and artifacts to MLflow

Outputs:
    lab06/checkpoints/<run_name>/epoch_NNN.pt and last.pt   (git-ignored)
    lab06/mlruns/                                          (git-ignored)
"""

import argparse
import csv
import json
import math
import os
import random
import resource
import sys
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DEFAULT_CONFIG, flatten, load_config, resolve_path
from data import make_datasets, make_loaders
from model import build_model


# ============================================================
# 1. SETUP
# ============================================================

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_optimizer(cfg, model):
    opt = cfg["optimizer"]

    if opt["name"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=opt["lr"], weight_decay=opt["weight_decay"])
    if opt["name"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=opt["lr"], momentum=opt["momentum"],
                               weight_decay=opt["weight_decay"])
    raise ValueError(f"Unknown optimizer {opt['name']!r}")


def build_scheduler(cfg, optimizer):
    sched = cfg["scheduler"]

    if sched["name"] == "none":
        return None
    if sched["name"] == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=sched["step_size"],
                                               gamma=sched["gamma"])
    if sched["name"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=sched["t_max"])
    if sched["name"] == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=sched["plateau_factor"],
                                                          patience=sched["plateau_patience"])
    raise ValueError(f"Unknown scheduler {sched['name']!r}")


def peak_memory_mb():
    """Highest resident memory of this process so far.

    ru_maxrss is in bytes on macOS and in kilobytes on Linux.
    """

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


# ============================================================
# 2. ONE EPOCH
# ============================================================

def batches_for(loader, max_samples):
    """Number of batches to train on. A sample limit keeps runs with different
    batch sizes or accumulation settings on the same amount of data."""

    if max_samples is None:
        return len(loader)
    return min(len(loader), math.ceil(max_samples / loader.batch_size))


def train_one_epoch(model, loader, optimizer, accumulation, max_samples, device):
    """Train for one epoch with optional gradient accumulation.

    Each micro-batch loss is divided by the number of samples in its whole
    accumulation group. The summed gradients then equal the gradient of one
    big batch of the same samples, including the smaller last group.
    """

    model.train()
    batches = batches_for(loader, max_samples)
    total_samples = min(len(loader.dataset), batches * loader.batch_size)
    group_size = loader.batch_size * accumulation

    loss_sum = 0.0
    correct = 0
    seen = 0
    steps = 0

    optimizer.zero_grad()
    start = time.perf_counter()

    for i, (x, y) in enumerate(loader):
        if i == batches:
            break

        x, y = x.to(device), y.to(device)
        output = model(x)
        batch_loss = F.cross_entropy(output, y, reduction="sum")

        group_samples = min(group_size, total_samples - (i // accumulation) * group_size)
        (batch_loss / group_samples).backward()

        if (i + 1) % accumulation == 0 or i + 1 == batches:
            optimizer.step()
            optimizer.zero_grad()
            steps += 1

        loss_sum += batch_loss.item()
        correct += (output.argmax(dim=1) == y).sum().item()
        seen += len(y)

    seconds = time.perf_counter() - start

    return {
        "train_loss": loss_sum / seen,
        "train_accuracy": correct / seen,
        "train_samples": seen,
        "train_seconds": seconds,
        "throughput_samples_per_s": seen / seconds,
        "optimizer_steps": steps,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    seen = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss_sum += F.cross_entropy(output, y, reduction="sum").item()
        correct += (output.argmax(dim=1) == y).sum().item()
        seen += len(y)

    return loss_sum / seen, correct / seen


# ============================================================
# 3. CHECKPOINTS
# ============================================================

def save_checkpoint(path, state):
    # Write to a temporary file first. If the process is killed while saving,
    # the previous checkpoint is still complete.
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def checkpoint_state(epoch, model, optimizer, scheduler, generator, history, run_id, cfg):
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "history": history,
        "mlflow_run_id": run_id,
        "config": cfg,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "loader": generator.get_state(),
        },
    }


def restore_checkpoint(path, model, optimizer, scheduler, generator):
    state = torch.load(path, map_location="cpu")

    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])

    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"])
    generator.set_state(state["rng"]["loader"])

    return state


# ============================================================
# 4. MLFLOW
# ============================================================

class Tracker:
    """Thin wrapper so the training loop also runs with MLflow switched off."""

    def __init__(self, cfg, run_id=None):
        self.cfg = cfg
        self.enabled = cfg["mlflow"]["enabled"]
        self.resume_run_id = run_id
        self.run_id = None

    def __enter__(self):
        if self.enabled:
            mlflow.set_tracking_uri(resolve_path(self.cfg["mlflow"]["tracking_uri"]).as_uri())
            mlflow.set_experiment(self.cfg["experiment"]["mlflow_experiment"])
            run = mlflow.start_run(run_id=self.resume_run_id,
                                   run_name=None if self.resume_run_id else self.cfg["experiment"]["run_name"])
            self.run_id = run.info.run_id
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.enabled:
            mlflow.end_run(status="FINISHED" if exc_type is None else "FAILED")
        return False

    def params(self, cfg):
        if self.enabled:
            mlflow.log_params(flatten(cfg))
            mlflow.log_dict(cfg, "config.yaml")

    def metrics(self, values, step=None):
        if self.enabled:
            mlflow.log_metrics(values, step=step)

    def artifact(self, path):
        if self.enabled:
            mlflow.log_artifact(str(path))


def write_history(history, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


# ============================================================
# 5. FULL RUN
# ============================================================

def run(cfg):
    exp = cfg["experiment"]
    set_seeds(exp["seed"])
    if exp["torch_threads"]:
        torch.set_num_threads(exp["torch_threads"])
    device = torch.device(exp["device"])

    train_set, val_set, test_set = make_datasets(cfg)
    train_loader, val_loader, test_loader, generator = make_loaders(cfg, train_set, val_set, test_set)
    memory_after_data_mb = peak_memory_mb()

    model = build_model(cfg).to(device)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    checkpoint_dir = resolve_path(cfg["checkpoint"]["dir"]) / exp["run_name"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = []
    start_epoch = 1
    run_id = None

    if cfg["checkpoint"]["resume"]:
        state = restore_checkpoint(resolve_path(cfg["checkpoint"]["resume"]),
                                   model, optimizer, scheduler, generator)
        history = state["history"]
        start_epoch = state["epoch"] + 1
        run_id = state["mlflow_run_id"]
        print(f"Resumed from epoch {state['epoch']}")
    else:
        # Old checkpoints from an earlier run with the same name would mix in.
        for old in checkpoint_dir.glob("epoch_*.pt"):
            old.unlink()

    with Tracker(cfg, run_id) as tracker:
        if start_epoch == 1:
            tracker.params(cfg)

        for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
            stats = train_one_epoch(model, train_loader, optimizer, cfg["train"]["accumulation_steps"],
                                    cfg["data"]["max_train_samples"], device)
            val_loss, val_accuracy = evaluate(model, val_loader, device)
            lr = optimizer.param_groups[0]["lr"]

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            elif scheduler is not None:
                scheduler.step()

            row = {"epoch": epoch, **stats, "val_loss": val_loss, "val_accuracy": val_accuracy, "lr": lr}
            history.append(row)
            tracker.metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)

            state = checkpoint_state(epoch, model, optimizer, scheduler, generator, history,
                                     tracker.run_id, cfg)
            save_checkpoint(checkpoint_dir / f"epoch_{epoch:03d}.pt", state)
            save_checkpoint(checkpoint_dir / "last.pt", state)

            print(f"epoch {epoch:3d}  train_loss {stats['train_loss']:.4f}  "
                  f"val_acc {val_accuracy:.4f}  lr {lr:.2e}  "
                  f"{stats['throughput_samples_per_s']:.0f} samples/s", flush=True)

        test_loss, test_accuracy = evaluate(model, test_loader, device)
        train_seconds = sum(row["train_seconds"] for row in history)
        train_samples = sum(row["train_samples"] for row in history)

        summary = {
            "run_name": exp["run_name"],
            "mlflow_run_id": tracker.run_id,
            "epochs": cfg["train"]["epochs"],
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "final_val_accuracy": history[-1]["val_accuracy"],
            "best_val_accuracy": max(row["val_accuracy"] for row in history),
            "throughput_samples_per_s": train_samples / train_seconds,
            "memory_after_data_mb": memory_after_data_mb,
            "peak_memory_mb": peak_memory_mb(),
            "history": history,
        }

        tracker.metrics({k: v for k, v in summary.items()
                         if isinstance(v, (int, float)) and k != "epochs"})

        with tempfile.TemporaryDirectory() as tmp:
            history_path = Path(tmp) / "history.csv"
            write_history(history, history_path)
            tracker.artifact(history_path)
        tracker.artifact(checkpoint_dir / "last.pt")

    return summary


# ============================================================
# 6. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Lab 06 training pipeline")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. train.epochs=10")
    parser.add_argument("--summary", help="write the run summary to this JSON file")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    print(yaml.safe_dump(cfg, sort_keys=False))

    summary = run(cfg)

    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, indent=2))

    print(f"test accuracy {summary['test_accuracy']:.4f}  "
          f"throughput {summary['throughput_samples_per_s']:.0f} samples/s  "
          f"peak memory {summary['peak_memory_mb']:.0f} MB")


if __name__ == "__main__":
    main()
