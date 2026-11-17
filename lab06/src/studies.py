"""
Lab 06: Training studies (Tasks 3 to 7)

Every configuration runs lab06/src/train.py in a new process, so peak memory
is measured per configuration and nothing is shared between runs. Every run
is also an MLflow run.

Usage:
    python lab06/src/studies.py workers        # Task 3
    python lab06/src/studies.py batch          # Task 4
    python lab06/src/studies.py schedules      # Task 5
    python lab06/src/studies.py accumulation   # Task 6
    python lab06/src/studies.py resume         # Task 7
    python lab06/src/studies.py export         # MLflow runs -> CSV
    python lab06/src/studies.py all

Add --set KEY=VALUE to change the base config for a study, e.g.
    python lab06/src/studies.py batch --set data.max_train_samples=2560

Outputs (lab06/results/):
    workers_study.csv, workers_study.png
    batch_study.csv, batch_study.png
    schedule_study.csv, schedule_curves.csv, schedule_study.png
    accumulation_study.csv, accumulation_curves.csv, accumulation_study.png
    resume_study.csv, resume_study.png
    mlflow_runs.csv
"""

import argparse
import csv
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DEFAULT_CONFIG, LAB_DIR, load_config, resolve_path
from data import make_datasets, make_loaders


# ============================================================
# PATHS AND STUDY SETTINGS
# ============================================================

RESULTS_DIR = LAB_DIR / "results"
TRAIN_SCRIPT = LAB_DIR / "src" / "train.py"

WORKER_COUNTS = [0, 1, 2, 4, 8]
BATCH_SIZES = [32, 64, 128, 256, 512]
BATCH_EPOCHS = 5
SCHEDULE_EPOCHS = 10
OPTIMIZERS = {
    "sgd": ["optimizer.name=sgd", "optimizer.lr=0.05", "optimizer.momentum=0.9"],
    "adam": ["optimizer.name=adam", "optimizer.lr=0.001"],
}
SCHEDULERS = ["none", "step", "cosine", "plateau"]
ACCUMULATION_EPOCHS = 5
RESUME_EPOCHS = 6
INTERRUPT_AFTER_EPOCH = 3

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


# ============================================================
# 1. HELPERS
# ============================================================

def run_training(base_overrides, overrides):
    """Run train.py in a new process and return its summary."""

    with tempfile.TemporaryDirectory() as tmp:
        summary_path = Path(tmp) / "summary.json"
        command = [sys.executable, str(TRAIN_SCRIPT), "--config", str(DEFAULT_CONFIG),
                   "--set", *base_overrides, *overrides, "--summary", str(summary_path)]
        print("\n$ " + " ".join(command[1:]), flush=True)
        subprocess.run(command, check=True)
        return json.loads(summary_path.read_text())


def save_csv(rows, name):
    path = RESULTS_DIR / name
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {path}")


def style(axes):
    axes.spines[["top", "right"]].set_visible(False)
    axes.grid(axis="y", color="#e6e5e0", linewidth=0.8)


def save_figure(figure, name):
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / name, dpi=150)
    plt.close(figure)
    print(f"saved {RESULTS_DIR / name}")


def history_rows(study, label, summary):
    return [{"study": study, "run": label, **row} for row in summary["history"]]


# ============================================================
# 2. TASK 3: NUM_WORKERS
# ============================================================

def measure_loader(cfg):
    """Time the training DataLoader alone, with no model.

    Startup is the time until the first batch arrives, which includes
    starting the worker processes. Throughput is measured after that.
    """

    train_set, val_set, test_set = make_datasets(cfg)
    loader, _, _, _ = make_loaders(cfg, train_set, val_set, test_set)

    start = time.perf_counter()
    iterator = iter(loader)
    first_x, _ = next(iterator)
    startup = time.perf_counter() - start

    seen = 0
    start = time.perf_counter()
    for x, _ in iterator:
        seen += len(x)
    seconds = time.perf_counter() - start

    return startup, seen / seconds


def workers_study(base):
    rows = []

    for workers in WORKER_COUNTS:
        # Two epochs: the first includes starting the worker processes, the
        # second shows the steady state, because the workers are kept alive.
        overrides = [f"data.num_workers={workers}", "train.epochs=2",
                     f"experiment.run_name=workers_{workers}"]

        cfg = load_config(DEFAULT_CONFIG, [*base, *overrides])
        startup, loader_throughput = measure_loader(cfg)
        summary = run_training(base, overrides)

        rows.append({
            "num_workers": workers,
            "loader_only_samples_per_s": loader_throughput,
            "loader_startup_s": startup,
            "training_samples_per_s_epoch_1": summary["history"][0]["throughput_samples_per_s"],
            "training_samples_per_s": summary["history"][1]["throughput_samples_per_s"],
            "epoch_2_train_seconds": summary["history"][1]["train_seconds"],
            "loader_vs_training_ratio": loader_throughput / summary["history"][1]["throughput_samples_per_s"],
        })

    save_csv(rows, "workers_study.csv")

    figure, axes = plt.subplots(figsize=(6.5, 4))
    x = [row["num_workers"] for row in rows]
    axes.plot(x, [row["loader_only_samples_per_s"] for row in rows], marker="o",
              color=COLORS[0], linewidth=2, label="DataLoader alone")
    axes.plot(x, [row["training_samples_per_s"] for row in rows], marker="o",
              color=COLORS[1], linewidth=2, label="Training, epoch 2")
    axes.set_xticks(x)
    axes.set_xlabel("num_workers")
    axes.set_ylabel("Samples per second")
    axes.set_title("Input pipeline vs training throughput", loc="left", fontsize=11)
    axes.legend(frameon=False)
    style(axes)
    save_figure(figure, "workers_study.png")


# ============================================================
# 3. TASK 4: BATCH SIZE
# ============================================================

def batch_study(base):
    rows = []

    for batch_size in BATCH_SIZES:
        summary = run_training(base, [f"data.batch_size={batch_size}",
                                      f"train.epochs={BATCH_EPOCHS}",
                                      f"experiment.run_name=batch_{batch_size}"])
        rows.append({
            "batch_size": batch_size,
            "epochs": BATCH_EPOCHS,
            "throughput_samples_per_s": summary["throughput_samples_per_s"],
            "memory_after_data_mb": summary["memory_after_data_mb"],
            "peak_memory_mb": summary["peak_memory_mb"],
            "final_val_accuracy": summary["final_val_accuracy"],
            "test_accuracy": summary["test_accuracy"],
        })

    save_csv(rows, "batch_study.csv")

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    x = [row["batch_size"] for row in rows]
    panels = [
        ("throughput_samples_per_s", "Samples per second", "Throughput"),
        ("peak_memory_mb", "MB", "Peak process memory"),
        ("test_accuracy", "Accuracy", f"Test accuracy after {BATCH_EPOCHS} epochs"),
    ]
    for ax, (key, label, title) in zip(axes, panels):
        ax.plot(x, [row[key] for row in rows], marker="o", color=COLORS[0], linewidth=2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(x, [str(v) for v in x])
        ax.set_xlabel("Batch size")
        ax.set_ylabel(label)
        ax.set_title(title, loc="left", fontsize=11)
        style(ax)
    save_figure(figure, "batch_study.png")


# ============================================================
# 4. TASK 5: SCHEDULES AND OPTIMIZERS
# ============================================================

def schedules_study(base):
    rows = []
    curves = []

    for optimizer, optimizer_overrides in OPTIMIZERS.items():
        for scheduler in SCHEDULERS:
            label = f"{optimizer}_{scheduler}"
            summary = run_training(base, [*optimizer_overrides,
                                          f"scheduler.name={scheduler}",
                                          f"scheduler.t_max={SCHEDULE_EPOCHS}",
                                          f"train.epochs={SCHEDULE_EPOCHS}",
                                          f"experiment.run_name=schedule_{label}"])
            rows.append({
                "optimizer": optimizer,
                "scheduler": scheduler,
                "epochs": SCHEDULE_EPOCHS,
                "final_train_loss": summary["history"][-1]["train_loss"],
                "final_val_accuracy": summary["final_val_accuracy"],
                "best_val_accuracy": summary["best_val_accuracy"],
                "test_accuracy": summary["test_accuracy"],
                "final_lr": summary["history"][-1]["lr"],
            })
            curves += history_rows("schedules", label, summary)

    save_csv(rows, "schedule_study.csv")
    save_csv(curves, "schedule_curves.csv")

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for column, optimizer in enumerate(OPTIMIZERS):
        for color, scheduler in zip(COLORS, SCHEDULERS):
            run = [row for row in curves if row["run"] == f"{optimizer}_{scheduler}"]
            epochs = [row["epoch"] for row in run]
            # Plateau equals "none" until validation loss stalls, so it is
            # dashed to keep both lines visible when they overlap.
            linestyle = (0, (4, 4)) if scheduler == "plateau" else "-"
            axes[0, column].plot(epochs, [row["val_accuracy"] for row in run],
                                 color=color, linewidth=2, linestyle=linestyle, label=scheduler)
            axes[1, column].plot(epochs, [row["lr"] for row in run],
                                 color=color, linewidth=2, linestyle=linestyle, label=scheduler)
        axes[0, column].set_title(f"{optimizer.upper()}: validation accuracy", loc="left", fontsize=11)
        axes[1, column].set_title(f"{optimizer.upper()}: learning rate", loc="left", fontsize=11)
        axes[1, column].set_yscale("log")
        axes[1, column].set_xlabel("Epoch")
        axes[0, column].legend(frameon=False)
        style(axes[0, column])
        style(axes[1, column])
    save_figure(figure, "schedule_study.png")


# ============================================================
# 5. TASK 6: GRADIENT ACCUMULATION
# ============================================================

ACCUMULATION_RUNS = {
    "batch_256": ["data.batch_size=256", "train.accumulation_steps=1"],
    "batch_64_accumulate_4": ["data.batch_size=64", "train.accumulation_steps=4"],
    "batch_64_no_accumulation": ["data.batch_size=64", "train.accumulation_steps=1"],
}


def accumulation_study(base):
    summaries = {}
    curves = []

    for label, overrides in ACCUMULATION_RUNS.items():
        summaries[label] = run_training(base, [*overrides, f"train.epochs={ACCUMULATION_EPOCHS}",
                                               f"experiment.run_name=accumulation_{label}"])
        curves += history_rows("accumulation", label, summaries[label])

    reference = summaries["batch_256"]["history"]
    rows = []
    for label, summary in summaries.items():
        history = summary["history"]
        rows.append({
            "run": label,
            "optimizer_steps_per_epoch": history[0]["optimizer_steps"],
            "final_train_loss": history[-1]["train_loss"],
            "test_accuracy": summary["test_accuracy"],
            "throughput_samples_per_s": summary["throughput_samples_per_s"],
            "max_abs_train_loss_diff_vs_batch_256": max(
                abs(a["train_loss"] - b["train_loss"]) for a, b in zip(history, reference)
            ),
        })

    save_csv(rows, "accumulation_study.csv")
    save_csv(curves, "accumulation_curves.csv")

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for color, label in zip(COLORS, ACCUMULATION_RUNS):
        run = [row for row in curves if row["run"] == label]
        epochs = [row["epoch"] for row in run]
        linestyle = (0, (4, 4)) if label == "batch_64_accumulate_4" else "-"
        axes[0].plot(epochs, [row["train_loss"] for row in run], color=color,
                     linewidth=2, linestyle=linestyle, label=label)
        axes[1].plot(epochs, [row["val_accuracy"] for row in run], color=color,
                     linewidth=2, linestyle=linestyle, label=label)
    axes[0].set_title("Training loss", loc="left", fontsize=11)
    axes[1].set_title("Validation accuracy", loc="left", fontsize=11)
    for ax in axes:
        ax.set_xlabel("Epoch")
        style(ax)
    axes[0].legend(frameon=False)
    save_figure(figure, "accumulation_study.png")


# ============================================================
# 6. TASK 7: INTERRUPT AND RESUME
# ============================================================

def resume_study(base):
    baseline = run_training(base, [f"train.epochs={RESUME_EPOCHS}", "experiment.run_name=resume_baseline"])

    # Start a second, identical run and kill it once the epoch 3 checkpoint
    # exists. SIGKILL gives the program no chance to clean up, like a crash.
    run_name = "resume_interrupted"
    cfg = load_config(DEFAULT_CONFIG, base)
    checkpoint_dir = resolve_path(cfg["checkpoint"]["dir"]) / run_name
    for old in checkpoint_dir.glob("*.pt"):
        old.unlink()
    target = checkpoint_dir / f"epoch_{INTERRUPT_AFTER_EPOCH:03d}.pt"

    command = [sys.executable, str(TRAIN_SCRIPT), "--config", str(DEFAULT_CONFIG), "--set",
               *base, f"train.epochs={RESUME_EPOCHS}", f"experiment.run_name={run_name}"]
    print("\n$ " + " ".join(command[1:]), flush=True)
    process = subprocess.Popen(command)
    while not target.exists():
        if process.poll() is not None:
            raise RuntimeError("Training ended before the interruption point")
        time.sleep(0.2)
    process.kill()
    process.wait()
    print(f"Killed training after {target.name} was written")

    resumed = run_training(base, [f"train.epochs={RESUME_EPOCHS}", f"experiment.run_name={run_name}",
                                  f"checkpoint.resume={target}"])

    rows = []
    for a, b in zip(baseline["history"], resumed["history"]):
        rows.append({
            "epoch": a["epoch"],
            "segment": "before interruption" if a["epoch"] <= INTERRUPT_AFTER_EPOCH else "after resume",
            "baseline_train_loss": a["train_loss"],
            "resumed_train_loss": b["train_loss"],
            "baseline_val_accuracy": a["val_accuracy"],
            "resumed_val_accuracy": b["val_accuracy"],
            "abs_train_loss_diff": abs(a["train_loss"] - b["train_loss"]),
        })
    save_csv(rows, "resume_study.csv")

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    epochs = [row["epoch"] for row in rows]
    for ax, key, title in [(axes[0], "train_loss", "Training loss"),
                           (axes[1], "val_accuracy", "Validation accuracy")]:
        ax.plot(epochs, [row[f"baseline_{key}"] for row in rows], color=COLORS[0],
                linewidth=2, label="Uninterrupted")
        ax.plot(epochs, [row[f"resumed_{key}"] for row in rows], color=COLORS[1],
                linewidth=2, linestyle=(0, (4, 4)), marker="o", label="Killed at epoch 3, resumed")
        ax.axvline(INTERRUPT_AFTER_EPOCH, color="#8a8984", linewidth=1, linestyle=":")
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("Epoch")
        style(ax)
    axes[0].legend(frameon=False)
    save_figure(figure, "resume_study.png")

    print(f"Largest train-loss difference: {max(row['abs_train_loss_diff'] for row in rows):.3e}")


# ============================================================
# 7. EXPORT MLFLOW RUNS
# ============================================================

def export_runs(base):
    cfg = load_config(DEFAULT_CONFIG, base)
    mlflow.set_tracking_uri(resolve_path(cfg["mlflow"]["tracking_uri"]).as_uri())
    runs = mlflow.search_runs(experiment_names=[cfg["experiment"]["mlflow_experiment"]])

    keep = ["tags.mlflow.runName", "status", "start_time", "params.data.batch_size",
            "params.data.num_workers", "params.optimizer.name", "params.optimizer.lr",
            "params.scheduler.name", "params.train.epochs", "params.train.accumulation_steps",
            "metrics.test_accuracy", "metrics.best_val_accuracy",
            "metrics.throughput_samples_per_s", "metrics.peak_memory_mb"]
    columns = [column for column in keep if column in runs.columns]
    runs[columns].sort_values("start_time").to_csv(RESULTS_DIR / "mlflow_runs.csv", index=False)
    print(f"saved {RESULTS_DIR / 'mlflow_runs.csv'} ({len(runs)} runs)")


# ============================================================
# 8. MAIN
# ============================================================

STUDIES = {
    "workers": workers_study,
    "batch": batch_study,
    "schedules": schedules_study,
    "accumulation": accumulation_study,
    "resume": resume_study,
    "export": export_runs,
}


def main():
    parser = argparse.ArgumentParser(description="Lab 06 training studies")
    parser.add_argument("study", choices=[*STUDIES, "all"])
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override the base config for every run in the study")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    selected = list(STUDIES) if args.study == "all" else [args.study]

    for name in selected:
        print(f"\n=== {name} ===", flush=True)
        STUDIES[name](args.set)


if __name__ == "__main__":
    main()
