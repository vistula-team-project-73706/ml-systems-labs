"""
Lab 07 Task 1: oversized baseline CNN

Trains TeacherCNN (997,434 parameters) on Fashion-MNIST and records test
accuracy, parameters, size and latency. The trained model is the starting
point for pruning (Task 2) and the teacher for distillation (Task 4).

Usage:
    python lab07/src/baseline.py
    python lab07/src/baseline.py --epochs 10

Outputs:
    lab07/checkpoints/teacher.pt          (git-ignored)
    lab07/results/baseline.csv
    lab07/results/baseline_history.csv
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (base_parser, count_macs, dense_size_bytes, evaluate, fit, make_loaders,
                    measure_latency, set_seeds, write_csv)
from models import TeacherCNN, count_parameters, teacher_state


def describe(model, test_loader):
    """Accuracy, size, compute and latency of one model, as one CSV row."""

    test = evaluate(model, test_loader)
    dense_macs, nonzero_macs = count_macs(model)
    latency_1, latency_1_p90 = measure_latency(model, batch_size=1)
    latency_128, _ = measure_latency(model, batch_size=128, runs=30)
    return {
        "parameters": count_parameters(model),
        "size_mb": dense_size_bytes(model) / 1e6,
        "test_accuracy": test["accuracy"],
        "test_loss": test["loss"],
        "macs": dense_macs,
        "latency_ms_batch1": latency_1,
        "latency_ms_batch1_p90": latency_1_p90,
        "throughput_samples_per_s_batch128": 128 / (latency_128 / 1000),
    }


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    set_seeds()
    train_loader, val_loader, test_loader = make_loaders(args)
    model = TeacherCNN()
    print(f"TeacherCNN: {count_parameters(model):,} parameters")

    start = time.perf_counter()
    history = fit(model, train_loader, val_loader, args.epochs, label="baseline")
    train_seconds = time.perf_counter() - start

    args.checkpoints.mkdir(parents=True, exist_ok=True)
    torch.save(teacher_state(model), args.checkpoints / "teacher.pt")

    row = {"model": "baseline_teacher", "epochs": args.epochs,
           "train_samples_seen": args.epochs * len(train_loader.dataset),
           "train_seconds": train_seconds, **describe(model, test_loader)}
    write_csv(args.results / "baseline.csv", [row])
    write_csv(args.results / "baseline_history.csv", history)

    for key, value in row.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
