"""
Lab 06: Profile one training epoch (Task 2)

Each phase of every step is wrapped in torch.profiler.record_function, so
the profiler reports how much time went to:
    data_loading    fetching the next batch from the DataLoader
    forward         model output and loss
    backward        loss.backward()
    optimizer_step  optimizer.step() and zero_grad()

Wall-clock timers around the same phases are recorded as a cross-check.

The profiler records every operation, including each image transform, so a
full-epoch trace is very large (about 2 MB per batch). The breakdown uses the
whole epoch, but the Chrome trace only covers a few batches.

Usage:
    python lab06/src/profile_epoch.py
    python lab06/src/profile_epoch.py --set data.num_workers=2

Outputs:
    lab06/results/profile_breakdown.csv
    lab06/results/profile_top_ops.txt
    lab06/profiles/trace.json   (first few batches, open in chrome://tracing, git-ignored)
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DEFAULT_CONFIG, LAB_DIR, load_config
from data import make_datasets, make_loaders
from model import build_model
from train import Tracker, batches_for, build_optimizer, set_seeds


RESULTS_DIR = LAB_DIR / "results"
PROFILES_DIR = LAB_DIR / "profiles"

PHASES = ["data_loading", "forward", "backward", "optimizer_step"]
TRACE_BATCHES = 5


def profile_one_epoch(cfg):
    set_seeds(cfg["experiment"]["seed"])
    if cfg["experiment"]["torch_threads"]:
        torch.set_num_threads(cfg["experiment"]["torch_threads"])
    device = torch.device(cfg["experiment"]["device"])

    train_set, val_set, test_set = make_datasets(cfg)
    train_loader, _, _, _ = make_loaders(cfg, train_set, val_set, test_set)
    model = build_model(cfg).to(device)
    optimizer = build_optimizer(cfg, model)
    model.train()

    batches = batches_for(train_loader, cfg["data"]["max_train_samples"])

    wall = {phase: 0.0 for phase in PHASES}

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        # Timed inside the profiler block, so the profiler's own start-up and
        # post-processing are not counted as part of the epoch.
        epoch_start = time.perf_counter()
        t0 = time.perf_counter()
        with record_function("data_loading"):
            iterator = iter(train_loader)
        wall["data_loading"] += time.perf_counter() - t0

        for _ in range(batches):
            t0 = time.perf_counter()
            with record_function("data_loading"):
                x, y = next(iterator)
                x, y = x.to(device), y.to(device)
            t1 = time.perf_counter()
            with record_function("forward"):
                loss = F.cross_entropy(model(x), y)
            t2 = time.perf_counter()
            with record_function("backward"):
                loss.backward()
            t3 = time.perf_counter()
            with record_function("optimizer_step"):
                optimizer.step()
                optimizer.zero_grad()
            t4 = time.perf_counter()

            wall["data_loading"] += t1 - t0
            wall["forward"] += t2 - t1
            wall["backward"] += t3 - t2
            wall["optimizer_step"] += t4 - t3

        epoch_seconds = time.perf_counter() - epoch_start
    return prof, wall, epoch_seconds, batches


def phase_breakdown(averages, wall, epoch_seconds):
    """Share of time per phase, from the profiler and from wall-clock timers.

    averages is prof.key_averages(). It is slow on a full epoch, so the
    caller computes it once.
    """

    totals = {event.key: event.cpu_time_total for event in averages}
    profiler_us = {phase: totals.get(phase, 0.0) for phase in PHASES}
    profiler_total = sum(profiler_us.values())
    wall_total = sum(wall.values())

    rows = []
    for phase in PHASES:
        rows.append({
            "phase": phase,
            "profiler_ms": profiler_us[phase] / 1000,
            "profiler_share": profiler_us[phase] / profiler_total,
            "wall_ms": wall[phase] * 1000,
            "wall_share": wall[phase] / wall_total,
        })

    rows.append({
        "phase": "outside_phases",
        "profiler_ms": "",
        "profiler_share": "",
        "wall_ms": (epoch_seconds - wall_total) * 1000,
        "wall_share": "",
    })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Profile one training epoch")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, ["experiment.run_name=profile_one_epoch", *args.set])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    prof, wall, epoch_seconds, batches = profile_one_epoch(cfg)
    averages = prof.key_averages()
    rows = phase_breakdown(averages, wall, epoch_seconds)

    breakdown_path = RESULTS_DIR / "profile_breakdown.csv"
    with open(breakdown_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    top_ops_path = RESULTS_DIR / "profile_top_ops.txt"
    top_ops_path.write_text(averages.table(sort_by="cpu_time_total", row_limit=25))
    del prof, averages

    trace_cfg = load_config(args.config, ["experiment.run_name=profile_one_epoch", *args.set,
                                          f"data.max_train_samples={TRACE_BATCHES * 128}",
                                          "data.batch_size=128"])
    trace_prof, _, _, _ = profile_one_epoch(trace_cfg)
    trace_path = PROFILES_DIR / "trace.json"
    trace_prof.export_chrome_trace(str(trace_path))

    with Tracker(cfg) as tracker:
        tracker.params(cfg)
        tracker.metrics({"epoch_seconds": epoch_seconds, "batches": batches})
        tracker.metrics({f"{row['phase']}_share": row["profiler_share"]
                         for row in rows if row["profiler_share"] != ""})
        tracker.artifact(breakdown_path)
        tracker.artifact(top_ops_path)
        tracker.artifact(trace_path)

    print(f"Profiled {batches} batches in {epoch_seconds:.1f} s")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
