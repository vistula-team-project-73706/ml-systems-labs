"""
Lab 05: PyTorch framework analysis

Parts:
    1. Inspect the graph of a small model (autograd graph and traced graph)
    2. Eager vs torch.jit.trace vs torch.compile after warm-up
    3. Inspect tensor shape, dtype, strides and contiguity
    4. Measure how non-contiguous views change operation speed

All timing uses one CPU thread, so the numbers show the cost of the execution
mode and not how well each mode splits work across cores.

Outputs:
    lab05/results/pytorch_graph.txt
    lab05/results/eager_vs_compiled_runs.csv
    lab05/results/eager_vs_compiled.csv
    lab05/results/tensor_inspection.csv
    lab05/results/noncontiguous_runs.csv
    lab05/results/noncontiguous_timing.csv
"""

import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_moons import (
    build_torch_model,
    graph_nodes_per_step,
    initial_weights,
    prepare_data,
)


# ============================================================
# PATHS AND EXPERIMENT SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "lab05" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
REPEATS = 5

COMPILE_WARMUP_CALLS = 100
COMPILE_MEASURE_CALLS = 300
EARLY_CALLS = 10

VIEW_SIZES = [256, 2048]
VIEW_WARMUP_CALLS = 5
VIEW_MEASURE_CALLS = 20


# ============================================================
# 1. GRAPH INSPECTION
# ============================================================

def autograd_graph(root):
    """Walk grad_fn back from the loss. Returns printable lines and a count."""

    lines = []
    seen = set()

    def visit(fn, depth):
        indent = "  " * depth
        name = type(fn).__name__

        if fn in seen:
            lines.append(f"{indent}{name} (already shown)")
            return

        seen.add(fn)

        if hasattr(fn, "variable"):
            lines.append(f"{indent}{name} shape={list(fn.variable.shape)} "
                         f"dtype={fn.variable.dtype}")
        else:
            lines.append(f"{indent}{name}")

        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                visit(next_fn, depth + 1)

    visit(root, 0)
    return lines, len(seen)


def inspect_graph():
    X_train, _, y_train, _ = prepare_data()
    X = torch.from_numpy(X_train)
    y = torch.from_numpy(y_train).unsqueeze(1)

    weights = initial_weights(SEED)
    model = build_torch_model(weights)
    loss = nn.BCELoss()(model(X), y)

    lines, node_count = autograd_graph(loss.grad_fn)
    traced = torch.jit.trace(model, X)
    engine_nodes = graph_nodes_per_step(X_train, y_train)

    text = [
        "MODEL",
        str(model),
        "",
        "PARAMETERS",
    ]
    for name, parameter in model.named_parameters():
        text.append(f"{name}: shape={list(parameter.shape)} dtype={parameter.dtype} "
                    f"requires_grad={parameter.requires_grad} "
                    f"contiguous={parameter.is_contiguous()}")

    text += [
        "",
        f"AUTOGRAD GRAPH of the BCE loss, batch of {len(X)} ({node_count} nodes)",
        *lines,
        "",
        "TRACED GRAPH (torch.jit.trace)",
        # The source-location comments hold absolute paths from this machine.
        re.sub(r" # /.*", "", str(traced.inlined_graph)),
        "",
        "GRAPH SIZE FOR ONE TRAINING STEP",
        f"PyTorch autograd nodes:  {node_count}",
        f"Value engine nodes:      {engine_nodes}",
        f"Value nodes per PyTorch node: {engine_nodes / node_count:.1f}",
    ]

    (RESULTS_DIR / "pytorch_graph.txt").write_text("\n".join(text) + "\n")
    print("\n".join(lines))
    print(f"PyTorch nodes: {node_count}  Value nodes: {engine_nodes}")


# ============================================================
# 2. EAGER VS COMPILED
# ============================================================

class LargeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    """The Lab 03 CNN."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(1568, 10)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class OpChain(nn.Module):
    """Many cheap pointwise operations and no large kernel.

    Here the time goes on dispatching operators and allocating temporaries,
    which is the overhead a compiler can remove by fusing the operations.
    """

    def __init__(self, length=100):
        super().__init__()
        self.length = length

    def forward(self, x):
        for _ in range(self.length):
            x = (x * 1.01 + 0.01).relu()
        return x


def tiny_model():
    return build_torch_model(initial_weights(SEED)).float()


WORKLOADS = {
    "tiny_batch_1": (tiny_model, (1, 2)),
    "tiny_batch_240": (tiny_model, (240, 2)),
    "large_mlp_batch_128": (LargeMLP, (128, 784)),
    "cnn_batch_128": (SmallCNN, (128, 1, 28, 28)),
    "op_chain_1024": (OpChain, (1024,)),
}


def prime_compiler():
    """Pay one-time costs before anything is timed.

    The first compile in a process also imports the compiler and runs
    `brew --prefix libomp` once. That has nothing to do with the model.
    """

    compiled = torch.compile(nn.Linear(3, 5))
    with torch.no_grad():
        compiled(torch.rand(4, 3))


def prepare_mode(mode, model, example):
    if mode == "eager":
        return model
    if mode == "jit_trace":
        return torch.jit.trace(model, example)
    if mode == "torch_compile":
        return torch.compile(model)
    raise ValueError(mode)


def timed_calls(model, example, calls):
    times = []

    for _ in range(calls):
        start = time.perf_counter()
        model(example)
        times.append((time.perf_counter() - start) * 1000)

    return times


def measure_mode(workload, mode, repeat):
    build, shape = WORKLOADS[workload]

    torch.manual_seed(SEED)
    model = build().eval()
    example = torch.rand(shape)
    expected = model(example).detach()

    with torch.no_grad():
        # torch.compile is lazy, the real compile happens in the first call.
        start = time.perf_counter()
        prepared = prepare_mode(mode, model, example)
        output = prepared(example)
        setup_ms = (time.perf_counter() - start) * 1000

        warmup = timed_calls(prepared, example, COMPILE_WARMUP_CALLS)
        measured = timed_calls(prepared, example, COMPILE_MEASURE_CALLS)

    return {
        "workload": workload,
        "mode": mode,
        "repeat": repeat,
        "setup_and_first_call_ms": setup_ms,
        "early_calls_median_ms": statistics.median(warmup[:EARLY_CALLS]),
        "after_warmup_median_ms": statistics.median(measured),
        "output_matches_eager": torch.allclose(output, expected, rtol=1e-4, atol=1e-5),
    }


def measure_in_fresh_process(workload, mode, repeat):
    """Run one measurement in a new process with an empty compile cache.

    torch.compile keeps compiled code on disk and in memory, and an identical
    model then "compiles" in a few ms. A new process has neither, so every
    repeat pays the real compile cost. All three modes go through this, so they
    start from the same state.
    """

    with tempfile.TemporaryDirectory() as cache_dir:
        env = {
            **os.environ,
            "TORCHINDUCTOR_CACHE_DIR": cache_dir,
            "HOMEBREW_NO_AUTO_UPDATE": "1",
        }
        result = subprocess.run(
            [sys.executable, __file__, "--measure", workload, mode, str(repeat)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )

    return json.loads(result.stdout.splitlines()[-1])


def measure_worker(workload, mode, repeat):
    torch.set_num_threads(1)

    if mode == "torch_compile":
        prime_compiler()

    print(json.dumps(measure_mode(workload, mode, repeat)))


def compile_summary(rows):
    """One row per workload and mode, medians over the repeats."""

    summary = []

    for workload in dict.fromkeys(row["workload"] for row in rows):
        selected = [row for row in rows if row["workload"] == workload]

        def median_of(mode, field):
            return statistics.median(row[field] for row in selected if row["mode"] == mode)

        eager_ms = median_of("eager", "after_warmup_median_ms")
        eager_setup = median_of("eager", "setup_and_first_call_ms")

        for mode in dict.fromkeys(row["mode"] for row in selected):
            mode_ms = median_of(mode, "after_warmup_median_ms")
            setup = median_of(mode, "setup_and_first_call_ms")
            saved = eager_ms - mode_ms

            summary.append({
                "workload": workload,
                "mode": mode,
                "repeats": sum(row["mode"] == mode for row in selected),
                "setup_and_first_call_ms": setup,
                "setup_min_ms": min(row["setup_and_first_call_ms"] for row in selected
                                    if row["mode"] == mode),
                "setup_max_ms": max(row["setup_and_first_call_ms"] for row in selected
                                    if row["mode"] == mode),
                "early_calls_median_ms": median_of(mode, "early_calls_median_ms"),
                "after_warmup_median_ms": mode_ms,
                "speedup_vs_eager": eager_ms / mode_ms,
                # Calls needed before the time saved pays for the setup.
                "break_even_calls": "" if mode == "eager"
                else (math.ceil((setup - eager_setup) / saved) if saved > 0 else "never"),
                "all_outputs_match_eager": all(
                    row["output_matches_eager"] for row in selected if row["mode"] == mode
                ),
            })

    return summary


def compile_graph_counts():
    """Confirm torch.compile really captured each model as one graph."""

    counts = {}

    for workload, (build, shape) in WORKLOADS.items():
        torch._dynamo.reset()
        torch.manual_seed(SEED)
        explanation = torch._dynamo.explain(build().eval())(torch.rand(shape))
        counts[workload] = (explanation.graph_count, explanation.graph_break_count)

    torch._dynamo.reset()
    return counts


def compare_execution_modes():
    rows = []

    # Modes take turns inside each repeat, so slow drift hits all of them.
    for repeat in range(REPEATS):
        for workload in WORKLOADS:
            for mode in ("eager", "jit_trace", "torch_compile"):
                row = measure_in_fresh_process(workload, mode, repeat)
                rows.append(row)
                print(f"repeat {repeat} {workload:<20} {mode:<14} "
                      f"setup {row['setup_and_first_call_ms']:10.2f} ms  "
                      f"steady {row['after_warmup_median_ms']:9.4f} ms", flush=True)

    summary = compile_summary(rows)
    graph_counts = compile_graph_counts()

    for row in summary:
        graphs, breaks = graph_counts[row["workload"]]
        row["dynamo_graphs"] = graphs if row["mode"] == "torch_compile" else ""
        row["dynamo_graph_breaks"] = breaks if row["mode"] == "torch_compile" else ""

    save_csv(rows, RESULTS_DIR / "eager_vs_compiled_runs.csv")
    save_csv(summary, RESULTS_DIR / "eager_vs_compiled.csv")

    for row in summary:
        print(row)


# ============================================================
# 3. TENSOR INSPECTION
# ============================================================

def describe(name, tensor):
    return {
        "tensor": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "strides": list(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
        "storage_elements": tensor.untyped_storage().nbytes() // tensor.element_size(),
        "elements": tensor.numel(),
        "is_contiguous": tensor.is_contiguous(),
    }


def inspect_tensors():
    X_train, _, _, _ = prepare_data()
    batch = torch.from_numpy(X_train)
    model = build_torch_model(initial_weights(SEED))

    base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    cube = torch.zeros(2, 3, 4)

    rows = [
        describe("moons_batch", batch),
        describe("hidden_weight", model[0].weight.detach()),
        describe("hidden_bias", model[0].bias.detach()),
        describe("base", base),
        describe("base_reshape_3x8", base.reshape(3, 8)),
        describe("base_rows_1_to_2", base[1:3]),
        describe("base_columns_1_to_2", base[:, 1:3]),
        describe("base_every_2nd_row", base[::2]),
        describe("base_every_2nd_column", base[:, ::2]),
        describe("base_transposed", base.t()),
        describe("base_every_2nd_column_contiguous", base[:, ::2].contiguous()),
        describe("cube_permuted_2_0_1", cube.permute(2, 0, 1)),
    ]

    save_csv(rows, RESULTS_DIR / "tensor_inspection.csv")

    for row in rows:
        print(row)

    # base[:, ::2] is not contiguous but it still has one even step through
    # memory, so view(-1) works on it. A transposed tensor has no such step.
    print(f"view(-1) on every 2nd column works: {base[:, ::2].view(-1).shape}")

    try:
        base.t().view(-1)
    except RuntimeError as error:
        print(f"view() on a non-contiguous tensor fails: {str(error).splitlines()[0]}")


# ============================================================
# 4. NON-CONTIGUOUS SPEED
# ============================================================

def make_view_cases(n):
    """Each tensor is n x n float32, only the memory layout differs."""

    return {
        "contiguous": torch.randn(n, n),
        "transposed": torch.randn(n, n).t(),
        "every_2nd_row": torch.randn(2 * n, n)[::2],
        "every_2nd_column": torch.randn(n, 2 * n)[:, ::2],
        "every_8th_column": torch.randn(n, 8 * n)[:, ::8],
    }


VIEW_OPERATIONS = {
    "sum": lambda x, v: x.sum(),
    "add_scalar": lambda x, v: x + 1.0,
    "sum_dim0": lambda x, v: x.sum(dim=0),
    "matvec": lambda x, v: x @ v,
}


def median_call_ms(operation, x, v):
    for _ in range(VIEW_WARMUP_CALLS):
        operation(x, v)

    times = []
    for _ in range(VIEW_MEASURE_CALLS):
        start = time.perf_counter()
        operation(x, v)
        times.append((time.perf_counter() - start) * 1000)

    return statistics.median(times)


def measure_views():
    torch.manual_seed(SEED)
    rows = []

    for repeat in range(REPEATS):
        for n in VIEW_SIZES:
            v = torch.randn(n)
            cases = make_view_cases(n)

            for case, x in cases.items():
                copy_ms = 0.0 if case == "contiguous" else median_call_ms(
                    lambda t, _: t.contiguous(), x, v
                )

                for op_name, operation in VIEW_OPERATIONS.items():
                    rows.append({
                        "n": n,
                        "case": case,
                        "operation": op_name,
                        "repeat": repeat,
                        "strides": list(x.stride()),
                        "is_contiguous": x.is_contiguous(),
                        "median_ms": median_call_ms(operation, x, v),
                        "copy_to_contiguous_ms": copy_ms,
                    })

            del cases
        print(f"view timing repeat {repeat} done", flush=True)

    return rows


def view_summary(rows):
    keys = list(dict.fromkeys((row["n"], row["case"], row["operation"]) for row in rows))
    summary = []

    def median_of(n, case, operation, field):
        return statistics.median(
            row[field] for row in rows
            if (row["n"], row["case"], row["operation"]) == (n, case, operation)
        )

    for n, case, operation in keys:
        first = next(row for row in rows
                     if (row["n"], row["case"], row["operation"]) == (n, case, operation))
        contiguous_ms = median_of(n, "contiguous", operation, "median_ms")
        view_ms = median_of(n, case, operation, "median_ms")
        copy_ms = median_of(n, case, operation, "copy_to_contiguous_ms")
        penalty = view_ms - contiguous_ms

        summary.append({
            "n": n,
            "case": case,
            "operation": operation,
            "strides": first["strides"],
            "is_contiguous": first["is_contiguous"],
            "repeats": sum((row["n"], row["case"], row["operation"]) == (n, case, operation)
                           for row in rows),
            "median_ms": view_ms,
            "ratio_vs_contiguous": view_ms / contiguous_ms,
            "copy_to_contiguous_ms": copy_ms,
            # How many calls before paying once for a copy would have been cheaper.
            "copy_pays_off_after_calls": "" if case == "contiguous"
            else (math.ceil(copy_ms / penalty) if penalty > 0 else "never"),
        })

    return summary


def compare_views():
    rows = measure_views()
    summary = view_summary(rows)

    save_csv(rows, RESULTS_DIR / "noncontiguous_runs.csv")
    save_csv(summary, RESULTS_DIR / "noncontiguous_timing.csv")

    for row in summary:
        print(row)


# ============================================================
# 5. MAIN
# ============================================================

def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    if sys.argv[1:2] == ["--measure"]:
        measure_worker(sys.argv[2], sys.argv[3], int(sys.argv[4]))
        return

    torch.set_num_threads(1)

    print("\n=== 1. Graph inspection ===")
    inspect_graph()

    print("\n=== 3. Tensor inspection ===")
    inspect_tensors()

    print("\n=== 2. Eager vs compiled ===")
    compare_execution_modes()

    print("\n=== 4. Non-contiguous speed ===")
    compare_views()


if __name__ == "__main__":
    main()
