"""
Lab 07 Task 3: latency and on-disk size, dense vs sparse

For the baseline and every pruned model from prune.py this measures:
    - latency at batch 1 (median and p90 of 200 calls, 1 CPU thread)
    - throughput at batch 128
    - file size as a normal dense state_dict, as sparse COO, as sparse CSR,
      as non-zero values + a bitmask, and the dense file compressed with gzip
    - dense MACs (what the CPU really does) and non-zero MACs (what a
      perfect sparse kernel would do)

It also times one layer (fc1, 6,272 -> 144, 90% of all weights) as a dense
matrix multiply against torch.sparse.mm with a CSR matrix at each sparsity.
That shows directly whether sparse kernels are faster on this CPU.

Run it after distill.py, so the two students are measured too.

Usage:
    python lab07/src/latency.py

Outputs:
    lab07/results/latency_size.csv
    lab07/results/sparse_kernel_benchmark.csv
    lab07/results/latency_size.png
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (LATENCY_THREADS, base_parser, count_macs, count_nonzero_parameters,
                    dense_size_bytes, gzip_size_bytes, measure_latency, sparse_size_bytes,
                    time_calls, weight_sparsity, write_csv)
from models import StudentCNN, build_teacher_from_state, count_parameters
from prune import SPARSITIES

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")


def model_files(checkpoints):
    """(name, kind, path) for the baseline, every pruned model and the students.

    All models are measured here, in one process, so their latency and
    throughput are comparable. The students exist only after distill.py.
    """

    files = [("baseline", "baseline", checkpoints / "teacher.pt")]
    for method in ["global", "layer"]:
        for path in sorted((checkpoints / "pruned").glob(f"unstructured_{method}_*.pt"),
                           key=lambda p: float(p.stem.split("_")[2])):
            files.append((path.stem, f"unstructured_{method}", path))
    for path in sorted((checkpoints / "pruned").glob("structured_*.pt"),
                       key=lambda p: float(p.stem.split("_")[1])):
        files.append((path.stem, "structured", path))
    for name in ["student_plain", "student_distilled"]:
        if (checkpoints / f"{name}.pt").exists():
            files.append((name, "student", checkpoints / f"{name}.pt"))
    return files


def load_model(kind, path):
    if kind == "student":
        model = StudentCNN()
        model.load_state_dict(torch.load(path))
        return model
    return build_teacher_from_state(torch.load(path))


def measure_model(name, kind, model):
    dense_macs, nonzero_macs = count_macs(model)
    latency_1, latency_1_p90 = measure_latency(model, batch_size=1)
    latency_128, _ = measure_latency(model, batch_size=128, runs=30)
    row = {
        "model": name,
        "kind": kind,
        "weight_sparsity": weight_sparsity(model),
        "parameters": count_parameters(model),
        "nonzero_parameters": count_nonzero_parameters(model),
        "dense_size_mb": dense_size_bytes(model) / 1e6,
        "sparse_coo_size_mb": sparse_size_bytes(model, "coo") / 1e6,
        "sparse_csr_size_mb": sparse_size_bytes(model, "csr") / 1e6,
        "sparse_bitmask_size_mb": sparse_size_bytes(model, "bitmask") / 1e6,
        "dense_gzip_size_mb": gzip_size_bytes(model) / 1e6,
        "dense_macs": dense_macs,
        "nonzero_macs": nonzero_macs,
        "latency_ms_batch1": latency_1,
        "latency_ms_batch1_p90": latency_1_p90,
        "throughput_samples_per_s_batch128": 128 / (latency_128 / 1000),
    }
    print(f"  {name:28s} sparsity {row['weight_sparsity']:.2f}  {latency_1:.3f} ms  "
          f"dense {row['dense_size_mb']:.2f} MB  csr {row['sparse_csr_size_mb']:.2f} MB", flush=True)
    return row


def sparse_kernel_benchmark(teacher):
    """Dense vs CSR matrix multiply for fc1 at each sparsity, 1 thread."""

    weight = teacher.fc1.weight.detach().clone()
    rows = []
    previous = torch.get_num_threads()
    torch.set_num_threads(LATENCY_THREADS)
    try:
        for sparsity in SPARSITIES:
            w = weight.clone()
            k = int(sparsity * w.numel())
            if k > 0:
                threshold = w.abs().flatten().kthvalue(k).values
                w[w.abs() <= threshold] = 0
            csr = w.to_sparse_csr()
            for batch in [1, 128]:
                x = torch.randn(w.shape[1], batch)
                with torch.inference_mode():
                    dense_ms, _ = time_calls(lambda: w @ x, runs=300)
                    sparse_ms, _ = time_calls(lambda: torch.sparse.mm(csr, x), runs=300)
                rows.append({
                    "layer": "fc1",
                    "shape": f"{w.shape[0]}x{w.shape[1]}",
                    "sparsity": (w == 0).float().mean().item(),
                    "batch": batch,
                    "dense_ms": dense_ms,
                    "sparse_csr_ms": sparse_ms,
                    "sparse_speedup": dense_ms / sparse_ms,
                })
                print(f"  fc1 sparsity {sparsity:.2f} batch {batch:3d}: dense {dense_ms:.4f} ms, "
                      f"csr {sparse_ms:.4f} ms", flush=True)
    finally:
        torch.set_num_threads(previous)
    return rows


def plot(rows, kernel_rows, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    for kind, color in [("unstructured_global", "tab:blue"), ("unstructured_layer", "tab:orange")]:
        sub = [r for r in rows if r["kind"] == kind]
        ax.plot([r["weight_sparsity"] for r in sub], [r["latency_ms_batch1"] for r in sub], "o-",
                color=color, label=kind.replace("_", " "))
    sub = [r for r in rows if r["kind"] == "structured"]
    base_params = next(r["parameters"] for r in rows if r["kind"] == "baseline")
    ax.plot([1 - r["parameters"] / base_params for r in sub], [r["latency_ms_batch1"] for r in sub], "s-",
            color="tab:green", label="structured (x = parameters removed)")
    for r, style in zip([r for r in rows if r["kind"] == "student"], ["--", "-."]):
        ax.axhline(r["latency_ms_batch1"], color="tab:purple", linestyle=style, label=r["model"])
    ax.set_xlabel("sparsity / parameters removed")
    ax.set_ylabel("latency, batch 1 (ms, median)")
    ax.set_title("Latency")
    ax.set_ylim(bottom=0)

    ax = axes[1]
    sub = [r for r in rows if r["kind"] == "unstructured_global"]
    x = [r["weight_sparsity"] for r in sub]
    for key, label in [("dense_size_mb", "dense .pt"), ("sparse_coo_size_mb", "sparse COO"),
                       ("sparse_csr_size_mb", "sparse CSR"),
                       ("sparse_bitmask_size_mb", "values + bitmask"), ("dense_gzip_size_mb", "dense + gzip")]:
        ax.plot(x, [r[key] for r in sub], "o-", label=label)
    ax.set_xlabel("weight sparsity (global pruning)")
    ax.set_ylabel("file size (MB)")
    ax.set_title("On-disk size")

    ax = axes[2]
    for batch, marker in [(1, "o"), (128, "s")]:
        sub = [r for r in kernel_rows if r["batch"] == batch]
        ax.plot([r["sparsity"] for r in sub], [r["sparse_speedup"] for r in sub], marker + "-",
                label=f"batch {batch}")
    ax.axhline(1.0, color="grey", linestyle=":")
    ax.set_yscale("log")
    ax.set_xlabel("fc1 sparsity")
    ax.set_ylabel("dense time / CSR time  (>1 = sparse faster)")
    ax.set_title("fc1: dense vs sparse CSR kernel")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"wrote {path}")


def main():
    args = base_parser(__doc__).parse_args()

    files = model_files(args.checkpoints)

    # Discarded warm-up: the first model measured in a process came out
    # about 7% slower than an identical copy measured right after it.
    warm = load_model(files[0][1], files[0][2])
    measure_latency(warm, batch_size=1)
    measure_latency(warm, batch_size=128, runs=30)

    rows = []
    for name, kind, path in files:
        model = load_model(kind, path)
        rows.append(measure_model(name, kind, model))
    write_csv(args.results / "latency_size.csv", rows)

    teacher = build_teacher_from_state(torch.load(args.checkpoints / "teacher.pt"))
    kernel_rows = sparse_kernel_benchmark(teacher)
    write_csv(args.results / "sparse_kernel_benchmark.csv", kernel_rows)

    plot(rows, kernel_rows, args.results / "latency_size.png")


if __name__ == "__main__":
    main()
