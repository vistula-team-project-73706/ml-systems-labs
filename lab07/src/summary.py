"""
Lab 07 Task 5: one summary table by Chapter 9 efficiency pillar

Reads the CSV files written by baseline.py, prune.py, latency.py and
distill.py and builds one table with five models as columns:

    baseline | global pruned (90%) | structured (50% filters) | plain student | distilled student

and rows grouped by the Chapter 9 pillars: data, model, compute, energy.

Energy is NOT measured: macOS gives no per-process energy counter without
sudo. It is estimated as time x an assumed constant power, so it is only
as good as that assumption. The relative column (vs baseline) does not
depend on the assumed power, only on the measured time.

Usage:
    python lab07/src/summary.py
    python lab07/src/summary.py --pruned-sparsity 0.95 --inference-watts 5 --training-watts 20

Outputs:
    lab07/results/efficiency_summary.csv
    lab07/results/efficiency_summary.md
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import base_parser, read_csv, write_csv


def pick(rows, **match):
    for r in rows:
        if all(str(r[k]) == str(v) for k, v in match.items()):
            return r
    raise KeyError(f"no row with {match}")


def fmt(value, digits=4):
    if isinstance(value, str):
        return value
    if isinstance(value, int) or float(value).is_integer() and abs(value) >= 1000:
        return f"{int(value):,}"
    return f"{value:.{digits}f}"


def build(results, pruned_sparsity, structured_fraction, inference_watts, training_watts):
    baseline = read_csv(results / "baseline.csv")[0]
    unstructured = read_csv(results / "pruning_unstructured.csv")
    structured = read_csv(results / "pruning_structured.csv")
    latency = read_csv(results / "latency_size.csv")
    distill = read_csv(results / "distillation.csv")

    pruned = pick(unstructured, method="global", target_sparsity=pruned_sparsity)
    struct = pick(structured, filter_fraction_removed=structured_fraction)
    lat_base = pick(latency, model="baseline")
    lat_pruned = pick(latency, model=f"unstructured_global_{float(pruned_sparsity):g}")
    lat_struct = pick(latency, model=f"structured_{float(structured_fraction):g}")
    plain = pick(distill, model="student_plain")
    distilled = pick(distill, model="student_distilled")

    base_train = float(baseline["train_seconds"])
    base_samples = int(baseline["train_samples_seen"])
    finetune_samples = 3 * base_samples // int(baseline["epochs"])

    def model_cols(lat):
        best_sparse = min(float(lat[k]) for k in
                          ["sparse_coo_size_mb", "sparse_csr_size_mb", "sparse_bitmask_size_mb"])
        return {
            "parameters": int(lat["parameters"]),
            "nonzero_parameters": int(lat["nonzero_parameters"]),
            "dense_size_mb": float(lat["dense_size_mb"]),
            "best_sparse_size_mb": best_sparse,
            "gzip_size_mb": float(lat["dense_gzip_size_mb"]),
            "dense_macs": int(lat["dense_macs"]),
            "nonzero_macs": int(lat["nonzero_macs"]),
            "latency_ms": float(lat["latency_ms_batch1"]),
            "throughput": float(lat["throughput_samples_per_s_batch128"]),
        }

    columns = {
        "baseline": {**model_cols(lat_base), "accuracy": float(baseline["test_accuracy"]),
                     "samples": base_samples, "needs_teacher": "no", "train_s": base_train},
        f"global pruned {float(pruned_sparsity):.0%}": {
            **model_cols(lat_pruned), "accuracy": float(pruned["test_accuracy_after_finetune"]),
            "samples": base_samples + finetune_samples, "needs_teacher": "yes (prunes it)",
            "train_s": base_train + float(pruned["finetune_seconds"])},
        f"structured {float(structured_fraction):.0%} filters": {
            **model_cols(lat_struct), "accuracy": float(struct["test_accuracy_after_finetune"]),
            "samples": base_samples + finetune_samples, "needs_teacher": "yes (prunes it)",
            "train_s": base_train + float(struct["finetune_seconds"])},
        "plain student": {**model_cols(pick(latency, model="student_plain")), "accuracy": float(plain["test_accuracy"]),
                          "samples": int(plain["train_samples_seen"]), "needs_teacher": "no",
                          "train_s": float(plain["train_seconds"])},
        "distilled student": {**model_cols(pick(latency, model="student_distilled")), "accuracy": float(distilled["test_accuracy"]),
                              "samples": base_samples + int(distilled["train_samples_seen"]),
                              "needs_teacher": "yes (soft labels)",
                              "train_s": base_train + float(distilled["train_seconds"])},
    }

    base_latency = columns["baseline"]["latency_ms"]
    for c in columns.values():
        c["inference_mj"] = c["latency_ms"] * inference_watts  # ms x W = mJ
        c["relative_energy"] = c["latency_ms"] / base_latency
        c["training_kj"] = c["train_s"] * training_watts / 1000

    metrics = [
        ("quality", "test accuracy", "accuracy"),
        ("data", "training samples seen, incl. baseline it depends on", "samples"),
        ("data", "needs the trained baseline", "needs_teacher"),
        ("model", "parameters (stored)", "parameters"),
        ("model", "non-zero parameters", "nonzero_parameters"),
        ("model", "size, dense float32 (MB)", "dense_size_mb"),
        ("model", "size, best sparse format (MB)", "best_sparse_size_mb"),
        ("model", "size, dense + gzip (MB)", "gzip_size_mb"),
        ("compute", "MACs per image, dense kernels", "dense_macs"),
        ("compute", "MACs per image, ideal sparse kernel", "nonzero_macs"),
        ("compute", "latency batch 1, 1 thread (ms, median)", "latency_ms"),
        ("compute", "throughput batch 128, 1 thread (samples/s)", "throughput"),
        ("compute", "training time incl. baseline (s)", "train_s"),
        ("energy", f"inference energy per image (mJ, estimate at {inference_watts:g} W)", "inference_mj"),
        ("energy", "inference energy relative to baseline (= latency ratio)", "relative_energy"),
        ("energy", f"training energy (kJ, estimate at {training_watts:g} W)", "training_kj"),
    ]

    rows = []
    for pillar, label, key in metrics:
        rows.append({"pillar": pillar, "metric": label,
                     **{name: c[key] for name, c in columns.items()}})
    return rows, list(columns)


def to_markdown(rows, names):
    lines = ["| Pillar | Metric | " + " | ".join(names) + " |",
             "|---|---|" + "---:|" * len(names)]
    for r in rows:
        lines.append(f"| {r['pillar']} | {r['metric']} | " + " | ".join(fmt(r[n]) for n in names) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--pruned-sparsity", default="0.9")
    parser.add_argument("--structured-fraction", default="0.5")
    parser.add_argument("--inference-watts", type=float, default=5.0,
                        help="assumed CPU power for 1-thread inference (not measured)")
    parser.add_argument("--training-watts", type=float, default=20.0,
                        help="assumed CPU package power during training (not measured)")
    args = parser.parse_args()

    rows, names = build(args.results, args.pruned_sparsity, args.structured_fraction,
                        args.inference_watts, args.training_watts)
    write_csv(args.results / "efficiency_summary.csv", rows)
    markdown = to_markdown(rows, names)
    (args.results / "efficiency_summary.md").write_text(markdown)
    print(f"wrote {args.results / 'efficiency_summary.md'}\n")
    print(markdown)


if __name__ == "__main__":
    main()
