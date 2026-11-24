"""
Lab 07 Task 2: magnitude pruning

Unstructured pruning: rank weights by absolute value and zero the smallest
ones at sparsity [0, 0.25, 0.5, 0.75, 0.9, 0.95], once with one global
threshold over all layers and once with the same fraction in every layer.
Then fine-tune 3 epochs with the mask fixed.

Structured pruning: remove whole conv filters with the smallest L2 norm at
[10%, 25%, 50%] of the filters in every conv layer (0% is the reference).
The layers are rebuilt smaller, so the model really has fewer parameters.
Then fine-tune 3 epochs.

Usage:
    python lab07/src/prune.py                 # both
    python lab07/src/prune.py --only unstructured
    python lab07/src/prune.py --only structured

Needs lab07/checkpoints/teacher.pt from baseline.py.

Outputs:
    lab07/results/pruning_unstructured.csv
    lab07/results/pruning_global_layer_sparsity.csv
    lab07/results/pruning_structured.csv
    lab07/results/pruning_curves.png
    lab07/checkpoints/pruned/*.pt         (git-ignored, used by latency.py)
"""

import copy
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.utils.prune as prune

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (base_parser, count_nonzero_parameters, evaluate, fit, make_loaders, read_csv,
                    set_seeds, weight_sparsity, write_csv)
from models import (TeacherCNN, build_teacher_from_state, count_parameters, prunable_layers,
                    teacher_state)


SPARSITIES = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95]
FILTER_FRACTIONS = [0.0, 0.10, 0.25, 0.50]
FINETUNE_EPOCHS = 3
FINETUNE_LR = 5e-4


# ============================================================
# 1. UNSTRUCTURED PRUNING
# ============================================================

def apply_unstructured(model, method, sparsity):
    """Attach L1 (absolute value) masks to every conv and linear weight.

    torch.nn.utils.prune replaces `weight` by `weight_orig * weight_mask`,
    so the masked weights stay zero during fine-tuning: their gradient is
    multiplied by 0.
    """

    layers = [m for _, m in prunable_layers(model)]

    if sparsity == 0:
        for m in layers:
            prune.identity(m, "weight")
    elif method == "global":
        prune.global_unstructured([(m, "weight") for m in layers],
                                  pruning_method=prune.L1Unstructured, amount=sparsity)
    elif method == "layer":
        for m in layers:
            prune.l1_unstructured(m, "weight", amount=sparsity)
    else:
        raise ValueError(f"Unknown method {method!r}")


def remove_masks(model):
    """Make the pruning permanent: weight becomes a normal tensor with zeros."""

    for _, m in prunable_layers(model):
        prune.remove(m, "weight")


def layer_sparsities(model):
    return {name: (m.weight == 0).float().mean().item() for name, m in prunable_layers(model)}


def unstructured_study(teacher, loaders, args, epochs):
    train_loader, val_loader, test_loader = loaders
    out_dir = args.checkpoints / "pruned"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, layer_rows = [], []

    for method in ["global", "layer"]:
        for sparsity in SPARSITIES:
            label = f"{method}_{sparsity:g}"
            print(f"\nunstructured {label}")
            set_seeds()
            model = copy.deepcopy(teacher)
            apply_unstructured(model, method, sparsity)
            before = evaluate(model, test_loader)

            start = time.perf_counter()
            fit(model, train_loader, val_loader, epochs, lr=FINETUNE_LR, label=label)
            finetune_seconds = time.perf_counter() - start

            remove_masks(model)
            after = evaluate(model, test_loader)
            val = evaluate(model, val_loader)
            per_layer = layer_sparsities(model)

            rows.append({
                "method": method,
                "target_sparsity": sparsity,
                "achieved_sparsity": weight_sparsity(model),
                "nonzero_parameters": count_nonzero_parameters(model),
                "test_accuracy_before_finetune": before["accuracy"],
                "test_accuracy_after_finetune": after["accuracy"],
                "val_accuracy": val["accuracy"],
                "finetune_seconds": finetune_seconds,
            })
            layer_rows.append({"method": method, "target_sparsity": sparsity,
                               **{f"{name}_sparsity": value for name, value in per_layer.items()}})
            torch.save(teacher_state(model), out_dir / f"unstructured_{label}.pt")

    write_csv(args.results / "pruning_unstructured.csv", rows)
    write_csv(args.results / "pruning_global_layer_sparsity.csv", layer_rows)


# ============================================================
# 2. STRUCTURED PRUNING
# ============================================================

def filters_to_keep(conv, fraction):
    """Indices of the filters with the largest L2 norm, in their original order."""

    n = conv.out_channels
    keep = n - int(n * fraction)
    norms = conv.weight.detach().reshape(n, -1).norm(p=2, dim=1)
    return torch.sort(torch.topk(norms, keep).indices).values


def structured_prune(model, fraction):
    """Return a new, smaller TeacherCNN without the lowest-L2-norm filters.

    Removing output filter j of one conv layer also removes input channel j
    of the next conv layer. For conv3, it removes the 7 x 7 = 49 matching
    input columns of fc1, because flatten puts channels first.
    """

    k1 = filters_to_keep(model.conv1, fraction)
    k2 = filters_to_keep(model.conv2, fraction)
    k3 = filters_to_keep(model.conv3, fraction)
    small = TeacherCNN(len(k1), len(k2), len(k3), hidden=model.hidden)

    with torch.no_grad():
        small.conv1.weight.copy_(model.conv1.weight[k1])
        small.conv1.bias.copy_(model.conv1.bias[k1])
        small.conv2.weight.copy_(model.conv2.weight[k2][:, k1])
        small.conv2.bias.copy_(model.conv2.bias[k2])
        small.conv3.weight.copy_(model.conv3.weight[k3][:, k2])
        small.conv3.bias.copy_(model.conv3.bias[k3])
        fc1 = model.fc1.weight.reshape(model.hidden, model.channels[2], 49)
        small.fc1.weight.copy_(fc1[:, k3, :].reshape(model.hidden, -1))
        small.fc1.bias.copy_(model.fc1.bias)
        small.fc2.weight.copy_(model.fc2.weight)
        small.fc2.bias.copy_(model.fc2.bias)

    return small


def structured_study(teacher, loaders, args, epochs):
    train_loader, val_loader, test_loader = loaders
    out_dir = args.checkpoints / "pruned"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_params = count_parameters(teacher)
    rows = []

    for fraction in FILTER_FRACTIONS:
        label = f"structured_{fraction:g}"
        print(f"\n{label}")
        set_seeds()
        model = structured_prune(teacher, fraction)
        before = evaluate(model, test_loader)

        start = time.perf_counter()
        fit(model, train_loader, val_loader, epochs, lr=FINETUNE_LR, label=label)
        finetune_seconds = time.perf_counter() - start

        after = evaluate(model, test_loader)
        val = evaluate(model, val_loader)
        params = count_parameters(model)
        rows.append({
            "filter_fraction_removed": fraction,
            "channels": "-".join(str(c) for c in model.channels),
            "parameters": params,
            "parameter_reduction": 1 - params / base_params,
            "test_accuracy_before_finetune": before["accuracy"],
            "test_accuracy_after_finetune": after["accuracy"],
            "val_accuracy": val["accuracy"],
            "finetune_seconds": finetune_seconds,
        })
        torch.save(teacher_state(model), out_dir / f"{label}.pt")

    write_csv(args.results / "pruning_structured.csv", rows)


# ============================================================
# 3. PLOT
# ============================================================

def plot(results_dir):
    unstructured_path = results_dir / "pruning_unstructured.csv"
    structured_path = results_dir / "pruning_structured.csv"
    if not unstructured_path.exists() or not structured_path.exists():
        print("plot skipped: run both studies first")
        return

    unstructured = read_csv(unstructured_path)
    structured = read_csv(structured_path)
    baseline = None
    if (results_dir / "baseline.csv").exists():
        baseline = float(read_csv(results_dir / "baseline.csv")[0]["test_accuracy"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    ax = axes[0]
    for method, color in [("global", "tab:blue"), ("layer", "tab:orange")]:
        rows = [r for r in unstructured if r["method"] == method]
        x = [float(r["target_sparsity"]) for r in rows]
        ax.plot(x, [float(r["test_accuracy_after_finetune"]) for r in rows], "o-", color=color,
                label=f"{method}, after 3-epoch fine-tune")
        ax.plot(x, [float(r["test_accuracy_before_finetune"]) for r in rows], "x--", color=color,
                alpha=0.6, label=f"{method}, before fine-tune")
    ax.set_xlabel("weight sparsity (fraction of weights set to zero)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Unstructured magnitude pruning")

    ax = axes[1]
    x = [float(r["parameter_reduction"]) for r in structured]
    ax.plot(x, [float(r["test_accuracy_after_finetune"]) for r in structured], "o-", color="tab:green",
            label="after 3-epoch fine-tune")
    ax.plot(x, [float(r["test_accuracy_before_finetune"]) for r in structured], "x--", color="tab:green",
            alpha=0.6, label="before fine-tune")
    for r, xi in zip(structured, x):
        ax.annotate(f"{float(r['filter_fraction_removed']):.0%} filters", (xi, float(r["test_accuracy_after_finetune"])),
                    textcoords="offset points", xytext=(4, 6), fontsize=8)
    ax.set_xlabel("parameters removed (fraction)")
    ax.set_title("Structured filter pruning (L2 norm)")

    for ax in axes:
        if baseline is not None:
            ax.axhline(baseline, color="grey", linestyle=":", label="baseline")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    path = results_dir / "pruning_curves.png"
    fig.savefig(path, dpi=120)
    print(f"wrote {path}")


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--only", choices=["unstructured", "structured", "all"], default="all")
    parser.add_argument("--finetune-epochs", type=int, default=FINETUNE_EPOCHS)
    args = parser.parse_args()

    teacher = build_teacher_from_state(torch.load(args.checkpoints / "teacher.pt"))
    loaders = make_loaders(args)

    if args.only in ("unstructured", "all"):
        unstructured_study(teacher, loaders, args, args.finetune_epochs)
    if args.only in ("structured", "all"):
        structured_study(teacher, loaders, args, args.finetune_epochs)
    plot(args.results)


if __name__ == "__main__":
    main()
