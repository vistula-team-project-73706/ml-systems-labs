"""
Lab 03: Architectures and their costs
Dataset: Fashion-MNIST

Models:
    MLP:       784 -> 256 -> 128 -> 10
    CNN:       Conv(1,16) -> Pool -> Conv(16,32) -> Pool -> Linear(1568,10)
    Attention: 16 patches of 7x7 -> Linear 64 -> 2 Transformer layers -> mean pool -> 10

Steps:
    1. Load data (55k train / 5k val / 10k test)
    2. Count parameters (torchinfo)
    3. Hand-calculate FLOPs and compare with thop
    4. Measure latency for batch 1 and batch 128
    5. Train each model for 5 epochs with Adam
    6. Plot accuracy vs parameters and vs latency

Outputs:
    lab03/results/model_costs.csv
    lab03/results/flops_comparison.csv
    lab03/results/latency_results.csv
    lab03/results/training_history.csv
    lab03/results/final_results.csv
    lab03/results/torchinfo_summaries.txt
    lab03/results/accuracy_vs_params.png
    lab03/results/accuracy_vs_latency.png
"""

import csv
import random
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from thop import profile
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from torchvision import datasets, transforms


# ============================================================
# PATHS AND EXPERIMENT SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "lab03" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 1e-3
DEVICE = torch.device("cpu")  # CPU so latency numbers are repeatable


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 1. DATASET
# ============================================================

def prepare_data():
    """Fashion-MNIST with 55k/5k train/validation split and official test set."""

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    full_train = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root=DATA_DIR, train=False, download=True, transform=transform)

    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(full_train, [55_000, 5_000], generator=generator)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    print(f"Train: {len(train_set)}  Val: {len(val_set)}  Test: {len(test_set)}")
    return train_loader, val_loader, test_loader


# ============================================================
# 2. MODELS
# ============================================================

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x):
        return self.net(x)


class PatchAttention(nn.Module):
    """Small vision transformer: 16 patches of 7x7 pixels."""

    def __init__(self, patch_size=7, d_model=64, nhead=4, dim_feedforward=128, num_layers=2):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (28 // patch_size) ** 2  # 16

        self.patch_proj = nn.Linear(patch_size * patch_size, d_model)
        # Learned position embedding so the model knows where each patch came from.
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 10)

    def to_patches(self, x):
        # (B, 1, 28, 28) -> (B, 16, 49)
        p = self.patch_size
        b = x.shape[0]
        x = x.unfold(2, p, p).unfold(3, p, p)          # (B, 1, 4, 4, 7, 7)
        return x.reshape(b, self.num_patches, p * p)

    def forward(self, x):
        x = self.to_patches(x)
        x = self.patch_proj(x) + self.pos_embedding
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(x)


def build_models():
    set_seeds()
    return {"MLP": MLP(), "CNN": CNN(), "Attention": PatchAttention()}


# ============================================================
# 3. COST: PARAMETERS AND FLOPS
# ============================================================

def linear_flops(n_in, n_out, tokens=1):
    return 2 * n_in * n_out * tokens


def conv_flops(k, c_in, c_out, h_out, w_out):
    return 2 * k * k * c_in * c_out * h_out * w_out


def attention_layer_flops(seq, d_model, d_ff):
    """FLOPs for one Transformer encoder layer (matmuls only)."""
    qkv = 2 * seq * d_model * 3 * d_model
    scores = 2 * seq * seq * d_model
    weighted_values = 2 * seq * seq * d_model
    out_proj = 2 * seq * d_model * d_model
    ffn = 2 * seq * d_model * d_ff + 2 * seq * d_ff * d_model
    return {
        "qkv_projection": qkv,
        "attention_scores": scores,
        "attention_x_values": weighted_values,
        "output_projection": out_proj,
        "feed_forward": ffn,
        "total": qkv + scores + weighted_values + out_proj + ffn,
    }


def hand_flops():
    """Hand-calculated forward FLOPs for one image (weights only, no bias/ReLU/pool)."""

    mlp = {
        "fc1 784->256": linear_flops(784, 256),
        "fc2 256->128": linear_flops(256, 128),
        "fc3 128->10": linear_flops(128, 10),
    }

    cnn = {
        "conv1 3x3 1->16 @28x28": conv_flops(3, 1, 16, 28, 28),
        "conv2 3x3 16->32 @14x14": conv_flops(3, 16, 32, 14, 14),
        "fc 1568->10": linear_flops(32 * 7 * 7, 10),
    }

    layer = attention_layer_flops(seq=16, d_model=64, d_ff=128)
    attention = {
        "patch projection 49->64 x16": linear_flops(49, 64, tokens=16),
        "encoder layer 1": layer["total"],
        "encoder layer 2": layer["total"],
        "head 64->10": linear_flops(64, 10),
    }

    return {"MLP": mlp, "CNN": cnn, "Attention": attention}, layer


def count_parameters(models):
    rows = []
    with open(RESULTS_DIR / "torchinfo_summaries.txt", "w") as f:
        for name, model in models.items():
            info = summary(model, input_size=(1, 1, 28, 28), verbose=0,
                           col_names=["output_size", "num_params"])
            f.write(f"===== {name} =====\n{info}\n\n")
            rows.append({"model": name, "parameters": info.total_params})
            print(f"{name}: {info.total_params:,} parameters")
    return rows


def compare_flops(models):
    per_layer, attn_layer = hand_flops()
    rows = []

    for name, model in models.items():
        x = torch.randn(1, 1, 28, 28)
        macs, _ = profile(model, inputs=(x,), verbose=False)
        hand_total = sum(per_layer[name].values())
        thop_flops = 2 * macs  # thop reports multiply-accumulates
        rows.append({
            "model": name,
            "hand_flops": hand_total,
            "thop_macs": int(macs),
            "thop_flops_2x_macs": int(thop_flops),
            "ratio_thop_over_hand": round(thop_flops / hand_total, 3),
        })
        print(f"{name}: hand {hand_total:,} FLOPs | thop {int(macs):,} MACs = {int(thop_flops):,} FLOPs")

    # Layer-by-layer hand calculations for the report
    with open(RESULTS_DIR / "hand_flops_per_layer.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "layer", "hand_flops"])
        for name, layers in per_layer.items():
            for layer_name, value in layers.items():
                writer.writerow([name, layer_name, value])
        for part, value in attn_layer.items():
            writer.writerow(["Attention (one encoder layer)", part, value])

    write_csv(RESULTS_DIR / "flops_comparison.csv", rows)
    return rows


# ============================================================
# 4. LATENCY
# ============================================================

def measure_latency(model, batch_size, repeats):
    model.eval()
    x = torch.randn(batch_size, 1, 28, 28)
    times = []
    with torch.inference_mode():
        for _ in range(5):  # warm-up
            model(x)
        for _ in range(repeats):
            start = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def latency_results(models):
    rows = []
    for name, model in models.items():
        b1 = measure_latency(model, batch_size=1, repeats=100)
        b128 = measure_latency(model, batch_size=128, repeats=20)
        rows.append({
            "model": name,
            "batch1_median_ms": round(b1, 4),
            "batch128_median_ms": round(b128, 4),
            "batch128_per_image_ms": round(b128 / 128, 5),
            "threads": torch.get_num_threads(),
        })
        print(f"{name}: batch 1 = {b1:.4f} ms, batch 128 = {b128:.4f} ms")
    write_csv(RESULTS_DIR / "latency_results.csv", rows)
    return rows


# ============================================================
# 5. TRAINING
# ============================================================

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(DEVICE)).argmax(dim=1)
            correct += (pred == y.to(DEVICE)).sum().item()
            total += y.size(0)
    return correct / total


def train_model(name, model, train_loader, val_loader):
    set_seeds()
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, n = 0.0, 0
        start = time.perf_counter()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)
            n += y.size(0)
        epoch_time = time.perf_counter() - start

        val_acc = evaluate(model, val_loader)
        history.append({
            "model": name,
            "epoch": epoch,
            "train_loss": round(running_loss / n, 4),
            "val_accuracy": round(val_acc, 4),
            "epoch_time_s": round(epoch_time, 2),
        })
        print(f"{name} epoch {epoch}: loss {running_loss / n:.4f}, val acc {val_acc:.4f}, {epoch_time:.1f}s")

    return history


# ============================================================
# 6. PLOTS
# ============================================================

def make_plots(final_rows):
    colors = {"MLP": "tab:blue", "CNN": "tab:orange", "Attention": "tab:green"}

    fig, ax = plt.subplots(figsize=(6, 4))
    for r in final_rows:
        ax.scatter(r["parameters"], r["test_accuracy"], s=80, color=colors[r["model"]])
        ax.annotate(r["model"], (r["parameters"], r["test_accuracy"]),
                    textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy vs parameters (Fashion-MNIST, 5 epochs)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "accuracy_vs_params.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in [
        (axes[0], "batch1_median_ms", "Batch 1 latency (ms)"),
        (axes[1], "batch128_median_ms", "Batch 128 latency (ms)"),
    ]:
        for r in final_rows:
            ax.scatter(r[key], r["test_accuracy"], s=80, color=colors[r["model"]])
            ax.annotate(r["model"], (r[key], r["test_accuracy"]),
                        textcoords="offset points", xytext=(6, 6))
        ax.set_xlabel(title)
        ax.set_ylabel("Test accuracy")
        ax.grid(alpha=0.3)
    fig.suptitle("Accuracy vs inference latency (CPU, median)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "accuracy_vs_latency.png", dpi=150)
    plt.close(fig)


# ============================================================
# HELPERS AND MAIN
# ============================================================

def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    set_seeds()
    train_loader, val_loader, test_loader = prepare_data()
    models = build_models()

    print("\n--- Parameters ---")
    param_rows = count_parameters(models)
    write_csv(RESULTS_DIR / "model_costs.csv", param_rows)

    print("\n--- FLOPs ---")
    flop_rows = compare_flops(models)

    print("\n--- Training ---")
    history = []
    test_acc = {}
    for name, model in models.items():
        history += train_model(name, model, train_loader, val_loader)
        test_acc[name] = evaluate(model, test_loader)
        print(f"{name} test accuracy: {test_acc[name]:.4f}\n")
    write_csv(RESULTS_DIR / "training_history.csv", history)

    # Latency after training so the measured models are the trained ones.
    print("--- Latency ---")
    latency_rows = latency_results(models)

    final_rows = []
    for p, fl, lat in zip(param_rows, flop_rows, latency_rows):
        name = p["model"]
        final_rows.append({
            "model": name,
            "parameters": p["parameters"],
            "hand_flops": fl["hand_flops"],
            "thop_flops_2x_macs": fl["thop_flops_2x_macs"],
            "batch1_median_ms": lat["batch1_median_ms"],
            "batch128_median_ms": lat["batch128_median_ms"],
            "test_accuracy": round(test_acc[name], 4),
        })
    write_csv(RESULTS_DIR / "final_results.csv", final_rows)
    make_plots(final_rows)
    print("Saved results to", RESULTS_DIR)


if __name__ == "__main__":
    main()
