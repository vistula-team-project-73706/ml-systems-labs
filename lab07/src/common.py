"""
Lab 07: Shared code

Data, seeding, training, evaluation and measurement (size, latency, MACs)
used by every script. Fashion-MNIST uses the same 55k / 5k split, seed and
normalisation as Labs 3 and 6.
"""

import argparse
import csv
import gzip
import io
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


LAB_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = LAB_DIR.parent
RESULTS = LAB_DIR / "results"
CHECKPOINTS = LAB_DIR / "checkpoints"
DATA_ROOT = PROJECT_ROOT / "data"

SEED = 42
MEAN, STD = 0.2860, 0.3530
TRAIN_SIZE, VAL_SIZE = 55_000, 5_000
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 512
LR = 1e-3

# Latency: median of RUNS timed calls after WARMUP untimed calls, on one CPU
# thread so that numbers are stable and comparable between models.
LATENCY_THREADS = 1
WARMUP = 20
RUNS = 200


# ============================================================
# 1. COMMAND LINE
# ============================================================

def base_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--fake", action="store_true",
                        help="random images instead of Fashion-MNIST (tests only)")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="use only the first N training samples (quick check, not for the report)")
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--checkpoints", type=Path, default=CHECKPOINTS)
    return parser


# ============================================================
# 2. DATA
# ============================================================

def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_datasets(fake=False, max_train_samples=None):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])

    if fake:
        train_size, val_size = 400, 100
        full_train = datasets.FakeData(size=train_size + val_size, image_size=(1, 28, 28),
                                       num_classes=10, transform=transform)
        test_set = datasets.FakeData(size=100, image_size=(1, 28, 28), num_classes=10,
                                     transform=transform, random_offset=1_000_000)
    else:
        train_size, val_size = TRAIN_SIZE, VAL_SIZE
        full_train = datasets.FashionMNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(root=DATA_ROOT, train=False, download=True, transform=transform)

    split = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=split)

    if max_train_samples is not None:
        train_set = Subset(train_set, range(min(max_train_samples, len(train_set))))

    return train_set, val_set, test_set


def make_loaders(args):
    train_set, val_set, test_set = make_datasets(args.fake, args.max_train_samples)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_set, batch_size=EVAL_BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=EVAL_BATCH_SIZE)
    return train_loader, val_loader, test_loader


# ============================================================
# 3. TRAINING AND EVALUATION
# ============================================================

def cross_entropy_loss(model, x, y):
    return F.cross_entropy(model(x), y)


def train_one_epoch(model, loader, optimizer, loss_fn=cross_entropy_loss, after_step=None):
    """One epoch. loss_fn(model, x, y) returns the batch loss.

    after_step, if given, runs after every optimizer step (used to keep
    pruned weights at zero).
    """

    model.train()
    total_loss, total = 0.0, 0
    start = time.perf_counter()

    for x, y in loader:
        optimizer.zero_grad()
        loss = loss_fn(model, x, y)
        loss.backward()
        optimizer.step()
        if after_step is not None:
            after_step()
        total_loss += loss.item() * len(y)
        total += len(y)

    return {"train_loss": total_loss / total, "train_seconds": time.perf_counter() - start}


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    for x, y in loader:
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return {"loss": total_loss / total, "accuracy": correct / total}


def fit(model, train_loader, val_loader, epochs, lr=LR, loss_fn=cross_entropy_loss,
        after_step=None, label=""):
    """Train for a fixed number of epochs with Adam and return the history."""

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    history = []
    for epoch in range(1, epochs + 1):
        stats = train_one_epoch(model, train_loader, optimizer, loss_fn, after_step)
        val = evaluate(model, val_loader)
        row = {"epoch": epoch, **stats, "val_loss": val["loss"], "val_accuracy": val["accuracy"]}
        history.append(row)
        print(f"  {label} epoch {epoch}/{epochs}  loss {stats['train_loss']:.4f}  "
              f"val_acc {val['accuracy']:.4f}  {stats['train_seconds']:.1f}s", flush=True)
    return history


# ============================================================
# 4. MEASUREMENT
# ============================================================

def state_dict_bytes(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return buffer.getbuffer().nbytes


def dense_size_bytes(model):
    """Size of the normal float32 state_dict, as torch.save writes it."""

    return state_dict_bytes(model.state_dict())


def gzip_size_bytes(model):
    """Dense file compressed with gzip. Runs of zeros compress well."""

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(gzip.compress(buffer.getvalue(), compresslevel=6))


def sparse_state(model, layout):
    """state_dict with every conv and linear weight stored in a sparse layout.

    layout:
        "coo"     PyTorch COO: int64 row and column index per non-zero (20 bytes each)
        "csr"     PyTorch CSR: int64 column index per non-zero plus row pointers (12 bytes each)
        "bitmask" the non-zero float32 values plus 1 bit per weight saying
                  where they go: a compact custom format (4 bytes per non-zero
                  plus 1/8 byte per weight)
    Conv weights are reshaped to 2-D (out_channels, in * k * k) first.
    """

    state = {}
    for name, tensor in model.state_dict().items():
        if name.endswith("weight") and tensor.dim() >= 2:
            matrix = tensor.reshape(tensor.shape[0], -1)
            if layout == "coo":
                state[name] = matrix.to_sparse()
            elif layout == "csr":
                state[name] = matrix.to_sparse_csr()
            elif layout == "bitmask":
                mask = (matrix != 0).flatten().numpy()
                state[name] = {"shape": tuple(tensor.shape),
                               "mask": torch.from_numpy(np.packbits(mask)),
                               "values": matrix.flatten()[torch.from_numpy(mask)].clone()}
            else:
                raise ValueError(f"Unknown layout {layout!r}")
        else:
            state[name] = tensor
    return state


def bitmask_to_dense(entry):
    numel = int(np.prod(entry["shape"]))
    mask = torch.from_numpy(np.unpackbits(entry["mask"].numpy())[:numel].astype(bool))
    dense = torch.zeros(numel)
    dense[mask] = entry["values"]
    return dense.reshape(entry["shape"])


def sparse_size_bytes(model, layout):
    return state_dict_bytes(sparse_state(model, layout))


def weight_sparsity(model):
    """Fraction of zero weights over all conv and linear weights (biases excluded)."""

    zeros, total = 0, 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            zeros += (m.weight == 0).sum().item()
            total += m.weight.numel()
    return zeros / total


def count_nonzero_parameters(model):
    return sum((p != 0).sum().item() for p in model.parameters())


def count_macs(model):
    """Multiply-accumulates for one 28x28 image.

    dense_macs counts every weight. nonzero_macs counts only non-zero
    weights: the work a perfect sparse kernel would do. Dense CPU kernels
    always do dense_macs.
    """

    dense, nonzero = 0, 0
    hooks = []

    def conv_hook(module, inputs, output):
        nonlocal dense, nonzero
        positions = output.shape[2] * output.shape[3]
        dense += module.weight.numel() * positions
        nonzero += (module.weight != 0).sum().item() * positions

    def linear_hook(module, inputs, output):
        nonlocal dense, nonzero
        dense += module.weight.numel()
        nonzero += (module.weight != 0).sum().item()

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, 28, 28))
    for h in hooks:
        h.remove()
    return dense, nonzero


def time_calls(fn, warmup=WARMUP, runs=RUNS):
    """Median and 90th-percentile time of fn() in milliseconds."""

    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return statistics.median(times), times[int(0.9 * (len(times) - 1))]


def measure_latency(model, batch_size=1, warmup=WARMUP, runs=RUNS, threads=LATENCY_THREADS):
    """Inference latency per batch in ms (median, p90) on `threads` CPU threads."""

    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        model.eval()
        x = torch.randn(batch_size, 1, 28, 28)
        with torch.inference_mode():
            return time_calls(lambda: model(x), warmup, runs)
    finally:
        torch.set_num_threads(previous)


# ============================================================
# 5. OUTPUT
# ============================================================

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))
