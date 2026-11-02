"""
Lab 05: Tiny neural network on make_moons, Value engine vs PyTorch
Dataset: sklearn make_moons

Architecture:
    2 -> 8 -> ReLU -> 1 -> Sigmoid
    Loss: mean binary cross entropy
    Optimiser: full-batch gradient descent

Both frameworks start from the same weights and use float64, so they compute
the same numbers. The loss curves are therefore also a test of the engine.

Outputs:
    lab05/results/nn_gradient_check.csv
    lab05/results/training_runs.csv
    lab05/results/training_speed.csv
    lab05/results/training_history.csv
    lab05/results/loss_curves.png
"""

import csv
import math
import statistics
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from value import Value


# ============================================================
# PATHS AND EXPERIMENT SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "lab05" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_SEED = 42
RUN_SEEDS = [42, 43, 44, 45, 46]
SAMPLES = 300
NOISE = 0.15
TEST_FRACTION = 0.20
HIDDEN = 8
INIT_STD = 0.5
LEARNING_RATE = 1.0
EPOCHS = 1000
TORCH_WARMUP_STEPS = 20
FINITE_DIFFERENCE_H = 1e-6

# After this log(p) is treated as a constant, the same floor as nn.BCELoss.
LOG_FLOOR = -100.0


# ============================================================
# 1. DATASET AND SHARED INITIAL WEIGHTS
# ============================================================

def prepare_data():
    """Split make_moons and scale it with statistics from the training set."""

    X, y = make_moons(n_samples=SAMPLES, noise=NOISE, random_state=DATA_SEED)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_FRACTION,
        random_state=DATA_SEED,
        stratify=y,
    )

    scaler = StandardScaler().fit(X_train)

    return (
        scaler.transform(X_train),
        scaler.transform(X_test),
        y_train.astype(np.float64),
        y_test.astype(np.float64),
    )


def initial_weights(seed):
    """Sample one set of weights, used by both frameworks."""

    rng = np.random.default_rng(seed)

    return {
        "hidden_weight": rng.normal(0, INIT_STD, size=(HIDDEN, 2)),
        "hidden_bias": rng.normal(0, INIT_STD, size=HIDDEN),
        "output_weight": rng.normal(0, INIT_STD, size=(1, HIDDEN)),
        "output_bias": rng.normal(0, INIT_STD, size=1),
    }


# ============================================================
# 2. MODEL ON THE VALUE ENGINE
# ============================================================

class ValueNetwork:
    """2 -> 8 -> 1 network where every weight is one Value."""

    def __init__(self, weights):
        self.hidden_weight = [[Value(w) for w in row] for row in weights["hidden_weight"].tolist()]
        self.hidden_bias = [Value(b) for b in weights["hidden_bias"].tolist()]
        self.output_weight = [Value(w) for w in weights["output_weight"][0].tolist()]
        self.output_bias = Value(weights["output_bias"][0])

    def parameters(self):
        params = [w for row in self.hidden_weight for w in row]
        params += self.hidden_bias
        params += self.output_weight
        params.append(self.output_bias)
        return params

    def forward(self, x):
        output = self.output_bias

        for j in range(HIDDEN):
            w1, w2 = self.hidden_weight[j]
            hidden = (w1 * x[0] + w2 * x[1] + self.hidden_bias[j]).relu()
            output = output + self.output_weight[j] * hidden

        return output.sigmoid()

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


def floored_log(p):
    if p.data <= math.exp(LOG_FLOOR):
        return Value(LOG_FLOOR)
    return p.log()


def value_bce_loss(model, X, y):
    total = Value(0.0)

    for x, target in zip(X, y):
        p = model.forward(x)
        total = total + -(target * floored_log(p) + (1.0 - target) * floored_log(1.0 - p))

    return total / len(X)


def value_accuracy(model, X, y):
    correct = sum((model.forward(x).data > 0.5) == (target > 0.5) for x, target in zip(X, y))
    return correct / len(X)


def train_value(weights, X_train, X_test, y_train, y_test):
    model = ValueNetwork(weights)
    params = model.parameters()

    # Python floats, a numpy float would slow every operation down.
    X_list, y_list = X_train.tolist(), y_train.tolist()

    history = []
    start = time.perf_counter()

    for _ in range(EPOCHS):
        model.zero_grad()
        loss = value_bce_loss(model, X_list, y_list)
        loss.backward()

        for p in params:
            p.data -= LEARNING_RATE * p.grad

        history.append(loss.data)

    seconds = time.perf_counter() - start
    accuracy = value_accuracy(model, X_test.tolist(), y_test.tolist())

    return seconds, history, accuracy


# ============================================================
# 3. THE SAME MODEL IN PYTORCH
# ============================================================

def build_torch_model(weights):
    model = nn.Sequential(
        nn.Linear(2, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, 1),
        nn.Sigmoid(),
    ).double()

    with torch.no_grad():
        model[0].weight.copy_(torch.from_numpy(weights["hidden_weight"]))
        model[0].bias.copy_(torch.from_numpy(weights["hidden_bias"]))
        model[2].weight.copy_(torch.from_numpy(weights["output_weight"]))
        model[2].bias.copy_(torch.from_numpy(weights["output_bias"]))

    return model


def train_torch(weights, X_train, X_test, y_train, y_test, threads):
    torch.set_num_threads(threads)

    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train).unsqueeze(1)
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test).unsqueeze(1)

    loss_fn = nn.BCELoss()

    # A throwaway model, so one-time library start-up is not in the timing.
    warmup_model = build_torch_model(weights)
    warmup_optimizer = torch.optim.SGD(warmup_model.parameters(), lr=LEARNING_RATE)
    for _ in range(TORCH_WARMUP_STEPS):
        warmup_optimizer.zero_grad()
        loss_fn(warmup_model(X_train), y_train).backward()
        warmup_optimizer.step()

    model = build_torch_model(weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    history = []
    start = time.perf_counter()

    for _ in range(EPOCHS):
        optimizer.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        optimizer.step()

        history.append(loss.item())

    seconds = time.perf_counter() - start

    with torch.no_grad():
        accuracy = ((model(X_test) > 0.5).double() == y_test).double().mean().item()

    return seconds, history, accuracy


# ============================================================
# 4. GRADIENT CHECK ON THE FULL NETWORK
# ============================================================

def value_gradients(weights, X, y):
    model = ValueNetwork(weights)
    loss = value_bce_loss(model, X.tolist(), y.tolist())
    loss.backward()
    return [p.grad for p in model.parameters()]


def torch_gradients(weights, X, y):
    model = build_torch_model(weights)
    loss = nn.BCELoss()(model(torch.from_numpy(X)), torch.from_numpy(y).unsqueeze(1))
    loss.backward()

    # Same order as ValueNetwork.parameters().
    grads = model[0].weight.grad.view(-1).tolist()
    grads += model[0].bias.grad.tolist()
    grads += model[2].weight.grad.view(-1).tolist()
    grads += model[2].bias.grad.tolist()
    return grads


def numerical_gradients(weights, X, y, h=FINITE_DIFFERENCE_H):
    X_list, y_list = X.tolist(), y.tolist()
    model = ValueNetwork(weights)
    grads = []

    for p in model.parameters():
        original = p.data
        p.data = original + h
        above = value_bce_loss(model, X_list, y_list).data
        p.data = original - h
        below = value_bce_loss(model, X_list, y_list).data
        p.data = original
        grads.append((above - below) / (2 * h))

    return grads


def check_network_gradients(X, y):
    rows = []

    for seed in RUN_SEEDS:
        weights = initial_weights(seed)
        engine = value_gradients(weights, X, y)
        reference = torch_gradients(weights, X, y)
        numeric = numerical_gradients(weights, X, y)

        rows.append({
            "seed": seed,
            "parameters": len(engine),
            "max_abs_diff_engine_vs_torch": max(abs(p - q) for p, q in zip(engine, reference)),
            "max_abs_diff_engine_vs_numeric": max(abs(p - q) for p, q in zip(engine, numeric)),
        })

    return rows


# ============================================================
# 5. RUN THE COMPARISON
# ============================================================

def graph_nodes_per_step(X, y):
    model = ValueNetwork(initial_weights(RUN_SEEDS[0]))
    return len(value_bce_loss(model, X.tolist(), y.tolist()).topological_order())


def run_comparison(data):
    default_threads = torch.get_num_threads()
    runs = []
    histories = {}

    configurations = [
        ("value_engine", lambda w: train_value(w, *data)),
        ("pytorch_1_thread", lambda w: train_torch(w, *data, threads=1)),
        (f"pytorch_{default_threads}_threads", lambda w: train_torch(w, *data, threads=default_threads)),
    ]

    # One run of each per seed, so slow drift hits all of them equally.
    for seed in RUN_SEEDS:
        weights = initial_weights(seed)

        for name, train in configurations:
            seconds, history, accuracy = train(weights)
            histories[(name, seed)] = history
            runs.append({
                "framework": name,
                "seed": seed,
                "epochs": EPOCHS,
                "seconds": seconds,
                "ms_per_step": 1000 * seconds / EPOCHS,
                "final_train_loss": history[-1],
                "test_accuracy": accuracy,
            })
            print(f"seed {seed} {name:<20} {seconds:9.3f} s  "
                  f"loss {history[-1]:.6f}  accuracy {accuracy:.4f}")

    torch.set_num_threads(default_threads)
    return runs, histories


def summarise_speed(runs):
    frameworks = list(dict.fromkeys(run["framework"] for run in runs))
    baseline = {}
    rows = []

    for framework in frameworks:
        selected = [run for run in runs if run["framework"] == framework]
        baseline[framework] = statistics.median(run["seconds"] for run in selected)

    for framework in frameworks:
        selected = [run for run in runs if run["framework"] == framework]
        rows.append({
            "framework": framework,
            "runs": len(selected),
            "median_seconds": baseline[framework],
            "min_seconds": min(run["seconds"] for run in selected),
            "max_seconds": max(run["seconds"] for run in selected),
            "median_ms_per_step": 1000 * baseline[framework] / EPOCHS,
            "median_final_train_loss": statistics.median(run["final_train_loss"] for run in selected),
            "median_test_accuracy": statistics.median(run["test_accuracy"] for run in selected),
            "times_slower_than_value_engine": baseline[framework] / baseline["value_engine"],
            "value_engine_times_slower": baseline["value_engine"] / baseline[framework],
        })

    return rows


# ============================================================
# 6. OUTPUTS
# ============================================================

def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_history(histories, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["framework", "seed", "epoch", "train_loss"])
        for (framework, seed), history in histories.items():
            for epoch, loss in enumerate(history):
                writer.writerow([framework, seed, epoch, loss])


def plot_loss_curves(histories, path):
    seed = RUN_SEEDS[0]
    engine = np.array(histories[("value_engine", seed)])
    reference = np.array(histories[("pytorch_1_thread", seed)])

    difference = np.abs(engine - reference)

    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))

    left.plot(engine, color="#2a78d6", linewidth=2, label="Value engine")
    left.plot(reference, color="#eb6834", linewidth=2, linestyle=(0, (4, 4)), label="PyTorch")
    left.set_title(f"Training loss, seed {seed}", loc="left", fontsize=11)
    left.set_xlabel("Epoch")
    left.set_ylabel("BCE loss")
    left.legend(frameon=False)

    right.scatter(np.arange(EPOCHS), difference, color="#2a78d6", s=8, linewidths=0)
    right.set_title("Absolute difference between the two curves", loc="left", fontsize=11)
    right.set_xlabel("Epoch")
    right.set_ylabel("|Value engine − PyTorch|")

    for axes in (left, right):
        axes.spines[["top", "right"]].set_visible(False)
        axes.grid(axis="y", color="#e6e5e0", linewidth=0.8)

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def max_history_difference(histories):
    engine_runs = {seed: h for (name, seed), h in histories.items() if name == "value_engine"}
    worst = 0.0

    for (name, seed), history in histories.items():
        if name != "value_engine":
            worst = max(worst, max(abs(p - q) for p, q in zip(history, engine_runs[seed])))

    return worst


def main():
    data = prepare_data()
    X_train, _, y_train, _ = data

    print(f"Train: {len(X_train)}  Test: {len(data[1])}")
    print(f"Value nodes in one training step: {graph_nodes_per_step(X_train, y_train)}")

    gradient_rows = check_network_gradients(X_train, y_train)
    save_csv(gradient_rows, RESULTS_DIR / "nn_gradient_check.csv")
    for row in gradient_rows:
        print(row)

    runs, histories = run_comparison(data)
    speed = summarise_speed(runs)

    save_csv(runs, RESULTS_DIR / "training_runs.csv")
    save_csv(speed, RESULTS_DIR / "training_speed.csv")
    save_history(histories, RESULTS_DIR / "training_history.csv")
    plot_loss_curves(histories, RESULTS_DIR / "loss_curves.png")

    for row in speed:
        print(row)
    print(f"Largest loss difference from the Value engine, any seed and epoch: "
          f"{max_history_difference(histories):.3e}")


if __name__ == "__main__":
    main()
