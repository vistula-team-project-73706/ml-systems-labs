"""
Lab 02: Neural Network From Scratch
Dataset: sklearn digits

Architecture:
    64 -> W -> ReLU -> W -> ReLU -> 10 -> Softmax

Experiments:
    Hidden widths: 16, 64
    Learning rates: 0.001, 0.01, 0.1
    Batch size: 32
    Epochs: 50

Outputs:
    lab02/results/loss_curves.png
    lab02/results/experiment_results.csv
    lab02/results/gradient_check_results.csv
    lab02/results/timing_results.csv
    lab02/results/report.md
"""

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS AND EXPERIMENT SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "lab02" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HIDDEN_WIDTHS = [16, 64]
LEARNING_RATES = [0.001, 0.01, 0.1]
EPOCHS = 50
BATCH_SIZE = 32
RANDOM_STATE = 42


# ============================================================
# 1. DATASET PREPARATION
# ============================================================

def prepare_data():
    """Load digits, scale features, one-hot encode labels, and split."""

    X, y = load_digits(return_X_y=True)

    # Scale pixel values from [0, 16] to [0, 1].
    X = X.astype(np.float64) / 16.0

    # Convert labels 0-9 into one-hot vectors.
    Y = np.eye(10, dtype=np.float64)[y]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_train, X_test, Y_train, Y_test


# ============================================================
# 2. NEURAL NETWORK
# ============================================================

class NeuralNetwork:
    """A three-layer neural network implemented using NumPy."""

    def __init__(self, hidden_width, seed=42):
        rng = np.random.default_rng(seed)
        width = hidden_width

        # Weights are sampled from N(0, 0.01).
        self.W1 = rng.normal(0, 0.01, size=(64, width))
        self.b1 = np.zeros(width)

        self.W2 = rng.normal(0, 0.01, size=(width, width))
        self.b2 = np.zeros(width)

        self.W3 = rng.normal(0, 0.01, size=(width, 10))
        self.b3 = np.zeros(10)

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def softmax(z):
        # Subtracting the row maximum improves numerical stability.
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        """Calculate probabilities and cache intermediate values."""

        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)

        z2 = a1 @ self.W2 + self.b2
        a2 = self.relu(z2)

        z3 = a2 @ self.W3 + self.b3
        probabilities = self.softmax(z3)

        cache = (X, z1, a1, z2, a2, probabilities)

        return probabilities, cache

    @staticmethod
    def cross_entropy(Y, probabilities):
        """Calculate mean categorical cross-entropy."""

        eps = 1e-12
        probabilities = np.clip(probabilities, eps, 1.0)

        return -np.mean(
            np.sum(Y * np.log(probabilities), axis=1)
        )

    def backward(self, Y, cache):
        """Calculate gradients using the chain rule."""

        X, z1, a1, z2, a2, probabilities = cache
        batch_size = X.shape[0]

        # Output layer: softmax combined with cross-entropy.
        dz3 = (probabilities - Y) / batch_size

        dW3 = a2.T @ dz3
        db3 = np.sum(dz3, axis=0)

        # Second hidden layer.
        da2 = dz3 @ self.W3.T
        dz2 = da2 * (z2 > 0)

        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        # First hidden layer.
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)

        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        return {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
            "W3": dW3,
            "b3": db3,
        }

    def update(self, gradients, learning_rate):
        """Apply a gradient-descent update."""

        self.W1 -= learning_rate * gradients["W1"]
        self.b1 -= learning_rate * gradients["b1"]

        self.W2 -= learning_rate * gradients["W2"]
        self.b2 -= learning_rate * gradients["b2"]

        self.W3 -= learning_rate * gradients["W3"]
        self.b3 -= learning_rate * gradients["b3"]

    def parameters(self):
        return {
            "W1": self.W1,
            "W2": self.W2,
            "W3": self.W3,
        }

    def gradient_check(
        self,
        X,
        Y,
        num_weights=10,
        epsilon=1e-5,
        seed=123,
        max_attempts=500,
    ):
        """
        Compare analytical and numerical gradients for random weights.

        ReLU is not differentiable at zero. If perturbing a weight by
        +/- epsilon changes a ReLU activation pattern, that numerical
        estimate may cross a nondifferentiable point. Such a candidate
        is recorded as a rejected boundary crossing and another weight
        is sampled.

        The returned checks therefore contain num_weights valid
        comparisons, provided enough stable candidates can be found.
        """

        rng = np.random.default_rng(seed)

        probabilities, cache = self.forward(X)
        analytical_gradients = self.backward(Y, cache)

        parameters = self.parameters()
        checks = []
        rejected_boundary_crossings = 0
        attempts = 0

        # Baseline ReLU activation patterns.
        baseline_z1 = cache[1]
        baseline_z2 = cache[3]
        baseline_mask1 = baseline_z1 > 0
        baseline_mask2 = baseline_z2 > 0

        while len(checks) < num_weights and attempts < max_attempts:
            attempts += 1

            parameter_name = rng.choice(list(parameters.keys()))
            parameter = parameters[parameter_name]

            index = tuple(
                rng.integers(0, size)
                for size in parameter.shape
            )

            original_value = parameter[index]

            try:
                # Loss at parameter + epsilon.
                parameter[index] = original_value + epsilon
                p_plus, cache_plus = self.forward(X)
                loss_plus = self.cross_entropy(Y, p_plus)

                # Loss at parameter - epsilon.
                parameter[index] = original_value - epsilon
                p_minus, cache_minus = self.forward(X)
                loss_minus = self.cross_entropy(Y, p_minus)

            finally:
                # Always restore the original value.
                parameter[index] = original_value

            # Check that the perturbation did not cross a ReLU boundary.
            stable_mask1 = (
                np.array_equal(cache_plus[1] > 0, baseline_mask1)
                and np.array_equal(cache_minus[1] > 0, baseline_mask1)
            )

            stable_mask2 = (
                np.array_equal(cache_plus[3] > 0, baseline_mask2)
                and np.array_equal(cache_minus[3] > 0, baseline_mask2)
            )

            if not (stable_mask1 and stable_mask2):
                rejected_boundary_crossings += 1
                continue

            numerical_gradient = (
                loss_plus - loss_minus
            ) / (2 * epsilon)

            analytical_gradient = analytical_gradients[
                parameter_name
            ][index]

            denominator = max(
                1e-12,
                abs(numerical_gradient)
                + abs(analytical_gradient),
            )

            relative_error = (
                abs(numerical_gradient - analytical_gradient)
                / denominator
            )

            checks.append(
                {
                    "parameter": parameter_name,
                    "index": str(index),
                    "analytical": float(analytical_gradient),
                    "numerical": float(numerical_gradient),
                    "relative_error": float(relative_error),
                    "passed": bool(relative_error < 1e-6),
                }
            )

        if len(checks) < num_weights:
            raise RuntimeError(
                "Could not find enough stable gradient-check candidates. "
                "Try a different seed or check batch."
            )

        return checks, rejected_boundary_crossings


# ============================================================
# 3. SANITY CHECKS AND GRADIENT CHECKING
# ============================================================

def run_sanity_checks(X_train, Y_train):
    """Check output dimensions, softmax, initial loss, and gradients."""

    all_gradient_results = []

    print("\n" + "=" * 65)
    print("SANITY CHECKS")
    print("=" * 65)

    X_check = X_train[:32]
    Y_check = Y_train[:32]

    for width in HIDDEN_WIDTHS:
        model = NeuralNetwork(hidden_width=width, seed=RANDOM_STATE)

        probabilities, _ = model.forward(X_train)

        row_sum_error = np.max(
            np.abs(probabilities.sum(axis=1) - 1.0)
        )

        initial_loss = model.cross_entropy(Y_train, probabilities)

        print(f"\nHidden width: {width}")
        print("W1 shape:", model.W1.shape)
        print("W2 shape:", model.W2.shape)
        print("W3 shape:", model.W3.shape)
        print("Output shape:", probabilities.shape)
        print(f"Maximum softmax row-sum error: {row_sum_error:.3e}")
        print(f"Initial cross-entropy loss: {initial_loss:.8f}")

        assert probabilities.shape == (len(X_train), 10)
        assert row_sum_error < 1e-6

        # An untrained 10-class classifier should have loss near ln(10).
        assert abs(initial_loss - np.log(10)) < 0.01

        checks, rejected = model.gradient_check(
            X_check,
            Y_check,
            num_weights=10,
            epsilon=1e-5,
            seed=123,
        )

        passed = sum(row["passed"] for row in checks)

        print(f"Gradient checks passed: {passed}/10")
        print(f"Rejected ReLU-boundary candidates: {rejected}")

        for number, check in enumerate(checks, start=1):
            print(
                f"  {number:02d}. "
                f"{check['parameter']}{check['index']} | "
                f"analytical={check['analytical']:.8e} | "
                f"numerical={check['numerical']:.8e} | "
                f"relative error={check['relative_error']:.3e} | "
                f"{'PASS' if check['passed'] else 'FAIL'}"
            )

            all_gradient_results.append(
                {
                    "hidden_width": width,
                    **check,
                }
            )

        if passed < 9:
            print(
                "WARNING: Fewer than 9/10 gradient checks passed. "
                "Review the failures before submitting."
            )

    save_csv(
        RESULTS_DIR / "gradient_check_results.csv",
        all_gradient_results,
    )

    return all_gradient_results


# ============================================================
# 4. TRAINING
# ============================================================

def train_model(
    model,
    X_train,
    Y_train,
    learning_rate,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    seed=RANDOM_STATE,
):
    """Train using shuffled mini-batches and gradient descent."""

    rng = np.random.default_rng(seed)
    n_samples = X_train.shape[0]
    history = []

    for epoch in range(epochs):
        permutation = rng.permutation(n_samples)
        X_shuffled = X_train[permutation]
        Y_shuffled = Y_train[permutation]

        for start in range(0, n_samples, batch_size):
            end = start + batch_size

            X_batch = X_shuffled[start:end]
            Y_batch = Y_shuffled[start:end]

            probabilities, cache = model.forward(X_batch)
            gradients = model.backward(Y_batch, cache)

            model.update(gradients, learning_rate)

        # Measure loss over the full training set after each epoch.
        train_probabilities, _ = model.forward(X_train)
        epoch_loss = model.cross_entropy(
            Y_train,
            train_probabilities,
        )
        history.append(float(epoch_loss))

        print(
            f"  Epoch {epoch + 1:02d}/{epochs} "
            f"- loss: {epoch_loss:.6f}"
        )

    return history


def accuracy(model, X, Y):
    """Calculate classification accuracy."""

    probabilities, _ = model.forward(X)

    predicted = np.argmax(probabilities, axis=1)
    actual = np.argmax(Y, axis=1)

    return float(np.mean(predicted == actual))


# ============================================================
# 5. TIMING
# ============================================================

def measure_timings(model, X, Y, repetitions=100):
    """
    Measure forward and backward passes separately.

    Each measurement is a single forward or backward call on the
    same fixed batch. Return the median in milliseconds.
    """

    X_batch = X[:BATCH_SIZE]
    Y_batch = Y[:BATCH_SIZE]

    forward_times = []
    backward_times = []

    # Warm-up calls.
    for _ in range(5):
        _, cache = model.forward(X_batch)
        model.backward(Y_batch, cache)

    for _ in range(repetitions):
        start = time.perf_counter()
        _, cache = model.forward(X_batch)
        forward_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        model.backward(Y_batch, cache)
        backward_times.append(time.perf_counter() - start)

    return {
        "forward_median_ms": float(
            np.median(forward_times) * 1000
        ),
        "backward_median_ms": float(
            np.median(backward_times) * 1000
        ),
        "repetitions": repetitions,
        "batch_size": len(X_batch),
    }


# ============================================================
# 6. CSV HELPER
# ============================================================

def save_csv(path, rows):
    """Write a list of dictionaries to a CSV file."""

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 7. PLOT LOSS CURVES
# ============================================================

def save_loss_plot(histories):
    """Plot training loss for all six configurations."""

    plt.figure(figsize=(11, 7))

    for result in histories:
        label = (
            f"Width={result['hidden_width']}, "
            f"LR={result['learning_rate']}"
        )

        plt.plot(
            range(1, len(result["loss_history"]) + 1),
            result["loss_history"],
            label=label,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Training cross-entropy loss")
    plt.title("Lab 2: Training Loss for Different Configurations")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = RESULTS_DIR / "loss_curves.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nSaved loss plot: {output_path}")


# ============================================================
# 8. REPORT
# ============================================================

def write_report(
    X_train,
    X_test,
    gradient_results,
    experiment_results,
    timing_results,
):
    """Create a report containing measured results."""

    gradient_summary = []

    for width in HIDDEN_WIDTHS:
        rows = [
            row for row in gradient_results
            if row["hidden_width"] == width
        ]
        passed = sum(row["passed"] for row in rows)
        gradient_summary.append(
            f"- Width {width}: {passed}/10 checks passed."
        )

    best = max(
        experiment_results,
        key=lambda row: row["test_accuracy"],
    )

    timing_summary = []

    for result in timing_results:
        timing_summary.append(
            "- Width {hidden_width}: forward median "
            "{forward_median_ms:.6f} ms; backward median "
            "{backward_median_ms:.6f} ms (batch size {batch_size}, "
            "{repetitions} repetitions).".format(**result)
        )

    experiment_summary = []

    for result in experiment_results:
        experiment_summary.append(
            "| {hidden_width} | {learning_rate} | "
            "{final_train_loss:.6f} | {test_accuracy:.4f} |".format(
                **result
            )
        )

    report = f"""# Lab 2 — Neural Network From Scratch

## Dataset and preprocessing

- Dataset: scikit-learn handwritten digits.
- Input features: 64 pixels per sample.
- Pixel values scaled from 0–16 to 0–1 by dividing by 16.
- Labels: one-hot encoded into 10 classes.
- Split: stratified 70% training and 30% testing.
- Random state: {RANDOM_STATE}.
- Training samples: {len(X_train)}.
- Testing samples: {len(X_test)}.

## Architecture

The network was implemented from scratch using NumPy:

`64 → W → ReLU → W → ReLU → 10 → Softmax`

The hidden widths tested were 16 and 64. Weights were initialized
from a normal distribution with mean 0 and standard deviation 0.01.
Biases were initialized to zero.

The loss function was categorical cross-entropy. Parameters were
updated using mini-batch gradient descent.

## Gradient checking

Numerical gradients were calculated using central finite differences
with epsilon = 1e-5. The numerical and analytical gradients were
compared using relative error, with a pass threshold of 1e-6.

ReLU activation patterns can change when parameters are perturbed
near zero. Candidate perturbations that changed the ReLU activation
patterns were rejected and resampled because the central difference
would cross a nondifferentiable point.

{chr(10).join(gradient_summary)}

See `gradient_check_results.csv` for individual measurements.

## Training configuration

- Epochs: {EPOCHS}
- Mini-batch size: {BATCH_SIZE}
- Learning rates: {LEARNING_RATES}
- Hidden widths: {HIDDEN_WIDTHS}

## Experiment results

| Hidden width | Learning rate | Final training loss | Test accuracy |
|---:|---:|---:|---:|
{chr(10).join(experiment_summary)}

Best measured test accuracy: **{best['test_accuracy']:.4f}**
with hidden width {best['hidden_width']} and learning rate
{best['learning_rate']}.

See `experiment_results.csv` for the complete results and
`loss_curves.png` for the training loss curves.

## Timing

Forward and backward passes were measured separately using a fixed
mini-batch. Each timing is the median of 100 repetitions.

{chr(10).join(timing_summary)}

Forward propagation performs matrix multiplications to calculate
layer activations. Backpropagation also performs matrix multiplications
to propagate gradients and calculate weight gradients. Which stage
takes longer depends on the hidden width, batch size, implementation,
and hardware. The measured medians above should be used to describe
the observed result rather than assuming one stage is always slower.

## Outputs

- `loss_curves.png`
- `experiment_results.csv`
- `gradient_check_results.csv`
- `timing_results.csv`
- `report.md`
"""

    report_path = RESULTS_DIR / "report.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"Saved report: {report_path}")


# ============================================================
# 9. MAIN PROGRAM
# ============================================================

def main():
    X_train, X_test, Y_train, Y_test = prepare_data()

    print("=" * 65)
    print("LAB 2: NEURAL NETWORK FROM SCRATCH")
    print("=" * 65)
    print("Training input:", X_train.shape)
    print("Testing input:", X_test.shape)
    print("Training labels:", Y_train.shape)
    print("Testing labels:", Y_test.shape)
    print(f"Input range: [{X_train.min():.1f}, {X_train.max():.1f}]")

    # First validate the implementation.
    gradient_results = run_sanity_checks(X_train, Y_train)

    experiment_results = []
    timing_results = []
    histories = []

    # Run the six required combinations.
    for width in HIDDEN_WIDTHS:
        for learning_rate in LEARNING_RATES:
            print("\n" + "=" * 65)
            print(
                f"Training width={width}, "
                f"learning_rate={learning_rate}"
            )
            print("=" * 65)

            # Use a reproducible seed for each configuration.
            model_seed = (
                RANDOM_STATE
                + width * 100
                + int(learning_rate * 10000)
            )

            model = NeuralNetwork(
                hidden_width=width,
                seed=model_seed,
            )

            loss_history = train_model(
                model,
                X_train,
                Y_train,
                learning_rate=learning_rate,
            )

            test_accuracy = accuracy(model, X_test, Y_test)

            final_train_loss = loss_history[-1]

            result = {
                "hidden_width": width,
                "learning_rate": learning_rate,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "final_train_loss": float(final_train_loss),
                "test_accuracy": float(test_accuracy),
                "loss_history": loss_history,
            }

            histories.append(result)

            experiment_results.append(
                {
                    "hidden_width": width,
                    "learning_rate": learning_rate,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "final_train_loss": float(final_train_loss),
                    "test_accuracy": float(test_accuracy),
                }
            )

            print(f"Final training loss: {final_train_loss:.6f}")
            print(f"Test accuracy: {test_accuracy:.4f}")

            # Measure timings on a fixed batch.
            timing = measure_timings(
                model,
                X_train,
                Y_train,
                repetitions=100,
            )

            timing["hidden_width"] = width
            timing["learning_rate"] = learning_rate
            timing_results.append(timing)

            print(
                "Forward median: "
                f"{timing['forward_median_ms']:.6f} ms"
            )
            print(
                "Backward median: "
                f"{timing['backward_median_ms']:.6f} ms"
            )

    # Save the six experiment results.
    save_csv(
        RESULTS_DIR / "experiment_results.csv",
        experiment_results,
    )

    # Save timing results.
    save_csv(
        RESULTS_DIR / "timing_results.csv",
        timing_results,
    )

    # Plot training loss for all configurations.
    save_loss_plot(histories)

    # Create a report using actual measurements.
    write_report(
        X_train,
        X_test,
        gradient_results,
        experiment_results,
        timing_results,
    )

    best = max(
        experiment_results,
        key=lambda row: row["test_accuracy"],
    )

    print("\n" + "=" * 65)
    print("ALL EXPERIMENTS FINISHED")
    print("=" * 65)
    print(f"Best test accuracy: {best['test_accuracy']:.4f}")
    print(f"Hidden width: {best['hidden_width']}")
    print(f"Learning rate: {best['learning_rate']}")
    print(f"\nResults directory: {RESULTS_DIR}")


if __name__ == "__main__":
    main()