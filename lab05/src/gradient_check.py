"""
Lab 05: Gradient validation

Compares the Value engine against two independent references:
    1. PyTorch autograd in float64
    2. Central finite differences, (f(x + h) - f(x - h)) / 2h

The required expression is ((a*b + c)**2).relu().sum().

Its input to relu is a square, so it can never be negative and it never uses
the inactive branch of relu. Four more expressions are checked to cover that
branch, nodes used more than once, sigmoid, log, division and negative powers.

Outputs:
    lab05/results/gradient_check.csv
    lab05/results/gradient_check_summary.csv
"""

import csv
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from value import Value


# ============================================================
# PATHS AND EXPERIMENT SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "lab05" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
RANDOM_POINTS = 100
POINT_RANGE = 3.0
FINITE_DIFFERENCE_H = 1e-5
MAX_ALLOWED_DIFFERENCE = 1e-6


# ============================================================
# 1. EXPRESSIONS
# ============================================================
# Each one runs unchanged on Value objects and on torch tensors.

def spec_expression(a, b, c):
    return ((a * b + c) ** 2).relu().sum()


def relu_both_branches(a, b, c):
    return (a * b + c).relu() ** 2 + (a - c).relu() * b + (b - a).relu()


def shared_nodes(a, b, c):
    d = a * b
    e = d + c
    return d * d + d * e + e * e + a * a


def division_and_negative_power(a, b, c):
    return a / (b ** 2 + 1) + c ** 3 - (a ** 2 + 2) ** -2


def sigmoid_and_log(a, b, c):
    return (a * b + c).sigmoid().log() + (1 - (a - c).sigmoid()).log()


EXPRESSIONS = {
    "spec": spec_expression,
    "relu_both_branches": relu_both_branches,
    "shared_nodes": shared_nodes,
    "division_and_negative_power": division_and_negative_power,
    "sigmoid_and_log": sigmoid_and_log,
}

# Each point exercises the spec expression in a different way. The other
# expressions only use the random points. A central difference is not valid
# exactly on a relu kink, and some of these points sit on one.
SPEC_ONLY_POINTS = {
    "positive_inner": (2.0, -3.0, 10.0),
    "negative_inner": (2.0, -3.0, 1.0),
    "zero_inner": (2.0, 3.0, -6.0),
    "zero_a": (0.0, 3.0, 2.0),
}


# ============================================================
# 2. THE THREE WAYS OF COMPUTING A GRADIENT
# ============================================================

def engine_gradient(expression, point):
    a, b, c = (Value(x) for x in point)
    output = expression(a, b, c)
    output.backward()
    return output.data, [a.grad, b.grad, c.grad]


def torch_gradient(expression, point):
    # float64 is required, float32 rounding alone is larger than 1e-6.
    a, b, c = (torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in point)
    output = expression(a, b, c)
    output.backward()
    return output.item(), [a.grad.item(), b.grad.item(), c.grad.item()]


def numerical_gradient(expression, point, h=FINITE_DIFFERENCE_H):
    gradient = []

    for i in range(len(point)):
        above = list(point)
        below = list(point)
        above[i] += h
        below[i] -= h

        f_above = expression(*(Value(x) for x in above)).data
        f_below = expression(*(Value(x) for x in below)).data
        gradient.append((f_above - f_below) / (2 * h))

    return gradient


def max_abs_difference(x, y):
    return max(abs(p - q) for p, q in zip(x, y))


# ============================================================
# 3. CHECK ONE POINT
# ============================================================

def check_point(name, expression, point_name, point):
    value, engine_grad = engine_gradient(expression, point)
    torch_value, torch_grad = torch_gradient(expression, point)
    numeric_grad = numerical_gradient(expression, point)

    return {
        "expression": name,
        "point": point_name,
        "a": point[0],
        "b": point[1],
        "c": point[2],
        "engine_value": value,
        "torch_value": torch_value,
        "engine_grad_a": engine_grad[0],
        "engine_grad_b": engine_grad[1],
        "engine_grad_c": engine_grad[2],
        "torch_grad_a": torch_grad[0],
        "torch_grad_b": torch_grad[1],
        "torch_grad_c": torch_grad[2],
        "numeric_grad_a": numeric_grad[0],
        "numeric_grad_b": numeric_grad[1],
        "numeric_grad_c": numeric_grad[2],
        "abs_diff_value": abs(value - torch_value),
        "max_abs_diff_engine_vs_torch": max_abs_difference(engine_grad, torch_grad),
        "max_abs_diff_engine_vs_numeric": max_abs_difference(engine_grad, numeric_grad),
    }


def random_points():
    rng = random.Random(SEED)
    return {
        f"random_{i:03d}": tuple(rng.uniform(-POINT_RANGE, POINT_RANGE) for _ in range(3))
        for i in range(RANDOM_POINTS)
    }


def run_checks():
    points = random_points()
    rows = []

    for point_name, point in SPEC_ONLY_POINTS.items():
        rows.append(check_point("spec", spec_expression, point_name, point))

    for name, expression in EXPRESSIONS.items():
        for point_name, point in points.items():
            rows.append(check_point(name, expression, point_name, point))

    return rows


def summarise(rows):
    summary = []

    for name in EXPRESSIONS:
        selected = [row for row in rows if row["expression"] == name]
        worst_torch = max(row["max_abs_diff_engine_vs_torch"] for row in selected)
        worst_numeric = max(row["max_abs_diff_engine_vs_numeric"] for row in selected)

        summary.append({
            "expression": name,
            "points": len(selected),
            "max_abs_diff_engine_vs_torch": worst_torch,
            "max_abs_diff_engine_vs_numeric": worst_numeric,
            "limit": MAX_ALLOWED_DIFFERENCE,
            "passed": worst_torch < MAX_ALLOWED_DIFFERENCE
                      and worst_numeric < MAX_ALLOWED_DIFFERENCE,
        })

    return summary


# ============================================================
# 4. MAIN
# ============================================================

def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = run_checks()
    summary = summarise(rows)

    save_csv(rows, RESULTS_DIR / "gradient_check.csv")
    save_csv(summary, RESULTS_DIR / "gradient_check_summary.csv")

    print(f"{'expression':<30}{'points':>7}{'vs torch':>12}{'vs numeric':>12}  passed")
    for row in summary:
        print(f"{row['expression']:<30}{row['points']:>7}"
              f"{row['max_abs_diff_engine_vs_torch']:>12.2e}"
              f"{row['max_abs_diff_engine_vs_numeric']:>12.2e}  {row['passed']}")

    if not all(row["passed"] for row in summary):
        raise SystemExit("Gradient check failed")


if __name__ == "__main__":
    main()
