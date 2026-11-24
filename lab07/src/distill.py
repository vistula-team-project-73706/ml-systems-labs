"""
Lab 07 Task 4: knowledge distillation

Teacher = the oversized baseline from baseline.py (997,434 parameters).
Student = StudentCNN (105,866 parameters).

The student is trained twice, from the same initial weights and with the
same batch order, for the same number of epochs:
    plain:     loss = CE(student, labels)
    distilled: loss = alpha * CE(student, labels)
                      + (1 - alpha) * T^2 * KL(teacher_T || student_T)
with T = 4 and alpha = 0.5, where *_T = softmax(logits / T).

KL direction: F.kl_div(log p_student, p_teacher) computes
sum p_teacher * log(p_teacher / p_student), the standard Hinton et al. (2015)
form. Its gradient pushes the student's softened distribution towards the
teacher's. The T^2 factor keeps the soft-target gradient on the same scale
as the CE gradient when T changes.

Usage:
    python lab07/src/distill.py
    python lab07/src/distill.py --epochs 10

Outputs:
    lab07/results/distillation.csv
    lab07/results/distillation_curves.csv
    lab07/results/distillation.png
    lab07/checkpoints/student_plain.pt, student_distilled.pt   (git-ignored)
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline import describe
from common import base_parser, fit, make_loaders, set_seeds, write_csv
from models import StudentCNN, build_teacher_from_state

TEMPERATURE = 4.0
ALPHA = 0.5


def distillation_loss(student_logits, teacher_logits, labels, temperature=TEMPERATURE, alpha=ALPHA):
    hard = F.cross_entropy(student_logits, labels)
    soft = F.kl_div(F.log_softmax(student_logits / temperature, dim=1),
                    F.softmax(teacher_logits / temperature, dim=1),
                    reduction="batchmean")
    return alpha * hard + (1 - alpha) * temperature ** 2 * soft


def make_distill_loss_fn(teacher, temperature=TEMPERATURE, alpha=ALPHA):
    teacher.eval()

    def loss_fn(student, x, y):
        with torch.no_grad():
            teacher_logits = teacher(x)
        return distillation_loss(student(x), teacher_logits, y, temperature, alpha)

    return loss_fn


def train_student(name, loaders, epochs, loss_fn=None):
    train_loader, val_loader, _ = loaders
    set_seeds()  # same initial weights and batch order for both students
    student = StudentCNN()
    start = time.perf_counter()
    kwargs = {} if loss_fn is None else {"loss_fn": loss_fn}
    history = fit(student, train_loader, val_loader, epochs, label=name, **kwargs)
    return student, history, time.perf_counter() - start


def plot(curves, rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for run, style in [("student_plain", "o-"), ("student_distilled", "s-")]:
        sub = [r for r in curves if r["run"] == run]
        ax.plot([r["epoch"] for r in sub], [r["val_accuracy"] for r in sub], style, label=run)
    teacher_acc = next(r["test_accuracy"] for r in rows if r["model"] == "teacher")
    ax.axhline(teacher_acc, color="grey", linestyle=":", label="teacher (test)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy")
    ax.set_title("Student training curves")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    names = [r["model"] for r in rows]
    accuracy = [r["test_accuracy"] for r in rows]
    bars = ax.bar(names, accuracy, color=["grey", "tab:blue", "tab:orange"])
    for bar, r in zip(bars, rows):
        ax.annotate(f"{r['test_accuracy']:.4f}\n{r['parameters']:,} params\n{r['latency_ms_batch1']:.3f} ms",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha="center",
                    xytext=(0, 3), textcoords="offset points", fontsize=8)
    low = min(accuracy)
    ax.set_ylim(max(0.0, low - 0.05), min(1.0, max(accuracy) + 0.04))
    ax.set_ylabel("test accuracy")
    ax.set_title("Teacher vs students")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"wrote {path}")


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    loaders = make_loaders(args)
    test_loader = loaders[2]
    teacher = build_teacher_from_state(torch.load(args.checkpoints / "teacher.pt"))
    teacher.eval()
    samples = args.epochs * len(loaders[0].dataset)

    rows = [{"model": "teacher", "epochs": "", "train_samples_seen": "", "train_seconds": "",
             **describe(teacher, test_loader)}]
    curves = []

    for name, loss_fn in [("student_plain", None),
                          ("student_distilled", make_distill_loss_fn(teacher))]:
        print(f"\n{name}")
        student, history, seconds = train_student(name, loaders, args.epochs, loss_fn)
        torch.save(student.state_dict(), args.checkpoints / f"{name}.pt")
        rows.append({"model": name, "epochs": args.epochs, "train_samples_seen": samples,
                     "train_seconds": seconds, **describe(student, test_loader)})
        curves += [{"run": name, **h} for h in history]

    for r in rows:
        r["temperature"] = TEMPERATURE if r["model"] == "student_distilled" else ""
        r["alpha"] = ALPHA if r["model"] == "student_distilled" else ""

    write_csv(args.results / "distillation.csv", rows)
    write_csv(args.results / "distillation_curves.csv", curves)
    plot(curves, rows, args.results / "distillation.png")


if __name__ == "__main__":
    main()
