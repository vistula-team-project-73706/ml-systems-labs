"""Lab 07 tests. They use small random data, so they run in seconds."""

import copy
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from common import (bitmask_to_dense, count_macs, count_nonzero_parameters, dense_size_bytes, measure_latency,
                    sparse_size_bytes, sparse_state, weight_sparsity)
from distill import distillation_loss
from models import StudentCNN, TeacherCNN, build_teacher_from_state, count_parameters, teacher_state
from prune import apply_unstructured, layer_sparsities, remove_masks, structured_prune


def trained_like_teacher(seed=0):
    torch.manual_seed(seed)
    model = TeacherCNN()
    model.eval()
    return model


# ---------- models ----------

def test_teacher_has_about_one_million_parameters():
    assert count_parameters(TeacherCNN()) == 997_434


def test_student_has_about_100k_parameters():
    assert count_parameters(StudentCNN()) == 105_866


def test_models_output_ten_logits():
    x = torch.randn(4, 1, 28, 28)
    assert TeacherCNN()(x).shape == (4, 10)
    assert StudentCNN()(x).shape == (4, 10)


def test_teacher_state_round_trip():
    model = trained_like_teacher()
    copy_ = build_teacher_from_state(teacher_state(model))
    x = torch.randn(2, 1, 28, 28)
    copy_.eval()
    assert torch.equal(model(x), copy_(x))


# ---------- unstructured pruning ----------

@pytest.mark.parametrize("sparsity", [0.25, 0.5, 0.9, 0.95])
def test_global_pruning_hits_target_sparsity(sparsity):
    model = trained_like_teacher()
    apply_unstructured(model, "global", sparsity)
    remove_masks(model)
    assert weight_sparsity(model) == pytest.approx(sparsity, abs=1e-4)


def test_global_pruning_removes_the_smallest_weights():
    model = trained_like_teacher()
    weights = torch.cat([m.weight.detach().abs().flatten()
                         for m in [model.conv1, model.conv2, model.conv3, model.fc1, model.fc2]])
    threshold = weights.kthvalue(int(0.5 * weights.numel())).values
    apply_unstructured(model, "global", 0.5)
    remove_masks(model)
    for m in [model.conv1, model.conv2, model.conv3, model.fc1, model.fc2]:
        kept = m.weight[m.weight != 0].abs()
        assert kept.numel() == 0 or kept.min() >= threshold


def test_layer_pruning_gives_same_sparsity_in_every_layer():
    model = trained_like_teacher()
    apply_unstructured(model, "layer", 0.75)
    remove_masks(model)
    for value in layer_sparsities(model).values():
        assert value == pytest.approx(0.75, abs=0.01)


def test_global_pruning_prunes_layers_unevenly():
    model = trained_like_teacher()
    apply_unstructured(model, "global", 0.75)
    remove_masks(model)
    values = list(layer_sparsities(model).values())
    assert max(values) - min(values) > 0.05


def test_mask_stays_fixed_during_fine_tuning():
    model = trained_like_teacher()
    model.train()
    apply_unstructured(model, "global", 0.9)
    zero_before = model.fc1.weight == 0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(3):
        optimizer.zero_grad()
        F.cross_entropy(model(torch.randn(8, 1, 28, 28)), torch.randint(0, 10, (8,))).backward()
        optimizer.step()
    remove_masks(model)
    assert torch.all(model.fc1.weight[zero_before] == 0)
    assert weight_sparsity(model) == pytest.approx(0.9, abs=1e-4)


# ---------- structured pruning ----------

def test_structured_pruning_zero_fraction_keeps_the_model():
    model = trained_like_teacher()
    small = structured_prune(model, 0.0)
    small.eval()
    x = torch.randn(3, 1, 28, 28)
    assert small.channels == model.channels
    assert torch.allclose(model(x), small(x))


def test_structured_pruning_shrinks_channels_and_parameters():
    model = trained_like_teacher()
    small = structured_prune(model, 0.5)
    assert small.channels == (16, 32, 64)
    assert count_parameters(small) < 0.55 * count_parameters(model)


def test_structured_pruning_equals_zeroing_the_removed_filters():
    """Removing a filter must give the same output as setting it to zero."""

    model = trained_like_teacher()
    small = structured_prune(model, 0.25)
    small.eval()

    zeroed = copy.deepcopy(model)
    with torch.no_grad():
        for conv in [zeroed.conv1, zeroed.conv2, zeroed.conv3]:
            n = conv.out_channels
            norms = conv.weight.reshape(n, -1).norm(dim=1)
            drop = torch.argsort(norms)[: int(n * 0.25)]
            conv.weight[drop] = 0
            conv.bias[drop] = 0
    zeroed.eval()

    x = torch.randn(3, 1, 28, 28)
    assert torch.allclose(zeroed(x), small(x), atol=1e-5)


# ---------- measurement ----------

def test_sparse_state_round_trips_to_dense():
    model = trained_like_teacher()
    apply_unstructured(model, "global", 0.9)
    remove_masks(model)
    for layout in ["coo", "csr"]:
        state = sparse_state(model, layout)
        dense = state["conv2.weight"].to_dense().reshape(model.conv2.weight.shape)
        assert torch.equal(dense, model.conv2.weight)
    entry = sparse_state(model, "bitmask")["conv2.weight"]
    assert torch.equal(bitmask_to_dense(entry), model.conv2.weight)


def test_sparse_file_only_smaller_at_high_sparsity():
    low, high = trained_like_teacher(), trained_like_teacher()
    apply_unstructured(low, "global", 0.25)
    remove_masks(low)
    apply_unstructured(high, "global", 0.95)
    remove_masks(high)
    assert sparse_size_bytes(low, "coo") > dense_size_bytes(low)
    assert sparse_size_bytes(high, "csr") < dense_size_bytes(high)
    assert sparse_size_bytes(low, "bitmask") < dense_size_bytes(low)


def test_dense_size_does_not_change_with_sparsity():
    model = trained_like_teacher()
    before = dense_size_bytes(model)
    apply_unstructured(model, "global", 0.9)
    remove_masks(model)
    assert dense_size_bytes(model) == pytest.approx(before, rel=0.01)


def test_mac_counts():
    model = trained_like_teacher()
    dense, nonzero = count_macs(model)
    expected = 32 * 9 * 784 + 64 * 32 * 9 * 196 + 128 * 64 * 9 * 49 + 6272 * 144 + 144 * 10
    assert dense == expected
    assert nonzero == dense
    apply_unstructured(model, "global", 0.5)
    remove_masks(model)
    assert count_macs(model)[0] == expected
    assert count_macs(model)[1] < expected


def test_nonzero_parameter_count():
    model = trained_like_teacher()
    apply_unstructured(model, "layer", 0.5)
    remove_masks(model)
    assert count_nonzero_parameters(model) < 0.55 * count_parameters(model)


def test_latency_is_positive_and_restores_threads():
    threads = torch.get_num_threads()
    median, p90 = measure_latency(StudentCNN(), batch_size=1, warmup=2, runs=5)
    assert 0 < median <= p90
    assert torch.get_num_threads() == threads


# ---------- distillation ----------

def test_distillation_loss_alpha_one_is_cross_entropy():
    s, t = torch.randn(8, 10), torch.randn(8, 10)
    y = torch.randint(0, 10, (8,))
    assert torch.allclose(distillation_loss(s, t, y, alpha=1.0), F.cross_entropy(s, y))


def test_distillation_soft_term_is_zero_when_student_equals_teacher():
    s = torch.randn(8, 10)
    y = torch.randint(0, 10, (8,))
    assert distillation_loss(s, s.clone(), y, alpha=0.0).abs() < 1e-6


def test_distillation_loss_matches_formula():
    s, t = torch.randn(8, 10), torch.randn(8, 10)
    y = torch.randint(0, 10, (8,))
    T, alpha = 4.0, 0.5
    p_t = F.softmax(t / T, dim=1)
    kl = (p_t * (p_t.log() - F.log_softmax(s / T, dim=1))).sum(1).mean()
    expected = alpha * F.cross_entropy(s, y) + (1 - alpha) * T * T * kl
    assert torch.allclose(distillation_loss(s, t, y, T, alpha), expected, atol=1e-6)


# ---------- end to end ----------

def test_all_scripts_run_on_fake_data(tmp_path):
    python = sys.executable
    common = ["--fake", "--results", str(tmp_path / "results"), "--checkpoints", str(tmp_path / "ckpt")]

    def run(script, *extra):
        subprocess.run([python, str(SRC / script), *common, *extra], check=True,
                       capture_output=True, text=True)

    run("baseline.py", "--epochs", "1")
    run("prune.py", "--finetune-epochs", "1")
    run("distill.py", "--epochs", "1")
    run("latency.py")
    run("summary.py")

    results = tmp_path / "results"
    for name in ["baseline.csv", "pruning_unstructured.csv", "pruning_structured.csv",
                 "pruning_curves.png", "latency_size.csv", "sparse_kernel_benchmark.csv",
                 "latency_size.png", "distillation.csv", "distillation.png",
                 "efficiency_summary.csv", "efficiency_summary.md"]:
        assert (results / name).exists(), name
