"""Tests for the PyTorch framework analysis helpers (Task 6)."""

import pytest
import torch
import torch.nn as nn

import framework_analysis
from framework_analysis import (
    OpChain,
    WORKLOADS,
    autograd_graph,
    compile_summary,
    describe,
    make_view_cases,
    measure_in_fresh_process,
    measure_mode,
    view_summary,
)


def test_autograd_graph_contains_operations_and_leaf_parameters():
    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1), nn.Sigmoid())
    loss = nn.BCELoss()(model(torch.rand(4, 2)), torch.ones(4, 1))

    lines, count = autograd_graph(loss.grad_fn)
    text = "\n".join(lines)

    assert "BinaryCrossEntropyBackward0" in text
    assert "SigmoidBackward0" in text
    assert "ReluBackward0" in text
    assert text.count("AccumulateGrad") == 4
    assert count == len(lines)


def test_describe_reports_layout_of_a_strided_view():
    base = torch.zeros(4, 6)
    row = describe("view", base[:, ::2])

    assert row["shape"] == [4, 3]
    assert row["strides"] == [6, 2]
    assert row["is_contiguous"] is False
    assert row["storage_elements"] == 24
    assert row["elements"] == 12


def test_view_shares_memory_and_contiguous_copies():
    base = torch.zeros(4, 6)
    view = base[:, ::2]
    copy = view.contiguous()

    view[0, 0] = 1.0

    assert base[0, 0] == 1.0
    assert copy[0, 0] == 0.0
    assert copy.is_contiguous()


def test_view_method_needs_one_even_step_through_memory():
    base = torch.zeros(4, 6)

    # Not contiguous, but every element is exactly 2 apart.
    assert base[:, ::2].view(-1).shape == (12,)

    with pytest.raises(RuntimeError):
        base.t().view(-1)


def test_view_cases_differ_only_in_layout():
    cases = make_view_cases(16)

    for case, x in cases.items():
        assert x.shape == (16, 16)
        assert x.is_contiguous() == (case == "contiguous")


@pytest.mark.parametrize("workload", WORKLOADS)
def test_jit_trace_matches_eager(workload):
    row = measure_with_few_calls(workload, "jit_trace")

    assert row["output_matches_eager"]
    assert row["setup_and_first_call_ms"] > 0
    assert row["after_warmup_median_ms"] > 0


def measure_with_few_calls(workload, mode):
    framework_analysis.COMPILE_WARMUP_CALLS = 10
    framework_analysis.COMPILE_MEASURE_CALLS = 5
    return measure_mode(workload, mode, repeat=0)


def test_torch_compile_matches_eager():
    row = measure_with_few_calls("tiny_batch_1", "torch_compile")

    assert row["output_matches_eager"]


def test_every_fresh_process_pays_the_real_compile_cost():
    # Guards against timing a cache hit. A hit takes about 30 ms in-process,
    # so a second repeat near the first one means it really compiled.
    first = measure_in_fresh_process("tiny_batch_1", "torch_compile", 0)
    second = measure_in_fresh_process("tiny_batch_1", "torch_compile", 1)

    assert first["output_matches_eager"]
    assert second["setup_and_first_call_ms"] > 0.5 * first["setup_and_first_call_ms"]


def test_op_chain_applies_every_step():
    out = OpChain(length=2)(torch.tensor([1.0]))

    assert out.item() == pytest.approx((1.0 * 1.01 + 0.01) * 1.01 + 0.01)


def compile_row(mode, setup, steady):
    return {
        "workload": "tiny_batch_1",
        "mode": mode,
        "repeat": 0,
        "setup_and_first_call_ms": setup,
        "early_calls_median_ms": steady,
        "after_warmup_median_ms": steady,
        "output_matches_eager": True,
    }


def test_compile_summary_break_even():
    rows = [
        compile_row("eager", 1.0, 2.0),
        compile_row("torch_compile", 101.0, 1.0),
        compile_row("jit_trace", 11.0, 3.0),
    ]

    summary = {row["mode"]: row for row in compile_summary(rows)}

    assert summary["torch_compile"]["speedup_vs_eager"] == 2.0
    assert summary["torch_compile"]["break_even_calls"] == 100
    assert summary["jit_trace"]["break_even_calls"] == "never"
    assert summary["eager"]["break_even_calls"] == ""


def test_view_summary_ratio_and_copy_payoff():
    def row(case, median, copy):
        return {"n": 8, "case": case, "operation": "sum", "repeat": 0, "strides": [1, 8],
                "is_contiguous": case == "contiguous", "median_ms": median,
                "copy_to_contiguous_ms": copy}

    summary = view_summary([row("contiguous", 2.0, 0.0), row("every_2nd_column", 6.0, 20.0)])

    assert summary[1]["ratio_vs_contiguous"] == 3.0
    assert summary[1]["copy_pays_off_after_calls"] == 5


def test_copy_that_pays_off_on_the_first_call_reports_one_call():
    def row(case, median, copy):
        return {"n": 8, "case": case, "operation": "sum", "repeat": 0, "strides": [1, 8],
                "is_contiguous": case == "contiguous", "median_ms": median,
                "copy_to_contiguous_ms": copy}

    summary = view_summary([row("contiguous", 1.0, 0.0), row("every_2nd_column", 10.0, 2.0)])

    assert summary[1]["copy_pays_off_after_calls"] == 1
