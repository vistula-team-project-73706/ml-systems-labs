"""Tests for the Value node, its operations and the backward pass."""

import math

import pytest

from value import Value


# ============================================================
# TASK 1: THE NODE
# ============================================================

def test_new_node_stores_data_and_zero_gradient():
    x = Value(3)

    assert x.data == 3.0
    assert x.grad == 0.0
    assert x.parents == ()
    assert x.rule is None


def test_operation_node_stores_parents_and_rule():
    a, b = Value(2.0), Value(3.0)
    c = a * b

    assert c.parents == (a, b)
    assert c.op == "*"
    assert c.rule(c) == (3.0, 2.0)


# ============================================================
# TASK 2: OPERATIONS BUILD THE GRAPH
# ============================================================

@pytest.mark.parametrize("build, expected, op", [
    (lambda a, b: a + b, 5.0, "+"),
    (lambda a, b: a * b, 6.0, "*"),
    (lambda a, b: a ** 3, 8.0, "**3"),
    (lambda a, b: a.relu(), 2.0, "relu"),
    (lambda a, b: a.sum(), 2.0, "sum"),
])
def test_primitive_forward_value_and_op(build, expected, op):
    out = build(Value(2.0), Value(3.0))

    assert out.data == expected
    assert out.op == op
    assert len(out.parents) > 0


def test_relu_clips_negative_input():
    assert Value(-2.0).relu().data == 0.0


def test_plain_numbers_mix_with_values_on_either_side():
    x = Value(4.0)

    assert (x + 1).data == 5.0
    assert (1 + x).data == 5.0
    assert (x * 2).data == 8.0
    assert (2 * x).data == 8.0
    assert (x - 1).data == 3.0
    assert (1 - x).data == -3.0
    assert (x / 2).data == 2.0
    assert (2 / x).data == 0.5
    assert (-x).data == -4.0


def test_forward_pass_builds_expected_graph():
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    out = ((a * b + c) ** 2).relu().sum()

    ops = [node.op for node in out.topological_order() if node.op]

    assert ops == ["*", "+", "**2", "relu", "sum"]


def test_pow_rejects_value_exponent():
    with pytest.raises(TypeError):
        Value(2.0) ** Value(2.0)


def test_pow_rejects_fractional_power_of_negative_base():
    with pytest.raises(ValueError):
        Value(-8.0) ** 0.5


def test_log_rejects_non_positive_input():
    with pytest.raises(ValueError):
        Value(0.0).log()


def test_sigmoid_does_not_overflow():
    assert Value(-1000.0).sigmoid().data == 0.0
    assert Value(1000.0).sigmoid().data == 1.0


# ============================================================
# TASK 3: TOPOLOGICAL SORT AND BACKWARD
# ============================================================

def test_topological_order_puts_parents_first():
    a, b = Value(1.0), Value(2.0)
    c = a * b
    d = c + a
    out = d * c

    order = out.topological_order()
    position = {node: i for i, node in enumerate(order)}

    assert len(order) == 5
    for node in order:
        for parent in node.parents:
            assert position[parent] < position[node]


def test_topological_order_lists_shared_node_once():
    x = Value(2.0)
    y = x * x
    out = y + y

    assert len(out.topological_order()) == 3


def test_output_gradient_is_one():
    x = Value(5.0)
    out = x * 2.0
    out.backward()

    assert out.grad == 1.0


def test_gradients_of_each_primitive():
    a, b = Value(2.0), Value(3.0)
    (a + b).backward()
    assert (a.grad, b.grad) == (1.0, 1.0)

    a, b = Value(2.0), Value(3.0)
    (a * b).backward()
    assert (a.grad, b.grad) == (3.0, 2.0)

    a = Value(2.0)
    (a ** 3).backward()
    assert a.grad == 12.0

    a = Value(2.0)
    a.relu().backward()
    assert a.grad == 1.0

    a = Value(-2.0)
    a.relu().backward()
    assert a.grad == 0.0


def test_relu_gradient_at_zero_is_zero_like_pytorch():
    a = Value(0.0)
    a.relu().backward()

    assert a.grad == 0.0


def test_zero_exponent_at_zero_has_zero_gradient():
    a = Value(0.0)
    (a ** 0).backward()

    assert a.grad == 0.0


def test_gradient_accumulates_when_a_node_is_used_twice():
    # If backward overwrote gradients these would be 1 and 3.
    x = Value(3.0)
    (x + x).backward()
    assert x.grad == 2.0

    x = Value(3.0)
    (x * x).backward()
    assert x.grad == 6.0


def test_gradient_accumulates_through_two_paths():
    # out = (a * b) + (a + b), so d(out)/da = b + 1 and d(out)/db = a + 1.
    a, b = Value(2.0), Value(5.0)
    ((a * b) + (a + b)).backward()

    assert a.grad == 6.0
    assert b.grad == 3.0


def test_node_is_finished_before_it_passes_its_gradient_on():
    # c reaches out through two paths. A wrong visiting order would pass on a
    # partial c.grad, and a.grad would not be the correct 2 * 2 * a * a * ...
    a = Value(3.0)
    c = a * a
    d = c * 2.0
    out = c * d       # out = 2 * a ** 4
    out.backward()

    assert a.grad == pytest.approx(8.0 * 3.0 ** 3)


def test_second_backward_accumulates_leaves_without_double_counting():
    x = Value(3.0)
    out = (x * x) + x

    out.backward()
    assert x.grad == 7.0

    # Same as PyTorch with retain_graph=True: 7 + 7, not 7 + 21.
    out.backward()
    assert x.grad == 14.0


def test_very_deep_graph_does_not_hit_recursion_limit():
    x = Value(1.0)
    out = x
    for _ in range(20_000):
        out = out + 1.0
    out.backward()

    assert out.data == 20_001.0
    assert x.grad == 1.0


def test_derived_operation_gradients():
    a, b = Value(6.0), Value(3.0)
    (a - b).backward()
    assert (a.grad, b.grad) == (1.0, -1.0)

    a, b = Value(6.0), Value(3.0)
    (a / b).backward()
    assert a.grad == pytest.approx(1.0 / 3.0)
    assert b.grad == pytest.approx(-6.0 / 9.0)


def test_sigmoid_and_log_gradients():
    x = Value(0.5)
    x.sigmoid().backward()
    s = 1.0 / (1.0 + math.exp(-0.5))
    assert x.grad == pytest.approx(s * (1 - s))

    x = Value(4.0)
    x.log().backward()
    assert x.grad == pytest.approx(0.25)
