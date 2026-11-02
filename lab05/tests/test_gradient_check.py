"""Tests that the engine matches PyTorch and finite differences (Task 4)."""

import pytest

from gradient_check import (
    EXPRESSIONS,
    MAX_ALLOWED_DIFFERENCE,
    SPEC_ONLY_POINTS,
    engine_gradient,
    max_abs_difference,
    numerical_gradient,
    random_points,
    spec_expression,
    summarise,
    run_checks,
    torch_gradient,
)


def test_hand_calculated_gradient_of_spec_expression():
    # a*b + c = 4, f = 16, df/da = 2*4*b, df/db = 2*4*a, df/dc = 2*4
    value, grad = engine_gradient(spec_expression, (2.0, -3.0, 10.0))

    assert value == 16.0
    assert grad == [-24.0, 16.0, 8.0]


@pytest.mark.parametrize("point", SPEC_ONLY_POINTS.values(), ids=SPEC_ONLY_POINTS.keys())
def test_spec_expression_at_named_points(point):
    value, grad = engine_gradient(spec_expression, point)
    torch_value, torch_grad = torch_gradient(spec_expression, point)
    numeric = numerical_gradient(spec_expression, point)

    assert abs(value - torch_value) < MAX_ALLOWED_DIFFERENCE
    assert max_abs_difference(grad, torch_grad) < MAX_ALLOWED_DIFFERENCE
    assert max_abs_difference(grad, numeric) < MAX_ALLOWED_DIFFERENCE


@pytest.mark.parametrize("name", EXPRESSIONS)
def test_every_expression_at_random_points(name):
    expression = EXPRESSIONS[name]

    for point in random_points().values():
        _, grad = engine_gradient(expression, point)
        _, torch_grad = torch_gradient(expression, point)

        assert max_abs_difference(grad, torch_grad) < MAX_ALLOWED_DIFFERENCE
        assert max_abs_difference(grad, numerical_gradient(expression, point)) < MAX_ALLOWED_DIFFERENCE


def test_extra_expressions_reach_the_inactive_relu_branch():
    # This is why the extra expressions exist. The spec expression cannot do it.
    from value import Value

    relu_inputs = []
    for a, b, c in random_points().values():
        relu_inputs.append((Value(a) - Value(c)).data)

    assert min(relu_inputs) < 0 < max(relu_inputs)


def test_check_detects_a_wrong_gradient():
    # A checker that can never fail proves nothing.
    rows = run_checks()
    rows[0]["max_abs_diff_engine_vs_torch"] = 1.0

    assert summarise(rows)[0]["passed"] is False
