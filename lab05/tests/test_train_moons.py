"""Tests for the make_moons network on the Value engine (Task 5)."""

import numpy as np
import pytest

import train_moons
from train_moons import (
    ValueNetwork,
    check_network_gradients,
    floored_log,
    initial_weights,
    prepare_data,
    train_torch,
    train_value,
)
from value import Value


def test_data_is_scaled_with_training_statistics_only():
    X_train, X_test, _, _ = prepare_data()

    assert np.allclose(X_train.mean(axis=0), 0.0)
    assert np.allclose(X_train.std(axis=0), 1.0)
    assert not np.allclose(X_test.mean(axis=0), 0.0)


def test_split_sizes_match_settings():
    X_train, X_test, y_train, y_test = prepare_data()

    assert len(X_train) == len(y_train) == 240
    assert len(X_test) == len(y_test) == 60


def test_architecture_has_33_parameters():
    # (2 * 8 + 8) hidden + (8 + 1) output
    assert len(ValueNetwork(initial_weights(42)).parameters()) == 33


def test_output_is_a_probability():
    p = ValueNetwork(initial_weights(42)).forward([0.5, -0.5])

    assert 0.0 < p.data < 1.0
    assert p.op == "sigmoid"


def test_network_gradients_match_pytorch_and_finite_differences():
    X_train, _, y_train, _ = prepare_data()

    for row in check_network_gradients(X_train[:40], y_train[:40]):
        assert row["max_abs_diff_engine_vs_torch"] < 1e-6
        assert row["max_abs_diff_engine_vs_numeric"] < 1e-6


def test_floored_log_matches_bce_loss_floor():
    assert floored_log(Value(0.0)).data == -100.0
    assert floored_log(Value(1.0)).data == 0.0


def test_short_training_matches_pytorch_step_for_step(monkeypatch):
    monkeypatch.setattr(train_moons, "EPOCHS", 30)
    data = prepare_data()
    weights = initial_weights(42)

    _, engine_history, _ = train_value(weights, *data)
    _, torch_history, _ = train_torch(weights, *data, threads=1)

    assert len(engine_history) == 30
    assert engine_history == pytest.approx(torch_history, abs=1e-10)
    assert engine_history[-1] < engine_history[0]
