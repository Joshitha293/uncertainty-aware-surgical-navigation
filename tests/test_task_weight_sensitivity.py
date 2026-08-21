"""Tests for task-weight sensitivity analysis."""

import numpy as np
import pytest

from src.simulation.task_weight_sensitivity import (
    WeightSensitivityResult,
    WeightSensitivityStudy,
    default_task_weights,
    run_sensitivity,
    run_weight,
)


def test_default_weights_are_present() -> None:
    weights = default_task_weights()

    assert weights == (
        0.0,
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,
    )


def test_weights_are_non_negative() -> None:
    assert all(
        weight >= 0.0
        for weight in default_task_weights()
    )


def test_negative_weight_is_rejected() -> None:
    with pytest.raises(ValueError):
        run_weight(
            task_weight=-1.0
        )


def test_single_weight_returns_result() -> None:
    result = run_weight(
        task_weight=1.0
    )

    assert isinstance(
        result,
        WeightSensitivityResult,
    )

    assert result.task_weight == 1.0


def test_result_metrics_are_finite() -> None:
    result = run_weight(
        task_weight=1.0
    )

    assert np.isfinite(
        result.selected_uncertainty
    )

    assert np.isfinite(
        result.selected_alignment
    )

    assert np.isfinite(
        result.selected_generic_score
    )

    assert np.isfinite(
        result.selected_final_score
    )


def test_alignment_is_bounded() -> None:
    result = run_weight(
        task_weight=1.0
    )

    assert (
        0.0
        <= result.selected_alignment
        <= 1.0
    )


def test_weight_zero_reproduces_generic_selection() -> None:
    result = run_weight(
        task_weight=0.0
    )

    assert (
        result.selection_difference_from_generic
        is False
    )


def test_custom_weights_are_supported() -> None:
    study = run_sensitivity(
        weights=(
            0.0,
            2.0,
        )
    )

    assert len(
        study.results
    ) == 2


def test_empty_weights_are_rejected() -> None:
    with pytest.raises(ValueError):
        run_sensitivity(
            weights=()
        )


def test_default_sensitivity_contains_six_results() -> None:
    study = run_sensitivity()

    assert isinstance(
        study,
        WeightSensitivityStudy,
    )

    assert len(
        study.results
    ) == 6


def test_results_are_in_weight_order() -> None:
    study = run_sensitivity()

    weights = [
        result.task_weight
        for result in study.results
    ]

    assert weights == list(
        default_task_weights()
    )