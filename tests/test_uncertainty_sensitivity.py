"""Tests for perception-uncertainty sensitivity analysis."""

import numpy as np
import pytest

from src.simulation.uncertainty_sensitivity import (
    UncertaintySensitivityResult,
    UncertaintySensitivityStudy,
    default_uncertainty_levels,
    run_sensitivity,
    run_uncertainty_level,
    scale_candidate_uncertainty,
)


def test_default_uncertainty_levels() -> None:
    levels = default_uncertainty_levels()

    assert levels == (
        0.001,
        0.002,
        0.005,
        0.010,
        0.020,
        0.030,
    )


def test_uncertainty_levels_are_positive() -> None:
    assert all(
        level > 0.0
        for level in default_uncertainty_levels()
    )


def test_scaling_requires_positive_sigma() -> None:
    values = np.array(
        [0.001, 0.002, 0.003],
        dtype=float,
    )

    with pytest.raises(ValueError):
        scale_candidate_uncertainty(
            values,
            0.0,
        )


def test_scaling_preserves_shape() -> None:
    values = np.array(
        [0.001, 0.002, 0.003],
        dtype=float,
    )

    result = scale_candidate_uncertainty(
        values,
        0.010,
    )

    assert result.shape == values.shape


def test_scaling_returns_finite_values() -> None:
    values = np.array(
        [0.001, 0.002, 0.003],
        dtype=float,
    )

    result = scale_candidate_uncertainty(
        values,
        0.010,
    )

    assert np.all(
        np.isfinite(result)
    )


def test_scaling_returns_positive_values() -> None:
    values = np.array(
        [0.001, 0.002, 0.003],
        dtype=float,
    )

    result = scale_candidate_uncertainty(
        values,
        0.010,
    )

    assert np.all(
        result > 0.0
    )


def test_single_uncertainty_level_returns_result() -> None:
    result = run_uncertainty_level(
        sigma_m=0.005
    )

    assert isinstance(
        result,
        UncertaintySensitivityResult,
    )

    assert result.sigma_m == 0.005


def test_result_values_are_finite() -> None:
    result = run_uncertainty_level(
        sigma_m=0.005
    )

    assert np.isfinite(
        result.generic_selected_uncertainty_m
    )

    assert np.isfinite(
        result.task_aware_selected_uncertainty_m
    )

    assert np.isfinite(
        result.task_aware_alignment
    )


def test_alignment_is_bounded() -> None:
    result = run_uncertainty_level(
        sigma_m=0.005
    )

    assert (
        0.0
        <= result.task_aware_alignment
        <= 1.0
    )


def test_negative_sigma_is_rejected() -> None:
    with pytest.raises(ValueError):
        run_uncertainty_level(
            sigma_m=-0.001
        )


def test_empty_levels_are_rejected() -> None:
    with pytest.raises(ValueError):
        run_sensitivity(
            levels=()
        )


def test_full_sensitivity_contains_six_results() -> None:
    study = run_sensitivity()

    assert isinstance(
        study,
        UncertaintySensitivityStudy,
    )

    assert len(
        study.results
    ) == 6


def test_results_are_in_level_order() -> None:
    study = run_sensitivity()

    levels = [
        result.sigma_m
        for result in study.results
    ]

    assert levels == list(
        default_uncertainty_levels()
    )