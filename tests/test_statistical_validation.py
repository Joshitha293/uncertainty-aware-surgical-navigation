"""Tests for statistical validation utilities."""

import numpy as np
import pytest

from src.simulation.statistical_validation import (
    StatisticalValidationResult,
    TrialOutcome,
    build_validation_result,
    exact_two_sided_binomial_p_value,
)


def make_trial(
    trial: int,
    safe: bool,
    clearance: float,
) -> TrialOutcome:
    """Create a simple synthetic trial outcome."""

    return TrialOutcome(
        trial=trial,
        planning_success=True,
        safe=safe,
        collision=not safe,
        safety_margin_violation=False,
        minimum_surface_clearance=clearance,
        minimum_safety_clearance=clearance,
        path_cost=1.0,
    )


def test_exact_binomial_p_value_is_bounded() -> None:
    value = exact_two_sided_binomial_p_value(
        successes=5,
        trials=10,
    )

    assert 0.0 <= value <= 1.0


def test_balanced_binomial_has_high_p_value() -> None:
    value = exact_two_sided_binomial_p_value(
        successes=5,
        trials=10,
    )

    assert value == pytest.approx(
        1.0
    )


def test_extreme_binomial_has_low_p_value() -> None:
    value = exact_two_sided_binomial_p_value(
        successes=10,
        trials=10,
    )

    assert value == pytest.approx(
        0.001953125
    )


def test_validation_result_can_be_built() -> None:
    generic = tuple(
        make_trial(
            trial=index,
            safe=False,
            clearance=0.001,
        )
        for index in range(10)
    )

    task_aware = tuple(
        make_trial(
            trial=index,
            safe=True,
            clearance=0.003,
        )
        for index in range(10)
    )

    result = build_validation_result(
        generic=generic,
        task_aware=task_aware,
    )

    assert isinstance(
        result,
        StatisticalValidationResult,
    )


def test_clearance_difference_is_positive() -> None:
    generic = tuple(
        make_trial(
            trial=index,
            safe=False,
            clearance=0.001,
        )
        for index in range(10)
    )

    task_aware = tuple(
        make_trial(
            trial=index,
            safe=True,
            clearance=0.003,
        )
        for index in range(10)
    )

    result = build_validation_result(
        generic=generic,
        task_aware=task_aware,
    )

    assert (
        result.paired_clearance.mean_difference
        == pytest.approx(0.002)
    )


def test_bootstrap_interval_is_ordered() -> None:
    generic = tuple(
        make_trial(
            trial=index,
            safe=False,
            clearance=0.001,
        )
        for index in range(10)
    )

    task_aware = tuple(
        make_trial(
            trial=index,
            safe=True,
            clearance=0.003,
        )
        for index in range(10)
    )

    result = build_validation_result(
        generic=generic,
        task_aware=task_aware,
    )

    clearance = result.paired_clearance

    assert (
        clearance.bootstrap_ci_lower
        <= clearance.bootstrap_ci_upper
    )


def test_cohens_dz_handles_zero_variance() -> None:
    generic = tuple(
        make_trial(
            trial=index,
            safe=False,
            clearance=0.001,
        )
        for index in range(10)
    )

    task_aware = tuple(
        make_trial(
            trial=index,
            safe=True,
            clearance=0.003,
        )
        for index in range(10)
    )

    result = build_validation_result(
        generic=generic,
        task_aware=task_aware,
    )

    dz = (
        result.paired_clearance.cohens_dz
    )

    assert (
        np.isinf(dz)
        or np.isfinite(dz)
    )


def test_cohens_dz_is_finite_for_variable_data() -> None:
    generic = tuple(
        make_trial(
            trial=index,
            safe=False,
            clearance=(
                0.001
                + index * 0.0001
            ),
        )
        for index in range(10)
    )

    task_aware = tuple(
        make_trial(
            trial=index,
            safe=True,
            clearance=(
                0.003
                + index * 0.0002
            ),
        )
        for index in range(10)
    )

    result = build_validation_result(
        generic=generic,
        task_aware=task_aware,
    )

    dz = (
        result.paired_clearance.cohens_dz
    )

    assert np.isfinite(dz)


def test_safety_rates_are_calculated() -> None:
    generic = tuple(
        make_trial(
            trial=index,
            safe=False,
            clearance=0.001,
        )
        for index in range(10)
    )

    task_aware = tuple(
        make_trial(
            trial=index,
            safe=True,
            clearance=0.003,
        )
        for index in range(10)
    )

    result = build_validation_result(
        generic=generic,
        task_aware=task_aware,
    )

    assert (
        result.paired_safety
        .generic_safe_rate_percent
        == pytest.approx(0.0)
    )

    assert (
        result.paired_safety
        .task_aware_safe_rate_percent
        == pytest.approx(100.0)
    )


def test_paired_safety_p_value_is_bounded() -> None:
    generic = tuple(
        make_trial(
            trial=index,
            safe=False,
            clearance=0.001,
        )
        for index in range(10)
    )

    task_aware = tuple(
        make_trial(
            trial=index,
            safe=True,
            clearance=0.003,
        )
        for index in range(10)
    )

    result = build_validation_result(
        generic=generic,
        task_aware=task_aware,
    )

    p_value = (
        result.paired_safety
        .exact_two_sided_p_value
    )

    assert (
        0.0
        <= p_value
        <= 1.0
    )