"""Tests for uncertainty-heterogeneity sensitivity analysis."""

import numpy as np
import pytest

from src.simulation.uncertainty_heterogeneity_sensitivity import (
    HeterogeneitySensitivityResult,
    HeterogeneitySensitivityStudy,
    evaluate_condition,
    make_conditions,
    make_observation_model,
    run_sensitivity,
)


def test_five_conditions_exist() -> None:
    conditions = make_conditions()

    assert len(conditions) == 5


def test_condition_names_are_unique() -> None:
    conditions = make_conditions()

    names = [
        condition.name
        for condition in conditions
    ]

    assert len(names) == len(
        set(names)
    )


def test_condition_weights_are_non_negative() -> None:
    for condition in make_conditions():
        assert (
            condition.distance_weight
            >= 0.0
        )

        assert (
            condition.angle_weight
            >= 0.0
        )


def test_observation_model_can_be_created() -> None:
    condition = make_conditions()[0]

    model = make_observation_model(
        condition
    )

    assert model is not None


def test_condition_evaluation_returns_result() -> None:
    condition = make_conditions()[0]

    result = evaluate_condition(
        condition
    )

    assert isinstance(
        result,
        HeterogeneitySensitivityResult,
    )


def test_candidate_indices_are_non_negative() -> None:
    result = evaluate_condition(
        make_conditions()[0]
    )

    assert (
        result.generic_candidate_index
        >= 0
    )

    assert (
        result.task_aware_candidate_index
        >= 0
    )


def test_uncertainty_metrics_are_finite() -> None:
    result = evaluate_condition(
        make_conditions()[0]
    )

    assert np.isfinite(
        result.generic_selected_uncertainty_m
    )

    assert np.isfinite(
        result.task_aware_selected_uncertainty_m
    )

    assert np.isfinite(
        result.uncertainty_range_m
    )

    assert np.isfinite(
        result.uncertainty_coefficient_of_variation
    )


def test_uncertainty_range_is_non_negative() -> None:
    result = evaluate_condition(
        make_conditions()[0]
    )

    assert (
        result.uncertainty_range_m
        >= 0.0
    )


def test_coefficient_of_variation_is_non_negative() -> None:
    result = evaluate_condition(
        make_conditions()[0]
    )

    assert (
        result.uncertainty_coefficient_of_variation
        >= 0.0
    )


def test_alignment_is_bounded() -> None:
    result = evaluate_condition(
        make_conditions()[0]
    )

    assert (
        0.0
        <= result.task_aware_alignment
        <= 1.0
    )


def test_full_study_contains_five_results() -> None:
    study = run_sensitivity()

    assert isinstance(
        study,
        HeterogeneitySensitivityStudy,
    )

    assert len(
        study.results
    ) == 5


def test_result_order_matches_conditions() -> None:
    study = run_sensitivity()

    expected = [
        condition.name
        for condition in make_conditions()
    ]

    actual = [
        result.condition
        for result in study.results
    ]

    assert actual == expected