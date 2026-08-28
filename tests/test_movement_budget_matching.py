"""Tests for Phase 1 movement-budget matching."""

import numpy as np
import pytest

from src.simulation.fair_scene_development_sweep import (
    FairSceneDevelopmentSummary,
)
from src.simulation.movement_budget_matching import (
    GenericBudgetCandidate,
    _normalise_lower_is_better,
    match_generic_movement_budget,
    select_generic_budget_match,
    select_task_operating_point,
)
from src.simulation.three_strategy_robustness_benchmark import (
    default_scenarios,
)


def make_summary(
    *,
    weight: float,
    task_movement: float,
    task_sigma: float,
) -> FairSceneDevelopmentSummary:
    """Construct a minimal valid synthetic summary."""

    return FairSceneDevelopmentSummary(
        movement_weight=weight,
        record_count=10,
        mean_generic_camera_movement=0.05,
        mean_task_camera_movement=(
            task_movement
        ),
        mean_camera_movement_difference=0.0,
        mean_generic_localisation_error=0.01,
        mean_task_localisation_error=0.01,
        mean_localisation_error_difference=0.0,
        mean_generic_predicted_sigma=0.01,
        mean_task_predicted_sigma=(
            task_sigma
        ),
        mean_predicted_sigma_difference=0.0,
        mean_generic_alignment=0.9,
        mean_task_alignment=0.95,
        mean_alignment_difference=0.05,
        mean_generic_scene_information=0.8,
        mean_task_scene_information=0.8,
        different_candidate_rate=0.5,
    )


def test_normalisation_maps_extremes_to_zero_and_one():
    values = np.asarray(
        [
            2.0,
            4.0,
            6.0,
        ]
    )

    result = (
        _normalise_lower_is_better(
            values
        )
    )

    assert np.allclose(
        result,
        [
            0.0,
            0.5,
            1.0,
        ],
    )


def test_constant_values_are_stable():
    result = (
        _normalise_lower_is_better(
            np.asarray(
                [
                    3.0,
                    3.0,
                    3.0,
                ]
            )
        )
    )

    assert np.allclose(
        result,
        0.0,
    )


def test_task_operating_point_uses_tradeoff_not_error():
    summaries = (
        make_summary(
            weight=0.0,
            task_movement=0.20,
            task_sigma=0.005,
        ),
        make_summary(
            weight=0.2,
            task_movement=0.10,
            task_sigma=0.010,
        ),
        make_summary(
            weight=1.0,
            task_movement=0.01,
            task_sigma=0.030,
        ),
    )

    selected = (
        select_task_operating_point(
            summaries
        )
    )

    assert (
        selected.movement_weight
        == pytest.approx(
            0.2
        )
    )


def test_generic_budget_match_selects_smallest_gap():
    candidates = (
        GenericBudgetCandidate(
            movement_weight=0.15,
            mean_camera_movement=0.075,
            absolute_budget_gap=0.008,
            relative_budget_gap=0.10,
        ),
        GenericBudgetCandidate(
            movement_weight=0.19,
            mean_camera_movement=0.068,
            absolute_budget_gap=0.001,
            relative_budget_gap=0.015,
        ),
        GenericBudgetCandidate(
            movement_weight=0.25,
            mean_camera_movement=0.055,
            absolute_budget_gap=0.012,
            relative_budget_gap=0.18,
        ),
    )

    selected = (
        select_generic_budget_match(
            candidates
        )
    )

    assert (
        selected.movement_weight
        == pytest.approx(
            0.19
        )
    )


def test_invalid_target_budget_rejected():
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        match_generic_movement_budget(
            target_camera_movement=0.0,
            scenarios=tuple(
                default_scenarios()[
                    :1
                ]
            ),
            repetitions=1,
            generic_weights=(
                0.2,
            ),
        )


def test_small_real_budget_search_returns_all_weights():
    weights = (
        0.15,
        0.20,
        0.25,
    )

    candidates = (
        match_generic_movement_budget(
            target_camera_movement=0.07,
            scenarios=tuple(
                default_scenarios()[
                    :2
                ]
            ),
            repetitions=1,
            generic_weights=weights,
        )
    )

    assert len(
        candidates
    ) == len(
        weights
    )

    for candidate in candidates:
        assert (
            candidate.mean_camera_movement
            >= 0.0
        )

        assert (
            candidate.absolute_budget_gap
            >= 0.0
        )

        assert (
            candidate.relative_budget_gap
            >= 0.0
        )

        assert np.isfinite(
            candidate.mean_camera_movement
        )