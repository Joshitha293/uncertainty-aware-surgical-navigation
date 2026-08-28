"""Tests for validation-only Generic movement-budget calibration."""

import dataclasses

import numpy as np
import pytest

from src.simulation.phase1_scenario_splits import (
    build_phase1_splits,
)
from src.simulation.phase1_validation_budget_calibration import (
    DEFAULT_GENERIC_WEIGHT_GRID,
    GenericWeightCandidate,
    ValidationBudgetTrial,
    _validate_weights,
    collect_validation_budget_trials,
    evaluate_generic_weight_grid,
    select_generic_weight,
)


def test_default_grid_contains_development_and_task_scale_weights():
    """Search grid must include key previously used movement penalties."""

    assert any(
        np.isclose(
            weight,
            0.170,
        )
        for weight
        in DEFAULT_GENERIC_WEIGHT_GRID
    )

    assert any(
        np.isclose(
            weight,
            0.200,
        )
        for weight
        in DEFAULT_GENERIC_WEIGHT_GRID
    )


def test_negative_grid_weight_rejected():
    """Calibration weights must be non-negative."""

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        _validate_weights(
            (
                0.1,
                -0.1,
            )
        )


def test_small_validation_collection_has_expected_count():
    """Every scenario/repetition pair must produce one calibration trial."""

    splits = (
        build_phase1_splits()
    )

    scenarios = tuple(
        splits.validation[
            :2
        ]
    )

    trials = (
        collect_validation_budget_trials(
            scenarios=scenarios,
            repetitions=2,
        )
    )

    assert len(
        trials
    ) == 4


def test_calibration_trial_contains_no_outcome_metrics():
    """Weight selection data must exclude downstream performance outcomes."""

    field_names = {
        field.name
        for field
        in dataclasses.fields(
            ValidationBudgetTrial
        )
    }

    forbidden = (
        "localisation_error",
        "predicted_sigma",
        "planning_success",
        "safe_navigation_success",
        "collision",
        "safety_violation",
    )

    for name in forbidden:
        assert (
            name
            not in field_names
        )


def test_grid_evaluation_returns_one_candidate_per_weight():
    """Every requested Generic weight must be evaluated."""

    trial = (
        ValidationBudgetTrial(
            scenario_id=1000,
            scenario_name="synthetic",
            repetition=0,
            initial_seed=1,
            task_camera_movement=0.10,
            generic_scene_information=(
                1.0,
                0.95,
            ),
            generic_normalised_movement=(
                1.0,
                0.0,
            ),
            generic_camera_movements=(
                0.12,
                0.05,
            ),
        ),
    )

    weights = (
        0.0,
        0.1,
        0.5,
    )

    candidates = (
        evaluate_generic_weight_grid(
            trials=trial,
            weights=weights,
        )
    )

    assert len(
        candidates
    ) == len(
        weights
    )


def test_generic_weight_selection_uses_smallest_budget_gap():
    """Closest movement budget must win regardless of weight magnitude."""

    candidates = (
        GenericWeightCandidate(
            movement_weight=0.10,
            mean_generic_camera_movement=0.12,
            target_task_camera_movement=0.10,
            absolute_budget_gap=0.02,
            relative_budget_gap=0.20,
        ),
        GenericWeightCandidate(
            movement_weight=0.30,
            mean_generic_camera_movement=0.101,
            target_task_camera_movement=0.10,
            absolute_budget_gap=0.001,
            relative_budget_gap=0.01,
        ),
        GenericWeightCandidate(
            movement_weight=0.80,
            mean_generic_camera_movement=0.06,
            target_task_camera_movement=0.10,
            absolute_budget_gap=0.04,
            relative_budget_gap=0.40,
        ),
    )

    selected = (
        select_generic_weight(
            candidates
        )
    )

    assert (
        selected.movement_weight
        == pytest.approx(
            0.30
        )
    )


def test_real_validation_grid_evaluation_is_finite():
    """Real validation data with a non-zero task budget must be finite."""

    splits = (
        build_phase1_splits()
    )

    trials = (
        collect_validation_budget_trials(
            scenarios=(
                splits.validation
            ),
            repetitions=1,
        )
    )

    target_task_movement = float(
        np.mean(
            [
                trial.task_camera_movement
                for trial
                in trials
            ]
        )
    )

    assert (
        target_task_movement
        > 0.0
    )

    candidates = (
        evaluate_generic_weight_grid(
            trials=trials,
            weights=(
                0.17,
                0.25,
                0.40,
            ),
        )
    )

    assert len(
        candidates
    ) == 3

    for candidate in candidates:
        assert np.isfinite(
            candidate
            .mean_generic_camera_movement
        )

        assert np.isfinite(
            candidate
            .absolute_budget_gap
        )

        assert np.isfinite(
            candidate
            .relative_budget_gap
        )

        assert (
            candidate
            .mean_generic_camera_movement
            >= 0.0
        )

        assert (
            candidate
            .absolute_budget_gap
            >= 0.0
        )

        assert (
            candidate
            .relative_budget_gap
            >= 0.0
        )