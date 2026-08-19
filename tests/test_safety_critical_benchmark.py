"""Tests for perception-to-planning safety evaluation."""

import numpy as np
import pytest

from src.geometry.workspace import SphericalStructure
from src.simulation.safety_critical_benchmark import (
    PathSafetyResult,
    PlanningBenchmarkConfig,
    StrategyConfig,
    estimate_structure,
    evaluate_path_against_truth,
    make_goal_configuration,
    make_instrument,
    make_start_configuration,
    make_true_structure,
    run_benchmark,
)


def test_strategy_requires_positive_sigma() -> None:
    with pytest.raises(ValueError):
        StrategyConfig(
            name="invalid",
            predicted_sigma=0.0,
        )

    with pytest.raises(ValueError):
        StrategyConfig(
            name="invalid",
            predicted_sigma=-0.01,
        )


def test_default_planning_config_is_valid() -> None:
    config = PlanningBenchmarkConfig()

    assert config.trial_count > 0
    assert config.max_iterations > 0
    assert config.step_size > 0.0
    assert config.edge_resolution >= 2


def test_instrument_is_created() -> None:
    instrument = make_instrument()

    assert instrument.rcm_position.shape == (
        3,
    )


def test_start_configuration_has_correct_shape() -> None:
    q = make_start_configuration()

    assert q.shape == (4,)
    assert np.all(np.isfinite(q))


def test_goal_configuration_has_correct_shape() -> None:
    q = make_goal_configuration()

    assert q.shape == (4,)
    assert np.all(np.isfinite(q))


def test_start_and_goal_are_distinct() -> None:
    start = make_start_configuration()
    goal = make_goal_configuration()

    assert not np.allclose(
        start,
        goal,
    )


def test_true_structure_is_valid() -> None:
    structure = make_true_structure()

    assert structure.centre.shape == (
        3,
    )

    assert structure.physical_radius > 0.0

    assert structure.safety_margin >= 0.0


def test_estimated_structure_preserves_geometry() -> None:
    true_structure = (
        make_true_structure()
    )

    rng = np.random.default_rng(
        123
    )

    estimated = estimate_structure(
        true_structure=true_structure,
        predicted_sigma=0.002,
        rng=rng,
    )

    assert (
        estimated.physical_radius
        == true_structure.physical_radius
    )

    assert (
        estimated.safety_margin
        == true_structure.safety_margin
    )


def test_estimated_structure_is_deterministic_for_same_seed() -> None:
    true_structure = (
        make_true_structure()
    )

    first = estimate_structure(
        true_structure=true_structure,
        predicted_sigma=0.002,
        rng=np.random.default_rng(
            42
        ),
    )

    second = estimate_structure(
        true_structure=true_structure,
        predicted_sigma=0.002,
        rng=np.random.default_rng(
            42
        ),
    )

    np.testing.assert_allclose(
        first.centre,
        second.centre,
    )


def test_larger_sigma_produces_larger_possible_error() -> None:
    true_structure = (
        make_true_structure()
    )

    low = estimate_structure(
        true_structure=true_structure,
        predicted_sigma=0.002,
        rng=np.random.default_rng(
            1
        ),
    )

    high = estimate_structure(
        true_structure=true_structure,
        predicted_sigma=0.030,
        rng=np.random.default_rng(
            1
        ),
    )

    low_error = np.linalg.norm(
        low.centre
        - true_structure.centre
    )

    high_error = np.linalg.norm(
        high.centre
        - true_structure.centre
    )

    assert high_error > low_error


def test_path_evaluation_requires_two_points() -> None:
    instrument = make_instrument()

    path = np.zeros(
        (1, 4),
        dtype=float,
    )

    with pytest.raises(ValueError):
        evaluate_path_against_truth(
            instrument=instrument,
            path=path,
            true_structures=(
                make_true_structure(),
            ),
            instrument_radius=0.005,
            proximal_length=0.10,
            resolution=10,
        )


def test_path_evaluation_rejects_invalid_shape() -> None:
    instrument = make_instrument()

    path = np.zeros(
        (2, 3),
        dtype=float,
    )

    with pytest.raises(ValueError):
        evaluate_path_against_truth(
            instrument=instrument,
            path=path,
            true_structures=(
                make_true_structure(),
            ),
            instrument_radius=0.005,
            proximal_length=0.10,
            resolution=10,
        )


def test_path_evaluation_rejects_invalid_resolution() -> None:
    instrument = make_instrument()

    path = np.vstack(
        [
            make_start_configuration(),
            make_goal_configuration(),
        ]
    )

    with pytest.raises(ValueError):
        evaluate_path_against_truth(
            instrument=instrument,
            path=path,
            true_structures=(
                make_true_structure(),
            ),
            instrument_radius=0.005,
            proximal_length=0.10,
            resolution=1,
        )


def test_path_evaluation_returns_result() -> None:
    instrument = make_instrument()

    path = np.vstack(
        [
            make_start_configuration(),
            make_start_configuration(),
        ]
    )

    # The path is deliberately invalid because consecutive
    # configurations coincide, so use a small valid motion instead.
    path = np.vstack(
        [
            make_start_configuration(),
            np.array(
                [
                    np.deg2rad(-20.0),
                    np.deg2rad(8.6),
                    0.22,
                    0.0,
                ]
            ),
        ]
    )

    result = evaluate_path_against_truth(
        instrument=instrument,
        path=path,
        true_structures=(
            make_true_structure(),
        ),
        instrument_radius=0.005,
        proximal_length=0.10,
        resolution=10,
    )

    assert isinstance(
        result,
        PathSafetyResult,
    )

    assert result.evaluated


def test_path_clearance_is_finite() -> None:
    instrument = make_instrument()

    path = np.vstack(
        [
            make_start_configuration(),
            np.array(
                [
                    np.deg2rad(-20.0),
                    np.deg2rad(8.6),
                    0.22,
                    0.0,
                ]
            ),
        ]
    )

    result = evaluate_path_against_truth(
        instrument=instrument,
        path=path,
        true_structures=(
            make_true_structure(),
        ),
        instrument_radius=0.005,
        proximal_length=0.10,
        resolution=10,
    )

    assert np.isfinite(
        result.minimum_surface_clearance
    )

    assert np.isfinite(
        result.minimum_safety_clearance
    )


def test_benchmark_can_be_run_with_small_trial_count() -> None:
    config = PlanningBenchmarkConfig(
        trial_count=1,
        max_iterations=500,
    )

    result = run_benchmark(
        config
    )

    assert result.fixed_view.trials == 1
    assert (
        result.generic_active_perception.trials
        == 1
    )
    assert (
        result.task_aware_active_perception.trials
        == 1
    )


def test_benchmark_returns_three_strategies() -> None:
    config = PlanningBenchmarkConfig(
        trial_count=1,
        max_iterations=500,
    )

    result = run_benchmark(
        config
    )

    assert (
        result.fixed_view.strategy
        == "Fixed view"
    )

    assert (
        result.generic_active_perception.strategy
        == "Generic active perception"
    )

    assert (
        result.task_aware_active_perception.strategy
        == "Task-aware active perception"
    )


def test_strategy_rates_are_bounded() -> None:
    config = PlanningBenchmarkConfig(
        trial_count=1,
        max_iterations=500,
    )

    result = run_benchmark(
        config
    )

    summaries = (
        result.fixed_view,
        result.generic_active_perception,
        result.task_aware_active_perception,
    )

    for summary in summaries:
        assert (
            0.0
            <= summary.planning_success_rate_percent
            <= 100.0
        )

        assert (
            0.0
            <= summary.ground_truth_safe_rate_percent
            <= 100.0
        )

        assert (
            0.0
            <= summary.collision_rate_percent
            <= 100.0
        )

        assert (
            0.0
            <= summary.unsafe_path_rate_percent
            <= 100.0
        )