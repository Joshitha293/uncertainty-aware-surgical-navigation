"""Tests for the final three-strategy end-to-end experiment."""

import numpy as np
import pytest

from src.simulation.statistical_benchmark import (
    make_true_structures,
)
from src.simulation.three_strategy_perception import (
    PerceptionStrategy,
)
from src.simulation.three_strategy_trial import (
    ThreeStrategyTrialConfig,
    build_trial_inputs,
    make_trial_task,
    run_three_strategy_perception,
    run_three_strategy_trial,
)


def test_default_trial_config_is_valid():
    config = ThreeStrategyTrialConfig()

    assert config.trial == 0
    assert config.perception_seed >= 0
    assert config.planner_seed >= 0
    assert config.sigma_multiplier >= 0.0
    assert config.instrument_radius > 0.0
    assert config.proximal_length > 0.0


def test_invalid_trial_config_is_rejected():
    with pytest.raises(
        ValueError,
        match="trial must be non-negative",
    ):
        ThreeStrategyTrialConfig(
            trial=-1
        )

    with pytest.raises(
        ValueError,
        match="sigma_multiplier",
    ):
        ThreeStrategyTrialConfig(
            sigma_multiplier=-1.0
        )


def test_trial_inputs_use_validated_hidden_anatomy():
    inputs = build_trial_inputs()

    expected = make_true_structures()

    assert len(inputs.true_structures) == len(
        expected
    )

    for actual, reference in zip(
        inputs.true_structures,
        expected,
    ):
        assert np.allclose(
            actual.centre,
            reference.centre,
        )

        assert actual.physical_radius == pytest.approx(
            reference.physical_radius
        )

        assert actual.safety_margin == pytest.approx(
            reference.safety_margin
        )

    assert np.allclose(
        inputs.target.centre,
        expected[0].centre,
    )


def test_task_is_derived_from_navigation_problem():
    inputs = build_trial_inputs()

    task = make_trial_task(
        instrument=inputs.instrument,
        start_q=inputs.start_q,
        goal_q=inputs.goal_q,
        true_structures=(
            inputs.true_structures
        ),
    )

    assert task.trajectory.ndim == 2
    assert task.trajectory.shape[1] == 3
    assert len(task.trajectory) >= 2

    expected_start = (
        inputs.instrument.forward_position(
            inputs.start_q
        )
    )

    expected_goal = (
        inputs.instrument.forward_position(
            inputs.goal_q
        )
    )

    assert np.allclose(
        task.trajectory[0],
        expected_start,
    )

    assert np.allclose(
        task.trajectory[-1],
        expected_goal,
    )

    expected_critical_points = np.asarray(
        [
            structure.centre
            for structure
            in inputs.true_structures
        ]
    )

    assert np.allclose(
        task.safety_critical_points,
        expected_critical_points,
    )


def test_perception_stage_returns_all_three_strategies():
    config = ThreeStrategyTrialConfig()

    results = (
        run_three_strategy_perception(
            config
        )
    )

    assert len(results) == 3

    assert results[0].strategy is (
        PerceptionStrategy.FIXED
    )

    assert results[1].strategy is (
        PerceptionStrategy.GENERIC_ACTIVE
    )

    assert results[2].strategy is (
        PerceptionStrategy.TASK_AWARE_ACTIVE
    )

    for result in results:
        assert (
            len(
                result
                .perception_result
                .estimated_structures
            )
            == 2
        )

        assert result.mean_predicted_sigma > 0.0
        assert result.mean_localisation_error >= 0.0


def test_matched_perception_is_reproducible():
    config = ThreeStrategyTrialConfig(
        perception_seed=12345
    )

    first = (
        run_three_strategy_perception(
            config
        )
    )

    second = (
        run_three_strategy_perception(
            config
        )
    )

    for first_result, second_result in zip(
        first,
        second,
    ):
        assert np.allclose(
            first_result.selected_pose.position,
            second_result.selected_pose.position,
        )

        assert np.allclose(
            first_result
            .perception_result
            .localisation_errors,
            second_result
            .perception_result
            .localisation_errors,
        )

        first_centres = np.asarray(
            [
                item.estimated_centre
                for item
                in first_result
                .perception_result
                .estimated_structures
            ]
        )

        second_centres = np.asarray(
            [
                item.estimated_centre
                for item
                in second_result
                .perception_result
                .estimated_structures
            ]
        )

        assert np.allclose(
            first_centres,
            second_centres,
        )


def test_end_to_end_trial_returns_three_navigation_results():
    config = ThreeStrategyTrialConfig(
        trial=3,
        perception_seed=321,
        planner_seed=4321,
    )

    result = run_three_strategy_trial(
        config
    )

    assert len(result.results) == 3

    expected_strategies = (
        "fixed",
        "generic_active",
        "task_aware_active",
    )

    assert tuple(
        item.strategy
        for item in result.results
    ) == expected_strategies

    for navigation in result.results:
        assert navigation.trial == 3

        assert (
            navigation.planner_result.method
            == navigation.strategy
        )

        assert (
            len(
                navigation
                .planning_perception
                .structures
            )
            == 2
        )

        assert (
            navigation.mean_planning_safety_margin
            >= 0.0
        )

        assert isinstance(
            navigation.planning_success,
            bool,
        )

        assert isinstance(
            navigation.collision_against_truth,
            bool,
        )

        assert isinstance(
            navigation.safety_violation_against_truth,
            bool,
        )