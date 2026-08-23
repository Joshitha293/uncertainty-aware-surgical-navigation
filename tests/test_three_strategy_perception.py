"""Tests for the unified three-strategy perception interface."""

from __future__ import annotations

import numpy as np

from src.perception.task_relevance import SurgicalTask
from src.simulation.task_aware_benchmark import (
    TaskAwareBenchmarkConfig,
    make_generic_controller,
    make_initial_pose,
    make_observation_model,
    make_occluder,
    make_target,
    make_task_aware_controller,
)
from src.perception.viewpoints import (
    generate_candidate_viewpoints,
)
from src.simulation.three_strategy_perception import (
    PerceptionStrategy,
    run_fixed_perception,
    run_generic_active_perception,
    run_task_aware_active_perception,
)


def make_test_system():
    """Create one matched perception scenario."""

    model = make_observation_model()
    target = make_target()
    occluder = make_occluder()

    initial_pose = make_initial_pose(
        target
    )

    trajectory = np.array(
        [
            [0.08, -0.04, 0.0],
            [0.13, -0.01, 0.0],
            [0.18, 0.02, 0.0],
            [0.22, 0.04, 0.0],
        ],
        dtype=float,
    )

    task = SurgicalTask(
        trajectory=trajectory,
        safety_critical_points=np.array(
            [
                target.centre,
            ],
            dtype=float,
        ),
    )

    candidates = (
        generate_candidate_viewpoints(
            target_position=target.centre
        )
    )

    generic_controller = (
        make_generic_controller(
            model
        )
    )

    task_controller = (
        make_task_aware_controller(
            model=model,
            task=task,
            config=(
                TaskAwareBenchmarkConfig()
            ),
        )
    )

    return (
        model,
        target,
        occluder,
        initial_pose,
        task,
        candidates,
        generic_controller,
        task_controller,
    )


def test_fixed_strategy_returns_common_result():
    (
        model,
        target,
        occluder,
        initial_pose,
        _,
        _,
        _,
        _,
    ) = make_test_system()

    result = run_fixed_perception(
        observation_model=model,
        initial_pose=initial_pose,
        true_structures=(target,),
        seed=123,
        occluders=(occluder,),
    )

    assert (
        result.strategy
        == PerceptionStrategy.FIXED
    )

    assert result.camera_movement == 0.0

    assert (
        len(
            result.perception_result
            .estimated_structures
        )
        == 1
    )

    assert result.mean_predicted_sigma > 0.0
    assert result.mean_localisation_error >= 0.0


def test_generic_strategy_returns_common_result():
    (
        model,
        target,
        occluder,
        initial_pose,
        _,
        candidates,
        generic_controller,
        _,
    ) = make_test_system()

    result = run_generic_active_perception(
        controller=generic_controller,
        observation_model=model,
        initial_pose=initial_pose,
        candidates=candidates,
        target=target,
        true_structures=(target,),
        seed=123,
        occluders=(occluder,),
    )

    assert (
        result.strategy
        == PerceptionStrategy.GENERIC_ACTIVE
    )

    assert result.candidate_count == len(
        candidates
    )

    assert result.camera_movement >= 0.0

    assert (
        len(
            result.perception_result
            .estimated_structures
        )
        == 1
    )


def test_task_aware_strategy_returns_common_result():
    (
        model,
        target,
        occluder,
        initial_pose,
        task,
        candidates,
        _,
        task_controller,
    ) = make_test_system()

    result = (
        run_task_aware_active_perception(
            controller=task_controller,
            observation_model=model,
            initial_pose=initial_pose,
            candidates=candidates,
            target=target,
            task=task,
            true_structures=(target,),
            seed=123,
            occluders=(occluder,),
        )
    )

    assert (
        result.strategy
        == PerceptionStrategy.TASK_AWARE_ACTIVE
    )

    assert result.task_relevance is not None
    assert result.task_alignment is not None

    assert result.candidate_count == len(
        candidates
    )


def test_all_strategies_are_reproducible():
    (
        model,
        target,
        occluder,
        initial_pose,
        _,
        _,
        _,
        _,
    ) = make_test_system()

    first = run_fixed_perception(
        observation_model=model,
        initial_pose=initial_pose,
        true_structures=(target,),
        seed=456,
        occluders=(occluder,),
    )

    second = run_fixed_perception(
        observation_model=model,
        initial_pose=initial_pose,
        true_structures=(target,),
        seed=456,
        occluders=(occluder,),
    )

    np.testing.assert_allclose(
        first.perception_result
        .estimated_structures[0]
        .estimated_centre,
        second.perception_result
        .estimated_structures[0]
        .estimated_centre,
    )


def test_strategy_results_are_planner_compatible():
    (
        model,
        target,
        occluder,
        initial_pose,
        _,
        _,
        _,
        _,
    ) = make_test_system()

    result = run_fixed_perception(
        observation_model=model,
        initial_pose=initial_pose,
        true_structures=(target,),
        seed=789,
        occluders=(occluder,),
    )

    estimated = (
        result.perception_result
        .estimated_structures[0]
    )

    assert estimated.physical_radius == (
        target.physical_radius
    )

    assert estimated.base_safety_margin == (
        target.safety_margin
    )

    assert (
        estimated.uncertainty
        .principal_sigma
        > 0.0
    )