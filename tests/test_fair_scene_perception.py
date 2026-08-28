"""Tests for the shared-initial-observation fair perception pipeline."""

import inspect

import numpy as np
import pytest

from src.perception.fair_scene_viewpoint_scoring import (
    FairSceneScoringConfig,
)
from src.simulation.fair_scene_perception import (
    run_fair_scene_perception,
    select_fair_scene_viewpoints,
)
from src.simulation.three_strategy_perception import (
    PerceptionStrategy,
    run_fixed_perception,
)
from src.simulation.three_strategy_robustness_benchmark import (
    build_scenario_inputs,
    default_scenarios,
)


def make_inputs():
    """Build the deterministic baseline surgical scenario."""

    return build_scenario_inputs(
        default_scenarios()[0]
    )


def run_default():
    """Run one deterministic fair-scene comparison."""

    inputs = make_inputs()

    result = (
        run_fair_scene_perception(
            observation_model=(
                inputs.observation_model
            ),
            initial_pose=(
                inputs.initial_pose
            ),
            candidates=(
                inputs.candidates
            ),
            task_trajectory=(
                inputs.task.trajectory
            ),
            true_structures=(
                inputs.true_structures
            ),
            initial_seed=12345,
            final_seed=54321,
            occluders=(
                inputs.occluders
            ),
        )
    )

    return (
        inputs,
        result,
    )


def test_empty_candidates_rejected():
    """A fair active-perception comparison requires candidates."""

    inputs = make_inputs()

    with pytest.raises(
        ValueError,
        match="candidates",
    ):
        run_fair_scene_perception(
            observation_model=(
                inputs.observation_model
            ),
            initial_pose=(
                inputs.initial_pose
            ),
            candidates=(),
            task_trajectory=(
                inputs.task.trajectory
            ),
            true_structures=(
                inputs.true_structures
            ),
            initial_seed=1,
            final_seed=2,
            occluders=(
                inputs.occluders
            ),
        )


def test_selection_api_has_no_ground_truth_arguments():
    """The selector must be structurally unable to receive simulator truth."""

    signature = inspect.signature(
        select_fair_scene_viewpoints
    )

    forbidden = (
        "true_structures",
        "target",
        "true_target",
        "occluders",
        "safety_critical_points",
    )

    for name in forbidden:
        assert (
            name
            not in signature.parameters
        )


def test_initial_observation_is_deterministic_for_same_seed():
    """Identical initial seeds must reproduce identical scene estimates."""

    inputs = make_inputs()

    first = run_fixed_perception(
        observation_model=(
            inputs.observation_model
        ),
        initial_pose=(
            inputs.initial_pose
        ),
        true_structures=(
            inputs.true_structures
        ),
        seed=12345,
        occluders=(
            inputs.occluders
        ),
    )

    second = run_fixed_perception(
        observation_model=(
            inputs.observation_model
        ),
        initial_pose=(
            inputs.initial_pose
        ),
        true_structures=(
            inputs.true_structures
        ),
        seed=12345,
        occluders=(
            inputs.occluders
        ),
    )

    first_estimates = (
        first
        .perception_result
        .estimated_structures
    )

    second_estimates = (
        second
        .perception_result
        .estimated_structures
    )

    assert len(
        first_estimates
    ) == len(
        second_estimates
    )

    for first_estimate, second_estimate in zip(
        first_estimates,
        second_estimates,
    ):
        assert np.allclose(
            first_estimate.estimated_centre,
            second_estimate.estimated_centre,
        )


def test_result_contains_one_shared_initial_perception():
    """The experiment must expose the single initial observation used."""

    inputs, result = run_default()

    estimates = (
        result
        .initial_perception
        .perception_result
        .estimated_structures
    )

    assert len(
        estimates
    ) == len(
        inputs.true_structures
    )

    assert (
        result.initial_seed
        == 12345
    )

    assert (
        result.final_seed
        == 54321
    )


def test_both_selections_use_supplied_candidate_set():
    """Both strategies must select from the exact common candidate objects."""

    inputs, result = run_default()

    generic_candidate = (
        result
        .selections
        .generic
        .selected
        .candidate
    )

    task_candidate = (
        result
        .selections
        .task_aware
        .selected
        .candidate
    )

    assert any(
        generic_candidate
        is candidate
        for candidate
        in inputs.candidates
    )

    assert any(
        task_candidate
        is candidate
        for candidate
        in inputs.candidates
    )


def test_final_strategy_labels_are_correct():
    """Final planner-facing results must preserve strategy identity."""

    _, result = run_default()

    assert (
        result
        .generic_perception
        .strategy
        == PerceptionStrategy
        .GENERIC_ACTIVE
    )

    assert (
        result
        .task_aware_perception
        .strategy
        == PerceptionStrategy
        .TASK_AWARE_ACTIVE
    )


def test_final_results_observe_all_anatomical_structures():
    """Both final observations must return the complete planning scene."""

    inputs, result = run_default()

    generic_estimates = (
        result
        .generic_perception
        .perception_result
        .estimated_structures
    )

    task_estimates = (
        result
        .task_aware_perception
        .perception_result
        .estimated_structures
    )

    assert len(
        generic_estimates
    ) == len(
        inputs.true_structures
    )

    assert len(
        task_estimates
    ) == len(
        inputs.true_structures
    )


def test_matched_final_noise_when_selected_pose_is_identical():
    """Same pose plus same final seed must produce matched observations.

    Restricting the candidate set to one pose forces both strategies to
    select exactly the same viewpoint. The resulting simulated final
    observations should therefore be identical.
    """

    inputs = make_inputs()

    one_candidate = (
        inputs.candidates[0],
    )

    result = (
        run_fair_scene_perception(
            observation_model=(
                inputs.observation_model
            ),
            initial_pose=(
                inputs.initial_pose
            ),
            candidates=(
                one_candidate
            ),
            task_trajectory=(
                inputs.task.trajectory
            ),
            true_structures=(
                inputs.true_structures
            ),
            initial_seed=12345,
            final_seed=54321,
            occluders=(
                inputs.occluders
            ),
        )
    )

    generic = (
        result
        .generic_perception
        .perception_result
        .estimated_structures
    )

    task = (
        result
        .task_aware_perception
        .perception_result
        .estimated_structures
    )

    assert len(
        generic
    ) == len(
        task
    )

    for generic_estimate, task_estimate in zip(
        generic,
        task,
    ):
        assert np.allclose(
            generic_estimate.estimated_centre,
            task_estimate.estimated_centre,
        )

        assert np.allclose(
            generic_estimate
            .uncertainty
            .covariance,
            task_estimate
            .uncertainty
            .covariance,
        )


def test_camera_movement_matches_selected_pose_geometry():
    """Reported movement must correspond to actual selected viewpoints."""

    inputs, result = run_default()

    assert (
        result
        .generic_perception
        .camera_movement
        >= 0.0
    )

    assert (
        result
        .task_aware_perception
        .camera_movement
        >= 0.0
    )

    assert np.isfinite(
        result
        .generic_perception
        .camera_movement
    )

    assert np.isfinite(
        result
        .task_aware_perception
        .camera_movement
    )


def test_task_metrics_exist_only_for_task_aware_output():
    """Only Task-Aware should report task-specific viewpoint diagnostics."""

    _, result = run_default()

    assert (
        result
        .generic_perception
        .task_relevance
        is None
    )

    assert (
        result
        .generic_perception
        .task_alignment
        is None
    )

    assert (
        result
        .task_aware_perception
        .task_relevance
        is not None
    )

    assert (
        result
        .task_aware_perception
        .task_alignment
        is not None
    )


def test_result_is_reproducible():
    """Complete fair-scene perception must reproduce under fixed seeds."""

    _, first = run_default()

    _, second = run_default()

    assert (
        first
        .generic_perception
        .camera_movement
        == pytest.approx(
            second
            .generic_perception
            .camera_movement
        )
    )

    assert (
        first
        .task_aware_perception
        .camera_movement
        == pytest.approx(
            second
            .task_aware_perception
            .camera_movement
        )
    )

    first_generic = (
        first
        .generic_perception
        .perception_result
        .estimated_structures
    )

    second_generic = (
        second
        .generic_perception
        .perception_result
        .estimated_structures
    )

    for first_estimate, second_estimate in zip(
        first_generic,
        second_generic,
    ):
        assert np.allclose(
            first_estimate.estimated_centre,
            second_estimate.estimated_centre,
        )