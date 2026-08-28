"""Tests for strictly separated fair scene strategies."""

import inspect

import numpy as np
import pytest

from src.perception.fair_scene_strategies import (
    GenericSceneScoringConfig,
    GenericSceneViewpointScorer,
    TaskAwareSceneScoringConfig,
    TaskAwareSceneViewpointScorer,
)
from src.simulation.three_strategy_perception import (
    run_fixed_perception,
)
from src.simulation.three_strategy_robustness_benchmark import (
    build_scenario_inputs,
    default_scenarios,
)


def make_state():
    """Build one common noisy initial scene estimate."""

    inputs = build_scenario_inputs(
        default_scenarios()[0]
    )

    initial = run_fixed_perception(
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

    return (
        inputs,
        initial.perception_result,
    )


def test_generic_constructor_has_no_task_information():
    """Generic must be structurally unable to receive the trajectory."""

    signature = inspect.signature(
        GenericSceneViewpointScorer
    )

    forbidden = (
        "task_trajectory",
        "task",
        "true_structures",
        "true_target",
        "target",
        "occluders",
    )

    for name in forbidden:
        assert (
            name
            not in signature.parameters
        )


def test_generic_selection_api_has_no_task_or_truth_information():
    """Generic public selection interface must remain task agnostic."""

    signature = inspect.signature(
        GenericSceneViewpointScorer
        .select_viewpoint
    )

    forbidden = (
        "task_trajectory",
        "task",
        "true_structures",
        "target",
        "occluders",
    )

    for name in forbidden:
        assert (
            name
            not in signature.parameters
        )


def test_task_aware_constructor_requires_trajectory_but_not_truth():
    """Task-Aware may receive the plan but not simulator anatomy."""

    signature = inspect.signature(
        TaskAwareSceneViewpointScorer
    )

    assert (
        "task_trajectory"
        in signature.parameters
    )

    forbidden = (
        "true_structures",
        "true_target",
        "target",
        "occluders",
    )

    for name in forbidden:
        assert (
            name
            not in signature.parameters
        )


def test_negative_weights_rejected():
    """Both strategy configurations must validate weights."""

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        GenericSceneScoringConfig(
            movement_weight=-0.1
        )

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        TaskAwareSceneScoringConfig(
            alignment_weight=-0.1
        )


def test_generic_scene_information_is_uniform_average():
    """Generic must assign equal importance to every estimated structure."""

    inputs, initial = (
        make_state()
    )

    scorer = (
        GenericSceneViewpointScorer(
            observation_model=(
                inputs.observation_model
            ),
            config=(
                GenericSceneScoringConfig(
                    movement_weight=0.17
                )
            ),
        )
    )

    scores = scorer.score_candidates(
        current_pose=(
            inputs.initial_pose
        ),
        candidates=(
            inputs.candidates
        ),
        initial_perception=(
            initial
        ),
    )

    for score in scores:
        assert (
            score.scene_information
            == pytest.approx(
                np.mean(
                    score
                    .information_scores
                )
            )
        )


def test_generic_and_task_share_identical_primitive_information():
    """Only task weighting should distinguish their perception objective."""

    inputs, initial = (
        make_state()
    )

    generic = (
        GenericSceneViewpointScorer(
            observation_model=(
                inputs.observation_model
            ),
            config=(
                GenericSceneScoringConfig(
                    movement_weight=0.17
                )
            ),
        )
    )

    task = (
        TaskAwareSceneViewpointScorer(
            observation_model=(
                inputs.observation_model
            ),
            task_trajectory=(
                inputs.task.trajectory
            ),
            config=(
                TaskAwareSceneScoringConfig(
                    movement_weight=0.20
                )
            ),
        )
    )

    generic_scores = (
        generic.score_candidates(
            current_pose=(
                inputs.initial_pose
            ),
            candidates=(
                inputs.candidates
            ),
            initial_perception=(
                initial
            ),
        )
    )

    task_scores = (
        task.score_candidates(
            current_pose=(
                inputs.initial_pose
            ),
            candidates=(
                inputs.candidates
            ),
            initial_perception=(
                initial
            ),
        )
    )

    assert len(
        generic_scores
    ) == len(
        task_scores
    )

    for generic_score, task_score in zip(
        generic_scores,
        task_scores,
    ):
        assert (
            generic_score.candidate
            is task_score.candidate
        )

        assert np.allclose(
            generic_score
            .predicted_sigmas,
            task_score
            .predicted_sigmas,
        )

        assert np.allclose(
            generic_score
            .information_scores,
            task_score
            .information_scores,
        )

        assert (
            generic_score
            .movement_cost
            == pytest.approx(
                task_score
                .movement_cost
            )
        )

        assert (
            generic_score
            .normalised_movement_cost
            == pytest.approx(
                task_score
                .normalised_movement_cost
            )
        )


def test_both_strategies_select_from_identical_candidate_objects():
    """Both selectors must return supplied candidate instances."""

    inputs, initial = (
        make_state()
    )

    generic = (
        GenericSceneViewpointScorer(
            observation_model=(
                inputs.observation_model
            ),
            config=(
                GenericSceneScoringConfig(
                    movement_weight=0.17
                )
            ),
        )
    )

    task = (
        TaskAwareSceneViewpointScorer(
            observation_model=(
                inputs.observation_model
            ),
            task_trajectory=(
                inputs.task.trajectory
            ),
            config=(
                TaskAwareSceneScoringConfig(
                    movement_weight=0.20
                )
            ),
        )
    )

    generic_selection = (
        generic.select_viewpoint(
            current_pose=(
                inputs.initial_pose
            ),
            candidates=(
                inputs.candidates
            ),
            initial_perception=(
                initial
            ),
        )
    )

    task_selection = (
        task.select_viewpoint(
            current_pose=(
                inputs.initial_pose
            ),
            candidates=(
                inputs.candidates
            ),
            initial_perception=(
                initial
            ),
        )
    )

    assert any(
        generic_selection
        .selected
        .candidate
        is candidate
        for candidate
        in inputs.candidates
    )

    assert any(
        task_selection
        .selected
        .candidate
        is candidate
        for candidate
        in inputs.candidates
    )


def test_independent_movement_weights_are_supported():
    """Generic and Task-Aware must permit separately frozen movement weights."""

    generic = (
        GenericSceneScoringConfig(
            movement_weight=0.17
        )
    )

    task = (
        TaskAwareSceneScoringConfig(
            movement_weight=0.20
        )
    )

    assert (
        generic.movement_weight
        == pytest.approx(
            0.17
        )
    )

    assert (
        task.movement_weight
        == pytest.approx(
            0.20
        )
    )