"""Tests for scene-wide fair active-perception scoring."""

import inspect

import numpy as np
import pytest

from src.perception.fair_scene_viewpoint_scoring import (
    FairSceneScoringConfig,
    FairSceneViewpointScorer,
    _estimated_scene,
    _normalise_information_matrix,
    _normalise_movement,
    _relevance_weights,
)
from src.perception.task_relevance import (
    TaskRelevanceConfig,
)
from src.simulation.three_strategy_robustness_benchmark import (
    build_scenario_inputs,
    default_scenarios,
)
from src.simulation.three_strategy_perception import (
    run_fixed_perception,
)


def make_initial_state():
    """Generate a shared noisy initial scene estimate."""

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

    scorer = (
        FairSceneViewpointScorer(
            observation_model=(
                inputs.observation_model
            ),
            task_trajectory=(
                inputs.task.trajectory
            ),
        )
    )

    return (
        inputs,
        initial,
        scorer,
    )


def test_negative_scoring_weight_rejected():
    """Scoring weights must be non-negative."""

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        FairSceneScoringConfig(
            movement_weight=-1.0
        )


def test_information_normalised_per_structure():
    """Each anatomical structure must be normalised independently."""

    sigmas = np.asarray(
        [
            [2.0, 10.0],
            [4.0, 20.0],
            [6.0, 30.0],
        ],
        dtype=float,
    )

    information = (
        _normalise_information_matrix(
            sigmas
        )
    )

    expected = np.asarray(
        [
            [1.0, 1.0],
            [0.5, 0.5],
            [0.0, 0.0],
        ],
        dtype=float,
    )

    assert np.allclose(
        information,
        expected,
    )


def test_constant_structure_information_is_stable():
    """A structure with equal candidate sigma must not generate NaNs."""

    sigmas = np.asarray(
        [
            [2.0, 5.0],
            [2.0, 6.0],
            [2.0, 7.0],
        ],
        dtype=float,
    )

    information = (
        _normalise_information_matrix(
            sigmas
        )
    )

    assert np.all(
        np.isfinite(
            information
        )
    )

    assert np.allclose(
        information[:, 0],
        1.0,
    )


def test_movement_normalisation_is_relative_cost():
    """Cheapest movement must receive the smallest cost."""

    movement = np.asarray(
        [
            0.01,
            0.02,
            0.03,
        ],
        dtype=float,
    )

    cost = _normalise_movement(
        movement
    )

    assert np.allclose(
        cost,
        np.asarray(
            [
                0.0,
                0.5,
                1.0,
            ]
        ),
    )


def test_estimated_scene_uses_estimated_centres():
    """Scoring geometry must come from planner-facing estimates."""

    (
        inputs,
        initial,
        _,
    ) = make_initial_state()

    scene = _estimated_scene(
        initial.perception_result
    )

    estimates = (
        initial
        .perception_result
        .estimated_structures
    )

    assert len(
        scene
    ) == len(
        estimates
    )

    for structure, estimate in zip(
        scene,
        estimates,
    ):
        assert np.allclose(
            structure.centre,
            estimate.estimated_centre,
        )


def test_relevance_is_calculated_from_estimated_scene():
    """Estimated structures closer to the trajectory must be more relevant."""

    (
        inputs,
        initial,
        _,
    ) = make_initial_state()

    scene = _estimated_scene(
        initial.perception_result
    )

    relevance = _relevance_weights(
        structures=scene,
        trajectory=(
            inputs.task.trajectory
        ),
        config=(
            TaskRelevanceConfig()
        ),
    )

    assert relevance.shape == (
        len(scene),
    )

    assert np.all(
        relevance >= 0.0
    )

    assert np.all(
        relevance <= 1.0
    )


def test_scorer_does_not_accept_true_structures():
    """Public scoring API must not expose ground-truth anatomy."""

    signature = inspect.signature(
        FairSceneViewpointScorer
        .score_candidates
    )

    assert (
        "true_structures"
        not in signature.parameters
    )

    assert (
        "target"
        not in signature.parameters
    )

    assert (
        "occluders"
        not in signature.parameters
    )


def test_all_scene_score_terms_are_finite():
    """Scene-wide score diagnostics must remain finite."""

    (
        inputs,
        initial,
        scorer,
    ) = make_initial_state()

    scores = scorer.score_candidates(
        current_pose=(
            inputs.initial_pose
        ),
        candidates=(
            inputs.candidates
        ),
        initial_perception=(
            initial.perception_result
        ),
    )

    assert len(
        scores
    ) == len(
        inputs.candidates
    )

    for score in scores:
        assert len(
            score.predicted_sigmas
        ) == len(
            inputs.true_structures
        )

        assert len(
            score.information_scores
        ) == len(
            inputs.true_structures
        )

        assert len(
            score.relevance_weights
        ) == len(
            inputs.true_structures
        )

        assert (
            0.0
            <= score
            .generic_scene_information
            <= 1.0
        )

        assert (
            0.0
            <= score
            .task_scene_information
            <= 1.0
        )

        assert (
            0.0
            <= score.task_alignment
            <= 1.0
        )

        assert np.isfinite(
            score.generic_score
        )

        assert np.isfinite(
            score.task_aware_score
        )


def test_generic_information_is_uniform_structure_average():
    """Generic must treat every anatomical structure equally."""

    (
        inputs,
        initial,
        scorer,
    ) = make_initial_state()

    scores = scorer.score_candidates(
        current_pose=(
            inputs.initial_pose
        ),
        candidates=(
            inputs.candidates
        ),
        initial_perception=(
            initial.perception_result
        ),
    )

    for score in scores:
        expected = float(
            np.mean(
                np.asarray(
                    score.information_scores,
                    dtype=float,
                )
            )
        )

        assert (
            score
            .generic_scene_information
            == pytest.approx(
                expected
            )
        )


def test_both_strategies_select_from_same_candidate_objects():
    """Generic and Task-Aware must use the identical candidate set."""

    (
        inputs,
        initial,
        scorer,
    ) = make_initial_state()

    generic = scorer.select_generic(
        current_pose=(
            inputs.initial_pose
        ),
        candidates=(
            inputs.candidates
        ),
        initial_perception=(
            initial.perception_result
        ),
    )

    task = scorer.select_task_aware(
        current_pose=(
            inputs.initial_pose
        ),
        candidates=(
            inputs.candidates
        ),
        initial_perception=(
            initial.perception_result
        ),
    )

    assert any(
        generic.selected.candidate
        is candidate
        for candidate
        in inputs.candidates
    )

    assert any(
        task.selected.candidate
        is candidate
        for candidate
        in inputs.candidates
    )


def test_large_movement_penalty_reduces_or_preserves_task_movement():
    """Movement cost must remain meaningful in the task-aware objective."""

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

    low = FairSceneViewpointScorer(
        observation_model=(
            inputs.observation_model
        ),
        task_trajectory=(
            inputs.task.trajectory
        ),
        config=(
            FairSceneScoringConfig(
                movement_weight=0.0
            )
        ),
    )

    high = FairSceneViewpointScorer(
        observation_model=(
            inputs.observation_model
        ),
        task_trajectory=(
            inputs.task.trajectory
        ),
        config=(
            FairSceneScoringConfig(
                movement_weight=100.0
            )
        ),
    )

    low_selection = (
        low.select_task_aware(
            current_pose=(
                inputs.initial_pose
            ),
            candidates=(
                inputs.candidates
            ),
            initial_perception=(
                initial.perception_result
            ),
        )
        .selected
    )

    high_selection = (
        high.select_task_aware(
            current_pose=(
                inputs.initial_pose
            ),
            candidates=(
                inputs.candidates
            ),
            initial_perception=(
                initial.perception_result
            ),
        )
        .selected
    )

    assert (
        high_selection.movement_cost
        <= low_selection.movement_cost
        + 1e-12
    )