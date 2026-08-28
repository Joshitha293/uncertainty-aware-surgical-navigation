"""Tests for Phase 1 Random and Oracle baselines."""

import inspect

import numpy as np

from src.perception.fair_scene_viewpoint_scoring import (
    FairSceneScoringConfig,
)
from src.perception.phase1_baselines import (
    score_oracle_candidates,
    select_oracle_viewpoint,
    select_random_viewpoint,
)
from src.simulation.three_strategy_robustness_benchmark import (
    build_scenario_inputs,
    default_scenarios,
)


def make_inputs():
    return build_scenario_inputs(
        default_scenarios()[0]
    )


def test_random_baseline_is_reproducible():
    """Same random seed must select the same candidate."""

    inputs = make_inputs()

    first = select_random_viewpoint(
        current_pose=(
            inputs.initial_pose
        ),
        candidates=(
            inputs.candidates
        ),
        seed=1234,
    )

    second = select_random_viewpoint(
        current_pose=(
            inputs.initial_pose
        ),
        candidates=(
            inputs.candidates
        ),
        seed=1234,
    )

    assert (
        first.candidate_index
        == second.candidate_index
    )

    assert (
        first.candidate
        is second.candidate
    )


def test_random_baseline_uses_common_candidate_set():
    """Random strategy must return one supplied candidate."""

    inputs = make_inputs()

    selection = (
        select_random_viewpoint(
            current_pose=(
                inputs.initial_pose
            ),
            candidates=(
                inputs.candidates
            ),
            seed=77,
        )
    )

    assert any(
        selection.candidate
        is candidate
        for candidate
        in inputs.candidates
    )


def test_random_baseline_api_has_no_truth_inputs():
    """Random Active must not receive anatomical truth."""

    signature = inspect.signature(
        select_random_viewpoint
    )

    forbidden = (
        "true_structures",
        "target",
        "true_target",
        "occluders",
        "task_trajectory",
    )

    for name in forbidden:
        assert (
            name
            not in signature.parameters
        )


def test_oracle_explicitly_requires_truth():
    """Oracle must make its privileged information access explicit."""

    signature = inspect.signature(
        select_oracle_viewpoint
    )

    assert (
        "true_structures"
        in signature.parameters
    )

    assert (
        "true_occluders"
        in signature.parameters
    )


def test_oracle_scores_all_candidates():
    """Oracle must compare the complete common candidate set."""

    inputs = make_inputs()

    scores = score_oracle_candidates(
        observation_model=(
            inputs.observation_model
        ),
        current_pose=(
            inputs.initial_pose
        ),
        candidates=(
            inputs.candidates
        ),
        true_structures=(
            inputs.true_structures
        ),
        true_occluders=(
            inputs.occluders
        ),
        task_trajectory=(
            inputs.task.trajectory
        ),
        config=(
            FairSceneScoringConfig(
                movement_weight=0.2
            )
        ),
    )

    assert len(
        scores
    ) == len(
        inputs.candidates
    )


def test_oracle_terms_are_bounded_and_finite():
    """Privileged scoring terms must remain interpretable."""

    inputs = make_inputs()

    scores = score_oracle_candidates(
        observation_model=(
            inputs.observation_model
        ),
        current_pose=(
            inputs.initial_pose
        ),
        candidates=(
            inputs.candidates
        ),
        true_structures=(
            inputs.true_structures
        ),
        true_occluders=(
            inputs.occluders
        ),
        task_trajectory=(
            inputs.task.trajectory
        ),
    )

    for score in scores:
        assert (
            0.0
            <= score.task_scene_information
            <= 1.0
        )

        assert (
            0.0
            <= score.task_alignment
            <= 1.0
        )

        assert (
            0.0
            <= score.normalised_movement_cost
            <= 1.0
        )

        assert np.isfinite(
            score.oracle_score
        )


def test_oracle_selection_comes_from_common_candidates():
    """Oracle must not invent a new viewpoint."""

    inputs = make_inputs()

    selection = (
        select_oracle_viewpoint(
            observation_model=(
                inputs.observation_model
            ),
            current_pose=(
                inputs.initial_pose
            ),
            candidates=(
                inputs.candidates
            ),
            true_structures=(
                inputs.true_structures
            ),
            true_occluders=(
                inputs.occluders
            ),
            task_trajectory=(
                inputs.task.trajectory
            ),
        )
    )

    assert any(
        selection.selected.candidate
        is candidate
        for candidate
        in inputs.candidates
    )


def test_oracle_relevance_count_matches_anatomy():
    """Oracle should assign one relevance value to every structure."""

    inputs = make_inputs()

    selection = (
        select_oracle_viewpoint(
            observation_model=(
                inputs.observation_model
            ),
            current_pose=(
                inputs.initial_pose
            ),
            candidates=(
                inputs.candidates
            ),
            true_structures=(
                inputs.true_structures
            ),
            true_occluders=(
                inputs.occluders
            ),
            task_trajectory=(
                inputs.task.trajectory
            ),
        )
    )

    assert len(
        selection
        .selected
        .relevance_weights
    ) == len(
        inputs.true_structures
    )