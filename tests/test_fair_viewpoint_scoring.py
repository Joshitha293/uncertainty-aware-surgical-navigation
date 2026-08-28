"""Tests for fair primitive-term viewpoint scoring."""

import numpy as np
import pytest

from src.perception.fair_viewpoint_scoring import (
    FairScoringConfig,
    FairViewpointScorer,
    normalise_cost,
    normalise_information,
)
from src.simulation.three_strategy_robustness_benchmark import (
    build_scenario_inputs,
    default_scenarios,
)


def make_scorer(
    *,
    movement_weight: float = 0.10,
    alignment_weight: float = 1.0,
):
    """Build a fair scorer using one established scenario."""

    inputs = build_scenario_inputs(
        default_scenarios()[0]
    )

    scorer = FairViewpointScorer(
        observation_model=(
            inputs.observation_model
        ),
        task=inputs.task,
        config=FairScoringConfig(
            perception_weight=1.0,
            movement_weight=(
                movement_weight
            ),
            alignment_weight=(
                alignment_weight
            ),
        ),
    )

    return (
        inputs,
        scorer,
    )


def test_negative_weight_rejected():
    """Weights must be non-negative."""

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        FairScoringConfig(
            movement_weight=-0.1
        )


def test_information_normalisation_rewards_low_sigma():
    """Lower uncertainty must produce greater information."""

    values = np.asarray(
        [
            0.002,
            0.004,
            0.010,
        ],
        dtype=float,
    )

    result = normalise_information(
        values
    )

    assert np.allclose(
        result,
        np.asarray(
            [
                1.0,
                0.75,
                0.0,
            ]
        ),
    )


def test_equal_information_is_neutral_for_ranking():
    """Equal uncertainty should provide equal information."""

    result = normalise_information(
        np.asarray(
            [
                0.002,
                0.002,
                0.002,
            ],
            dtype=float,
        )
    )

    assert np.allclose(
        result,
        1.0,
    )


def test_cost_normalisation_penalises_larger_cost():
    """Larger camera displacement must have larger normalised cost."""

    result = normalise_cost(
        np.asarray(
            [
                0.01,
                0.02,
                0.03,
            ],
            dtype=float,
        )
    )

    assert np.allclose(
        result,
        np.asarray(
            [
                0.0,
                0.5,
                1.0,
            ]
        ),
    )


def test_equal_cost_has_no_relative_penalty():
    """Equal candidate movement should not alter ranking."""

    result = normalise_cost(
        np.asarray(
            [
                0.02,
                0.02,
                0.02,
            ],
            dtype=float,
        )
    )

    assert np.allclose(
        result,
        0.0,
    )


def test_all_candidate_terms_are_bounded_and_finite():
    """Primitive score terms must remain interpretable."""

    (
        inputs,
        scorer,
    ) = make_scorer()

    scores = scorer.score_candidates(
        current_pose=(
            inputs.initial_pose
        ),
        candidates=(
            inputs.candidates
        ),
        target=(
            inputs.target
        ),
        occluders=(
            inputs.occluders
        ),
    )

    assert len(
        scores
    ) == len(
        inputs.candidates
    )

    for score in scores:
        assert (
            0.0
            <= score.information_score
            <= 1.0
        )

        assert (
            0.0
            <= score.normalised_movement_cost
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


def test_task_aware_differs_from_generic_only_by_alignment_term():
    """Task-aware must receive exactly one additional information term."""

    (
        inputs,
        scorer,
    ) = make_scorer(
        alignment_weight=0.75
    )

    scores = scorer.score_candidates(
        current_pose=(
            inputs.initial_pose
        ),
        candidates=(
            inputs.candidates
        ),
        target=(
            inputs.target
        ),
        occluders=(
            inputs.occluders
        ),
    )

    for score in scores:
        expected_difference = (
            0.75
            * score.task_alignment
        )

        actual_difference = (
            score.task_aware_score
            - score.generic_score
        )

        assert actual_difference == pytest.approx(
            expected_difference
        )


def test_zero_alignment_weight_makes_strategies_identical():
    """Removing task information must recover Generic exactly."""

    (
        inputs,
        scorer,
    ) = make_scorer(
        alignment_weight=0.0
    )

    generic = scorer.select_generic(
        current_pose=(
            inputs.initial_pose
        ),
        candidates=(
            inputs.candidates
        ),
        target=(
            inputs.target
        ),
        occluders=(
            inputs.occluders
        ),
    )

    task_aware = scorer.select_task_aware(
        current_pose=(
            inputs.initial_pose
        ),
        candidates=(
            inputs.candidates
        ),
        target=(
            inputs.target
        ),
        occluders=(
            inputs.occluders
        ),
    )

    assert (
        task_aware.selected.candidate
        is generic.selected.candidate
    )

    assert (
        task_aware
        .selected
        .task_aware_score
        == pytest.approx(
            generic
            .selected
            .generic_score
        )
    )


def test_selected_candidates_come_from_common_candidate_set():
    """Both strategies must operate on exactly the supplied candidates."""

    (
        inputs,
        scorer,
    ) = make_scorer()

    generic = scorer.select_generic(
        current_pose=(
            inputs.initial_pose
        ),
        candidates=(
            inputs.candidates
        ),
        target=(
            inputs.target
        ),
        occluders=(
            inputs.occluders
        ),
    )

    task_aware = scorer.select_task_aware(
        current_pose=(
            inputs.initial_pose
        ),
        candidates=(
            inputs.candidates
        ),
        target=(
            inputs.target
        ),
        occluders=(
            inputs.occluders
        ),
    )

    generic_is_supplied_candidate = any(
        generic.selected.candidate
        is candidate
        for candidate
        in inputs.candidates
    )

    task_aware_is_supplied_candidate = any(
        task_aware.selected.candidate
        is candidate
        for candidate
        in inputs.candidates
    )

    assert (
        generic_is_supplied_candidate
    )

    assert (
        task_aware_is_supplied_candidate
    )


def test_high_movement_weight_cannot_increase_selected_task_movement():
    """A very large movement penalty should not increase displacement."""

    (
        inputs,
        low_penalty,
    ) = make_scorer(
        movement_weight=0.0
    )

    (
        _,
        high_penalty,
    ) = make_scorer(
        movement_weight=100.0
    )

    low = (
        low_penalty
        .select_task_aware(
            current_pose=(
                inputs.initial_pose
            ),
            candidates=(
                inputs.candidates
            ),
            target=(
                inputs.target
            ),
            occluders=(
                inputs.occluders
            ),
        )
        .selected
    )

    high = (
        high_penalty
        .select_task_aware(
            current_pose=(
                inputs.initial_pose
            ),
            candidates=(
                inputs.candidates
            ),
            target=(
                inputs.target
            ),
            occluders=(
                inputs.occluders
            ),
        )
        .selected
    )

    assert (
        high.movement_cost
        <= low.movement_cost
        + 1e-12
    )