"""Tests for scale-controlled task-aware viewpoint scoring."""

import numpy as np
import pytest

from src.perception.normalised_task_aware_scoring import (
    NormalisedTaskAwareConfig,
    NormalisedTaskAwareViewpointScorer,
    _normalise_higher_is_better,
    _normalise_lower_is_better,
)
from src.perception.viewpoint_scoring import (
    GenericViewpointScorer,
)
from src.simulation.three_strategy_robustness_benchmark import (
    build_scenario_inputs,
    default_scenarios,
)


def make_scorer():
    """Create a scorer using the baseline robustness scene."""

    inputs = build_scenario_inputs(
        default_scenarios()[0]
    )

    generic = GenericViewpointScorer(
        observation_model=(
            inputs.observation_model
        )
    )

    scorer = (
        NormalisedTaskAwareViewpointScorer(
            generic_scorer=generic,
            task=inputs.task,
        )
    )

    return (
        inputs,
        scorer,
    )


def test_invalid_negative_weight_rejected():
    """Scoring weights must be non-negative."""

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        NormalisedTaskAwareConfig(
            movement_weight=-1.0
        )


def test_higher_is_better_normalisation():
    """Higher-is-better scaling must map extrema to zero and one."""

    values = np.asarray(
        [2.0, 4.0, 6.0],
        dtype=float,
    )

    normalised = (
        _normalise_higher_is_better(
            values
        )
    )

    assert np.allclose(
        normalised,
        np.asarray(
            [0.0, 0.5, 1.0]
        ),
    )


def test_lower_is_better_normalisation():
    """Lower-is-better scaling must reward the smallest value."""

    values = np.asarray(
        [2.0, 4.0, 6.0],
        dtype=float,
    )

    normalised = (
        _normalise_lower_is_better(
            values
        )
    )

    assert np.allclose(
        normalised,
        np.asarray(
            [1.0, 0.5, 0.0]
        ),
    )


def test_constant_values_are_stable():
    """Constant candidate terms must not produce NaN values."""

    values = np.asarray(
        [3.0, 3.0, 3.0],
        dtype=float,
    )

    high = (
        _normalise_higher_is_better(
            values
        )
    )

    low = (
        _normalise_lower_is_better(
            values
        )
    )

    assert np.all(
        np.isfinite(
            high
        )
    )

    assert np.all(
        np.isfinite(
            low
        )
    )

    assert np.allclose(
        high,
        1.0,
    )

    assert np.allclose(
        low,
        1.0,
    )


def test_candidate_terms_are_bounded():
    """All normalised continuous terms must lie in [0, 1]."""

    (
        inputs,
        scorer,
    ) = make_scorer()

    scores = (
        scorer.score_candidates(
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
    )

    assert len(
        scores
    ) == len(
        inputs.candidates
    )

    for score in scores:
        assert (
            0.0
            <= score
            .normalised_generic_utility
            <= 1.0
        )

        assert (
            0.0
            <= score
            .normalised_uncertainty_information
            <= 1.0
        )

        assert (
            0.0
            <= score
            .normalised_movement_cost
            <= 1.0
        )

        assert (
            0.0
            <= score.task_alignment
            <= 1.0
        )

        assert np.isfinite(
            score.final_score
        )


def test_selected_viewpoint_is_from_common_candidates():
    """The scorer must select only from the supplied candidate set."""

    (
        inputs,
        scorer,
    ) = make_scorer()

    selected = (
        scorer.select_viewpoint(
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
    )

    assert any(
        selected.candidate
        is candidate
        for candidate
        in inputs.candidates
    )


def test_large_movement_weight_cannot_be_numerically_ignored():
    """Movement weighting must influence selection on a common scale."""

    inputs = build_scenario_inputs(
        default_scenarios()[0]
    )

    generic = (
        GenericViewpointScorer(
            observation_model=(
                inputs.observation_model
            )
        )
    )

    low_penalty = (
        NormalisedTaskAwareViewpointScorer(
            generic_scorer=generic,
            task=inputs.task,
            config=(
                NormalisedTaskAwareConfig(
                    generic_weight=1.0,
                    alignment_weight=1.0,
                    uncertainty_weight=1.0,
                    movement_weight=0.0,
                )
            ),
        )
    )

    high_penalty = (
        NormalisedTaskAwareViewpointScorer(
            generic_scorer=generic,
            task=inputs.task,
            config=(
                NormalisedTaskAwareConfig(
                    generic_weight=1.0,
                    alignment_weight=1.0,
                    uncertainty_weight=1.0,
                    movement_weight=100.0,
                )
            ),
        )
    )

    low_selection = (
        low_penalty.select_viewpoint(
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
    )

    high_selection = (
        high_penalty.select_viewpoint(
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
    )

    assert (
        high_selection
        .movement_cost
        <= low_selection
        .movement_cost
        + 1e-12
    )


def test_original_generic_score_remains_available():
    """Normalisation must preserve raw generic diagnostics."""

    (
        inputs,
        scorer,
    ) = make_scorer()

    scores = (
        scorer.score_candidates(
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
    )

    for score in scores:
        assert np.isfinite(
            score
            .generic_score
            .score
        )

        assert np.isfinite(
            score
            .generic_score
            .movement_cost
        )