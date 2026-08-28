"""Tests for the fair scene-wide development sweep."""

import numpy as np
import pytest

from src.simulation.fair_scene_development_sweep import (
    analyse_development_records,
    run_development_records,
    run_fair_scene_development_sweep,
)
from src.simulation.three_strategy_robustness_benchmark import (
    default_scenarios,
)


def small_scenarios():
    """Use two established scenarios for fast tests."""

    return tuple(
        default_scenarios()[
            :2
        ]
    )


def test_negative_weight_rejected():
    """Movement weights must be valid."""

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        run_development_records(
            scenarios=small_scenarios(),
            movement_weights=(
                -0.1,
            ),
            repetitions=1,
        )


def test_nonpositive_repetitions_rejected():
    """Development repetition count must be positive."""

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        run_development_records(
            scenarios=small_scenarios(),
            movement_weights=(
                0.1,
            ),
            repetitions=0,
        )


def test_record_count_matches_matched_design():
    """Every scenario/repetition/weight combination must produce one record."""

    weights = (
        0.0,
        0.1,
        0.5,
    )

    repetitions = 2

    records = (
        run_development_records(
            scenarios=small_scenarios(),
            movement_weights=weights,
            repetitions=(
                repetitions
            ),
        )
    )

    assert len(
        records
    ) == (
        len(
            small_scenarios()
        )
        * repetitions
        * len(
            weights
        )
    )


def test_all_record_metrics_are_valid():
    """Development metrics must remain finite and interpretable."""

    records = (
        run_development_records(
            scenarios=small_scenarios(),
            movement_weights=(
                0.0,
                0.1,
            ),
            repetitions=1,
        )
    )

    for record in records:
        assert (
            record.generic_camera_movement
            >= 0.0
        )

        assert (
            record.task_camera_movement
            >= 0.0
        )

        assert (
            record.generic_predicted_sigma
            > 0.0
        )

        assert (
            record.task_predicted_sigma
            > 0.0
        )

        assert (
            record.generic_localisation_error
            >= 0.0
        )

        assert (
            record.task_localisation_error
            >= 0.0
        )

        assert (
            0.0
            <= record.generic_alignment
            <= 1.0
        )

        assert (
            0.0
            <= record.task_alignment
            <= 1.0
        )

        assert np.isfinite(
            record.generic_score
        )

        assert np.isfinite(
            record.task_score
        )


def test_same_seed_pair_used_for_matched_strategies():
    """Each record must preserve shared initial/final random conditions."""

    records = (
        run_development_records(
            scenarios=small_scenarios(),
            movement_weights=(
                0.1,
            ),
            repetitions=2,
        )
    )

    for record in records:
        assert isinstance(
            record.initial_seed,
            int,
        )

        assert isinstance(
            record.final_seed,
            int,
        )


def test_analysis_returns_one_summary_per_weight():
    """Aggregate analysis must preserve the weight sweep."""

    weights = (
        0.0,
        0.1,
        0.5,
    )

    records = (
        run_development_records(
            scenarios=small_scenarios(),
            movement_weights=weights,
            repetitions=1,
        )
    )

    summaries = (
        analyse_development_records(
            records=records,
            movement_weights=weights,
        )
    )

    assert len(
        summaries
    ) == len(
        weights
    )


def test_summary_differences_are_paired_and_finite():
    """Reported paired differences must remain finite."""

    weights = (
        0.0,
        0.1,
    )

    records = (
        run_development_records(
            scenarios=small_scenarios(),
            movement_weights=weights,
            repetitions=2,
        )
    )

    summaries = (
        analyse_development_records(
            records=records,
            movement_weights=weights,
        )
    )

    for summary in summaries:
        assert np.isfinite(
            summary
            .mean_camera_movement_difference
        )

        assert np.isfinite(
            summary
            .mean_localisation_error_difference
        )

        assert np.isfinite(
            summary
            .mean_predicted_sigma_difference
        )

        assert np.isfinite(
            summary
            .mean_alignment_difference
        )

        assert (
            0.0
            <= summary
            .different_candidate_rate
            <= 1.0
        )


def test_complete_sweep_structure():
    """Top-level development result must retain design metadata."""

    weights = (
        0.0,
        0.1,
    )

    result = (
        run_fair_scene_development_sweep(
            scenarios=small_scenarios(),
            movement_weights=weights,
            repetitions=2,
        )
    )

    assert (
        result.movement_weights
        == weights
    )

    assert (
        result.repetitions
        == 2
    )

    assert len(
        result.summaries
    ) == 2

    assert len(
        result.records
    ) == (
        len(
            small_scenarios()
        )
        * 2
        * 2
    )