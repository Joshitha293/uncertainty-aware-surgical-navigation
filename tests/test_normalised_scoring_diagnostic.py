"""Tests for the Phase 1 normalised-scoring diagnostic."""

import numpy as np
import pytest

from src.simulation.normalised_scoring_diagnostic import (
    DEFAULT_MOVEMENT_WEIGHTS,
    analyse_records,
    run_normalised_scoring_diagnostic,
    run_diagnostic_records,
)
from src.simulation.three_strategy_robustness_benchmark import (
    default_scenarios,
)


def small_scenarios():
    """Return two deterministic development scenarios."""

    return tuple(
        default_scenarios()[
            :2
        ]
    )


def test_default_weights_are_nonnegative():
    """The diagnostic sweep must use valid movement weights."""

    assert len(
        DEFAULT_MOVEMENT_WEIGHTS
    ) > 1

    assert all(
        weight >= 0.0
        for weight
        in DEFAULT_MOVEMENT_WEIGHTS
    )


def test_negative_weight_rejected():
    """Invalid movement-weight sweeps must fail."""

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        run_diagnostic_records(
            scenarios=small_scenarios(),
            movement_weights=(
                -1.0,
            ),
        )


def test_record_count_matches_scenarios_and_weights():
    """Every scenario/weight pair must produce one record."""

    weights = (
        0.0,
        1.0,
        4.0,
    )

    records = (
        run_diagnostic_records(
            scenarios=small_scenarios(),
            movement_weights=weights,
        )
    )

    assert len(
        records
    ) == (
        len(
            small_scenarios()
        )
        * len(
            weights
        )
    )


def test_records_have_valid_metrics():
    """Diagnostic outputs must be numerically valid."""

    records = (
        run_diagnostic_records(
            scenarios=small_scenarios(),
            movement_weights=(
                0.0,
                1.0,
            ),
        )
    )

    for record in records:
        assert (
            record.camera_movement
            >= 0.0
        )

        assert (
            record.predicted_sigma
            > 0.0
        )

        assert (
            0.0
            <= record.task_alignment
            <= 1.0
        )

        assert (
            0.0
            <= record
            .normalised_generic_utility
            <= 1.0
        )

        assert (
            0.0
            <= record
            .normalised_uncertainty_information
            <= 1.0
        )

        assert (
            0.0
            <= record
            .normalised_movement_cost
            <= 1.0
        )

        assert np.isfinite(
            record.final_score
        )


def test_large_movement_penalty_does_not_increase_mean_movement():
    """A strong movement penalty should not increase average displacement."""

    records = (
        run_diagnostic_records(
            scenarios=small_scenarios(),
            movement_weights=(
                0.0,
                100.0,
            ),
        )
    )

    summaries = (
        analyse_records(
            records=records,
            movement_weights=(
                0.0,
                100.0,
            ),
        )
    )

    assert (
        summaries[
            1
        ].mean_camera_movement
        <= summaries[
            0
        ].mean_camera_movement
        + 1e-12
    )


def test_full_diagnostic_returns_one_summary_per_weight():
    """Top-level diagnostic must retain complete sweep structure."""

    weights = (
        0.0,
        0.5,
        1.0,
    )

    result = (
        run_normalised_scoring_diagnostic(
            scenarios=small_scenarios(),
            movement_weights=weights,
        )
    )

    assert (
        result.movement_weights
        == weights
    )

    assert len(
        result.summaries
    ) == len(
        weights
    )

    assert len(
        result.records
    ) == (
        len(
            small_scenarios()
        )
        * len(
            weights
        )
    )