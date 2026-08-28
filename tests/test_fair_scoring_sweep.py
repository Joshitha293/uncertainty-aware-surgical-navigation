"""Tests for the Phase 1 fair-scoring sweep."""

import numpy as np
import pytest

from src.simulation.fair_scoring_sweep import (
    DEFAULT_MOVEMENT_WEIGHTS,
    analyse_fair_sweep,
    run_fair_scoring_sweep,
    run_fair_sweep_records,
)
from src.simulation.three_strategy_robustness_benchmark import (
    default_scenarios,
)


def small_scenarios():
    """Use two deterministic scenarios for fast tests."""

    return tuple(
        default_scenarios()[
            :2
        ]
    )


def test_default_weights_are_valid():
    """Fine sweep must contain valid non-negative weights."""

    assert len(
        DEFAULT_MOVEMENT_WEIGHTS
    ) >= 5

    assert all(
        weight >= 0.0
        for weight
        in DEFAULT_MOVEMENT_WEIGHTS
    )


def test_negative_movement_weight_rejected():
    """Invalid movement weights must fail."""

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        run_fair_sweep_records(
            scenarios=small_scenarios(),
            movement_weights=(
                -0.01,
            ),
        )


def test_record_count_matches_design():
    """Every scenario/weight pair must yield one paired comparison."""

    weights = (
        0.0,
        0.1,
        0.5,
    )

    records = (
        run_fair_sweep_records(
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


def test_all_metrics_are_finite_and_valid():
    """Sweep metrics must remain numerically interpretable."""

    records = (
        run_fair_sweep_records(
            scenarios=small_scenarios(),
            movement_weights=(
                0.0,
                0.1,
            ),
        )
    )

    for record in records:
        assert (
            record.generic_movement
            >= 0.0
        )

        assert (
            record.task_movement
            >= 0.0
        )

        assert (
            record.generic_sigma
            > 0.0
        )

        assert (
            record.task_sigma
            > 0.0
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

        assert (
            0.0
            <= record.generic_information
            <= 1.0
        )

        assert (
            0.0
            <= record.task_information
            <= 1.0
        )

        assert np.isfinite(
            record.generic_score
        )

        assert np.isfinite(
            record.task_score
        )


def test_candidate_indices_are_valid():
    """Both strategies must select candidates from the common set."""

    scenarios = small_scenarios()

    records = (
        run_fair_sweep_records(
            scenarios=scenarios,
            movement_weights=(
                0.1,
            ),
        )
    )

    for record, scenario in zip(
        records,
        scenarios,
    ):
        candidate_count = len(
            __import__(
                "src.simulation."
                "three_strategy_robustness_benchmark",
                fromlist=[
                    "build_scenario_inputs"
                ],
            )
            .build_scenario_inputs(
                scenario
            )
            .candidates
        )

        assert (
            0
            <= record.generic_candidate_index
            < candidate_count
        )

        assert (
            0
            <= record.task_candidate_index
            < candidate_count
        )


def test_zero_alignment_weight_would_be_identical_in_principle():
    """The fair scorer architecture must recover Generic without task input."""

    records = (
        run_fair_sweep_records(
            scenarios=small_scenarios(),
            movement_weights=(
                0.0,
                0.1,
            ),
            alignment_weight=0.0,
        )
    )

    for record in records:
        assert (
            record.generic_candidate_index
            == record.task_candidate_index
        )

        assert (
            record.generic_movement
            == pytest.approx(
                record.task_movement
            )
        )

        assert (
            record.generic_sigma
            == pytest.approx(
                record.task_sigma
            )
        )


def test_summary_count_matches_weights():
    """One aggregate summary must be produced per movement weight."""

    weights = (
        0.0,
        0.05,
        0.1,
    )

    result = (
        run_fair_scoring_sweep(
            scenarios=small_scenarios(),
            movement_weights=weights,
        )
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


def test_analysis_produces_finite_differences():
    """Paired aggregate differences must remain finite."""

    weights = (
        0.0,
        0.1,
    )

    records = (
        run_fair_sweep_records(
            scenarios=small_scenarios(),
            movement_weights=weights,
        )
    )

    summaries = (
        analyse_fair_sweep(
            records=records,
            movement_weights=weights,
        )
    )

    for summary in summaries:
        assert np.isfinite(
            summary.mean_movement_difference
        )

        assert np.isfinite(
            summary.mean_sigma_difference
        )

        assert np.isfinite(
            summary.mean_alignment_difference
        )

        assert (
            0.0
            <= summary.different_candidate_rate
            <= 1.0
        )