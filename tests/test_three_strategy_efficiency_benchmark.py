"""Tests for the supplementary three-strategy efficiency benchmark."""

import json

import numpy as np
import pytest

from src.simulation.three_strategy_efficiency_benchmark import (
    EfficiencyBenchmarkConfig,
    STRATEGIES,
    balanced_strategy_order,
    run_efficiency_benchmark,
    save_efficiency_outputs,
)
from src.simulation.three_strategy_robustness_benchmark import (
    default_scenarios,
)


def small_config():
    """Return a lightweight benchmark configuration."""

    return EfficiencyBenchmarkConfig(
        repetitions_per_scenario=1,
        bootstrap_samples=20,
        bootstrap_seed=1234,
    )


def small_scenarios():
    """Return one scenario for targeted tests."""

    return (
        default_scenarios()[0],
    )


@pytest.fixture(
    scope="module",
)
def small_result():
    """Run the expensive small benchmark once."""

    return run_efficiency_benchmark(
        small_config(),
        scenarios=small_scenarios(),
    )


def test_invalid_configuration_is_rejected():
    """Invalid repetition counts must fail."""

    with pytest.raises(
        ValueError,
        match="repetitions_per_scenario",
    ):
        EfficiencyBenchmarkConfig(
            repetitions_per_scenario=0
        )

    with pytest.raises(
        ValueError,
        match="bootstrap_samples",
    ):
        EfficiencyBenchmarkConfig(
            bootstrap_samples=0
        )


def test_strategy_execution_order_is_cyclically_balanced():
    """The three possible execution orders must rotate."""

    assert (
        balanced_strategy_order(
            0
        )
        == (
            "fixed",
            "generic_active",
            "task_aware_active",
        )
    )

    assert (
        balanced_strategy_order(
            1
        )
        == (
            "generic_active",
            "task_aware_active",
            "fixed",
        )
    )

    assert (
        balanced_strategy_order(
            2
        )
        == (
            "task_aware_active",
            "fixed",
            "generic_active",
        )
    )


def test_small_run_contains_three_matched_strategies(
    small_result,
):
    """One matched unit must contain all three strategies."""

    records = (
        small_result.records
    )

    assert len(
        records
    ) == 3

    assert {
        record.strategy
        for record
        in records
    } == set(
        STRATEGIES
    )

    assert len(
        {
            record.perception_seed
            for record
            in records
        }
    ) == 1

    assert len(
        {
            record.planner_seed
            for record
            in records
        }
    ) == 1

    assert {
        record.execution_position
        for record
        in records
    } == {
        0,
        1,
        2,
    }


def test_efficiency_records_have_valid_compute_metrics(
    small_result,
):
    """Timing, iterations and path-cost semantics must be valid."""

    for record in (
        small_result.records
    ):
        assert np.isfinite(
            record
            .planning_time_seconds
        )

        assert (
            record
            .planning_time_seconds
            >= 0.0
        )

        assert (
            record.iterations
            >= 0
        )

        if (
            record.planning_success
        ):
            assert np.isfinite(
                record.path_cost
            )

            assert (
                record.path_cost
                >= 0.0
            )

        else:
            assert np.isinf(
                record.path_cost
            )


def test_safe_navigation_endpoint_is_consistent(
    small_result,
):
    """Safe navigation requires a successful, non-violating plan."""

    for record in (
        small_result.records
    ):
        expected = (
            record.planning_success
            and not record
            .collision_against_truth
            and not record
            .safety_violation_against_truth
        )

        assert (
            record
            .safe_navigation_success
            == expected
        )


def test_summaries_condition_path_cost_on_success(
    small_result,
):
    """Path-cost sample count must equal successful-plan count."""

    assert len(
        small_result.summaries
    ) == 3

    for summary in (
        small_result.summaries
    ):
        records = [
            record
            for record
            in small_result.records
            if (
                record.strategy
                == summary.strategy
            )
        ]

        successful = [
            record
            for record
            in records
            if (
                record.planning_success
            )
        ]

        assert (
            summary
            .path_cost_given_success
            .n
            == len(
                successful
            )
        )

        assert (
            summary
            .planning_time_all_attempts
            .n
            == len(
                records
            )
        )

        assert (
            summary
            .iterations_all_attempts
            .n
            == len(
                records
            )
        )


def test_paired_comparisons_and_outputs(
    small_result,
    tmp_path,
):
    """Paired comparisons and machine-readable outputs must be produced."""

    assert len(
        small_result
        .paired_comparisons
    ) == 2

    assert {
        comparison.comparator
        for comparison
        in small_result
        .paired_comparisons
    } == {
        "fixed",
        "generic_active",
    }

    for comparison in (
        small_result
        .paired_comparisons
    ):
        assert (
            comparison
            .matched_count
            == 1
        )

        assert (
            0
            <= comparison
            .both_success_count
            <= 1
        )

    (
        raw_path,
        summary_path,
    ) = (
        save_efficiency_outputs(
            small_result,
            output_directory=(
                tmp_path
            ),
        )
    )

    assert raw_path.exists()

    assert summary_path.exists()

    payload = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    assert len(
        payload[
            "summaries"
        ]
    ) == 3

    assert len(
        payload[
            "paired_comparisons"
        ]
    ) == 2

    assert (
        "planning_time_all_attempts"
        in payload[
            "metric_definitions"
        ]
    )