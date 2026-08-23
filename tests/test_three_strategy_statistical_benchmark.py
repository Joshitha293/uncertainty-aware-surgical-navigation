"""Tests for the three-strategy statistical benchmark."""

import json

import numpy as np
import pytest

from src.simulation.three_strategy_statistical_benchmark import (
    ThreeStrategyStatisticalConfig,
    analyse_records,
    run_matched_trials,
    run_statistical_benchmark,
    save_benchmark_outputs,
)


def small_config():
    """Return a fast deterministic benchmark configuration."""

    return ThreeStrategyStatisticalConfig(
        trial_count=3,
        bootstrap_samples=100,
        bootstrap_seed=12345,
    )


def test_statistical_config_defaults_are_valid():
    config = (
        ThreeStrategyStatisticalConfig()
    )

    assert config.trial_count == 100
    assert config.confidence_level == pytest.approx(
        0.95
    )

    assert config.bootstrap_samples > 0
    assert config.sigma_multiplier >= 0.0


def test_invalid_statistical_config_is_rejected():
    with pytest.raises(
        ValueError,
        match="trial_count",
    ):
        ThreeStrategyStatisticalConfig(
            trial_count=0
        )

    with pytest.raises(
        ValueError,
        match="confidence_level",
    ):
        ThreeStrategyStatisticalConfig(
            confidence_level=1.0
        )

    with pytest.raises(
        ValueError,
        match="bootstrap_samples",
    ):
        ThreeStrategyStatisticalConfig(
            bootstrap_samples=0
        )


def test_matched_trials_create_three_records_per_trial():
    config = small_config()

    records = run_matched_trials(
        config
    )

    assert len(records) == (
        config.trial_count * 3
    )

    for trial in range(
        config.trial_count
    ):
        trial_records = [
            record
            for record in records
            if record.trial == trial
        ]

        assert len(
            trial_records
        ) == 3

        strategies = {
            record.strategy
            for record in trial_records
        }

        assert strategies == {
            "fixed",
            "generic_active",
            "task_aware_active",
        }

        perception_seeds = {
            record.perception_seed
            for record in trial_records
        }

        planner_seeds = {
            record.planner_seed
            for record in trial_records
        }

        assert len(
            perception_seeds
        ) == 1

        assert len(
            planner_seeds
        ) == 1


def test_matched_trial_benchmark_is_reproducible():
    config = small_config()

    first = run_matched_trials(
        config
    )

    second = run_matched_trials(
        config
    )

    assert len(first) == len(second)

    for first_record, second_record in zip(
        first,
        second,
    ):
        assert (
            first_record.strategy
            == second_record.strategy
        )

        assert (
            first_record.planning_success
            == second_record.planning_success
        )

        assert (
            first_record
            .collision_against_truth
            == second_record
            .collision_against_truth
        )

        assert (
            first_record
            .safety_violation_against_truth
            == second_record
            .safety_violation_against_truth
        )

        assert (
            first_record
            .mean_localisation_error
            == pytest.approx(
                second_record
                .mean_localisation_error
            )
        )

        assert (
            first_record
            .mean_predicted_sigma
            == pytest.approx(
                second_record
                .mean_predicted_sigma
            )
        )


def test_analysis_produces_three_summaries_and_two_paired_comparisons():
    config = small_config()

    records = run_matched_trials(
        config
    )

    result = analyse_records(
        records=records,
        config=config,
    )

    assert len(
        result.summaries
    ) == 3

    assert len(
        result.paired_comparisons
    ) == 2

    assert {
        summary.strategy
        for summary in result.summaries
    } == {
        "fixed",
        "generic_active",
        "task_aware_active",
    }

    assert {
        comparison.comparator
        for comparison
        in result.paired_comparisons
    } == {
        "fixed",
        "generic_active",
    }

    for summary in result.summaries:
        assert (
            summary.trial_count
            == config.trial_count
        )

        assert (
            summary
            .localisation_error
            .n
            == config.trial_count
        )

        assert (
            0.0
            <= summary
            .planning_success_rate
            .mean
            <= 1.0
        )

        assert (
            0.0
            <= summary
            .collision_rate
            .mean
            <= 1.0
        )

        assert (
            0.0
            <= summary
            .safety_violation_rate
            .mean
            <= 1.0
        )


def test_statistical_benchmark_returns_paired_effects():
    result = run_statistical_benchmark(
        small_config()
    )

    assert len(
        result.records
    ) == 9

    for comparison in (
        result.paired_comparisons
    ):
        assert (
            comparison
            .localisation_error_difference
            .n
            == 3
        )

        assert (
            comparison
            .planning_success_rate_difference
            .n
            == 3
        )

        estimate = (
            comparison
            .localisation_error_difference
        )

        assert np.isfinite(
            estimate.mean
        )

        assert (
            estimate.ci_low
            <= estimate.mean
            <= estimate.ci_high
        )


def test_benchmark_outputs_are_saved(
    tmp_path,
):
    result = (
        run_statistical_benchmark(
            small_config()
        )
    )

    raw_path, summary_path = (
        save_benchmark_outputs(
            result,
            output_directory=tmp_path,
        )
    )

    assert raw_path.exists()
    assert summary_path.exists()

    raw_text = raw_path.read_text(
        encoding="utf-8"
    )

    assert (
        "mean_localisation_error"
        in raw_text
    )

    assert (
        "task_aware_active"
        in raw_text
    )

    payload = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        payload["config"]["trial_count"]
        == 3
    )

    assert len(
        payload["summaries"]
    ) == 3

    assert len(
        payload[
            "paired_comparisons"
        ]
    ) == 2