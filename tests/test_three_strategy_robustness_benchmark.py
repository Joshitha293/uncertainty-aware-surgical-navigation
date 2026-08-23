"""Tests for E5 multi-scenario robustness validation."""

import json

import numpy as np
import pytest

from src.simulation.three_strategy_robustness_benchmark import (
    RobustnessConfig,
    RobustnessScenario,
    analyse_robustness_records,
    build_scenario_inputs,
    default_scenarios,
    run_robustness_benchmark,
    run_robustness_trials,
    save_robustness_outputs,
)


def small_config() -> RobustnessConfig:
    """Return a fast E5 test configuration."""

    return RobustnessConfig(
        repetitions_per_scenario=1,
        bootstrap_samples=50,
        bootstrap_seed=1234,
    )


def small_scenarios():
    """Return two inexpensive scenario variations."""

    scenarios = default_scenarios()

    return (
        scenarios[0],
        scenarios[1],
    )


def test_robustness_config_defaults_are_valid():
    config = RobustnessConfig()

    assert (
        config.repetitions_per_scenario
        == 10
    )

    assert config.bootstrap_samples > 0

    assert (
        0.0
        < config.confidence_level
        < 1.0
    )


def test_invalid_robustness_config_is_rejected():
    with pytest.raises(
        ValueError,
        match="repetitions_per_scenario",
    ):
        RobustnessConfig(
            repetitions_per_scenario=0
        )

    with pytest.raises(
        ValueError,
        match="confidence_level",
    ):
        RobustnessConfig(
            confidence_level=1.0
        )


def test_default_scenarios_include_multiple_scene_variations():
    scenarios = default_scenarios()

    assert len(scenarios) == 10

    assert len(
        {
            scenario.scenario_id
            for scenario in scenarios
        }
    ) == 10

    assert any(
        scenario.translation
        != (0.0, 0.0, 0.0)
        for scenario in scenarios
    )

    assert any(
        scenario.radius_scale
        != 1.0
        for scenario in scenarios
    )

    assert any(
        scenario.occluder_radius
        > 0.0
        for scenario in scenarios
    )


def test_scenario_inputs_change_anatomy_and_viewpoint():
    baseline = build_scenario_inputs(
        default_scenarios()[0]
    )

    varied = build_scenario_inputs(
        default_scenarios()[1]
    )

    assert not np.allclose(
        baseline.target.centre,
        varied.target.centre,
    )

    assert (
        len(
            baseline.candidates
        )
        > 0
    )

    assert (
        len(
            varied.candidates
        )
        > 0
    )

    assert not np.allclose(
        baseline.initial_pose.position,
        varied.initial_pose.position,
    )


def test_matched_robustness_trials_have_three_strategies():
    config = small_config()

    scenarios = small_scenarios()

    records = run_robustness_trials(
        config,
        scenarios=scenarios,
    )

    assert len(records) == (
        len(scenarios)
        * config.repetitions_per_scenario
        * 3
    )

    for scenario in scenarios:
        scenario_records = [
            record
            for record in records
            if record.scenario_id
            == scenario.scenario_id
        ]

        assert {
            record.strategy
            for record
            in scenario_records
        } == {
            "fixed",
            "generic_active",
            "task_aware_active",
        }

        assert len(
            {
                record.perception_seed
                for record
                in scenario_records
            }
        ) == 1

        assert len(
            {
                record.planner_seed
                for record
                in scenario_records
            }
        ) == 1

        for record in scenario_records:
            expected_safe = (
                record.planning_success
                and not record
                .collision_against_truth
                and not record
                .safety_violation_against_truth
            )

            assert (
                record.safe_navigation_success
                == expected_safe
            )


def test_analysis_reports_corrected_safety_endpoint():
    config = small_config()

    scenarios = small_scenarios()

    records = run_robustness_trials(
        config,
        scenarios=scenarios,
    )

    result = analyse_robustness_records(
        records=records,
        scenarios=scenarios,
        config=config,
    )

    assert len(
        result.summaries
    ) == 3

    assert len(
        result.paired_comparisons
    ) == 2

    for summary in result.summaries:
        assert (
            0.0
            <= summary
            .safe_navigation_success_rate
            .mean
            <= 1.0
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
            .worst_scenario_safe_navigation_rate
            <= 1.0
        )


def test_robustness_outputs_are_saved(
    tmp_path,
):
    result = run_robustness_benchmark(
        small_config(),
        scenarios=small_scenarios(),
    )

    raw_path, summary_path = (
        save_robustness_outputs(
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
        "safe_navigation_success"
        in raw_text
    )

    payload = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    assert len(
        payload["scenarios"]
    ) == 2

    assert len(
        payload["summaries"]
    ) == 3

    assert len(
        payload[
            "paired_comparisons"
        ]
    ) == 2