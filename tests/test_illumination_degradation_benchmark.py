"""Tests for the supplementary illumination-degradation benchmark."""

import json

import pytest

from src.simulation.illumination_degradation_benchmark import (
    IlluminationBenchmarkConfig,
    IlluminationCondition,
    analyse_illumination_records,
    apply_illumination_condition,
    default_illumination_conditions,
    run_illumination_benchmark,
    run_illumination_trials,
    save_illumination_outputs,
)
from src.simulation.three_strategy_robustness_benchmark import (
    build_scenario_inputs,
    default_scenarios,
)


def small_config():
    return IlluminationBenchmarkConfig(
        repetitions_per_condition=1,
        bootstrap_samples=20,
        bootstrap_seed=1234,
    )


def small_scenarios():
    scenarios = (
        default_scenarios()
    )

    return (
        scenarios[0],
    )


def small_conditions():
    conditions = (
        default_illumination_conditions()
    )

    return (
        conditions[0],
        conditions[-1],
    )


def test_default_conditions_cover_normal_to_severe():
    conditions = (
        default_illumination_conditions()
    )

    assert len(
        conditions
    ) == 4

    assert (
        conditions[0].quality
        == pytest.approx(
            1.0
        )
    )

    assert (
        conditions[-1].quality
        == pytest.approx(
            0.25
        )
    )

    assert (
        conditions[-1]
        .sigma_multiplier
        == pytest.approx(
            2.0
        )
    )


def test_invalid_illumination_quality_is_rejected():
    with pytest.raises(
        ValueError,
        match="quality",
    ):
        IlluminationCondition(
            name="invalid",
            quality=0.0,
        )

    with pytest.raises(
        ValueError,
        match="quality",
    ):
        IlluminationCondition(
            name="invalid",
            quality=1.1,
        )


def test_illumination_scaling_preserves_geometry_and_scales_noise():
    scenario = (
        default_scenarios()[0]
    )

    inputs = (
        build_scenario_inputs(
            scenario
        )
    )

    condition = (
        IlluminationCondition(
            name="severe",
            quality=0.25,
        )
    )

    degraded = (
        apply_illumination_condition(
            inputs=inputs,
            condition=condition,
        )
    )

    assert (
        degraded
        .observation_model
        .config
        .base_sigma
        == pytest.approx(
            inputs
            .observation_model
            .config
            .base_sigma
            * 2.0
        )
    )

    assert (
        degraded
        .observation_model
        .config
        .occluded_sigma
        == pytest.approx(
            inputs
            .observation_model
            .config
            .occluded_sigma
            * 2.0
        )
    )

    assert (
        degraded.target
        is inputs.target
    )

    assert (
        degraded.candidates
        == inputs.candidates
    )


def test_small_trial_run_is_matched_across_three_strategies():
    records = (
        run_illumination_trials(
            small_config(),
            scenarios=small_scenarios(),
            conditions=small_conditions(),
        )
    )

    assert len(
        records
    ) == 6

    for condition in (
        small_conditions()
    ):
        selected = [
            record
            for record
            in records
            if (
                record
                .illumination_name
                == condition.name
            )
        ]

        assert len(
            selected
        ) == 3

        assert {
            record.strategy
            for record
            in selected
        } == {
            "fixed",
            "generic_active",
            "task_aware_active",
        }

        assert len(
            {
                record.perception_seed
                for record
                in selected
            }
        ) == 1

        assert len(
            {
                record.planner_seed
                for record
                in selected
            }
        ) == 1


def test_safe_navigation_endpoint_is_consistent():
    records = (
        run_illumination_trials(
            small_config(),
            scenarios=small_scenarios(),
            conditions=(
                small_conditions()[:1]
            ),
        )
    )

    for record in records:
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


def test_analysis_produces_summary_for_every_condition_and_strategy():
    config = (
        small_config()
    )

    conditions = (
        small_conditions()
    )

    records = (
        run_illumination_trials(
            config,
            scenarios=small_scenarios(),
            conditions=conditions,
        )
    )

    summaries = (
        analyse_illumination_records(
            records=records,
            config=config,
            conditions=conditions,
        )
    )

    assert len(
        summaries
    ) == 6

    assert {
        summary.strategy
        for summary
        in summaries
    } == {
        "fixed",
        "generic_active",
        "task_aware_active",
    }


def test_complete_benchmark_outputs_are_saved(
    tmp_path,
):
    result = (
        run_illumination_benchmark(
            small_config(),
            scenarios=small_scenarios(),
            conditions=small_conditions(),
        )
    )

    raw_path, summary_path = (
        save_illumination_outputs(
            result,
            output_directory=tmp_path,
        )
    )

    assert raw_path.exists()

    assert summary_path.exists()

    payload = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        payload[
            "model"
        ][
            "physical_camera_model"
        ]
        is False
    )

    assert len(
        payload[
            "conditions"
        ]
    ) == 2

    assert len(
        payload[
            "summaries"
        ]
    ) == 6