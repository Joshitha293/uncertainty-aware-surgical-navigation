"""Tests for E7 explicit uncertainty sensitivity."""

import json

import pytest

from src.simulation.three_strategy_robustness_benchmark import (
    default_scenarios,
)
from src.simulation.uncertainty_stress_benchmark import (
    UncertaintyStressConfig,
    analyse_selection_sweep,
    confirmation_variants,
    default_stress_profiles,
    run_end_to_end_confirmation,
    run_selection_sweep,
    run_uncertainty_stress_benchmark,
    save_outputs,
)


def small_config():
    return UncertaintyStressConfig(
        repetitions=1,
        uncertainty_weights=(
            0.0,
            1.0,
            16.0,
        ),
        bootstrap_samples=30,
        bootstrap_seed=1234,
    )


def small_scenarios():
    scenarios = default_scenarios()

    return (
        scenarios[0],
        scenarios[1],
    )


def small_profiles():
    profiles = (
        default_stress_profiles()
    )

    return (
        profiles[0],
        profiles[1],
    )


def test_config_defaults_are_valid():
    config = (
        UncertaintyStressConfig()
    )

    assert config.repetitions == 2

    assert (
        0.0
        in config.uncertainty_weights
    )

    assert max(
        config.uncertainty_weights
    ) == pytest.approx(
        16.0
    )


def test_invalid_config_is_rejected():
    with pytest.raises(
        ValueError,
        match="repetitions",
    ):
        UncertaintyStressConfig(
            repetitions=0
        )

    with pytest.raises(
        ValueError,
        match="uncertainty weights",
    ):
        UncertaintyStressConfig(
            uncertainty_weights=(
                -1.0,
            )
        )


def test_stress_profiles_cover_distinct_uncertainty_regimes():
    profiles = (
        default_stress_profiles()
    )

    assert len(profiles) == 4

    assert any(
        profile.distance_multiplier
        > profile.angle_multiplier
        for profile in profiles
    )

    assert any(
        profile.angle_multiplier
        > profile.distance_multiplier
        for profile in profiles
    )

    assert any(
        profile.occluded_sigma_multiplier
        > 1.0
        for profile in profiles
    )


def test_selection_sweep_covers_all_weights_and_conditions():
    config = small_config()

    records = (
        run_selection_sweep(
            config,
            scenarios=small_scenarios(),
            profiles=small_profiles(),
        )
    )

    expected = (
        len(
            small_scenarios()
        )
        * len(
            small_profiles()
        )
        * len(
            config.uncertainty_weights
        )
    )

    assert len(records) == expected

    summaries = (
        analyse_selection_sweep(
            records
        )
    )

    assert len(summaries) == len(
        config.uncertainty_weights
    )

    zero = next(
        summary
        for summary in summaries
        if summary.uncertainty_weight
        == 0.0
    )

    assert (
        zero.selection_change_rate
        == pytest.approx(
            0.0
        )
    )


def test_confirmation_defines_four_interpretable_variants():
    variants = (
        confirmation_variants()
    )

    assert tuple(
        variant.name
        for variant in variants
    ) == (
        "generic_baseline",
        "high_uncertainty_only",
        "alignment_only",
        "full_task_aware",
    )


def test_end_to_end_confirmation_is_matched():
    config = small_config()

    records = (
        run_end_to_end_confirmation(
            config,
            scenarios=small_scenarios(),
            profiles=(
                small_profiles()[:1]
            ),
        )
    )

    assert len(records) == (
        len(
            small_scenarios()
        )
        * 4
    )

    for scenario in (
        small_scenarios()
    ):
        selected = [
            record
            for record in records
            if record.scenario_id
            == scenario.scenario_id
        ]

        assert len(selected) == 4

        assert len(
            {
                record.perception_seed
                for record in selected
            }
        ) == 1

        assert len(
            {
                record.planner_seed
                for record in selected
            }
        ) == 1

        for record in selected:
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


def test_full_benchmark_outputs_are_saved(
    tmp_path,
):
    result = (
        run_uncertainty_stress_benchmark(
            small_config(),
            scenarios=small_scenarios(),
            profiles=small_profiles(),
        )
    )

    paths = save_outputs(
        result,
        output_directory=tmp_path,
    )

    assert all(
        path.exists()
        for path in paths
    )

    payload = json.loads(
        paths[2].read_text(
            encoding="utf-8"
        )
    )

    assert len(
        payload[
            "selection_summaries"
        ]
    ) == 3

    assert len(
        payload[
            "variant_summaries"
        ]
    ) == 4