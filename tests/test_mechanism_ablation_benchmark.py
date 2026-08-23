"""Tests for E6 end-to-end task-aware mechanism ablation."""

import json

import numpy as np
import pytest

from src.simulation.mechanism_ablation_benchmark import (
    MechanismAblationConfig,
    analyse_mechanism_records,
    make_variants,
    run_mechanism_ablation,
    run_mechanism_trials,
    save_mechanism_outputs,
)
from src.simulation.three_strategy_robustness_benchmark import (
    default_scenarios,
)


def small_config():
    """Return a fast deterministic E6 configuration."""

    return MechanismAblationConfig(
        repetitions_per_scenario=1,
        bootstrap_samples=50,
        bootstrap_seed=12345,
    )


def small_scenarios():
    """Return two different surgical scenarios."""

    scenarios = default_scenarios()

    return (
        scenarios[0],
        scenarios[1],
    )


def test_mechanism_config_defaults_are_valid():
    config = (
        MechanismAblationConfig()
    )

    assert (
        config.repetitions_per_scenario
        == 10
    )

    assert config.task_weight == pytest.approx(
        2.0
    )

    assert config.bootstrap_samples > 0

    assert (
        0.0
        < config.confidence_level
        < 1.0
    )


def test_invalid_mechanism_config_is_rejected():
    with pytest.raises(
        ValueError,
        match="repetitions_per_scenario",
    ):
        MechanismAblationConfig(
            repetitions_per_scenario=0
        )

    with pytest.raises(
        ValueError,
        match="confidence_level",
    ):
        MechanismAblationConfig(
            confidence_level=1.0
        )


def test_four_mechanistic_variants_are_defined():
    variants = make_variants()

    assert tuple(
        variant.name
        for variant in variants
    ) == (
        "generic_baseline",
        "alignment_only",
        "uncertainty_only",
        "full_task_aware",
    )

    generic = variants[0]

    alignment = variants[1]

    uncertainty = variants[2]

    full = variants[3]

    assert (
        generic.alignment_weight
        == 0.0
    )

    assert (
        generic.uncertainty_weight
        == 0.0
    )

    assert (
        alignment.alignment_weight
        == 1.0
    )

    assert (
        alignment.uncertainty_weight
        == 0.0
    )

    assert (
        uncertainty.alignment_weight
        == 0.0
    )

    assert (
        uncertainty.uncertainty_weight
        == 1.0
    )

    assert (
        full.alignment_weight
        == 1.0
    )

    assert (
        full.uncertainty_weight
        == 1.0
    )


def test_matched_trials_create_four_records_per_unit():
    config = small_config()

    scenarios = small_scenarios()

    records = (
        run_mechanism_trials(
            config,
            scenarios=scenarios,
        )
    )

    assert len(records) == (
        len(scenarios)
        * config.repetitions_per_scenario
        * 4
    )

    for scenario in scenarios:
        selected = [
            record
            for record in records
            if record.scenario_id
            == scenario.scenario_id
        ]

        assert {
            record.variant
            for record in selected
        } == {
            "generic_baseline",
            "alignment_only",
            "uncertainty_only",
            "full_task_aware",
        }

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
                record
                .safe_navigation_success
                == expected_safe
            )


def test_generic_ablation_is_reference_selection():
    records = run_mechanism_trials(
        small_config(),
        scenarios=small_scenarios(),
    )

    generic_records = [
        record
        for record in records
        if record.variant
        == "generic_baseline"
    ]

    assert len(
        generic_records
    ) == 2

    assert all(
        not record.differs_from_generic
        for record
        in generic_records
    )


def test_analysis_produces_four_summaries_and_three_comparisons():
    config = small_config()

    scenarios = small_scenarios()

    records = (
        run_mechanism_trials(
            config,
            scenarios=scenarios,
        )
    )

    result = analyse_mechanism_records(
        records=records,
        scenarios=scenarios,
        config=config,
    )

    assert len(
        result.summaries
    ) == 4

    assert len(
        result.full_comparisons
    ) == 3

    assert {
        summary.variant
        for summary
        in result.summaries
    } == {
        "generic_baseline",
        "alignment_only",
        "uncertainty_only",
        "full_task_aware",
    }

    assert {
        comparison.comparator
        for comparison
        in result.full_comparisons
    } == {
        "generic_baseline",
        "alignment_only",
        "uncertainty_only",
    }

    for summary in result.summaries:
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
            .safe_navigation_success_rate
            .mean
            <= 1.0
        )

        assert (
            0.0
            <= summary
            .selection_difference_from_generic_rate
            .mean
            <= 1.0
        )

    for comparison in (
        result.full_comparisons
    ):
        assert (
            0.0
            <= comparison
            .viewpoint_difference_rate
            .mean
            <= 1.0
        )

        assert np.isfinite(
            comparison
            .localisation_error_difference
            .mean
        )


def test_mechanism_outputs_are_saved(
    tmp_path,
):
    result = run_mechanism_ablation(
        small_config(),
        scenarios=small_scenarios(),
    )

    raw_path, summary_path = (
        save_mechanism_outputs(
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

    assert (
        "selected_candidate_index"
        in raw_text
    )

    payload = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    assert len(
        payload["variants"]
    ) == 4

    assert len(
        payload["summaries"]
    ) == 4

    assert len(
        payload[
            "full_comparisons"
        ]
    ) == 3