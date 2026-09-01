"""Tests for final Phase 1 secondary and ablation analysis."""

import csv

import pytest

from src.simulation.phase1_frozen_protocol import (
    StrategyId,
)
from src.simulation.phase1_secondary_analysis import (
    EXPECTED_RAW_RECORD_COUNT,
    build_strategy_summaries,
    file_sha256,
    load_secondary_records,
    run_secondary_analysis,
)
from src.simulation.phase1_statistics import (
    FROZEN_PROTOCOL_SHA256,
)


def _write_synthetic_csv(
    path,
) -> None:
    """Write a compact complete eight-strategy synthetic experiment."""

    fieldnames = [
        "scenario_id",
        "scenario_name",
        "repetition",
        "strategy",
        "initial_perception_seed",
        "final_perception_seed",
        "random_viewpoint_seed",
        "planner_seed",
        "selected_candidate_index",
        "movement_weight",
        "camera_movement",
        "mean_localisation_error",
        "mean_predicted_sigma",
        "task_relevance",
        "task_alignment",
        "planning_success",
        "collision_against_truth",
        "safety_violation_against_truth",
        "safe_navigation_success",
        "minimum_true_safety_clearance",
        "mean_planning_safety_margin",
        "maximum_planning_safety_radius",
        "planner_iterations",
        "planning_time_seconds",
        "path_cost",
        "maximum_rcm_error",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for scenario_id in (
            1,
            2,
            3,
        ):
            for repetition in (
                0,
                1,
            ):
                for strategy_id in StrategyId:
                    strategy = (
                        strategy_id.value
                    )

                    task = (
                        strategy_id
                        == StrategyId
                        .FULL_TASK_AWARE
                    )

                    comparator = (
                        strategy_id
                        == StrategyId
                        .BUDGET_MATCHED_GENERIC
                    )

                    if task:
                        safe = True
                        planning = True
                        movement = 0.08
                        error = 0.008
                        sigma = 0.007
                        path_cost = 10.0
                        alignment = 0.98

                    elif comparator:
                        safe = bool(
                            scenario_id
                            != 1
                        )
                        planning = bool(
                            scenario_id
                            != 1
                        )
                        movement = 0.08
                        error = 0.010
                        sigma = 0.009
                        path_cost = (
                            12.0
                            if planning
                            else float(
                                "inf"
                            )
                        )
                        alignment = None

                    else:
                        safe = False
                        planning = False
                        movement = 0.04
                        error = 0.015
                        sigma = 0.012
                        path_cost = float(
                            "inf"
                        )

                        if strategy_id in (
                            StrategyId.ALIGNMENT_ONLY,
                            StrategyId.INFORMATION_ONLY,
                            StrategyId.ORACLE,
                        ):
                            alignment = 0.90

                        else:
                            alignment = None

                    seed_offset = (
                        scenario_id
                        * 100
                        + repetition
                    )

                    writer.writerow(
                        {
                            "scenario_id": (
                                scenario_id
                            ),
                            "scenario_name": (
                                f"s{scenario_id}"
                            ),
                            "repetition": (
                                repetition
                            ),
                            "strategy": (
                                strategy
                            ),
                            "initial_perception_seed": (
                                100000
                                + seed_offset
                            ),
                            "final_perception_seed": (
                                200000
                                + seed_offset
                            ),
                            "random_viewpoint_seed": (
                                300000
                                + seed_offset
                            ),
                            "planner_seed": (
                                400000
                                + seed_offset
                            ),
                            "selected_candidate_index": (
                                ""
                                if strategy_id
                                == StrategyId.FIXED
                                else 0
                            ),
                            "movement_weight": "",
                            "camera_movement": (
                                movement
                            ),
                            "mean_localisation_error": (
                                error
                            ),
                            "mean_predicted_sigma": (
                                sigma
                            ),
                            "task_relevance": (
                                0.8
                                if alignment
                                is not None
                                else ""
                            ),
                            "task_alignment": (
                                alignment
                                if alignment
                                is not None
                                else ""
                            ),
                            "planning_success": (
                                planning
                            ),
                            "collision_against_truth": (
                                False
                            ),
                            "safety_violation_against_truth": (
                                False
                            ),
                            "safe_navigation_success": (
                                safe
                            ),
                            "minimum_true_safety_clearance": (
                                0.01
                            ),
                            "mean_planning_safety_margin": (
                                0.01
                            ),
                            "maximum_planning_safety_radius": (
                                0.02
                            ),
                            "planner_iterations": (
                                10
                            ),
                            "planning_time_seconds": (
                                0.1
                            ),
                            "path_cost": (
                                path_cost
                            ),
                            "maximum_rcm_error": (
                                0.0
                            ),
                        }
                    )


def test_frozen_protocol_hash_is_unchanged():
    """Secondary analysis must reference the frozen pre-held-out protocol."""

    assert (
        FROZEN_PROTOCOL_SHA256
        == (
            "dc6537d3ff75832ccbe48c9b2690c8e9"
            "66711b31806076cfbc2721488dbf6361"
        )
    )


def test_expected_final_record_count_is_2400():
    """Frozen experiment size must remain unchanged."""

    assert (
        EXPECTED_RAW_RECORD_COUNT
        == 2400
    )


def test_secondary_loader_accepts_failed_infinite_path_cost(
    tmp_path,
):
    """Failed plans may legitimately preserve path cost as infinity."""

    path = (
        tmp_path
        / "synthetic.csv"
    )

    _write_synthetic_csv(
        path
    )

    records = (
        load_secondary_records(
            path
        )
    )

    failed = [
        record
        for record
        in records
        if not record.planning_success
    ]

    assert failed

    assert any(
        record.path_cost
        == float(
            "inf"
        )
        for record
        in failed
    )


def test_strategy_summary_contains_all_eight_strategies(
    tmp_path,
):
    """Every frozen strategy must receive a summary."""

    path = (
        tmp_path
        / "synthetic.csv"
    )

    _write_synthetic_csv(
        path
    )

    summaries = (
        build_strategy_summaries(
            load_secondary_records(
                path
            )
        )
    )

    assert len(
        summaries
    ) == 8

    assert {
        summary.strategy
        for summary
        in summaries
    } == {
        strategy.value
        for strategy
        in StrategyId
    }


def test_task_aware_synthetic_summary_is_successful(
    tmp_path,
):
    """Synthetic Task-Aware records should aggregate correctly."""

    path = (
        tmp_path
        / "synthetic.csv"
    )

    _write_synthetic_csv(
        path
    )

    summaries = {
        summary.strategy:
        summary
        for summary
        in build_strategy_summaries(
            load_secondary_records(
                path
            )
        )
    }

    task = summaries[
        StrategyId
        .FULL_TASK_AWARE
        .value
    ]

    assert (
        task.safe_navigation_rate
        == pytest.approx(
            1.0
        )
    )

    assert (
        task.planning_success_rate
        == pytest.approx(
            1.0
        )
    )

    assert (
        task.mean_camera_movement_mm
        == pytest.approx(
            80.0
        )
    )


def test_task_alignment_is_retained(
    tmp_path,
):
    """Mechanism diagnostics must survive CSV loading."""

    path = (
        tmp_path
        / "synthetic.csv"
    )

    _write_synthetic_csv(
        path
    )

    summaries = {
        summary.strategy:
        summary
        for summary
        in build_strategy_summaries(
            load_secondary_records(
                path
            )
        )
    }

    task = summaries[
        StrategyId
        .FULL_TASK_AWARE
        .value
    ]

    assert (
        task.mean_task_alignment
        == pytest.approx(
            0.98
        )
    )


def test_raw_file_digest_is_reproducible(
    tmp_path,
):
    """Raw evidence must have deterministic provenance."""

    path = (
        tmp_path
        / "synthetic.csv"
    )

    _write_synthetic_csv(
        path
    )

    assert (
        file_sha256(
            path
        )
        == file_sha256(
            path
        )
    )


def test_secondary_analysis_builds_all_frozen_comparisons(
    tmp_path,
):
    """All comparisons declared before held-out testing must be analysed."""

    path = (
        tmp_path
        / "synthetic.csv"
    )

    _write_synthetic_csv(
        path
    )

    analysis = (
        run_secondary_analysis(
            raw_path=path,
            permutation_resamples=1000,
            bootstrap_resamples=1000,
            require_frozen_shape=False,
        )
    )

    assert len(
        analysis.comparisons
    ) == 7

    assert (
        analysis
        .planning_secondary_family_size
        == 7
    )


def test_secondary_planning_p_values_are_holm_adjusted(
    tmp_path,
):
    """Every planning-success comparison must receive an adjusted p-value."""

    path = (
        tmp_path
        / "synthetic.csv"
    )

    _write_synthetic_csv(
        path
    )

    analysis = (
        run_secondary_analysis(
            raw_path=path,
            permutation_resamples=1000,
            bootstrap_resamples=1000,
            require_frozen_shape=False,
        )
    )

    for comparison in (
        analysis.comparisons
    ):
        assert (
            0.0
            <= comparison
            .planning_permutation_p_raw
            <= 1.0
        )

        assert (
            0.0
            <= comparison
            .planning_permutation_p_holm
            <= 1.0
        )

        assert (
            comparison
            .planning_permutation_p_holm
            + 1e-12
            >= comparison
            .planning_permutation_p_raw
        )


def test_primary_comparator_synthetic_effect_has_correct_direction(
    tmp_path,
):
    """Positive success effect must indicate Task-Aware advantage."""

    path = (
        tmp_path
        / "synthetic.csv"
    )

    _write_synthetic_csv(
        path
    )

    analysis = (
        run_secondary_analysis(
            raw_path=path,
            permutation_resamples=1000,
            bootstrap_resamples=1000,
            require_frozen_shape=False,
        )
    )

    primary = next(
        comparison
        for comparison
        in analysis.comparisons
        if comparison.comparison_name
        == "primary_task_vs_budget_matched_generic"
    )

    assert (
        primary.safe_navigation_effect_pp
        > 0.0
    )

    assert (
        primary.localisation_effect_mm
        < 0.0
    )

    assert (
        primary.sigma_effect_mm
        < 0.0
    )


def test_joint_success_path_cost_excludes_failed_pairs(
    tmp_path,
):
    """Path comparison must use only jointly successful planning pairs."""

    path = (
        tmp_path
        / "synthetic.csv"
    )

    _write_synthetic_csv(
        path
    )

    analysis = (
        run_secondary_analysis(
            raw_path=path,
            permutation_resamples=1000,
            bootstrap_resamples=1000,
            require_frozen_shape=False,
        )
    )

    primary = next(
        comparison
        for comparison
        in analysis.comparisons
        if comparison.comparison_name
        == "primary_task_vs_budget_matched_generic"
    )

    # Scenario 1 comparator always fails, so only scenarios 2 and 3
    # contribute to jointly successful path-cost analysis.
    assert (
        primary
        .jointly_successful_path_scenario_count
        == 2
    )

    assert (
        primary.path_cost_effect
        == pytest.approx(
            -2.0
        )
    )