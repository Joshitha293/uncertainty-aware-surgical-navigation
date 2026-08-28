"""Tests for the frozen pre-held-out Phase 1 protocol."""

import json

import pytest

from src.simulation.phase1_frozen_protocol import (
    BOOTSTRAP_RESAMPLES,
    FROZEN_GENERIC_MOVEMENT_WEIGHT,
    FROZEN_SCENARIO_MANIFEST_SHA256,
    FROZEN_TASK_MOVEMENT_WEIGHT,
    HELD_OUT_REPETITIONS,
    HELD_OUT_SCENARIO_COUNT,
    PERMUTATION_RESAMPLES,
    StrategyId,
    build_frozen_phase1_protocol,
    held_out_trial_seeds,
    protocol_digest,
    save_protocol,
    validate_frozen_protocol,
)


def test_frozen_manifest_is_exact():
    """Protocol must point to the already frozen scenario manifest."""

    protocol = (
        build_frozen_phase1_protocol()
    )

    assert (
        protocol
        .scenario_manifest_sha256
        == FROZEN_SCENARIO_MANIFEST_SHA256
    )

    assert (
        protocol
        .scenario_manifest_sha256
        == (
            "490922fbc9f91743262257ff594a7440"
            "651016085207b3e8867b9e1bdce8ddd9"
        )
    )


def test_held_out_design_is_frozen():
    """Final sample size must be fixed before execution."""

    protocol = (
        build_frozen_phase1_protocol()
    )

    assert (
        protocol.held_out_scenario_count
        == HELD_OUT_SCENARIO_COUNT
        == 30
    )

    assert (
        protocol.repetitions_per_scenario
        == HELD_OUT_REPETITIONS
        == 10
    )

    assert (
        protocol.total_paired_trials
        == 300
    )


def test_complete_strategy_family_is_frozen():
    """All eight pre-specified strategies must be present."""

    protocol = (
        build_frozen_phase1_protocol()
    )

    observed = {
        strategy.strategy_id
        for strategy
        in protocol.strategies
    }

    assert observed == set(
        StrategyId
    )

    assert len(
        observed
    ) == 8


def test_primary_comparison_is_task_vs_budget_matched_generic():
    """There must be exactly one primary strategy comparison."""

    protocol = (
        build_frozen_phase1_protocol()
    )

    primary = [
        comparison
        for comparison
        in protocol.comparisons
        if comparison.role
        == "primary"
    ]

    assert len(
        primary
    ) == 1

    assert (
        primary[
            0
        ]
        .strategy_a
        == StrategyId.FULL_TASK_AWARE
    )

    assert (
        primary[
            0
        ]
        .strategy_b
        == StrategyId
        .BUDGET_MATCHED_GENERIC
    )


def test_primary_endpoint_is_safe_navigation():
    """Primary outcome must not change after held-out results are seen."""

    protocol = (
        build_frozen_phase1_protocol()
    )

    primary = [
        outcome
        for outcome
        in protocol.outcomes
        if outcome.role
        == "primary"
    ]

    assert len(
        primary
    ) == 1

    assert (
        primary[
            0
        ]
        .name
        == "safe_navigation_success_rate"
    )


def test_corrected_movement_weights_are_frozen():
    """Final development-selected weights must be immutable."""

    protocol = (
        build_frozen_phase1_protocol()
    )

    strategies = {
        strategy.strategy_id:
        strategy
        for strategy
        in protocol.strategies
    }

    assert (
        FROZEN_GENERIC_MOVEMENT_WEIGHT
        == pytest.approx(
            0.072
        )
    )

    assert (
        FROZEN_TASK_MOVEMENT_WEIGHT
        == pytest.approx(
            0.200
        )
    )

    assert (
        strategies[
            StrategyId
            .BUDGET_MATCHED_GENERIC
        ]
        .movement_weight
        == pytest.approx(
            0.072
        )
    )

    assert (
        strategies[
            StrategyId
            .FULL_TASK_AWARE
        ]
        .movement_weight
        == pytest.approx(
            0.200
        )
    )

    assert (
        strategies[
            StrategyId
            .GENERIC_ACTIVE
        ]
        .movement_weight
        == pytest.approx(
            0.200
        )
    )


def test_generic_strategies_are_strictly_task_agnostic():
    """Neither Generic comparator may receive the surgical trajectory."""

    protocol = (
        build_frozen_phase1_protocol()
    )

    strategies = {
        strategy.strategy_id:
        strategy
        for strategy
        in protocol.strategies
    }

    for strategy_id in (
        StrategyId.GENERIC_ACTIVE,
        StrategyId.BUDGET_MATCHED_GENERIC,
    ):
        strategy = strategies[
            strategy_id
        ]

        assert (
            strategy.receives_task_trajectory
            is False
        )

        assert (
            strategy.uses_task_weighted_information
            is False
        )

        assert (
            strategy.uses_task_alignment
            is False
        )

        assert (
            strategy.uses_ground_truth_for_selection
            is False
        )


def test_only_oracle_receives_truth_for_selection():
    """Privileged selection must be isolated to the explicit Oracle."""

    protocol = (
        build_frozen_phase1_protocol()
    )

    truth_users = [
        strategy.strategy_id
        for strategy
        in protocol.strategies
        if strategy
        .uses_ground_truth_for_selection
    ]

    assert truth_users == [
        StrategyId.ORACLE
    ]


def test_candidate_generation_is_ground_truth_isolated():
    """The common active search space must come from noisy perception."""

    protocol = (
        build_frozen_phase1_protocol()
    )

    isolation = (
        protocol
        .information_isolation
    )

    assert (
        isolation.initial_pose_uses_ground_truth
        is False
    )

    assert (
        isolation.candidate_generation_uses_ground_truth
        is False
    )

    assert (
        isolation.initial_observation_is_shared
        is True
    )

    assert (
        isolation.active_candidate_centre
        == "shared noisy estimated target centre"
    )

    assert (
        isolation.deployable_strategies_receive_ground_truth
        is False
    )


def test_fixed_receives_matched_final_observation():
    """Fixed must receive the same observation count as active strategies."""

    protocol = (
        build_frozen_phase1_protocol()
    )

    assert (
        protocol
        .information_isolation
        .fixed_receives_matched_final_observation
        is True
    )

    fixed = next(
        strategy
        for strategy
        in protocol.strategies
        if strategy.strategy_id
        == StrategyId.FIXED
    )

    assert (
        fixed.final_observation
        is True
    )


def test_statistical_plan_uses_scenario_as_inferential_unit():
    """Repeated stochastic trials must not be treated as independent scenes."""

    protocol = (
        build_frozen_phase1_protocol()
    )

    assert (
        "scenario"
        in protocol
        .statistics
        .inferential_unit
    )

    assert (
        protocol.statistics.alpha
        == pytest.approx(
            0.05
        )
    )

    assert (
        protocol
        .statistics
        .bootstrap_resamples
        == BOOTSTRAP_RESAMPLES
        == 10_000
    )

    assert (
        protocol
        .statistics
        .permutation_resamples
        == PERMUTATION_RESAMPLES
        == 100_000
    )


def test_seed_families_are_unique_for_held_out_trials():
    """Matched stochastic conditions must use deterministic non-colliding seeds."""

    protocol = (
        build_frozen_phase1_protocol()
    )

    observed = set()

    for scenario_id in range(
        2000,
        2030,
    ):
        for repetition in range(
            10
        ):
            seeds = (
                held_out_trial_seeds(
                    protocol=protocol,
                    scenario_id=(
                        scenario_id
                    ),
                    repetition=(
                        repetition
                    ),
                )
            )

            assert len(
                set(
                    seeds.values()
                )
            ) == 4

            for seed in seeds.values():
                assert (
                    seed
                    not in observed
                )

                observed.add(
                    seed
                )

    assert len(
        observed
    ) == (
        30
        * 10
        * 4
    )


def test_post_held_out_tuning_is_explicitly_forbidden():
    """Protocol must prohibit changing the analysis after seeing test data."""

    protocol = (
        build_frozen_phase1_protocol()
    )

    lower = (
        protocol
        .tuning_rule
        .lower()
    )

    assert (
        "no strategy definition"
        in lower
    )

    assert (
        "held-out"
        in lower
    )


def test_protocol_validation_passes():
    """Frozen protocol should satisfy all internal invariants."""

    protocol = (
        build_frozen_phase1_protocol()
    )

    validate_frozen_protocol(
        protocol
    )


def test_protocol_digest_is_reproducible():
    """Identical frozen protocol definitions must have identical digests."""

    first = (
        build_frozen_phase1_protocol()
    )

    second = (
        build_frozen_phase1_protocol()
    )

    assert (
        protocol_digest(
            first
        )
        == protocol_digest(
            second
        )
    )


def test_saved_protocol_contains_digest(
    tmp_path,
):
    """Saved protocol evidence must include its cryptographic checksum."""

    protocol = (
        build_frozen_phase1_protocol()
    )

    path, digest = (
        save_protocol(
            protocol,
            output_path=(
                tmp_path
                / "frozen_protocol.json"
            ),
        )
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(
            handle
        )

    assert (
        payload[
            "protocol_sha256"
        ]
        == digest
    )

    assert (
        payload[
            "scenario_manifest_sha256"
        ]
        == FROZEN_SCENARIO_MANIFEST_SHA256
    )

    assert (
        payload[
            "held_out_scenario_count"
        ]
        == 30
    )

    assert (
        payload[
            "repetitions_per_scenario"
        ]
        == 10
    )