"""Tests for the frozen Phase 1 statistical engine."""

import numpy as np
import pytest

from src.simulation.phase1_frozen_protocol import (
    StrategyId,
)
from src.simulation.phase1_statistics import (
    FROZEN_PROTOCOL_SHA256,
    StatisticalRecord,
    analyse_primary,
    cluster_bootstrap_mean_difference,
    holm_adjust,
    paired_scenario_effects,
    paired_success_path_cost_differences,
    scenario_success_rates,
    sign_flip_permutation_test,
    validate_raw_records,
)


def make_record(
    *,
    scenario_id: int,
    repetition: int,
    strategy: str,
    safe: bool,
    planning: bool | None = None,
    path_cost: float = 1.0,
) -> StatisticalRecord:
    """Build one deterministic synthetic record."""

    if planning is None:
        planning = safe

    return StatisticalRecord(
        scenario_id=(
            scenario_id
        ),
        repetition=(
            repetition
        ),
        strategy=(
            strategy
        ),
        initial_perception_seed=(
            100000
            + scenario_id * 100
            + repetition
        ),
        final_perception_seed=(
            200000
            + scenario_id * 100
            + repetition
        ),
        random_viewpoint_seed=(
            300000
            + scenario_id * 100
            + repetition
        ),
        planner_seed=(
            400000
            + scenario_id * 100
            + repetition
        ),
        camera_movement=0.05,
        mean_localisation_error=0.01,
        mean_predicted_sigma=0.008,
        planning_success=(
            planning
        ),
        collision_against_truth=False,
        safety_violation_against_truth=False,
        safe_navigation_success=(
            safe
        ),
        path_cost=(
            path_cost
        ),
    )


def make_complete_trial(
    *,
    scenario_id: int,
    repetition: int,
    task_safe: bool,
    comparator_safe: bool,
) -> tuple[
    StatisticalRecord,
    ...,
]:
    """Return all eight strategy records for one synthetic trial."""

    records = []

    for strategy in StrategyId:
        if (
            strategy
            == StrategyId.FULL_TASK_AWARE
        ):
            safe = task_safe

        elif (
            strategy
            == StrategyId
            .BUDGET_MATCHED_GENERIC
        ):
            safe = comparator_safe

        else:
            safe = False

        records.append(
            make_record(
                scenario_id=(
                    scenario_id
                ),
                repetition=(
                    repetition
                ),
                strategy=(
                    strategy.value
                ),
                safe=(
                    safe
                ),
            )
        )

    return tuple(
        records
    )


def test_frozen_protocol_hash_is_exact():
    """Statistics must point to the frozen pre-held-out protocol."""

    assert (
        FROZEN_PROTOCOL_SHA256
        == (
            "dc6537d3ff75832ccbe48c9b2690c8e9"
            "66711b31806076cfbc2721488dbf6361"
        )
    )


def test_scenario_success_rate_aggregates_repetitions():
    """Repetitions must be nested within scenario."""

    records = (
        make_record(
            scenario_id=1,
            repetition=0,
            strategy=(
                StrategyId
                .FULL_TASK_AWARE
                .value
            ),
            safe=True,
        ),
        make_record(
            scenario_id=1,
            repetition=1,
            strategy=(
                StrategyId
                .FULL_TASK_AWARE
                .value
            ),
            safe=False,
        ),
    )

    rates = (
        scenario_success_rates(
            records,
            strategy=(
                StrategyId
                .FULL_TASK_AWARE
                .value
            ),
        )
    )

    assert (
        rates[
            1
        ]
        == pytest.approx(
            0.5
        )
    )


def test_paired_effect_is_task_minus_comparator():
    """Positive effects must favour Task-Aware."""

    effects = (
        paired_scenario_effects(
            task_values={
                1: 1.0,
                2: 0.8,
            },
            comparator_values={
                1: 0.5,
                2: 0.6,
            },
        )
    )

    assert (
        effects[
            0
        ]
        .difference
        == pytest.approx(
            0.5
        )
    )

    assert (
        effects[
            1
        ]
        .difference
        == pytest.approx(
            0.2
        )
    )


def test_sign_flip_permutation_is_reproducible():
    """Frozen seed must produce deterministic inference."""

    differences = np.asarray(
        [
            0.1,
            0.2,
            0.0,
            0.3,
            -0.1,
        ],
        dtype=float,
    )

    first = (
        sign_flip_permutation_test(
            differences,
            resamples=5000,
            seed=1234,
        )
    )

    second = (
        sign_flip_permutation_test(
            differences,
            resamples=5000,
            seed=1234,
        )
    )

    assert (
        first
        .p_value_two_sided
        == second
        .p_value_two_sided
    )

    assert (
        first
        .observed_mean_difference
        == pytest.approx(
            0.1
        )
    )


def test_zero_effect_permutation_is_not_significant():
    """A zero observed effect must not generate superiority."""

    result = (
        sign_flip_permutation_test(
            np.zeros(
                10,
                dtype=float,
            ),
            resamples=2000,
            seed=123,
        )
    )

    assert (
        result
        .observed_mean_difference
        == pytest.approx(
            0.0
        )
    )

    assert (
        result
        .p_value_two_sided
        == pytest.approx(
            1.0
        )
    )


def test_cluster_bootstrap_is_reproducible():
    """Scenario-level bootstrap must be deterministic."""

    differences = np.asarray(
        [
            0.1,
            0.2,
            0.3,
            0.4,
        ],
        dtype=float,
    )

    first = (
        cluster_bootstrap_mean_difference(
            differences,
            resamples=2000,
            seed=999,
        )
    )

    second = (
        cluster_bootstrap_mean_difference(
            differences,
            resamples=2000,
            seed=999,
        )
    )

    assert (
        first.lower
        == second.lower
    )

    assert (
        first.upper
        == second.upper
    )

    assert (
        first
        .observed_mean_difference
        == pytest.approx(
            0.25
        )
    )


def test_holm_adjustment_preserves_original_order():
    """Secondary-family multiplicity correction must be valid."""

    adjusted = (
        holm_adjust(
            (
                0.01,
                0.04,
                0.03,
            )
        )
    )

    assert len(
        adjusted
    ) == 3

    assert all(
        0.0
        <= value
        <= 1.0
        for value
        in adjusted
    )

    assert (
        adjusted[
            0
        ]
        <= adjusted[
            1
        ]
    )


def test_paired_path_cost_uses_only_joint_successes():
    """Failed strategy pairs must not receive artificial path costs."""

    task = (
        StrategyId
        .FULL_TASK_AWARE
        .value
    )

    comparator = (
        StrategyId
        .BUDGET_MATCHED_GENERIC
        .value
    )

    records = (
        make_record(
            scenario_id=1,
            repetition=0,
            strategy=task,
            safe=True,
            planning=True,
            path_cost=10.0,
        ),
        make_record(
            scenario_id=1,
            repetition=0,
            strategy=comparator,
            safe=True,
            planning=True,
            path_cost=12.0,
        ),
        make_record(
            scenario_id=1,
            repetition=1,
            strategy=task,
            safe=True,
            planning=True,
            path_cost=8.0,
        ),
        make_record(
            scenario_id=1,
            repetition=1,
            strategy=comparator,
            safe=False,
            planning=False,
            path_cost=0.0,
        ),
    )

    effects = (
        paired_success_path_cost_differences(
            records,
            task_strategy=task,
            comparator_strategy=(
                comparator
            ),
        )
    )

    assert (
        effects[
            1
        ]
        == pytest.approx(
            -2.0
        )
    )


def test_raw_validation_rejects_duplicate_strategy_record():
    """Duplicate experimental records must be rejected."""

    record = (
        make_record(
            scenario_id=1,
            repetition=0,
            strategy=(
                StrategyId
                .FULL_TASK_AWARE
                .value
            ),
            safe=True,
        )
    )

    with pytest.raises(
        ValueError,
        match="Duplicate",
    ):
        validate_raw_records(
            (
                record,
                record,
            ),
            require_complete_strategy_family=False,
        )


def test_primary_analysis_uses_scenario_not_trial_as_unit():
    """Three scenarios must yield scenario_count=3 regardless of repeats."""

    records = []

    # Scenario 1:
    # Task 100%, Comparator 0%.
    for repetition in range(
        2
    ):
        records.extend(
            make_complete_trial(
                scenario_id=1,
                repetition=repetition,
                task_safe=True,
                comparator_safe=False,
            )
        )

    # Scenario 2:
    # Both 100%.
    for repetition in range(
        2
    ):
        records.extend(
            make_complete_trial(
                scenario_id=2,
                repetition=repetition,
                task_safe=True,
                comparator_safe=True,
            )
        )

    # Scenario 3:
    # Both 0%.
    for repetition in range(
        2
    ):
        records.extend(
            make_complete_trial(
                scenario_id=3,
                repetition=repetition,
                task_safe=False,
                comparator_safe=False,
            )
        )

    records = tuple(
        records
    )

    validate_raw_records(
        records
    )

    result = (
        analyse_primary(
            records
        )
    )

    assert (
        result.scenario_count
        == 3
    )

    assert (
        result.task_mean_success_rate
        == pytest.approx(
            2.0 / 3.0
        )
    )

    assert (
        result.comparator_mean_success_rate
        == pytest.approx(
            1.0 / 3.0
        )
    )

    assert (
        result.absolute_effect
        == pytest.approx(
            1.0 / 3.0
        )
    )