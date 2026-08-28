"""Tests for the frozen eight-strategy Phase 1 runner."""

import numpy as np
import pytest

from src.simulation.phase1_final_runner import (
    FROZEN_PROTOCOL_SHA256,
    HELD_OUT_ID_MAX_EXCLUSIVE,
    HELD_OUT_ID_MIN,
    assert_execution_permission,
    build_eight_strategy_perceptions,
)
from src.simulation.phase1_frozen_protocol import (
    FROZEN_GENERIC_MOVEMENT_WEIGHT,
    FROZEN_TASK_MOVEMENT_WEIGHT,
    StrategyId,
)
from src.simulation.three_strategy_robustness_benchmark import (
    RobustnessScenario,
    default_scenarios,
)


def build_development_bundles():
    """Build all eight strategies without running the RRT."""

    scenario = (
        default_scenarios()[
            0
        ]
    )

    return (
        build_eight_strategy_perceptions(
            scenario=scenario,
            repetition=0,
        )
    )


def test_frozen_protocol_hash_is_exact():
    """Runner must point to the already frozen protocol."""

    assert (
        FROZEN_PROTOCOL_SHA256
        == (
            "dc6537d3ff75832ccbe48c9b2690c8e9"
            "66711b31806076cfbc2721488dbf6361"
        )
    )


def test_all_eight_strategies_are_built():
    """Final perception runner must contain the complete strategy family."""

    bundles = (
        build_development_bundles()
    )

    assert len(
        bundles
    ) == 8

    assert {
        bundle.strategy_id
        for bundle
        in bundles
    } == set(
        StrategyId
    )


def test_fixed_view_has_zero_camera_movement():
    """Fixed View must remain at the common initial camera pose."""

    bundles = (
        build_development_bundles()
    )

    fixed = next(
        bundle
        for bundle
        in bundles
        if bundle.strategy_id
        == StrategyId.FIXED
    )

    assert (
        fixed
        .perception
        .camera_movement
        == pytest.approx(
            0.0
        )
    )

    assert (
        fixed.selected_candidate_index
        is None
    )


def test_frozen_strategy_weights_are_used():
    """The final runner must use the development-frozen weights."""

    bundles = (
        build_development_bundles()
    )

    by_id = {
        bundle.strategy_id:
        bundle
        for bundle
        in bundles
    }

    assert (
        by_id[
            StrategyId
            .BUDGET_MATCHED_GENERIC
        ]
        .movement_weight
        == pytest.approx(
            FROZEN_GENERIC_MOVEMENT_WEIGHT
        )
    )

    assert (
        by_id[
            StrategyId
            .BUDGET_MATCHED_GENERIC
        ]
        .movement_weight
        == pytest.approx(
            0.072
        )
    )

    for strategy_id in (
        StrategyId.GENERIC_ACTIVE,
        StrategyId.ALIGNMENT_ONLY,
        StrategyId.INFORMATION_ONLY,
        StrategyId.FULL_TASK_AWARE,
        StrategyId.ORACLE,
    ):
        assert (
            by_id[
                strategy_id
            ]
            .movement_weight
            == pytest.approx(
                FROZEN_TASK_MOVEMENT_WEIGHT
            )
        )


def test_active_strategies_select_valid_candidates():
    """Every active strategy must select a real common candidate."""

    bundles = (
        build_development_bundles()
    )

    for bundle in bundles:
        if (
            bundle.strategy_id
            == StrategyId.FIXED
        ):
            continue

        assert (
            bundle.selected_candidate_index
            is not None
        )

        assert (
            bundle.selected_candidate_index
            >= 0
        )

        assert (
            bundle.perception
            .candidate_count
            > 0
        )

        assert (
            bundle.selected_candidate_index
            < bundle
            .perception
            .candidate_count
        )


def test_task_metrics_are_attached_to_task_variants():
    """Task-aware mechanism variants must expose task diagnostics."""

    bundles = (
        build_development_bundles()
    )

    by_id = {
        bundle.strategy_id:
        bundle
        for bundle
        in bundles
    }

    for strategy_id in (
        StrategyId.ALIGNMENT_ONLY,
        StrategyId.INFORMATION_ONLY,
        StrategyId.FULL_TASK_AWARE,
        StrategyId.ORACLE,
    ):
        perception = (
            by_id[
                strategy_id
            ]
            .perception
        )

        assert (
            perception.task_relevance
            is not None
        )

        assert (
            perception.task_alignment
            is not None
        )

        assert (
            0.0
            <= perception.task_alignment
            <= 1.0
        )


def test_matched_final_observations_are_reproducible():
    """Fixed seeds must reproduce selected poses and final observations."""

    first = (
        build_development_bundles()
    )

    second = (
        build_development_bundles()
    )

    assert len(
        first
    ) == len(
        second
    )

    for first_bundle, second_bundle in zip(
        first,
        second,
    ):
        assert (
            first_bundle.strategy_id
            == second_bundle.strategy_id
        )

        assert (
            first_bundle
            .selected_candidate_index
            == second_bundle
            .selected_candidate_index
        )

        assert np.allclose(
            first_bundle
            .perception
            .selected_pose
            .position,
            second_bundle
            .perception
            .selected_pose
            .position,
        )

        first_estimates = (
            first_bundle
            .perception
            .perception_result
            .estimated_structures
        )

        second_estimates = (
            second_bundle
            .perception
            .perception_result
            .estimated_structures
        )

        assert len(
            first_estimates
        ) == len(
            second_estimates
        )

        for first_estimate, second_estimate in zip(
            first_estimates,
            second_estimates,
        ):
            assert np.allclose(
                first_estimate
                .estimated_centre,
                second_estimate
                .estimated_centre,
            )


def test_held_out_execution_requires_explicit_permission():
    """Frozen held-out IDs must never run accidentally."""

    held_out = (
        RobustnessScenario(
            scenario_id=(
                HELD_OUT_ID_MIN
            ),
            name=(
                "synthetic_held_out_guard"
            ),
        ),
    )

    with pytest.raises(
        PermissionError,
        match="Held-out",
    ):
        assert_execution_permission(
            scenarios=held_out,
            allow_held_out=False,
        )

    assert_execution_permission(
        scenarios=held_out,
        allow_held_out=True,
    )

    assert (
        HELD_OUT_ID_MAX_EXCLUSIVE
        == 2030
    )