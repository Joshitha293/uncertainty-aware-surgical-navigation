"""Tests for Phase 6 closed-loop risk supervision."""

from __future__ import annotations

import numpy as np
import pytest

from src.perception.uncertainty import (
    EstimatedStructure,
    PositionUncertainty,
)
from src.robotics.closed_loop_risk_manager import (
    ClosedLoopRiskConfig,
    ClosedLoopRiskManager,
    RiskManagementAction,
)
from src.robotics.risk_aware_planning import (
    ChanceConstraintConfig,
)
from src.simulation.phase6_chance_constraint_benchmark import (
    INSTRUMENT_RADIUS,
    PROXIMAL_LENGTH,
    SHAFT_SAMPLE_SPACING,
    goal_configuration,
    make_instrument,
    select_disagreement_clearance,
    start_configuration,
)


def _base_estimate() -> EstimatedStructure:
    (
        _clearance,
        conditions,
    ) = (
        select_disagreement_clearance()
    )

    tangential, _radial = (
        conditions
    )

    return tangential.estimate


def _estimate_with_covariance_scale(
    scale: float,
) -> EstimatedStructure:
    base = (
        _base_estimate()
    )

    return EstimatedStructure(
        estimated_centre=(
            base
            .estimated_centre
            .copy()
        ),
        physical_radius=(
            base.physical_radius
        ),
        base_safety_margin=(
            base.base_safety_margin
        ),
        uncertainty=PositionUncertainty(
            covariance=(
                base
                .uncertainty
                .covariance
                * scale**2
            )
        ),
    )


def _path() -> np.ndarray:
    return np.vstack(
        [
            start_configuration(),
            goal_configuration(),
        ]
    )


def _chance_config() -> ChanceConstraintConfig:
    return ChanceConstraintConfig(
        max_point_violation_probability=0.05,
        sample_spacing=(
            SHAFT_SAMPLE_SPACING
        ),
    )


def _manager() -> ClosedLoopRiskManager:
    return ClosedLoopRiskManager(
        ClosedLoopRiskConfig(
            replan_probability=0.05,
            reacquire_probability=0.20,
            stop_probability=0.45,
            reacquire_principal_sigma=0.015,
            stop_principal_sigma=0.030,
            minimum_replan_interval_steps=2,
        )
    )


def test_nominal_tangential_path_continues() -> None:
    decision = (
        _manager().evaluate(
            instrument=make_instrument(),
            remaining_path=_path(),
            estimated_structures=(
                _base_estimate(),
            ),
            instrument_radius=(
                INSTRUMENT_RADIUS
            ),
            chance_config=(
                _chance_config()
            ),
            step_index=0,
            proximal_length=(
                PROXIMAL_LENGTH
            ),
            shaft_sample_spacing=(
                SHAFT_SAMPLE_SPACING
            ),
            edge_resolution=31,
        )
    )

    assert (
        decision.action
        == RiskManagementAction.CONTINUE
    )

    assert decision.path_accepted is True


def test_moderately_increased_uncertainty_requests_replan() -> None:
    decision = (
        _manager().evaluate(
            instrument=make_instrument(),
            remaining_path=_path(),
            estimated_structures=(
                _estimate_with_covariance_scale(
                    1.20
                ),
            ),
            instrument_radius=(
                INSTRUMENT_RADIUS
            ),
            chance_config=(
                _chance_config()
            ),
            step_index=3,
            proximal_length=(
                PROXIMAL_LENGTH
            ),
            shaft_sample_spacing=(
                SHAFT_SAMPLE_SPACING
            ),
            edge_resolution=31,
        )
    )

    assert (
        decision.action
        in {
            RiskManagementAction.REPLAN,
            RiskManagementAction.REACQUIRE,
        }
    )


def test_large_uncertainty_requests_reacquisition_or_stop() -> None:
    decision = (
        _manager().evaluate(
            instrument=make_instrument(),
            remaining_path=_path(),
            estimated_structures=(
                _estimate_with_covariance_scale(
                    1.70
                ),
            ),
            instrument_radius=(
                INSTRUMENT_RADIUS
            ),
            chance_config=(
                _chance_config()
            ),
            step_index=4,
            proximal_length=(
                PROXIMAL_LENGTH
            ),
            shaft_sample_spacing=(
                SHAFT_SAMPLE_SPACING
            ),
            edge_resolution=31,
        )
    )

    assert (
        decision.action
        in {
            RiskManagementAction.REACQUIRE,
            RiskManagementAction.STOP,
        }
    )


def test_extreme_sigma_forces_stop() -> None:
    decision = (
        _manager().evaluate(
            instrument=make_instrument(),
            remaining_path=_path(),
            estimated_structures=(
                _estimate_with_covariance_scale(
                    3.1
                ),
            ),
            instrument_radius=(
                INSTRUMENT_RADIUS
            ),
            chance_config=(
                _chance_config()
            ),
            step_index=5,
        )
    )

    assert (
        decision.action
        == RiskManagementAction.STOP
    )

    assert decision.path_accepted is False


def test_replan_registration_updates_state() -> None:
    manager = _manager()

    assert (
        manager.last_replan_step
        is None
    )

    manager.register_replan(
        7
    )

    assert (
        manager.last_replan_step
        == 7
    )


def test_negative_replan_step_is_rejected() -> None:
    manager = _manager()

    with pytest.raises(
        ValueError
    ):
        manager.register_replan(
            -1
        )


def test_invalid_probability_order_is_rejected() -> None:
    with pytest.raises(
        ValueError
    ):
        ClosedLoopRiskConfig(
            replan_probability=0.20,
            reacquire_probability=0.10,
            stop_probability=0.40,
        )


def test_stop_sigma_must_exceed_reacquire_sigma() -> None:
    with pytest.raises(
        ValueError
    ):
        ClosedLoopRiskConfig(
            reacquire_principal_sigma=0.020,
            stop_principal_sigma=0.010,
        )


def test_runtime_decision_uses_no_ground_truth_argument() -> None:
    import inspect

    parameter_names = set(
        inspect.signature(
            ClosedLoopRiskManager.evaluate
        ).parameters
    )

    forbidden = {
        "ground_truth",
        "true_structure",
        "true_structures",
        "true_centre",
        "true_position",
    }

    assert (
        parameter_names
        & forbidden
        == set()
    )