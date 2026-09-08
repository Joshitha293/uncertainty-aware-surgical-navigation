"""Regression test for future-risk versus immediate-risk supervision."""

from __future__ import annotations

import numpy as np

from src.perception.uncertainty import (
    EstimatedStructure,
    PositionUncertainty,
)
from src.robotics.closed_loop_risk_manager import (
    ClosedLoopRiskManager,
    RiskManagementAction,
)
from src.simulation.phase6_closed_loop_benchmark import (
    CLOSED_LOOP_CONFIG,
    EXECUTION_SAMPLES,
    INSTRUMENT_RADIUS,
    PROXIMAL_LENGTH,
    SHAFT_SAMPLE_SPACING,
    _base_condition,
    _plan_nominal,
    chance_config,
    make_instrument,
    resample_joint_path,
)
from src.simulation.phase6_chance_constraint_benchmark import (
    start_configuration,
)


FROZEN_SECONDARY_CENTRE = np.asarray(
    [
        0.0420764,
        0.00799174,
        0.02476202,
    ],
    dtype=float,
)


def test_high_future_risk_with_safe_current_state_requests_replan() -> None:
    """Future-only risk should REPLAN rather than STOP.

    Frozen development candidate:

    update onset:
        index 11

    current configuration:
        nominal path index 10

    secondary structure:
        centre [0.0420764, 0.00799174, 0.02476202]
        physical radius 5 mm
        safety margin 4 mm
        isotropic positional sigma 2 mm

    Development search established:

    - pre-update/current configuration safe;
    - one future nominal safety-margin violation;
    - zero physical collisions;
    - alternative chance-constrained replanning succeeded 3/3.
    """

    base_condition = (
        _base_condition()
    )

    nominal_plan = (
        _plan_nominal(
            start_q=(
                start_configuration()
            ),
            estimate=(
                base_condition
                .estimate
            ),
            seed=777,
        )
    )

    assert nominal_plan.success

    nominal_path = (
        resample_joint_path(
            nominal_plan.path,
            EXECUTION_SAMPLES,
        )
    )

    secondary = (
        EstimatedStructure(
            estimated_centre=(
                FROZEN_SECONDARY_CENTRE
            ),
            physical_radius=0.005,
            base_safety_margin=0.004,
            uncertainty=(
                PositionUncertainty(
                    covariance=(
                        np.eye(
                            3,
                            dtype=float,
                        )
                        * (
                            0.002**2
                        )
                    )
                )
            ),
        )
    )

    manager = (
        ClosedLoopRiskManager(
            CLOSED_LOOP_CONFIG
        )
    )

    remaining_path = np.vstack(
        [
            nominal_path[
                10
            ],
            nominal_path[
                11:
            ],
        ]
    )

    decision = (
        manager.evaluate(
            instrument=(
                make_instrument()
            ),
            remaining_path=(
                remaining_path
            ),
            estimated_structures=(
                base_condition.estimate,
                secondary,
            ),
            instrument_radius=(
                INSTRUMENT_RADIUS
            ),
            chance_config=(
                chance_config()
            ),
            step_index=11,
            proximal_length=(
                PROXIMAL_LENGTH
            ),
            shaft_sample_spacing=(
                SHAFT_SAMPLE_SPACING
            ),
            edge_resolution=16,
        )
    )

    assert (
        decision.action
        == RiskManagementAction.REPLAN
    )

    # The complete remaining path deliberately exceeds the global STOP
    # probability level. This regression guard verifies that future-only
    # high risk is no longer confused with immediate current-state danger.
    assert (
        decision
        .maximum_point_violation_probability
        > CLOSED_LOOP_CONFIG
        .stop_probability
    )

    assert (
        decision
        .maximum_principal_sigma
        < CLOSED_LOOP_CONFIG
        .reacquire_principal_sigma
    )

    assert (
        decision.action
        != RiskManagementAction.STOP
    )