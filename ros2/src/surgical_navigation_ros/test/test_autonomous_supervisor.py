"""Tests for the ROS autonomous-supervisor bridge."""

import numpy as np

from surgical_navigation_ros.autonomous_supervisor_node import (
    AutonomousSupervisorLogic,
)
from surgical_navigation_ros.autonomous_supervisor_node import (
    command_from_decision,
)
from surgical_navigation_ros.core_adapter import (
    build_runtime_snapshot,
)
from surgical_navigation_ros.core_adapter import (
    ROS_JOINT_NAMES,
)


def make_snapshot(
    *,
    step: int = 10,
    latest_perception_step: int = 10,
    sigma: float = 0.005,
    clearance: float = 0.010,
    tracking_error: float = 0.001,
):
    """Create one deterministic ROS-order runtime snapshot."""
    return build_runtime_snapshot(
        names=ROS_JOINT_NAMES,
        positions=np.asarray(
            [
                0.0,
                0.0,
                0.0,
                0.175,
            ],
            dtype=float,
        ),
        step_index=step,
        latest_perception_step=latest_perception_step,
        maximum_principal_sigma=sigma,
        predicted_clearance=clearance,
        tracking_error=tracking_error,
        execution_steps=step,
    )


def started_logic() -> AutonomousSupervisorLogic:
    """Create and start one clean autonomous controller."""
    logic = AutonomousSupervisorLogic()

    decision = logic.process(
        make_snapshot()
    )

    assert decision.action.value == 'start'
    assert decision.state.value == 'executing'

    return logic


def test_nominal_snapshot_continues() -> None:
    """Nominal runtime data should permit continued execution."""
    logic = started_logic()

    decision = logic.process(
        make_snapshot(
            step=11,
            latest_perception_step=11,
        )
    )

    assert decision.action.value == 'continue'
    assert decision.state.value == 'executing'
    assert command_from_decision(decision) is None


def test_recoverable_uncertainty_requests_reacquisition() -> None:
    """Recoverable positional uncertainty should trigger reacquisition."""
    logic = started_logic()

    decision = logic.process(
        make_snapshot(
            step=11,
            latest_perception_step=11,
            sigma=0.020,
        )
    )

    assert decision.action.value == 'reacquire'
    assert command_from_decision(decision) == 'reacquire'

    assert (
        logic.controller.reacquisition_count
        == 1
    )


def test_recoverable_clearance_requests_replan() -> None:
    """Recoverable clearance concern should trigger replanning."""
    logic = started_logic()

    decision = logic.process(
        make_snapshot(
            step=11,
            latest_perception_step=11,
            clearance=0.001,
        )
    )

    assert decision.action.value == 'replan'
    assert command_from_decision(decision) == 'replan'

    assert logic.controller.replan_count == 1


def test_tracking_error_requests_recovery() -> None:
    """Recoverable tracking error should trigger recovery."""
    logic = started_logic()

    decision = logic.process(
        make_snapshot(
            step=11,
            latest_perception_step=11,
            tracking_error=0.010,
        )
    )

    assert decision.action.value == 'recover'
    assert command_from_decision(decision) == 'recover'

    assert logic.controller.recovery_count == 1


def test_critical_uncertainty_requests_stop() -> None:
    """Critical uncertainty should cause fail-safe stopping."""
    logic = started_logic()

    decision = logic.process(
        make_snapshot(
            step=11,
            latest_perception_step=11,
            sigma=0.040,
        )
    )

    assert decision.action.value == 'stop'
    assert decision.state.value == 'stopped'
    assert command_from_decision(decision) == 'stop'

    assert logic.controller.stop_count == 1
