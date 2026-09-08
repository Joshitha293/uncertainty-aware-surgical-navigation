"""Tests for the Phase 7 autonomous execution controller."""

from __future__ import annotations

import numpy as np
import pytest

from src.robotics.autonomous_execution_controller import (
    AutonomousControllerAction,
    AutonomousExecutionConfig,
    AutonomousExecutionController,
)
from src.robotics.runtime_safety_monitors import (
    RuntimeSafetySnapshot,
)
from src.robotics.safety_state_machine import (
    SafetyHazard,
    SafetyState,
)


def _snapshot(
    **overrides,
) -> RuntimeSafetySnapshot:
    """Return nominal runtime data."""

    values = {
        "step_index": 10,
        "latest_perception_step": 10,
        "maximum_principal_sigma": 0.005,
        "predicted_clearance": 0.010,
        "joint_positions": np.asarray(
            [0.0, 0.0, 0.0, 0.0],
            dtype=float,
        ),
        "joint_lower_limits": np.asarray(
            [-1.0, -1.0, -1.0, -1.0],
            dtype=float,
        ),
        "joint_upper_limits": np.asarray(
            [1.0, 1.0, 1.0, 1.0],
            dtype=float,
        ),
        "tracking_error": 0.001,
        "execution_steps": 10,
    }

    values.update(
        overrides
    )

    return RuntimeSafetySnapshot(
        **values
    )


def _controller(
    *,
    planner=lambda snapshot: True,
    reacquire=lambda snapshot: True,
    recover=lambda snapshot: True,
    execution_config=None,
) -> AutonomousExecutionController:
    """Return controller with deterministic test callbacks."""

    return AutonomousExecutionController(
        planner_callback=planner,
        reacquisition_callback=reacquire,
        recovery_callback=recover,
        execution_config=execution_config,
    )


def _started_controller(
    **kwargs,
) -> AutonomousExecutionController:
    """Return controller already in EXECUTING."""

    controller = _controller(
        **kwargs
    )

    decision = controller.start(
        _snapshot(
            step_index=0,
            latest_perception_step=0,
            execution_steps=0,
        )
    )

    assert (
        decision.action
        == AutonomousControllerAction.START
    )

    assert (
        controller.state
        == SafetyState.EXECUTING
    )

    return controller


def test_successful_start_enters_execution() -> None:
    """Initialisation and planning should reach EXECUTING."""

    controller = (
        _controller()
    )

    decision = controller.start(
        _snapshot(
            step_index=0,
            latest_perception_step=0,
            execution_steps=0,
        )
    )

    assert (
        decision.action
        == AutonomousControllerAction.START
    )

    assert (
        controller.state
        == SafetyState.EXECUTING
    )


def test_initial_planning_failure_enters_failed() -> None:
    """Initial planner failure must terminate the episode."""

    controller = (
        _controller(
            planner=lambda snapshot: False
        )
    )

    decision = controller.start(
        _snapshot(
            step_index=0,
            latest_perception_step=0,
            execution_steps=0,
        )
    )

    assert (
        decision.action
        == AutonomousControllerAction.FAIL
    )

    assert (
        controller.state
        == SafetyState.FAILED
    )


def test_safe_snapshot_continues() -> None:
    """Nominal state should continue execution."""

    controller = (
        _started_controller()
    )

    decision = (
        controller.process_snapshot(
            _snapshot()
        )
    )

    assert (
        decision.action
        == AutonomousControllerAction.CONTINUE
    )

    assert (
        controller.state
        == SafetyState.EXECUTING
    )


def test_stale_perception_reacquires_and_replans() -> None:
    """Recoverable stale perception should recover autonomously."""

    controller = (
        _started_controller()
    )

    decision = (
        controller.process_snapshot(
            _snapshot(
                step_index=10,
                latest_perception_step=5,
            )
        )
    )

    assert (
        decision.action
        == AutonomousControllerAction.REACQUIRE
    )

    assert (
        controller.state
        == SafetyState.EXECUTING
    )

    assert (
        controller.reacquisition_count
        == 1
    )

    assert (
        controller.replan_count
        == 1
    )


def test_failed_reacquisition_stops() -> None:
    """Persistent perception recovery failure must stop."""

    controller = (
        _started_controller(
            reacquire=lambda snapshot: False,
            execution_config=(
                AutonomousExecutionConfig(
                    maximum_reacquisition_attempts=2,
                    maximum_recovery_attempts=2,
                )
            ),
        )
    )

    decision = (
        controller.process_snapshot(
            _snapshot(
                step_index=10,
                latest_perception_step=5,
            )
        )
    )

    assert (
        decision.action
        == AutonomousControllerAction.STOP
    )

    assert (
        controller.state
        == SafetyState.STOPPED
    )

    assert (
        controller.reacquisition_count
        == 2
    )


def test_recoverable_uncertainty_reacquires() -> None:
    """Moderate uncertainty should trigger perception recovery."""

    controller = (
        _started_controller()
    )

    decision = (
        controller.process_snapshot(
            _snapshot(
                maximum_principal_sigma=0.020,
            )
        )
    )

    assert (
        decision.action
        == AutonomousControllerAction.REACQUIRE
    )

    assert (
        controller.state
        == SafetyState.EXECUTING
    )


def test_critical_uncertainty_stops() -> None:
    """Extreme uncertainty must produce a fail-safe stop."""

    controller = (
        _started_controller()
    )

    decision = (
        controller.process_snapshot(
            _snapshot(
                maximum_principal_sigma=0.035,
            )
        )
    )

    assert (
        decision.action
        == AutonomousControllerAction.STOP
    )

    assert (
        controller.state
        == SafetyState.STOPPED
    )


def test_recoverable_clearance_requests_replan() -> None:
    """Positive but small clearance should trigger replanning."""

    controller = (
        _started_controller()
    )

    decision = (
        controller.process_snapshot(
            _snapshot(
                predicted_clearance=0.001,
            )
        )
    )

    assert (
        decision.action
        == AutonomousControllerAction.REPLAN
    )

    assert (
        controller.state
        == SafetyState.EXECUTING
    )

    assert (
        controller.replan_count
        == 1
    )


def test_clearance_violation_stops() -> None:
    """Non-positive predicted clearance must stop."""

    controller = (
        _started_controller()
    )

    decision = (
        controller.process_snapshot(
            _snapshot(
                predicted_clearance=-0.001,
            )
        )
    )

    assert (
        decision.action
        == AutonomousControllerAction.STOP
    )

    assert (
        controller.state
        == SafetyState.STOPPED
    )


def test_tracking_error_recovers_and_replans() -> None:
    """Recoverable trajectory error should invoke recovery."""

    controller = (
        _started_controller()
    )

    decision = (
        controller.process_snapshot(
            _snapshot(
                tracking_error=0.007,
            )
        )
    )

    assert (
        decision.action
        == AutonomousControllerAction.RECOVER
    )

    assert (
        controller.recovery_count
        == 1
    )

    assert (
        controller.replan_count
        == 1
    )

    assert (
        controller.state
        == SafetyState.EXECUTING
    )


def test_failed_execution_recovery_stops() -> None:
    """Repeated low-level recovery failure must stop."""

    controller = (
        _started_controller(
            recover=lambda snapshot: False
        )
    )

    decision = (
        controller.process_snapshot(
            _snapshot(
                tracking_error=0.007,
            )
        )
    )

    assert (
        decision.action
        == AutonomousControllerAction.STOP
    )

    assert (
        controller.state
        == SafetyState.STOPPED
    )


def test_joint_limit_approach_recovers() -> None:
    """Near-limit joint state should invoke autonomous recovery."""

    controller = (
        _started_controller()
    )

    decision = (
        controller.process_snapshot(
            _snapshot(
                joint_positions=np.asarray(
                    [0.97, 0.0, 0.0, 0.0],
                    dtype=float,
                )
            )
        )
    )

    assert (
        decision.action
        == AutonomousControllerAction.RECOVER
    )

    assert (
        controller.state
        == SafetyState.EXECUTING
    )


def test_joint_limit_violation_stops() -> None:
    """Hard joint-limit violation must terminate execution."""

    controller = (
        _started_controller()
    )

    decision = (
        controller.process_snapshot(
            _snapshot(
                joint_positions=np.asarray(
                    [1.01, 0.0, 0.0, 0.0],
                    dtype=float,
                )
            )
        )
    )

    assert (
        decision.action
        == AutonomousControllerAction.STOP
    )


def test_timeout_stops() -> None:
    """Execution timeout is a terminal safety condition."""

    controller = (
        _started_controller()
    )

    decision = (
        controller.process_snapshot(
            _snapshot(
                execution_steps=60,
            )
        )
    )

    assert (
        decision.action
        == AutonomousControllerAction.STOP
    )

    assert (
        decision.triggering_event
        is not None
    )

    assert (
        decision.triggering_event.hazard
        == SafetyHazard.EXECUTION_TIMEOUT
    )


def test_invalid_data_stops() -> None:
    """Non-finite runtime data must stop autonomous execution."""

    controller = (
        _started_controller()
    )

    decision = (
        controller.process_snapshot(
            _snapshot(
                tracking_error=float(
                    "nan"
                )
            )
        )
    )

    assert (
        decision.action
        == AutonomousControllerAction.STOP
    )

    assert (
        controller.state
        == SafetyState.STOPPED
    )


def test_replan_failure_enters_failed() -> None:
    """Failure to find replacement trajectory must enter FAILED."""

    call_count = {
        "value": 0
    }

    def planner(
        snapshot,
    ):
        call_count[
            "value"
        ] += 1

        return (
            call_count[
                "value"
            ]
            == 1
        )

    controller = (
        _started_controller(
            planner=planner
        )
    )

    decision = (
        controller.process_snapshot(
            _snapshot(
                predicted_clearance=0.001,
            )
        )
    )

    assert (
        decision.action
        == AutonomousControllerAction.FAIL
    )

    assert (
        controller.state
        == SafetyState.FAILED
    )


def test_successful_completion_is_terminal() -> None:
    """Completed task must enter COMPLETED."""

    controller = (
        _started_controller()
    )

    decision = (
        controller.complete(
            step_index=25
        )
    )

    assert (
        decision.action
        == AutonomousControllerAction.COMPLETE
    )

    assert (
        controller.state
        == SafetyState.COMPLETED
    )

    assert (
        controller.state_machine.is_terminal
    )


def test_snapshot_processing_requires_execution_state() -> None:
    """Snapshots cannot be processed before execution begins."""

    controller = (
        _controller()
    )

    with pytest.raises(
        RuntimeError
    ):
        controller.process_snapshot(
            _snapshot()
        )


def test_start_cannot_be_called_twice() -> None:
    """A running episode cannot be restarted silently."""

    controller = (
        _started_controller()
    )

    with pytest.raises(
        RuntimeError
    ):
        controller.start(
            _snapshot()
        )


def test_invalid_recovery_configuration_rejected() -> None:
    """Recovery attempt budgets must be positive."""

    with pytest.raises(
        ValueError
    ):
        AutonomousExecutionConfig(
            maximum_reacquisition_attempts=0
        )