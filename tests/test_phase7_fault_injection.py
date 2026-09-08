"""Verification of Phase 7 controlled fault injection."""

from __future__ import annotations

import numpy as np
import pytest

from src.robotics.autonomous_execution_controller import (
    AutonomousControllerAction,
    AutonomousExecutionController,
)
from src.robotics.runtime_safety_monitors import (
    RuntimeSafetyConfig,
    evaluate_runtime_safety,
)
from src.robotics.safety_state_machine import (
    SafetyHazard,
    SafetySeverity,
    SafetyState,
)
from src.simulation.phase7_fault_injection import (
    FaultType,
    all_fault_types,
    inject_fault,
    nominal_runtime_snapshot,
)


def _controller(
    *,
    planner=lambda snapshot: True,
    reacquire=lambda snapshot: True,
    recover=lambda snapshot: True,
) -> AutonomousExecutionController:
    """Return deterministic controller for fault-response tests."""

    controller = AutonomousExecutionController(
        planner_callback=planner,
        reacquisition_callback=reacquire,
        recovery_callback=recover,
    )

    controller.start(
        nominal_runtime_snapshot(
            step_index=0
        )
    )

    return controller


def test_nominal_snapshot_has_no_hazard() -> None:
    """Baseline snapshot must be safety-monitor clean."""

    snapshot = (
        nominal_runtime_snapshot()
    )

    assert (
        evaluate_runtime_safety(
            snapshot
        )
        == ()
    )


def test_frozen_fault_set_contains_all_required_faults() -> None:
    """Phase 7 verification must expose all intended fault classes."""

    faults = set(
        all_fault_types()
    )

    assert (
        FaultType.STALE_PERCEPTION
        in faults
    )

    assert (
        FaultType.UNCERTAINTY_RECOVERABLE
        in faults
    )

    assert (
        FaultType.UNCERTAINTY_CRITICAL
        in faults
    )

    assert (
        FaultType.CLEARANCE_RECOVERABLE
        in faults
    )

    assert (
        FaultType.CLEARANCE_CRITICAL
        in faults
    )

    assert (
        FaultType.JOINT_LIMIT_APPROACH
        in faults
    )

    assert (
        FaultType.JOINT_LIMIT_VIOLATION
        in faults
    )

    assert (
        FaultType.TRACKING_ERROR_RECOVERABLE
        in faults
    )

    assert (
        FaultType.TRACKING_ERROR_CRITICAL
        in faults
    )

    assert (
        FaultType.INVALID_NUMERICAL_DATA
        in faults
    )

    assert (
        FaultType.EXECUTION_TIMEOUT
        in faults
    )


def test_none_fault_preserves_snapshot() -> None:
    """NONE must leave nominal runtime values unchanged."""

    original = (
        nominal_runtime_snapshot()
    )

    injected = (
        inject_fault(
            original,
            FaultType.NONE,
        ).snapshot
    )

    assert (
        injected.step_index
        == original.step_index
    )

    assert (
        injected.latest_perception_step
        == original.latest_perception_step
    )

    assert np.allclose(
        injected.joint_positions,
        original.joint_positions,
    )

    assert np.isclose(
        injected.tracking_error,
        original.tracking_error,
    )


@pytest.mark.parametrize(
    "fault",
    all_fault_types(),
)
def test_every_injected_fault_is_detected(
    fault: FaultType,
) -> None:
    """Every frozen injected fault must activate at least one monitor."""

    snapshot = (
        nominal_runtime_snapshot()
    )

    injected = (
        inject_fault(
            snapshot,
            fault,
        )
    )

    events = (
        evaluate_runtime_safety(
            injected.snapshot
        )
    )

    assert len(
        events
    ) >= 1


def test_stale_perception_maps_to_expected_hazard() -> None:
    """Injected perception staleness must be detected correctly."""

    record = (
        inject_fault(
            nominal_runtime_snapshot(),
            FaultType.STALE_PERCEPTION,
        )
    )

    events = (
        evaluate_runtime_safety(
            record.snapshot
        )
    )

    assert (
        events[
            0
        ].hazard
        == SafetyHazard.STALE_PERCEPTION
    )

    assert (
        events[
            0
        ].severity
        == SafetySeverity.RECOVERABLE
    )


def test_recoverable_uncertainty_is_not_critical() -> None:
    """Moderate uncertainty must remain a recovery condition."""

    record = (
        inject_fault(
            nominal_runtime_snapshot(),
            FaultType.UNCERTAINTY_RECOVERABLE,
        )
    )

    events = (
        evaluate_runtime_safety(
            record.snapshot
        )
    )

    assert any(
        event.hazard
        == SafetyHazard.EXCESSIVE_UNCERTAINTY
        and event.severity
        == SafetySeverity.RECOVERABLE
        for event in events
    )


def test_critical_uncertainty_is_critical() -> None:
    """Severe uncertainty must activate critical response."""

    record = (
        inject_fault(
            nominal_runtime_snapshot(),
            FaultType.UNCERTAINTY_CRITICAL,
        )
    )

    events = (
        evaluate_runtime_safety(
            record.snapshot
        )
    )

    assert any(
        event.hazard
        == SafetyHazard.EXCESSIVE_UNCERTAINTY
        and event.severity
        == SafetySeverity.CRITICAL
        for event in events
    )


def test_recoverable_clearance_triggers_replan() -> None:
    """Small positive clearance must request replanning."""

    controller = (
        _controller()
    )

    record = (
        inject_fault(
            nominal_runtime_snapshot(),
            FaultType.CLEARANCE_RECOVERABLE,
        )
    )

    decision = (
        controller.process_snapshot(
            record.snapshot
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


def test_critical_clearance_stops() -> None:
    """Critical protected-region clearance must stop."""

    controller = (
        _controller()
    )

    record = (
        inject_fault(
            nominal_runtime_snapshot(),
            FaultType.CLEARANCE_CRITICAL,
        )
    )

    decision = (
        controller.process_snapshot(
            record.snapshot
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
    """Near-limit joint fault must invoke recovery."""

    controller = (
        _controller()
    )

    record = (
        inject_fault(
            nominal_runtime_snapshot(),
            FaultType.JOINT_LIMIT_APPROACH,
        )
    )

    decision = (
        controller.process_snapshot(
            record.snapshot
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
    """Hard joint-limit violation must stop execution."""

    controller = (
        _controller()
    )

    record = (
        inject_fault(
            nominal_runtime_snapshot(),
            FaultType.JOINT_LIMIT_VIOLATION,
        )
    )

    decision = (
        controller.process_snapshot(
            record.snapshot
        )
    )

    assert (
        decision.action
        == AutonomousControllerAction.STOP
    )


def test_recoverable_tracking_fault_recovers() -> None:
    """Moderate tracking error must invoke recovery."""

    controller = (
        _controller()
    )

    record = (
        inject_fault(
            nominal_runtime_snapshot(),
            FaultType.TRACKING_ERROR_RECOVERABLE,
        )
    )

    decision = (
        controller.process_snapshot(
            record.snapshot
        )
    )

    assert (
        decision.action
        == AutonomousControllerAction.RECOVER
    )


def test_critical_tracking_fault_stops() -> None:
    """Severe tracking error must stop."""

    controller = (
        _controller()
    )

    record = (
        inject_fault(
            nominal_runtime_snapshot(),
            FaultType.TRACKING_ERROR_CRITICAL,
        )
    )

    decision = (
        controller.process_snapshot(
            record.snapshot
        )
    )

    assert (
        decision.action
        == AutonomousControllerAction.STOP
    )


def test_stale_perception_reacquires() -> None:
    """Temporary perception loss must invoke reacquisition."""

    controller = (
        _controller()
    )

    record = (
        inject_fault(
            nominal_runtime_snapshot(),
            FaultType.STALE_PERCEPTION,
        )
    )

    decision = (
        controller.process_snapshot(
            record.snapshot
        )
    )

    assert (
        decision.action
        == AutonomousControllerAction.REACQUIRE
    )

    assert (
        controller.reacquisition_count
        >= 1
    )


def test_recoverable_uncertainty_reacquires() -> None:
    """Recoverable uncertainty must invoke perception recovery."""

    controller = (
        _controller()
    )

    record = (
        inject_fault(
            nominal_runtime_snapshot(),
            FaultType.UNCERTAINTY_RECOVERABLE,
        )
    )

    decision = (
        controller.process_snapshot(
            record.snapshot
        )
    )

    assert (
        decision.action
        == AutonomousControllerAction.REACQUIRE
    )


def test_critical_uncertainty_stops_controller() -> None:
    """Critical uncertainty must terminate autonomous execution."""

    controller = (
        _controller()
    )

    record = (
        inject_fault(
            nominal_runtime_snapshot(),
            FaultType.UNCERTAINTY_CRITICAL,
        )
    )

    decision = (
        controller.process_snapshot(
            record.snapshot
        )
    )

    assert (
        decision.action
        == AutonomousControllerAction.STOP
    )


def test_invalid_numerical_data_stops() -> None:
    """NaN state data must fail safely."""

    controller = (
        _controller()
    )

    record = (
        inject_fault(
            nominal_runtime_snapshot(),
            FaultType.INVALID_NUMERICAL_DATA,
        )
    )

    decision = (
        controller.process_snapshot(
            record.snapshot
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
        == SafetyHazard.INVALID_DATA
    )


def test_timeout_stops() -> None:
    """Timeout fault must terminate execution."""

    controller = (
        _controller()
    )

    record = (
        inject_fault(
            nominal_runtime_snapshot(),
            FaultType.EXECUTION_TIMEOUT,
        )
    )

    decision = (
        controller.process_snapshot(
            record.snapshot
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


def test_persistent_reacquisition_failure_stops() -> None:
    """A recoverable perception fault becomes terminal if recovery fails."""

    controller = (
        _controller(
            reacquire=lambda snapshot: False
        )
    )

    record = (
        inject_fault(
            nominal_runtime_snapshot(),
            FaultType.STALE_PERCEPTION,
        )
    )

    decision = (
        controller.process_snapshot(
            record.snapshot
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


def test_persistent_execution_recovery_failure_stops() -> None:
    """A recoverable tracking fault becomes terminal if recovery fails."""

    controller = (
        _controller(
            recover=lambda snapshot: False
        )
    )

    record = (
        inject_fault(
            nominal_runtime_snapshot(),
            FaultType.TRACKING_ERROR_RECOVERABLE,
        )
    )

    decision = (
        controller.process_snapshot(
            record.snapshot
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


def test_replan_failure_enters_failed_state() -> None:
    """Recoverable clearance cannot resume when replanning fails."""

    calls = {
        "count": 0
    }

    def planner(
        snapshot,
    ) -> bool:
        calls[
            "count"
        ] += 1

        return (
            calls[
                "count"
            ]
            == 1
        )

    controller = (
        _controller(
            planner=planner
        )
    )

    record = (
        inject_fault(
            nominal_runtime_snapshot(),
            FaultType.CLEARANCE_RECOVERABLE,
        )
    )

    decision = (
        controller.process_snapshot(
            record.snapshot
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


def test_fault_injection_is_deterministic() -> None:
    """Same input and fault must generate identical injected snapshots."""

    snapshot = (
        nominal_runtime_snapshot()
    )

    first = (
        inject_fault(
            snapshot,
            FaultType.JOINT_LIMIT_APPROACH,
        )
    )

    second = (
        inject_fault(
            snapshot,
            FaultType.JOINT_LIMIT_APPROACH,
        )
    )

    assert np.allclose(
        first.snapshot.joint_positions,
        second.snapshot.joint_positions,
    )

    assert (
        first.description
        == second.description
    )


def test_original_snapshot_is_not_mutated() -> None:
    """Fault injection must not change the caller's original snapshot."""

    original = (
        nominal_runtime_snapshot()
    )

    original_q = (
        original.joint_positions.copy()
    )

    inject_fault(
        original,
        FaultType.JOINT_LIMIT_VIOLATION,
    )

    assert np.allclose(
        original.joint_positions,
        original_q,
    )


def test_invalid_fault_argument_rejected() -> None:
    """Fault type must use the declared enumeration."""

    with pytest.raises(
        TypeError
    ):
        inject_fault(
            nominal_runtime_snapshot(),
            "stale",
        )


def test_custom_monitor_thresholds_are_respected() -> None:
    """Fault injection should derive values from supplied monitor config."""

    config = RuntimeSafetyConfig(
        uncertainty_reacquire_sigma=0.010,
        uncertainty_stop_sigma=0.020,
    )

    record = (
        inject_fault(
            nominal_runtime_snapshot(),
            FaultType.UNCERTAINTY_RECOVERABLE,
            config=config,
        )
    )

    assert np.isclose(
        record.snapshot.maximum_principal_sigma,
        0.010,
    )