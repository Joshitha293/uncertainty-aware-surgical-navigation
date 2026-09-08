"""Tests for the Phase 7 safety state machine."""

from __future__ import annotations

import pytest

from src.robotics.safety_state_machine import (
    SafetyEvent,
    SafetyHazard,
    SafetySeverity,
    SafetyState,
    SafetyStateMachine,
    recommended_state_for_event,
)


def _advance_to_executing(
    machine: SafetyStateMachine,
) -> None:
    """Move a machine through the normal startup sequence."""

    machine.transition(
        SafetyState.INITIALISING,
        reason="Initialisation requested.",
    )

    machine.transition(
        SafetyState.READY,
        reason="Initialisation successful.",
    )

    machine.transition(
        SafetyState.PLANNING,
        reason="Planning requested.",
    )

    machine.transition(
        SafetyState.EXECUTING,
        reason="Plan accepted.",
    )


def test_machine_starts_idle() -> None:
    """Default state must be IDLE."""

    machine = (
        SafetyStateMachine()
    )

    assert (
        machine.state
        == SafetyState.IDLE
    )

    assert not (
        machine.is_terminal
    )

    assert (
        machine.history
        == ()
    )


def test_normal_startup_sequence() -> None:
    """Normal startup must reach EXECUTING."""

    machine = (
        SafetyStateMachine()
    )

    _advance_to_executing(
        machine
    )

    assert (
        machine.state
        == SafetyState.EXECUTING
    )

    assert (
        len(
            machine.history
        )
        == 4
    )


def test_forbidden_idle_to_executing_transition() -> None:
    """Autonomous execution cannot bypass initialisation and planning."""

    machine = (
        SafetyStateMachine()
    )

    with pytest.raises(
        ValueError
    ):
        machine.transition(
            SafetyState.EXECUTING,
            reason="Invalid shortcut.",
        )


def test_execution_can_complete() -> None:
    """Successful execution can enter COMPLETED."""

    machine = (
        SafetyStateMachine()
    )

    _advance_to_executing(
        machine
    )

    machine.transition(
        SafetyState.COMPLETED,
        reason="Goal reached.",
    )

    assert (
        machine.state
        == SafetyState.COMPLETED
    )

    assert (
        machine.is_terminal
    )


def test_terminal_state_cannot_resume_automatically() -> None:
    """COMPLETED must not transition directly back to execution."""

    machine = (
        SafetyStateMachine()
    )

    _advance_to_executing(
        machine
    )

    machine.transition(
        SafetyState.COMPLETED,
        reason="Goal reached.",
    )

    with pytest.raises(
        RuntimeError
    ):
        machine.transition(
            SafetyState.EXECUTING,
            reason="Invalid restart.",
        )


def test_reset_requires_terminal_state() -> None:
    """Active execution cannot be silently reset."""

    machine = (
        SafetyStateMachine()
    )

    with pytest.raises(
        RuntimeError
    ):
        machine.reset()


def test_terminal_reset_returns_to_idle() -> None:
    """Explicit reset begins a new independent episode."""

    machine = (
        SafetyStateMachine()
    )

    machine.transition(
        SafetyState.STOPPED,
        reason="Manual stop.",
    )

    assert (
        machine.is_terminal
    )

    machine.reset()

    assert (
        machine.state
        == SafetyState.IDLE
    )

    assert (
        machine.history
        == ()
    )


def test_stale_perception_requests_reacquisition() -> None:
    """Recoverable stale perception should request REACQUIRING."""

    event = SafetyEvent(
        hazard=(
            SafetyHazard.STALE_PERCEPTION
        ),
        severity=(
            SafetySeverity.RECOVERABLE
        ),
        message="Perception observation is stale.",
        step_index=10,
    )

    assert (
        recommended_state_for_event(
            event
        )
        == SafetyState.REACQUIRING
    )


def test_excessive_uncertainty_requests_reacquisition() -> None:
    """Recoverable localisation uncertainty should request reacquisition."""

    event = SafetyEvent(
        hazard=(
            SafetyHazard.EXCESSIVE_UNCERTAINTY
        ),
        severity=(
            SafetySeverity.RECOVERABLE
        ),
        message="Estimated uncertainty exceeded recovery threshold.",
    )

    assert (
        recommended_state_for_event(
            event
        )
        == SafetyState.REACQUIRING
    )


def test_tracking_error_requests_recovery() -> None:
    """Recoverable tracking error should request RECOVERING."""

    event = SafetyEvent(
        hazard=(
            SafetyHazard.TRACKING_ERROR
        ),
        severity=(
            SafetySeverity.RECOVERABLE
        ),
        message="Tracking error exceeded nominal tolerance.",
    )

    assert (
        recommended_state_for_event(
            event
        )
        == SafetyState.RECOVERING
    )


def test_planning_failure_requests_failed_state() -> None:
    """Unrecoverable planning failure maps to FAILED."""

    event = SafetyEvent(
        hazard=(
            SafetyHazard.PLANNING_FAILURE
        ),
        severity=(
            SafetySeverity.RECOVERABLE
        ),
        message="Planner failed to produce a trajectory.",
    )

    assert (
        recommended_state_for_event(
            event
        )
        == SafetyState.FAILED
    )


def test_critical_event_always_requests_stop() -> None:
    """Critical severity overrides recoverable hazard category."""

    event = SafetyEvent(
        hazard=(
            SafetyHazard.STALE_PERCEPTION
        ),
        severity=(
            SafetySeverity.CRITICAL
        ),
        message="Persistent perception loss.",
    )

    assert (
        recommended_state_for_event(
            event
        )
        == SafetyState.STOPPED
    )


def test_emergency_stop_from_execution() -> None:
    """Emergency-stop event must terminate active execution."""

    machine = (
        SafetyStateMachine()
    )

    _advance_to_executing(
        machine
    )

    event = SafetyEvent(
        hazard=(
            SafetyHazard.EMERGENCY_STOP
        ),
        severity=(
            SafetySeverity.CRITICAL
        ),
        message="Emergency stop requested.",
        step_index=12,
    )

    transition = (
        machine.handle_event(
            event
        )
    )

    assert (
        transition is not None
    )

    assert (
        machine.state
        == SafetyState.STOPPED
    )

    assert (
        transition.hazard
        == SafetyHazard.EMERGENCY_STOP
    )

    assert (
        transition.step_index
        == 12
    )


def test_stale_perception_moves_execution_to_reacquiring() -> None:
    """Recoverable stale perception should alter execution state."""

    machine = (
        SafetyStateMachine()
    )

    _advance_to_executing(
        machine
    )

    event = SafetyEvent(
        hazard=(
            SafetyHazard.STALE_PERCEPTION
        ),
        severity=(
            SafetySeverity.RECOVERABLE
        ),
        message="Perception age exceeded threshold.",
        step_index=8,
    )

    machine.handle_event(
        event
    )

    assert (
        machine.state
        == SafetyState.REACQUIRING
    )


def test_tracking_error_moves_execution_to_recovering() -> None:
    """Recoverable tracking deviation should initiate RECOVERING."""

    machine = (
        SafetyStateMachine()
    )

    _advance_to_executing(
        machine
    )

    event = SafetyEvent(
        hazard=(
            SafetyHazard.TRACKING_ERROR
        ),
        severity=(
            SafetySeverity.RECOVERABLE
        ),
        message="Trajectory tracking deviation detected.",
        step_index=7,
    )

    machine.handle_event(
        event
    )

    assert (
        machine.state
        == SafetyState.RECOVERING
    )


def test_advisory_event_does_not_change_state() -> None:
    """Advisory events should not force a state transition."""

    machine = (
        SafetyStateMachine()
    )

    _advance_to_executing(
        machine
    )

    history_before = (
        machine.history
    )

    event = SafetyEvent(
        hazard=(
            SafetyHazard.JOINT_LIMIT_APPROACH
        ),
        severity=(
            SafetySeverity.ADVISORY
        ),
        message="Joint is approaching warning region.",
    )

    transition = (
        machine.handle_event(
            event
        )
    )

    assert (
        transition is None
    )

    assert (
        machine.state
        == SafetyState.EXECUTING
    )

    assert (
        machine.history
        == history_before
    )


def test_recovery_can_return_to_execution() -> None:
    """Successful recovery may resume execution."""

    machine = (
        SafetyStateMachine()
    )

    _advance_to_executing(
        machine
    )

    machine.transition(
        SafetyState.RECOVERING,
        reason="Tracking recovery required.",
    )

    machine.transition(
        SafetyState.EXECUTING,
        reason="Tracking recovery successful.",
    )

    assert (
        machine.state
        == SafetyState.EXECUTING
    )


def test_reacquisition_can_lead_to_replanning() -> None:
    """Updated perception can trigger a fresh trajectory plan."""

    machine = (
        SafetyStateMachine()
    )

    _advance_to_executing(
        machine
    )

    machine.transition(
        SafetyState.REACQUIRING,
        reason="Perception reacquisition required.",
    )

    machine.transition(
        SafetyState.REPLANNING,
        reason="Updated perception available.",
    )

    machine.transition(
        SafetyState.EXECUTING,
        reason="Updated trajectory accepted.",
    )

    assert (
        machine.state
        == SafetyState.EXECUTING
    )


def test_negative_event_step_is_rejected() -> None:
    """Safety events cannot use negative execution indices."""

    with pytest.raises(
        ValueError
    ):
        SafetyEvent(
            hazard=(
                SafetyHazard.TRACKING_ERROR
            ),
            severity=(
                SafetySeverity.RECOVERABLE
            ),
            message="Invalid event.",
            step_index=-1,
        )


def test_empty_event_message_is_rejected() -> None:
    """Safety events require auditable reasons."""

    with pytest.raises(
        ValueError
    ):
        SafetyEvent(
            hazard=(
                SafetyHazard.INVALID_DATA
            ),
            severity=(
                SafetySeverity.CRITICAL
            ),
            message="",
        )


def test_transition_history_is_auditable() -> None:
    """Transition history must retain state, hazard and step metadata."""

    machine = (
        SafetyStateMachine()
    )

    _advance_to_executing(
        machine
    )

    event = SafetyEvent(
        hazard=(
            SafetyHazard.CLEARANCE_VIOLATION
        ),
        severity=(
            SafetySeverity.CRITICAL
        ),
        message="Protected clearance violated.",
        step_index=14,
    )

    machine.handle_event(
        event
    )

    final_transition = (
        machine.history[
            -1
        ]
    )

    assert (
        final_transition.from_state
        == SafetyState.EXECUTING
    )

    assert (
        final_transition.to_state
        == SafetyState.STOPPED
    )

    assert (
        final_transition.hazard
        == SafetyHazard.CLEARANCE_VIOLATION
    )

    assert (
        final_transition.step_index
        == 14
    )