"""Safety-oriented autonomous execution state machine.

This module provides the deterministic supervisory state model used by
Phase 7 of the simulated surgical-navigation research platform.

It defines:

- execution states;
- safety hazards;
- hazard severity;
- safety events;
- auditable state transitions;
- allowed and forbidden transitions;
- high-level hazard-response behaviour.

The implementation is simulation/research software. The state machine and
hazard-response policy do not establish medical-device safety, clinical
safety, or suitability for autonomous surgery.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SafetyState(str, Enum):
    """Discrete states of autonomous task execution."""

    IDLE = "idle"
    INITIALISING = "initialising"
    READY = "ready"
    PLANNING = "planning"
    EXECUTING = "executing"

    REPLANNING = "replanning"
    REACQUIRING = "reacquiring"
    RECOVERING = "recovering"

    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


class SafetyHazard(str, Enum):
    """Safety-relevant runtime hazards."""

    STALE_PERCEPTION = "stale_perception"
    EXCESSIVE_UNCERTAINTY = "excessive_uncertainty"

    CLEARANCE_VIOLATION = "clearance_violation"

    JOINT_LIMIT_APPROACH = "joint_limit_approach"
    TRACKING_ERROR = "tracking_error"

    PLANNING_FAILURE = "planning_failure"

    EXECUTION_TIMEOUT = "execution_timeout"

    INVALID_STATE = "invalid_state"
    INVALID_DATA = "invalid_data"

    EMERGENCY_STOP = "emergency_stop"


class SafetySeverity(str, Enum):
    """Severity associated with a safety event."""

    ADVISORY = "advisory"
    RECOVERABLE = "recoverable"
    CRITICAL = "critical"


@dataclass(frozen=True)
class SafetyEvent:
    """One safety-relevant runtime observation."""

    hazard: SafetyHazard

    severity: SafetySeverity

    message: str

    step_index: int | None = None

    def __post_init__(
        self,
    ) -> None:
        """Validate safety-event metadata."""

        if not isinstance(
            self.hazard,
            SafetyHazard,
        ):
            raise TypeError(
                "hazard must be a SafetyHazard."
            )

        if not isinstance(
            self.severity,
            SafetySeverity,
        ):
            raise TypeError(
                "severity must be a SafetySeverity."
            )

        if (
            not isinstance(
                self.message,
                str,
            )
            or not self.message.strip()
        ):
            raise ValueError(
                "message must be a non-empty string."
            )

        if (
            self.step_index is not None
            and self.step_index < 0
        ):
            raise ValueError(
                "step_index must be non-negative when provided."
            )


@dataclass(frozen=True)
class SafetyTransition:
    """Auditable transition between execution states."""

    from_state: SafetyState

    to_state: SafetyState

    reason: str

    hazard: SafetyHazard | None = None

    step_index: int | None = None

    def __post_init__(
        self,
    ) -> None:
        """Validate transition metadata."""

        if not isinstance(
            self.from_state,
            SafetyState,
        ):
            raise TypeError(
                "from_state must be a SafetyState."
            )

        if not isinstance(
            self.to_state,
            SafetyState,
        ):
            raise TypeError(
                "to_state must be a SafetyState."
            )

        if (
            not isinstance(
                self.reason,
                str,
            )
            or not self.reason.strip()
        ):
            raise ValueError(
                "reason must be a non-empty string."
            )

        if (
            self.hazard is not None
            and not isinstance(
                self.hazard,
                SafetyHazard,
            )
        ):
            raise TypeError(
                "hazard must be a SafetyHazard or None."
            )

        if (
            self.step_index is not None
            and self.step_index < 0
        ):
            raise ValueError(
                "step_index must be non-negative when provided."
            )


# ---------------------------------------------------------------------------
# State-transition model
# ---------------------------------------------------------------------------

_ALLOWED_TRANSITIONS: dict[
    SafetyState,
    frozenset[
        SafetyState
    ],
] = {
    SafetyState.IDLE: frozenset(
        {
            SafetyState.INITIALISING,
            SafetyState.STOPPED,
        }
    ),

    SafetyState.INITIALISING: frozenset(
        {
            SafetyState.READY,
            SafetyState.FAILED,
            SafetyState.STOPPED,
        }
    ),

    SafetyState.READY: frozenset(
        {
            SafetyState.PLANNING,
            SafetyState.FAILED,
            SafetyState.STOPPED,
        }
    ),

    SafetyState.PLANNING: frozenset(
        {
            SafetyState.EXECUTING,
            SafetyState.REPLANNING,
            SafetyState.FAILED,
            SafetyState.STOPPED,
        }
    ),

    SafetyState.EXECUTING: frozenset(
        {
            SafetyState.REPLANNING,
            SafetyState.REACQUIRING,
            SafetyState.RECOVERING,
            SafetyState.COMPLETED,
            SafetyState.FAILED,
            SafetyState.STOPPED,
        }
    ),

    SafetyState.REPLANNING: frozenset(
        {
            SafetyState.EXECUTING,
            SafetyState.REACQUIRING,
            SafetyState.RECOVERING,
            SafetyState.FAILED,
            SafetyState.STOPPED,
        }
    ),

    SafetyState.REACQUIRING: frozenset(
        {
            SafetyState.REPLANNING,
            SafetyState.RECOVERING,
            SafetyState.FAILED,
            SafetyState.STOPPED,
        }
    ),

    SafetyState.RECOVERING: frozenset(
        {
            SafetyState.EXECUTING,
            SafetyState.REPLANNING,
            SafetyState.REACQUIRING,
            SafetyState.FAILED,
            SafetyState.STOPPED,
        }
    ),

    # Terminal states deliberately have no automatic outgoing transitions.
    # reset() must be called to begin a new independent episode.
    SafetyState.STOPPED: frozenset(),

    SafetyState.COMPLETED: frozenset(),

    SafetyState.FAILED: frozenset(),
}


_TERMINAL_STATES = frozenset(
    {
        SafetyState.STOPPED,
        SafetyState.COMPLETED,
        SafetyState.FAILED,
    }
)


# ---------------------------------------------------------------------------
# Hazard-response policy
# ---------------------------------------------------------------------------

def recommended_state_for_event(
    event: SafetyEvent,
) -> SafetyState | None:
    """Return the high-level response associated with one safety event.

    Policy
    ------

    ADVISORY
        No forced state transition.

    CRITICAL
        STOPPED.

    Recoverable perception faults
        REACQUIRING.

    Recoverable protected-clearance concern
        REPLANNING.

    Recoverable tracking / joint-limit concern
        RECOVERING.

    Planning failure
        FAILED.

    Timeout, invalid data/state, and emergency-stop hazards
        STOPPED.

    The returned state is only a requested high-level response. Actual state
    transitions remain subject to the deterministic transition table.
    """

    if not isinstance(
        event,
        SafetyEvent,
    ):
        raise TypeError(
            "event must be a SafetyEvent."
        )

    # Critical severity has highest priority regardless of hazard category.
    if (
        event.severity
        == SafetySeverity.CRITICAL
    ):
        return SafetyState.STOPPED

    if (
        event.severity
        == SafetySeverity.ADVISORY
    ):
        return None

    if event.hazard in {
        SafetyHazard.STALE_PERCEPTION,
        SafetyHazard.EXCESSIVE_UNCERTAINTY,
    }:
        return SafetyState.REACQUIRING

    if (
        event.hazard
        == SafetyHazard.CLEARANCE_VIOLATION
    ):
        return SafetyState.REPLANNING

    if event.hazard in {
        SafetyHazard.JOINT_LIMIT_APPROACH,
        SafetyHazard.TRACKING_ERROR,
    }:
        return SafetyState.RECOVERING

    if (
        event.hazard
        == SafetyHazard.PLANNING_FAILURE
    ):
        return SafetyState.FAILED

    if event.hazard in {
        SafetyHazard.EXECUTION_TIMEOUT,
        SafetyHazard.INVALID_STATE,
        SafetyHazard.INVALID_DATA,
        SafetyHazard.EMERGENCY_STOP,
    }:
        return SafetyState.STOPPED

    raise ValueError(
        f"No response policy exists for hazard {event.hazard!r}."
    )


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class SafetyStateMachine:
    """Deterministic autonomous-execution safety state machine."""

    def __init__(
        self,
        initial_state: SafetyState = SafetyState.IDLE,
    ) -> None:
        """Create a new safety state machine."""

        if not isinstance(
            initial_state,
            SafetyState,
        ):
            raise TypeError(
                "initial_state must be a SafetyState."
            )

        self._state = (
            initial_state
        )

        self._history: list[
            SafetyTransition
        ] = []

    @property
    def state(
        self,
    ) -> SafetyState:
        """Return current execution state."""

        return self._state

    @property
    def history(
        self,
    ) -> tuple[
        SafetyTransition,
        ...,
    ]:
        """Return immutable transition history."""

        return tuple(
            self._history
        )

    @property
    def is_terminal(
        self,
    ) -> bool:
        """Return whether the machine is in a terminal state."""

        return bool(
            self._state
            in _TERMINAL_STATES
        )

    def can_transition(
        self,
        target_state: SafetyState,
    ) -> bool:
        """Return whether target_state is currently allowed."""

        if not isinstance(
            target_state,
            SafetyState,
        ):
            raise TypeError(
                "target_state must be a SafetyState."
            )

        return bool(
            target_state
            in _ALLOWED_TRANSITIONS[
                self._state
            ]
        )

    def transition(
        self,
        target_state: SafetyState,
        *,
        reason: str,
        hazard: SafetyHazard | None = None,
        step_index: int | None = None,
    ) -> SafetyTransition:
        """Perform one validated and auditable state transition."""

        if not isinstance(
            target_state,
            SafetyState,
        ):
            raise TypeError(
                "target_state must be a SafetyState."
            )

        if (
            not isinstance(
                reason,
                str,
            )
            or not reason.strip()
        ):
            raise ValueError(
                "reason must be a non-empty string."
            )

        if (
            hazard is not None
            and not isinstance(
                hazard,
                SafetyHazard,
            )
        ):
            raise TypeError(
                "hazard must be a SafetyHazard or None."
            )

        if (
            step_index is not None
            and step_index < 0
        ):
            raise ValueError(
                "step_index must be non-negative when provided."
            )

        if self.is_terminal:
            raise RuntimeError(
                "Terminal safety states cannot transition automatically. "
                "Call reset() to begin a new independent episode."
            )

        if not self.can_transition(
            target_state
        ):
            raise ValueError(
                "Forbidden safety-state transition: "
                f"{self._state.value} -> {target_state.value}."
            )

        transition = SafetyTransition(
            from_state=(
                self._state
            ),
            to_state=(
                target_state
            ),
            reason=reason,
            hazard=hazard,
            step_index=step_index,
        )

        self._state = (
            target_state
        )

        self._history.append(
            transition
        )

        return transition

    def handle_event(
        self,
        event: SafetyEvent,
    ) -> SafetyTransition | None:
        """Apply the state response associated with one safety event.

        Advisory events do not force a transition.

        Recoverable and critical events are mapped to an intended state using
        recommended_state_for_event(). The requested transition must still be
        valid according to the state-transition table.

        This prevents the hazard-response policy from bypassing execution-state
        invariants.
        """

        if not isinstance(
            event,
            SafetyEvent,
        ):
            raise TypeError(
                "event must be a SafetyEvent."
            )

        if self.is_terminal:
            raise RuntimeError(
                "Cannot handle runtime events after entering a terminal state."
            )

        target_state = (
            recommended_state_for_event(
                event
            )
        )

        if target_state is None:
            return None

        if not self.can_transition(
            target_state
        ):
            raise RuntimeError(
                "Hazard response is incompatible with current state: "
                f"{self._state.value} -> {target_state.value} "
                f"for {event.hazard.value}."
            )

        return self.transition(
            target_state,
            reason=(
                event.message
            ),
            hazard=(
                event.hazard
            ),
            step_index=(
                event.step_index
            ),
        )

    def reset(
        self,
    ) -> None:
        """Reset a terminal episode back to IDLE.

        Resetting an active autonomous task is deliberately prohibited.
        """

        if not self.is_terminal:
            raise RuntimeError(
                "reset() is permitted only from a terminal state."
            )

        self._state = (
            SafetyState.IDLE
        )

        self._history.clear()