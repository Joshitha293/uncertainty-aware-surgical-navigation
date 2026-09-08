"""Autonomous execution controller for Phase 7 safety supervision.

The controller integrates:

- the Phase 7 safety state machine;
- independent runtime safety monitors;
- perception reacquisition;
- execution recovery;
- replanning;
- fail-safe stopping;
- terminal task completion.

Planning, perception reacquisition, and low-level recovery are supplied as
callbacks so the controller remains independent of any particular simulator
or robot implementation.

This is simulation/research software. The controller does not establish
clinical autonomy, medical-device safety, or suitability for surgery.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from src.robotics.runtime_safety_monitors import (
    RuntimeSafetyConfig,
    RuntimeSafetySnapshot,
    evaluate_runtime_safety,
    highest_severity_event,
)
from src.robotics.safety_state_machine import (
    SafetyEvent,
    SafetyHazard,
    SafetyState,
    SafetyStateMachine,
)


class AutonomousControllerAction(str, Enum):
    """High-level action taken during one controller update."""

    START = "start"
    CONTINUE = "continue"
    REPLAN = "replan"
    REACQUIRE = "reacquire"
    RECOVER = "recover"
    STOP = "stop"
    FAIL = "fail"
    COMPLETE = "complete"


@dataclass(frozen=True)
class AutonomousExecutionConfig:
    """Configuration for autonomous recovery behaviour."""

    maximum_reacquisition_attempts: int = 2
    maximum_recovery_attempts: int = 2

    def __post_init__(self) -> None:
        """Validate recovery-attempt limits."""

        if self.maximum_reacquisition_attempts < 1:
            raise ValueError(
                "maximum_reacquisition_attempts must be positive."
            )

        if self.maximum_recovery_attempts < 1:
            raise ValueError(
                "maximum_recovery_attempts must be positive."
            )


@dataclass(frozen=True)
class AutonomousControllerDecision:
    """Auditable result of one controller operation."""

    action: AutonomousControllerAction

    state: SafetyState

    triggering_event: SafetyEvent | None

    active_events: tuple[
        SafetyEvent,
        ...
    ]


PlannerCallback = Callable[
    [RuntimeSafetySnapshot],
    bool,
]

RecoveryCallback = Callable[
    [RuntimeSafetySnapshot],
    bool,
]

ReacquisitionCallback = Callable[
    [RuntimeSafetySnapshot],
    bool,
]


class AutonomousExecutionController:
    """Safety-supervised autonomous task execution controller."""

    def __init__(
        self,
        *,
        planner_callback: PlannerCallback,
        reacquisition_callback: ReacquisitionCallback,
        recovery_callback: RecoveryCallback,
        safety_config: RuntimeSafetyConfig | None = None,
        execution_config: AutonomousExecutionConfig | None = None,
    ) -> None:
        """Create the autonomous controller."""

        if not callable(planner_callback):
            raise TypeError(
                "planner_callback must be callable."
            )

        if not callable(reacquisition_callback):
            raise TypeError(
                "reacquisition_callback must be callable."
            )

        if not callable(recovery_callback):
            raise TypeError(
                "recovery_callback must be callable."
            )

        self._planner_callback = planner_callback
        self._reacquisition_callback = reacquisition_callback
        self._recovery_callback = recovery_callback

        self.safety_config = (
            RuntimeSafetyConfig()
            if safety_config is None
            else safety_config
        )

        self.execution_config = (
            AutonomousExecutionConfig()
            if execution_config is None
            else execution_config
        )

        self.state_machine = SafetyStateMachine()

        self.replan_count = 0
        self.reacquisition_count = 0
        self.recovery_count = 0
        self.stop_count = 0
        self.failure_count = 0

    @property
    def state(self) -> SafetyState:
        """Return current autonomous execution state."""

        return self.state_machine.state

    def _planning_failed(
        self,
        *,
        reason: str,
        step_index: int,
    ) -> AutonomousControllerDecision:
        """Enter FAILED after unsuccessful planning."""

        self.failure_count += 1

        self.state_machine.transition(
            SafetyState.FAILED,
            reason=reason,
            hazard=SafetyHazard.PLANNING_FAILURE,
            step_index=step_index,
        )

        return AutonomousControllerDecision(
            action=AutonomousControllerAction.FAIL,
            state=self.state,
            triggering_event=None,
            active_events=(),
        )

    def _stop(
        self,
        *,
        reason: str,
        hazard: SafetyHazard,
        step_index: int,
        triggering_event: SafetyEvent | None,
        active_events: tuple[
            SafetyEvent,
            ...
        ],
    ) -> AutonomousControllerDecision:
        """Enter terminal STOPPED state."""

        self.stop_count += 1

        self.state_machine.transition(
            SafetyState.STOPPED,
            reason=reason,
            hazard=hazard,
            step_index=step_index,
        )

        return AutonomousControllerDecision(
            action=AutonomousControllerAction.STOP,
            state=self.state,
            triggering_event=triggering_event,
            active_events=active_events,
        )

    def _run_replan(
        self,
        snapshot: RuntimeSafetySnapshot,
        *,
        triggering_event: SafetyEvent | None,
        active_events: tuple[
            SafetyEvent,
            ...
        ],
    ) -> AutonomousControllerDecision:
        """Attempt trajectory replanning."""

        self.replan_count += 1

        try:
            success = bool(
                self._planner_callback(
                    snapshot
                )
            )
        except Exception:
            success = False

        if not success:
            return self._planning_failed(
                reason=(
                    "Replanning failed to produce an acceptable trajectory."
                ),
                step_index=snapshot.step_index,
            )

        self.state_machine.transition(
            SafetyState.EXECUTING,
            reason="Updated trajectory accepted.",
            step_index=snapshot.step_index,
        )

        return AutonomousControllerDecision(
            action=AutonomousControllerAction.REPLAN,
            state=self.state,
            triggering_event=triggering_event,
            active_events=active_events,
        )

    def start(
        self,
        snapshot: RuntimeSafetySnapshot,
    ) -> AutonomousControllerDecision:
        """Initialise, plan, and enter autonomous execution."""

        if not isinstance(
            snapshot,
            RuntimeSafetySnapshot,
        ):
            raise TypeError(
                "snapshot must be a RuntimeSafetySnapshot."
            )

        if self.state != SafetyState.IDLE:
            raise RuntimeError(
                "start() is permitted only from IDLE."
            )

        self.state_machine.transition(
            SafetyState.INITIALISING,
            reason="Autonomous task initialisation started.",
            step_index=snapshot.step_index,
        )

        self.state_machine.transition(
            SafetyState.READY,
            reason="Autonomous task initialisation completed.",
            step_index=snapshot.step_index,
        )

        self.state_machine.transition(
            SafetyState.PLANNING,
            reason="Initial trajectory planning requested.",
            step_index=snapshot.step_index,
        )

        try:
            success = bool(
                self._planner_callback(
                    snapshot
                )
            )
        except Exception:
            success = False

        if not success:
            return self._planning_failed(
                reason=(
                    "Initial planner failed to produce an acceptable "
                    "trajectory."
                ),
                step_index=snapshot.step_index,
            )

        self.state_machine.transition(
            SafetyState.EXECUTING,
            reason="Initial trajectory accepted.",
            step_index=snapshot.step_index,
        )

        return AutonomousControllerDecision(
            action=AutonomousControllerAction.START,
            state=self.state,
            triggering_event=None,
            active_events=(),
        )

    def process_snapshot(
        self,
        snapshot: RuntimeSafetySnapshot,
    ) -> AutonomousControllerDecision:
        """Evaluate one runtime snapshot and perform required response."""

        if not isinstance(
            snapshot,
            RuntimeSafetySnapshot,
        ):
            raise TypeError(
                "snapshot must be a RuntimeSafetySnapshot."
            )

        if self.state != SafetyState.EXECUTING:
            raise RuntimeError(
                "Runtime snapshots may be processed only while EXECUTING."
            )

        events = evaluate_runtime_safety(
            snapshot,
            self.safety_config,
        )

        event = highest_severity_event(
            events
        )

        if event is None:
            return AutonomousControllerDecision(
                action=AutonomousControllerAction.CONTINUE,
                state=self.state,
                triggering_event=None,
                active_events=(),
            )

        transition = self.state_machine.handle_event(
            event
        )

        if transition is None:
            return AutonomousControllerDecision(
                action=AutonomousControllerAction.CONTINUE,
                state=self.state,
                triggering_event=event,
                active_events=events,
            )

        if self.state == SafetyState.STOPPED:
            self.stop_count += 1

            return AutonomousControllerDecision(
                action=AutonomousControllerAction.STOP,
                state=self.state,
                triggering_event=event,
                active_events=events,
            )

        if self.state == SafetyState.FAILED:
            self.failure_count += 1

            return AutonomousControllerDecision(
                action=AutonomousControllerAction.FAIL,
                state=self.state,
                triggering_event=event,
                active_events=events,
            )

        if self.state == SafetyState.REPLANNING:
            return self._run_replan(
                snapshot,
                triggering_event=event,
                active_events=events,
            )

        if self.state == SafetyState.REACQUIRING:
            success = False

            for _ in range(
                self.execution_config
                .maximum_reacquisition_attempts
            ):
                self.reacquisition_count += 1

                try:
                    success = bool(
                        self._reacquisition_callback(
                            snapshot
                        )
                    )
                except Exception:
                    success = False

                if success:
                    break

            if not success:
                return self._stop(
                    reason=(
                        "Perception reacquisition failed after the "
                        "configured maximum attempts."
                    ),
                    hazard=event.hazard,
                    step_index=snapshot.step_index,
                    triggering_event=event,
                    active_events=events,
                )

            self.state_machine.transition(
                SafetyState.REPLANNING,
                reason=(
                    "Perception reacquisition succeeded; updated "
                    "trajectory required."
                ),
                hazard=event.hazard,
                step_index=snapshot.step_index,
            )

            decision = self._run_replan(
                snapshot,
                triggering_event=event,
                active_events=events,
            )

            if (
                decision.action
                == AutonomousControllerAction.REPLAN
            ):
                return AutonomousControllerDecision(
                    action=AutonomousControllerAction.REACQUIRE,
                    state=decision.state,
                    triggering_event=event,
                    active_events=events,
                )

            return decision

        if self.state == SafetyState.RECOVERING:
            success = False

            for _ in range(
                self.execution_config
                .maximum_recovery_attempts
            ):
                self.recovery_count += 1

                try:
                    success = bool(
                        self._recovery_callback(
                            snapshot
                        )
                    )
                except Exception:
                    success = False

                if success:
                    break

            if not success:
                return self._stop(
                    reason=(
                        "Execution recovery failed after the configured "
                        "maximum attempts."
                    ),
                    hazard=event.hazard,
                    step_index=snapshot.step_index,
                    triggering_event=event,
                    active_events=events,
                )

            self.state_machine.transition(
                SafetyState.REPLANNING,
                reason=(
                    "Execution recovery succeeded; updated trajectory "
                    "required."
                ),
                hazard=event.hazard,
                step_index=snapshot.step_index,
            )

            decision = self._run_replan(
                snapshot,
                triggering_event=event,
                active_events=events,
            )

            if (
                decision.action
                == AutonomousControllerAction.REPLAN
            ):
                return AutonomousControllerDecision(
                    action=AutonomousControllerAction.RECOVER,
                    state=decision.state,
                    triggering_event=event,
                    active_events=events,
                )

            return decision

        raise RuntimeError(
            f"Unhandled controller state {self.state.value}."
        )

    def complete(
        self,
        *,
        step_index: int,
    ) -> AutonomousControllerDecision:
        """Mark successful autonomous task completion."""

        if step_index < 0:
            raise ValueError(
                "step_index must be non-negative."
            )

        if self.state != SafetyState.EXECUTING:
            raise RuntimeError(
                "complete() is permitted only while EXECUTING."
            )

        self.state_machine.transition(
            SafetyState.COMPLETED,
            reason="Autonomous task goal reached.",
            step_index=step_index,
        )

        return AutonomousControllerDecision(
            action=AutonomousControllerAction.COMPLETE,
            state=self.state,
            triggering_event=None,
            active_events=(),
        )