"""Runtime safety monitors for simulated autonomous execution.

This module implements independent safety checks used by the Phase 7
autonomous-execution supervisor.

The monitors evaluate:

- perception freshness;
- positional uncertainty;
- predicted protected-region clearance;
- joint-limit proximity and violation;
- trajectory tracking error;
- execution timeout;
- invalid numerical state.

Each monitor produces zero or one SafetyEvent. The combined monitor returns
all active events so that the higher-level execution supervisor can decide
how to respond.

The thresholds implemented here are engineering simulation parameters. They
are not clinical safety limits and do not establish medical-device safety.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.robotics.safety_state_machine import (
    SafetyEvent,
    SafetyHazard,
    SafetySeverity,
)


@dataclass(frozen=True)
class RuntimeSafetyConfig:
    """Configuration for Phase 7 runtime safety monitoring."""

    max_perception_age_steps: int = 3

    uncertainty_reacquire_sigma: float = 0.015
    uncertainty_stop_sigma: float = 0.030

    clearance_recovery_threshold: float = 0.002
    clearance_stop_threshold: float = 0.0

    joint_limit_warning_margin: float = 0.05

    tracking_error_recovery_threshold: float = 0.005
    tracking_error_stop_threshold: float = 0.015

    maximum_execution_steps: int = 60

    def __post_init__(
        self,
    ) -> None:
        """Validate monitor configuration."""

        if self.max_perception_age_steps < 0:
            raise ValueError(
                "max_perception_age_steps must be non-negative."
            )

        if (
            not np.isfinite(
                self.uncertainty_reacquire_sigma
            )
            or self.uncertainty_reacquire_sigma <= 0.0
        ):
            raise ValueError(
                "uncertainty_reacquire_sigma must be finite and positive."
            )

        if (
            not np.isfinite(
                self.uncertainty_stop_sigma
            )
            or self.uncertainty_stop_sigma <= 0.0
        ):
            raise ValueError(
                "uncertainty_stop_sigma must be finite and positive."
            )

        if (
            self.uncertainty_reacquire_sigma
            >= self.uncertainty_stop_sigma
        ):
            raise ValueError(
                "uncertainty_reacquire_sigma must be smaller than "
                "uncertainty_stop_sigma."
            )

        if not np.isfinite(
            self.clearance_recovery_threshold
        ):
            raise ValueError(
                "clearance_recovery_threshold must be finite."
            )

        if not np.isfinite(
            self.clearance_stop_threshold
        ):
            raise ValueError(
                "clearance_stop_threshold must be finite."
            )

        if (
            self.clearance_stop_threshold
            > self.clearance_recovery_threshold
        ):
            raise ValueError(
                "clearance_stop_threshold must not exceed "
                "clearance_recovery_threshold."
            )

        if (
            not np.isfinite(
                self.joint_limit_warning_margin
            )
            or self.joint_limit_warning_margin < 0.0
        ):
            raise ValueError(
                "joint_limit_warning_margin must be finite and non-negative."
            )

        if (
            not np.isfinite(
                self.tracking_error_recovery_threshold
            )
            or self.tracking_error_recovery_threshold <= 0.0
        ):
            raise ValueError(
                "tracking_error_recovery_threshold must be finite and positive."
            )

        if (
            not np.isfinite(
                self.tracking_error_stop_threshold
            )
            or self.tracking_error_stop_threshold <= 0.0
        ):
            raise ValueError(
                "tracking_error_stop_threshold must be finite and positive."
            )

        if (
            self.tracking_error_recovery_threshold
            >= self.tracking_error_stop_threshold
        ):
            raise ValueError(
                "tracking_error_recovery_threshold must be smaller than "
                "tracking_error_stop_threshold."
            )

        if self.maximum_execution_steps < 1:
            raise ValueError(
                "maximum_execution_steps must be positive."
            )


@dataclass(frozen=True)
class RuntimeSafetySnapshot:
    """Runtime information required by the safety monitors."""

    step_index: int

    latest_perception_step: int

    maximum_principal_sigma: float

    predicted_clearance: float

    joint_positions: np.ndarray

    joint_lower_limits: np.ndarray

    joint_upper_limits: np.ndarray

    tracking_error: float

    execution_steps: int

    def __post_init__(
        self,
    ) -> None:
        """Validate basic snapshot shape and indexing."""

        if self.step_index < 0:
            raise ValueError(
                "step_index must be non-negative."
            )

        if self.latest_perception_step < 0:
            raise ValueError(
                "latest_perception_step must be non-negative."
            )

        if self.latest_perception_step > self.step_index:
            raise ValueError(
                "latest_perception_step cannot be later than step_index."
            )

        positions = np.asarray(
            self.joint_positions,
            dtype=float,
        )

        lower = np.asarray(
            self.joint_lower_limits,
            dtype=float,
        )

        upper = np.asarray(
            self.joint_upper_limits,
            dtype=float,
        )

        if positions.ndim != 1:
            raise ValueError(
                "joint_positions must be one-dimensional."
            )

        if lower.shape != positions.shape:
            raise ValueError(
                "joint_lower_limits must match joint_positions shape."
            )

        if upper.shape != positions.shape:
            raise ValueError(
                "joint_upper_limits must match joint_positions shape."
            )

        if positions.size < 1:
            raise ValueError(
                "At least one joint must be provided."
            )

        if self.execution_steps < 0:
            raise ValueError(
                "execution_steps must be non-negative."
            )


def check_invalid_data(
    snapshot: RuntimeSafetySnapshot,
) -> SafetyEvent | None:
    """Detect NaN, infinity, or invalid joint-limit definitions."""

    scalar_values = np.asarray(
        [
            snapshot.maximum_principal_sigma,
            snapshot.predicted_clearance,
            snapshot.tracking_error,
        ],
        dtype=float,
    )

    positions = np.asarray(
        snapshot.joint_positions,
        dtype=float,
    )

    lower = np.asarray(
        snapshot.joint_lower_limits,
        dtype=float,
    )

    upper = np.asarray(
        snapshot.joint_upper_limits,
        dtype=float,
    )

    if (
        not np.all(
            np.isfinite(
                scalar_values
            )
        )
        or not np.all(
            np.isfinite(
                positions
            )
        )
        or not np.all(
            np.isfinite(
                lower
            )
        )
        or not np.all(
            np.isfinite(
                upper
            )
        )
    ):
        return SafetyEvent(
            hazard=(
                SafetyHazard.INVALID_DATA
            ),
            severity=(
                SafetySeverity.CRITICAL
            ),
            message=(
                "Runtime state contained non-finite numerical data."
            ),
            step_index=(
                snapshot.step_index
            ),
        )

    if np.any(
        lower >= upper
    ):
        return SafetyEvent(
            hazard=(
                SafetyHazard.INVALID_DATA
            ),
            severity=(
                SafetySeverity.CRITICAL
            ),
            message=(
                "Joint-limit definition was invalid."
            ),
            step_index=(
                snapshot.step_index
            ),
        )

    if (
        snapshot.maximum_principal_sigma
        < 0.0
        or snapshot.tracking_error < 0.0
    ):
        return SafetyEvent(
            hazard=(
                SafetyHazard.INVALID_DATA
            ),
            severity=(
                SafetySeverity.CRITICAL
            ),
            message=(
                "Runtime uncertainty or tracking-error value was negative."
            ),
            step_index=(
                snapshot.step_index
            ),
        )

    return None


def check_perception_freshness(
    snapshot: RuntimeSafetySnapshot,
    config: RuntimeSafetyConfig,
) -> SafetyEvent | None:
    """Detect stale perception information."""

    age_steps = (
        snapshot.step_index
        - snapshot.latest_perception_step
    )

    if (
        age_steps
        > config.max_perception_age_steps
    ):
        return SafetyEvent(
            hazard=(
                SafetyHazard.STALE_PERCEPTION
            ),
            severity=(
                SafetySeverity.RECOVERABLE
            ),
            message=(
                "Perception age exceeded the configured freshness limit."
            ),
            step_index=(
                snapshot.step_index
            ),
        )

    return None


def check_uncertainty(
    snapshot: RuntimeSafetySnapshot,
    config: RuntimeSafetyConfig,
) -> SafetyEvent | None:
    """Detect excessive positional uncertainty."""

    sigma = float(
        snapshot.maximum_principal_sigma
    )

    if (
        sigma
        >= config.uncertainty_stop_sigma
    ):
        return SafetyEvent(
            hazard=(
                SafetyHazard.EXCESSIVE_UNCERTAINTY
            ),
            severity=(
                SafetySeverity.CRITICAL
            ),
            message=(
                "Positional uncertainty exceeded the fail-safe limit."
            ),
            step_index=(
                snapshot.step_index
            ),
        )

    if (
        sigma
        >= config.uncertainty_reacquire_sigma
    ):
        return SafetyEvent(
            hazard=(
                SafetyHazard.EXCESSIVE_UNCERTAINTY
            ),
            severity=(
                SafetySeverity.RECOVERABLE
            ),
            message=(
                "Positional uncertainty exceeded the perception-recovery "
                "threshold."
            ),
            step_index=(
                snapshot.step_index
            ),
        )

    return None


def check_clearance(
    snapshot: RuntimeSafetySnapshot,
    config: RuntimeSafetyConfig,
) -> SafetyEvent | None:
    """Monitor predicted clearance to the protected region."""

    clearance = float(
        snapshot.predicted_clearance
    )

    if (
        clearance
        <= config.clearance_stop_threshold
    ):
        return SafetyEvent(
            hazard=(
                SafetyHazard.CLEARANCE_VIOLATION
            ),
            severity=(
                SafetySeverity.CRITICAL
            ),
            message=(
                "Predicted protected-region clearance reached the "
                "fail-safe limit."
            ),
            step_index=(
                snapshot.step_index
            ),
        )

    if (
        clearance
        <= config.clearance_recovery_threshold
    ):
        return SafetyEvent(
            hazard=(
                SafetyHazard.CLEARANCE_VIOLATION
            ),
            severity=(
                SafetySeverity.RECOVERABLE
            ),
            message=(
                "Predicted protected-region clearance entered the "
                "recovery region."
            ),
            step_index=(
                snapshot.step_index
            ),
        )

    return None


def check_joint_limits(
    snapshot: RuntimeSafetySnapshot,
    config: RuntimeSafetyConfig,
) -> SafetyEvent | None:
    """Detect joint-limit violation or approach."""

    q = np.asarray(
        snapshot.joint_positions,
        dtype=float,
    )

    lower = np.asarray(
        snapshot.joint_lower_limits,
        dtype=float,
    )

    upper = np.asarray(
        snapshot.joint_upper_limits,
        dtype=float,
    )

    if (
        np.any(
            q < lower
        )
        or np.any(
            q > upper
        )
    ):
        return SafetyEvent(
            hazard=(
                SafetyHazard.JOINT_LIMIT_APPROACH
            ),
            severity=(
                SafetySeverity.CRITICAL
            ),
            message=(
                "Joint position exceeded a configured hard limit."
            ),
            step_index=(
                snapshot.step_index
            ),
        )

    distance_to_lower = (
        q
        - lower
    )

    distance_to_upper = (
        upper
        - q
    )

    minimum_limit_distance = float(
        np.min(
            np.minimum(
                distance_to_lower,
                distance_to_upper,
            )
        )
    )

    if (
        minimum_limit_distance
        <= config.joint_limit_warning_margin
    ):
        return SafetyEvent(
            hazard=(
                SafetyHazard.JOINT_LIMIT_APPROACH
            ),
            severity=(
                SafetySeverity.RECOVERABLE
            ),
            message=(
                "Joint position entered the configured limit-warning region."
            ),
            step_index=(
                snapshot.step_index
            ),
        )

    return None


def check_tracking_error(
    snapshot: RuntimeSafetySnapshot,
    config: RuntimeSafetyConfig,
) -> SafetyEvent | None:
    """Monitor commanded-versus-estimated execution error."""

    error = float(
        snapshot.tracking_error
    )

    if (
        error
        >= config.tracking_error_stop_threshold
    ):
        return SafetyEvent(
            hazard=(
                SafetyHazard.TRACKING_ERROR
            ),
            severity=(
                SafetySeverity.CRITICAL
            ),
            message=(
                "Trajectory tracking error exceeded the fail-safe limit."
            ),
            step_index=(
                snapshot.step_index
            ),
        )

    if (
        error
        >= config.tracking_error_recovery_threshold
    ):
        return SafetyEvent(
            hazard=(
                SafetyHazard.TRACKING_ERROR
            ),
            severity=(
                SafetySeverity.RECOVERABLE
            ),
            message=(
                "Trajectory tracking error exceeded the recovery threshold."
            ),
            step_index=(
                snapshot.step_index
            ),
        )

    return None


def check_execution_timeout(
    snapshot: RuntimeSafetySnapshot,
    config: RuntimeSafetyConfig,
) -> SafetyEvent | None:
    """Detect execution exceeding the configured step budget."""

    if (
        snapshot.execution_steps
        >= config.maximum_execution_steps
    ):
        return SafetyEvent(
            hazard=(
                SafetyHazard.EXECUTION_TIMEOUT
            ),
            severity=(
                SafetySeverity.CRITICAL
            ),
            message=(
                "Execution exceeded the configured maximum step budget."
            ),
            step_index=(
                snapshot.step_index
            ),
        )

    return None


def evaluate_runtime_safety(
    snapshot: RuntimeSafetySnapshot,
    config: RuntimeSafetyConfig | None = None,
) -> tuple[
    SafetyEvent,
    ...,
]:
    """Evaluate all independent runtime safety monitors.

    Invalid numerical data is evaluated first. If the snapshot itself is
    invalid, no threshold-based interpretation is attempted.
    """

    if not isinstance(
        snapshot,
        RuntimeSafetySnapshot,
    ):
        raise TypeError(
            "snapshot must be a RuntimeSafetySnapshot."
        )

    if config is None:
        config = RuntimeSafetyConfig()

    if not isinstance(
        config,
        RuntimeSafetyConfig,
    ):
        raise TypeError(
            "config must be a RuntimeSafetyConfig."
        )

    invalid_event = (
        check_invalid_data(
            snapshot
        )
    )

    if invalid_event is not None:
        return (
            invalid_event,
        )

    monitors = (
        check_perception_freshness,
        check_uncertainty,
        check_clearance,
        check_joint_limits,
        check_tracking_error,
        check_execution_timeout,
    )

    events: list[
        SafetyEvent
    ] = []

    for monitor in monitors:
        event = monitor(
            snapshot,
            config,
        )

        if event is not None:
            events.append(
                event
            )

    return tuple(
        events
    )


def highest_severity_event(
    events: tuple[
        SafetyEvent,
        ...,
    ],
) -> SafetyEvent | None:
    """Return the highest-severity event using deterministic ordering."""

    if len(
        events
    ) == 0:
        return None

    priority = {
        SafetySeverity.ADVISORY: 0,
        SafetySeverity.RECOVERABLE: 1,
        SafetySeverity.CRITICAL: 2,
    }

    return max(
        events,
        key=lambda event: (
            priority[
                event.severity
            ],
            -events.index(
                event
            ),
        ),
    )