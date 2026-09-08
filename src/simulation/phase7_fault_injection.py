"""Deterministic fault injection for Phase 7 autonomous safety evaluation.

This module creates controlled runtime faults for verification of the
autonomous-execution safety supervisor.

Injected faults include:

- stale perception;
- moderate and severe uncertainty;
- recoverable and critical clearance loss;
- joint-limit approach and violation;
- recoverable and critical tracking error;
- invalid numerical data;
- execution timeout.

Fault injection is used only for simulation and verification. It does not
represent patient-specific failure probabilities or clinical risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.robotics.runtime_safety_monitors import (
    RuntimeSafetyConfig,
    RuntimeSafetySnapshot,
)


class FaultType(str, Enum):
    """Controlled Phase 7 runtime fault types."""

    NONE = "none"

    STALE_PERCEPTION = "stale_perception"

    UNCERTAINTY_RECOVERABLE = "uncertainty_recoverable"
    UNCERTAINTY_CRITICAL = "uncertainty_critical"

    CLEARANCE_RECOVERABLE = "clearance_recoverable"
    CLEARANCE_CRITICAL = "clearance_critical"

    JOINT_LIMIT_APPROACH = "joint_limit_approach"
    JOINT_LIMIT_VIOLATION = "joint_limit_violation"

    TRACKING_ERROR_RECOVERABLE = "tracking_error_recoverable"
    TRACKING_ERROR_CRITICAL = "tracking_error_critical"

    INVALID_NUMERICAL_DATA = "invalid_numerical_data"

    EXECUTION_TIMEOUT = "execution_timeout"


@dataclass(frozen=True)
class FaultInjectionRecord:
    """Auditable description of one injected simulation fault."""

    fault: FaultType

    step_index: int

    description: str

    snapshot: RuntimeSafetySnapshot


def nominal_runtime_snapshot(
    *,
    step_index: int = 10,
) -> RuntimeSafetySnapshot:
    """Return a deterministic nominally safe runtime snapshot."""

    if step_index < 0:
        raise ValueError(
            "step_index must be non-negative."
        )

    return RuntimeSafetySnapshot(
        step_index=step_index,
        latest_perception_step=step_index,
        maximum_principal_sigma=0.005,
        predicted_clearance=0.010,
        joint_positions=np.asarray(
            [
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=float,
        ),
        joint_lower_limits=np.asarray(
            [
                -1.0,
                -1.0,
                -1.0,
                -1.0,
            ],
            dtype=float,
        ),
        joint_upper_limits=np.asarray(
            [
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype=float,
        ),
        tracking_error=0.001,
        execution_steps=step_index,
    )


def inject_fault(
    snapshot: RuntimeSafetySnapshot,
    fault: FaultType,
    *,
    config: RuntimeSafetyConfig | None = None,
) -> FaultInjectionRecord:
    """Return a copy of snapshot containing one controlled fault."""

    if not isinstance(
        snapshot,
        RuntimeSafetySnapshot,
    ):
        raise TypeError(
            "snapshot must be a RuntimeSafetySnapshot."
        )

    if not isinstance(
        fault,
        FaultType,
    ):
        raise TypeError(
            "fault must be a FaultType."
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

    values = {
        "step_index": snapshot.step_index,
        "latest_perception_step": snapshot.latest_perception_step,
        "maximum_principal_sigma": snapshot.maximum_principal_sigma,
        "predicted_clearance": snapshot.predicted_clearance,
        "joint_positions": np.asarray(
            snapshot.joint_positions,
            dtype=float,
        ).copy(),
        "joint_lower_limits": np.asarray(
            snapshot.joint_lower_limits,
            dtype=float,
        ).copy(),
        "joint_upper_limits": np.asarray(
            snapshot.joint_upper_limits,
            dtype=float,
        ).copy(),
        "tracking_error": snapshot.tracking_error,
        "execution_steps": snapshot.execution_steps,
    }

    description = "No fault injected."

    if fault == FaultType.NONE:
        pass

    elif fault == FaultType.STALE_PERCEPTION:
        age = (
            config.max_perception_age_steps
            + 1
        )

        values[
            "latest_perception_step"
        ] = max(
            0,
            snapshot.step_index
            - age,
        )

        description = (
            "Injected stale perception beyond the configured age limit."
        )

    elif (
        fault
        == FaultType.UNCERTAINTY_RECOVERABLE
    ):
        values[
            "maximum_principal_sigma"
        ] = float(
            config.uncertainty_reacquire_sigma
        )

        description = (
            "Injected positional uncertainty at the recoverable threshold."
        )

    elif (
        fault
        == FaultType.UNCERTAINTY_CRITICAL
    ):
        values[
            "maximum_principal_sigma"
        ] = float(
            config.uncertainty_stop_sigma
        )

        description = (
            "Injected positional uncertainty at the critical threshold."
        )

    elif (
        fault
        == FaultType.CLEARANCE_RECOVERABLE
    ):
        values[
            "predicted_clearance"
        ] = float(
            0.5
            * (
                config.clearance_recovery_threshold
                + config.clearance_stop_threshold
            )
        )

        description = (
            "Injected protected clearance inside the recovery region."
        )

    elif (
        fault
        == FaultType.CLEARANCE_CRITICAL
    ):
        values[
            "predicted_clearance"
        ] = float(
            config.clearance_stop_threshold
        )

        description = (
            "Injected protected clearance at the fail-safe limit."
        )

    elif (
        fault
        == FaultType.JOINT_LIMIT_APPROACH
    ):
        q = np.asarray(
            values[
                "joint_positions"
            ],
            dtype=float,
        )

        upper = np.asarray(
            values[
                "joint_upper_limits"
            ],
            dtype=float,
        )

        q[
            0
        ] = (
            upper[
                0
            ]
            - 0.5
            * config.joint_limit_warning_margin
        )

        values[
            "joint_positions"
        ] = q

        description = (
            "Injected joint position inside the configured warning margin."
        )

    elif (
        fault
        == FaultType.JOINT_LIMIT_VIOLATION
    ):
        q = np.asarray(
            values[
                "joint_positions"
            ],
            dtype=float,
        )

        upper = np.asarray(
            values[
                "joint_upper_limits"
            ],
            dtype=float,
        )

        q[
            0
        ] = (
            upper[
                0
            ]
            + 0.01
        )

        values[
            "joint_positions"
        ] = q

        description = (
            "Injected joint position beyond the configured hard limit."
        )

    elif (
        fault
        == FaultType.TRACKING_ERROR_RECOVERABLE
    ):
        values[
            "tracking_error"
        ] = float(
            config.tracking_error_recovery_threshold
        )

        description = (
            "Injected trajectory tracking error at the recovery threshold."
        )

    elif (
        fault
        == FaultType.TRACKING_ERROR_CRITICAL
    ):
        values[
            "tracking_error"
        ] = float(
            config.tracking_error_stop_threshold
        )

        description = (
            "Injected trajectory tracking error at the critical threshold."
        )

    elif (
        fault
        == FaultType.INVALID_NUMERICAL_DATA
    ):
        values[
            "tracking_error"
        ] = float(
            "nan"
        )

        description = (
            "Injected non-finite runtime tracking data."
        )

    elif (
        fault
        == FaultType.EXECUTION_TIMEOUT
    ):
        values[
            "execution_steps"
        ] = int(
            config.maximum_execution_steps
        )

        description = (
            "Injected execution step count at the timeout threshold."
        )

    else:
        raise ValueError(
            f"Unhandled fault type: {fault!r}."
        )

    injected_snapshot = (
        RuntimeSafetySnapshot(
            **values
        )
    )

    return FaultInjectionRecord(
        fault=fault,
        step_index=(
            snapshot.step_index
        ),
        description=description,
        snapshot=injected_snapshot,
    )


def all_fault_types() -> tuple[
    FaultType,
    ...,
]:
    """Return the frozen Phase 7 fault set excluding NONE."""

    return tuple(
        fault
        for fault in FaultType
        if fault != FaultType.NONE
    )