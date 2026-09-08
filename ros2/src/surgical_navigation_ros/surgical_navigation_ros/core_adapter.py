"""Adapt ROS runtime data to the existing surgical-navigation research core."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


ROS_JOINT_NAMES = (
    'yaw_joint',
    'pitch_joint',
    'roll_joint',
    'insertion_joint',
)

CORE_Q_NAMES = (
    'yaw_joint',
    'pitch_joint',
    'insertion_joint',
    'roll_joint',
)

EXPECTED_JOINT_NAMES = ROS_JOINT_NAMES


# ROS-order limits:
#
# [yaw, pitch, roll, insertion]
#
# These correspond exactly to the frozen Phase 2
# SurgicalInstrument position limits:
#
# core q = [yaw, pitch, insertion, roll]
#
# yaw       : [-pi/3,  pi/3]
# pitch     : [-pi/4,  pi/4]
# insertion : [0.05,   0.30] m
# roll      : [-pi,    pi]
JOINT_LOWER_LIMITS = np.asarray(
    [
        -np.pi / 3.0,
        -np.pi / 4.0,
        -np.pi,
        0.05,
    ],
    dtype=float,
)

JOINT_UPPER_LIMITS = np.asarray(
    [
        np.pi / 3.0,
        np.pi / 4.0,
        np.pi,
        0.30,
    ],
    dtype=float,
)


def _find_repository_root() -> Path:
    """Find the repository containing the existing research-core modules."""
    module_path = Path(__file__).resolve()
    working_directory = Path.cwd().resolve()

    candidates = (
        module_path.parent,
        *module_path.parents,
        working_directory,
        *working_directory.parents,
    )

    visited: set[Path] = set()

    for candidate in candidates:
        if candidate in visited:
            continue

        visited.add(candidate)

        marker = (
            candidate
            / 'src'
            / 'robotics'
            / 'runtime_safety_monitors.py'
        )

        if marker.is_file():
            return candidate

    raise RuntimeError(
        'Unable to locate the uncertainty-aware surgical-navigation '
        'repository root.'
    )


REPOSITORY_ROOT = _find_repository_root()

repository_root_string = str(
    REPOSITORY_ROOT
)

if repository_root_string not in sys.path:
    sys.path.insert(
        0,
        repository_root_string,
    )


_runtime_safety_module = importlib.import_module(
    'src.robotics.runtime_safety_monitors'
)

_safety_state_module = importlib.import_module(
    'src.robotics.safety_state_machine'
)

_autonomous_execution_module = importlib.import_module(
    'src.robotics.autonomous_execution_controller'
)


RuntimeSafetyConfig = (
    _runtime_safety_module.RuntimeSafetyConfig
)

RuntimeSafetySnapshot = (
    _runtime_safety_module.RuntimeSafetySnapshot
)

evaluate_runtime_safety = (
    _runtime_safety_module.evaluate_runtime_safety
)

highest_severity_event = (
    _runtime_safety_module.highest_severity_event
)


SafetyEvent = (
    _safety_state_module.SafetyEvent
)

SafetyHazard = (
    _safety_state_module.SafetyHazard
)

SafetySeverity = (
    _safety_state_module.SafetySeverity
)

SafetyState = (
    _safety_state_module.SafetyState
)

SafetyStateMachine = (
    _safety_state_module.SafetyStateMachine
)

recommended_state_for_event = (
    _safety_state_module.recommended_state_for_event
)

AutonomousExecutionController = (
    _autonomous_execution_module.AutonomousExecutionController
)


def _position_mapping(
    names: Sequence[str],
    positions: Sequence[float],
) -> dict[str, float]:
    """Build a validated mapping from joint names to joint positions."""
    joint_names = tuple(
        str(name)
        for name in names
    )

    joint_positions = np.asarray(
        positions,
        dtype=float,
    )

    if joint_positions.ndim != 1:
        raise ValueError(
            'positions must be one-dimensional.'
        )

    if len(joint_names) != joint_positions.size:
        raise ValueError(
            'Joint name and position counts must match.'
        )

    if len(joint_names) == 0:
        raise ValueError(
            'At least one joint must be supplied.'
        )

    if any(
        not name
        for name in joint_names
    ):
        raise ValueError(
            'Joint names must be non-empty.'
        )

    if len(set(joint_names)) != len(joint_names):
        raise ValueError(
            'Joint names must be unique.'
        )

    if not np.all(
        np.isfinite(
            joint_positions
        )
    ):
        raise ValueError(
            'Joint positions must contain finite values.'
        )

    return {
        name: float(position)
        for name, position in zip(
            joint_names,
            joint_positions,
        )
    }


def _required_joint_positions(
    *,
    mapping: dict[str, float],
    required_names: tuple[str, ...],
) -> np.ndarray:
    """Extract required joints from a validated position mapping."""
    missing = tuple(
        name
        for name in required_names
        if name not in mapping
    )

    if missing:
        raise ValueError(
            'Missing required joint(s): '
            + ', '.join(missing)
        )

    return np.asarray(
        [
            mapping[name]
            for name in required_names
        ],
        dtype=float,
    )


def ordered_joint_positions(
    names: Sequence[str],
    positions: Sequence[float],
) -> np.ndarray:
    """Return joint positions in canonical ROS controller order."""
    mapping = _position_mapping(
        names,
        positions,
    )

    return _required_joint_positions(
        mapping=mapping,
        required_names=ROS_JOINT_NAMES,
    )


def ros_joint_state_to_core_q(
    names: Sequence[str],
    positions: Sequence[float],
) -> np.ndarray:
    """Convert a named ROS JointState to Phase 2 core q ordering."""
    mapping = _position_mapping(
        names,
        positions,
    )

    return _required_joint_positions(
        mapping=mapping,
        required_names=CORE_Q_NAMES,
    )


def core_q_to_ros_positions(
    q: Sequence[float],
) -> np.ndarray:
    """Convert one Phase 2 core configuration into ROS joint order."""
    core_q = np.asarray(
        q,
        dtype=float,
    )

    if core_q.shape != (4,):
        raise ValueError(
            'Core configuration must have shape (4,).'
        )

    if not np.all(
        np.isfinite(
            core_q
        )
    ):
        raise ValueError(
            'Core configuration must contain finite values.'
        )

    yaw = core_q[0]
    pitch = core_q[1]
    insertion = core_q[2]
    roll = core_q[3]

    return np.asarray(
        [
            yaw,
            pitch,
            roll,
            insertion,
        ],
        dtype=float,
    )


def core_path_to_ros_positions(
    path: np.ndarray,
) -> np.ndarray:
    """Convert an N-by-4 Phase 2 path into ROS joint ordering."""
    core_path = np.asarray(
        path,
        dtype=float,
    )

    if (
        core_path.ndim != 2
        or core_path.shape[1] != 4
    ):
        raise ValueError(
            'Core path must have shape (N, 4).'
        )

    if not np.all(
        np.isfinite(
            core_path
        )
    ):
        raise ValueError(
            'Core path must contain finite values.'
        )

    return core_path[
        :,
        [
            0,
            1,
            3,
            2,
        ],
    ].copy()


def build_runtime_snapshot(
    *,
    names: Sequence[str] | None = None,
    joint_names: Sequence[str] | None = None,
    positions: Sequence[float],
    step_index: int,
    latest_perception_step: int,
    maximum_principal_sigma: float,
    predicted_clearance: float,
    tracking_error: float,
    execution_steps: int,
) -> RuntimeSafetySnapshot:
    """Build a Phase 7 RuntimeSafetySnapshot from ROS runtime data."""
    if (
        names is not None
        and joint_names is not None
    ):
        raise ValueError(
            'Provide either names or joint_names, not both.'
        )

    selected_names = (
        joint_names
        if joint_names is not None
        else names
    )

    if selected_names is None:
        raise ValueError(
            'Joint names must be provided.'
        )

    ros_positions = ordered_joint_positions(
        selected_names,
        positions,
    )

    return RuntimeSafetySnapshot(
        step_index=int(
            step_index
        ),
        latest_perception_step=int(
            latest_perception_step
        ),
        maximum_principal_sigma=float(
            maximum_principal_sigma
        ),
        predicted_clearance=float(
            predicted_clearance
        ),
        joint_positions=ros_positions,
        joint_lower_limits=(
            JOINT_LOWER_LIMITS.copy()
        ),
        joint_upper_limits=(
            JOINT_UPPER_LIMITS.copy()
        ),
        tracking_error=float(
            tracking_error
        ),
        execution_steps=int(
            execution_steps
        ),
    )
