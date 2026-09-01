"""Trajectory time-parameterisation and execution metrics.

Phase 2 extends geometric path planning into executable simulated
joint trajectories.

The timing limits used here are simulation parameters. They are not
claimed to represent a particular commercial surgical robot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.robotics.advanced_kinematics import (
    evaluate_kinematics,
)
from src.robotics.instrument import (
    SurgicalInstrument,
)
from src.robotics.planner import (
    path_cost,
)


@dataclass(frozen=True)
class TimeParameterisationConfig:
    """Simulation limits for joint-trajectory generation."""

    maximum_joint_velocities: tuple[
        float,
        float,
        float,
        float,
    ] = (
        0.75,
        0.75,
        0.08,
        1.50,
    )

    sample_period: float = 0.02

    minimum_segment_duration: float = 0.02

    def __post_init__(self) -> None:
        velocities = np.asarray(
            self.maximum_joint_velocities,
            dtype=float,
        )

        if velocities.shape != (4,):
            raise ValueError(
                "maximum_joint_velocities must contain four values."
            )

        if (
            not np.all(
                np.isfinite(
                    velocities
                )
            )
            or np.any(
                velocities <= 0.0
            )
        ):
            raise ValueError(
                "maximum_joint_velocities must be finite and positive."
            )

        if (
            not np.isfinite(
                self.sample_period
            )
            or self.sample_period <= 0.0
        ):
            raise ValueError(
                "sample_period must be finite and positive."
            )

        if (
            not np.isfinite(
                self.minimum_segment_duration
            )
            or self.minimum_segment_duration <= 0.0
        ):
            raise ValueError(
                "minimum_segment_duration must be finite and positive."
            )


@dataclass(frozen=True)
class TimedTrajectory:
    """A time-parameterised joint trajectory."""

    times: np.ndarray

    positions: np.ndarray

    velocities: np.ndarray

    accelerations: np.ndarray


@dataclass(frozen=True)
class ExecutionMetrics:
    """Quantitative metrics for one simulated trajectory."""

    duration: float

    sample_count: int

    scaled_joint_path_cost: float

    cartesian_tip_distance: float

    peak_absolute_velocity: np.ndarray

    peak_absolute_acceleration: np.ndarray

    minimum_joint_limit_margin: float

    minimum_manipulability: float

    maximum_condition_number: float

    maximum_rcm_error: float

    acceleration_energy: float


def _validate_path(
    instrument: SurgicalInstrument,
    path: np.ndarray,
) -> np.ndarray:
    """Validate one joint-space path."""

    path = np.asarray(
        path,
        dtype=float,
    )

    if (
        path.ndim != 2
        or path.shape[1] != 4
    ):
        raise ValueError(
            "path must have shape (N, 4)."
        )

    if path.shape[0] < 2:
        raise ValueError(
            "path must contain at least two configurations."
        )

    if not np.all(
        np.isfinite(
            path
        )
    ):
        raise ValueError(
            "path must contain finite values."
        )

    for q in path:
        instrument.validate_configuration(
            q
        )

    return path


def _segment_duration(
    delta: np.ndarray,
    maximum_joint_velocities: np.ndarray,
    minimum_duration: float,
) -> float:
    """Return velocity-limited duration for one path segment."""

    required_times = (
        np.abs(
            delta
        )
        / maximum_joint_velocities
    )

    return float(
        max(
            float(
                np.max(
                    required_times
                )
            ),
            minimum_duration,
        )
    )


def time_parameterise_path(
    instrument: SurgicalInstrument,
    path: np.ndarray,
    config: TimeParameterisationConfig | None = None,
) -> TimedTrajectory:
    """Convert a geometric path into a sampled timed trajectory.

    Each path segment receives enough duration that no configured
    joint-velocity limit is exceeded by the resulting linear interpolation.
    """

    path = _validate_path(
        instrument,
        path,
    )

    if config is None:
        config = (
            TimeParameterisationConfig()
        )

    velocity_limits = np.asarray(
        config.maximum_joint_velocities,
        dtype=float,
    )

    position_samples: list[
        np.ndarray
    ] = [
        np.array(
            path[0],
            dtype=float,
            copy=True,
        )
    ]

    time_samples: list[float] = [
        0.0
    ]

    current_time = 0.0

    for index in range(
        path.shape[0] - 1
    ):
        start = path[
            index
        ]

        goal = path[
            index + 1
        ]

        delta = (
            goal
            - start
        )

        duration = _segment_duration(
            delta,
            velocity_limits,
            config.minimum_segment_duration,
        )

        interval_count = max(
            1,
            int(
                np.ceil(
                    duration
                    / config.sample_period
                )
            ),
        )

        actual_duration = (
            interval_count
            * config.sample_period
        )

        for interval_index in range(
            1,
            interval_count + 1
        ):
            fraction = (
                interval_index
                / interval_count
            )

            position = (
                start
                + fraction
                * delta
            )

            current_time += (
                config.sample_period
            )

            position_samples.append(
                np.asarray(
                    position,
                    dtype=float,
                )
            )

            time_samples.append(
                float(
                    current_time
                )
            )

        # actual_duration is deliberately retained conceptually:
        # interpolation uses the discrete sample interval exactly.
        _ = actual_duration

    positions = np.vstack(
        position_samples
    )

    times = np.asarray(
        time_samples,
        dtype=float,
    )

    dt = np.diff(
        times
    )

    velocities = (
        np.diff(
            positions,
            axis=0,
        )
        / dt[
            :,
            None
        ]
    )

    if velocities.shape[0] >= 2:
        velocity_times = (
            0.5
            * (
                times[:-1]
                + times[1:]
            )
        )

        velocity_dt = np.diff(
            velocity_times
        )

        accelerations = (
            np.diff(
                velocities,
                axis=0,
            )
            / velocity_dt[
                :,
                None
            ]
        )

    else:
        accelerations = np.empty(
            (
                0,
                4,
            ),
            dtype=float,
        )

    return TimedTrajectory(
        times=times,
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
    )


def evaluate_execution_metrics(
    instrument: SurgicalInstrument,
    trajectory: TimedTrajectory,
) -> ExecutionMetrics:
    """Evaluate kinematic and execution metrics along a trajectory."""

    positions = np.asarray(
        trajectory.positions,
        dtype=float,
    )

    times = np.asarray(
        trajectory.times,
        dtype=float,
    )

    velocities = np.asarray(
        trajectory.velocities,
        dtype=float,
    )

    accelerations = np.asarray(
        trajectory.accelerations,
        dtype=float,
    )

    if (
        positions.ndim != 2
        or positions.shape[1] != 4
        or positions.shape[0] < 2
    ):
        raise ValueError(
            "trajectory positions must have shape (N, 4), N >= 2."
        )

    if (
        times.ndim != 1
        or times.shape[0] != positions.shape[0]
    ):
        raise ValueError(
            "trajectory times must correspond to position samples."
        )

    if not np.all(
        np.diff(
            times
        )
        > 0.0
    ):
        raise ValueError(
            "trajectory times must be strictly increasing."
        )

    tip_positions: list[np.ndarray] = []

    margins: list[float] = []
    manipulabilities: list[float] = []
    condition_numbers: list[float] = []
    rcm_errors: list[float] = []

    for q in positions:
        state = evaluate_kinematics(
            instrument,
            q,
        )

        tip_positions.append(
            instrument.forward_position(
                q
            )
        )

        margins.append(
            state.normalised_joint_limit_margin
        )

        manipulabilities.append(
            state.manipulability
        )

        condition_numbers.append(
            state.condition_number
        )

        rcm_errors.append(
            state.rcm_error
        )

    tips = np.asarray(
        tip_positions,
        dtype=float,
    )

    cartesian_tip_distance = float(
        np.sum(
            np.linalg.norm(
                np.diff(
                    tips,
                    axis=0,
                ),
                axis=1,
            )
        )
    )

    if velocities.size:
        peak_velocity = np.max(
            np.abs(
                velocities
            ),
            axis=0,
        )
    else:
        peak_velocity = np.zeros(
            4,
            dtype=float,
        )

    if accelerations.size:
        peak_acceleration = np.max(
            np.abs(
                accelerations
            ),
            axis=0,
        )

        acceleration_energy = float(
            np.sum(
                accelerations**2
            )
            * float(
                np.mean(
                    np.diff(
                        times
                    )
                )
            )
        )

    else:
        peak_acceleration = np.zeros(
            4,
            dtype=float,
        )

        acceleration_energy = 0.0

    return ExecutionMetrics(
        duration=float(
            times[-1]
            - times[0]
        ),
        sample_count=int(
            positions.shape[0]
        ),
        scaled_joint_path_cost=path_cost(
            positions
        ),
        cartesian_tip_distance=(
            cartesian_tip_distance
        ),
        peak_absolute_velocity=np.asarray(
            peak_velocity,
            dtype=float,
        ),
        peak_absolute_acceleration=np.asarray(
            peak_acceleration,
            dtype=float,
        ),
        minimum_joint_limit_margin=float(
            np.min(
                margins
            )
        ),
        minimum_manipulability=float(
            np.min(
                manipulabilities
            )
        ),
        maximum_condition_number=float(
            np.max(
                condition_numbers
            )
        ),
        maximum_rcm_error=float(
            np.max(
                rcm_errors
            )
        ),
        acceleration_energy=(
            acceleration_energy
        ),
    )