"""Trajectory generation utilities for the surgical instrument."""

import numpy as np


def interpolate_joint_trajectory(
    start_q: np.ndarray,
    goal_q: np.ndarray,
    num_steps: int,
) -> np.ndarray:
    """Generate a linear joint-space trajectory.

    Parameters
    ----------
    start_q:
        Initial joint configuration with shape (4,).

    goal_q:
        Final joint configuration with shape (4,).

    num_steps:
        Number of trajectory samples, including start and goal.

    Returns
    -------
    np.ndarray
        Array with shape (num_steps, 4).

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    start_q = np.asarray(start_q, dtype=float)
    goal_q = np.asarray(goal_q, dtype=float)

    if start_q.shape != (4,):
        raise ValueError(
            "start_q must have shape (4,)."
        )

    if goal_q.shape != (4,):
        raise ValueError(
            "goal_q must have shape (4,)."
        )

    if num_steps < 2:
        raise ValueError(
            "num_steps must be at least 2."
        )

    trajectory = np.linspace(
        start_q,
        goal_q,
        num=num_steps,
        dtype=float,
    )

    return trajectory


def trajectory_joint_velocity(
    trajectory: np.ndarray,
    timestep: float,
) -> np.ndarray:
    """Estimate discrete joint velocities along a trajectory.

    Parameters
    ----------
    trajectory:
        Joint trajectory with shape (N, 4).

    timestep:
        Time interval between consecutive samples.

    Returns
    -------
    np.ndarray
        Joint velocity array with shape (N - 1, 4).

    Raises
    ------
    ValueError
        If the trajectory or timestep is invalid.
    """
    trajectory = np.asarray(
        trajectory,
        dtype=float,
    )

    if (
        trajectory.ndim != 2
        or trajectory.shape[1] != 4
    ):
        raise ValueError(
            "trajectory must have shape (N, 4)."
        )

    if trajectory.shape[0] < 2:
        raise ValueError(
            "trajectory must contain at least two samples."
        )

    if timestep <= 0.0:
        raise ValueError(
            "timestep must be positive."
        )

    velocity = np.diff(
        trajectory,
        axis=0,
    ) / timestep

    return velocity