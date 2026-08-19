import numpy as np
import pytest

from src.robotics.trajectory import (
    interpolate_joint_trajectory,
    trajectory_joint_velocity,
)


def test_trajectory_includes_start_and_goal():
    start_q = np.array([0.0, 0.0, 0.10, 0.0])
    goal_q = np.array([0.2, -0.1, 0.20, 0.5])

    trajectory = interpolate_joint_trajectory(
        start_q,
        goal_q,
        num_steps=5,
    )

    np.testing.assert_allclose(
        trajectory[0],
        start_q,
        atol=1e-8,
    )

    np.testing.assert_allclose(
        trajectory[-1],
        goal_q,
        atol=1e-8,
    )


def test_trajectory_has_correct_shape():
    start_q = np.zeros(4)
    goal_q = np.ones(4)

    trajectory = interpolate_joint_trajectory(
        start_q,
        goal_q,
        num_steps=10,
    )

    assert trajectory.shape == (10, 4)


def test_midpoint_is_correct():
    start_q = np.zeros(4)
    goal_q = np.array([1.0, 2.0, 3.0, 4.0])

    trajectory = interpolate_joint_trajectory(
        start_q,
        goal_q,
        num_steps=3,
    )

    expected_midpoint = np.array(
        [0.5, 1.0, 1.5, 2.0]
    )

    np.testing.assert_allclose(
        trajectory[1],
        expected_midpoint,
        atol=1e-8,
    )


def test_constant_velocity_for_linear_trajectory():
    start_q = np.zeros(4)
    goal_q = np.array([1.0, 2.0, 3.0, 4.0])

    trajectory = interpolate_joint_trajectory(
        start_q,
        goal_q,
        num_steps=5,
    )

    velocity = trajectory_joint_velocity(
        trajectory,
        timestep=0.5,
    )

    expected_velocity = np.array(
        [0.5, 1.0, 1.5, 2.0]
    )

    for row in velocity:
        np.testing.assert_allclose(
            row,
            expected_velocity,
            atol=1e-8,
        )


def test_invalid_start_shape_is_rejected():
    with pytest.raises(ValueError):
        interpolate_joint_trajectory(
            np.array([0.0, 0.0, 0.0]),
            np.zeros(4),
            num_steps=5,
        )


def test_invalid_goal_shape_is_rejected():
    with pytest.raises(ValueError):
        interpolate_joint_trajectory(
            np.zeros(4),
            np.array([0.0, 0.0, 0.0]),
            num_steps=5,
        )


def test_too_few_steps_is_rejected():
    with pytest.raises(ValueError):
        interpolate_joint_trajectory(
            np.zeros(4),
            np.ones(4),
            num_steps=1,
        )


def test_invalid_trajectory_shape_is_rejected():
    with pytest.raises(ValueError):
        trajectory_joint_velocity(
            np.zeros((5, 3)),
            timestep=0.1,
        )


def test_single_sample_trajectory_is_rejected():
    with pytest.raises(ValueError):
        trajectory_joint_velocity(
            np.zeros((1, 4)),
            timestep=0.1,
        )


def test_non_positive_timestep_is_rejected():
    trajectory = np.zeros((5, 4))

    with pytest.raises(ValueError):
        trajectory_joint_velocity(
            trajectory,
            timestep=0.0,
        )