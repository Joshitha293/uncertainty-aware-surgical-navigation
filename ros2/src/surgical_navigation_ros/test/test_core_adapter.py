"""Tests for the ROS-to-research-core adapter."""

import numpy as np
import pytest

from surgical_navigation_ros.core_adapter import (
    build_runtime_snapshot,
    core_path_to_ros_positions,
    CORE_Q_NAMES,
    core_q_to_ros_positions,
    EXPECTED_JOINT_NAMES,
    ordered_joint_positions,
    ROS_JOINT_NAMES,
    ros_joint_state_to_core_q,
)


def test_joint_positions_are_reordered() -> None:
    """ROS JointState data should be reordered into canonical ROS order."""
    names = (
        'insertion_joint',
        'roll_joint',
        'yaw_joint',
        'pitch_joint',
    )

    positions = (
        0.20,
        0.30,
        0.10,
        -0.20,
    )

    ordered = ordered_joint_positions(
        names,
        positions,
    )

    expected = np.asarray(
        [
            0.10,
            -0.20,
            0.30,
            0.20,
        ],
        dtype=float,
    )

    np.testing.assert_allclose(
        ordered,
        expected,
    )


def test_ros_joint_state_converts_to_core_q() -> None:
    """ROS joint ordering should convert to Phase 2 core q ordering."""
    names = (
        'roll_joint',
        'insertion_joint',
        'pitch_joint',
        'yaw_joint',
    )

    positions = (
        0.30,
        0.20,
        -0.20,
        0.10,
    )

    core_q = ros_joint_state_to_core_q(
        names,
        positions,
    )

    expected = np.asarray(
        [
            0.10,
            -0.20,
            0.20,
            0.30,
        ],
        dtype=float,
    )

    np.testing.assert_allclose(
        core_q,
        expected,
    )


def test_core_q_converts_to_ros_positions() -> None:
    """Phase 2 q should convert explicitly into ROS controller order."""
    core_q = np.asarray(
        [
            0.10,
            -0.20,
            0.20,
            0.30,
        ],
        dtype=float,
    )

    ros_positions = core_q_to_ros_positions(
        core_q
    )

    expected = np.asarray(
        [
            0.10,
            -0.20,
            0.30,
            0.20,
        ],
        dtype=float,
    )

    np.testing.assert_allclose(
        ros_positions,
        expected,
    )


def test_core_path_converts_to_ros_positions() -> None:
    """Every waypoint should be converted from core to ROS ordering."""
    core_path = np.asarray(
        [
            [
                0.10,
                -0.20,
                0.20,
                0.30,
            ],
            [
                0.40,
                0.25,
                0.24,
                -0.50,
            ],
        ],
        dtype=float,
    )

    ros_path = core_path_to_ros_positions(
        core_path
    )

    expected = np.asarray(
        [
            [
                0.10,
                -0.20,
                0.30,
                0.20,
            ],
            [
                0.40,
                0.25,
                -0.50,
                0.24,
            ],
        ],
        dtype=float,
    )

    np.testing.assert_allclose(
        ros_path,
        expected,
    )


def test_round_trip_joint_mapping() -> None:
    """Core-to-ROS-to-core conversion should preserve the configuration."""
    original_core_q = np.asarray(
        [
            0.35,
            0.25,
            0.18,
            -0.40,
        ],
        dtype=float,
    )

    ros_positions = core_q_to_ros_positions(
        original_core_q
    )

    reconstructed_core_q = (
        ros_joint_state_to_core_q(
            ROS_JOINT_NAMES,
            ros_positions,
        )
    )

    np.testing.assert_allclose(
        reconstructed_core_q,
        original_core_q,
    )


def test_missing_required_joint_is_rejected() -> None:
    """A ROS state missing a required joint should be rejected."""
    names = (
        'yaw_joint',
        'pitch_joint',
        'roll_joint',
    )

    positions = (
        0.0,
        0.0,
        0.0,
    )

    with pytest.raises(
        ValueError,
        match='Missing required joint',
    ):
        ordered_joint_positions(
            names,
            positions,
        )


def test_non_finite_joint_state_is_rejected() -> None:
    """Non-finite ROS joint positions should be rejected."""
    positions = (
        0.0,
        0.0,
        np.nan,
        0.15,
    )

    with pytest.raises(
        ValueError,
        match='finite',
    ):
        ordered_joint_positions(
            ROS_JOINT_NAMES,
            positions,
        )


def test_invalid_core_configuration_shape_is_rejected() -> None:
    """A Phase 2 configuration must contain exactly four joints."""
    invalid_q = np.asarray(
        [
            0.0,
            0.0,
            0.15,
        ],
        dtype=float,
    )

    with pytest.raises(
        ValueError,
        match=r'shape \(4,\)',
    ):
        core_q_to_ros_positions(
            invalid_q
        )


def test_runtime_snapshot_uses_robot_limits() -> None:
    """Runtime snapshots should use the frozen Phase 2 position limits."""
    snapshot = build_runtime_snapshot(
        names=EXPECTED_JOINT_NAMES,
        positions=np.asarray(
            [
                0.0,
                0.0,
                0.0,
                0.175,
            ],
            dtype=float,
        ),
        step_index=10,
        latest_perception_step=10,
        maximum_principal_sigma=0.005,
        predicted_clearance=0.010,
        tracking_error=0.001,
        execution_steps=10,
    )

    expected_lower = np.asarray(
        [
            -np.pi / 3.0,
            -np.pi / 4.0,
            -np.pi,
            0.05,
        ],
        dtype=float,
    )

    expected_upper = np.asarray(
        [
            np.pi / 3.0,
            np.pi / 4.0,
            np.pi,
            0.30,
        ],
        dtype=float,
    )

    np.testing.assert_allclose(
        snapshot.joint_lower_limits,
        expected_lower,
    )

    np.testing.assert_allclose(
        snapshot.joint_upper_limits,
        expected_upper,
    )

    np.testing.assert_allclose(
        snapshot.joint_positions,
        np.asarray(
            [
                0.0,
                0.0,
                0.0,
                0.175,
            ],
            dtype=float,
        ),
    )


def test_joint_contract_names_are_explicit() -> None:
    """ROS and research-core joint contracts should remain explicit."""
    assert ROS_JOINT_NAMES == (
        'yaw_joint',
        'pitch_joint',
        'roll_joint',
        'insertion_joint',
    )

    assert CORE_Q_NAMES == (
        'yaw_joint',
        'pitch_joint',
        'insertion_joint',
        'roll_joint',
    )

    assert EXPECTED_JOINT_NAMES == (
        ROS_JOINT_NAMES
    )
