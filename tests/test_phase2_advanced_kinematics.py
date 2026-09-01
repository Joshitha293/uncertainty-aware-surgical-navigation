"""Verification tests for Phase 2 advanced surgical kinematics."""

import numpy as np
import pytest

from src.robotics.advanced_kinematics import (
    evaluate_kinematics,
    forward_transform,
    geometric_jacobian,
    jacobian_max_abs_error,
    manipulability_index,
    normalised_joint_limit_margin,
    numerical_geometric_jacobian,
    numerical_position_jacobian,
    position_jacobian,
    position_jacobian_condition_number,
    tool_rotation_matrix,
    tool_twist,
    verify_position_ik,
)
from src.robotics.instrument import (
    JointLimits,
    SurgicalInstrument,
)


def make_instrument() -> SurgicalInstrument:
    """Create a deterministic test instrument."""

    return SurgicalInstrument(
        rcm_position=np.asarray(
            [
                0.10,
                -0.02,
                0.03,
            ],
            dtype=float,
        )
    )


def interior_configuration() -> np.ndarray:
    """Return a configuration comfortably inside all limits."""

    return np.asarray(
        [
            0.20,
            -0.15,
            0.18,
            0.40,
        ],
        dtype=float,
    )


def test_zero_angles_and_roll_give_identity_orientation():
    """Zero yaw/pitch/roll should align the tool with world axes."""

    instrument = (
        make_instrument()
    )

    q = np.asarray(
        [
            0.0,
            0.0,
            0.15,
            0.0,
        ],
        dtype=float,
    )

    rotation = (
        tool_rotation_matrix(
            instrument,
            q,
        )
    )

    assert np.allclose(
        rotation,
        np.eye(
            3
        ),
        atol=1e-12,
    )


def test_rotation_matrix_is_proper_and_tracks_shaft_axis():
    """Tool orientation must remain orthonormal and right-handed."""

    instrument = (
        make_instrument()
    )

    q = (
        interior_configuration()
    )

    rotation = (
        tool_rotation_matrix(
            instrument,
            q,
        )
    )

    assert np.allclose(
        rotation.T
        @ rotation,
        np.eye(
            3
        ),
        atol=1e-12,
    )

    assert np.linalg.det(
        rotation
    ) == pytest.approx(
        1.0,
        abs=1e-12,
    )

    expected_shaft = (
        instrument.shaft_direction(
            q[
                0
            ],
            q[
                1
            ],
        )
    )

    assert np.allclose(
        rotation[
            :,
            0,
        ],
        expected_shaft,
        atol=1e-12,
    )


def test_forward_transform_contains_existing_tip_position():
    """Full FK translation must exactly preserve Phase 1 position FK."""

    instrument = (
        make_instrument()
    )

    q = (
        interior_configuration()
    )

    transform = (
        forward_transform(
            instrument,
            q,
        )
    )

    expected_position = (
        instrument.forward_position(
            q
        )
    )

    assert transform.shape == (
        4,
        4,
    )

    assert np.allclose(
        transform[
            :3,
            3,
        ],
        expected_position,
        atol=1e-12,
    )

    assert np.allclose(
        transform[
            3,
            :,
        ],
        np.asarray(
            [
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        ),
        atol=1e-12,
    )


def test_roll_changes_orientation_but_not_tip_position():
    """Axial roll must not move the shaft tip."""

    instrument = (
        make_instrument()
    )

    q_zero_roll = np.asarray(
        [
            0.25,
            0.10,
            0.16,
            0.0,
        ],
        dtype=float,
    )

    q_rolled = np.asarray(
        [
            0.25,
            0.10,
            0.16,
            0.75,
        ],
        dtype=float,
    )

    position_zero = (
        instrument.forward_position(
            q_zero_roll
        )
    )

    position_rolled = (
        instrument.forward_position(
            q_rolled
        )
    )

    rotation_zero = (
        tool_rotation_matrix(
            instrument,
            q_zero_roll,
        )
    )

    rotation_rolled = (
        tool_rotation_matrix(
            instrument,
            q_rolled,
        )
    )

    assert np.allclose(
        position_zero,
        position_rolled,
        atol=1e-12,
    )

    assert not np.allclose(
        rotation_zero,
        rotation_rolled,
    )

    assert np.allclose(
        rotation_zero[
            :,
            0,
        ],
        rotation_rolled[
            :,
            0,
        ],
        atol=1e-12,
    )


def test_analytical_position_jacobian_matches_numerical():
    """Position derivatives must agree with finite-difference FK."""

    instrument = (
        make_instrument()
    )

    q = (
        interior_configuration()
    )

    analytical = (
        position_jacobian(
            instrument,
            q,
        )
    )

    numerical = (
        numerical_position_jacobian(
            instrument,
            q,
            step=1e-7,
        )
    )

    assert analytical.shape == (
        3,
        4,
    )

    assert np.allclose(
        analytical,
        numerical,
        atol=2e-7,
        rtol=2e-6,
    )


def test_analytical_geometric_jacobian_matches_numerical():
    """Linear and angular analytical kinematics must both verify."""

    instrument = (
        make_instrument()
    )

    q = (
        interior_configuration()
    )

    analytical = (
        geometric_jacobian(
            instrument,
            q,
        )
    )

    numerical = (
        numerical_geometric_jacobian(
            instrument,
            q,
            step=1e-7,
        )
    )

    assert analytical.shape == (
        6,
        4,
    )

    assert np.allclose(
        analytical,
        numerical,
        atol=3e-7,
        rtol=3e-6,
    )

    assert (
        jacobian_max_abs_error(
            instrument,
            q,
            step=1e-7,
        )
        < 3e-7
    )


def test_velocity_kinematics_maps_joint_rates_to_tool_twist():
    """The geometric Jacobian must implement x_dot = J q_dot."""

    instrument = (
        make_instrument()
    )

    q = (
        interior_configuration()
    )

    q_dot = np.asarray(
        [
            0.10,
            -0.05,
            0.02,
            0.20,
        ],
        dtype=float,
    )

    expected = (
        geometric_jacobian(
            instrument,
            q,
        )
        @ q_dot
    )

    actual = (
        tool_twist(
            instrument,
            q,
            q_dot,
        )
    )

    assert actual.shape == (
        6,
    )

    assert np.allclose(
        actual,
        expected,
        atol=1e-12,
    )


def test_position_inverse_kinematics_round_trip():
    """Existing analytical IK must reconstruct an FK-generated target."""

    instrument = (
        make_instrument()
    )

    original_q = (
        interior_configuration()
    )

    target = (
        instrument.forward_position(
            original_q
        )
    )

    recovered_q = (
        instrument.inverse_position(
            target,
            roll=float(
                original_q[
                    3
                ]
            ),
        )
    )

    reconstructed = (
        instrument.forward_position(
            recovered_q
        )
    )

    assert np.allclose(
        recovered_q,
        original_q,
        atol=1e-12,
    )

    assert np.allclose(
        reconstructed,
        target,
        atol=1e-12,
    )


def test_verified_position_inverse_kinematics_satisfies_rcm():
    """IK verification must check both Cartesian and RCM errors."""

    instrument = (
        make_instrument()
    )

    target = np.asarray(
        [
            0.24,
            0.02,
            0.08,
        ],
        dtype=float,
    )

    result = (
        verify_position_ik(
            instrument,
            target,
            roll=0.25,
        )
    )

    assert result.success

    assert (
        result.position_error
        <= 1e-9
    )

    assert (
        result.rcm_error
        <= 1e-8
    )

    assert np.allclose(
        result.reconstructed_position,
        target,
        atol=1e-9,
    )


def test_unreachable_target_is_rejected_by_joint_limits():
    """IK must reject targets beyond configured insertion range."""

    instrument = (
        make_instrument()
    )

    target = (
        instrument.rcm_position
        + np.asarray(
            [
                1.0,
                0.0,
                0.0,
            ],
            dtype=float,
        )
    )

    with pytest.raises(
        ValueError,
        match="Insertion joint",
    ):
        verify_position_ik(
            instrument,
            target,
        )


def test_normalised_joint_limit_margin_detects_boundary():
    """Joint-limit margin must fall to zero at a configured boundary."""

    instrument = (
        make_instrument()
    )

    limits: JointLimits = (
        instrument.joint_limits
    )

    centre = np.asarray(
        [
            0.5
            * (
                limits.yaw_min
                + limits.yaw_max
            ),
            0.5
            * (
                limits.pitch_min
                + limits.pitch_max
            ),
            0.5
            * (
                limits.insertion_min
                + limits.insertion_max
            ),
            0.5
            * (
                limits.roll_min
                + limits.roll_max
            ),
        ],
        dtype=float,
    )

    boundary = np.array(
        centre,
        copy=True,
    )

    boundary[
        0
    ] = limits.yaw_max

    assert (
        normalised_joint_limit_margin(
            instrument,
            centre,
        )
        == pytest.approx(
            0.5
        )
    )

    assert (
        normalised_joint_limit_margin(
            instrument,
            boundary,
        )
        == pytest.approx(
            0.0
        )
    )


def test_manipulability_and_conditioning_are_finite_inside_workspace():
    """An interior configuration must retain full positional rank."""

    instrument = (
        make_instrument()
    )

    q = (
        interior_configuration()
    )

    manipulability = (
        manipulability_index(
            instrument,
            q,
        )
    )

    condition_number = (
        position_jacobian_condition_number(
            instrument,
            q,
        )
    )

    state = (
        evaluate_kinematics(
            instrument,
            q,
        )
    )

    assert np.isfinite(
        manipulability
    )

    assert (
        manipulability
        > 0.0
    )

    assert np.isfinite(
        condition_number
    )

    assert (
        condition_number
        >= 1.0
    )

    assert np.all(
        state.singular_values
        > 0.0
    )

    assert (
        state.manipulability
        == pytest.approx(
            manipulability
        )
    )

    assert (
        state.condition_number
        == pytest.approx(
            condition_number
        )
    )


def test_rcm_constraint_is_preserved_across_valid_configurations():
    """All direct model configurations must retain the fixed RCM."""

    instrument = (
        make_instrument()
    )

    configurations = (
        np.asarray(
            [
                0.0,
                0.0,
                0.10,
                0.0,
            ],
            dtype=float,
        ),
        np.asarray(
            [
                0.40,
                0.25,
                0.20,
                1.0,
            ],
            dtype=float,
        ),
        np.asarray(
            [
                -0.50,
                -0.30,
                0.27,
                -1.5,
            ],
            dtype=float,
        ),
    )

    for q in configurations:
        assert (
            instrument
            .satisfies_rcm_constraint(
                q,
                tolerance=1e-10,
            )
        )

        state = (
            evaluate_kinematics(
                instrument,
                q,
            )
        )

        assert (
            state.rcm_error
            <= 1e-10
        )