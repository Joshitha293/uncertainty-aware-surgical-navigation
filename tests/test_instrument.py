import numpy as np
import pytest

from src.robotics.instrument import SurgicalInstrument


def test_shaft_direction_is_unit_length():
    direction = SurgicalInstrument.shaft_direction(
        yaw=np.deg2rad(30.0),
        pitch=np.deg2rad(20.0),
    )

    np.testing.assert_allclose(
        np.linalg.norm(direction),
        1.0,
        atol=1e-8,
    )


def test_forward_position_zero_angles():
    instrument = SurgicalInstrument(
        rcm_position=np.zeros(3),
    )

    q = np.array([0.0, 0.0, 0.20, 0.0])

    result = instrument.forward_position(q)

    expected = np.array([0.20, 0.0, 0.0])

    np.testing.assert_allclose(
        result,
        expected,
        atol=1e-8,
    )


def test_forward_position_with_rcm_offset():
    instrument = SurgicalInstrument(
        rcm_position=np.array([1.0, 2.0, 3.0]),
    )

    q = np.array([0.0, 0.0, 0.10, 0.0])

    result = instrument.forward_position(q)

    expected = np.array([1.10, 2.0, 3.0])

    np.testing.assert_allclose(
        result,
        expected,
        atol=1e-8,
    )


def test_roll_does_not_change_tip_position():
    instrument = SurgicalInstrument(
        rcm_position=np.zeros(3),
    )

    q_a = np.array(
        [
            np.deg2rad(20.0),
            np.deg2rad(10.0),
            0.20,
            0.0,
        ]
    )

    q_b = np.array(
        [
            np.deg2rad(20.0),
            np.deg2rad(10.0),
            0.20,
            np.deg2rad(90.0),
        ]
    )

    position_a = instrument.forward_position(q_a)
    position_b = instrument.forward_position(q_b)

    np.testing.assert_allclose(
        position_a,
        position_b,
        atol=1e-8,
    )


def test_tip_distance_from_rcm_equals_insertion():
    instrument = SurgicalInstrument(
        rcm_position=np.array([0.3, -0.2, 0.5]),
    )

    insertion = 0.25

    q = np.array(
        [
            np.deg2rad(25.0),
            np.deg2rad(-15.0),
            insertion,
            np.deg2rad(40.0),
        ]
    )

    tip = instrument.forward_position(q)

    distance = np.linalg.norm(
        tip - instrument.rcm_position
    )

    np.testing.assert_allclose(
        distance,
        insertion,
        atol=1e-8,
    )


def test_inverse_position_known_target():
    instrument = SurgicalInstrument(
        rcm_position=np.zeros(3),
    )

    target = np.array([0.20, 0.0, 0.0])

    q = instrument.inverse_position(target)

    expected = np.array(
        [
            0.0,
            0.0,
            0.20,
            0.0,
        ]
    )

    np.testing.assert_allclose(
        q,
        expected,
        atol=1e-8,
    )


def test_forward_inverse_round_trip():
    instrument = SurgicalInstrument(
        rcm_position=np.array([0.1, -0.05, 0.02]),
    )

    original_q = np.array(
        [
            np.deg2rad(25.0),
            np.deg2rad(15.0),
            0.20,
            np.deg2rad(30.0),
        ]
    )

    target = instrument.forward_position(
        original_q
    )

    recovered_q = instrument.inverse_position(
        target,
        roll=original_q[3],
    )

    recovered_target = instrument.forward_position(
        recovered_q
    )

    np.testing.assert_allclose(
        recovered_target,
        target,
        atol=1e-8,
    )


def test_inverse_recovers_joint_configuration():
    instrument = SurgicalInstrument(
        rcm_position=np.zeros(3),
    )

    original_q = np.array(
        [
            np.deg2rad(30.0),
            np.deg2rad(-20.0),
            0.25,
            np.deg2rad(45.0),
        ]
    )

    target = instrument.forward_position(
        original_q
    )

    recovered_q = instrument.inverse_position(
        target,
        roll=original_q[3],
    )

    np.testing.assert_allclose(
        recovered_q,
        original_q,
        atol=1e-8,
    )


def test_rcm_error_is_zero_for_nominal_configuration():
    instrument = SurgicalInstrument(
        rcm_position=np.zeros(3),
    )

    q = np.array(
        [
            0.0,
            0.0,
            0.20,
            0.0,
        ]
    )

    error = instrument.rcm_error(q)

    np.testing.assert_allclose(
        error,
        0.0,
        atol=1e-10,
    )


def test_rcm_constraint_with_nonzero_yaw_and_pitch():
    instrument = SurgicalInstrument(
        rcm_position=np.array([0.1, -0.05, 0.02]),
    )

    q = np.array(
        [
            np.deg2rad(35.0),
            np.deg2rad(-25.0),
            0.22,
            np.deg2rad(30.0),
        ]
    )

    assert instrument.satisfies_rcm_constraint(
        q,
        tolerance=1e-8,
    )


def test_rcm_constraint_across_multiple_configurations():
    instrument = SurgicalInstrument(
        rcm_position=np.array([0.02, 0.01, -0.03]),
    )

    configurations = [
        np.array(
            [
                np.deg2rad(-40.0),
                np.deg2rad(-20.0),
                0.10,
                0.0,
            ]
        ),
        np.array(
            [
                np.deg2rad(20.0),
                np.deg2rad(30.0),
                0.18,
                np.deg2rad(45.0),
            ]
        ),
        np.array(
            [
                np.deg2rad(55.0),
                np.deg2rad(-35.0),
                0.28,
                np.deg2rad(-90.0),
            ]
        ),
    ]

    for q in configurations:
        assert instrument.satisfies_rcm_constraint(
            q,
            tolerance=1e-8,
        )


def test_rcm_error_remains_small_across_configurations():
    instrument = SurgicalInstrument(
        rcm_position=np.array([0.1, 0.2, -0.1]),
    )

    configurations = [
        np.array(
            [
                np.deg2rad(-30.0),
                np.deg2rad(10.0),
                0.08,
                0.0,
            ]
        ),
        np.array(
            [
                np.deg2rad(10.0),
                np.deg2rad(-15.0),
                0.15,
                0.0,
            ]
        ),
        np.array(
            [
                np.deg2rad(45.0),
                np.deg2rad(35.0),
                0.25,
                0.0,
            ]
        ),
    ]

    errors = np.array(
        [
            instrument.rcm_error(q)
            for q in configurations
        ]
    )

    assert np.max(errors) <= 1e-8


def test_rcm_lies_between_proximal_point_and_tip():
    instrument = SurgicalInstrument(
        rcm_position=np.array([0.1, 0.2, 0.3]),
    )

    q = np.array(
        [
            np.deg2rad(20.0),
            np.deg2rad(-10.0),
            0.20,
            0.0,
        ]
    )

    proximal, tip = instrument.shaft_segment(
        q,
        proximal_length=0.10,
    )

    full_length = np.linalg.norm(
        tip - proximal
    )

    split_length = (
        np.linalg.norm(
            instrument.rcm_position - proximal
        )
        + np.linalg.norm(
            tip - instrument.rcm_position
        )
    )

    np.testing.assert_allclose(
        split_length,
        full_length,
        atol=1e-8,
    )


def test_point_to_line_distance_known_case():
    point = np.array([0.0, 1.0, 0.0])

    line_start = np.array([0.0, 0.0, 0.0])
    line_end = np.array([1.0, 0.0, 0.0])

    distance = (
        SurgicalInstrument.point_to_line_distance(
            point,
            line_start,
            line_end,
        )
    )

    np.testing.assert_allclose(
        distance,
        1.0,
        atol=1e-8,
    )


def test_invalid_joint_vector_shape_is_rejected():
    instrument = SurgicalInstrument(
        rcm_position=np.zeros(3),
    )

    with pytest.raises(ValueError):
        instrument.forward_position(
            np.array([0.0, 0.0, 0.10])
        )


def test_excessive_yaw_is_rejected():
    instrument = SurgicalInstrument(
        rcm_position=np.zeros(3),
    )

    q = np.array(
        [
            np.deg2rad(80.0),
            0.0,
            0.20,
            0.0,
        ]
    )

    with pytest.raises(ValueError):
        instrument.forward_position(q)


def test_excessive_pitch_is_rejected():
    instrument = SurgicalInstrument(
        rcm_position=np.zeros(3),
    )

    q = np.array(
        [
            0.0,
            np.deg2rad(60.0),
            0.20,
            0.0,
        ]
    )

    with pytest.raises(ValueError):
        instrument.forward_position(q)


def test_excessive_insertion_is_rejected():
    instrument = SurgicalInstrument(
        rcm_position=np.zeros(3),
    )

    q = np.array(
        [
            0.0,
            0.0,
            0.50,
            0.0,
        ]
    )

    with pytest.raises(ValueError):
        instrument.forward_position(q)


def test_inverse_rejects_target_at_rcm():
    instrument = SurgicalInstrument(
        rcm_position=np.zeros(3),
    )

    with pytest.raises(ValueError):
        instrument.inverse_position(
            np.zeros(3)
        )


def test_inverse_rejects_unreachable_target():
    instrument = SurgicalInstrument(
        rcm_position=np.zeros(3),
    )

    target = np.array([0.50, 0.0, 0.0])

    with pytest.raises(ValueError):
        instrument.inverse_position(target)


def test_negative_rcm_tolerance_is_rejected():
    instrument = SurgicalInstrument(
        rcm_position=np.zeros(3),
    )

    q = np.array(
        [
            0.0,
            0.0,
            0.20,
            0.0,
        ]
    )

    with pytest.raises(ValueError):
        instrument.satisfies_rcm_constraint(
            q,
            tolerance=-1.0,
        )


def test_invalid_proximal_length_is_rejected():
    instrument = SurgicalInstrument(
        rcm_position=np.zeros(3),
    )

    q = np.array(
        [
            0.0,
            0.0,
            0.20,
            0.0,
        ]
    )

    with pytest.raises(ValueError):
        instrument.shaft_segment(
            q,
            proximal_length=0.0,
        )