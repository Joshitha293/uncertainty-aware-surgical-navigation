import numpy as np
import pytest

from src.robotics.instrument import (
    JointLimits,
    SurgicalInstrument,
)


def test_shaft_direction_is_unit_length() -> None:
    direction = SurgicalInstrument.shaft_direction(
        yaw=np.deg2rad(30.0),
        pitch=np.deg2rad(20.0),
    )

    np.testing.assert_allclose(
        np.linalg.norm(direction),
        1.0,
        atol=1e-8,
    )


def test_forward_position_zero_angles() -> None:
    instrument = SurgicalInstrument(
        rcm_position=np.array([0.0, 0.0, 0.0]),
    )

    q = np.array(
        [
            0.0,
            0.0,
            0.20,
            0.0,
        ]
    )

    result = instrument.forward_position(q)

    expected = np.array(
        [
            0.20,
            0.0,
            0.0,
        ]
    )

    np.testing.assert_allclose(
        result,
        expected,
        atol=1e-8,
    )


def test_forward_position_with_rcm_offset() -> None:
    instrument = SurgicalInstrument(
        rcm_position=np.array([1.0, 2.0, 3.0]),
    )

    q = np.array(
        [
            0.0,
            0.0,
            0.10,
            0.0,
        ]
    )

    result = instrument.forward_position(q)

    expected = np.array(
        [
            1.10,
            2.0,
            3.0,
        ]
    )

    np.testing.assert_allclose(
        result,
        expected,
        atol=1e-8,
    )


def test_roll_does_not_change_tip_position() -> None:
    instrument = SurgicalInstrument(
        rcm_position=np.zeros(3),
    )

    q1 = np.array(
        [
            np.deg2rad(20.0),
            np.deg2rad(10.0),
            0.20,
            0.0,
        ]
    )

    q2 = np.array(
        [
            np.deg2rad(20.0),
            np.deg2rad(10.0),
            0.20,
            np.deg2rad(90.0),
        ]
    )

    position_1 = instrument.forward_position(q1)
    position_2 = instrument.forward_position(q2)

    np.testing.assert_allclose(
        position_1,
        position_2,
        atol=1e-8,
    )


def test_forward_position_preserves_insertion_distance_from_rcm() -> None:
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


def test_invalid_joint_vector_shape_is_rejected() -> None:
    instrument = SurgicalInstrument(
        rcm_position=np.zeros(3),
    )

    with pytest.raises(ValueError):
        instrument.forward_position(
            np.array([0.0, 0.0, 0.1])
        )


def test_excessive_yaw_is_rejected() -> None:
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


def test_excessive_insertion_is_rejected() -> None:
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

  python -m pytest tests/test_instrument.py -v
