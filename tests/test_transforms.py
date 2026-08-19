import numpy as np
import pytest

from src.geometry.transforms import (
    compose_transforms,
    invert_transform,
    is_homogeneous_transform,
    is_rotation_matrix,
    make_transform,
    transform_point,
)


def test_identity_rotation_is_valid():
    rotation = np.eye(3)

    assert is_rotation_matrix(rotation)


def test_reflection_matrix_is_rejected():
    reflection = np.diag([1.0, 1.0, -1.0])

    assert not is_rotation_matrix(reflection)


def test_non_orthogonal_matrix_is_rejected():
    invalid_rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    assert not is_rotation_matrix(invalid_rotation)


def test_make_transform_identity():
    transform = make_transform(
        np.eye(3),
        np.zeros(3),
    )

    np.testing.assert_allclose(
        transform,
        np.eye(4),
        atol=1e-8,
    )


def test_translation_only_transform():
    transform = make_transform(
        np.eye(3),
        np.array([1.0, 2.0, 3.0]),
    )

    point = np.array([4.0, 5.0, 6.0])

    result = transform_point(
        transform,
        point,
    )

    expected = np.array([5.0, 7.0, 9.0])

    np.testing.assert_allclose(
        result,
        expected,
        atol=1e-8,
    )


def test_rotation_90_degrees_about_z():
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    transform = make_transform(
        rotation,
        np.zeros(3),
    )

    point = np.array([1.0, 0.0, 0.0])

    result = transform_point(
        transform,
        point,
    )

    expected = np.array([0.0, 1.0, 0.0])

    np.testing.assert_allclose(
        result,
        expected,
        atol=1e-8,
    )


def test_valid_homogeneous_transform_is_accepted():
    transform = make_transform(
        np.eye(3),
        np.array([0.2, 0.3, 0.4]),
    )

    assert is_homogeneous_transform(transform)


def test_inverse_multiplies_to_identity():
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    transform = make_transform(
        rotation,
        np.array([1.0, -2.0, 0.5]),
    )

    inverse = invert_transform(transform)

    np.testing.assert_allclose(
        inverse @ transform,
        np.eye(4),
        atol=1e-8,
    )


def test_composed_transform_matches_sequential_application():
    first = make_transform(
        np.eye(3),
        np.array([1.0, 0.0, 0.0]),
    )

    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    second = make_transform(
        rotation,
        np.array([0.0, 2.0, 0.0]),
    )

    point = np.array([1.0, 1.0, 0.0])

    sequential_result = transform_point(
        first,
        transform_point(second, point),
    )

    combined = compose_transforms(
        first,
        second,
    )

    composed_result = transform_point(
        combined,
        point,
    )

    np.testing.assert_allclose(
        composed_result,
        sequential_result,
        atol=1e-8,
    )


def test_round_trip_recovers_original_point():
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    transform = make_transform(
        rotation,
        np.array([0.5, -1.2, 2.0]),
    )

    point = np.array([2.4, -0.7, 1.1])

    transformed = transform_point(
        transform,
        point,
    )

    recovered = transform_point(
        invert_transform(transform),
        transformed,
    )

    np.testing.assert_allclose(
        recovered,
        point,
        atol=1e-8,
    )


def test_invalid_translation_shape_is_rejected():
    with pytest.raises(ValueError):
        make_transform(
            np.eye(3),
            np.array([1.0, 2.0]),
        )


def test_invalid_rotation_is_rejected_by_make_transform():
    invalid_rotation = np.diag(
        [1.0, 1.0, -1.0]
    )

    with pytest.raises(ValueError):
        make_transform(
            invalid_rotation,
            np.zeros(3),
        )


def test_invalid_point_shape_is_rejected():
    transform = np.eye(4)

    with pytest.raises(ValueError):
        transform_point(
            transform,
            np.array([1.0, 2.0]),
        )


def test_invalid_homogeneous_bottom_row_is_rejected():
    transform = np.eye(4)

    transform[3, :] = np.array(
        [1.0, 0.0, 0.0, 1.0]
    )

    assert not is_homogeneous_transform(
        transform
    )