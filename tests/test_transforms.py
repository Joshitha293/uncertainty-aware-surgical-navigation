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


def rotation_z(theta: float) -> np.ndarray:
    """Return a 3x3 rotation matrix for rotation about the z-axis."""
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def test_identity_rotation_is_valid() -> None:
    rotation = np.eye(3)

    assert is_rotation_matrix(rotation)


def test_reflection_matrix_is_rejected() -> None:
    reflection = np.diag([1.0, 1.0, -1.0])

    assert not is_rotation_matrix(reflection)


def test_non_orthogonal_matrix_is_rejected() -> None:
    invalid_rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    assert not is_rotation_matrix(invalid_rotation)


def test_make_transform_identity() -> None:
    transform = make_transform(
        np.eye(3),
        np.zeros(3),
    )

    np.testing.assert_allclose(
        transform,
        np.eye(4),
    )


def test_translation_only_transform() -> None:
    transform = make_transform(
        np.eye(3),
        np.array([1.0, 2.0, 3.0]),
    )

    point = np.array([4.0, 5.0, 6.0])

    result = transform_point(transform, point)

    expected = np.array([5.0, 7.0, 9.0])

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_rotation_90_degrees_about_z() -> None:
    rotation = rotation_z(np.pi / 2)

    transform = make_transform(
        rotation,
        np.zeros(3),
    )

    point = np.array([1.0, 0.0, 0.0])

    result = transform_point(transform, point)

    expected = np.array([0.0, 1.0, 0.0])

    np.testing.assert_allclose(
        result,
        expected,
        atol=1e-8,
    )


def test_transform_composition_matches_sequential_application() -> None:
    first = make_transform(
        rotation_z(np.pi / 2),
        np.array([1.0, 0.0, 0.0]),
    )

    second = make_transform(
        np.eye(3),
        np.array([0.0, 2.0, 0.0]),
    )

    point = np.array([1.0, 1.0, 0.0])

    sequential = transform_point(
        first,
        transform_point(second, point),
    )

    composed = compose_transforms(
        first,
        second,
    )

    composed_result = transform_point(
        composed,
        point,
    )

    np.testing.assert_allclose(
        sequential,
        composed_result,
        atol=1e-8,
    )


def test_inverse_transform_multiplies_to_identity() -> None:
    transform = make_transform(
        rotation_z(np.pi / 4),
        np.array([1.0, -2.0, 0.5]),
    )

    inverse = invert_transform(transform)

    np.testing.assert_allclose(
        inverse @ transform,
        np.eye(4),
        atol=1e-8,
    )


def test_round_trip_point_transformation() -> None:
    transform = make_transform(
        rotation_z(np.pi / 3),
        np.array([0.5, -1.2, 2.0]),
    )

    point = np.array([2.4, -0.7, 1.1])

    transformed_point = transform_point(
        transform,
        point,
    )

    recovered_point = transform_point(
        invert_transform(transform),
        transformed_point,
    )

    np.testing.assert_allclose(
        recovered_point,
        point,
        atol=1e-8,
    )


def test_valid_homogeneous_transform_is_accepted() -> None:
    transform = make_transform(
        rotation_z(np.pi / 6),
        np.array([0.2, 0.3, 0.4]),
    )

    assert is_homogeneous_transform(transform)


def test_invalid_bottom_row_is_rejected() -> None:
    transform = np.eye(4)
    transform[3, 0] = 1.0

    assert not is_homogeneous_transform(transform)


def test_make_transform_rejects_invalid_translation_shape() -> None:
    with pytest.raises(ValueError):
        make_transform(
            np.eye(3),
            np.array([1.0, 2.0]),
        )


def test_transform_point_rejects_invalid_point_shape() -> None:
    transform = np.eye(4)

    with pytest.raises(ValueError):
        transform_point(
            transform,
            np.array([1.0, 2.0]),
        )


def test_compose_transforms_requires_at_least_two_inputs() -> None:
    with pytest.raises(ValueError):
        compose_transforms(np.eye(4))
