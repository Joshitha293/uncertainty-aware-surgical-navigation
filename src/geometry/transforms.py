"""Rigid-body transformation utilities for the surgical navigation system."""

import numpy as np


def is_rotation_matrix(
    rotation: np.ndarray,
    atol: float = 1e-8,
) -> bool:
    """Check whether a matrix is a valid proper 3-D rotation matrix."""
    rotation = np.asarray(rotation, dtype=float)

    if rotation.shape != (3, 3):
        return False

    orthonormal = np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=atol,
    )

    proper = np.isclose(
        np.linalg.det(rotation),
        1.0,
        atol=atol,
    )

    return bool(orthonormal and proper)


def make_transform(
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Construct a 4 x 4 homogeneous rigid-body transformation."""
    rotation = np.asarray(rotation, dtype=float)
    translation = np.asarray(translation, dtype=float)

    if not is_rotation_matrix(rotation):
        raise ValueError(
            "rotation must be a valid 3 x 3 proper rotation matrix."
        )

    if translation.shape != (3,):
        raise ValueError(
            "translation must have shape (3,)."
        )

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation

    return transform


def transform_point(
    transform: np.ndarray,
    point: np.ndarray,
) -> np.ndarray:
    """Transform a 3-D Cartesian point using a homogeneous transform."""
    transform = np.asarray(transform, dtype=float)
    point = np.asarray(point, dtype=float)

    if transform.shape != (4, 4):
        raise ValueError(
            "transform must have shape (4, 4)."
        )

    if point.shape != (3,):
        raise ValueError(
            "point must have shape (3,)."
        )

    homogeneous_point = np.array(
        [point[0], point[1], point[2], 1.0],
        dtype=float,
    )

    transformed_point = transform @ homogeneous_point

    return transformed_point[:3]


def is_homogeneous_transform(
    transform: np.ndarray,
    atol: float = 1e-8,
) -> bool:
    """Check whether a matrix is a valid 4 x 4 rigid transform."""
    transform = np.asarray(transform, dtype=float)

    if transform.shape != (4, 4):
        return False

    if not is_rotation_matrix(
        transform[:3, :3],
        atol=atol,
    ):
        return False

    valid_bottom_row = np.allclose(
        transform[3, :],
        np.array([0.0, 0.0, 0.0, 1.0]),
        atol=atol,
    )

    return bool(valid_bottom_row)


def invert_transform(
    transform: np.ndarray,
) -> np.ndarray:
    """Compute the analytical inverse of a rigid-body transform."""
    transform = np.asarray(transform, dtype=float)

    if not is_homogeneous_transform(transform):
        raise ValueError(
            "transform must be a valid 4 x 4 rigid transformation."
        )

    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    inverse_rotation = rotation.T
    inverse_translation = -inverse_rotation @ translation

    inverse = np.eye(4, dtype=float)
    inverse[:3, :3] = inverse_rotation
    inverse[:3, 3] = inverse_translation

    return inverse


def compose_transforms(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    """Compose two rigid-body transformations.

    The result is equivalent to:

        first @ second

    For example:

        ^W T_C = ^W T_B @ ^B T_C

    Parameters
    ----------
    first:
        First valid 4 x 4 homogeneous transformation.

    second:
        Second valid 4 x 4 homogeneous transformation.

    Returns
    -------
    np.ndarray
        Composite homogeneous transformation.

    Raises
    ------
    ValueError
        If either input is not a valid rigid transformation.
    """
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)

    if not is_homogeneous_transform(first):
        raise ValueError(
            "first must be a valid rigid transformation."
        )

    if not is_homogeneous_transform(second):
        raise ValueError(
            "second must be a valid rigid transformation."
        )

    composed = first @ second

    return composed