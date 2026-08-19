"""Rigid-body transformation utilities.

This module implements the core coordinate-transformation operations used
throughout the surgical navigation simulation.

A transform written mathematically as ^A T_B represents the pose of frame B
expressed in frame A.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]

DEFAULT_ATOL = 1e-8


def is_rotation_matrix(
    rotation: FloatArray,
    atol: float = DEFAULT_ATOL,
) -> bool:
    """Check whether a matrix is a valid proper 3-D rotation matrix.

    A proper rotation matrix belongs to SO(3) and therefore satisfies:

        R.T @ R = I
        det(R) = +1

    Parameters
    ----------
    rotation:
        Candidate rotation matrix.
    atol:
        Absolute numerical tolerance.

    Returns
    -------
    bool
        True when the matrix satisfies the required properties.
    """
    rotation = np.asarray(rotation, dtype=np.float64)

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
    rotation: FloatArray,
    translation: FloatArray,
) -> FloatArray:
    """Construct a homogeneous rigid-body transformation matrix.

    Parameters
    ----------
    rotation:
        3 x 3 proper rotation matrix.
    translation:
        Translation vector with shape (3,).

    Returns
    -------
    FloatArray
        Homogeneous transformation with shape (4, 4).

    Raises
    ------
    ValueError
        If the rotation or translation is invalid.
    """
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)

    if not is_rotation_matrix(rotation):
        raise ValueError(
            "rotation must be a valid 3 x 3 proper rotation matrix."
        )

    if translation.shape != (3,):
        raise ValueError(
            "translation must have shape (3,)."
        )

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation

    return transform


def is_homogeneous_transform(
    transform: FloatArray,
    atol: float = DEFAULT_ATOL,
) -> bool:
    """Check whether a matrix is a valid rigid homogeneous transform."""
    transform = np.asarray(transform, dtype=np.float64)

    if transform.shape != (4, 4):
        return False

    valid_bottom_row = np.allclose(
        transform[3, :],
        np.array([0.0, 0.0, 0.0, 1.0]),
        atol=atol,
    )

    if not valid_bottom_row:
        return False

    return is_rotation_matrix(
        transform[:3, :3],
        atol=atol,
    )


def transform_point(
    transform: FloatArray,
    point: FloatArray,
) -> FloatArray:
    """Transform a Cartesian 3-D point between coordinate frames.

    Parameters
    ----------
    transform:
        Valid 4 x 4 homogeneous transformation.
    point:
        Cartesian point with shape (3,).

    Returns
    -------
    FloatArray
        Transformed Cartesian point with shape (3,).
    """
    transform = np.asarray(transform, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)

    if not is_homogeneous_transform(transform):
        raise ValueError(
            "transform must be a valid 4 x 4 rigid transformation."
        )

    if point.shape != (3,):
        raise ValueError(
            "point must have shape (3,)."
        )

    homogeneous_point = np.concatenate(
        (point, np.array([1.0]))
    )

    transformed_point = transform @ homogeneous_point

    return transformed_point[:3]


def compose_transforms(
    *transforms: FloatArray,
) -> FloatArray:
    """Compose two or more rigid-body transformations.

    Transform order follows matrix multiplication.

    For example:

        ^W T_T = ^W T_B @ ^B T_T

    can be evaluated using:

        compose_transforms(T_W_B, T_B_T)
    """
    if len(transforms) < 2:
        raise ValueError(
            "At least two transforms are required."
        )

    result = np.eye(4, dtype=np.float64)

    for transform in transforms:
        transform = np.asarray(
            transform,
            dtype=np.float64,
        )

        if not is_homogeneous_transform(transform):
            raise ValueError(
                "Every input must be a valid rigid transformation."
            )

        result = result @ transform

    return result


def invert_transform(
    transform: FloatArray,
) -> FloatArray:
    """Compute the analytical inverse of a rigid-body transform.

    For

        T = [R  t]
            [0  1]

    the inverse is

        T^-1 = [R.T  -R.T @ t]
               [ 0        1    ]
    """
    transform = np.asarray(
        transform,
        dtype=np.float64,
    )

    if not is_homogeneous_transform(transform):
        raise ValueError(
            "transform must be a valid rigid transformation."
        )

    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    rotation_inverse = rotation.T
    translation_inverse = (
        -rotation_inverse @ translation
    )

    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation_inverse
    inverse[:3, 3] = translation_inverse

    return inverse
