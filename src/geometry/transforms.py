"""Rigid-body transformation utilities for the surgical navigation project.

This module provides validated homogeneous transformation operations used
throughout the simulation, including coordinate-frame construction,
point transformation, transform composition, and rigid-transform inversion.

Coordinate transforms follow the convention:

    ^A T_B

which represents the pose of frame B expressed in frame A.

A point expressed in frame B can therefore be transformed into frame A using:

    ^A p = ^A T_B @ ^B p
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


def is_rotation_matrix(
    rotation: Array,
    atol: float = 1e-8,
) -> bool:
    """Return True if the input is a valid 3x3 proper rotation matrix.

    A valid rotation matrix must satisfy:

        R.T @ R = I
        det(R) = +1

    Parameters
    ----------
    rotation:
        Candidate 3x3 rotation matrix.
    atol:
        Absolute numerical tolerance used for validation.

    Returns
    -------
    bool
        True if the matrix is a valid member of SO(3), otherwise False.
    """
    rotation = np.asarray(rotation, dtype=float)

    if rotation.shape != (3, 3):
        return False

    identity = np.eye(3)

    orthonormal = np.allclose(
        rotation.T @ rotation,
        identity,
        atol=atol,
    )

    proper_orientation = np.isclose(
        np.linalg.det(rotation),
        1.0,
        atol=atol,
    )

    return bool(orthonormal and proper_orientation)


def make_transform(
    rotation: Array,
    translation: Array,
) -> Array:
    """Construct a 4x4 homogeneous rigid-body transformation matrix.

    Parameters
    ----------
    rotation:
        Valid 3x3 rotation matrix.
    translation:
        Translation vector with shape (3,).

    Returns
    -------
    numpy.ndarray
        Homogeneous transformation matrix with shape (4, 4).

    Raises
    ------
    ValueError
        If the rotation matrix or translation vector is invalid.
    """
    rotation = np.asarray(rotation, dtype=float)
    translation = np.asarray(translation, dtype=float)

    if not is_rotation_matrix(rotation):
        raise ValueError(
            "rotation must be a valid 3x3 proper rotation matrix."
        )

    if translation.shape != (3,):
        raise ValueError(
            "translation must have shape (3,)."
        )

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation

    return transform


def is_homogeneous_transform(
    transform: Array,
    atol: float = 1e-8,
) -> bool:
    """Return True if the input is a valid 4x4 rigid transformation.

    Parameters
    ----------
    transform:
        Candidate homogeneous transformation matrix.
    atol:
        Absolute numerical tolerance used for validation.

    Returns
    -------
    bool
        True if the matrix represents a valid rigid transform.
    """
    transform = np.asarray(transform, dtype=float)

    if transform.shape != (4, 4):
        return False

    if not np.allclose(
        transform[3, :],
        np.array([0.0, 0.0, 0.0, 1.0]),
        atol=atol,
    ):
        return False

    return is_rotation_matrix(
        transform[:3, :3],
        atol=atol,
    )


def transform_point(
    transform: Array,
    point: Array,
) -> Array:
    """Transform a 3D point using a homogeneous rigid-body transform.

    Parameters
    ----------
    transform:
        Valid 4x4 homogeneous transformation matrix.
    point:
        Cartesian point with shape (3,).

    Returns
    -------
    numpy.ndarray
        Transformed point with shape (3,).

    Raises
    ------
    ValueError
        If the transform or point has an invalid shape or structure.
    """
    transform = np.asarray(transform, dtype=float)
    point = np.asarray(point, dtype=float)

    if not is_homogeneous_transform(transform):
        raise ValueError(
            "transform must be a valid 4x4 rigid transformation."
        )

    if point.shape != (3,):
        raise ValueError(
            "point must have shape (3,)."
        )

    homogeneous_point = np.append(point, 1.0)

    transformed = transform @ homogeneous_point

    return transformed[:3]


def compose_transforms(*transforms: Array) -> Array:
    """Compose multiple homogeneous transformations in order.

    For example:

        ^W T_T = ^W T_B @ ^B T_T

    can be computed as:

        compose_transforms(T_W_B, T_B_T)

    Parameters
    ----------
    *transforms:
        Two or more valid 4x4 homogeneous transformations.

    Returns
    -------
    numpy.ndarray
        Composite 4x4 homogeneous transformation.

    Raises
    ------
    ValueError
        If fewer than two transforms are provided or any transform is invalid.
    """
    if len(transforms) < 2:
        raise ValueError(
            "At least two transforms are required for composition."
        )

    result = np.eye(4, dtype=float)

    for transform in transforms:
        transform = np.asarray(transform, dtype=float)

        if not is_homogeneous_transform(transform):
            raise ValueError(
                "All inputs must be valid 4x4 rigid transformations."
            )

        result = result @ transform

    return result


def invert_transform(
    transform: Array,
) -> Array:
    """Invert a rigid-body homogeneous transformation analytically.

    For:

        T = [R  p]
            [0  1]

    the rigid inverse is:

        T^-1 = [R.T  -R.T @ p]
               [ 0        1   ]

    Parameters
    ----------
    transform:
        Valid 4x4 homogeneous rigid-body transformation.

    Returns
    -------
    numpy.ndarray
        Inverse transformation matrix.

    Raises
    ------
    ValueError
        If the input is not a valid rigid transform.
    """
    transform = np.asarray(transform, dtype=float)

    if not is_homogeneous_transform(transform):
        raise ValueError(
            "transform must be a valid 4x4 rigid transformation."
        )

    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    inverse = np.eye(4, dtype=float)

    rotation_inverse = rotation.T

    inverse[:3, :3] = rotation_inverse
    inverse[:3, 3] = -rotation_inverse @ translation

    return inverse
