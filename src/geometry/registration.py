"""Rigid point-set registration for surgical navigation.

This module estimates a rigid transformation between corresponding 3-D
landmarks using the Kabsch/SVD algorithm.

Given corresponding source and target points:

    p_target ~= R @ p_source + t

the algorithm:

1. computes source and target centroids;
2. centres both point sets;
3. forms the cross-covariance matrix;
4. solves the optimal rotation using singular value decomposition;
5. corrects improper reflections when required;
6. computes the translation;
7. builds the project's existing 4 x 4 homogeneous transformation;
8. evaluates fiducial registration residuals.

The implementation is intended for simulated engineering validation and
does not constitute clinical image-to-patient registration validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry.transforms import (
    is_homogeneous_transform,
    make_transform,
    transform_point,
)


@dataclass(frozen=True)
class RigidRegistrationResult:
    """Result of corresponding-point rigid registration."""

    transform: np.ndarray

    rotation: np.ndarray

    translation: np.ndarray

    transformed_source_points: np.ndarray

    residual_vectors: np.ndarray

    residual_distances: np.ndarray

    fre_rms: float

    fre_mean: float

    fre_max: float

    singular_values: np.ndarray


@dataclass(frozen=True)
class TransformError:
    """Difference between estimated and reference rigid transforms."""

    translation_error: float

    rotation_angle_error_radians: float

    rotation_angle_error_degrees: float


def _validate_point_set(
    points: np.ndarray,
    *,
    name: str,
    minimum_points: int = 1,
) -> np.ndarray:
    """Validate an N x 3 Cartesian point set."""

    points = np.asarray(
        points,
        dtype=float,
    )

    if (
        points.ndim != 2
        or points.shape[1] != 3
    ):
        raise ValueError(
            f"{name} must have shape (N, 3)."
        )

    if points.shape[0] < minimum_points:
        raise ValueError(
            f"{name} must contain at least {minimum_points} points."
        )

    if not np.all(
        np.isfinite(
            points
        )
    ):
        raise ValueError(
            f"{name} must contain finite values."
        )

    return points


def transform_points(
    transform: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Transform an N x 3 point set using an existing homogeneous transform."""

    transform = np.asarray(
        transform,
        dtype=float,
    )

    if not is_homogeneous_transform(
        transform
    ):
        raise ValueError(
            "transform must be a valid 4 x 4 rigid transformation."
        )

    points = _validate_point_set(
        points,
        name="points",
    )

    return np.vstack(
        [
            transform_point(
                transform,
                point,
            )
            for point in points
        ]
    )


def rms_point_error(
    first_points: np.ndarray,
    second_points: np.ndarray,
) -> float:
    """Return RMS Euclidean error between corresponding 3-D points."""

    first = _validate_point_set(
        first_points,
        name="first_points",
    )

    second = _validate_point_set(
        second_points,
        name="second_points",
    )

    if first.shape != second.shape:
        raise ValueError(
            "first_points and second_points must have identical shapes."
        )

    distances_squared = np.sum(
        (
            first
            - second
        )
        ** 2,
        axis=1,
    )

    return float(
        np.sqrt(
            np.mean(
                distances_squared
            )
        )
    )


def _check_registration_geometry(
    centred_source: np.ndarray,
) -> None:
    """Reject point configurations that cannot determine a unique rotation."""

    rank = int(
        np.linalg.matrix_rank(
            centred_source,
            tol=1e-12,
        )
    )

    if rank < 2:
        raise ValueError(
            "source_points are geometrically degenerate; "
            "at least three non-collinear points are required."
        )


def register_corresponding_points(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> RigidRegistrationResult:
    """Estimate rigid transform from corresponding 3-D landmarks.

    Parameters
    ----------
    source_points:
        N x 3 landmarks expressed in the source coordinate frame.

    target_points:
        Corresponding N x 3 landmarks expressed in the target frame.

    Returns
    -------
    RigidRegistrationResult
        Estimated rigid transform and registration residual metrics.
    """

    source = _validate_point_set(
        source_points,
        name="source_points",
        minimum_points=3,
    )

    target = _validate_point_set(
        target_points,
        name="target_points",
        minimum_points=3,
    )

    if source.shape != target.shape:
        raise ValueError(
            "source_points and target_points must have identical shapes."
        )

    source_centroid = np.mean(
        source,
        axis=0,
    )

    target_centroid = np.mean(
        target,
        axis=0,
    )

    centred_source = (
        source
        - source_centroid
    )

    centred_target = (
        target
        - target_centroid
    )

    _check_registration_geometry(
        centred_source
    )

    covariance = (
        centred_source.T
        @ centred_target
    )

    u, singular_values, vt = (
        np.linalg.svd(
            covariance
        )
    )

    rotation = (
        vt.T
        @ u.T
    )

    # Kabsch reflection correction:
    # enforce det(R) = +1 for a proper rigid-body rotation.
    if np.linalg.det(
        rotation
    ) < 0.0:
        vt = vt.copy()

        vt[
            -1,
            :
        ] *= -1.0

        rotation = (
            vt.T
            @ u.T
        )

    translation = (
        target_centroid
        - rotation
        @ source_centroid
    )

    transform = make_transform(
        rotation=rotation,
        translation=translation,
    )

    transformed_source = transform_points(
        transform,
        source,
    )

    residual_vectors = (
        transformed_source
        - target
    )

    residual_distances = np.linalg.norm(
        residual_vectors,
        axis=1,
    )

    fre_rms = float(
        np.sqrt(
            np.mean(
                residual_distances
                ** 2
            )
        )
    )

    fre_mean = float(
        np.mean(
            residual_distances
        )
    )

    fre_max = float(
        np.max(
            residual_distances
        )
    )

    return RigidRegistrationResult(
        transform=np.asarray(
            transform,
            dtype=float,
        ),
        rotation=np.asarray(
            rotation,
            dtype=float,
        ),
        translation=np.asarray(
            translation,
            dtype=float,
        ),
        transformed_source_points=np.asarray(
            transformed_source,
            dtype=float,
        ),
        residual_vectors=np.asarray(
            residual_vectors,
            dtype=float,
        ),
        residual_distances=np.asarray(
            residual_distances,
            dtype=float,
        ),
        fre_rms=fre_rms,
        fre_mean=fre_mean,
        fre_max=fre_max,
        singular_values=np.asarray(
            singular_values,
            dtype=float,
        ),
    )


def rotation_angle_between(
    first_rotation: np.ndarray,
    second_rotation: np.ndarray,
) -> float:
    """Return geodesic rotation-angle difference in radians."""

    first_rotation = np.asarray(
        first_rotation,
        dtype=float,
    )

    second_rotation = np.asarray(
        second_rotation,
        dtype=float,
    )

    if first_rotation.shape != (
        3,
        3,
    ):
        raise ValueError(
            "first_rotation must have shape (3, 3)."
        )

    if second_rotation.shape != (
        3,
        3,
    ):
        raise ValueError(
            "second_rotation must have shape (3, 3)."
        )

    relative = (
        first_rotation.T
        @ second_rotation
    )

    cosine_angle = float(
        (
            np.trace(
                relative
            )
            - 1.0
        )
        / 2.0
    )

    cosine_angle = float(
        np.clip(
            cosine_angle,
            -1.0,
            1.0,
        )
    )

    return float(
        np.arccos(
            cosine_angle
        )
    )


def transform_error(
    estimated_transform: np.ndarray,
    reference_transform: np.ndarray,
) -> TransformError:
    """Measure translation and rotation error between two rigid transforms."""

    estimated = np.asarray(
        estimated_transform,
        dtype=float,
    )

    reference = np.asarray(
        reference_transform,
        dtype=float,
    )

    if not is_homogeneous_transform(
        estimated
    ):
        raise ValueError(
            "estimated_transform must be a valid rigid transformation."
        )

    if not is_homogeneous_transform(
        reference
    ):
        raise ValueError(
            "reference_transform must be a valid rigid transformation."
        )

    translation_difference = (
        estimated[
            :3,
            3
        ]
        - reference[
            :3,
            3
        ]
    )

    translation_error_value = float(
        np.linalg.norm(
            translation_difference
        )
    )

    rotation_error = (
        rotation_angle_between(
            estimated[
                :3,
                :3,
            ],
            reference[
                :3,
                :3,
            ],
        )
    )

    return TransformError(
        translation_error=(
            translation_error_value
        ),
        rotation_angle_error_radians=float(
            rotation_error
        ),
        rotation_angle_error_degrees=float(
            np.degrees(
                rotation_error
            )
        ),
    )


def registration_error_at_targets(
    estimated_transform: np.ndarray,
    reference_transform: np.ndarray,
    target_points_source_frame: np.ndarray,
) -> np.ndarray:
    """Return target registration errors for evaluation points.

    This is an evaluation utility requiring a known reference transform.
    It is therefore intended for simulation/benchmark use, not runtime
    estimation.
    """

    targets = _validate_point_set(
        target_points_source_frame,
        name="target_points_source_frame",
    )

    estimated_points = transform_points(
        estimated_transform,
        targets,
    )

    reference_points = transform_points(
        reference_transform,
        targets,
    )

    return np.linalg.norm(
        estimated_points
        - reference_points,
        axis=1,
    )