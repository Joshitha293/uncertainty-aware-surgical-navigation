"""Uncertainty propagation for calibrated stereo reconstruction.

Phase 3 converts image-coordinate measurement uncertainty into a full
three-dimensional world-frame positional covariance.

The measurement vector is:

    z = [u_left, v_left, u_right, v_right]

with independent pixel measurement noise.

A numerical Jacobian of the calibrated triangulation function is evaluated
around the measured stereo correspondence:

    J = d X_world / d z

and first-order covariance propagation gives:

    Sigma_X = J Sigma_z J^T

The resulting covariance is represented directly by the project's existing
PositionUncertainty class.

This is a simulation and algorithm-validation model. Pixel-noise parameters
are not claimed to represent a specific clinical imaging system.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.perception.camera import CameraPose
from src.perception.image_geometry import (
    PixelCameraIntrinsics,
)
from src.perception.stereo_geometry import (
    triangulate_world_point_from_pixels,
)
from src.perception.uncertainty import (
    PositionUncertainty,
)


@dataclass(frozen=True)
class StereoUncertaintyResult:
    """Stereo localisation uncertainty and propagation diagnostics."""

    uncertainty: PositionUncertainty

    covariance: np.ndarray

    pixel_covariance: np.ndarray

    triangulation_jacobian: np.ndarray

    principal_sigma: float


def stereo_measurement_covariance(
    pixel_sigma: float,
) -> np.ndarray:
    """Return covariance for four independent pixel coordinates."""

    if (
        not np.isfinite(
            pixel_sigma
        )
        or pixel_sigma < 0.0
    ):
        raise ValueError(
            "pixel_sigma must be finite and non-negative."
        )

    return (
        float(
            pixel_sigma
        )
        ** 2
    ) * np.eye(
        4,
        dtype=float,
    )


def _measurement_vector(
    left_pixel: np.ndarray,
    right_pixel: np.ndarray,
) -> np.ndarray:
    """Combine two image coordinates into one four-element vector."""

    left = np.asarray(
        left_pixel,
        dtype=float,
    )

    right = np.asarray(
        right_pixel,
        dtype=float,
    )

    if (
        left.shape != (2,)
        or right.shape != (2,)
    ):
        raise ValueError(
            "left_pixel and right_pixel must each have shape (2,)."
        )

    if (
        not np.all(
            np.isfinite(
                left
            )
        )
        or not np.all(
            np.isfinite(
                right
            )
        )
    ):
        raise ValueError(
            "Stereo pixels must contain finite values."
        )

    return np.concatenate(
        (
            left,
            right,
        )
    )


def _triangulate_measurement(
    *,
    intrinsics: PixelCameraIntrinsics,
    left_pose: CameraPose,
    right_pose: CameraPose,
    measurement: np.ndarray,
) -> np.ndarray:
    """Triangulate one four-element stereo measurement vector."""

    measurement = np.asarray(
        measurement,
        dtype=float,
    )

    if measurement.shape != (4,):
        raise ValueError(
            "measurement must have shape (4,)."
        )

    result = (
        triangulate_world_point_from_pixels(
            intrinsics=intrinsics,
            left_pose=left_pose,
            right_pose=right_pose,
            left_pixel=measurement[
                :2
            ],
            right_pixel=measurement[
                2:
            ],
        )
    )

    return np.asarray(
        result.point_world,
        dtype=float,
    )


def numerical_stereo_jacobian(
    *,
    intrinsics: PixelCameraIntrinsics,
    left_pose: CameraPose,
    right_pose: CameraPose,
    left_pixel: np.ndarray,
    right_pixel: np.ndarray,
    finite_difference_step: float = 1e-3,
) -> np.ndarray:
    """Numerically differentiate world position with respect to pixels.

    Central finite differences are used for all four image coordinates.
    """

    if (
        not np.isfinite(
            finite_difference_step
        )
        or finite_difference_step <= 0.0
    ):
        raise ValueError(
            "finite_difference_step must be finite and positive."
        )

    measurement = (
        _measurement_vector(
            left_pixel,
            right_pixel,
        )
    )

    jacobian = np.zeros(
        (
            3,
            4,
        ),
        dtype=float,
    )

    for dimension in range(
        4
    ):
        plus = measurement.copy()
        minus = measurement.copy()

        plus[
            dimension
        ] += finite_difference_step

        minus[
            dimension
        ] -= finite_difference_step

        point_plus = (
            _triangulate_measurement(
                intrinsics=intrinsics,
                left_pose=left_pose,
                right_pose=right_pose,
                measurement=plus,
            )
        )

        point_minus = (
            _triangulate_measurement(
                intrinsics=intrinsics,
                left_pose=left_pose,
                right_pose=right_pose,
                measurement=minus,
            )
        )

        jacobian[
            :,
            dimension
        ] = (
            point_plus
            - point_minus
        ) / (
            2.0
            * finite_difference_step
        )

    return jacobian


def propagate_stereo_pixel_uncertainty(
    *,
    intrinsics: PixelCameraIntrinsics,
    left_pose: CameraPose,
    right_pose: CameraPose,
    left_pixel: np.ndarray,
    right_pixel: np.ndarray,
    pixel_sigma: float,
    finite_difference_step: float = 1e-3,
) -> StereoUncertaintyResult:
    """Propagate stereo pixel noise into world-frame position covariance."""

    pixel_covariance = (
        stereo_measurement_covariance(
            pixel_sigma
        )
    )

    jacobian = (
        numerical_stereo_jacobian(
            intrinsics=intrinsics,
            left_pose=left_pose,
            right_pose=right_pose,
            left_pixel=left_pixel,
            right_pixel=right_pixel,
            finite_difference_step=(
                finite_difference_step
            ),
        )
    )

    covariance = (
        jacobian
        @ pixel_covariance
        @ jacobian.T
    )

    # Remove numerical asymmetry.
    covariance = (
        0.5
        * (
            covariance
            + covariance.T
        )
    )

    # Numerical round-off can introduce tiny negative eigenvalues.
    eigenvalues, eigenvectors = (
        np.linalg.eigh(
            covariance
        )
    )

    eigenvalues = np.maximum(
        eigenvalues,
        0.0,
    )

    covariance = (
        eigenvectors
        @ np.diag(
            eigenvalues
        )
        @ eigenvectors.T
    )

    covariance = (
        0.5
        * (
            covariance
            + covariance.T
        )
    )

    uncertainty = PositionUncertainty(
        covariance=covariance
    )

    return StereoUncertaintyResult(
        uncertainty=uncertainty,
        covariance=np.asarray(
            covariance,
            dtype=float,
        ),
        pixel_covariance=np.asarray(
            pixel_covariance,
            dtype=float,
        ),
        triangulation_jacobian=np.asarray(
            jacobian,
            dtype=float,
        ),
        principal_sigma=float(
            uncertainty.principal_sigma
        ),
    )