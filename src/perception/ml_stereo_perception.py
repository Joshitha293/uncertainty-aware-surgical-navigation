"""Learned stereo perception integrated with navigation uncertainty.

This module connects Phase 4 learned image segmentation to the existing
stereo geometry and uncertainty-aware planning representation.

Repeated perturbed inference provides an empirical image-centroid
distribution independently for the left and right images. Those 2-D
covariances are assembled into a 4-D stereo measurement covariance and
propagated through the numerical triangulation Jacobian:

    Sigma_X = J Sigma_pixels J^T

A conservative isotropic pixel-variance floor is added because repeated
network predictions can be stable while still being systematically wrong,
particularly under distribution shift.

The resulting world-frame covariance is represented using the existing
PositionUncertainty class and therefore connects directly to the existing
uncertainty-aware planner.

This is simulation-only engineering validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.perception.camera import CameraPose
from src.perception.image_geometry import (
    PixelCameraIntrinsics,
)
from src.perception.ml_uncertainty import (
    PredictiveSegmentationUncertainty,
    predict_with_perturbation_ensemble,
)
from src.perception.perception import (
    PerceptionResult,
)
from src.perception.stereo_geometry import (
    StereoTriangulationResult,
    triangulate_world_point_from_pixels,
)
from src.perception.stereo_uncertainty import (
    numerical_stereo_jacobian,
)
from src.perception.uncertainty import (
    EstimatedStructure,
    PositionUncertainty,
)


class LearnedStereoPerceptionFailure(
    RuntimeError
):
    """Raised when learned image evidence cannot support stereo perception."""


@dataclass(frozen=True)
class LearnedStereoPerceptionResult:
    """Output of learned uncertainty-aware stereo perception."""

    perception_result: PerceptionResult

    left_prediction: PredictiveSegmentationUncertainty

    right_prediction: PredictiveSegmentationUncertainty

    triangulation: StereoTriangulationResult

    stereo_pixel_covariance: np.ndarray

    world_position_covariance: np.ndarray

    triangulation_jacobian: np.ndarray


def _regularise_centroid_covariance(
    prediction: PredictiveSegmentationUncertainty,
    *,
    minimum_pixel_sigma: float,
) -> np.ndarray:
    """Return positive-semidefinite 2-D covariance with safety floor."""

    if (
        not np.isfinite(
            minimum_pixel_sigma
        )
        or minimum_pixel_sigma < 0.0
    ):
        raise ValueError(
            "minimum_pixel_sigma must be finite and non-negative."
        )

    if prediction.centroid_mean is None:
        raise LearnedStereoPerceptionFailure(
            "Learned segmentation produced no usable centroid."
        )

    if prediction.centroid_covariance is None:
        empirical = np.zeros(
            (
                2,
                2,
            ),
            dtype=float,
        )
    else:
        empirical = np.asarray(
            prediction.centroid_covariance,
            dtype=float,
        )

    if empirical.shape != (
        2,
        2,
    ):
        raise ValueError(
            "centroid covariance must have shape (2, 2)."
        )

    empirical = (
        0.5
        * (
            empirical
            + empirical.T
        )
    )

    eigenvalues, eigenvectors = (
        np.linalg.eigh(
            empirical
        )
    )

    eigenvalues = np.maximum(
        eigenvalues,
        0.0,
    )

    empirical = (
        eigenvectors
        @ np.diag(
            eigenvalues
        )
        @ eigenvectors.T
    )

    floor = (
        minimum_pixel_sigma
        ** 2
    ) * np.eye(
        2,
        dtype=float,
    )

    covariance = (
        empirical
        + floor
    )

    return (
        0.5
        * (
            covariance
            + covariance.T
        )
    )


def stereo_centroid_measurement_covariance(
    *,
    left_prediction: PredictiveSegmentationUncertainty,
    right_prediction: PredictiveSegmentationUncertainty,
    minimum_pixel_sigma: float = 0.5,
) -> np.ndarray:
    """Construct covariance for [uL, vL, uR, vR]."""

    left_covariance = (
        _regularise_centroid_covariance(
            left_prediction,
            minimum_pixel_sigma=(
                minimum_pixel_sigma
            ),
        )
    )

    right_covariance = (
        _regularise_centroid_covariance(
            right_prediction,
            minimum_pixel_sigma=(
                minimum_pixel_sigma
            ),
        )
    )

    covariance = np.zeros(
        (
            4,
            4,
        ),
        dtype=float,
    )

    covariance[
        0:2,
        0:2,
    ] = left_covariance

    covariance[
        2:4,
        2:4,
    ] = right_covariance

    return covariance


def propagate_learned_stereo_covariance(
    *,
    intrinsics: PixelCameraIntrinsics,
    left_pose: CameraPose,
    right_pose: CameraPose,
    left_pixel: np.ndarray,
    right_pixel: np.ndarray,
    pixel_covariance: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Propagate a full stereo pixel covariance into world coordinates."""

    pixel_covariance = np.asarray(
        pixel_covariance,
        dtype=float,
    )

    if pixel_covariance.shape != (
        4,
        4,
    ):
        raise ValueError(
            "pixel_covariance must have shape (4, 4)."
        )

    jacobian = (
        numerical_stereo_jacobian(
            intrinsics=intrinsics,
            left_pose=left_pose,
            right_pose=right_pose,
            left_pixel=left_pixel,
            right_pixel=right_pixel,
        )
    )

    covariance = (
        jacobian
        @ pixel_covariance
        @ jacobian.T
    )

    covariance = (
        0.5
        * (
            covariance
            + covariance.T
        )
    )

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

    return (
        covariance,
        jacobian,
    )


def perceive_structure_from_learned_stereo(
    *,
    model: nn.Module,
    left_image: np.ndarray,
    right_image: np.ndarray,
    intrinsics: PixelCameraIntrinsics,
    left_pose: CameraPose,
    right_pose: CameraPose,
    physical_radius: float,
    base_safety_margin: float,
    device: torch.device,
    ensemble_samples: int = 20,
    minimum_pixel_sigma: float = 0.5,
    noise_sigma: float = 4.0,
    brightness_sigma: float = 0.04,
    seed: int = 4901,
) -> LearnedStereoPerceptionResult:
    """Estimate one navigation structure from learned stereo perception.

    Hidden simulator truth is deliberately absent from this runtime
    interface.
    """

    if (
        not np.isfinite(
            physical_radius
        )
        or physical_radius <= 0.0
    ):
        raise ValueError(
            "physical_radius must be finite and positive."
        )

    if (
        not np.isfinite(
            base_safety_margin
        )
        or base_safety_margin < 0.0
    ):
        raise ValueError(
            "base_safety_margin must be finite and non-negative."
        )

    left_prediction = (
        predict_with_perturbation_ensemble(
            model=model,
            image_bgr=left_image,
            device=device,
            sample_count=ensemble_samples,
            noise_sigma=noise_sigma,
            brightness_sigma=brightness_sigma,
            seed=seed,
        )
    )

    right_prediction = (
        predict_with_perturbation_ensemble(
            model=model,
            image_bgr=right_image,
            device=device,
            sample_count=ensemble_samples,
            noise_sigma=noise_sigma,
            brightness_sigma=brightness_sigma,
            seed=seed + 1,
        )
    )

    if left_prediction.centroid_mean is None:
        raise LearnedStereoPerceptionFailure(
            "Left learned segmentation produced no target centroid."
        )

    if right_prediction.centroid_mean is None:
        raise LearnedStereoPerceptionFailure(
            "Right learned segmentation produced no target centroid."
        )

    left_pixel = np.asarray(
        left_prediction.centroid_mean,
        dtype=float,
    )

    right_pixel = np.asarray(
        right_prediction.centroid_mean,
        dtype=float,
    )

    triangulation = (
        triangulate_world_point_from_pixels(
            intrinsics=intrinsics,
            left_pose=left_pose,
            right_pose=right_pose,
            left_pixel=left_pixel,
            right_pixel=right_pixel,
        )
    )

    stereo_pixel_covariance = (
        stereo_centroid_measurement_covariance(
            left_prediction=(
                left_prediction
            ),
            right_prediction=(
                right_prediction
            ),
            minimum_pixel_sigma=(
                minimum_pixel_sigma
            ),
        )
    )

    (
        world_covariance,
        jacobian,
    ) = (
        propagate_learned_stereo_covariance(
            intrinsics=intrinsics,
            left_pose=left_pose,
            right_pose=right_pose,
            left_pixel=left_pixel,
            right_pixel=right_pixel,
            pixel_covariance=(
                stereo_pixel_covariance
            ),
        )
    )

    uncertainty = PositionUncertainty(
        covariance=world_covariance
    )

    estimate = EstimatedStructure(
        estimated_centre=np.asarray(
            triangulation.point_world,
            dtype=float,
        ),
        physical_radius=float(
            physical_radius
        ),
        base_safety_margin=float(
            base_safety_margin
        ),
        uncertainty=uncertainty,
    )

    perception_result = PerceptionResult(
        estimated_structures=(
            estimate,
        ),
        # Ground truth remains evaluation-only.
        localisation_errors=np.empty(
            (
                0,
            ),
            dtype=float,
        ),
    )

    return LearnedStereoPerceptionResult(
        perception_result=(
            perception_result
        ),
        left_prediction=(
            left_prediction
        ),
        right_prediction=(
            right_prediction
        ),
        triangulation=(
            triangulation
        ),
        stereo_pixel_covariance=(
            stereo_pixel_covariance
        ),
        world_position_covariance=(
            world_covariance
        ),
        triangulation_jacobian=(
            jacobian
        ),
    )