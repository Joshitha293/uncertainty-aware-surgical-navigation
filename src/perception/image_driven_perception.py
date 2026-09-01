"""Image-derived uncertainty-aware perception for Phase 3.

This module connects the Phase 3 computer-vision pipeline to the existing
uncertainty-aware navigation representation.

Runtime information flow:

    left/right BGR images
        -> OpenCV segmentation
        -> measured image centroids
        -> calibrated stereo triangulation
        -> world-frame 3-D estimate
        -> stereo uncertainty propagation
        -> EstimatedStructure
        -> PerceptionResult

Ground-truth anatomy is deliberately absent from this runtime interface.

Physical radius and base safety margin are supplied as known structure
metadata. They describe the object model rather than its hidden position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.perception.camera import CameraPose
from src.perception.image_geometry import (
    PixelCameraIntrinsics,
)
from src.perception.image_processing import (
    SegmentationConfig,
    SegmentationResult,
    segment_target_bgr,
)
from src.perception.perception import (
    PerceptionResult,
)
from src.perception.stereo_geometry import (
    StereoTriangulationResult,
    triangulate_world_point_from_pixels,
)
from src.perception.stereo_uncertainty import (
    StereoUncertaintyResult,
    propagate_stereo_pixel_uncertainty,
)
from src.perception.uncertainty import (
    EstimatedStructure,
)


class ImagePerceptionFailure(
    RuntimeError
):
    """Raised when image evidence is insufficient for 3-D perception."""


@dataclass(frozen=True)
class StereoImagePerceptionResult:
    """Complete output of one image-derived stereo perception step."""

    perception_result: PerceptionResult

    left_segmentation: SegmentationResult

    right_segmentation: SegmentationResult

    triangulation: StereoTriangulationResult

    uncertainty: StereoUncertaintyResult


def perceive_structure_from_stereo_images(
    *,
    left_image: np.ndarray,
    right_image: np.ndarray,
    intrinsics: PixelCameraIntrinsics,
    left_pose: CameraPose,
    right_pose: CameraPose,
    physical_radius: float,
    base_safety_margin: float,
    pixel_sigma: float = 0.5,
    segmentation_config: (
        SegmentationConfig
        | None
    ) = None,
) -> StereoImagePerceptionResult:
    """Estimate one 3-D structure directly from a calibrated stereo pair.

    No simulator ground-truth position is accepted by this function.

    ``localisation_errors`` in the returned PerceptionResult is empty because
    true position is not available to the runtime perception system.
    Ground-truth error is an evaluation-only quantity calculated separately
    by experiments.
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

    left_segmentation = (
        segment_target_bgr(
            left_image,
            config=(
                segmentation_config
            ),
        )
    )

    right_segmentation = (
        segment_target_bgr(
            right_image,
            config=(
                segmentation_config
            ),
        )
    )

    if (
        not left_segmentation
        .detection
        .found
    ):
        raise ImagePerceptionFailure(
            "Target was not detected in the left image."
        )

    if (
        not right_segmentation
        .detection
        .found
    ):
        raise ImagePerceptionFailure(
            "Target was not detected in the right image."
        )

    left_pixel = np.asarray(
        left_segmentation
        .detection
        .centroid,
        dtype=float,
    )

    right_pixel = np.asarray(
        right_segmentation
        .detection
        .centroid,
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

    uncertainty = (
        propagate_stereo_pixel_uncertainty(
            intrinsics=intrinsics,
            left_pose=left_pose,
            right_pose=right_pose,
            left_pixel=left_pixel,
            right_pixel=right_pixel,
            pixel_sigma=pixel_sigma,
        )
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
        uncertainty=(
            uncertainty.uncertainty
        ),
    )

    perception = PerceptionResult(
        estimated_structures=(
            estimate,
        ),
        # Ground truth is intentionally unavailable here.
        localisation_errors=np.empty(
            (
                0,
            ),
            dtype=float,
        ),
    )

    return StereoImagePerceptionResult(
        perception_result=perception,
        left_segmentation=(
            left_segmentation
        ),
        right_segmentation=(
            right_segmentation
        ),
        triangulation=(
            triangulation
        ),
        uncertainty=uncertainty,
    )


def evaluate_image_perception_error(
    result: StereoImagePerceptionResult,
    reference_position: np.ndarray,
) -> float:
    """Evaluate one stereo estimate against simulation truth.

    This function is deliberately separate from runtime perception so hidden
    simulator truth remains evaluation-only.
    """

    reference = np.asarray(
        reference_position,
        dtype=float,
    )

    if reference.shape != (3,):
        raise ValueError(
            "reference_position must have shape (3,)."
        )

    if not np.all(
        np.isfinite(
            reference
        )
    ):
        raise ValueError(
            "reference_position must contain finite values."
        )

    estimate = (
        result
        .perception_result
        .estimated_structures[
            0
        ]
        .estimated_centre
    )

    return float(
        np.linalg.norm(
            estimate
            - reference
        )
    )