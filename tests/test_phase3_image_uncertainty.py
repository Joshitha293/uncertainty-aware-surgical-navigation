"""Tests for Phase 3 image-derived uncertainty-aware perception."""

import numpy as np
import pytest

from src.perception.camera import (
    CameraPose,
)
from src.perception.image_driven_perception import (
    ImagePerceptionFailure,
    evaluate_image_perception_error,
    perceive_structure_from_stereo_images,
)
from src.perception.image_geometry import (
    PixelCameraIntrinsics,
    project_world_point,
)
from src.perception.image_processing import (
    render_synthetic_marker_frame,
)
from src.perception.stereo_uncertainty import (
    numerical_stereo_jacobian,
    propagate_stereo_pixel_uncertainty,
    stereo_measurement_covariance,
)
from src.perception.uncertainty import (
    PositionUncertainty,
    inflate_estimated_structures,
)


def make_intrinsics() -> PixelCameraIntrinsics:
    return PixelCameraIntrinsics(
        width=320,
        height=240,
        fx=250.0,
        fy=250.0,
        cx=159.5,
        cy=119.5,
    )


def left_pose() -> CameraPose:
    return CameraPose(
        position=np.zeros(
            3,
            dtype=float,
        ),
        rotation=np.eye(
            3,
            dtype=float,
        ),
    )


def right_pose(
    baseline: float = 0.02,
) -> CameraPose:
    return CameraPose(
        position=np.asarray(
            [
                baseline,
                0.0,
                0.0,
            ],
            dtype=float,
        ),
        rotation=np.eye(
            3,
            dtype=float,
        ),
    )


def target_position() -> np.ndarray:
    return np.asarray(
        [
            0.020,
            -0.010,
            0.250,
        ],
        dtype=float,
    )


def stereo_pixels():
    intrinsics = (
        make_intrinsics()
    )

    target = target_position()

    left = (
        project_world_point(
            intrinsics,
            left_pose(),
            target,
        )
        .pixel
    )

    right = (
        project_world_point(
            intrinsics,
            right_pose(),
            target,
        )
        .pixel
    )

    return (
        left,
        right,
    )


def make_stereo_images():
    intrinsics = (
        make_intrinsics()
    )

    left_pixel, right_pixel = (
        stereo_pixels()
    )

    left_image = (
        render_synthetic_marker_frame(
            width=intrinsics.width,
            height=intrinsics.height,
            marker_pixel=(
                left_pixel
            ),
            marker_radius=14,
            seed=101,
        )
    )

    right_image = (
        render_synthetic_marker_frame(
            width=intrinsics.width,
            height=intrinsics.height,
            marker_pixel=(
                right_pixel
            ),
            marker_radius=14,
            seed=102,
        )
    )

    return (
        left_image,
        right_image,
    )


def test_pixel_covariance_has_expected_shape():
    covariance = (
        stereo_measurement_covariance(
            0.5
        )
    )

    assert covariance.shape == (
        4,
        4,
    )

    np.testing.assert_allclose(
        covariance,
        0.25
        * np.eye(
            4
        ),
        atol=1e-12,
    )


def test_zero_pixel_sigma_produces_zero_measurement_covariance():
    covariance = (
        stereo_measurement_covariance(
            0.0
        )
    )

    np.testing.assert_allclose(
        covariance,
        np.zeros(
            (
                4,
                4,
            )
        ),
        atol=1e-12,
    )


def test_negative_pixel_sigma_is_rejected():
    with pytest.raises(
        ValueError,
        match="pixel_sigma",
    ):
        stereo_measurement_covariance(
            -0.1
        )


def test_stereo_jacobian_has_correct_shape_and_finite_values():
    left_pixel, right_pixel = (
        stereo_pixels()
    )

    jacobian = (
        numerical_stereo_jacobian(
            intrinsics=(
                make_intrinsics()
            ),
            left_pose=left_pose(),
            right_pose=right_pose(),
            left_pixel=left_pixel,
            right_pixel=right_pixel,
        )
    )

    assert jacobian.shape == (
        3,
        4,
    )

    assert np.all(
        np.isfinite(
            jacobian
        )
    )


def test_propagated_covariance_is_symmetric_psd():
    left_pixel, right_pixel = (
        stereo_pixels()
    )

    result = (
        propagate_stereo_pixel_uncertainty(
            intrinsics=(
                make_intrinsics()
            ),
            left_pose=left_pose(),
            right_pose=right_pose(),
            left_pixel=left_pixel,
            right_pixel=right_pixel,
            pixel_sigma=0.5,
        )
    )

    np.testing.assert_allclose(
        result.covariance,
        result.covariance.T,
        atol=1e-12,
    )

    eigenvalues = (
        np.linalg.eigvalsh(
            result.covariance
        )
    )

    assert np.all(
        eigenvalues
        >= -1e-12
    )


def test_propagated_result_uses_existing_position_uncertainty():
    left_pixel, right_pixel = (
        stereo_pixels()
    )

    result = (
        propagate_stereo_pixel_uncertainty(
            intrinsics=(
                make_intrinsics()
            ),
            left_pose=left_pose(),
            right_pose=right_pose(),
            left_pixel=left_pixel,
            right_pixel=right_pixel,
            pixel_sigma=0.5,
        )
    )

    assert isinstance(
        result.uncertainty,
        PositionUncertainty,
    )

    assert (
        result.principal_sigma
        == pytest.approx(
            result
            .uncertainty
            .principal_sigma
        )
    )


def test_larger_pixel_noise_produces_larger_position_uncertainty():
    left_pixel, right_pixel = (
        stereo_pixels()
    )

    low = (
        propagate_stereo_pixel_uncertainty(
            intrinsics=(
                make_intrinsics()
            ),
            left_pose=left_pose(),
            right_pose=right_pose(),
            left_pixel=left_pixel,
            right_pixel=right_pixel,
            pixel_sigma=0.25,
        )
    )

    high = (
        propagate_stereo_pixel_uncertainty(
            intrinsics=(
                make_intrinsics()
            ),
            left_pose=left_pose(),
            right_pose=right_pose(),
            left_pixel=left_pixel,
            right_pixel=right_pixel,
            pixel_sigma=1.0,
        )
    )

    assert (
        high.principal_sigma
        > low.principal_sigma
    )


def test_image_pair_produces_existing_perception_result_structure():
    left_image, right_image = (
        make_stereo_images()
    )

    result = (
        perceive_structure_from_stereo_images(
            left_image=left_image,
            right_image=right_image,
            intrinsics=(
                make_intrinsics()
            ),
            left_pose=left_pose(),
            right_pose=right_pose(),
            physical_radius=0.025,
            base_safety_margin=0.015,
            pixel_sigma=0.5,
        )
    )

    assert (
        len(
            result
            .perception_result
            .estimated_structures
        )
        == 1
    )

    estimate = (
        result
        .perception_result
        .estimated_structures[
            0
        ]
    )

    assert isinstance(
        estimate.uncertainty,
        PositionUncertainty,
    )


def test_runtime_perception_does_not_populate_ground_truth_error():
    left_image, right_image = (
        make_stereo_images()
    )

    result = (
        perceive_structure_from_stereo_images(
            left_image=left_image,
            right_image=right_image,
            intrinsics=(
                make_intrinsics()
            ),
            left_pose=left_pose(),
            right_pose=right_pose(),
            physical_radius=0.025,
            base_safety_margin=0.015,
        )
    )

    assert (
        result
        .perception_result
        .localisation_errors
        .size
        == 0
    )


def test_image_derived_3d_estimate_is_close_to_simulation_truth():
    left_image, right_image = (
        make_stereo_images()
    )

    result = (
        perceive_structure_from_stereo_images(
            left_image=left_image,
            right_image=right_image,
            intrinsics=(
                make_intrinsics()
            ),
            left_pose=left_pose(),
            right_pose=right_pose(),
            physical_radius=0.025,
            base_safety_margin=0.015,
            pixel_sigma=0.5,
        )
    )

    error = (
        evaluate_image_perception_error(
            result,
            target_position(),
        )
    )

    assert error < 0.002


def test_image_derived_uncertainty_can_inflate_planner_margin():
    left_image, right_image = (
        make_stereo_images()
    )

    result = (
        perceive_structure_from_stereo_images(
            left_image=left_image,
            right_image=right_image,
            intrinsics=(
                make_intrinsics()
            ),
            left_pose=left_pose(),
            right_pose=right_pose(),
            physical_radius=0.025,
            base_safety_margin=0.015,
            pixel_sigma=0.5,
        )
    )

    estimate = (
        result
        .perception_result
        .estimated_structures[
            0
        ]
    )

    inflated = (
        inflate_estimated_structures(
            estimated_structures=(
                estimate,
            ),
            sigma_multiplier=2.0,
        )
    )

    assert len(
        inflated
    ) == 1

    assert (
        inflated[
            0
        ].safety_margin
        > estimate.base_safety_margin
    )


def test_missing_image_target_fails_explicitly():
    blank = np.zeros(
        (
            240,
            320,
            3,
        ),
        dtype=np.uint8,
    )

    with pytest.raises(
        ImagePerceptionFailure,
        match="left image",
    ):
        perceive_structure_from_stereo_images(
            left_image=blank,
            right_image=blank,
            intrinsics=(
                make_intrinsics()
            ),
            left_pose=left_pose(),
            right_pose=right_pose(),
            physical_radius=0.025,
            base_safety_margin=0.015,
        )