"""Tests for learned stereo perception -> navigation uncertainty."""

import numpy as np
import pytest
import torch
from torch import nn

from src.perception.camera import (
    CameraPose,
)
from src.perception.image_geometry import (
    PixelCameraIntrinsics,
    project_world_point,
)
from src.perception.image_processing import (
    render_synthetic_marker_frame,
)
from src.perception.ml_stereo_perception import (
    perceive_structure_from_learned_stereo,
    propagate_learned_stereo_covariance,
)
from src.perception.uncertainty import (
    PositionUncertainty,
    inflate_estimated_structures,
)


class GreenChannelModel(
    nn.Module
):
    """Deterministic surrogate responding to green dominance.

    The model is used only for integration testing.

    Neutral or white/specular pixels are forced below the foreground
    threshold. This prevents pixels where R ~= G ~= B from producing
    sigmoid(0) == 0.5 and being incorrectly included by a >= 0.5 mask.
    """

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        red = x[
            :,
            0:1,
        ]

        green = x[
            :,
            1:2,
        ]

        blue = x[
            :,
            2:3,
        ]

        strongest_non_green = torch.maximum(
            red,
            blue,
        )

        green_dominance = (
            green
            - strongest_non_green
        )

        return (
            20.0
            * green_dominance
            - 1.0
        )


def make_intrinsics() -> PixelCameraIntrinsics:
    """Return synthetic stereo-camera intrinsics."""

    return PixelCameraIntrinsics(
        width=96,
        height=96,
        fx=180.0,
        fy=180.0,
        cx=47.5,
        cy=47.5,
    )


def make_left_pose() -> CameraPose:
    """Return left-camera pose."""

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


def make_right_pose() -> CameraPose:
    """Return right-camera pose with 20 mm stereo baseline."""

    return CameraPose(
        position=np.asarray(
            [
                0.02,
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
    """Return hidden simulation target used only for test evaluation."""

    return np.asarray(
        [
            0.010,
            -0.005,
            0.250,
        ],
        dtype=float,
    )


def make_images() -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Render deterministic synthetic stereo images."""

    intrinsics = (
        make_intrinsics()
    )

    target = (
        target_position()
    )

    left_pixel = (
        project_world_point(
            intrinsics,
            make_left_pose(),
            target,
        )
        .pixel
    )

    right_pixel = (
        project_world_point(
            intrinsics,
            make_right_pose(),
            target,
        )
        .pixel
    )

    left_image = (
        render_synthetic_marker_frame(
            width=96,
            height=96,
            marker_pixel=(
                left_pixel
            ),
            marker_radius=9,
            seed=811,
        )
    )

    right_image = (
        render_synthetic_marker_frame(
            width=96,
            height=96,
            marker_pixel=(
                right_pixel
            ),
            marker_radius=9,
            seed=812,
        )
    )

    return (
        left_image,
        right_image,
    )


def make_result():
    """Run one deterministic learned-stereo perception result."""

    left_image, right_image = (
        make_images()
    )

    return (
        perceive_structure_from_learned_stereo(
            model=GreenChannelModel(),
            left_image=left_image,
            right_image=right_image,
            intrinsics=(
                make_intrinsics()
            ),
            left_pose=(
                make_left_pose()
            ),
            right_pose=(
                make_right_pose()
            ),
            physical_radius=0.025,
            base_safety_margin=0.015,
            device=torch.device(
                "cpu"
            ),
            ensemble_samples=6,
            minimum_pixel_sigma=0.5,
            noise_sigma=2.0,
            brightness_sigma=0.02,
            seed=91,
        )
    )


def test_learned_stereo_produces_one_estimated_structure():
    """Learned stereo should return one navigation structure."""

    result = make_result()

    assert (
        len(
            result
            .perception_result
            .estimated_structures
        )
        == 1
    )


def test_runtime_result_contains_no_ground_truth_error():
    """Hidden simulator truth must not enter runtime perception."""

    result = make_result()

    assert (
        result
        .perception_result
        .localisation_errors
        .size
        == 0
    )


def test_learned_stereo_uses_existing_position_uncertainty():
    """Learned stereo must integrate with the existing uncertainty API."""

    result = make_result()

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


def test_world_covariance_is_symmetric_positive_semidefinite():
    """Propagated 3-D covariance must be valid."""

    result = make_result()

    covariance = (
        result
        .world_position_covariance
    )

    assert covariance.shape == (
        3,
        3,
    )

    np.testing.assert_allclose(
        covariance,
        covariance.T,
        atol=1e-12,
    )

    eigenvalues = (
        np.linalg.eigvalsh(
            covariance
        )
    )

    assert np.all(
        eigenvalues
        >= -1e-12
    )


def test_learned_estimate_is_close_to_simulation_truth():
    """Stereo estimate should remain accurate for the synthetic test target."""

    result = make_result()

    estimate = (
        result
        .perception_result
        .estimated_structures[
            0
        ]
        .estimated_centre
    )

    error = float(
        np.linalg.norm(
            estimate
            - target_position()
        )
    )

    # Keep the integration requirement strict.
    assert error < 0.015


def test_learned_centroids_preserve_reasonable_stereo_disparity():
    """Predicted stereo disparity should remain close to ideal geometry."""

    result = make_result()

    intrinsics = (
        make_intrinsics()
    )

    target = (
        target_position()
    )

    expected_left = (
        project_world_point(
            intrinsics,
            make_left_pose(),
            target,
        )
        .pixel
    )

    expected_right = (
        project_world_point(
            intrinsics,
            make_right_pose(),
            target,
        )
        .pixel
    )

    predicted_left = (
        result
        .left_prediction
        .centroid_mean
    )

    predicted_right = (
        result
        .right_prediction
        .centroid_mean
    )

    assert predicted_left is not None
    assert predicted_right is not None

    expected_disparity = float(
        expected_left[
            0
        ]
        - expected_right[
            0
        ]
    )

    predicted_disparity = float(
        predicted_left[
            0
        ]
        - predicted_right[
            0
        ]
    )

    assert predicted_disparity == pytest.approx(
        expected_disparity,
        abs=0.75,
    )


def test_learned_uncertainty_inflates_planner_margin():
    """Learned perception covariance must affect planning geometry."""

    result = make_result()

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


def test_pixel_covariance_has_expected_shape():
    """Learned stereo should produce full four-dimensional pixel covariance."""

    result = make_result()

    covariance = (
        result
        .stereo_pixel_covariance
    )

    assert covariance.shape == (
        4,
        4,
    )

    np.testing.assert_allclose(
        covariance,
        covariance.T,
        atol=1e-12,
    )


def test_triangulation_jacobian_has_expected_shape():
    """Stereo measurement Jacobian should map four pixels into 3-D."""

    result = make_result()

    jacobian = (
        result
        .triangulation_jacobian
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


def test_invalid_pixel_covariance_shape_is_rejected():
    """Malformed stereo covariance should fail explicitly."""

    with pytest.raises(
        ValueError,
        match="shape",
    ):
        propagate_learned_stereo_covariance(
            intrinsics=(
                make_intrinsics()
            ),
            left_pose=(
                make_left_pose()
            ),
            right_pose=(
                make_right_pose()
            ),
            left_pixel=np.asarray(
                [
                    50.0,
                    48.0,
                ],
                dtype=float,
            ),
            right_pixel=np.asarray(
                [
                    36.0,
                    48.0,
                ],
                dtype=float,
            ),
            pixel_covariance=np.eye(
                3,
                dtype=float,
            ),
        )


def test_larger_pixel_covariance_produces_larger_world_uncertainty():
    """Increasing image-space uncertainty should increase 3-D uncertainty."""

    intrinsics = (
        make_intrinsics()
    )

    target = (
        target_position()
    )

    left_pixel = (
        project_world_point(
            intrinsics,
            make_left_pose(),
            target,
        )
        .pixel
    )

    right_pixel = (
        project_world_point(
            intrinsics,
            make_right_pose(),
            target,
        )
        .pixel
    )

    low_covariance, _ = (
        propagate_learned_stereo_covariance(
            intrinsics=intrinsics,
            left_pose=(
                make_left_pose()
            ),
            right_pose=(
                make_right_pose()
            ),
            left_pixel=left_pixel,
            right_pixel=right_pixel,
            pixel_covariance=(
                0.25
                * np.eye(
                    4,
                    dtype=float,
                )
            ),
        )
    )

    high_covariance, _ = (
        propagate_learned_stereo_covariance(
            intrinsics=intrinsics,
            left_pose=(
                make_left_pose()
            ),
            right_pose=(
                make_right_pose()
            ),
            left_pixel=left_pixel,
            right_pixel=right_pixel,
            pixel_covariance=(
                1.00
                * np.eye(
                    4,
                    dtype=float,
                )
            ),
        )
    )

    low_sigma = float(
        np.sqrt(
            np.max(
                np.linalg.eigvalsh(
                    low_covariance
                )
            )
        )
    )

    high_sigma = float(
        np.sqrt(
            np.max(
                np.linalg.eigvalsh(
                    high_covariance
                )
            )
        )
    )

    assert (
        high_sigma
        > low_sigma
    )