"""Tests for Phase 3 stereo reconstruction and point clouds."""

import numpy as np
import pytest

from src.perception.camera import (
    CameraPose,
)
from src.perception.image_geometry import (
    PixelCameraIntrinsics,
    project_world_point,
)
from src.perception.image_processing import (
    render_synthetic_marker_frame,
    segment_target_bgr,
)
from src.perception.stereo_geometry import (
    camera_points_to_world,
    depth_image_to_camera_points,
    depth_image_to_world_points,
    localisation_error,
    reconstruct_rectified_stereo_point,
    rectified_stereo_depth,
    triangulate_world_point_from_pixels,
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


def test_rectified_depth_formula():
    depth = (
        rectified_stereo_depth(
            focal_length_pixels=250.0,
            baseline=0.02,
            disparity=20.0,
        )
    )

    assert depth == pytest.approx(
        0.25
    )


def test_non_positive_disparity_is_rejected():
    with pytest.raises(
        ValueError,
        match="disparity",
    ):
        rectified_stereo_depth(
            focal_length_pixels=250.0,
            baseline=0.02,
            disparity=0.0,
        )


def test_rectified_reconstruction_recovers_camera_point():
    intrinsics = (
        make_intrinsics()
    )

    point = np.asarray(
        [
            0.02,
            -0.01,
            0.25,
        ],
        dtype=float,
    )

    left = project_world_point(
        intrinsics,
        left_pose(),
        point,
    )

    right = project_world_point(
        intrinsics,
        right_pose(),
        point,
    )

    reconstructed = (
        reconstruct_rectified_stereo_point(
            intrinsics=intrinsics,
            left_pixel=left.pixel,
            right_pixel=right.pixel,
            baseline=0.02,
        )
    )

    np.testing.assert_allclose(
        reconstructed,
        point,
        atol=1e-12,
    )


def test_general_triangulation_recovers_exact_world_point():
    intrinsics = (
        make_intrinsics()
    )

    target = np.asarray(
        [
            0.02,
            -0.01,
            0.25,
        ],
        dtype=float,
    )

    left_pixel = (
        project_world_point(
            intrinsics,
            left_pose(),
            target,
        )
        .pixel
    )

    right_pixel = (
        project_world_point(
            intrinsics,
            right_pose(),
            target,
        )
        .pixel
    )

    result = (
        triangulate_world_point_from_pixels(
            intrinsics=intrinsics,
            left_pose=left_pose(),
            right_pose=right_pose(),
            left_pixel=left_pixel,
            right_pixel=right_pixel,
        )
    )

    np.testing.assert_allclose(
        result.point_world,
        target,
        atol=1e-10,
    )


def test_exact_triangulation_has_negligible_ray_gap():
    intrinsics = (
        make_intrinsics()
    )

    target = np.asarray(
        [
            0.03,
            0.015,
            0.30,
        ]
    )

    left_pixel = (
        project_world_point(
            intrinsics,
            left_pose(),
            target,
        )
        .pixel
    )

    right_pixel = (
        project_world_point(
            intrinsics,
            right_pose(),
            target,
        )
        .pixel
    )

    result = (
        triangulate_world_point_from_pixels(
            intrinsics=intrinsics,
            left_pose=left_pose(),
            right_pose=right_pose(),
            left_pixel=left_pixel,
            right_pixel=right_pixel,
        )
    )

    assert (
        result.closest_ray_gap
        < 1e-10
    )


def test_exact_triangulation_has_negligible_reprojection_error():
    intrinsics = (
        make_intrinsics()
    )

    target = np.asarray(
        [
            -0.01,
            0.02,
            0.28,
        ]
    )

    left_pixel = (
        project_world_point(
            intrinsics,
            left_pose(),
            target,
        )
        .pixel
    )

    right_pixel = (
        project_world_point(
            intrinsics,
            right_pose(),
            target,
        )
        .pixel
    )

    result = (
        triangulate_world_point_from_pixels(
            intrinsics=intrinsics,
            left_pose=left_pose(),
            right_pose=right_pose(),
            left_pixel=left_pixel,
            right_pixel=right_pixel,
        )
    )

    assert (
        result.left_reprojection_error
        < 1e-8
    )

    assert (
        result.right_reprojection_error
        < 1e-8
    )


def test_parallel_stereo_rays_are_rejected():
    intrinsics = (
        make_intrinsics()
    )

    centre = np.asarray(
        [
            intrinsics.cx,
            intrinsics.cy,
        ]
    )

    with pytest.raises(
        ValueError,
        match="parallel",
    ):
        triangulate_world_point_from_pixels(
            intrinsics=intrinsics,
            left_pose=left_pose(),
            right_pose=right_pose(),
            left_pixel=centre,
            right_pixel=centre,
        )


def test_depth_image_reconstructs_valid_camera_points_only():
    intrinsics = (
        PixelCameraIntrinsics(
            width=3,
            height=2,
            fx=100.0,
            fy=100.0,
            cx=1.0,
            cy=0.5,
        )
    )

    depth = np.asarray(
        [
            [
                0.20,
                np.nan,
                -1.0,
            ],
            [
                0.30,
                0.0,
                0.40,
            ],
        ]
    )

    points = (
        depth_image_to_camera_points(
            intrinsics=intrinsics,
            depth_image=depth,
        )
    )

    assert points.shape == (
        3,
        3,
    )

    assert np.all(
        points[
            :,
            2
        ]
        > 0.0
    )


def test_principal_depth_pixel_reconstructs_optical_axis():
    intrinsics = (
        PixelCameraIntrinsics(
            width=3,
            height=3,
            fx=100.0,
            fy=100.0,
            cx=1.0,
            cy=1.0,
        )
    )

    depth = np.zeros(
        (
            3,
            3,
        ),
        dtype=float,
    )

    depth[
        1,
        1
    ] = 0.25

    points = (
        depth_image_to_camera_points(
            intrinsics=intrinsics,
            depth_image=depth,
        )
    )

    assert points.shape == (
        1,
        3,
    )

    np.testing.assert_allclose(
        points[0],
        np.asarray(
            [
                0.0,
                0.0,
                0.25,
            ]
        ),
        atol=1e-12,
    )


def test_camera_point_cloud_transforms_to_world():
    pose = CameraPose(
        position=np.asarray(
            [
                0.10,
                -0.05,
                0.02,
            ]
        ),
        rotation=np.eye(
            3
        ),
    )

    camera_points = np.asarray(
        [
            [
                0.0,
                0.0,
                0.20,
            ],
            [
                0.01,
                0.02,
                0.25,
            ],
        ]
    )

    world = (
        camera_points_to_world(
            pose=pose,
            points_camera=(
                camera_points
            ),
        )
    )

    expected = (
        camera_points
        + pose.position
    )

    np.testing.assert_allclose(
        world,
        expected,
        atol=1e-12,
    )


def test_depth_image_to_world_point_cloud():
    intrinsics = (
        PixelCameraIntrinsics(
            width=3,
            height=3,
            fx=100.0,
            fy=100.0,
            cx=1.0,
            cy=1.0,
        )
    )

    depth = np.zeros(
        (
            3,
            3,
        )
    )

    depth[
        1,
        1
    ] = 0.25

    pose = CameraPose(
        position=np.asarray(
            [
                0.10,
                0.20,
                0.30,
            ]
        ),
        rotation=np.eye(
            3
        ),
    )

    points = (
        depth_image_to_world_points(
            intrinsics=intrinsics,
            pose=pose,
            depth_image=depth,
        )
    )

    np.testing.assert_allclose(
        points[0],
        np.asarray(
            [
                0.10,
                0.20,
                0.55,
            ]
        ),
        atol=1e-12,
    )


def test_stride_reduces_point_cloud_density():
    intrinsics = (
        PixelCameraIntrinsics(
            width=4,
            height=4,
            fx=100.0,
            fy=100.0,
            cx=1.5,
            cy=1.5,
        )
    )

    depth = np.full(
        (
            4,
            4,
        ),
        0.20,
    )

    dense = (
        depth_image_to_camera_points(
            intrinsics=intrinsics,
            depth_image=depth,
            stride=1,
        )
    )

    sparse = (
        depth_image_to_camera_points(
            intrinsics=intrinsics,
            depth_image=depth,
            stride=2,
        )
    )

    assert dense.shape[0] == 16
    assert sparse.shape[0] == 4


def test_localisation_error_is_euclidean_distance():
    error = localisation_error(
        np.asarray(
            [
                1.0,
                2.0,
                3.0,
            ]
        ),
        np.asarray(
            [
                1.0,
                2.0,
                5.0,
            ]
        ),
    )

    assert error == pytest.approx(
        2.0
    )


def test_segmented_stereo_images_reconstruct_3d_target():
    """Full image -> segmentation -> stereo -> 3-D reconstruction chain."""

    intrinsics = (
        make_intrinsics()
    )

    target_world = np.asarray(
        [
            0.020,
            -0.010,
            0.250,
        ],
        dtype=float,
    )

    left_projection = (
        project_world_point(
            intrinsics,
            left_pose(),
            target_world,
        )
    )

    right_projection = (
        project_world_point(
            intrinsics,
            right_pose(),
            target_world,
        )
    )

    left_image = (
        render_synthetic_marker_frame(
            width=intrinsics.width,
            height=intrinsics.height,
            marker_pixel=(
                left_projection.pixel
            ),
            marker_radius=14,
            seed=31,
        )
    )

    right_image = (
        render_synthetic_marker_frame(
            width=intrinsics.width,
            height=intrinsics.height,
            marker_pixel=(
                right_projection.pixel
            ),
            marker_radius=14,
            seed=32,
        )
    )

    left_detection = (
        segment_target_bgr(
            left_image
        )
        .detection
    )

    right_detection = (
        segment_target_bgr(
            right_image
        )
        .detection
    )

    assert left_detection.found
    assert right_detection.found

    result = (
        triangulate_world_point_from_pixels(
            intrinsics=intrinsics,
            left_pose=left_pose(),
            right_pose=right_pose(),
            left_pixel=(
                left_detection.centroid
            ),
            right_pixel=(
                right_detection.centroid
            ),
        )
    )

    error = localisation_error(
        result.point_world,
        target_world,
    )

    assert error < 0.002