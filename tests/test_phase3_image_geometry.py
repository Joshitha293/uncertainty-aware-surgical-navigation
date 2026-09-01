"""Tests for Phase 3 pixel-level camera geometry."""

import numpy as np
import pytest

from src.perception.camera import (
    CameraPose,
    look_at_rotation,
)
from src.perception.image_geometry import (
    PixelCameraIntrinsics,
    backproject_pixel_to_camera_ray,
    backproject_pixel_to_world_ray,
    project_camera_point,
    project_world_point,
    reconstruct_camera_point_from_depth,
    reconstruct_world_point_from_depth,
    reprojection_error,
)


def make_intrinsics() -> PixelCameraIntrinsics:
    return PixelCameraIntrinsics(
        width=640,
        height=480,
        fx=500.0,
        fy=500.0,
        cx=319.5,
        cy=239.5,
    )


def identity_pose() -> CameraPose:
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


def test_intrinsic_matrix_has_expected_form():
    intrinsics = (
        make_intrinsics()
    )

    expected = np.asarray(
        [
            [
                500.0,
                0.0,
                319.5,
            ],
            [
                0.0,
                500.0,
                239.5,
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
        ]
    )

    np.testing.assert_allclose(
        intrinsics.matrix,
        expected,
        atol=1e-12,
    )


def test_intrinsics_can_be_constructed_from_fov():
    intrinsics = (
        PixelCameraIntrinsics
        .from_field_of_view(
            width=640,
            height=480,
            horizontal_fov=np.deg2rad(
                60.0
            ),
            vertical_fov=np.deg2rad(
                45.0
            ),
        )
    )

    assert intrinsics.fx > 0.0
    assert intrinsics.fy > 0.0

    assert (
        intrinsics.cx
        == pytest.approx(
            319.5
        )
    )

    assert (
        intrinsics.cy
        == pytest.approx(
            239.5
        )
    )


def test_optical_axis_projects_to_principal_point():
    projection = (
        project_camera_point(
            make_intrinsics(),
            np.asarray(
                [
                    0.0,
                    0.0,
                    0.20,
                ]
            ),
        )
    )

    np.testing.assert_allclose(
        projection.pixel,
        np.asarray(
            [
                319.5,
                239.5,
            ]
        ),
        atol=1e-12,
    )

    assert projection.depth == pytest.approx(
        0.20
    )

    assert projection.inside_image


def test_positive_camera_x_moves_pixel_right():
    intrinsics = (
        make_intrinsics()
    )

    centre = project_camera_point(
        intrinsics,
        np.asarray(
            [
                0.0,
                0.0,
                0.20,
            ]
        ),
    )

    right = project_camera_point(
        intrinsics,
        np.asarray(
            [
                0.02,
                0.0,
                0.20,
            ]
        ),
    )

    assert (
        right.pixel[0]
        > centre.pixel[0]
    )


def test_positive_camera_y_moves_pixel_up():
    intrinsics = (
        make_intrinsics()
    )

    centre = project_camera_point(
        intrinsics,
        np.asarray(
            [
                0.0,
                0.0,
                0.20,
            ]
        ),
    )

    upper = project_camera_point(
        intrinsics,
        np.asarray(
            [
                0.0,
                0.02,
                0.20,
            ]
        ),
    )

    assert (
        upper.pixel[1]
        < centre.pixel[1]
    )


def test_point_behind_camera_is_rejected():
    with pytest.raises(
        ValueError,
        match="front",
    ):
        project_camera_point(
            make_intrinsics(),
            np.asarray(
                [
                    0.0,
                    0.0,
                    -0.10,
                ]
            ),
        )


def test_principal_pixel_backprojects_to_optical_axis():
    ray = (
        backproject_pixel_to_camera_ray(
            make_intrinsics(),
            np.asarray(
                [
                    319.5,
                    239.5,
                ]
            ),
        )
    )

    np.testing.assert_allclose(
        ray,
        np.asarray(
            [
                0.0,
                0.0,
                1.0,
            ]
        ),
        atol=1e-12,
    )


def test_projection_and_depth_reconstruction_round_trip():
    intrinsics = (
        make_intrinsics()
    )

    original = np.asarray(
        [
            0.025,
            -0.015,
            0.22,
        ],
        dtype=float,
    )

    projection = (
        project_camera_point(
            intrinsics,
            original,
        )
    )

    reconstructed = (
        reconstruct_camera_point_from_depth(
            intrinsics,
            projection.pixel,
            projection.depth,
        )
    )

    np.testing.assert_allclose(
        reconstructed,
        original,
        atol=1e-12,
    )


def test_world_projection_and_reconstruction_round_trip():
    intrinsics = (
        make_intrinsics()
    )

    camera_position = np.asarray(
        [
            0.10,
            -0.04,
            0.06,
        ],
        dtype=float,
    )

    target = np.asarray(
        [
            0.22,
            0.02,
            0.08,
        ],
        dtype=float,
    )

    pose = CameraPose(
        position=camera_position,
        rotation=look_at_rotation(
            camera_position,
            target,
        ),
    )

    projection = (
        project_world_point(
            intrinsics,
            pose,
            target,
        )
    )

    reconstructed = (
        reconstruct_world_point_from_depth(
            intrinsics,
            pose,
            projection.pixel,
            projection.depth,
        )
    )

    np.testing.assert_allclose(
        reconstructed,
        target,
        atol=1e-12,
    )


def test_world_ray_points_toward_projected_target():
    intrinsics = (
        make_intrinsics()
    )

    camera_position = np.asarray(
        [
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )

    target = np.asarray(
        [
            0.10,
            0.03,
            0.30,
        ],
        dtype=float,
    )

    pose = identity_pose()

    projection = (
        project_world_point(
            intrinsics,
            pose,
            target,
        )
    )

    ray = (
        backproject_pixel_to_world_ray(
            intrinsics,
            pose,
            projection.pixel,
        )
    )

    expected_direction = (
        target
        - camera_position
    )

    expected_direction = (
        expected_direction
        / np.linalg.norm(
            expected_direction
        )
    )

    np.testing.assert_allclose(
        ray.direction,
        expected_direction,
        atol=1e-12,
    )


def test_exact_projection_has_zero_reprojection_error():
    intrinsics = (
        make_intrinsics()
    )

    pose = identity_pose()

    point = np.asarray(
        [
            0.04,
            -0.02,
            0.25,
        ]
    )

    pixel = (
        project_world_point(
            intrinsics,
            pose,
            point,
        )
        .pixel
    )

    assert (
        reprojection_error(
            intrinsics,
            pose,
            point,
            pixel,
        )
        <= 1e-12
    )


def test_outside_pixel_is_reported():
    projection = (
        project_camera_point(
            make_intrinsics(),
            np.asarray(
                [
                    1.0,
                    0.0,
                    0.10,
                ],
                dtype=float,
            ),
        )
    )

    assert not (
        projection.inside_image
    )