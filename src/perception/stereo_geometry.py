"""Stereo reconstruction and point-cloud geometry for Phase 3.

This module extends the monocular pixel camera model with:

- rectified stereo disparity geometry;
- general two-view ray triangulation;
- reprojection diagnostics;
- closest-ray consistency error;
- depth-image to 3-D point-cloud reconstruction;
- world-frame localisation error metrics.

The implementation uses the project's established camera convention:

    +x = image right
    +y = image up
    +z = optical axis

where image row v increases downward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.perception.camera import CameraPose
from src.perception.image_geometry import (
    PixelCameraIntrinsics,
    backproject_pixel_to_world_ray,
    project_world_point,
    reconstruct_camera_point_from_depth,
)


@dataclass(frozen=True)
class StereoTriangulationResult:
    """Result from two-view 3-D triangulation."""

    point_world: np.ndarray

    left_ray_distance: float
    right_ray_distance: float

    closest_ray_gap: float

    left_reprojection_error: float
    right_reprojection_error: float


def rectified_stereo_depth(
    *,
    focal_length_pixels: float,
    baseline: float,
    disparity: float,
) -> float:
    """Recover optical-axis depth from rectified stereo disparity.

    For parallel cameras separated along camera +x:

        depth = focal_length * baseline / disparity
    """

    for name, value in (
        (
            "focal_length_pixels",
            focal_length_pixels,
        ),
        (
            "baseline",
            baseline,
        ),
        (
            "disparity",
            disparity,
        ),
    ):
        if not np.isfinite(
            value
        ):
            raise ValueError(
                f"{name} must be finite."
            )

    if focal_length_pixels <= 0.0:
        raise ValueError(
            "focal_length_pixels must be positive."
        )

    if baseline <= 0.0:
        raise ValueError(
            "baseline must be positive."
        )

    if disparity <= 0.0:
        raise ValueError(
            "disparity must be positive."
        )

    return float(
        focal_length_pixels
        * baseline
        / disparity
    )


def reconstruct_rectified_stereo_point(
    *,
    intrinsics: PixelCameraIntrinsics,
    left_pixel: np.ndarray,
    right_pixel: np.ndarray,
    baseline: float,
) -> np.ndarray:
    """Reconstruct a point in the left-camera frame from rectified stereo."""

    left_pixel = np.asarray(
        left_pixel,
        dtype=float,
    )

    right_pixel = np.asarray(
        right_pixel,
        dtype=float,
    )

    if (
        left_pixel.shape != (2,)
        or right_pixel.shape != (2,)
    ):
        raise ValueError(
            "left_pixel and right_pixel must each have shape (2,)."
        )

    disparity = float(
        left_pixel[0]
        - right_pixel[0]
    )

    depth = rectified_stereo_depth(
        focal_length_pixels=(
            intrinsics.fx
        ),
        baseline=baseline,
        disparity=disparity,
    )

    return (
        reconstruct_camera_point_from_depth(
            intrinsics,
            left_pixel,
            depth,
        )
    )


def triangulate_world_point_from_pixels(
    *,
    intrinsics: PixelCameraIntrinsics,
    left_pose: CameraPose,
    right_pose: CameraPose,
    left_pixel: np.ndarray,
    right_pixel: np.ndarray,
    parallel_tolerance: float = 1e-8,
) -> StereoTriangulationResult:
    """Triangulate a world point from two calibrated camera rays.

    For noisy observations, the rays generally do not intersect exactly.
    The reconstructed point is therefore the midpoint of the shortest
    segment joining the two rays.

    ``closest_ray_gap`` quantifies stereo geometric consistency.
    """

    if (
        parallel_tolerance <= 0.0
        or not np.isfinite(
            parallel_tolerance
        )
    ):
        raise ValueError(
            "parallel_tolerance must be finite and positive."
        )

    left_ray = (
        backproject_pixel_to_world_ray(
            intrinsics,
            left_pose,
            left_pixel,
        )
    )

    right_ray = (
        backproject_pixel_to_world_ray(
            intrinsics,
            right_pose,
            right_pixel,
        )
    )

    cross_product = np.cross(
        left_ray.direction,
        right_ray.direction,
    )

    if (
        np.linalg.norm(
            cross_product
        )
        < parallel_tolerance
    ):
        raise ValueError(
            "Stereo rays are parallel or nearly parallel."
        )

    # Solve:
    #
    # left_origin + t_left * left_direction
    #     ~= right_origin + t_right * right_direction
    #
    # [d_left, -d_right] [t_left, t_right]^T
    #     = right_origin - left_origin

    matrix = np.column_stack(
        (
            left_ray.direction,
            -right_ray.direction,
        )
    )

    right_hand_side = (
        right_ray.origin
        - left_ray.origin
    )

    distances, _residuals, _rank, _singular_values = (
        np.linalg.lstsq(
            matrix,
            right_hand_side,
            rcond=None,
        )
    )

    left_distance = float(
        distances[0]
    )

    right_distance = float(
        distances[1]
    )

    if (
        left_distance <= 0.0
        or right_distance <= 0.0
    ):
        raise ValueError(
            "Triangulated point lies behind at least one camera."
        )

    closest_left = (
        left_ray.origin
        + left_distance
        * left_ray.direction
    )

    closest_right = (
        right_ray.origin
        + right_distance
        * right_ray.direction
    )

    point_world = (
        0.5
        * (
            closest_left
            + closest_right
        )
    )

    gap = float(
        np.linalg.norm(
            closest_left
            - closest_right
        )
    )

    left_projection = (
        project_world_point(
            intrinsics,
            left_pose,
            point_world,
        )
    )

    right_projection = (
        project_world_point(
            intrinsics,
            right_pose,
            point_world,
        )
    )

    left_pixel_array = np.asarray(
        left_pixel,
        dtype=float,
    )

    right_pixel_array = np.asarray(
        right_pixel,
        dtype=float,
    )

    left_error = float(
        np.linalg.norm(
            left_projection.pixel
            - left_pixel_array
        )
    )

    right_error = float(
        np.linalg.norm(
            right_projection.pixel
            - right_pixel_array
        )
    )

    return StereoTriangulationResult(
        point_world=np.asarray(
            point_world,
            dtype=float,
        ),
        left_ray_distance=(
            left_distance
        ),
        right_ray_distance=(
            right_distance
        ),
        closest_ray_gap=gap,
        left_reprojection_error=(
            left_error
        ),
        right_reprojection_error=(
            right_error
        ),
    )


def depth_image_to_camera_points(
    *,
    intrinsics: PixelCameraIntrinsics,
    depth_image: np.ndarray,
    stride: int = 1,
) -> np.ndarray:
    """Convert valid optical-axis depth pixels into camera-frame points.

    Invalid, non-finite and non-positive depth values are omitted.
    """

    depth = np.asarray(
        depth_image,
        dtype=float,
    )

    if depth.ndim != 2:
        raise ValueError(
            "depth_image must be two-dimensional."
        )

    if (
        depth.shape[0]
        != intrinsics.height
        or depth.shape[1]
        != intrinsics.width
    ):
        raise ValueError(
            "depth_image dimensions must match camera intrinsics."
        )

    if stride < 1:
        raise ValueError(
            "stride must be at least 1."
        )

    rows = np.arange(
        0,
        intrinsics.height,
        stride,
    )

    columns = np.arange(
        0,
        intrinsics.width,
        stride,
    )

    u_grid, v_grid = np.meshgrid(
        columns,
        rows,
    )

    sampled_depth = depth[
        v_grid,
        u_grid,
    ]

    valid = (
        np.isfinite(
            sampled_depth
        )
        & (
            sampled_depth
            > 0.0
        )
    )

    if not np.any(
        valid
    ):
        return np.empty(
            (
                0,
                3,
            ),
            dtype=float,
        )

    z = sampled_depth[
        valid
    ]

    u = u_grid[
        valid
    ].astype(
        float
    )

    v = v_grid[
        valid
    ].astype(
        float
    )

    x = (
        u
        - intrinsics.cx
    ) * z / intrinsics.fx

    y = -(
        v
        - intrinsics.cy
    ) * z / intrinsics.fy

    return np.column_stack(
        (
            x,
            y,
            z,
        )
    ).astype(
        float
    )


def camera_points_to_world(
    *,
    pose: CameraPose,
    points_camera: np.ndarray,
) -> np.ndarray:
    """Transform an N x 3 camera point cloud into world coordinates."""

    points = np.asarray(
        points_camera,
        dtype=float,
    )

    if (
        points.ndim != 2
        or points.shape[1] != 3
    ):
        raise ValueError(
            "points_camera must have shape (N, 3)."
        )

    if points.shape[0] == 0:
        return np.empty(
            (
                0,
                3,
            ),
            dtype=float,
        )

    if not np.all(
        np.isfinite(
            points
        )
    ):
        raise ValueError(
            "points_camera must contain finite values."
        )

    return (
        (
            pose.rotation
            @ points.T
        ).T
        + pose.position[
            None,
            :
        ]
    )


def depth_image_to_world_points(
    *,
    intrinsics: PixelCameraIntrinsics,
    pose: CameraPose,
    depth_image: np.ndarray,
    stride: int = 1,
) -> np.ndarray:
    """Reconstruct a world-frame point cloud from a depth image."""

    camera_points = (
        depth_image_to_camera_points(
            intrinsics=intrinsics,
            depth_image=depth_image,
            stride=stride,
        )
    )

    return camera_points_to_world(
        pose=pose,
        points_camera=camera_points,
    )


def localisation_error(
    estimated_point: np.ndarray,
    reference_point: np.ndarray,
) -> float:
    """Return Euclidean 3-D localisation error."""

    estimated = np.asarray(
        estimated_point,
        dtype=float,
    )

    reference = np.asarray(
        reference_point,
        dtype=float,
    )

    if (
        estimated.shape != (3,)
        or reference.shape != (3,)
    ):
        raise ValueError(
            "Points must each have shape (3,)."
        )

    return float(
        np.linalg.norm(
            estimated
            - reference
        )
    )