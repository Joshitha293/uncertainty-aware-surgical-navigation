"""Pixel-level pinhole camera geometry for Phase 3 surgical vision.

The established Phase 1 camera model represents field of view and camera
pose geometrically but does not represent image pixels.

This module adds the pixel-domain camera model required for genuine
image-based perception:

- image resolution;
- focal lengths and principal point;
- conversion from field of view to pixel intrinsics;
- camera-point projection to pixels;
- world-point projection to pixels;
- pixel back-projection to camera rays;
- pixel back-projection to world rays;
- depth-assisted 3-D reconstruction.

Camera convention is inherited from CameraPose:

    +x = image right
    +y = image up
    +z = optical axis / forward

Image convention:

    u increases right
    v increases downward

The sign change between camera +y and image v is handled explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.perception.camera import CameraPose


@dataclass(frozen=True)
class PixelCameraIntrinsics:
    """Pinhole camera intrinsics expressed in pixel units."""

    width: int
    height: int

    fx: float
    fy: float

    cx: float
    cy: float

    def __post_init__(self) -> None:
        if self.width < 2:
            raise ValueError(
                "width must be at least 2 pixels."
            )

        if self.height < 2:
            raise ValueError(
                "height must be at least 2 pixels."
            )

        for name, value in (
            ("fx", self.fx),
            ("fy", self.fy),
            ("cx", self.cx),
            ("cy", self.cy),
        ):
            if not np.isfinite(
                value
            ):
                raise ValueError(
                    f"{name} must be finite."
                )

        if self.fx <= 0.0:
            raise ValueError(
                "fx must be positive."
            )

        if self.fy <= 0.0:
            raise ValueError(
                "fy must be positive."
            )

    @classmethod
    def from_field_of_view(
        cls,
        *,
        width: int,
        height: int,
        horizontal_fov: float,
        vertical_fov: float,
    ) -> "PixelCameraIntrinsics":
        """Construct pixel intrinsics from image size and angular FOV."""

        if not (
            0.0
            < horizontal_fov
            < np.pi
        ):
            raise ValueError(
                "horizontal_fov must lie between 0 and pi."
            )

        if not (
            0.0
            < vertical_fov
            < np.pi
        ):
            raise ValueError(
                "vertical_fov must lie between 0 and pi."
            )

        fx = (
            0.5
            * width
            / np.tan(
                0.5
                * horizontal_fov
            )
        )

        fy = (
            0.5
            * height
            / np.tan(
                0.5
                * vertical_fov
            )
        )

        cx = (
            width
            - 1
        ) / 2.0

        cy = (
            height
            - 1
        ) / 2.0

        return cls(
            width=width,
            height=height,
            fx=float(
                fx
            ),
            fy=float(
                fy
            ),
            cx=float(
                cx
            ),
            cy=float(
                cy
            ),
        )

    @property
    def matrix(self) -> np.ndarray:
        """Return conventional 3 x 3 intrinsic matrix.

        The matrix is suitable for the conventional image-coordinate model:

            u = fx * x / z + cx
            v = cy - fy * y / z

        The minus sign for y is applied explicitly during projection because
        this project's camera frame defines +y as image-up.
        """

        return np.asarray(
            [
                [
                    self.fx,
                    0.0,
                    self.cx,
                ],
                [
                    0.0,
                    self.fy,
                    self.cy,
                ],
                [
                    0.0,
                    0.0,
                    1.0,
                ],
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class PixelProjection:
    """Projection result for one 3-D point."""

    pixel: np.ndarray

    depth: float

    inside_image: bool


@dataclass(frozen=True)
class WorldRay:
    """World-frame ray originating from a camera centre."""

    origin: np.ndarray

    direction: np.ndarray


def project_camera_point(
    intrinsics: PixelCameraIntrinsics,
    point_camera: np.ndarray,
) -> PixelProjection:
    """Project a camera-frame point to image coordinates."""

    point = np.asarray(
        point_camera,
        dtype=float,
    )

    if point.shape != (3,):
        raise ValueError(
            "point_camera must have shape (3,)."
        )

    if not np.all(
        np.isfinite(
            point
        )
    ):
        raise ValueError(
            "point_camera must contain finite values."
        )

    x, y, z = (
        float(
            point[0]
        ),
        float(
            point[1]
        ),
        float(
            point[2]
        ),
    )

    if z <= 0.0:
        raise ValueError(
            "point_camera must lie in front of the camera."
        )

    u = (
        intrinsics.fx
        * x
        / z
        + intrinsics.cx
    )

    # Camera +y is image-up, whereas image row v increases downward.
    v = (
        intrinsics.cy
        - intrinsics.fy
        * y
        / z
    )

    inside = bool(
        0.0
        <= u
        <= intrinsics.width
        - 1
        and 0.0
        <= v
        <= intrinsics.height
        - 1
    )

    return PixelProjection(
        pixel=np.asarray(
            [
                u,
                v,
            ],
            dtype=float,
        ),
        depth=z,
        inside_image=inside,
    )


def project_world_point(
    intrinsics: PixelCameraIntrinsics,
    pose: CameraPose,
    point_world: np.ndarray,
) -> PixelProjection:
    """Project one world-frame point into the image."""

    point_world = np.asarray(
        point_world,
        dtype=float,
    )

    if point_world.shape != (3,):
        raise ValueError(
            "point_world must have shape (3,)."
        )

    point_camera = (
        pose.world_to_camera(
            point_world
        )
    )

    return project_camera_point(
        intrinsics,
        point_camera,
    )


def backproject_pixel_to_camera_ray(
    intrinsics: PixelCameraIntrinsics,
    pixel: np.ndarray,
) -> np.ndarray:
    """Return a unit camera-frame ray passing through one image pixel."""

    pixel = np.asarray(
        pixel,
        dtype=float,
    )

    if pixel.shape != (2,):
        raise ValueError(
            "pixel must have shape (2,)."
        )

    if not np.all(
        np.isfinite(
            pixel
        )
    ):
        raise ValueError(
            "pixel must contain finite values."
        )

    u = float(
        pixel[0]
    )

    v = float(
        pixel[1]
    )

    x = (
        u
        - intrinsics.cx
    ) / intrinsics.fx

    y = -(
        v
        - intrinsics.cy
    ) / intrinsics.fy

    ray = np.asarray(
        [
            x,
            y,
            1.0,
        ],
        dtype=float,
    )

    norm = float(
        np.linalg.norm(
            ray
        )
    )

    return (
        ray
        / norm
    )


def backproject_pixel_to_world_ray(
    intrinsics: PixelCameraIntrinsics,
    pose: CameraPose,
    pixel: np.ndarray,
) -> WorldRay:
    """Return a world-frame ray passing through an image pixel."""

    camera_ray = (
        backproject_pixel_to_camera_ray(
            intrinsics,
            pixel,
        )
    )

    world_direction = (
        pose.rotation
        @ camera_ray
    )

    world_direction = (
        world_direction
        / np.linalg.norm(
            world_direction
        )
    )

    return WorldRay(
        origin=np.array(
            pose.position,
            dtype=float,
            copy=True,
        ),
        direction=np.asarray(
            world_direction,
            dtype=float,
        ),
    )


def reconstruct_camera_point_from_depth(
    intrinsics: PixelCameraIntrinsics,
    pixel: np.ndarray,
    depth: float,
) -> np.ndarray:
    """Reconstruct a camera-frame point using optical-axis depth.

    ``depth`` represents z distance along the optical axis, not Euclidean
    range along the ray.
    """

    if (
        not np.isfinite(
            depth
        )
        or depth <= 0.0
    ):
        raise ValueError(
            "depth must be finite and positive."
        )

    pixel = np.asarray(
        pixel,
        dtype=float,
    )

    if pixel.shape != (2,):
        raise ValueError(
            "pixel must have shape (2,)."
        )

    u = float(
        pixel[0]
    )

    v = float(
        pixel[1]
    )

    x = (
        u
        - intrinsics.cx
    ) * depth / intrinsics.fx

    y = -(
        v
        - intrinsics.cy
    ) * depth / intrinsics.fy

    return np.asarray(
        [
            x,
            y,
            depth,
        ],
        dtype=float,
    )


def reconstruct_world_point_from_depth(
    intrinsics: PixelCameraIntrinsics,
    pose: CameraPose,
    pixel: np.ndarray,
    depth: float,
) -> np.ndarray:
    """Reconstruct a world-frame point from pixel and optical-axis depth."""

    point_camera = (
        reconstruct_camera_point_from_depth(
            intrinsics,
            pixel,
            depth,
        )
    )

    return pose.camera_to_world(
        point_camera
    )


def reprojection_error(
    intrinsics: PixelCameraIntrinsics,
    pose: CameraPose,
    point_world: np.ndarray,
    observed_pixel: np.ndarray,
) -> float:
    """Return Euclidean image reprojection error in pixels."""

    observed_pixel = np.asarray(
        observed_pixel,
        dtype=float,
    )

    if observed_pixel.shape != (2,):
        raise ValueError(
            "observed_pixel must have shape (2,)."
        )

    predicted = project_world_point(
        intrinsics,
        pose,
        point_world,
    )

    return float(
        np.linalg.norm(
            predicted.pixel
            - observed_pixel
        )
    )