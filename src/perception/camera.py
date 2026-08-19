"""Geometric camera model for simulated surgical perception."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Simplified pinhole-camera intrinsic parameters."""

    horizontal_fov: float
    vertical_fov: float
    near_distance: float
    far_distance: float

    def __post_init__(self) -> None:
        if not 0.0 < self.horizontal_fov < np.pi:
            raise ValueError(
                "horizontal_fov must lie between 0 and pi radians."
            )

        if not 0.0 < self.vertical_fov < np.pi:
            raise ValueError(
                "vertical_fov must lie between 0 and pi radians."
            )

        if self.near_distance <= 0.0:
            raise ValueError(
                "near_distance must be positive."
            )

        if self.far_distance <= self.near_distance:
            raise ValueError(
                "far_distance must exceed near_distance."
            )


@dataclass(frozen=True)
class CameraPose:
    """Position and orientation of a simulated surgical camera.

    Rotation maps vectors from camera coordinates into world coordinates.

    Camera convention:
        +x : image right
        +y : image up
        +z : optical axis / forward direction
    """

    position: np.ndarray
    rotation: np.ndarray

    def __post_init__(self) -> None:
        position = np.asarray(
            self.position,
            dtype=float,
        )

        rotation = np.asarray(
            self.rotation,
            dtype=float,
        )

        if position.shape != (3,):
            raise ValueError(
                "position must have shape (3,)."
            )

        if rotation.shape != (3, 3):
            raise ValueError(
                "rotation must have shape (3, 3)."
            )

        if not np.allclose(
            rotation.T @ rotation,
            np.eye(3),
            atol=1e-8,
        ):
            raise ValueError(
                "rotation must be orthonormal."
            )

        if not np.isclose(
            np.linalg.det(rotation),
            1.0,
            atol=1e-8,
        ):
            raise ValueError(
                "rotation must be a proper rotation matrix."
            )

        object.__setattr__(
            self,
            "position",
            position,
        )

        object.__setattr__(
            self,
            "rotation",
            rotation,
        )

    @property
    def forward(self) -> np.ndarray:
        """Return camera optical-axis direction in world coordinates."""
        return self.rotation[:, 2].copy()

    @property
    def right(self) -> np.ndarray:
        """Return camera right direction in world coordinates."""
        return self.rotation[:, 0].copy()

    @property
    def up(self) -> np.ndarray:
        """Return camera up direction in world coordinates."""
        return self.rotation[:, 1].copy()

    def world_to_camera(
        self,
        point_world: np.ndarray,
    ) -> np.ndarray:
        """Transform a world-frame point into camera coordinates."""
        point_world = np.asarray(
            point_world,
            dtype=float,
        )

        if point_world.shape != (3,):
            raise ValueError(
                "point_world must have shape (3,)."
            )

        return (
            self.rotation.T
            @ (point_world - self.position)
        )

    def camera_to_world(
        self,
        point_camera: np.ndarray,
    ) -> np.ndarray:
        """Transform a camera-frame point into world coordinates."""
        point_camera = np.asarray(
            point_camera,
            dtype=float,
        )

        if point_camera.shape != (3,):
            raise ValueError(
                "point_camera must have shape (3,)."
            )

        return (
            self.position
            + self.rotation @ point_camera
        )


@dataclass(frozen=True)
class VisibilityResult:
    """Geometric visibility information for one world point."""

    visible: bool
    distance: float
    horizontal_angle: float
    vertical_angle: float
    camera_point: np.ndarray


class SurgicalCamera:
    """Simplified geometric surgical camera."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
    ) -> None:
        self.intrinsics = intrinsics

    def evaluate_visibility(
        self,
        pose: CameraPose,
        point_world: np.ndarray,
    ) -> VisibilityResult:
        """Evaluate whether a world point lies inside the camera frustum."""

        point_camera = pose.world_to_camera(
            point_world
        )

        x = float(point_camera[0])
        y = float(point_camera[1])
        z = float(point_camera[2])

        distance = float(
            np.linalg.norm(point_camera)
        )

        if z <= 0.0:
            return VisibilityResult(
                visible=False,
                distance=distance,
                horizontal_angle=float("nan"),
                vertical_angle=float("nan"),
                camera_point=point_camera,
            )

        horizontal_angle = float(
            np.arctan2(x, z)
        )

        vertical_angle = float(
            np.arctan2(y, z)
        )

        inside_horizontal_fov = (
            abs(horizontal_angle)
            <= 0.5
            * self.intrinsics.horizontal_fov
        )

        inside_vertical_fov = (
            abs(vertical_angle)
            <= 0.5
            * self.intrinsics.vertical_fov
        )

        inside_depth_range = (
            self.intrinsics.near_distance
            <= distance
            <= self.intrinsics.far_distance
        )

        visible = bool(
            inside_horizontal_fov
            and inside_vertical_fov
            and inside_depth_range
        )

        return VisibilityResult(
            visible=visible,
            distance=distance,
            horizontal_angle=horizontal_angle,
            vertical_angle=vertical_angle,
            camera_point=point_camera,
        )


def look_at_rotation(
    camera_position: np.ndarray,
    target_position: np.ndarray,
    world_up: np.ndarray | None = None,
) -> np.ndarray:
    """Construct a camera-to-world rotation looking toward a target."""

    camera_position = np.asarray(
        camera_position,
        dtype=float,
    )

    target_position = np.asarray(
        target_position,
        dtype=float,
    )

    if (
        camera_position.shape != (3,)
        or target_position.shape != (3,)
    ):
        raise ValueError(
            "camera_position and target_position "
            "must each have shape (3,)."
        )

    if world_up is None:
        world_up = np.array(
            [0.0, 0.0, 1.0],
            dtype=float,
        )
    else:
        world_up = np.asarray(
            world_up,
            dtype=float,
        )

    if world_up.shape != (3,):
        raise ValueError(
            "world_up must have shape (3,)."
        )

    forward = (
        target_position
        - camera_position
    )

    forward_norm = np.linalg.norm(
        forward
    )

    if forward_norm < 1e-12:
        raise ValueError(
            "camera_position and target_position "
            "must be different."
        )

    forward = forward / forward_norm

    up_norm = np.linalg.norm(
        world_up
    )

    if up_norm < 1e-12:
        raise ValueError(
            "world_up must be non-zero."
        )

    world_up = world_up / up_norm

    right = np.cross(
        world_up,
        forward,
    )

    right_norm = np.linalg.norm(
        right
    )

    if right_norm < 1e-10:
        alternative_up = np.array(
            [0.0, 1.0, 0.0],
            dtype=float,
        )

        right = np.cross(
            alternative_up,
            forward,
        )

        right_norm = np.linalg.norm(
            right
        )

    right = right / right_norm

    up = np.cross(
        forward,
        right,
    )

    up = up / np.linalg.norm(
        up
    )

    return np.column_stack(
        (
            right,
            up,
            forward,
        )
    )