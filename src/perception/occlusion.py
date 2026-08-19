"""Geometric occlusion modelling for simulated surgical perception."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry.workspace import SphericalStructure
from src.perception.camera import CameraPose


@dataclass(frozen=True)
class OcclusionResult:
    """Result of testing whether a target is occluded."""

    occluded: bool
    occluding_structure_index: int | None
    nearest_occluder_distance: float


def ray_sphere_intersection_distance(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    sphere_centre: np.ndarray,
    sphere_radius: float,
) -> float | None:
    """Return first positive ray-sphere intersection distance.

    Parameters
    ----------
    ray_origin:
        Origin of the viewing ray.

    ray_direction:
        Unit vector defining the ray direction.

    sphere_centre:
        Centre of the spherical occluding structure.

    sphere_radius:
        Physical radius of the sphere.

    Returns
    -------
    float | None
        Distance from the ray origin to the first positive intersection.
        None is returned when the ray does not intersect the sphere.
    """

    ray_origin = np.asarray(
        ray_origin,
        dtype=float,
    )

    ray_direction = np.asarray(
        ray_direction,
        dtype=float,
    )

    sphere_centre = np.asarray(
        sphere_centre,
        dtype=float,
    )

    if ray_origin.shape != (3,):
        raise ValueError(
            "ray_origin must have shape (3,)."
        )

    if ray_direction.shape != (3,):
        raise ValueError(
            "ray_direction must have shape (3,)."
        )

    if sphere_centre.shape != (3,):
        raise ValueError(
            "sphere_centre must have shape (3,)."
        )

    if sphere_radius <= 0.0:
        raise ValueError(
            "sphere_radius must be positive."
        )

    direction_norm = float(
        np.linalg.norm(ray_direction)
    )

    if direction_norm < 1e-12:
        raise ValueError(
            "ray_direction must be non-zero."
        )

    direction = (
        ray_direction
        / direction_norm
    )

    origin_to_centre = (
        ray_origin
        - sphere_centre
    )

    b = 2.0 * float(
        np.dot(
            direction,
            origin_to_centre,
        )
    )

    c = float(
        np.dot(
            origin_to_centre,
            origin_to_centre,
        )
        - sphere_radius**2
    )

    discriminant = (
        b**2
        - 4.0 * c
    )

    if discriminant < 0.0:
        return None

    sqrt_discriminant = float(
        np.sqrt(
            max(discriminant, 0.0)
        )
    )

    t_near = (
        -b - sqrt_discriminant
    ) / 2.0

    t_far = (
        -b + sqrt_discriminant
    ) / 2.0

    positive_intersections = [
        value
        for value in (
            t_near,
            t_far,
        )
        if value > 1e-10
    ]

    if not positive_intersections:
        return None

    return float(
        min(positive_intersections)
    )


def evaluate_point_occlusion(
    camera_pose: CameraPose,
    target_position: np.ndarray,
    occluders: tuple[
        SphericalStructure,
        ...
    ],
) -> OcclusionResult:
    """Determine whether a target point is hidden by spherical anatomy."""

    target_position = np.asarray(
        target_position,
        dtype=float,
    )

    if target_position.shape != (3,):
        raise ValueError(
            "target_position must have shape (3,)."
        )

    camera_to_target = (
        target_position
        - camera_pose.position
    )

    target_distance = float(
        np.linalg.norm(
            camera_to_target
        )
    )

    if target_distance < 1e-12:
        raise ValueError(
            "target_position must differ "
            "from camera position."
        )

    ray_direction = (
        camera_to_target
        / target_distance
    )

    nearest_distance = float("inf")
    nearest_index: int | None = None

    for index, structure in enumerate(
        occluders
    ):
        intersection_distance = (
            ray_sphere_intersection_distance(
                ray_origin=(
                    camera_pose.position
                ),
                ray_direction=ray_direction,
                sphere_centre=(
                    structure.centre
                ),
                sphere_radius=(
                    structure.physical_radius
                ),
            )
        )

        if intersection_distance is None:
            continue

        if (
            intersection_distance
            < target_distance - 1e-10
            and intersection_distance
            < nearest_distance
        ):
            nearest_distance = (
                intersection_distance
            )

            nearest_index = index

    return OcclusionResult(
        occluded=(
            nearest_index is not None
        ),
        occluding_structure_index=(
            nearest_index
        ),
        nearest_occluder_distance=(
            nearest_distance
        ),
    )


def visible_after_occlusion(
    camera_pose: CameraPose,
    target_position: np.ndarray,
    occluders: tuple[
        SphericalStructure,
        ...
    ],
) -> bool:
    """Return True when no physical structure blocks the target."""

    result = evaluate_point_occlusion(
        camera_pose=camera_pose,
        target_position=target_position,
        occluders=occluders,
    )

    return not result.occluded