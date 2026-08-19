"""Candidate camera viewpoint generation for active surgical perception."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.perception.camera import (
    CameraPose,
    look_at_rotation,
)


@dataclass(frozen=True)
class ViewpointSamplingConfig:
    """Configuration for candidate camera viewpoint generation.

    Candidate camera positions are sampled on spherical shells around a
    target point. Each candidate is oriented toward the target using the
    geometric look-at camera model.

    Parameters
    ----------
    radii:
        Camera-to-target distances in metres.

    azimuth_count:
        Number of samples around each horizontal ring.

    elevation_angles:
        Elevation angles above or below the target plane in radians.

    minimum_height:
        Optional lower world-z bound for candidate camera positions.

    maximum_height:
        Optional upper world-z bound for candidate camera positions.
    """

    radii: tuple[float, ...] = (
        0.15,
        0.20,
        0.25,
    )

    azimuth_count: int = 12

    elevation_angles: tuple[float, ...] = (
        np.deg2rad(-20.0),
        0.0,
        np.deg2rad(20.0),
    )

    minimum_height: float | None = None
    maximum_height: float | None = None

    def __post_init__(self) -> None:
        if len(self.radii) == 0:
            raise ValueError(
                "At least one sampling radius is required."
            )

        if any(
            radius <= 0.0
            for radius in self.radii
        ):
            raise ValueError(
                "All sampling radii must be positive."
            )

        if self.azimuth_count < 1:
            raise ValueError(
                "azimuth_count must be at least 1."
            )

        if len(self.elevation_angles) == 0:
            raise ValueError(
                "At least one elevation angle is required."
            )

        for elevation in self.elevation_angles:
            if not (
                -0.5 * np.pi
                < elevation
                < 0.5 * np.pi
            ):
                raise ValueError(
                    "Elevation angles must lie strictly "
                    "between -pi/2 and pi/2."
                )

        if (
            self.minimum_height is not None
            and self.maximum_height is not None
            and self.minimum_height
            > self.maximum_height
        ):
            raise ValueError(
                "minimum_height must not exceed "
                "maximum_height."
            )


@dataclass(frozen=True)
class CandidateViewpoint:
    """One candidate camera pose considered by active perception."""

    pose: CameraPose
    radius: float
    azimuth: float
    elevation: float


def spherical_camera_position(
    target_position: np.ndarray,
    radius: float,
    azimuth: float,
    elevation: float,
) -> np.ndarray:
    """Return a camera position on a sphere around a target."""

    target_position = np.asarray(
        target_position,
        dtype=float,
    )

    if target_position.shape != (3,):
        raise ValueError(
            "target_position must have shape (3,)."
        )

    if radius <= 0.0:
        raise ValueError(
            "radius must be positive."
        )

    horizontal_radius = (
        radius
        * np.cos(elevation)
    )

    offset = np.array(
        [
            horizontal_radius
            * np.cos(azimuth),
            horizontal_radius
            * np.sin(azimuth),
            radius
            * np.sin(elevation),
        ],
        dtype=float,
    )

    return (
        target_position
        + offset
    )


def generate_candidate_viewpoints(
    target_position: np.ndarray,
    config: ViewpointSamplingConfig | None = None,
) -> tuple[CandidateViewpoint, ...]:
    """Generate camera viewpoints surrounding a target location."""

    target_position = np.asarray(
        target_position,
        dtype=float,
    )

    if target_position.shape != (3,):
        raise ValueError(
            "target_position must have shape (3,)."
        )

    if config is None:
        config = ViewpointSamplingConfig()

    azimuths = np.linspace(
        0.0,
        2.0 * np.pi,
        num=config.azimuth_count,
        endpoint=False,
        dtype=float,
    )

    candidates: list[
        CandidateViewpoint
    ] = []

    for radius in config.radii:
        for elevation in (
            config.elevation_angles
        ):
            for azimuth in azimuths:
                camera_position = (
                    spherical_camera_position(
                        target_position=(
                            target_position
                        ),
                        radius=radius,
                        azimuth=float(
                            azimuth
                        ),
                        elevation=float(
                            elevation
                        ),
                    )
                )

                if (
                    config.minimum_height
                    is not None
                    and camera_position[2]
                    < config.minimum_height
                ):
                    continue

                if (
                    config.maximum_height
                    is not None
                    and camera_position[2]
                    > config.maximum_height
                ):
                    continue

                rotation = look_at_rotation(
                    camera_position=(
                        camera_position
                    ),
                    target_position=(
                        target_position
                    ),
                )

                candidates.append(
                    CandidateViewpoint(
                        pose=CameraPose(
                            position=(
                                camera_position
                            ),
                            rotation=rotation,
                        ),
                        radius=float(
                            radius
                        ),
                        azimuth=float(
                            azimuth
                        ),
                        elevation=float(
                            elevation
                        ),
                    )
                )

    return tuple(
        candidates
    )


def viewpoint_displacement(
    current_pose: CameraPose,
    candidate_pose: CameraPose,
) -> float:
    """Return translational camera displacement between two poses."""

    return float(
        np.linalg.norm(
            candidate_pose.position
            - current_pose.position
        )
    )


def nearest_candidate_index(
    current_pose: CameraPose,
    candidates: tuple[
        CandidateViewpoint,
        ...
    ],
) -> int:
    """Return index of candidate nearest to the current camera position."""

    if len(candidates) == 0:
        raise ValueError(
            "candidates must not be empty."
        )

    distances = np.asarray(
        [
            viewpoint_displacement(
                current_pose=current_pose,
                candidate_pose=(
                    candidate.pose
                ),
            )
            for candidate in candidates
        ],
        dtype=float,
    )

    return int(
        np.argmin(
            distances
        )
    )