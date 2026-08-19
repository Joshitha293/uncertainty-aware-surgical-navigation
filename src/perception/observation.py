"""Viewpoint-dependent observation model for simulated surgical perception."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry.workspace import SphericalStructure
from src.perception.camera import (
    CameraPose,
    SurgicalCamera,
)
from src.perception.occlusion import (
    evaluate_point_occlusion,
)
from src.perception.uncertainty import (
    EstimatedStructure,
    PositionUncertainty,
)


@dataclass(frozen=True)
class ObservationModelConfig:
    """Parameters controlling simulated localisation quality."""

    base_sigma: float = 0.002
    reference_distance: float = 0.15
    distance_weight: float = 1.0
    angle_weight: float = 1.0
    invisible_sigma: float = 0.050
    occluded_sigma: float = 0.030

    def __post_init__(self) -> None:
        if self.base_sigma <= 0.0:
            raise ValueError(
                "base_sigma must be positive."
            )

        if self.reference_distance <= 0.0:
            raise ValueError(
                "reference_distance must be positive."
            )

        if self.distance_weight < 0.0:
            raise ValueError(
                "distance_weight must be non-negative."
            )

        if self.angle_weight < 0.0:
            raise ValueError(
                "angle_weight must be non-negative."
            )

        if self.invisible_sigma <= 0.0:
            raise ValueError(
                "invisible_sigma must be positive."
            )

        if self.occluded_sigma <= 0.0:
            raise ValueError(
                "occluded_sigma must be positive."
            )

        if self.invisible_sigma < self.base_sigma:
            raise ValueError(
                "invisible_sigma must not be smaller "
                "than base_sigma."
            )

        if self.occluded_sigma < self.base_sigma:
            raise ValueError(
                "occluded_sigma must not be smaller "
                "than base_sigma."
            )


@dataclass(frozen=True)
class ObservationQuality:
    """Quality metrics associated with one camera observation."""

    visible: bool
    occluded: bool
    distance: float
    off_axis_angle: float
    localisation_sigma: float


@dataclass(frozen=True)
class StructureObservation:
    """Simulated camera observation of one anatomical structure."""

    estimated_structure: EstimatedStructure
    quality: ObservationQuality
    localisation_error: float


class ViewpointObservationModel:
    """Generate viewpoint-dependent anatomical observations.

    Localisation uncertainty depends on camera-to-target distance,
    off-axis viewing geometry, camera-frustum visibility and anatomical
    occlusion.
    """

    def __init__(
        self,
        camera: SurgicalCamera,
        config: ObservationModelConfig | None = None,
    ) -> None:
        self.camera = camera

        if config is None:
            config = ObservationModelConfig()

        self.config = config

    def observation_quality(
        self,
        camera_pose: CameraPose,
        structure: SphericalStructure,
        occluders: tuple[
            SphericalStructure,
            ...
        ] = (),
    ) -> ObservationQuality:
        """Calculate expected localisation quality from one viewpoint."""

        visibility = self.camera.evaluate_visibility(
            pose=camera_pose,
            point_world=structure.centre,
        )

        distance = visibility.distance

        if not visibility.visible:
            return ObservationQuality(
                visible=False,
                occluded=False,
                distance=distance,
                off_axis_angle=float("inf"),
                localisation_sigma=(
                    self.config.invisible_sigma
                ),
            )

        occlusion = evaluate_point_occlusion(
            camera_pose=camera_pose,
            target_position=structure.centre,
            occluders=occluders,
        )

        horizontal_angle = abs(
            visibility.horizontal_angle
        )

        vertical_angle = abs(
            visibility.vertical_angle
        )

        off_axis_angle = float(
            np.hypot(
                horizontal_angle,
                vertical_angle,
            )
        )

        if occlusion.occluded:
            return ObservationQuality(
                visible=True,
                occluded=True,
                distance=distance,
                off_axis_angle=off_axis_angle,
                localisation_sigma=(
                    self.config.occluded_sigma
                ),
            )

        distance_ratio = (
            distance
            / self.config.reference_distance
        )

        distance_penalty = (
            self.config.distance_weight
            * max(
                distance_ratio - 1.0,
                0.0,
            )
        )

        horizontal_half_fov = (
            0.5
            * self.camera.intrinsics.horizontal_fov
        )

        vertical_half_fov = (
            0.5
            * self.camera.intrinsics.vertical_fov
        )

        normalised_horizontal_angle = (
            horizontal_angle
            / horizontal_half_fov
        )

        normalised_vertical_angle = (
            vertical_angle
            / vertical_half_fov
        )

        normalised_off_axis = float(
            np.hypot(
                normalised_horizontal_angle,
                normalised_vertical_angle,
            )
        )

        angle_penalty = (
            self.config.angle_weight
            * normalised_off_axis
        )

        sigma = (
            self.config.base_sigma
            * (
                1.0
                + distance_penalty
                + angle_penalty
            )
        )

        sigma = min(
            sigma,
            self.config.invisible_sigma,
        )

        return ObservationQuality(
            visible=True,
            occluded=False,
            distance=distance,
            off_axis_angle=off_axis_angle,
            localisation_sigma=float(
                sigma
            ),
        )

    def observe_structure(
        self,
        camera_pose: CameraPose,
        structure: SphericalStructure,
        rng: np.random.Generator,
        occluders: tuple[
            SphericalStructure,
            ...
        ] = (),
    ) -> StructureObservation:
        """Generate a noisy anatomical localisation from a viewpoint."""

        quality = self.observation_quality(
            camera_pose=camera_pose,
            structure=structure,
            occluders=occluders,
        )

        uncertainty = (
            PositionUncertainty.isotropic(
                sigma=quality.localisation_sigma
            )
        )

        localisation_noise = (
            rng.multivariate_normal(
                mean=np.zeros(
                    3,
                    dtype=float,
                ),
                cov=uncertainty.covariance,
            )
        )

        estimated_centre = (
            structure.centre
            + localisation_noise
        )

        localisation_error = float(
            np.linalg.norm(
                estimated_centre
                - structure.centre
            )
        )

        estimated_structure = EstimatedStructure(
            estimated_centre=estimated_centre,
            physical_radius=(
                structure.physical_radius
            ),
            base_safety_margin=(
                structure.safety_margin
            ),
            uncertainty=uncertainty,
        )

        return StructureObservation(
            estimated_structure=(
                estimated_structure
            ),
            quality=quality,
            localisation_error=(
                localisation_error
            ),
        )

    def observe_structures(
        self,
        camera_pose: CameraPose,
        structures: tuple[
            SphericalStructure,
            ...
        ],
        rng: np.random.Generator,
        occluders: tuple[
            SphericalStructure,
            ...
        ] = (),
    ) -> tuple[StructureObservation, ...]:
        """Observe multiple anatomical structures from one viewpoint."""

        return tuple(
            self.observe_structure(
                camera_pose=camera_pose,
                structure=structure,
                rng=rng,
                occluders=occluders,
            )
            for structure in structures
        )


def mean_observation_sigma(
    observations: tuple[
        StructureObservation,
        ...
    ],
) -> float:
    """Return mean predicted localisation standard deviation."""

    if len(observations) == 0:
        return 0.0

    return float(
        np.mean(
            [
                observation.quality.localisation_sigma
                for observation in observations
            ]
        )
    )


def mean_localisation_error(
    observations: tuple[
        StructureObservation,
        ...
    ],
) -> float:
    """Return mean realised localisation error."""

    if len(observations) == 0:
        return 0.0

    return float(
        np.mean(
            [
                observation.localisation_error
                for observation in observations
            ]
        )
    )


def visible_fraction(
    observations: tuple[
        StructureObservation,
        ...
    ],
) -> float:
    """Return fraction of observations inside the camera frustum."""

    if len(observations) == 0:
        return 0.0

    return float(
        np.mean(
            [
                observation.quality.visible
                for observation in observations
            ]
        )
    )


def occluded_fraction(
    observations: tuple[
        StructureObservation,
        ...
    ],
) -> float:
    """Return fraction of observations geometrically occluded."""

    if len(observations) == 0:
        return 0.0

    return float(
        np.mean(
            [
                observation.quality.occluded
                for observation in observations
            ]
        )
    )