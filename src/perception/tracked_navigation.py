"""Registered temporal perception for uncertainty-aware navigation.

This module connects:

    learned stereo 3-D perception
        ->
    rigid registration into a navigation frame
        ->
    covariance propagation
        ->
    temporal Kalman state estimation
        ->
    uncertainty-aware planner geometry

The stereo observation is expected to provide:

- triangulation.point_world
- world_position_covariance

which matches the interface exposed by LearnedStereoPerceptionResult.

A rigid transform maps the perception/world coordinate frame into the
navigation frame.

Measurement covariance is propagated through the rigid transform as:

    Sigma_nav = R Sigma_perception R^T

Optional small-angle registration-pose uncertainty can also be propagated:

    Sigma_nav =
        R Sigma_perception R^T
        +
        J_T Sigma_T J_T^T

where registration pose covariance is ordered:

    [tx, ty, tz, rx, ry, rz]

and rotational perturbations use a small-angle left-perturbation model in
the navigation frame.

This is simulation-only engineering software and does not constitute
clinical navigation or patient-registration validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.geometry.transforms import (
    is_homogeneous_transform,
)
from src.perception.perception import (
    PerceptionResult,
)
from src.perception.planning import (
    PlanningPerception,
    uncertainty_aware_planning_structures,
)
from src.perception.state_estimation import (
    ConstantVelocityKalmanFilter,
    TrackingState,
    tracked_estimated_structure,
)


class TriangulationLike(
    Protocol
):
    """Structural interface required from stereo triangulation."""

    point_world: np.ndarray


class LearnedStereoMeasurementLike(
    Protocol
):
    """Structural interface required from learned stereo perception."""

    triangulation: TriangulationLike

    world_position_covariance: np.ndarray


@dataclass(frozen=True)
class RegisteredPositionMeasurement:
    """One uncertain 3-D measurement in the navigation frame."""

    position: np.ndarray

    covariance: np.ndarray

    def __post_init__(
        self,
    ) -> None:
        position = np.asarray(
            self.position,
            dtype=float,
        )

        covariance = np.asarray(
            self.covariance,
            dtype=float,
        )

        if position.shape != (
            3,
        ):
            raise ValueError(
                "position must have shape (3,)."
            )

        if covariance.shape != (
            3,
            3,
        ):
            raise ValueError(
                "covariance must have shape (3, 3)."
            )

        if (
            not np.all(
                np.isfinite(
                    position
                )
            )
            or not np.all(
                np.isfinite(
                    covariance
                )
            )
        ):
            raise ValueError(
                "measurement must contain finite values."
            )

        if not np.allclose(
            covariance,
            covariance.T,
            atol=1e-10,
        ):
            raise ValueError(
                "covariance must be symmetric."
            )

        eigenvalues = np.linalg.eigvalsh(
            covariance
        )

        if np.any(
            eigenvalues < -1e-10
        ):
            raise ValueError(
                "covariance must be positive semi-definite."
            )

        object.__setattr__(
            self,
            "position",
            position,
        )

        object.__setattr__(
            self,
            "covariance",
            covariance,
        )


@dataclass(frozen=True)
class TrackedNavigationResult:
    """Output from one registered tracking/navigation update."""

    state: TrackingState

    registered_measurement: RegisteredPositionMeasurement

    measurement_accepted: bool

    mahalanobis_squared: float | None

    perception_result: PerceptionResult

    planning_perception: PlanningPerception


def _validate_covariance(
    covariance: np.ndarray,
    *,
    name: str,
    shape: tuple[int, int],
) -> np.ndarray:
    """Validate a symmetric positive-semidefinite covariance matrix."""

    covariance = np.asarray(
        covariance,
        dtype=float,
    )

    if covariance.shape != shape:
        raise ValueError(
            f"{name} must have shape {shape}."
        )

    if not np.all(
        np.isfinite(
            covariance
        )
    ):
        raise ValueError(
            f"{name} must contain finite values."
        )

    if not np.allclose(
        covariance,
        covariance.T,
        atol=1e-10,
    ):
        raise ValueError(
            f"{name} must be symmetric."
        )

    if np.any(
        np.linalg.eigvalsh(
            covariance
        )
        < -1e-10
    ):
        raise ValueError(
            f"{name} must be positive semi-definite."
        )

    return covariance


def skew_symmetric(
    vector: np.ndarray,
) -> np.ndarray:
    """Return the 3 x 3 cross-product matrix of a 3-D vector."""

    vector = np.asarray(
        vector,
        dtype=float,
    )

    if vector.shape != (
        3,
    ):
        raise ValueError(
            "vector must have shape (3,)."
        )

    if not np.all(
        np.isfinite(
            vector
        )
    ):
        raise ValueError(
            "vector must contain finite values."
        )

    x, y, z = vector

    return np.asarray(
        [
            [
                0.0,
                -z,
                y,
            ],
            [
                z,
                0.0,
                -x,
            ],
            [
                -y,
                x,
                0.0,
            ],
        ],
        dtype=float,
    )


def registration_pose_jacobian(
    *,
    point_perception_frame: np.ndarray,
    perception_to_navigation_transform: np.ndarray,
) -> np.ndarray:
    """Return point Jacobian with respect to a small rigid-pose perturbation.

    Registration-pose perturbation ordering is:

        [tx, ty, tz, rx, ry, rz]

    Rotational perturbations are defined as small left perturbations in the
    navigation frame.
    """

    point = np.asarray(
        point_perception_frame,
        dtype=float,
    )

    transform = np.asarray(
        perception_to_navigation_transform,
        dtype=float,
    )

    if point.shape != (
        3,
    ):
        raise ValueError(
            "point_perception_frame must have shape (3,)."
        )

    if not is_homogeneous_transform(
        transform
    ):
        raise ValueError(
            "perception_to_navigation_transform must be a valid "
            "rigid transformation."
        )

    rotation = transform[
        :3,
        :3
    ]

    rotated_point = (
        rotation
        @ point
    )

    jacobian = np.zeros(
        (
            3,
            6,
        ),
        dtype=float,
    )

    jacobian[
        :,
        :3
    ] = np.eye(
        3,
        dtype=float,
    )

    jacobian[
        :,
        3:
    ] = (
        -skew_symmetric(
            rotated_point
        )
    )

    return jacobian


def register_position_measurement(
    *,
    position: np.ndarray,
    covariance: np.ndarray,
    perception_to_navigation_transform: np.ndarray,
    registration_pose_covariance: np.ndarray | None = None,
) -> RegisteredPositionMeasurement:
    """Transform an uncertain position into the navigation frame."""

    position = np.asarray(
        position,
        dtype=float,
    )

    transform = np.asarray(
        perception_to_navigation_transform,
        dtype=float,
    )

    if position.shape != (
        3,
    ):
        raise ValueError(
            "position must have shape (3,)."
        )

    if not np.all(
        np.isfinite(
            position
        )
    ):
        raise ValueError(
            "position must contain finite values."
        )

    covariance = _validate_covariance(
        covariance,
        name="covariance",
        shape=(
            3,
            3,
        ),
    )

    if not is_homogeneous_transform(
        transform
    ):
        raise ValueError(
            "perception_to_navigation_transform must be a valid "
            "rigid transformation."
        )

    rotation = transform[
        :3,
        :3
    ]

    translation = transform[
        :3,
        3
    ]

    registered_position = (
        rotation
        @ position
        + translation
    )

    registered_covariance = (
        rotation
        @ covariance
        @ rotation.T
    )

    if (
        registration_pose_covariance
        is not None
    ):
        registration_pose_covariance = (
            _validate_covariance(
                registration_pose_covariance,
                name="registration_pose_covariance",
                shape=(
                    6,
                    6,
                ),
            )
        )

        jacobian = (
            registration_pose_jacobian(
                point_perception_frame=(
                    position
                ),
                perception_to_navigation_transform=(
                    transform
                ),
            )
        )

        registered_covariance = (
            registered_covariance
            + jacobian
            @ registration_pose_covariance
            @ jacobian.T
        )

    registered_covariance = (
        0.5
        * (
            registered_covariance
            + registered_covariance.T
        )
    )

    eigenvalues, eigenvectors = (
        np.linalg.eigh(
            registered_covariance
        )
    )

    eigenvalues = np.clip(
        eigenvalues,
        0.0,
        None,
    )

    registered_covariance = (
        eigenvectors
        @ np.diag(
            eigenvalues
        )
        @ eigenvectors.T
    )

    registered_covariance = (
        0.5
        * (
            registered_covariance
            + registered_covariance.T
        )
    )

    return RegisteredPositionMeasurement(
        position=registered_position,
        covariance=registered_covariance,
    )


def learned_stereo_registered_measurement(
    *,
    stereo_result: LearnedStereoMeasurementLike,
    perception_to_navigation_transform: np.ndarray,
    registration_pose_covariance: np.ndarray | None = None,
) -> RegisteredPositionMeasurement:
    """Extract and register a learned-stereo 3-D measurement."""

    return register_position_measurement(
        position=np.asarray(
            stereo_result
            .triangulation
            .point_world,
            dtype=float,
        ),
        covariance=np.asarray(
            stereo_result
            .world_position_covariance,
            dtype=float,
        ),
        perception_to_navigation_transform=(
            perception_to_navigation_transform
        ),
        registration_pose_covariance=(
            registration_pose_covariance
        ),
    )


def tracking_state_to_perception_result(
    *,
    state: TrackingState,
    physical_radius: float,
    base_safety_margin: float,
) -> PerceptionResult:
    """Convert the tracked state into the existing perception interface."""

    estimate = (
        tracked_estimated_structure(
            state,
            physical_radius=(
                physical_radius
            ),
            base_safety_margin=(
                base_safety_margin
            ),
        )
    )

    return PerceptionResult(
        estimated_structures=(
            estimate,
        ),
        # Ground truth remains evaluation-only.
        localisation_errors=np.empty(
            (
                0,
            ),
            dtype=float,
        ),
    )


def integrate_learned_stereo_observation(
    *,
    stereo_result: LearnedStereoMeasurementLike,
    perception_to_navigation_transform: np.ndarray,
    tracker: ConstantVelocityKalmanFilter,
    previous_state: TrackingState | None,
    timestamp: float,
    physical_radius: float,
    base_safety_margin: float,
    sigma_multiplier: float,
    registration_pose_covariance: np.ndarray | None = None,
) -> TrackedNavigationResult:
    """Register, track and expose one learned stereo observation to planning."""

    if (
        not np.isfinite(
            physical_radius
        )
        or physical_radius < 0.0
    ):
        raise ValueError(
            "physical_radius must be finite and non-negative."
        )

    if (
        not np.isfinite(
            base_safety_margin
        )
        or base_safety_margin < 0.0
    ):
        raise ValueError(
            "base_safety_margin must be finite and non-negative."
        )

    if (
        not np.isfinite(
            sigma_multiplier
        )
        or sigma_multiplier < 0.0
    ):
        raise ValueError(
            "sigma_multiplier must be finite and non-negative."
        )

    measurement = (
        learned_stereo_registered_measurement(
            stereo_result=(
                stereo_result
            ),
            perception_to_navigation_transform=(
                perception_to_navigation_transform
            ),
            registration_pose_covariance=(
                registration_pose_covariance
            ),
        )
    )

    if previous_state is None:
        state = tracker.initialise(
            measurement=(
                measurement.position
            ),
            measurement_covariance=(
                measurement.covariance
            ),
            timestamp=timestamp,
        )

        accepted = True

        mahalanobis_squared = None

    else:
        update = tracker.step(
            previous_state,
            measurement=(
                measurement.position
            ),
            measurement_covariance=(
                measurement.covariance
            ),
            timestamp=timestamp,
        )

        state = update.state

        accepted = (
            update.accepted
        )

        mahalanobis_squared = float(
            update.mahalanobis_squared
        )

    perception_result = (
        tracking_state_to_perception_result(
            state=state,
            physical_radius=(
                physical_radius
            ),
            base_safety_margin=(
                base_safety_margin
            ),
        )
    )

    planning_perception = (
        uncertainty_aware_planning_structures(
            perception_result=(
                perception_result
            ),
            sigma_multiplier=(
                sigma_multiplier
            ),
        )
    )

    return TrackedNavigationResult(
        state=state,
        registered_measurement=(
            measurement
        ),
        measurement_accepted=bool(
            accepted
        ),
        mahalanobis_squared=(
            mahalanobis_squared
        ),
        perception_result=(
            perception_result
        ),
        planning_perception=(
            planning_perception
        ),
    )