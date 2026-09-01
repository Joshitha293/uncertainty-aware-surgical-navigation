"""Tests for registered temporal perception-to-navigation integration."""

from dataclasses import dataclass

import numpy as np
import pytest

from src.geometry.transforms import (
    make_transform,
)
from src.perception.state_estimation import (
    ConstantVelocityKalmanFilter,
    KalmanFilterConfig,
)
from src.perception.tracked_navigation import (
    integrate_learned_stereo_observation,
    learned_stereo_registered_measurement,
    register_position_measurement,
    registration_pose_jacobian,
)


@dataclass(frozen=True)
class StubTriangulation:
    point_world: np.ndarray


@dataclass(frozen=True)
class StubStereoResult:
    triangulation: StubTriangulation

    world_position_covariance: np.ndarray


def rotation_z(
    angle_radians: float,
) -> np.ndarray:
    cosine = np.cos(
        angle_radians
    )

    sine = np.sin(
        angle_radians
    )

    return np.asarray(
        [
            [
                cosine,
                -sine,
                0.0,
            ],
            [
                sine,
                cosine,
                0.0,
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
        ],
        dtype=float,
    )


def registration_transform() -> np.ndarray:
    return make_transform(
        rotation=rotation_z(
            np.deg2rad(
                90.0
            )
        ),
        translation=np.asarray(
            [
                0.010,
                -0.020,
                0.030,
            ],
            dtype=float,
        ),
    )


def make_stereo_result(
    *,
    point: np.ndarray | None = None,
    sigma: float = 0.002,
) -> StubStereoResult:
    if point is None:
        point = np.asarray(
            [
                0.100,
                0.020,
                0.200,
            ],
            dtype=float,
        )

    return StubStereoResult(
        triangulation=(
            StubTriangulation(
                point_world=np.asarray(
                    point,
                    dtype=float,
                )
            )
        ),
        world_position_covariance=(
            sigma**2
            * np.eye(
                3,
                dtype=float,
            )
        ),
    )


def make_tracker(
    *,
    gated: bool = True,
) -> ConstantVelocityKalmanFilter:
    return ConstantVelocityKalmanFilter(
        KalmanFilterConfig(
            acceleration_sigma=0.025,
            initial_velocity_sigma=0.030,
            innovation_gate_threshold=(
                11.344866730144373
                if gated
                else None
            ),
        )
    )


def test_registration_transforms_position_into_navigation_frame():
    transform = (
        registration_transform()
    )

    point = np.asarray(
        [
            0.100,
            0.020,
            0.200,
        ]
    )

    result = (
        register_position_measurement(
            position=point,
            covariance=(
                1e-6
                * np.eye(
                    3
                )
            ),
            perception_to_navigation_transform=(
                transform
            ),
        )
    )

    expected = (
        transform[
            :3,
            :3
        ]
        @ point
        + transform[
            :3,
            3
        ]
    )

    np.testing.assert_allclose(
        result.position,
        expected,
        atol=1e-12,
    )


def test_registration_rotates_anisotropic_measurement_covariance():
    covariance = np.diag(
        [
            1e-6,
            4e-6,
            9e-6,
        ]
    )

    transform = (
        registration_transform()
    )

    result = (
        register_position_measurement(
            position=np.asarray(
                [
                    0.1,
                    0.0,
                    0.2,
                ]
            ),
            covariance=covariance,
            perception_to_navigation_transform=(
                transform
            ),
        )
    )

    rotation = transform[
        :3,
        :3
    ]

    expected = (
        rotation
        @ covariance
        @ rotation.T
    )

    np.testing.assert_allclose(
        result.covariance,
        expected,
        atol=1e-12,
    )


def test_registration_translation_does_not_change_measurement_covariance():
    covariance = np.diag(
        [
            2e-6,
            3e-6,
            5e-6,
        ]
    )

    identity_rotation = np.eye(
        3
    )

    transform = make_transform(
        rotation=identity_rotation,
        translation=np.asarray(
            [
                1.0,
                -2.0,
                3.0,
            ]
        ),
    )

    result = (
        register_position_measurement(
            position=np.asarray(
                [
                    0.1,
                    0.2,
                    0.3,
                ]
            ),
            covariance=covariance,
            perception_to_navigation_transform=(
                transform
            ),
        )
    )

    np.testing.assert_allclose(
        result.covariance,
        covariance,
        atol=1e-12,
    )


def test_registration_pose_uncertainty_increases_position_uncertainty():
    transform = (
        registration_transform()
    )

    point = np.asarray(
        [
            0.1,
            0.03,
            0.2,
        ]
    )

    measurement_covariance = (
        1e-6
        * np.eye(
            3
        )
    )

    without_pose_uncertainty = (
        register_position_measurement(
            position=point,
            covariance=(
                measurement_covariance
            ),
            perception_to_navigation_transform=(
                transform
            ),
        )
    )

    pose_covariance = np.diag(
        [
            1e-6,
            1e-6,
            1e-6,
            1e-4,
            1e-4,
            1e-4,
        ]
    )

    with_pose_uncertainty = (
        register_position_measurement(
            position=point,
            covariance=(
                measurement_covariance
            ),
            perception_to_navigation_transform=(
                transform
            ),
            registration_pose_covariance=(
                pose_covariance
            ),
        )
    )

    assert (
        np.trace(
            with_pose_uncertainty
            .covariance
        )
        > np.trace(
            without_pose_uncertainty
            .covariance
        )
    )


def test_registration_pose_jacobian_has_expected_shape():
    jacobian = (
        registration_pose_jacobian(
            point_perception_frame=np.asarray(
                [
                    0.1,
                    0.0,
                    0.2,
                ]
            ),
            perception_to_navigation_transform=(
                registration_transform()
            ),
        )
    )

    assert jacobian.shape == (
        3,
        6,
    )

    assert np.all(
        np.isfinite(
            jacobian
        )
    )


def test_learned_stereo_measurement_extracts_actual_interface_fields():
    stereo_result = (
        make_stereo_result()
    )

    result = (
        learned_stereo_registered_measurement(
            stereo_result=(
                stereo_result
            ),
            perception_to_navigation_transform=(
                np.eye(
                    4
                )
            ),
        )
    )

    np.testing.assert_allclose(
        result.position,
        stereo_result
        .triangulation
        .point_world,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        result.covariance,
        stereo_result
        .world_position_covariance,
        atol=1e-12,
    )


def test_first_observation_initialises_registered_tracking_state():
    stereo_result = (
        make_stereo_result()
    )

    result = (
        integrate_learned_stereo_observation(
            stereo_result=(
                stereo_result
            ),
            perception_to_navigation_transform=(
                registration_transform()
            ),
            tracker=make_tracker(),
            previous_state=None,
            timestamp=0.0,
            physical_radius=0.010,
            base_safety_margin=0.005,
            sigma_multiplier=2.0,
        )
    )

    np.testing.assert_allclose(
        result.state.position,
        result.registered_measurement.position,
        atol=1e-12,
    )

    assert result.measurement_accepted

    assert (
        result.mahalanobis_squared
        is None
    )


def test_second_consistent_observation_is_accepted():
    tracker = (
        make_tracker()
    )

    first = (
        integrate_learned_stereo_observation(
            stereo_result=(
                make_stereo_result()
            ),
            perception_to_navigation_transform=(
                np.eye(
                    4
                )
            ),
            tracker=tracker,
            previous_state=None,
            timestamp=0.0,
            physical_radius=0.010,
            base_safety_margin=0.005,
            sigma_multiplier=2.0,
        )
    )

    second_point = (
        first.state.position
        + np.asarray(
            [
                0.001,
                0.0,
                0.0,
            ]
        )
    )

    second = (
        integrate_learned_stereo_observation(
            stereo_result=(
                make_stereo_result(
                    point=second_point,
                    sigma=0.002,
                )
            ),
            perception_to_navigation_transform=(
                np.eye(
                    4
                )
            ),
            tracker=tracker,
            previous_state=(
                first.state
            ),
            timestamp=0.1,
            physical_radius=0.010,
            base_safety_margin=0.005,
            sigma_multiplier=2.0,
        )
    )

    assert second.measurement_accepted

    assert (
        second.state.accepted_measurements
        == first.state.accepted_measurements
        + 1
    )


def test_gross_stereo_outlier_is_rejected_by_tracker():
    tracker = (
        make_tracker()
    )

    first = (
        integrate_learned_stereo_observation(
            stereo_result=(
                make_stereo_result(
                    sigma=0.001
                )
            ),
            perception_to_navigation_transform=(
                np.eye(
                    4
                )
            ),
            tracker=tracker,
            previous_state=None,
            timestamp=0.0,
            physical_radius=0.010,
            base_safety_margin=0.005,
            sigma_multiplier=2.0,
        )
    )

    outlier = (
        integrate_learned_stereo_observation(
            stereo_result=(
                make_stereo_result(
                    point=np.asarray(
                        [
                            1.0,
                            1.0,
                            1.0,
                        ]
                    ),
                    sigma=0.001,
                )
            ),
            perception_to_navigation_transform=(
                np.eye(
                    4
                )
            ),
            tracker=tracker,
            previous_state=(
                first.state
            ),
            timestamp=0.1,
            physical_radius=0.010,
            base_safety_margin=0.005,
            sigma_multiplier=2.0,
        )
    )

    assert not outlier.measurement_accepted

    assert (
        outlier.state.accepted_measurements
        == first.state.accepted_measurements
    )

    np.testing.assert_allclose(
        outlier.state.position,
        first.state.position,
        atol=1e-6,
    )


def test_tracked_perception_keeps_ground_truth_out_of_runtime_result():
    result = (
        integrate_learned_stereo_observation(
            stereo_result=(
                make_stereo_result()
            ),
            perception_to_navigation_transform=(
                np.eye(
                    4
                )
            ),
            tracker=make_tracker(),
            previous_state=None,
            timestamp=0.0,
            physical_radius=0.010,
            base_safety_margin=0.005,
            sigma_multiplier=2.0,
        )
    )

    assert (
        result
        .perception_result
        .localisation_errors
        .size
        == 0
    )


def test_tracked_state_uncertainty_reaches_planner_margin():
    base_margin = 0.005

    result = (
        integrate_learned_stereo_observation(
            stereo_result=(
                make_stereo_result(
                    sigma=0.003
                )
            ),
            perception_to_navigation_transform=(
                np.eye(
                    4
                )
            ),
            tracker=make_tracker(),
            previous_state=None,
            timestamp=0.0,
            physical_radius=0.010,
            base_safety_margin=(
                base_margin
            ),
            sigma_multiplier=2.0,
        )
    )

    assert len(
        result
        .planning_perception
        .structures
    ) == 1

    assert (
        result
        .planning_perception
        .structures[
            0
        ]
        .safety_margin
        > base_margin
    )


def test_integrated_covariance_remains_symmetric_psd():
    result = (
        integrate_learned_stereo_observation(
            stereo_result=(
                make_stereo_result(
                    sigma=0.003
                )
            ),
            perception_to_navigation_transform=(
                registration_transform()
            ),
            tracker=make_tracker(),
            previous_state=None,
            timestamp=0.0,
            physical_radius=0.010,
            base_safety_margin=0.005,
            sigma_multiplier=2.0,
            registration_pose_covariance=np.diag(
                [
                    1e-6,
                    1e-6,
                    1e-6,
                    1e-5,
                    1e-5,
                    1e-5,
                ]
            ),
        )
    )

    covariance = (
        result
        .state
        .position_covariance
    )

    np.testing.assert_allclose(
        covariance,
        covariance.T,
        atol=1e-10,
    )

    assert np.all(
        np.linalg.eigvalsh(
            covariance
        )
        >= -1e-10
    )


def test_invalid_registration_pose_covariance_is_rejected():
    with pytest.raises(
        ValueError,
        match="registration_pose_covariance",
    ):
        register_position_measurement(
            position=np.zeros(
                3
            ),
            covariance=np.eye(
                3
            ),
            perception_to_navigation_transform=np.eye(
                4
            ),
            registration_pose_covariance=np.eye(
                3
            ),
        )