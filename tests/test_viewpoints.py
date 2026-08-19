"""Tests for candidate surgical camera viewpoint generation."""

import numpy as np
import pytest

from src.perception.camera import CameraPose
from src.perception.viewpoints import (
    ViewpointSamplingConfig,
    generate_candidate_viewpoints,
    nearest_candidate_index,
    spherical_camera_position,
    viewpoint_displacement,
)


def test_spherical_camera_position_has_correct_radius() -> None:
    target = np.array(
        [0.20, 0.05, 0.02],
        dtype=float,
    )

    radius = 0.20

    position = spherical_camera_position(
        target_position=target,
        radius=radius,
        azimuth=np.deg2rad(45.0),
        elevation=np.deg2rad(20.0),
    )

    distance = np.linalg.norm(
        position - target
    )

    assert distance == pytest.approx(
        radius
    )


def test_zero_azimuth_zero_elevation_places_camera_on_positive_x() -> None:
    target = np.array(
        [0.20, 0.05, 0.02],
        dtype=float,
    )

    position = spherical_camera_position(
        target_position=target,
        radius=0.15,
        azimuth=0.0,
        elevation=0.0,
    )

    expected = np.array(
        [0.35, 0.05, 0.02],
        dtype=float,
    )

    np.testing.assert_allclose(
        position,
        expected,
        atol=1e-12,
    )


def test_positive_elevation_increases_height() -> None:
    target = np.zeros(
        3,
        dtype=float,
    )

    position = spherical_camera_position(
        target_position=target,
        radius=0.20,
        azimuth=0.0,
        elevation=np.deg2rad(30.0),
    )

    assert position[2] > 0.0


def test_invalid_radius_is_rejected() -> None:
    with pytest.raises(ValueError):
        spherical_camera_position(
            target_position=np.zeros(3),
            radius=0.0,
            azimuth=0.0,
            elevation=0.0,
        )


def test_default_configuration_generates_108_candidates() -> None:
    candidates = generate_candidate_viewpoints(
        target_position=np.zeros(3),
    )

    assert len(candidates) == 108


def test_custom_configuration_candidate_count_is_correct() -> None:
    config = ViewpointSamplingConfig(
        radii=(
            0.15,
            0.20,
        ),
        azimuth_count=8,
        elevation_angles=(
            np.deg2rad(-15.0),
            0.0,
            np.deg2rad(15.0),
        ),
    )

    candidates = generate_candidate_viewpoints(
        target_position=np.zeros(3),
        config=config,
    )

    expected_count = (
        2
        * 8
        * 3
    )

    assert len(candidates) == expected_count


def test_all_generated_candidates_look_toward_target() -> None:
    target = np.array(
        [0.20, 0.03, 0.01],
        dtype=float,
    )

    config = ViewpointSamplingConfig(
        radii=(0.15,),
        azimuth_count=6,
        elevation_angles=(0.0,),
    )

    candidates = generate_candidate_viewpoints(
        target_position=target,
        config=config,
    )

    for candidate in candidates:
        expected_forward = (
            target
            - candidate.pose.position
        )

        expected_forward = (
            expected_forward
            / np.linalg.norm(
                expected_forward
            )
        )

        np.testing.assert_allclose(
            candidate.pose.forward,
            expected_forward,
            atol=1e-10,
        )


def test_height_constraints_filter_candidates() -> None:
    target = np.zeros(
        3,
        dtype=float,
    )

    config = ViewpointSamplingConfig(
        radii=(0.20,),
        azimuth_count=4,
        elevation_angles=(
            np.deg2rad(-30.0),
            0.0,
            np.deg2rad(30.0),
        ),
        minimum_height=0.0,
    )

    candidates = generate_candidate_viewpoints(
        target_position=target,
        config=config,
    )

    assert all(
        candidate.pose.position[2] >= 0.0
        for candidate in candidates
    )


def test_viewpoint_displacement_matches_euclidean_distance() -> None:
    pose_a = CameraPose(
        position=np.array(
            [0.0, 0.0, 0.0],
            dtype=float,
        ),
        rotation=np.eye(3),
    )

    pose_b = CameraPose(
        position=np.array(
            [0.03, 0.04, 0.0],
            dtype=float,
        ),
        rotation=np.eye(3),
    )

    displacement = viewpoint_displacement(
        current_pose=pose_a,
        candidate_pose=pose_b,
    )

    assert displacement == pytest.approx(
        0.05
    )


def test_nearest_candidate_index_is_correct() -> None:
    current_pose = CameraPose(
        position=np.zeros(3),
        rotation=np.eye(3),
    )

    target = np.array(
        [0.20, 0.0, 0.0],
        dtype=float,
    )

    config = ViewpointSamplingConfig(
        radii=(0.10,),
        azimuth_count=4,
        elevation_angles=(0.0,),
    )

    candidates = generate_candidate_viewpoints(
        target_position=target,
        config=config,
    )

    distances = np.array(
        [
            np.linalg.norm(
                candidate.pose.position
                - current_pose.position
            )
            for candidate in candidates
        ]
    )

    expected_index = int(
        np.argmin(distances)
    )

    result = nearest_candidate_index(
        current_pose=current_pose,
        candidates=candidates,
    )

    assert result == expected_index


def test_empty_candidate_set_is_rejected() -> None:
    current_pose = CameraPose(
        position=np.zeros(3),
        rotation=np.eye(3),
    )

    with pytest.raises(ValueError):
        nearest_candidate_index(
            current_pose=current_pose,
            candidates=(),
        )


def test_invalid_sampling_config_is_rejected() -> None:
    with pytest.raises(ValueError):
        ViewpointSamplingConfig(
            radii=(),
        )

    with pytest.raises(ValueError):
        ViewpointSamplingConfig(
            azimuth_count=0
        )

    with pytest.raises(ValueError):
        ViewpointSamplingConfig(
            elevation_angles=()
        )