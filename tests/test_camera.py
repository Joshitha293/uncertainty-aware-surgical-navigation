"""Tests for the simulated surgical camera."""

import numpy as np
import pytest

from src.perception.camera import (
    CameraIntrinsics,
    CameraPose,
    SurgicalCamera,
    look_at_rotation,
)


def make_camera() -> SurgicalCamera:
    """Create a camera used by the tests."""
    return SurgicalCamera(
        CameraIntrinsics(
            horizontal_fov=np.deg2rad(70.0),
            vertical_fov=np.deg2rad(55.0),
            near_distance=0.02,
            far_distance=0.50,
        )
    )


def test_look_at_points_forward_axis_at_target() -> None:
    camera_position = np.array(
        [0.0, 0.0, 0.0]
    )

    target = np.array(
        [0.20, 0.05, 0.02]
    )

    rotation = look_at_rotation(
        camera_position,
        target,
    )

    expected = (
        target - camera_position
    )

    expected = (
        expected
        / np.linalg.norm(expected)
    )

    assert np.allclose(
        rotation[:, 2],
        expected,
    )


def test_look_at_rotation_is_proper_rotation() -> None:
    rotation = look_at_rotation(
        np.array(
            [0.0, 0.0, 0.0]
        ),
        np.array(
            [0.20, 0.00, 0.00]
        ),
    )

    assert np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=1e-10,
    )

    assert np.isclose(
        np.linalg.det(rotation),
        1.0,
        atol=1e-10,
    )


def test_world_camera_round_trip() -> None:
    position = np.array(
        [0.03, -0.02, 0.01]
    )

    rotation = look_at_rotation(
        position,
        np.array(
            [0.20, 0.00, 0.03]
        ),
    )

    pose = CameraPose(
        position=position,
        rotation=rotation,
    )

    point_world = np.array(
        [0.15, 0.04, 0.02]
    )

    point_camera = pose.world_to_camera(
        point_world
    )

    reconstructed = pose.camera_to_world(
        point_camera
    )

    assert np.allclose(
        reconstructed,
        point_world,
        atol=1e-12,
    )


def test_target_on_optical_axis_is_visible() -> None:
    camera = make_camera()

    camera_position = np.zeros(3)

    target = np.array(
        [0.20, 0.00, 0.00]
    )

    pose = CameraPose(
        position=camera_position,
        rotation=look_at_rotation(
            camera_position,
            target,
        ),
    )

    result = camera.evaluate_visibility(
        pose,
        target,
    )

    assert result.visible
    assert np.isclose(
        result.horizontal_angle,
        0.0,
        atol=1e-10,
    )
    assert np.isclose(
        result.vertical_angle,
        0.0,
        atol=1e-10,
    )


def test_point_behind_camera_is_not_visible() -> None:
    camera = make_camera()

    pose = CameraPose(
        position=np.zeros(3),
        rotation=np.eye(3),
    )

    result = camera.evaluate_visibility(
        pose,
        np.array(
            [0.0, 0.0, -0.10]
        ),
    )

    assert not result.visible


def test_point_outside_horizontal_fov_is_not_visible() -> None:
    camera = make_camera()

    pose = CameraPose(
        position=np.zeros(3),
        rotation=np.eye(3),
    )

    point = np.array(
        [0.20, 0.0, 0.10]
    )

    result = camera.evaluate_visibility(
        pose,
        point,
    )

    assert not result.visible


def test_point_too_near_is_not_visible() -> None:
    camera = make_camera()

    pose = CameraPose(
        position=np.zeros(3),
        rotation=np.eye(3),
    )

    result = camera.evaluate_visibility(
        pose,
        np.array(
            [0.0, 0.0, 0.01]
        ),
    )

    assert not result.visible


def test_point_too_far_is_not_visible() -> None:
    camera = make_camera()

    pose = CameraPose(
        position=np.zeros(3),
        rotation=np.eye(3),
    )

    result = camera.evaluate_visibility(
        pose,
        np.array(
            [0.0, 0.0, 0.60]
        ),
    )

    assert not result.visible


def test_invalid_rotation_is_rejected() -> None:
    with pytest.raises(ValueError):
        CameraPose(
            position=np.zeros(3),
            rotation=np.ones(
                (3, 3)
            ),
        )


def test_invalid_intrinsics_are_rejected() -> None:
    with pytest.raises(ValueError):
        CameraIntrinsics(
            horizontal_fov=0.0,
            vertical_fov=np.deg2rad(55.0),
            near_distance=0.02,
            far_distance=0.50,
        )