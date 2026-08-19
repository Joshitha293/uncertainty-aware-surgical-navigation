"""Tests for geometric occlusion modelling."""

import numpy as np
import pytest

from src.geometry.workspace import SphericalStructure
from src.perception.camera import (
    CameraPose,
    look_at_rotation,
)
from src.perception.occlusion import (
    evaluate_point_occlusion,
    ray_sphere_intersection_distance,
    visible_after_occlusion,
)


def make_pose() -> CameraPose:
    camera_position = np.array(
        [0.0, 0.0, 0.0],
        dtype=float,
    )

    target_position = np.array(
        [0.30, 0.0, 0.0],
        dtype=float,
    )

    return CameraPose(
        position=camera_position,
        rotation=look_at_rotation(
            camera_position,
            target_position,
        ),
    )


def test_ray_intersects_sphere() -> None:
    distance = ray_sphere_intersection_distance(
        ray_origin=np.zeros(3),
        ray_direction=np.array(
            [1.0, 0.0, 0.0]
        ),
        sphere_centre=np.array(
            [0.20, 0.0, 0.0]
        ),
        sphere_radius=0.05,
    )

    assert distance is not None
    assert distance == pytest.approx(
        0.15
    )


def test_ray_misses_sphere() -> None:
    distance = ray_sphere_intersection_distance(
        ray_origin=np.zeros(3),
        ray_direction=np.array(
            [1.0, 0.0, 0.0]
        ),
        sphere_centre=np.array(
            [0.20, 0.20, 0.0]
        ),
        sphere_radius=0.05,
    )

    assert distance is None


def test_non_unit_ray_direction_is_supported() -> None:
    distance = ray_sphere_intersection_distance(
        ray_origin=np.zeros(3),
        ray_direction=np.array(
            [2.0, 0.0, 0.0]
        ),
        sphere_centre=np.array(
            [0.20, 0.0, 0.0]
        ),
        sphere_radius=0.05,
    )

    assert distance == pytest.approx(
        0.15
    )


def test_zero_ray_direction_is_rejected() -> None:
    with pytest.raises(ValueError):
        ray_sphere_intersection_distance(
            ray_origin=np.zeros(3),
            ray_direction=np.zeros(3),
            sphere_centre=np.array(
                [0.20, 0.0, 0.0]
            ),
            sphere_radius=0.05,
        )


def test_non_positive_sphere_radius_is_rejected() -> None:
    with pytest.raises(ValueError):
        ray_sphere_intersection_distance(
            ray_origin=np.zeros(3),
            ray_direction=np.array(
                [1.0, 0.0, 0.0]
            ),
            sphere_centre=np.array(
                [0.20, 0.0, 0.0]
            ),
            sphere_radius=0.0,
        )


def test_structure_between_camera_and_target_causes_occlusion() -> None:
    pose = make_pose()

    target = np.array(
        [0.30, 0.0, 0.0],
        dtype=float,
    )

    occluder = SphericalStructure(
        centre=np.array(
            [0.15, 0.0, 0.0],
            dtype=float,
        ),
        physical_radius=0.03,
        safety_margin=0.01,
    )

    result = evaluate_point_occlusion(
        camera_pose=pose,
        target_position=target,
        occluders=(occluder,),
    )

    assert result.occluded
    assert result.occluding_structure_index == 0
    assert result.nearest_occluder_distance < 0.30


def test_structure_beyond_target_does_not_occlude() -> None:
    pose = make_pose()

    target = np.array(
        [0.30, 0.0, 0.0],
        dtype=float,
    )

    occluder = SphericalStructure(
        centre=np.array(
            [0.45, 0.0, 0.0],
            dtype=float,
        ),
        physical_radius=0.03,
        safety_margin=0.01,
    )

    result = evaluate_point_occlusion(
        camera_pose=pose,
        target_position=target,
        occluders=(occluder,),
    )

    assert not result.occluded
    assert result.occluding_structure_index is None
    assert np.isinf(
        result.nearest_occluder_distance
    )


def test_offset_structure_does_not_occlude() -> None:
    pose = make_pose()

    target = np.array(
        [0.30, 0.0, 0.0],
        dtype=float,
    )

    occluder = SphericalStructure(
        centre=np.array(
            [0.15, 0.10, 0.0],
            dtype=float,
        ),
        physical_radius=0.03,
        safety_margin=0.01,
    )

    result = evaluate_point_occlusion(
        camera_pose=pose,
        target_position=target,
        occluders=(occluder,),
    )

    assert not result.occluded


def test_nearest_occluder_is_reported() -> None:
    pose = make_pose()

    target = np.array(
        [0.40, 0.0, 0.0],
        dtype=float,
    )

    near_occluder = SphericalStructure(
        centre=np.array(
            [0.12, 0.0, 0.0],
            dtype=float,
        ),
        physical_radius=0.02,
        safety_margin=0.01,
    )

    far_occluder = SphericalStructure(
        centre=np.array(
            [0.25, 0.0, 0.0],
            dtype=float,
        ),
        physical_radius=0.02,
        safety_margin=0.01,
    )

    result = evaluate_point_occlusion(
        camera_pose=pose,
        target_position=target,
        occluders=(
            far_occluder,
            near_occluder,
        ),
    )

    assert result.occluded

    # near_occluder is index 1 in the tuple
    assert result.occluding_structure_index == 1


def test_visible_after_occlusion_false_when_blocked() -> None:
    pose = make_pose()

    target = np.array(
        [0.30, 0.0, 0.0],
        dtype=float,
    )

    occluder = SphericalStructure(
        centre=np.array(
            [0.15, 0.0, 0.0],
            dtype=float,
        ),
        physical_radius=0.03,
        safety_margin=0.01,
    )

    assert not visible_after_occlusion(
        camera_pose=pose,
        target_position=target,
        occluders=(occluder,),
    )


def test_visible_after_occlusion_true_without_blocker() -> None:
    pose = make_pose()

    target = np.array(
        [0.30, 0.0, 0.0],
        dtype=float,
    )

    assert visible_after_occlusion(
        camera_pose=pose,
        target_position=target,
        occluders=(),
    )


def test_target_at_camera_position_is_rejected() -> None:
    pose = make_pose()

    with pytest.raises(ValueError):
        evaluate_point_occlusion(
            camera_pose=pose,
            target_position=pose.position,
            occluders=(),
        )