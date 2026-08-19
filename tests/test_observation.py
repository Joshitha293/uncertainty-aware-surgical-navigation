"""Tests for viewpoint-dependent surgical perception."""

import numpy as np
import pytest

from src.geometry.workspace import SphericalStructure
from src.perception.camera import (
    CameraIntrinsics,
    CameraPose,
    SurgicalCamera,
    look_at_rotation,
)
from src.perception.observation import (
    ObservationModelConfig,
    ViewpointObservationModel,
    mean_localisation_error,
    mean_observation_sigma,
    occluded_fraction,
    visible_fraction,
)


def make_camera() -> SurgicalCamera:
    return SurgicalCamera(
        CameraIntrinsics(
            horizontal_fov=np.deg2rad(70.0),
            vertical_fov=np.deg2rad(55.0),
            near_distance=0.02,
            far_distance=0.60,
        )
    )


def make_model() -> ViewpointObservationModel:
    return ViewpointObservationModel(
        camera=make_camera(),
        config=ObservationModelConfig(
            base_sigma=0.002,
            reference_distance=0.15,
            distance_weight=1.0,
            angle_weight=1.0,
            invisible_sigma=0.050,
            occluded_sigma=0.030,
        ),
    )


def make_structure() -> SphericalStructure:
    return SphericalStructure(
        centre=np.array(
            [0.20, 0.00, 0.00],
            dtype=float,
        ),
        physical_radius=0.025,
        safety_margin=0.015,
    )


def make_good_pose(
    target: np.ndarray,
) -> CameraPose:
    camera_position = np.array(
        [0.05, 0.00, 0.00],
        dtype=float,
    )

    return CameraPose(
        position=camera_position,
        rotation=look_at_rotation(
            camera_position,
            target,
        ),
    )


def test_good_viewpoint_has_low_uncertainty():
    model = make_model()
    structure = make_structure()

    pose = make_good_pose(
        structure.centre
    )

    quality = model.observation_quality(
        camera_pose=pose,
        structure=structure,
    )

    assert quality.visible
    assert not quality.occluded

    assert quality.localisation_sigma == pytest.approx(
        0.002,
        abs=1e-12,
    )


def test_farther_viewpoint_increases_uncertainty():
    model = make_model()
    structure = make_structure()

    near_position = np.array(
        [0.05, 0.00, 0.00],
        dtype=float,
    )

    far_position = np.array(
        [-0.10, 0.00, 0.00],
        dtype=float,
    )

    near_pose = CameraPose(
        position=near_position,
        rotation=look_at_rotation(
            near_position,
            structure.centre,
        ),
    )

    far_pose = CameraPose(
        position=far_position,
        rotation=look_at_rotation(
            far_position,
            structure.centre,
        ),
    )

    near_quality = model.observation_quality(
        camera_pose=near_pose,
        structure=structure,
    )

    far_quality = model.observation_quality(
        camera_pose=far_pose,
        structure=structure,
    )

    assert near_quality.visible
    assert far_quality.visible

    assert not near_quality.occluded
    assert not far_quality.occluded

    assert (
        far_quality.localisation_sigma
        > near_quality.localisation_sigma
    )


def test_off_axis_structure_increases_uncertainty():
    model = make_model()

    centred_structure = make_structure()

    off_axis_structure = SphericalStructure(
        centre=np.array(
            [0.20, 0.07, 0.00],
            dtype=float,
        ),
        physical_radius=0.025,
        safety_margin=0.015,
    )

    camera_position = np.zeros(
        3,
        dtype=float,
    )

    pose = CameraPose(
        position=camera_position,
        rotation=look_at_rotation(
            camera_position,
            centred_structure.centre,
        ),
    )

    centred_quality = model.observation_quality(
        camera_pose=pose,
        structure=centred_structure,
    )

    off_axis_quality = model.observation_quality(
        camera_pose=pose,
        structure=off_axis_structure,
    )

    assert centred_quality.visible
    assert off_axis_quality.visible

    assert (
        off_axis_quality.localisation_sigma
        > centred_quality.localisation_sigma
    )


def test_invisible_structure_receives_invisible_sigma():
    model = make_model()

    structure = SphericalStructure(
        centre=np.array(
            [0.0, 0.0, -0.20],
            dtype=float,
        ),
        physical_radius=0.025,
        safety_margin=0.015,
    )

    pose = CameraPose(
        position=np.zeros(
            3,
            dtype=float,
        ),
        rotation=np.eye(3),
    )

    quality = model.observation_quality(
        camera_pose=pose,
        structure=structure,
    )

    assert not quality.visible
    assert not quality.occluded

    assert quality.localisation_sigma == pytest.approx(
        0.050
    )


def test_occluded_structure_receives_occluded_sigma():
    model = make_model()
    structure = make_structure()

    pose = CameraPose(
        position=np.zeros(
            3,
            dtype=float,
        ),
        rotation=look_at_rotation(
            np.zeros(3),
            structure.centre,
        ),
    )

    occluder = SphericalStructure(
        centre=np.array(
            [0.10, 0.00, 0.00],
            dtype=float,
        ),
        physical_radius=0.025,
        safety_margin=0.010,
    )

    quality = model.observation_quality(
        camera_pose=pose,
        structure=structure,
        occluders=(occluder,),
    )

    assert quality.visible
    assert quality.occluded

    assert quality.localisation_sigma == pytest.approx(
        0.030
    )


def test_same_target_is_better_without_occluder():
    model = make_model()
    structure = make_structure()

    pose = CameraPose(
        position=np.zeros(
            3,
            dtype=float,
        ),
        rotation=look_at_rotation(
            np.zeros(3),
            structure.centre,
        ),
    )

    occluder = SphericalStructure(
        centre=np.array(
            [0.10, 0.00, 0.00],
            dtype=float,
        ),
        physical_radius=0.025,
        safety_margin=0.010,
    )

    clear_quality = model.observation_quality(
        camera_pose=pose,
        structure=structure,
        occluders=(),
    )

    blocked_quality = model.observation_quality(
        camera_pose=pose,
        structure=structure,
        occluders=(occluder,),
    )

    assert not clear_quality.occluded
    assert blocked_quality.occluded

    assert (
        blocked_quality.localisation_sigma
        > clear_quality.localisation_sigma
    )


def test_observation_is_reproducible_for_fixed_seed():
    model = make_model()
    structure = make_structure()

    pose = make_good_pose(
        structure.centre
    )

    observation_a = model.observe_structure(
        camera_pose=pose,
        structure=structure,
        rng=np.random.default_rng(123),
    )

    observation_b = model.observe_structure(
        camera_pose=pose,
        structure=structure,
        rng=np.random.default_rng(123),
    )

    np.testing.assert_allclose(
        observation_a.estimated_structure.estimated_centre,
        observation_b.estimated_structure.estimated_centre,
        atol=1e-12,
    )

    assert (
        observation_a.localisation_error
        == pytest.approx(
            observation_b.localisation_error
        )
    )


def test_observation_preserves_structure_geometry():
    model = make_model()
    structure = make_structure()

    pose = make_good_pose(
        structure.centre
    )

    observation = model.observe_structure(
        camera_pose=pose,
        structure=structure,
        rng=np.random.default_rng(7),
    )

    estimate = (
        observation.estimated_structure
    )

    assert estimate.physical_radius == pytest.approx(
        structure.physical_radius
    )

    assert estimate.base_safety_margin == pytest.approx(
        structure.safety_margin
    )


def test_observation_uncertainty_matches_quality_sigma():
    model = make_model()
    structure = make_structure()

    pose = make_good_pose(
        structure.centre
    )

    observation = model.observe_structure(
        camera_pose=pose,
        structure=structure,
        rng=np.random.default_rng(7),
    )

    assert (
        observation.estimated_structure.uncertainty.principal_sigma
        == pytest.approx(
            observation.quality.localisation_sigma
        )
    )


def test_multiple_structure_observation_preserves_count():
    model = make_model()

    structures = (
        make_structure(),
        SphericalStructure(
            centre=np.array(
                [0.22, 0.03, 0.02],
                dtype=float,
            ),
            physical_radius=0.020,
            safety_margin=0.012,
        ),
    )

    pose = make_good_pose(
        np.array(
            [0.20, 0.00, 0.00]
        )
    )

    observations = model.observe_structures(
        camera_pose=pose,
        structures=structures,
        rng=np.random.default_rng(9),
    )

    assert len(observations) == len(
        structures
    )


def test_mean_observation_sigma_matches_manual_mean():
    model = make_model()

    structures = (
        make_structure(),
        SphericalStructure(
            centre=np.array(
                [0.22, 0.03, 0.02],
                dtype=float,
            ),
            physical_radius=0.020,
            safety_margin=0.012,
        ),
    )

    pose = make_good_pose(
        np.array(
            [0.20, 0.00, 0.00]
        )
    )

    observations = model.observe_structures(
        camera_pose=pose,
        structures=structures,
        rng=np.random.default_rng(9),
    )

    expected = float(
        np.mean(
            [
                observation.quality.localisation_sigma
                for observation in observations
            ]
        )
    )

    assert mean_observation_sigma(
        observations
    ) == pytest.approx(
        expected
    )


def test_mean_localisation_error_matches_manual_mean():
    model = make_model()

    structures = (
        make_structure(),
        SphericalStructure(
            centre=np.array(
                [0.22, 0.03, 0.02],
                dtype=float,
            ),
            physical_radius=0.020,
            safety_margin=0.012,
        ),
    )

    pose = make_good_pose(
        np.array(
            [0.20, 0.00, 0.00]
        )
    )

    observations = model.observe_structures(
        camera_pose=pose,
        structures=structures,
        rng=np.random.default_rng(9),
    )

    expected = float(
        np.mean(
            [
                observation.localisation_error
                for observation in observations
            ]
        )
    )

    assert mean_localisation_error(
        observations
    ) == pytest.approx(
        expected
    )


def test_visible_fraction_is_correct():
    model = make_model()

    visible_structure = make_structure()

    invisible_structure = SphericalStructure(
        centre=np.array(
            [0.0, 0.0, -0.20],
            dtype=float,
        ),
        physical_radius=0.025,
        safety_margin=0.015,
    )

    pose = CameraPose(
        position=np.zeros(3),
        rotation=look_at_rotation(
            np.zeros(3),
            visible_structure.centre,
        ),
    )

    observations = model.observe_structures(
        camera_pose=pose,
        structures=(
            visible_structure,
            invisible_structure,
        ),
        rng=np.random.default_rng(1),
    )

    assert visible_fraction(
        observations
    ) == pytest.approx(
        0.5
    )


def test_occluded_fraction_is_correct():
    model = make_model()

    target_a = SphericalStructure(
        centre=np.array(
            [0.20, 0.00, 0.00]
        ),
        physical_radius=0.020,
        safety_margin=0.010,
    )

    target_b = SphericalStructure(
        centre=np.array(
            [0.20, 0.10, 0.00]
        ),
        physical_radius=0.020,
        safety_margin=0.010,
    )

    occluder = SphericalStructure(
        centre=np.array(
            [0.10, 0.00, 0.00]
        ),
        physical_radius=0.025,
        safety_margin=0.010,
    )

    pose = CameraPose(
        position=np.zeros(3),
        rotation=look_at_rotation(
            np.zeros(3),
            target_a.centre,
        ),
    )

    observations = model.observe_structures(
        camera_pose=pose,
        structures=(
            target_a,
            target_b,
        ),
        rng=np.random.default_rng(2),
        occluders=(occluder,),
    )

    assert occluded_fraction(
        observations
    ) == pytest.approx(
        0.5
    )


def test_empty_summary_metrics_are_zero():
    assert mean_observation_sigma(
        ()
    ) == 0.0

    assert mean_localisation_error(
        ()
    ) == 0.0

    assert visible_fraction(
        ()
    ) == 0.0

    assert occluded_fraction(
        ()
    ) == 0.0


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError):
        ObservationModelConfig(
            base_sigma=-0.001
        )