"""Tests for closed-loop generic active perception."""

import numpy as np
import pytest

from src.geometry.workspace import SphericalStructure
from src.perception.active_perception import GenericActivePerception
from src.perception.camera import (
    CameraIntrinsics,
    CameraPose,
    SurgicalCamera,
    look_at_rotation,
)
from src.perception.closed_loop import (
    ClosedLoopActivePerception,
    execute_reproducible_cycle,
    selected_viewpoint_position,
    uncertainty_improvement,
    uncertainty_reduction_fraction,
)
from src.perception.observation import (
    ObservationModelConfig,
    ViewpointObservationModel,
)
from src.perception.viewpoint_scoring import (
    GenericViewpointScorer,
    ViewpointScoringConfig,
)
from src.perception.viewpoints import (
    generate_candidate_viewpoints,
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


def make_target() -> SphericalStructure:
    return SphericalStructure(
        centre=np.array(
            [0.24, 0.00, 0.00],
            dtype=float,
        ),
        physical_radius=0.025,
        safety_margin=0.015,
    )


def make_current_pose(
    target: SphericalStructure,
) -> CameraPose:
    position = np.array(
        [0.00, 0.00, 0.00],
        dtype=float,
    )

    return CameraPose(
        position=position,
        rotation=look_at_rotation(
            position,
            target.centre,
        ),
    )


def make_controller() -> ClosedLoopActivePerception:
    model = make_model()

    scorer = GenericViewpointScorer(
        observation_model=model,
        config=ViewpointScoringConfig(
            uncertainty_weight=1.0,
            movement_weight=0.05,
            occlusion_penalty=2.0,
            invisibility_penalty=4.0,
        ),
    )

    active = GenericActivePerception(
        observation_model=model,
        scorer=scorer,
    )

    return ClosedLoopActivePerception(
        observation_model=model,
        active_perception=active,
    )


def test_closed_loop_returns_selection() -> None:
    controller = make_controller()
    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre,
    )

    result = execute_reproducible_cycle(
        controller=controller,
        current_pose=current_pose,
        candidates=candidates,
        target=target,
        seed=42,
    )

    assert result.selection is not None


def test_closed_loop_returns_observation() -> None:
    controller = make_controller()
    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre,
    )

    result = execute_reproducible_cycle(
        controller=controller,
        current_pose=current_pose,
        candidates=candidates,
        target=target,
        seed=42,
    )

    assert result.observation is not None
    assert result.localisation_error >= 0.0


def test_selected_pose_matches_selection() -> None:
    controller = make_controller()
    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre,
    )

    result = execute_reproducible_cycle(
        controller=controller,
        current_pose=current_pose,
        candidates=candidates,
        target=target,
        seed=42,
    )

    np.testing.assert_allclose(
        result.selected_pose.position,
        result.selection.selected_viewpoint.pose.position,
        atol=1e-12,
    )


def test_selected_viewpoint_position_has_three_coordinates() -> None:
    controller = make_controller()
    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre,
    )

    result = execute_reproducible_cycle(
        controller=controller,
        current_pose=current_pose,
        candidates=candidates,
        target=target,
        seed=42,
    )

    position = selected_viewpoint_position(
        result
    )

    assert position.shape == (3,)


def test_post_observation_quality_is_available() -> None:
    controller = make_controller()
    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre,
    )

    result = execute_reproducible_cycle(
        controller=controller,
        current_pose=current_pose,
        candidates=candidates,
        target=target,
        seed=42,
    )

    assert (
        result.post_observation_quality
        .localisation_sigma
        > 0.0
    )


def test_uncertainty_improvement_matches_sigma_difference() -> None:
    controller = make_controller()
    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre,
    )

    result = execute_reproducible_cycle(
        controller=controller,
        current_pose=current_pose,
        candidates=candidates,
        target=target,
        seed=42,
    )

    expected = (
        result.selection.current_quality
        .localisation_sigma
        - result.post_observation_quality
        .localisation_sigma
    )

    assert uncertainty_improvement(
        result
    ) == pytest.approx(
        expected
    )


def test_uncertainty_reduction_fraction_is_finite() -> None:
    controller = make_controller()
    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre,
    )

    result = execute_reproducible_cycle(
        controller=controller,
        current_pose=current_pose,
        candidates=candidates,
        target=target,
        seed=42,
    )

    assert np.isfinite(
        uncertainty_reduction_fraction(
            result
        )
    )


def test_same_seed_produces_same_observation_error() -> None:
    controller = make_controller()
    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre,
    )

    result_a = execute_reproducible_cycle(
        controller=controller,
        current_pose=current_pose,
        candidates=candidates,
        target=target,
        seed=123,
    )

    result_b = execute_reproducible_cycle(
        controller=controller,
        current_pose=current_pose,
        candidates=candidates,
        target=target,
        seed=123,
    )

    assert result_a.localisation_error == pytest.approx(
        result_b.localisation_error
    )


def test_different_seeds_can_produce_different_observations() -> None:
    controller = make_controller()
    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre,
    )

    result_a = execute_reproducible_cycle(
        controller=controller,
        current_pose=current_pose,
        candidates=candidates,
        target=target,
        seed=1,
    )

    result_b = execute_reproducible_cycle(
        controller=controller,
        current_pose=current_pose,
        candidates=candidates,
        target=target,
        seed=999,
    )

    assert (
        result_a.localisation_error
        != result_b.localisation_error
    )


def test_empty_candidates_are_rejected() -> None:
    controller = make_controller()
    target = make_target()
    current_pose = make_current_pose(
        target
    )

    with pytest.raises(ValueError):
        execute_reproducible_cycle(
            controller=controller,
            current_pose=current_pose,
            candidates=(),
            target=target,
            seed=42,
        )