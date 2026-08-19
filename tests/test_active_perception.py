"""Tests for generic active-perception viewpoint selection."""

import numpy as np
import pytest

from src.geometry.workspace import SphericalStructure
from src.perception.active_perception import (
    GenericActivePerception,
    candidate_score_array,
    selected_viewpoint_improves_uncertainty,
    selected_viewpoint_is_unoccluded,
    selected_viewpoint_is_visible,
)
from src.perception.camera import (
    CameraIntrinsics,
    CameraPose,
    SurgicalCamera,
    look_at_rotation,
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


def make_occluder() -> SphericalStructure:
    return SphericalStructure(
        centre=np.array(
            [0.12, 0.00, 0.00],
            dtype=float,
        ),
        physical_radius=0.035,
        safety_margin=0.010,
    )


def make_controller(
    movement_weight: float = 0.05,
) -> GenericActivePerception:
    model = make_model()

    scorer = GenericViewpointScorer(
        observation_model=model,
        config=ViewpointScoringConfig(
            uncertainty_weight=1.0,
            movement_weight=movement_weight,
            occlusion_penalty=2.0,
            invisibility_penalty=4.0,
        ),
    )

    return GenericActivePerception(
        observation_model=model,
        scorer=scorer,
    )


def test_current_view_can_be_evaluated() -> None:
    controller = make_controller()
    target = make_target()
    current_pose = make_current_pose(
        target
    )

    quality = controller.evaluate_current_view(
        current_pose=current_pose,
        target=target,
    )

    assert quality.visible
    assert not quality.occluded
    assert quality.localisation_sigma > 0.0


def test_selection_returns_one_of_the_candidates() -> None:
    controller = make_controller()
    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre,
    )

    result = controller.select_viewpoint(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    selected_position = (
        result.selected_viewpoint
        .pose.position
    )

    assert any(
        np.allclose(
            selected_position,
            candidate.pose.position,
            atol=1e-12,
        )
        for candidate in candidates
    )


def test_candidate_count_is_preserved() -> None:
    controller = make_controller()
    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre,
    )

    result = controller.select_viewpoint(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    assert result.candidate_count == len(
        candidates
    )


def test_selected_viewpoint_is_visible() -> None:
    controller = make_controller()
    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre,
    )

    result = controller.select_viewpoint(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    assert selected_viewpoint_is_visible(
        result
    )


def test_empty_candidate_set_is_rejected() -> None:
    controller = make_controller()
    target = make_target()
    current_pose = make_current_pose(
        target
    )

    with pytest.raises(ValueError):
        controller.select_viewpoint(
            current_pose=current_pose,
            candidates=(),
            target=target,
        )


def test_occlusion_drives_selection_toward_better_viewpoint() -> None:
    controller = make_controller()
    target = make_target()
    current_pose = make_current_pose(
        target
    )
    occluder = make_occluder()

    candidates = generate_candidate_viewpoints(
        target_position=target.centre,
    )

    result = controller.select_viewpoint(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
        occluders=(occluder,),
    )

    assert selected_viewpoint_is_visible(
        result
    )

    assert selected_viewpoint_is_unoccluded(
        result
    )


def test_information_only_selection_reduces_uncertainty() -> None:
    """With movement cost disabled, selection should optimise information."""

    controller = make_controller(
        movement_weight=0.0
    )

    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre,
    )

    result = controller.select_viewpoint(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    assert (
        result.selected_score
        .quality.localisation_sigma
        <= result.current_quality
        .localisation_sigma
    )


def test_uncertainty_improvement_flag_matches_result() -> None:
    controller = make_controller()
    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre,
    )

    result = controller.select_viewpoint(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    expected = (
        result.selected_score
        .quality.localisation_sigma
        < result.current_quality
        .localisation_sigma
    )

    assert (
        selected_viewpoint_improves_uncertainty(
            result
        )
        == expected
    )


def test_relative_reduction_is_finite() -> None:
    controller = make_controller()
    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre,
    )

    result = controller.select_viewpoint(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    assert np.isfinite(
        result.predicted_relative_uncertainty_reduction
    )


def test_absolute_uncertainty_reduction_matches_difference() -> None:
    controller = make_controller()
    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre,
    )

    result = controller.select_viewpoint(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    expected = (
        result.current_quality
        .localisation_sigma
        - result.selected_score
        .quality.localisation_sigma
    )

    assert (
        result.predicted_uncertainty_reduction
        == pytest.approx(
            expected
        )
    )


def test_selection_is_deterministic() -> None:
    controller = make_controller()
    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre,
    )

    result_a = controller.select_viewpoint(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    result_b = controller.select_viewpoint(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    np.testing.assert_allclose(
        result_a.selected_viewpoint
        .pose.position,
        result_b.selected_viewpoint
        .pose.position,
        atol=1e-12,
    )

    assert (
        result_a.selected_score.score
        == pytest.approx(
            result_b.selected_score.score
        )
    )


def test_selected_viewpoint_has_highest_score() -> None:
    controller = make_controller()
    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre,
    )

    scores = controller.scorer.score_candidates(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    result = controller.select_viewpoint(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    maximum_score = max(
        score.score
        for score in scores
    )

    assert (
        result.selected_score.score
        == pytest.approx(
            maximum_score
        )
    )


def test_candidate_score_array_has_correct_shape() -> None:
    controller = make_controller()
    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre,
    )

    scores = controller.scorer.score_candidates(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    score_array = candidate_score_array(
        scores
    )

    assert score_array.shape == (
        len(candidates),
    )


def test_empty_score_array_is_supported() -> None:
    result = candidate_score_array(
        ()
    )

    assert result.shape == (
        0,
    )

    assert result.dtype == float


def test_selected_viewpoint_is_unoccluded_when_occluder_exists() -> None:
    controller = make_controller()
    target = make_target()
    current_pose = make_current_pose(
        target
    )
    occluder = make_occluder()

    candidates = generate_candidate_viewpoints(
        target_position=target.centre,
    )

    result = controller.select_viewpoint(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
        occluders=(occluder,),
    )

    assert not result.selected_score.quality.occluded


def test_selected_viewpoint_distance_is_non_negative() -> None:
    controller = make_controller()
    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre,
    )

    result = controller.select_viewpoint(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    assert (
        result.selected_score.movement_cost
        >= 0.0
    )