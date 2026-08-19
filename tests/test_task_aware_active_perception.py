"""Tests for automatic task-aware active perception."""

"""Tests for automatic task-aware active perception."""

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
)
from src.perception.task_aware_active_perception import (
    TaskAwareActivePerception,
    candidate_score_array,
    selected_generic_score,
    selected_task_aware_position,
    selected_task_score,
    selection_changed_from_generic,
    task_aware_score_improvement,
)
from src.perception.task_aware_scoring import (
    TaskAwareScoringConfig,
    TaskAwareViewpointScorer,
)
from src.perception.task_relevance import (
    SurgicalTask,
)
from src.perception.viewpoint_scoring import (
    GenericViewpointScorer,
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
            [0.24, 0.0, 0.0],
            dtype=float,
        ),
        physical_radius=0.025,
        safety_margin=0.015,
    )


def make_task() -> SurgicalTask:
    trajectory = np.array(
        [
            [0.10, -0.05, 0.0],
            [0.15, 0.0, 0.0],
            [0.20, 0.05, 0.0],
        ],
        dtype=float,
    )

    critical_points = np.array(
        [
            [0.15, 0.0, 0.0],
            [0.16, 0.01, 0.0],
            [0.17, -0.01, 0.0],
        ],
        dtype=float,
    )

    return SurgicalTask(
        trajectory=trajectory,
        safety_critical_points=critical_points,
    )


def make_current_pose(
    target: SphericalStructure,
) -> CameraPose:
    position = np.array(
        [0.0, 0.0, 0.0],
        dtype=float,
    )

    return CameraPose(
        position=position,
        rotation=look_at_rotation(
            position,
            target.centre,
        ),
    )


def make_controller(
    task_weight: float = 2.0,
) -> TaskAwareActivePerception:
    model = make_model()

    generic = GenericViewpointScorer(
        observation_model=model
    )

    scorer = TaskAwareViewpointScorer(
        generic_scorer=generic,
        task=make_task(),
        task_config=TaskAwareScoringConfig(
            task_weight=task_weight,
            alignment_weight=1.0,
        ),
    )

    return TaskAwareActivePerception(
        scorer=scorer
    )


def test_controller_can_be_created() -> None:
    controller = make_controller()

    assert controller.task is not None
    assert controller.scorer is not None


def test_selection_returns_candidate() -> None:
    controller = make_controller()

    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre
    )

    result = controller.select_viewpoint(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    selected_position = (
        result.selected_position
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
        target_position=target.centre
    )

    result = controller.select_viewpoint(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    assert (
        result.candidate_count
        == len(candidates)
    )


def test_selected_position_has_correct_shape() -> None:
    controller = make_controller()

    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre
    )

    result = controller.select_viewpoint(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    position = selected_task_aware_position(
        result
    )

    assert position.shape == (3,)


def test_selected_position_is_copy() -> None:
    controller = make_controller()

    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre
    )

    result = controller.select_viewpoint(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    first = selected_task_aware_position(
        result
    )

    first[0] += 1.0

    second = selected_task_aware_position(
        result
    )

    assert not np.isclose(
        first[0],
        second[0],
    )


def test_selected_task_score_is_finite() -> None:
    controller = make_controller()

    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre
    )

    result = controller.select_viewpoint(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    assert np.isfinite(
        selected_task_score(result)
    )


def test_selected_generic_score_is_finite() -> None:
    controller = make_controller()

    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre
    )

    result = controller.select_viewpoint(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    assert np.isfinite(
        selected_generic_score(result)
    )


def test_task_aware_score_contribution_is_finite() -> None:
    controller = make_controller()

    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre
    )

    result = controller.select_viewpoint(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    assert np.isfinite(
        task_aware_score_improvement(
            result
        )
    )


def test_task_relevance_is_bounded() -> None:
    controller = make_controller()

    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre
    )

    result = controller.select_viewpoint(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    assert 0.0 <= result.task_relevance <= 1.0


def test_task_alignment_is_bounded() -> None:
    controller = make_controller()

    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre
    )

    result = controller.select_viewpoint(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    assert 0.0 <= result.task_alignment <= 1.0


def test_empty_candidates_are_rejected() -> None:
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


def test_zero_task_weight_matches_generic_score() -> None:
    controller = make_controller(
        task_weight=0.0
    )

    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre
    )

    result = controller.select_viewpoint(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    assert selected_task_score(
        result
    ) == pytest.approx(
        selected_generic_score(
            result
        )
    )


def test_task_aware_selection_is_deterministic() -> None:
    controller = make_controller()

    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre
    )

    first = controller.select_viewpoint(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    second = controller.select_viewpoint(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    np.testing.assert_allclose(
        first.selected_position,
        second.selected_position,
        atol=1e-12,
    )

    assert selected_task_score(
        first
    ) == pytest.approx(
        selected_task_score(
            second
        )
    )


def test_candidate_score_array_returns_correct_shape() -> None:
    controller = make_controller()

    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre
    )

    scores = controller.scorer.score_candidates(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    values = candidate_score_array(
        scores
    )

    assert values.shape == (
        len(candidates),
    )


def test_empty_score_array_is_supported() -> None:
    values = candidate_score_array(
        ()
    )

    assert values.shape == (0,)
    assert values.dtype == float


def test_selection_changed_function_detects_difference() -> None:
    controller = make_controller()

    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre
    )

    result = controller.select_viewpoint(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    different_position = (
        result.selected_position
        + np.array(
            [0.01, 0.0, 0.0]
        )
    )

    assert selection_changed_from_generic(
        task_aware_result=result,
        generic_position=different_position,
    )


def test_selection_changed_function_detects_same_position() -> None:
    controller = make_controller()

    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre
    )

    result = controller.select_viewpoint(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    assert not selection_changed_from_generic(
        task_aware_result=result,
        generic_position=result.selected_position,
    )


def test_negative_tolerance_is_rejected() -> None:
    controller = make_controller()

    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre
    )

    result = controller.select_viewpoint(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    with pytest.raises(ValueError):
        selection_changed_from_generic(
            task_aware_result=result,
            generic_position=np.zeros(3),
            tolerance=-1.0,
        )


def test_invalid_generic_position_shape_is_rejected() -> None:
    controller = make_controller()

    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre
    )

    result = controller.select_viewpoint(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    with pytest.raises(ValueError):
        selection_changed_from_generic(
            task_aware_result=result,
            generic_position=np.zeros(2),
        )