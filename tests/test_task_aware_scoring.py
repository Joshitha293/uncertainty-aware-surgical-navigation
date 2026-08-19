"""Tests for task-aware viewpoint scoring."""

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
from src.perception.task_aware_scoring import (
    TaskAwareScoringConfig,
    TaskAwareViewpointScorer,
    rank_task_aware_viewpoints,
    select_best_task_aware_viewpoint,
)
from src.perception.task_relevance import (
    SurgicalTask,
    TaskRelevanceConfig,
)
from src.perception.viewpoint_scoring import (
    GenericViewpointScorer,
)
from src.perception.viewpoints import (
    CandidateViewpoint,
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


def make_scorer() -> TaskAwareViewpointScorer:
    model = make_model()

    generic = GenericViewpointScorer(
        observation_model=model
    )

    return TaskAwareViewpointScorer(
        generic_scorer=generic,
        task=make_task(),
    )


def test_default_config_is_valid() -> None:
    config = TaskAwareScoringConfig()

    assert config.task_weight >= 0.0
    assert config.alignment_weight >= 0.0


def test_negative_task_weight_is_rejected() -> None:
    with pytest.raises(ValueError):
        TaskAwareScoringConfig(
            task_weight=-1.0
        )


def test_negative_alignment_weight_is_rejected() -> None:
    with pytest.raises(ValueError):
        TaskAwareScoringConfig(
            alignment_weight=-1.0
        )


def test_scorer_returns_finite_score() -> None:
    scorer = make_scorer()
    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidate = generate_candidate_viewpoints(
        target_position=target.centre,
        config=None,
    )[0]

    result = scorer.score_candidate(
        current_pose=current_pose,
        candidate=candidate,
        target=target,
    )

    assert np.isfinite(
        result.score
    )


def test_task_relevance_is_bounded() -> None:
    scorer = make_scorer()

    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidate = generate_candidate_viewpoints(
        target_position=target.centre,
    )[0]

    result = scorer.score_candidate(
        current_pose=current_pose,
        candidate=candidate,
        target=target,
    )

    assert 0.0 <= result.task_relevance <= 1.0


def test_task_alignment_is_bounded() -> None:
    scorer = make_scorer()

    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidate = generate_candidate_viewpoints(
        target_position=target.centre,
    )[0]

    result = scorer.score_candidate(
        current_pose=current_pose,
        candidate=candidate,
        target=target,
    )

    assert 0.0 <= result.task_alignment <= 1.0


def test_generic_score_is_preserved() -> None:
    model = make_model()

    generic = GenericViewpointScorer(
        observation_model=model
    )

    task_scorer = TaskAwareViewpointScorer(
        generic_scorer=generic,
        task=make_task(),
        task_config=TaskAwareScoringConfig(
            task_weight=0.0
        ),
    )

    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidate = generate_candidate_viewpoints(
        target_position=target.centre,
    )[0]

    generic_result = generic.score_candidate(
        current_pose=current_pose,
        candidate=candidate,
        target=target,
    )

    task_result = task_scorer.score_candidate(
        current_pose=current_pose,
        candidate=candidate,
        target=target,
    )

    assert task_result.score == pytest.approx(
        generic_result.score
    )


def test_task_weight_changes_score() -> None:
    model = make_model()

    generic = GenericViewpointScorer(
        observation_model=model
    )

    task = make_task()

    no_task = TaskAwareViewpointScorer(
        generic_scorer=generic,
        task=task,
        task_config=TaskAwareScoringConfig(
            task_weight=0.0
        ),
    )

    task_enabled = TaskAwareViewpointScorer(
        generic_scorer=generic,
        task=task,
        task_config=TaskAwareScoringConfig(
            task_weight=2.0
        ),
    )

    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidate = generate_candidate_viewpoints(
        target_position=target.centre,
    )[0]

    result_no_task = no_task.score_candidate(
        current_pose=current_pose,
        candidate=candidate,
        target=target,
    )

    result_task = task_enabled.score_candidate(
        current_pose=current_pose,
        candidate=candidate,
        target=target,
    )

    assert (
        result_task.score
        != pytest.approx(
            result_no_task.score
        )
    )


def test_score_candidates_preserves_count() -> None:
    scorer = make_scorer()

    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre,
    )

    scores = scorer.score_candidates(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    assert len(scores) == len(
        candidates
    )


def test_empty_candidates_are_rejected() -> None:
    scorer = make_scorer()

    target = make_target()
    current_pose = make_current_pose(
        target
    )

    with pytest.raises(ValueError):
        scorer.score_candidates(
            current_pose=current_pose,
            candidates=(),
            target=target,
        )


def test_best_task_aware_viewpoint_has_highest_score() -> None:
    scorer = make_scorer()

    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre,
    )

    scores = scorer.score_candidates(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    best = select_best_task_aware_viewpoint(
        scores
    )

    assert best.score == pytest.approx(
        max(
            score.score
            for score in scores
        )
    )


def test_empty_scores_are_rejected() -> None:
    with pytest.raises(ValueError):
        select_best_task_aware_viewpoint(
            ()
        )


def test_ranking_is_descending() -> None:
    scorer = make_scorer()

    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre,
    )

    scores = scorer.score_candidates(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    ranked = rank_task_aware_viewpoints(
        scores
    )

    for first, second in zip(
        ranked,
        ranked[1:],
    ):
        assert (
            first.score
            >= second.score
        )


def test_empty_ranking_is_rejected() -> None:
    with pytest.raises(ValueError):
        rank_task_aware_viewpoints(
            ()
        )


def test_task_score_contains_generic_score() -> None:
    scorer = make_scorer()

    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidate = generate_candidate_viewpoints(
        target_position=target.centre,
    )[0]

    result = scorer.score_candidate(
        current_pose=current_pose,
        candidate=candidate,
        target=target,
    )

    assert result.generic_score is not None


def test_task_aware_scoring_is_deterministic() -> None:
    scorer = make_scorer()

    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre,
    )

    first = scorer.score_candidates(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    second = scorer.score_candidates(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    for first_score, second_score in zip(
        first,
        second,
    ):
        assert first_score.score == pytest.approx(
            second_score.score
        )


def test_relevance_configuration_can_be_changed() -> None:
    model = make_model()

    generic = GenericViewpointScorer(
        observation_model=model
    )

    scorer = TaskAwareViewpointScorer(
        generic_scorer=generic,
        task=make_task(),
        relevance_config=TaskRelevanceConfig(
            relevance_sigma=0.01
        ),
    )

    target = make_target()
    current_pose = make_current_pose(
        target
    )

    candidate = generate_candidate_viewpoints(
        target_position=target.centre,
    )[0]

    result = scorer.score_candidate(
        current_pose=current_pose,
        candidate=candidate,
        target=target,
    )

    assert np.isfinite(
        result.task_relevance
    )