"""Tests for generic active-perception viewpoint scoring."""

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
from src.perception.viewpoint_scoring import (
    GenericViewpointScorer,
    ViewpointScoringConfig,
    rank_viewpoints,
    relative_uncertainty_reduction,
    score_improvement,
    select_best_viewpoint,
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
            [0.20, 0.00, 0.00],
            dtype=float,
        ),
        physical_radius=0.025,
        safety_margin=0.015,
    )


def make_current_pose() -> CameraPose:
    position = np.array(
        [0.05, 0.00, 0.00],
        dtype=float,
    )

    target = make_target()

    return CameraPose(
        position=position,
        rotation=look_at_rotation(
            position,
            target.centre,
        ),
    )


def make_candidate(
    position: np.ndarray,
    target: np.ndarray | None = None,
) -> CandidateViewpoint:
    if target is None:
        target = make_target().centre

    position = np.asarray(
        position,
        dtype=float,
    )

    return CandidateViewpoint(
        pose=CameraPose(
            position=position,
            rotation=look_at_rotation(
                position,
                target,
            ),
        ),
        radius=float(
            np.linalg.norm(
                position - target
            )
        ),
        azimuth=0.0,
        elevation=0.0,
    )


def test_default_scoring_config_is_valid() -> None:
    config = ViewpointScoringConfig()

    assert config.uncertainty_weight >= 0.0
    assert config.movement_weight >= 0.0
    assert config.occlusion_penalty >= 0.0
    assert config.invisibility_penalty >= 0.0


def test_negative_scoring_weights_are_rejected() -> None:
    with pytest.raises(ValueError):
        ViewpointScoringConfig(
            uncertainty_weight=-1.0
        )

    with pytest.raises(ValueError):
        ViewpointScoringConfig(
            movement_weight=-1.0
        )

    with pytest.raises(ValueError):
        ViewpointScoringConfig(
            occlusion_penalty=-1.0
        )

    with pytest.raises(ValueError):
        ViewpointScoringConfig(
            invisibility_penalty=-1.0
        )


def test_candidate_score_is_finite() -> None:
    model = make_model()

    scorer = GenericViewpointScorer(
        observation_model=model
    )

    target = make_target()
    current_pose = make_current_pose()

    candidate = make_candidate(
        np.array(
            [0.05, 0.05, 0.00]
        )
    )

    result = scorer.score_candidate(
        current_pose=current_pose,
        candidate=candidate,
        target=target,
    )

    assert np.isfinite(
        result.score
    )

    assert np.isfinite(
        result.movement_cost
    )

    assert result.movement_cost >= 0.0


def test_clear_view_scores_better_than_occluded_view() -> None:
    model = make_model()

    scorer = GenericViewpointScorer(
        observation_model=model,
        config=ViewpointScoringConfig(
            uncertainty_weight=1.0,
            movement_weight=0.0,
            occlusion_penalty=2.0,
            invisibility_penalty=4.0,
        ),
    )

    target = make_target()
    current_pose = make_current_pose()

    clear_candidate = make_candidate(
        np.array(
            [0.05, 0.12, 0.00]
        )
    )

    occluded_candidate = make_candidate(
        np.array(
            [0.05, 0.00, 0.00]
        )
    )

    occluder = SphericalStructure(
        centre=np.array(
            [0.12, 0.00, 0.00]
        ),
        physical_radius=0.035,
        safety_margin=0.010,
    )

    clear_score = scorer.score_candidate(
        current_pose=current_pose,
        candidate=clear_candidate,
        target=target,
        occluders=(occluder,),
    )

    occluded_score = scorer.score_candidate(
        current_pose=current_pose,
        candidate=occluded_candidate,
        target=target,
        occluders=(occluder,),
    )

    assert not clear_score.quality.occluded
    assert occluded_score.quality.occluded

    assert (
        clear_score.score
        > occluded_score.score
    )


def test_movement_cost_can_change_viewpoint_ranking() -> None:
    model = make_model()

    scorer = GenericViewpointScorer(
        observation_model=model,
        config=ViewpointScoringConfig(
            uncertainty_weight=1.0,
            movement_weight=10.0,
            occlusion_penalty=0.0,
            invisibility_penalty=0.0,
        ),
    )

    target = make_target()
    current_pose = make_current_pose()

    nearby = make_candidate(
        np.array(
            [0.05, 0.03, 0.00]
        )
    )

    far = make_candidate(
        np.array(
            [-0.20, 0.10, 0.10]
        )
    )

    nearby_score = scorer.score_candidate(
        current_pose=current_pose,
        candidate=nearby,
        target=target,
    )

    far_score = scorer.score_candidate(
        current_pose=current_pose,
        candidate=far,
        target=target,
    )

    assert (
        nearby_score.movement_cost
        < far_score.movement_cost
    )


def test_score_candidates_preserves_candidate_count() -> None:
    model = make_model()

    scorer = GenericViewpointScorer(
        observation_model=model
    )

    target = make_target()
    current_pose = make_current_pose()

    candidates = generate_candidate_viewpoints(
        target_position=target.centre,
        config=None,
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
    model = make_model()

    scorer = GenericViewpointScorer(
        observation_model=model
    )

    with pytest.raises(ValueError):
        scorer.score_candidates(
            current_pose=make_current_pose(),
            candidates=(),
            target=make_target(),
        )


def test_best_viewpoint_has_highest_score() -> None:
    model = make_model()

    scorer = GenericViewpointScorer(
        observation_model=model,
        config=ViewpointScoringConfig(
            uncertainty_weight=1.0,
            movement_weight=0.0,
            occlusion_penalty=5.0,
            invisibility_penalty=5.0,
        ),
    )

    target = make_target()
    current_pose = make_current_pose()

    clear_candidate = make_candidate(
        np.array(
            [0.05, 0.12, 0.00]
        )
    )

    occluded_candidate = make_candidate(
        np.array(
            [0.05, 0.00, 0.00]
        )
    )

    occluder = SphericalStructure(
        centre=np.array(
            [0.12, 0.00, 0.00]
        ),
        physical_radius=0.035,
        safety_margin=0.010,
    )

    scores = scorer.score_candidates(
        current_pose=current_pose,
        candidates=(
            occluded_candidate,
            clear_candidate,
        ),
        target=target,
        occluders=(occluder,),
    )

    best = select_best_viewpoint(
        scores
    )

    assert (
        best.candidate
        == clear_candidate
    )


def test_select_best_viewpoint_rejects_empty_scores() -> None:
    with pytest.raises(ValueError):
        select_best_viewpoint(
            ()
        )


def test_rank_viewpoints_orders_descending() -> None:
    model = make_model()

    scorer = GenericViewpointScorer(
        observation_model=model,
        config=ViewpointScoringConfig(
            movement_weight=0.0,
        ),
    )

    target = make_target()
    current_pose = make_current_pose()

    candidates = generate_candidate_viewpoints(
        target_position=target.centre,
    )

    scores = scorer.score_candidates(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
    )

    ranked = rank_viewpoints(
        scores
    )

    assert len(ranked) == len(
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


def test_rank_viewpoints_rejects_empty_scores() -> None:
    with pytest.raises(ValueError):
        rank_viewpoints(
            ()
        )


def test_score_improvement_is_uncertainty_difference() -> None:
    model = make_model()
    target = make_target()
    current_pose = make_current_pose()

    current_candidate = make_candidate(
        np.array(
            [0.05, 0.00, 0.00]
        )
    )

    improved_candidate = make_candidate(
        np.array(
            [0.05, 0.12, 0.00]
        )
    )

    current_quality = (
        model.observation_quality(
            camera_pose=current_candidate.pose,
            structure=target,
        )
    )

    improved_quality = (
        model.observation_quality(
            camera_pose=improved_candidate.pose,
            structure=target,
        )
    )

    improvement = score_improvement(
        current_quality=current_quality,
        candidate_quality=improved_quality,
    )

    expected = (
        current_quality.localisation_sigma
        - improved_quality.localisation_sigma
    )

    assert improvement == pytest.approx(
        expected
    )


def test_relative_uncertainty_reduction_is_correct() -> None:
    model = make_model()
    target = make_target()

    current_candidate = make_candidate(
        np.array(
            [0.05, 0.00, 0.00]
        )
    )

    improved_candidate = make_candidate(
        np.array(
            [0.05, 0.12, 0.00]
        )
    )

    current_quality = (
        model.observation_model
        if False
        else model.observation_quality(
            camera_pose=current_candidate.pose,
            structure=target,
        )
    )

    improved_quality = (
        model.observation_quality(
            camera_pose=improved_candidate.pose,
            structure=target,
        )
    )

    reduction = (
        relative_uncertainty_reduction(
            current_quality=current_quality,
            candidate_quality=improved_quality,
        )
    )

    expected = (
        current_quality.localisation_sigma
        - improved_quality.localisation_sigma
    ) / current_quality.localisation_sigma

    assert reduction == pytest.approx(
        expected
    )


def test_relative_uncertainty_reduction_rejects_zero_sigma() -> None:
    from src.perception.observation import ObservationQuality

    current = ObservationQuality(
        visible=True,
        occluded=False,
        distance=0.15,
        off_axis_angle=0.0,
        localisation_sigma=0.0,
    )

    candidate = ObservationQuality(
        visible=True,
        occluded=False,
        distance=0.15,
        off_axis_angle=0.0,
        localisation_sigma=0.001,
    )

    with pytest.raises(ValueError):
        relative_uncertainty_reduction(
            current_quality=current,
            candidate_quality=candidate,
        )