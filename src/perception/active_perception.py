"""Generic active-perception controller for surgical camera viewpoints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry.workspace import SphericalStructure
from src.perception.camera import CameraPose
from src.perception.observation import (
    ObservationQuality,
    ViewpointObservationModel,
)
from src.perception.viewpoint_scoring import (
    GenericViewpointScorer,
    ViewpointScore,
    relative_uncertainty_reduction,
    select_best_viewpoint,
)
from src.perception.viewpoints import CandidateViewpoint


@dataclass(frozen=True)
class ViewpointSelectionResult:
    """Result returned by generic active perception."""

    selected_viewpoint: CandidateViewpoint
    selected_score: ViewpointScore
    current_quality: ObservationQuality
    predicted_uncertainty_reduction: float
    predicted_relative_uncertainty_reduction: float
    candidate_count: int


class GenericActivePerception:
    """Select informative camera viewpoints without task information.

    The controller deliberately uses only generic observation quality and
    camera movement cost. It does not use the surgical trajectory,
    collision geometry or task-specific risk.
    """

    def __init__(
        self,
        observation_model: ViewpointObservationModel,
        scorer: GenericViewpointScorer | None = None,
    ) -> None:
        self.observation_model = observation_model

        if scorer is None:
            scorer = GenericViewpointScorer(
                observation_model=observation_model
            )

        self.scorer = scorer

    def evaluate_current_view(
        self,
        current_pose: CameraPose,
        target: SphericalStructure,
        occluders: tuple[
            SphericalStructure,
            ...
        ] = (),
    ) -> ObservationQuality:
        """Evaluate the quality of the current camera viewpoint."""

        return self.observation_model.observation_quality(
            camera_pose=current_pose,
            structure=target,
            occluders=occluders,
        )

    def select_viewpoint(
        self,
        current_pose: CameraPose,
        candidates: tuple[
            CandidateViewpoint,
            ...
        ],
        target: SphericalStructure,
        occluders: tuple[
            SphericalStructure,
            ...
        ] = (),
    ) -> ViewpointSelectionResult:
        """Select the highest-utility candidate viewpoint."""

        if len(candidates) == 0:
            raise ValueError(
                "candidates must not be empty."
            )

        current_quality = (
            self.evaluate_current_view(
                current_pose=current_pose,
                target=target,
                occluders=occluders,
            )
        )

        scores = self.scorer.score_candidates(
            current_pose=current_pose,
            candidates=candidates,
            target=target,
            occluders=occluders,
        )

        selected_score = (
            select_best_viewpoint(
                scores
            )
        )

        selected_viewpoint = (
            selected_score.candidate
        )

        uncertainty_reduction = (
            current_quality.localisation_sigma
            - selected_score.quality.localisation_sigma
        )

        relative_reduction = (
            relative_uncertainty_reduction(
                current_quality=current_quality,
                candidate_quality=(
                    selected_score.quality
                ),
            )
        )

        return ViewpointSelectionResult(
            selected_viewpoint=selected_viewpoint,
            selected_score=selected_score,
            current_quality=current_quality,
            predicted_uncertainty_reduction=float(
                uncertainty_reduction
            ),
            predicted_relative_uncertainty_reduction=float(
                relative_reduction
            ),
            candidate_count=len(
                candidates
            ),
        )


def selected_viewpoint_improves_uncertainty(
    result: ViewpointSelectionResult,
) -> bool:
    """Return whether the selected viewpoint is predicted to improve uncertainty."""

    return bool(
        result.selected_score.quality.localisation_sigma
        < result.current_quality.localisation_sigma
    )


def selected_viewpoint_is_visible(
    result: ViewpointSelectionResult,
) -> bool:
    """Return whether the selected viewpoint sees the target."""

    return bool(
        result.selected_score.quality.visible
    )


def selected_viewpoint_is_unoccluded(
    result: ViewpointSelectionResult,
) -> bool:
    """Return whether the selected viewpoint has no geometric occlusion."""

    return bool(
        not result.selected_score.quality.occluded
    )


def candidate_score_array(
    result_scores: tuple[
        ViewpointScore,
        ...
    ],
) -> np.ndarray:
    """Return candidate utility scores as a NumPy array."""

    if len(result_scores) == 0:
        return np.empty(
            (0,),
            dtype=float,
        )

    return np.asarray(
        [
            score.score
            for score in result_scores
        ],
        dtype=float,
    )