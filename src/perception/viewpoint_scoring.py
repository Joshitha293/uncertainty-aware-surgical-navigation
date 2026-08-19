"""Utility-based scoring for active surgical camera viewpoints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry.workspace import SphericalStructure
from src.perception.camera import CameraPose
from src.perception.observation import (
    ObservationQuality,
    ViewpointObservationModel,
)
from src.perception.viewpoints import (
    CandidateViewpoint,
    viewpoint_displacement,
)


@dataclass(frozen=True)
class ViewpointScoringConfig:
    """Weights controlling generic viewpoint utility."""

    uncertainty_weight: float = 1.0
    movement_weight: float = 0.10
    occlusion_penalty: float = 2.0
    invisibility_penalty: float = 4.0

    def __post_init__(self) -> None:
        if self.uncertainty_weight < 0.0:
            raise ValueError(
                "uncertainty_weight must be non-negative."
            )

        if self.movement_weight < 0.0:
            raise ValueError(
                "movement_weight must be non-negative."
            )

        if self.occlusion_penalty < 0.0:
            raise ValueError(
                "occlusion_penalty must be non-negative."
            )

        if self.invisibility_penalty < 0.0:
            raise ValueError(
                "invisibility_penalty must be non-negative."
            )


@dataclass(frozen=True)
class ViewpointScore:
    """Score and diagnostic information for one candidate viewpoint."""

    candidate: CandidateViewpoint
    quality: ObservationQuality
    movement_cost: float
    score: float


class GenericViewpointScorer:
    """Score camera viewpoints using generic perception quality.

    The scorer intentionally does not use surgical trajectory information,
    planned path geometry or task-specific risk. This preserves a clean
    generic active-perception baseline for later comparison against the
    task-aware strategy.
    """

    def __init__(
        self,
        observation_model: ViewpointObservationModel,
        config: ViewpointScoringConfig | None = None,
    ) -> None:
        self.observation_model = observation_model

        if config is None:
            config = ViewpointScoringConfig()

        self.config = config

    def score_candidate(
        self,
        current_pose: CameraPose,
        candidate: CandidateViewpoint,
        target: SphericalStructure,
        occluders: tuple[
            SphericalStructure,
            ...
        ] = (),
    ) -> ViewpointScore:
        """Calculate utility for one candidate viewpoint."""

        quality = (
            self.observation_model.observation_quality(
                camera_pose=candidate.pose,
                structure=target,
                occluders=occluders,
            )
        )

        movement_cost = viewpoint_displacement(
            current_pose=current_pose,
            candidate_pose=candidate.pose,
        )

        score = (
            -self.config.uncertainty_weight
            * quality.localisation_sigma
        )

        score -= (
            self.config.movement_weight
            * movement_cost
        )

        if quality.occluded:
            score -= (
                self.config.occlusion_penalty
            )

        if not quality.visible:
            score -= (
                self.config.invisibility_penalty
            )

        return ViewpointScore(
            candidate=candidate,
            quality=quality,
            movement_cost=movement_cost,
            score=float(score),
        )

    def score_candidates(
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
    ) -> tuple[ViewpointScore, ...]:
        """Score all candidate viewpoints."""

        if len(candidates) == 0:
            raise ValueError(
                "candidates must not be empty."
            )

        return tuple(
            self.score_candidate(
                current_pose=current_pose,
                candidate=candidate,
                target=target,
                occluders=occluders,
            )
            for candidate in candidates
        )


def select_best_viewpoint(
    scores: tuple[
        ViewpointScore,
        ...
    ],
) -> ViewpointScore:
    """Return the highest-utility viewpoint."""

    if len(scores) == 0:
        raise ValueError(
            "scores must not be empty."
        )

    return max(
        scores,
        key=lambda item: item.score,
    )


def score_improvement(
    current_quality: ObservationQuality,
    candidate_quality: ObservationQuality,
) -> float:
    """Return reduction in predicted localisation uncertainty."""

    return float(
        current_quality.localisation_sigma
        - candidate_quality.localisation_sigma
    )


def relative_uncertainty_reduction(
    current_quality: ObservationQuality,
    candidate_quality: ObservationQuality,
) -> float:
    """Return fractional uncertainty reduction."""

    current_sigma = (
        current_quality.localisation_sigma
    )

    if current_sigma <= 0.0:
        raise ValueError(
            "current localisation uncertainty "
            "must be positive."
        )

    return float(
        (
            current_sigma
            - candidate_quality.localisation_sigma
        )
        / current_sigma
    )


def rank_viewpoints(
    scores: tuple[
        ViewpointScore,
        ...
    ],
) -> tuple[ViewpointScore, ...]:
    """Return viewpoints ordered from highest to lowest utility."""

    if len(scores) == 0:
        raise ValueError(
            "scores must not be empty."
        )

    return tuple(
        sorted(
            scores,
            key=lambda item: item.score,
            reverse=True,
        )
    )