"""Normalised task-aware viewpoint scoring.

This module provides a scale-controlled task-aware viewpoint objective for
fair experimental comparison.

The original task-aware scorer is intentionally retained for historical
reproducibility.  This implementation is used for the examiner-proof
fair-baseline evaluation.

Candidate-level continuous quantities are normalised across the common
candidate set before weighted combination.  This prevents quantities with
different units or numerical ranges from dominating the final objective
purely because of scale.

The final score combines:

- generic perception utility;
- task alignment;
- uncertainty information;
- camera movement cost.

All continuous comparison terms are mapped to [0, 1] before combination.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry.workspace import (
    SphericalStructure,
)
from src.perception.camera import CameraPose
from src.perception.task_relevance import (
    SurgicalTask,
    TaskRelevanceConfig,
    task_relevance_weights,
)
from src.perception.viewpoint_scoring import (
    GenericViewpointScorer,
    ViewpointScore,
)
from src.perception.viewpoints import (
    CandidateViewpoint,
)


@dataclass(frozen=True)
class NormalisedTaskAwareConfig:
    """Weights for scale-controlled task-aware scoring."""

    generic_weight: float = 1.0

    alignment_weight: float = 1.0

    uncertainty_weight: float = 1.0

    movement_weight: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.generic_weight,
            self.alignment_weight,
            self.uncertainty_weight,
            self.movement_weight,
        )

        if not all(
            np.isfinite(
                value
            )
            for value in values
        ):
            raise ValueError(
                "All scoring weights must be finite."
            )

        if any(
            value < 0.0
            for value in values
        ):
            raise ValueError(
                "All scoring weights must be non-negative."
            )


@dataclass(frozen=True)
class NormalisedTaskAwareScore:
    """Normalised score and diagnostics for one candidate."""

    candidate: CandidateViewpoint

    generic_score: ViewpointScore

    task_relevance: float

    task_alignment: float

    predicted_sigma: float

    movement_cost: float

    normalised_generic_utility: float

    normalised_uncertainty_information: float

    normalised_movement_cost: float

    final_score: float


def _normalise_higher_is_better(
    values: np.ndarray,
) -> np.ndarray:
    """Normalise values to [0, 1] with larger values preferred."""

    values = np.asarray(
        values,
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError(
            "values must be one-dimensional."
        )

    if values.size == 0:
        raise ValueError(
            "values must not be empty."
        )

    if not np.all(
        np.isfinite(
            values
        )
    ):
        raise ValueError(
            "values must all be finite."
        )

    minimum = float(
        np.min(
            values
        )
    )

    maximum = float(
        np.max(
            values
        )
    )

    span = (
        maximum
        - minimum
    )

    if span <= 1e-12:
        return np.ones_like(
            values,
            dtype=float,
        )

    return (
        values
        - minimum
    ) / span


def _normalise_lower_is_better(
    values: np.ndarray,
) -> np.ndarray:
    """Normalise values to [0, 1] with smaller values preferred."""

    values = np.asarray(
        values,
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError(
            "values must be one-dimensional."
        )

    if values.size == 0:
        raise ValueError(
            "values must not be empty."
        )

    if not np.all(
        np.isfinite(
            values
        )
    ):
        raise ValueError(
            "values must all be finite."
        )

    minimum = float(
        np.min(
            values
        )
    )

    maximum = float(
        np.max(
            values
        )
    )

    span = (
        maximum
        - minimum
    )

    if span <= 1e-12:
        return np.ones_like(
            values,
            dtype=float,
        )

    return (
        maximum
        - values
    ) / span


class NormalisedTaskAwareViewpointScorer:
    """Score candidate viewpoints using comparable normalised terms."""

    def __init__(
        self,
        *,
        generic_scorer: GenericViewpointScorer,
        task: SurgicalTask,
        config: NormalisedTaskAwareConfig
        | None = None,
        relevance_config: TaskRelevanceConfig
        | None = None,
    ) -> None:
        self.generic_scorer = (
            generic_scorer
        )

        self.task = task

        if config is None:
            config = (
                NormalisedTaskAwareConfig()
            )

        if relevance_config is None:
            relevance_config = (
                TaskRelevanceConfig()
            )

        self.config = config

        self.relevance_config = (
            relevance_config
        )

        self._relevance_weights = (
            task_relevance_weights(
                task=task,
                config=relevance_config,
            )
        )

    def _task_relevance(
        self,
    ) -> float:
        """Return mean relevance of safety-critical task regions."""

        return float(
            np.mean(
                self._relevance_weights
            )
        )

    def _task_alignment(
        self,
        pose: CameraPose,
    ) -> float:
        """Calculate weighted alignment with task-critical regions."""

        camera_position = np.asarray(
            pose.position,
            dtype=float,
        )

        forward = np.asarray(
            pose.forward,
            dtype=float,
        )

        forward_norm = float(
            np.linalg.norm(
                forward
            )
        )

        if forward_norm <= 0.0:
            raise ValueError(
                "Camera forward vector must be non-zero."
            )

        forward = (
            forward
            / forward_norm
        )

        alignments: list[
            float
        ] = []

        for point in (
            self.task
            .safety_critical_points
        ):
            direction = (
                np.asarray(
                    point,
                    dtype=float,
                )
                - camera_position
            )

            direction_norm = float(
                np.linalg.norm(
                    direction
                )
            )

            if direction_norm <= 0.0:
                alignment = 1.0
            else:
                direction = (
                    direction
                    / direction_norm
                )

                alignment = float(
                    np.dot(
                        forward,
                        direction,
                    )
                )

            alignments.append(
                float(
                    np.clip(
                        alignment,
                        -1.0,
                        1.0,
                    )
                )
            )

        weighted_alignment = float(
            np.average(
                np.asarray(
                    alignments,
                    dtype=float,
                ),
                weights=(
                    self._relevance_weights
                ),
            )
        )

        return float(
            0.5
            * (
                weighted_alignment
                + 1.0
            )
        )

    def score_candidates(
        self,
        *,
        current_pose: CameraPose,
        candidates: tuple[
            CandidateViewpoint,
            ...,
        ],
        target: SphericalStructure,
        occluders: tuple[
            SphericalStructure,
            ...,
        ] = (),
    ) -> tuple[
        NormalisedTaskAwareScore,
        ...,
    ]:
        """Score a common candidate set using scale-controlled terms."""

        if len(
            candidates
        ) == 0:
            raise ValueError(
                "candidates must not be empty."
            )

        generic_scores = (
            self.generic_scorer
            .score_candidates(
                current_pose=current_pose,
                candidates=candidates,
                target=target,
                occluders=occluders,
            )
        )

        raw_generic = np.asarray(
            [
                score.score
                for score
                in generic_scores
            ],
            dtype=float,
        )

        sigmas = np.asarray(
            [
                score
                .quality
                .localisation_sigma
                for score
                in generic_scores
            ],
            dtype=float,
        )

        movement = np.asarray(
            [
                score.movement_cost
                for score
                in generic_scores
            ],
            dtype=float,
        )

        alignments = np.asarray(
            [
                self._task_alignment(
                    candidate.pose
                )
                for candidate
                in candidates
            ],
            dtype=float,
        )

        normalised_generic = (
            _normalise_higher_is_better(
                raw_generic
            )
        )

        uncertainty_information = (
            _normalise_lower_is_better(
                sigmas
            )
        )

        normalised_movement = (
            _normalise_higher_is_better(
                movement
            )
        )

        relevance = (
            self._task_relevance()
        )

        results: list[
            NormalisedTaskAwareScore
        ] = []

        for index, candidate in enumerate(
            candidates
        ):
            final_score = (
                self.config.generic_weight
                * normalised_generic[
                    index
                ]
            )

            final_score += (
                self.config.alignment_weight
                * relevance
                * alignments[
                    index
                ]
            )

            final_score += (
                self.config.uncertainty_weight
                * relevance
                * uncertainty_information[
                    index
                ]
            )

            final_score -= (
                self.config.movement_weight
                * normalised_movement[
                    index
                ]
            )

            results.append(
                NormalisedTaskAwareScore(
                    candidate=candidate,
                    generic_score=(
                        generic_scores[
                            index
                        ]
                    ),
                    task_relevance=float(
                        relevance
                    ),
                    task_alignment=float(
                        alignments[
                            index
                        ]
                    ),
                    predicted_sigma=float(
                        sigmas[
                            index
                        ]
                    ),
                    movement_cost=float(
                        movement[
                            index
                        ]
                    ),
                    normalised_generic_utility=float(
                        normalised_generic[
                            index
                        ]
                    ),
                    normalised_uncertainty_information=float(
                        uncertainty_information[
                            index
                        ]
                    ),
                    normalised_movement_cost=float(
                        normalised_movement[
                            index
                        ]
                    ),
                    final_score=float(
                        final_score
                    ),
                )
            )

        return tuple(
            results
        )

    def select_viewpoint(
        self,
        *,
        current_pose: CameraPose,
        candidates: tuple[
            CandidateViewpoint,
            ...,
        ],
        target: SphericalStructure,
        occluders: tuple[
            SphericalStructure,
            ...,
        ] = (),
    ) -> NormalisedTaskAwareScore:
        """Return the highest-scoring candidate."""

        scores = (
            self.score_candidates(
                current_pose=current_pose,
                candidates=candidates,
                target=target,
                occluders=occluders,
            )
        )

        return max(
            scores,
            key=lambda item:
            item.final_score,
        )


def rank_normalised_task_aware_viewpoints(
    scores: tuple[
        NormalisedTaskAwareScore,
        ...,
    ],
) -> tuple[
    NormalisedTaskAwareScore,
    ...,
]:
    """Rank normalised task-aware scores from highest to lowest."""

    if len(
        scores
    ) == 0:
        raise ValueError(
            "scores must not be empty."
        )

    return tuple(
        sorted(
            scores,
            key=lambda item:
            item.final_score,
            reverse=True,
        )
    )