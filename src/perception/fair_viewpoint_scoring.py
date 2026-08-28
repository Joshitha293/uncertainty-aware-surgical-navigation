"""Fair, scale-controlled viewpoint scoring for baseline comparison.

This module defines a candidate-set scoring formulation intended for the
examiner-proof active-perception comparison.

The objective deliberately operates on primitive, non-duplicated quantities.

Generic active perception uses:

    perception information
    - camera movement cost

Task-aware active perception uses exactly the same quantities plus:

    task-relevance-weighted alignment

Predicted localisation sigma is treated as the single perception-quality
quantity because the observation model already encodes distance, viewing
angle, visibility and occlusion in that uncertainty estimate.

This avoids:

- unit-scale domination;
- double-counting uncertainty;
- double-counting camera movement;
- separately penalising effects already represented by sigma.

The original project scorers are retained separately for reproducibility of
historical experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry.workspace import SphericalStructure
from src.perception.camera import CameraPose
from src.perception.observation import (
    ObservationQuality,
    ViewpointObservationModel,
)
from src.perception.task_relevance import (
    SurgicalTask,
    TaskRelevanceConfig,
    task_relevance_weights,
)
from src.perception.viewpoints import (
    CandidateViewpoint,
    viewpoint_displacement,
)


@dataclass(frozen=True)
class FairScoringConfig:
    """Weights for the fair active-perception objective."""

    perception_weight: float = 1.0
    movement_weight: float = 0.10
    alignment_weight: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.perception_weight,
            self.movement_weight,
            self.alignment_weight,
        )

        if not all(
            np.isfinite(value)
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
class FairCandidateScore:
    """Fair-score diagnostics for one candidate viewpoint."""

    candidate: CandidateViewpoint
    quality: ObservationQuality

    movement_cost: float

    task_alignment: float

    information_score: float
    normalised_movement_cost: float

    valid_observation: bool

    generic_score: float
    task_aware_score: float


@dataclass(frozen=True)
class FairSelection:
    """Selected candidate and complete candidate diagnostics."""

    selected: FairCandidateScore
    scores: tuple[
        FairCandidateScore,
        ...,
    ]


def normalise_information(
    sigmas: np.ndarray,
) -> np.ndarray:
    """Convert localisation uncertainty into information in [0, 1].

    Smaller predicted uncertainty represents better perception and therefore
    receives a larger information score.

    When all candidates have identical sigma, every candidate receives the
    same information score of 1.0. The term then contributes no ranking
    difference.
    """

    values = np.asarray(
        sigmas,
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError(
            "sigmas must be one-dimensional."
        )

    if values.size == 0:
        raise ValueError(
            "sigmas must not be empty."
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "sigmas must all be finite."
        )

    if np.any(
        values <= 0.0
    ):
        raise ValueError(
            "sigmas must all be positive."
        )

    minimum = float(
        np.min(values)
    )

    maximum = float(
        np.max(values)
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


def normalise_cost(
    costs: np.ndarray,
) -> np.ndarray:
    """Normalise a non-negative cost to [0, 1].

    Smaller values represent lower cost.

    A value of zero corresponds to the cheapest candidate in the supplied
    candidate set and one corresponds to the most expensive.

    If all candidates have the same cost, all receive zero because there is
    no relative cost difference to penalise.
    """

    values = np.asarray(
        costs,
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError(
            "costs must be one-dimensional."
        )

    if values.size == 0:
        raise ValueError(
            "costs must not be empty."
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "costs must all be finite."
        )

    if np.any(
        values < 0.0
    ):
        raise ValueError(
            "costs must all be non-negative."
        )

    minimum = float(
        np.min(values)
    )

    maximum = float(
        np.max(values)
    )

    span = (
        maximum
        - minimum
    )

    if span <= 1e-12:
        return np.zeros_like(
            values,
            dtype=float,
        )

    return (
        values
        - minimum
    ) / span


class FairViewpointScorer:
    """Construct Generic and Task-Aware scores from common primitive terms."""

    def __init__(
        self,
        *,
        observation_model: ViewpointObservationModel,
        task: SurgicalTask,
        config: FairScoringConfig | None = None,
        relevance_config: TaskRelevanceConfig | None = None,
    ) -> None:
        self.observation_model = (
            observation_model
        )

        self.task = task

        if config is None:
            config = FairScoringConfig()

        if relevance_config is None:
            relevance_config = (
                TaskRelevanceConfig()
            )

        self.config = config

        self.relevance_config = (
            relevance_config
        )

        self._task_weights = (
            task_relevance_weights(
                task=task,
                config=relevance_config,
            )
        )

    def _task_alignment(
        self,
        pose: CameraPose,
    ) -> float:
        """Return task-relevance-weighted alignment in [0, 1]."""

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

            alignment = float(
                np.clip(
                    alignment,
                    -1.0,
                    1.0,
                )
            )

            alignments.append(
                alignment
            )

        weighted_alignment = float(
            np.average(
                np.asarray(
                    alignments,
                    dtype=float,
                ),
                weights=(
                    self._task_weights
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
        FairCandidateScore,
        ...,
    ]:
        """Score the common candidate set for both strategies."""

        if len(candidates) == 0:
            raise ValueError(
                "candidates must not be empty."
            )

        qualities = tuple(
            self.observation_model
            .observation_quality(
                camera_pose=(
                    candidate.pose
                ),
                structure=target,
                occluders=occluders,
            )
            for candidate in candidates
        )

        sigmas = np.asarray(
            [
                quality.localisation_sigma
                for quality in qualities
            ],
            dtype=float,
        )

        movements = np.asarray(
            [
                viewpoint_displacement(
                    current_pose=current_pose,
                    candidate_pose=(
                        candidate.pose
                    ),
                )
                for candidate in candidates
            ],
            dtype=float,
        )

        alignments = np.asarray(
            [
                self._task_alignment(
                    candidate.pose
                )
                for candidate in candidates
            ],
            dtype=float,
        )

        information = (
            normalise_information(
                sigmas
            )
        )

        movement_costs = (
            normalise_cost(
                movements
            )
        )

        results: list[
            FairCandidateScore
        ] = []

        for index, candidate in enumerate(
            candidates
        ):
            quality = qualities[
                index
            ]

            valid_observation = bool(
                quality.visible
                and not quality.occluded
            )

            generic_score = (
                self.config
                .perception_weight
                * information[
                    index
                ]
                - self.config
                .movement_weight
                * movement_costs[
                    index
                ]
            )

            task_aware_score = (
                generic_score
                + self.config
                .alignment_weight
                * alignments[
                    index
                ]
            )

            results.append(
                FairCandidateScore(
                    candidate=candidate,
                    quality=quality,
                    movement_cost=float(
                        movements[
                            index
                        ]
                    ),
                    task_alignment=float(
                        alignments[
                            index
                        ]
                    ),
                    information_score=float(
                        information[
                            index
                        ]
                    ),
                    normalised_movement_cost=float(
                        movement_costs[
                            index
                        ]
                    ),
                    valid_observation=(
                        valid_observation
                    ),
                    generic_score=float(
                        generic_score
                    ),
                    task_aware_score=float(
                        task_aware_score
                    ),
                )
            )

        return tuple(
            results
        )

    @staticmethod
    def _selection_pool(
        scores: tuple[
            FairCandidateScore,
            ...,
        ],
    ) -> tuple[
        FairCandidateScore,
        ...,
    ]:
        """Prefer candidates providing valid unoccluded observations.

        If no valid candidate exists, retain the complete candidate set so the
        experiment remains defined rather than failing silently.
        """

        valid = tuple(
            score
            for score in scores
            if score.valid_observation
        )

        if len(valid) > 0:
            return valid

        return scores

    def select_generic(
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
    ) -> FairSelection:
        """Select the best task-agnostic candidate."""

        scores = self.score_candidates(
            current_pose=current_pose,
            candidates=candidates,
            target=target,
            occluders=occluders,
        )

        pool = self._selection_pool(
            scores
        )

        selected = max(
            pool,
            key=lambda score:
            score.generic_score,
        )

        return FairSelection(
            selected=selected,
            scores=scores,
        )

    def select_task_aware(
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
    ) -> FairSelection:
        """Select the best task-aware candidate."""

        scores = self.score_candidates(
            current_pose=current_pose,
            candidates=candidates,
            target=target,
            occluders=occluders,
        )

        pool = self._selection_pool(
            scores
        )

        selected = max(
            pool,
            key=lambda score:
            score.task_aware_score,
        )

        return FairSelection(
            selected=selected,
            scores=scores,
        )