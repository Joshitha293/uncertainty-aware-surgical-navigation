"""Task-aware viewpoint scoring for safety-critical surgical perception."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry.workspace import SphericalStructure
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
from src.perception.viewpoints import CandidateViewpoint


@dataclass(frozen=True)
class TaskAwareScoringConfig:
    """Configuration for task-aware viewpoint scoring."""

    task_weight: float = 2.0

    alignment_weight: float = 1.0

    uncertainty_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.task_weight < 0.0:
            raise ValueError(
                "task_weight must be non-negative."
            )

        if self.alignment_weight < 0.0:
            raise ValueError(
                "alignment_weight must be non-negative."
            )

        if self.uncertainty_weight < 0.0:
            raise ValueError(
                "uncertainty_weight must be non-negative."
            )


@dataclass(frozen=True)
class TaskAwareViewpointScore:
    """Score and task-relevance diagnostics for one viewpoint."""

    candidate: CandidateViewpoint

    generic_score: ViewpointScore

    task_relevance: float

    task_alignment: float

    task_uncertainty: float

    score: float


class TaskAwareViewpointScorer:
    """Score viewpoints using task-specific uncertainty.

    The generic perception score remains the baseline.

    The task-aware contribution estimates how much uncertainty remains
    around safety-critical regions associated with the surgical task.
    Candidate viewpoints are therefore rewarded when they reduce
    uncertainty where the planned task actually matters.
    """

    def __init__(
        self,
        generic_scorer: GenericViewpointScorer,
        task: SurgicalTask,
        task_config: TaskAwareScoringConfig | None = None,
        relevance_config: TaskRelevanceConfig | None = None,
    ) -> None:
        self.generic_scorer = generic_scorer
        self.task = task

        if task_config is None:
            task_config = TaskAwareScoringConfig()

        if relevance_config is None:
            relevance_config = TaskRelevanceConfig()

        self.task_config = task_config
        self.relevance_config = relevance_config

        self._weights = task_relevance_weights(
            task=task,
            config=relevance_config,
        )

    def _task_alignment(
        self,
        pose: CameraPose,
    ) -> float:
        """Measure how well a camera faces relevant task regions."""

        camera_position = np.asarray(
            pose.position,
            dtype=float,
        )

        forward = np.asarray(
            pose.forward,
            dtype=float,
        )

        forward_norm = np.linalg.norm(
            forward
        )

        if forward_norm <= 0.0:
            raise ValueError(
                "Camera forward vector must be non-zero."
            )

        forward = (
            forward
            / forward_norm
        )

        alignments: list[float] = []

        for point in self.task.safety_critical_points:
            direction = (
                np.asarray(
                    point,
                    dtype=float,
                )
                - camera_position
            )

            direction_norm = np.linalg.norm(
                direction
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
                weights=self._weights,
            )
        )

        return float(
            0.5
            * (
                weighted_alignment
                + 1.0
            )
        )

    def _task_relevance(
        self,
    ) -> float:
        """Return mean relevance across critical regions."""

        return float(
            np.mean(
                self._weights
            )
        )

    def _candidate_task_uncertainty(
        self,
        candidate: CandidateViewpoint,
        target: SphericalStructure,
        occluders: tuple[
            SphericalStructure,
            ...,
        ],
    ) -> float:
        """Estimate candidate uncertainty for the task-critical region.

        The observation model provides a viewpoint-dependent localisation
        uncertainty for the target. This uncertainty is weighted by the
        relevance of the safety-critical task regions.

        The target-level observation is used as the simulated proxy for
        uncertainty around the safety-critical structure.
        """

        quality = (
            self.generic_scorer
            .observation_model
            .observation_quality(
                camera_pose=candidate.pose,
                structure=target,
                occluders=occluders,
            )
        )

        target_sigma = float(
            quality.localisation_sigma
        )

        relevance = self._task_relevance()

        return float(
            target_sigma
            * relevance
        )

    def score_candidate(
        self,
        current_pose: CameraPose,
        candidate: CandidateViewpoint,
        target: SphericalStructure,
        occluders: tuple[
            SphericalStructure,
            ...,
        ] = (),
    ) -> TaskAwareViewpointScore:
        """Score one candidate using task-specific uncertainty."""

        generic_score = (
            self.generic_scorer.score_candidate(
                current_pose=current_pose,
                candidate=candidate,
                target=target,
                occluders=occluders,
            )
        )

        relevance = self._task_relevance()

        alignment = self._task_alignment(
            candidate.pose
        )

        task_uncertainty = (
            self._candidate_task_uncertainty(
                candidate=candidate,
                target=target,
                occluders=occluders,
            )
        )

        task_information = (
            self.task_config.alignment_weight
            * relevance
            * alignment
        )

        uncertainty_information = (
            self.task_config.uncertainty_weight
            * (
                1.0
                / (
                    task_uncertainty
                    + 1e-12
                )
            )
        )

        score = (
            generic_score.score
            + self.task_config.task_weight
            * (
                task_information
                + uncertainty_information
            )
        )

        return TaskAwareViewpointScore(
            candidate=candidate,
            generic_score=generic_score,
            task_relevance=float(
                relevance
            ),
            task_alignment=float(
                alignment
            ),
            task_uncertainty=float(
                task_uncertainty
            ),
            score=float(
                score
            ),
        )

    def score_candidates(
        self,
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
        TaskAwareViewpointScore,
        ...,
    ]:
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


def select_best_task_aware_viewpoint(
    scores: tuple[
        TaskAwareViewpointScore,
        ...,
    ],
) -> TaskAwareViewpointScore:
    """Return the highest-scoring task-aware viewpoint."""

    if len(scores) == 0:
        raise ValueError(
            "scores must not be empty."
        )

    return max(
        scores,
        key=lambda item: item.score,
    )


def rank_task_aware_viewpoints(
    scores: tuple[
        TaskAwareViewpointScore,
        ...,
    ],
) -> tuple[
    TaskAwareViewpointScore,
    ...,
]:
    """Rank viewpoints from highest to lowest utility."""

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