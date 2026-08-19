"""Automatic task-aware active perception for surgical navigation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry.workspace import SphericalStructure
from src.perception.camera import CameraPose
from src.perception.task_aware_scoring import (
    TaskAwareViewpointScore,
    TaskAwareViewpointScorer,
    select_best_task_aware_viewpoint,
)
from src.perception.task_relevance import SurgicalTask
from src.perception.viewpoints import CandidateViewpoint


@dataclass(frozen=True)
class TaskAwareSelectionResult:
    """Result returned by task-aware viewpoint selection."""

    selected_viewpoint: CandidateViewpoint

    selected_score: TaskAwareViewpointScore

    candidate_count: int

    selected_position: np.ndarray

    task_relevance: float

    task_alignment: float

    task_uncertainty: float


class TaskAwareActivePerception:
    """Select viewpoints using task-relevant perception utility.

    Unlike the generic active-perception controller, this controller
    incorporates the surgical trajectory and safety-critical regions
    when selecting the camera viewpoint.
    """

    def __init__(
        self,
        scorer: TaskAwareViewpointScorer,
    ) -> None:
        self.scorer = scorer

    @property
    def task(self) -> SurgicalTask:
        """Return the surgical task used by the scorer."""

        return self.scorer.task

    def select_viewpoint(
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
    ) -> TaskAwareSelectionResult:
        """Select the highest-utility task-aware viewpoint."""

        if len(candidates) == 0:
            raise ValueError(
                "candidates must not be empty."
            )

        scores = self.scorer.score_candidates(
            current_pose=current_pose,
            candidates=candidates,
            target=target,
            occluders=occluders,
        )

        selected_score = (
            select_best_task_aware_viewpoint(
                scores
            )
        )

        selected_viewpoint = (
            selected_score.candidate
        )

        selected_position = np.asarray(
            selected_viewpoint.pose.position,
            dtype=float,
        ).copy()

        return TaskAwareSelectionResult(
            selected_viewpoint=selected_viewpoint,
            selected_score=selected_score,
            candidate_count=len(
                candidates
            ),
            selected_position=selected_position,
            task_relevance=float(
                selected_score.task_relevance
            ),
            task_alignment=float(
                selected_score.task_alignment
            ),
            task_uncertainty=float(
                selected_score.task_uncertainty
            ),
        )


def selected_task_aware_position(
    result: TaskAwareSelectionResult,
) -> np.ndarray:
    """Return a copy of the selected camera position."""

    return np.asarray(
        result.selected_position,
        dtype=float,
    ).copy()


def selected_task_score(
    result: TaskAwareSelectionResult,
) -> float:
    """Return the selected task-aware utility."""

    return float(
        result.selected_score.score
    )


def selected_generic_score(
    result: TaskAwareSelectionResult,
) -> float:
    """Return the underlying generic score."""

    return float(
        result.selected_score.generic_score.score
    )


def task_aware_score_improvement(
    result: TaskAwareSelectionResult,
) -> float:
    """Return the task-aware utility contribution beyond the generic score."""

    return float(
        result.selected_score.score
        - result.selected_score.generic_score.score
    )


def selection_changed_from_generic(
    task_aware_result: TaskAwareSelectionResult,
    generic_position: np.ndarray,
    tolerance: float = 1e-10,
) -> bool:
    """Return whether task-aware selection differs from a generic selection."""

    if tolerance < 0.0:
        raise ValueError(
            "tolerance must be non-negative."
        )

    generic_position = np.asarray(
        generic_position,
        dtype=float,
    )

    if generic_position.shape != (3,):
        raise ValueError(
            "generic_position must have shape (3,)."
        )

    return bool(
        not np.allclose(
            task_aware_result.selected_position,
            generic_position,
            atol=tolerance,
            rtol=0.0,
        )
    )


def candidate_score_array(
    scores: tuple[
        TaskAwareViewpointScore,
        ...,
    ],
) -> np.ndarray:
    """Return task-aware candidate scores as a NumPy array."""

    if len(scores) == 0:
        return np.empty(
            (0,),
            dtype=float,
        )

    return np.asarray(
        [
            score.score
            for score in scores
        ],
        dtype=float,
    )