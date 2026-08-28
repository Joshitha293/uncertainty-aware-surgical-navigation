"""Strictly separated fair scene-wide viewpoint-selection strategies.

This module provides the final Phase 1 viewpoint-selection interfaces.

Generic Scene-Wide Active Perception
------------------------------------
Receives:

- the observation model;
- the current camera pose;
- the common candidate set;
- the shared noisy initial anatomical estimate.

It does NOT receive:

- the task trajectory;
- task relevance;
- true anatomy;
- true target coordinates;
- true occluders.

Task-Aware Active Perception
----------------------------
Receives the same estimated scene and candidate set, plus the known planned
instrument trajectory.

The two strategies may use independently frozen movement weights so that
Generic can be movement-budget matched to Task-Aware.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.perception.fair_scene_viewpoint_scoring import (
    _estimated_scene,
    _normalise_information_matrix,
    _normalise_movement,
    _relevance_weights,
    _validate_trajectory,
    _weighted_alignment,
)
from src.perception.observation import (
    ViewpointObservationModel,
)
from src.perception.perception import (
    PerceptionResult,
)
from src.perception.task_relevance import (
    TaskRelevanceConfig,
)
from src.perception.viewpoints import (
    CandidateViewpoint,
    viewpoint_displacement,
)
from src.perception.camera import (
    CameraPose,
)


@dataclass(frozen=True)
class GenericSceneScoringConfig:
    """Configuration for task-agnostic scene-wide scoring."""

    perception_weight: float = 1.0
    movement_weight: float = 0.10

    def __post_init__(self) -> None:
        values = (
            self.perception_weight,
            self.movement_weight,
        )

        if not all(
            np.isfinite(value)
            for value in values
        ):
            raise ValueError(
                "Generic scoring weights must be finite."
            )

        if any(
            value < 0.0
            for value in values
        ):
            raise ValueError(
                "Generic scoring weights must be non-negative."
            )


@dataclass(frozen=True)
class TaskAwareSceneScoringConfig:
    """Configuration for task-aware scene scoring."""

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
                "Task-aware scoring weights must be finite."
            )

        if any(
            value < 0.0
            for value in values
        ):
            raise ValueError(
                "Task-aware scoring weights must be non-negative."
            )


@dataclass(frozen=True)
class GenericSceneCandidateScore:
    """Task-agnostic diagnostics for one candidate."""

    candidate: CandidateViewpoint

    predicted_sigmas: tuple[
        float,
        ...,
    ]

    information_scores: tuple[
        float,
        ...,
    ]

    movement_cost: float
    normalised_movement_cost: float

    scene_information: float

    score: float


@dataclass(frozen=True)
class TaskAwareSceneCandidateScore:
    """Task-aware diagnostics for one candidate."""

    candidate: CandidateViewpoint

    predicted_sigmas: tuple[
        float,
        ...,
    ]

    information_scores: tuple[
        float,
        ...,
    ]

    relevance_weights: tuple[
        float,
        ...,
    ]

    movement_cost: float
    normalised_movement_cost: float

    task_scene_information: float

    task_alignment: float

    score: float


@dataclass(frozen=True)
class GenericSceneSelection:
    """Selected Generic candidate."""

    selected: GenericSceneCandidateScore

    scores: tuple[
        GenericSceneCandidateScore,
        ...,
    ]


@dataclass(frozen=True)
class TaskAwareSceneSelection:
    """Selected Task-Aware candidate."""

    selected: TaskAwareSceneCandidateScore

    scores: tuple[
        TaskAwareSceneCandidateScore,
        ...,
    ]


@dataclass(frozen=True)
class SceneScoringPrimitives:
    """Task-independent primitive candidate quantities."""

    predicted_sigmas: np.ndarray

    information_scores: np.ndarray

    movement_costs: np.ndarray

    normalised_movement_costs: np.ndarray


def _calculate_scene_primitives(
    *,
    observation_model: ViewpointObservationModel,
    current_pose: CameraPose,
    candidates: tuple[
        CandidateViewpoint,
        ...,
    ],
    initial_perception: PerceptionResult,
) -> SceneScoringPrimitives:
    """Calculate task-independent candidate quantities.

    No task trajectory or simulator ground truth enters this function.
    """

    if len(
        candidates
    ) == 0:
        raise ValueError(
            "candidates must not be empty."
        )

    estimated_structures = (
        _estimated_scene(
            initial_perception
        )
    )

    predicted_sigmas = np.asarray(
        [
            [
                observation_model
                .observation_quality(
                    camera_pose=(
                        candidate.pose
                    ),
                    structure=structure,
                )
                .localisation_sigma

                for structure
                in estimated_structures
            ]

            for candidate
            in candidates
        ],
        dtype=float,
    )

    information_scores = (
        _normalise_information_matrix(
            predicted_sigmas
        )
    )

    movement_costs = np.asarray(
        [
            viewpoint_displacement(
                current_pose=(
                    current_pose
                ),
                candidate_pose=(
                    candidate.pose
                ),
            )

            for candidate
            in candidates
        ],
        dtype=float,
    )

    normalised_movement_costs = (
        _normalise_movement(
            movement_costs
        )
    )

    return SceneScoringPrimitives(
        predicted_sigmas=(
            predicted_sigmas
        ),
        information_scores=(
            information_scores
        ),
        movement_costs=(
            movement_costs
        ),
        normalised_movement_costs=(
            normalised_movement_costs
        ),
    )


class GenericSceneViewpointScorer:
    """Task-agnostic scene-wide active-perception scorer."""

    def __init__(
        self,
        *,
        observation_model: ViewpointObservationModel,
        config: GenericSceneScoringConfig
        | None = None,
    ) -> None:
        self.observation_model = (
            observation_model
        )

        if config is None:
            config = (
                GenericSceneScoringConfig()
            )

        self.config = config

    def score_candidates(
        self,
        *,
        current_pose: CameraPose,
        candidates: tuple[
            CandidateViewpoint,
            ...,
        ],
        initial_perception: PerceptionResult,
    ) -> tuple[
        GenericSceneCandidateScore,
        ...,
    ]:
        """Score all estimated structures uniformly."""

        primitives = (
            _calculate_scene_primitives(
                observation_model=(
                    self.observation_model
                ),
                current_pose=(
                    current_pose
                ),
                candidates=candidates,
                initial_perception=(
                    initial_perception
                ),
            )
        )

        results = []

        for index, candidate in enumerate(
            candidates
        ):
            information = (
                primitives
                .information_scores[
                    index
                ]
            )

            scene_information = float(
                np.mean(
                    information
                )
            )

            score = (
                self.config
                .perception_weight
                * scene_information
                - self.config
                .movement_weight
                * primitives
                .normalised_movement_costs[
                    index
                ]
            )

            results.append(
                GenericSceneCandidateScore(
                    candidate=candidate,
                    predicted_sigmas=tuple(
                        float(value)
                        for value
                        in primitives
                        .predicted_sigmas[
                            index
                        ]
                    ),
                    information_scores=tuple(
                        float(value)
                        for value
                        in information
                    ),
                    movement_cost=float(
                        primitives
                        .movement_costs[
                            index
                        ]
                    ),
                    normalised_movement_cost=float(
                        primitives
                        .normalised_movement_costs[
                            index
                        ]
                    ),
                    scene_information=float(
                        scene_information
                    ),
                    score=float(
                        score
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
        initial_perception: PerceptionResult,
    ) -> GenericSceneSelection:
        """Select the highest-scoring task-agnostic viewpoint."""

        scores = (
            self.score_candidates(
                current_pose=current_pose,
                candidates=candidates,
                initial_perception=(
                    initial_perception
                ),
            )
        )

        selected = max(
            scores,
            key=lambda item:
            item.score,
        )

        return GenericSceneSelection(
            selected=selected,
            scores=scores,
        )


class TaskAwareSceneViewpointScorer:
    """Task-aware scene-wide active-perception scorer."""

    def __init__(
        self,
        *,
        observation_model: ViewpointObservationModel,
        task_trajectory: np.ndarray,
        config: TaskAwareSceneScoringConfig
        | None = None,
        relevance_config: TaskRelevanceConfig
        | None = None,
    ) -> None:
        self.observation_model = (
            observation_model
        )

        self.task_trajectory = (
            _validate_trajectory(
                task_trajectory
            )
        )

        if config is None:
            config = (
                TaskAwareSceneScoringConfig()
            )

        if relevance_config is None:
            relevance_config = (
                TaskRelevanceConfig()
            )

        self.config = config

        self.relevance_config = (
            relevance_config
        )

    def score_candidates(
        self,
        *,
        current_pose: CameraPose,
        candidates: tuple[
            CandidateViewpoint,
            ...,
        ],
        initial_perception: PerceptionResult,
    ) -> tuple[
        TaskAwareSceneCandidateScore,
        ...,
    ]:
        """Score estimated anatomy according to task relevance."""

        primitives = (
            _calculate_scene_primitives(
                observation_model=(
                    self.observation_model
                ),
                current_pose=(
                    current_pose
                ),
                candidates=candidates,
                initial_perception=(
                    initial_perception
                ),
            )
        )

        estimated_structures = (
            _estimated_scene(
                initial_perception
            )
        )

        relevance = (
            _relevance_weights(
                structures=(
                    estimated_structures
                ),
                trajectory=(
                    self.task_trajectory
                ),
                config=(
                    self.relevance_config
                ),
            )
        )

        relevance_sum = float(
            np.sum(
                relevance
            )
        )

        results = []

        for index, candidate in enumerate(
            candidates
        ):
            information = (
                primitives
                .information_scores[
                    index
                ]
            )

            if relevance_sum <= 1e-12:
                task_information = float(
                    np.mean(
                        information
                    )
                )

            else:
                task_information = float(
                    np.average(
                        information,
                        weights=relevance,
                    )
                )

            alignment = (
                _weighted_alignment(
                    pose=(
                        candidate.pose
                    ),
                    structures=(
                        estimated_structures
                    ),
                    relevance_weights=(
                        relevance
                    ),
                )
            )

            score = (
                self.config
                .perception_weight
                * task_information
                + self.config
                .alignment_weight
                * alignment
                - self.config
                .movement_weight
                * primitives
                .normalised_movement_costs[
                    index
                ]
            )

            results.append(
                TaskAwareSceneCandidateScore(
                    candidate=candidate,
                    predicted_sigmas=tuple(
                        float(value)
                        for value
                        in primitives
                        .predicted_sigmas[
                            index
                        ]
                    ),
                    information_scores=tuple(
                        float(value)
                        for value
                        in information
                    ),
                    relevance_weights=tuple(
                        float(value)
                        for value
                        in relevance
                    ),
                    movement_cost=float(
                        primitives
                        .movement_costs[
                            index
                        ]
                    ),
                    normalised_movement_cost=float(
                        primitives
                        .normalised_movement_costs[
                            index
                        ]
                    ),
                    task_scene_information=float(
                        task_information
                    ),
                    task_alignment=float(
                        alignment
                    ),
                    score=float(
                        score
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
        initial_perception: PerceptionResult,
    ) -> TaskAwareSceneSelection:
        """Select the highest-scoring Task-Aware viewpoint."""

        scores = (
            self.score_candidates(
                current_pose=current_pose,
                candidates=candidates,
                initial_perception=(
                    initial_perception
                ),
            )
        )

        selected = max(
            scores,
            key=lambda item:
            item.score,
        )

        return TaskAwareSceneSelection(
            selected=selected,
            scores=scores,
        )