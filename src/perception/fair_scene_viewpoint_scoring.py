"""Scene-wide fair viewpoint scoring using estimated anatomy only.

This module defines the strengthened Phase 1 active-perception comparison.

Both Generic and Task-Aware strategies operate on the same initial noisy
perception result.  Neither strategy receives ground-truth anatomical
positions or simulated occluder geometry during viewpoint selection.

Generic Active Perception
-------------------------
Scores candidate viewpoints according to mean expected perception
information across all currently estimated anatomical structures and camera
movement cost.

Task-Aware Active Perception
----------------------------
Uses the same candidate viewpoints, observation model, estimated anatomy and
movement cost, but additionally:

1. weights scene perception according to proximity of each estimated
   structure to the known planned instrument trajectory;
2. rewards camera alignment with task-relevant estimated anatomy.

Ground-truth anatomy remains the responsibility of the simulator and
evaluation layer only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry.workspace import (
    SphericalStructure,
)
from src.perception.camera import CameraPose
from src.perception.observation import (
    ViewpointObservationModel,
)
from src.perception.perception import (
    PerceptionResult,
)
from src.perception.task_relevance import (
    TaskRelevanceConfig,
    point_to_polyline_distance,
)
from src.perception.viewpoints import (
    CandidateViewpoint,
    viewpoint_displacement,
)


@dataclass(frozen=True)
class FairSceneScoringConfig:
    """Configuration for the fair scene-wide objective."""

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
class FairSceneCandidateScore:
    """Diagnostics for one candidate viewpoint."""

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

    generic_scene_information: float
    task_scene_information: float

    task_alignment: float

    generic_score: float
    task_aware_score: float


@dataclass(frozen=True)
class FairSceneSelection:
    """Selected viewpoint and all candidate diagnostics."""

    selected: FairSceneCandidateScore

    scores: tuple[
        FairSceneCandidateScore,
        ...,
    ]


def _validate_trajectory(
    trajectory: np.ndarray,
) -> np.ndarray:
    """Return a validated Cartesian task trajectory."""

    trajectory = np.asarray(
        trajectory,
        dtype=float,
    )

    if (
        trajectory.ndim != 2
        or trajectory.shape[1] != 3
    ):
        raise ValueError(
            "trajectory must have shape (N, 3)."
        )

    if len(
        trajectory
    ) < 2:
        raise ValueError(
            "trajectory must contain at least two points."
        )

    if not np.all(
        np.isfinite(
            trajectory
        )
    ):
        raise ValueError(
            "trajectory must contain finite values."
        )

    return trajectory


def _normalise_information_matrix(
    sigmas: np.ndarray,
) -> np.ndarray:
    """Normalise predicted uncertainty separately for each structure.

    Parameters
    ----------
    sigmas:
        Array with shape:

            (candidate_count, structure_count)

    Lower uncertainty corresponds to greater information.

    Each anatomical structure is normalised separately so that structures
    with different absolute uncertainty scales do not dominate the scene
    objective merely because of their numerical range.
    """

    sigmas = np.asarray(
        sigmas,
        dtype=float,
    )

    if sigmas.ndim != 2:
        raise ValueError(
            "sigmas must have shape "
            "(candidate_count, structure_count)."
        )

    if (
        sigmas.shape[0] == 0
        or sigmas.shape[1] == 0
    ):
        raise ValueError(
            "sigmas must not contain an empty dimension."
        )

    if not np.all(
        np.isfinite(
            sigmas
        )
    ):
        raise ValueError(
            "sigmas must contain finite values."
        )

    if np.any(
        sigmas <= 0.0
    ):
        raise ValueError(
            "sigmas must be positive."
        )

    minimum = np.min(
        sigmas,
        axis=0,
    )

    maximum = np.max(
        sigmas,
        axis=0,
    )

    span = (
        maximum
        - minimum
    )

    information = np.ones_like(
        sigmas,
        dtype=float,
    )

    informative = (
        span > 1e-12
    )

    if np.any(
        informative
    ):
        information[
            :,
            informative
        ] = (
            maximum[
                informative
            ]
            - sigmas[
                :,
                informative
            ]
        ) / span[
            informative
        ]

    return information


def _normalise_movement(
    movement: np.ndarray,
) -> np.ndarray:
    """Normalise camera displacement to a relative [0, 1] cost."""

    movement = np.asarray(
        movement,
        dtype=float,
    )

    if movement.ndim != 1:
        raise ValueError(
            "movement must be one-dimensional."
        )

    if movement.size == 0:
        raise ValueError(
            "movement must not be empty."
        )

    if not np.all(
        np.isfinite(
            movement
        )
    ):
        raise ValueError(
            "movement must contain finite values."
        )

    if np.any(
        movement < 0.0
    ):
        raise ValueError(
            "movement must be non-negative."
        )

    minimum = float(
        np.min(
            movement
        )
    )

    maximum = float(
        np.max(
            movement
        )
    )

    span = (
        maximum
        - minimum
    )

    if span <= 1e-12:
        return np.zeros_like(
            movement,
            dtype=float,
        )

    return (
        movement
        - minimum
    ) / span


def _estimated_scene(
    perception_result: PerceptionResult,
) -> tuple[
    SphericalStructure,
    ...,
]:
    """Convert planner-facing estimates into scoring geometry.

    Critically, structure centres come from ``estimated_centre`` rather than
    any simulator ground truth.
    """

    estimates = (
        perception_result
        .estimated_structures
    )

    if len(
        estimates
    ) == 0:
        raise ValueError(
            "perception_result must contain "
            "at least one estimated structure."
        )

    structures: list[
        SphericalStructure
    ] = []

    for estimate in estimates:
        centre = np.asarray(
            estimate.estimated_centre,
            dtype=float,
        )

        if centre.shape != (3,):
            raise ValueError(
                "estimated structure centre "
                "must have shape (3,)."
            )

        structures.append(
            SphericalStructure(
                centre=centre.copy(),
                physical_radius=float(
                    estimate.physical_radius
                ),
                safety_margin=float(
                    estimate.base_safety_margin
                ),
            )
        )

    return tuple(
        structures
    )


def _relevance_weights(
    *,
    structures: tuple[
        SphericalStructure,
        ...,
    ],
    trajectory: np.ndarray,
    config: TaskRelevanceConfig,
) -> np.ndarray:
    """Calculate task relevance from estimated anatomy only."""

    weights = []

    for structure in structures:
        distance = (
            point_to_polyline_distance(
                point=structure.centre,
                polyline=trajectory,
            )
        )

        relevance = np.exp(
            -0.5
            * (
                distance
                / config.relevance_sigma
            )
            ** 2
        )

        relevance = np.clip(
            relevance,
            config.minimum_relevance,
            config.maximum_relevance,
        )

        weights.append(
            float(
                relevance
            )
        )

    return np.asarray(
        weights,
        dtype=float,
    )


def _weighted_alignment(
    *,
    pose: CameraPose,
    structures: tuple[
        SphericalStructure,
        ...,
    ],
    relevance_weights: np.ndarray,
) -> float:
    """Return task-weighted camera-to-anatomy alignment in [0, 1]."""

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

    alignments = []

    for structure in structures:
        direction = (
            structure.centre
            - camera_position
        )

        direction_norm = float(
            np.linalg.norm(
                direction
            )
        )

        if direction_norm <= 0.0:
            cosine = 1.0

        else:
            direction = (
                direction
                / direction_norm
            )

            cosine = float(
                np.dot(
                    forward,
                    direction,
                )
            )

        cosine = float(
            np.clip(
                cosine,
                -1.0,
                1.0,
            )
        )

        alignments.append(
            0.5
            * (
                cosine
                + 1.0
            )
        )

    alignments_array = np.asarray(
        alignments,
        dtype=float,
    )

    weight_sum = float(
        np.sum(
            relevance_weights
        )
    )

    if weight_sum <= 1e-12:
        return float(
            np.mean(
                alignments_array
            )
        )

    return float(
        np.average(
            alignments_array,
            weights=relevance_weights,
        )
    )


class FairSceneViewpointScorer:
    """Score a common candidate set using estimated scene information."""

    def __init__(
        self,
        *,
        observation_model: ViewpointObservationModel,
        task_trajectory: np.ndarray,
        config: FairSceneScoringConfig | None = None,
        relevance_config: TaskRelevanceConfig | None = None,
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
                FairSceneScoringConfig()
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
        FairSceneCandidateScore,
        ...,
    ]:
        """Score candidates without access to simulator ground truth."""

        if len(
            candidates
        ) == 0:
            raise ValueError(
                "candidates must not be empty."
            )

        structures = (
            _estimated_scene(
                initial_perception
            )
        )

        relevance = (
            _relevance_weights(
                structures=structures,
                trajectory=(
                    self.task_trajectory
                ),
                config=(
                    self.relevance_config
                ),
            )
        )

        sigma_matrix = np.asarray(
            [
                [
                    self.observation_model
                    .observation_quality(
                        camera_pose=(
                            candidate.pose
                        ),
                        structure=structure,
                    )
                    .localisation_sigma

                    for structure
                    in structures
                ]

                for candidate
                in candidates
            ],
            dtype=float,
        )

        information_matrix = (
            _normalise_information_matrix(
                sigma_matrix
            )
        )

        movement = np.asarray(
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

        normalised_movement = (
            _normalise_movement(
                movement
            )
        )

        results: list[
            FairSceneCandidateScore
        ] = []

        relevance_sum = float(
            np.sum(
                relevance
            )
        )

        for index, candidate in enumerate(
            candidates
        ):
            candidate_information = (
                information_matrix[
                    index
                ]
            )

            generic_scene_information = float(
                np.mean(
                    candidate_information
                )
            )

            if relevance_sum <= 1e-12:
                task_scene_information = (
                    generic_scene_information
                )

            else:
                task_scene_information = float(
                    np.average(
                        candidate_information,
                        weights=relevance,
                    )
                )

            task_alignment = (
                _weighted_alignment(
                    pose=candidate.pose,
                    structures=structures,
                    relevance_weights=(
                        relevance
                    ),
                )
            )

            generic_score = (
                self.config
                .perception_weight
                * generic_scene_information
                - self.config
                .movement_weight
                * normalised_movement[
                    index
                ]
            )

            task_aware_score = (
                self.config
                .perception_weight
                * task_scene_information
                + self.config
                .alignment_weight
                * task_alignment
                - self.config
                .movement_weight
                * normalised_movement[
                    index
                ]
            )

            results.append(
                FairSceneCandidateScore(
                    candidate=candidate,
                    predicted_sigmas=tuple(
                        float(value)
                        for value
                        in sigma_matrix[
                            index
                        ]
                    ),
                    information_scores=tuple(
                        float(value)
                        for value
                        in candidate_information
                    ),
                    relevance_weights=tuple(
                        float(value)
                        for value
                        in relevance
                    ),
                    movement_cost=float(
                        movement[
                            index
                        ]
                    ),
                    normalised_movement_cost=float(
                        normalised_movement[
                            index
                        ]
                    ),
                    generic_scene_information=(
                        generic_scene_information
                    ),
                    task_scene_information=(
                        task_scene_information
                    ),
                    task_alignment=float(
                        task_alignment
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

    def select_generic(
        self,
        *,
        current_pose: CameraPose,
        candidates: tuple[
            CandidateViewpoint,
            ...,
        ],
        initial_perception: PerceptionResult,
    ) -> FairSceneSelection:
        """Select the highest-scoring scene-wide Generic viewpoint."""

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
            item.generic_score,
        )

        return FairSceneSelection(
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
        initial_perception: PerceptionResult,
    ) -> FairSceneSelection:
        """Select the highest-scoring task-aware viewpoint."""

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
            item.task_aware_score,
        )

        return FairSceneSelection(
            selected=selected,
            scores=scores,
        )