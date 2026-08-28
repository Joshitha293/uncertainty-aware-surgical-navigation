"""Additional Phase 1 viewpoint-selection baselines.

This module implements:

1. Random Active Perception
   A task-agnostic random candidate selection baseline with no access to
   anatomical ground truth.

2. Privileged Oracle
   An analysis-only reference that deliberately receives simulator ground
   truth. It uses the same candidate set and score structure as the fair
   Task-Aware method but evaluates expected viewpoint information using true
   anatomical geometry.

The oracle must never be presented as a deployable perception strategy. It
exists only to estimate how much performance is available under privileged
geometric information.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry.workspace import SphericalStructure
from src.perception.camera import CameraPose
from src.perception.fair_scene_viewpoint_scoring import (
    FairSceneScoringConfig,
    _normalise_information_matrix,
    _normalise_movement,
)
from src.perception.observation import (
    ViewpointObservationModel,
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
class RandomBaselineSelection:
    """Randomly selected viewpoint."""

    candidate: CandidateViewpoint

    candidate_index: int

    movement_cost: float

    random_seed: int


@dataclass(frozen=True)
class OracleCandidateScore:
    """Privileged score for one candidate viewpoint."""

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

    oracle_score: float


@dataclass(frozen=True)
class OracleSelection:
    """Privileged oracle viewpoint selection."""

    selected: OracleCandidateScore

    scores: tuple[
        OracleCandidateScore,
        ...,
    ]


def select_random_viewpoint(
    *,
    current_pose: CameraPose,
    candidates: tuple[
        CandidateViewpoint,
        ...,
    ],
    seed: int,
) -> RandomBaselineSelection:
    """Select uniformly from the common candidate set."""

    if len(
        candidates
    ) == 0:
        raise ValueError(
            "candidates must not be empty."
        )

    rng = np.random.default_rng(
        int(
            seed
        )
    )

    index = int(
        rng.integers(
            0,
            len(
                candidates
            ),
        )
    )

    candidate = candidates[
        index
    ]

    movement = float(
        viewpoint_displacement(
            current_pose=(
                current_pose
            ),
            candidate_pose=(
                candidate.pose
            ),
        )
    )

    return RandomBaselineSelection(
        candidate=candidate,
        candidate_index=index,
        movement_cost=movement,
        random_seed=int(
            seed
        ),
    )


def _validate_trajectory(
    trajectory: np.ndarray,
) -> np.ndarray:
    """Validate the known planned task trajectory."""

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


def _oracle_relevance_weights(
    *,
    true_structures: tuple[
        SphericalStructure,
        ...,
    ],
    trajectory: np.ndarray,
    config: TaskRelevanceConfig,
) -> np.ndarray:
    """Calculate relevance using privileged true anatomy."""

    if len(
        true_structures
    ) == 0:
        raise ValueError(
            "true_structures must not be empty."
        )

    values = []

    for structure in (
        true_structures
    ):
        distance = (
            point_to_polyline_distance(
                point=(
                    structure.centre
                ),
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

        values.append(
            float(
                relevance
            )
        )

    return np.asarray(
        values,
        dtype=float,
    )


def _oracle_alignment(
    *,
    pose: CameraPose,
    true_structures: tuple[
        SphericalStructure,
        ...,
    ],
    relevance: np.ndarray,
) -> float:
    """Calculate privileged task-weighted alignment."""

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

    for structure in (
        true_structures
    ):
        direction = (
            structure.centre
            - camera_position
        )

        direction_norm = float(
            np.linalg.norm(
                direction
            )
        )

        if direction_norm <= 1e-12:
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

    alignments = np.asarray(
        alignments,
        dtype=float,
    )

    relevance_sum = float(
        np.sum(
            relevance
        )
    )

    if relevance_sum <= 1e-12:
        return float(
            np.mean(
                alignments
            )
        )

    return float(
        np.average(
            alignments,
            weights=relevance,
        )
    )


def score_oracle_candidates(
    *,
    observation_model: ViewpointObservationModel,
    current_pose: CameraPose,
    candidates: tuple[
        CandidateViewpoint,
        ...,
    ],
    true_structures: tuple[
        SphericalStructure,
        ...,
    ],
    true_occluders: tuple[
        SphericalStructure,
        ...,
    ],
    task_trajectory: np.ndarray,
    config: FairSceneScoringConfig
    | None = None,
    relevance_config: TaskRelevanceConfig
    | None = None,
) -> tuple[
    OracleCandidateScore,
    ...,
]:
    """Score candidates using privileged simulator truth.

    Unlike the deployable Generic and Task-Aware selectors, this function
    intentionally receives true anatomy and true occluder geometry.
    """

    if len(
        candidates
    ) == 0:
        raise ValueError(
            "candidates must not be empty."
        )

    if len(
        true_structures
    ) == 0:
        raise ValueError(
            "true_structures must not be empty."
        )

    trajectory = (
        _validate_trajectory(
            task_trajectory
        )
    )

    if config is None:
        config = (
            FairSceneScoringConfig(
                movement_weight=0.200
            )
        )

    if relevance_config is None:
        relevance_config = (
            TaskRelevanceConfig()
        )

    relevance = (
        _oracle_relevance_weights(
            true_structures=(
                true_structures
            ),
            trajectory=trajectory,
            config=relevance_config,
        )
    )

    sigma_matrix = np.asarray(
        [
            [
                observation_model
                .observation_quality(
                    camera_pose=(
                        candidate.pose
                    ),
                    structure=(
                        structure
                    ),
                    occluders=(
                        true_occluders
                    ),
                )
                .localisation_sigma

                for structure
                in true_structures
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

    relevance_sum = float(
        np.sum(
            relevance
        )
    )

    scores = []

    for index, candidate in enumerate(
        candidates
    ):
        information = (
            information_matrix[
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
            _oracle_alignment(
                pose=(
                    candidate.pose
                ),
                true_structures=(
                    true_structures
                ),
                relevance=relevance,
            )
        )

        total = (
            config.perception_weight
            * task_information
            + config.alignment_weight
            * alignment
            - config.movement_weight
            * normalised_movement[
                index
            ]
        )

        scores.append(
            OracleCandidateScore(
                candidate=candidate,
                predicted_sigmas=tuple(
                    float(
                        value
                    )
                    for value
                    in sigma_matrix[
                        index
                    ]
                ),
                information_scores=tuple(
                    float(
                        value
                    )
                    for value
                    in information
                ),
                relevance_weights=tuple(
                    float(
                        value
                    )
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
                task_scene_information=float(
                    task_information
                ),
                task_alignment=float(
                    alignment
                ),
                oracle_score=float(
                    total
                ),
            )
        )

    return tuple(
        scores
    )


def select_oracle_viewpoint(
    *,
    observation_model: ViewpointObservationModel,
    current_pose: CameraPose,
    candidates: tuple[
        CandidateViewpoint,
        ...,
    ],
    true_structures: tuple[
        SphericalStructure,
        ...,
    ],
    true_occluders: tuple[
        SphericalStructure,
        ...,
    ],
    task_trajectory: np.ndarray,
    config: FairSceneScoringConfig
    | None = None,
    relevance_config: TaskRelevanceConfig
    | None = None,
) -> OracleSelection:
    """Return privileged analysis-only oracle selection."""

    scores = score_oracle_candidates(
        observation_model=(
            observation_model
        ),
        current_pose=(
            current_pose
        ),
        candidates=candidates,
        true_structures=(
            true_structures
        ),
        true_occluders=(
            true_occluders
        ),
        task_trajectory=(
            task_trajectory
        ),
        config=config,
        relevance_config=(
            relevance_config
        ),
    )

    selected = max(
        scores,
        key=lambda item:
        item.oracle_score,
    )

    return OracleSelection(
        selected=selected,
        scores=scores,
    )