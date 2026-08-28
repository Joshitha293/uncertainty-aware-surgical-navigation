"""Fair scene-wide perception pipeline with shared initial observations.

This module implements the strengthened Phase 1 comparison.

Experimental information flow
-----------------------------

Simulator ground truth
        |
        v
shared noisy INITIAL observation
        |
        +-----------------------------+
        |                             |
        v                             v
Generic Active                  Task-Aware Active
same estimated anatomy          same estimated anatomy
uniform scene utility           task trajectory + relevance
same candidate set              same candidate set
same movement objective         same movement objective
        |                             |
        v                             v
selected viewpoint              selected viewpoint
        |                             |
        +-------------+---------------+
                      |
                      v
         simulated FINAL observation
                      |
                      v
           planner-facing estimates

Ground-truth anatomy is required only by the simulator to generate
observations. It is not supplied to the viewpoint-selection API.

Matched random seeds are used for both final observations so that stochastic
localisation noise is controlled as closely as possible across strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect

import numpy as np

from src.geometry.workspace import SphericalStructure
from src.perception.camera import CameraPose
from src.perception.fair_scene_viewpoint_scoring import (
    FairSceneScoringConfig,
    FairSceneSelection,
    FairSceneViewpointScorer,
)
from src.perception.observation import (
    StructureObservation,
    ViewpointObservationModel,
)
from src.perception.perception import PerceptionResult
from src.perception.viewpoints import (
    CandidateViewpoint,
    viewpoint_displacement,
)
from src.simulation.three_strategy_perception import (
    PerceptionStrategy,
    StrategyPerceptionResult,
    run_fixed_perception,
)


@dataclass(frozen=True)
class FairSceneViewpointSelections:
    """Generic and Task-Aware selections from one shared initial estimate."""

    generic: FairSceneSelection
    task_aware: FairSceneSelection


@dataclass(frozen=True)
class FairScenePerceptionResult:
    """Complete matched fair-scene perception experiment."""

    initial_perception: StrategyPerceptionResult

    selections: FairSceneViewpointSelections

    generic_perception: StrategyPerceptionResult

    task_aware_perception: StrategyPerceptionResult

    initial_seed: int

    final_seed: int


def _observations_to_perception_result(
    observations: tuple[
        StructureObservation,
        ...,
    ],
) -> PerceptionResult:
    """Convert simulated observations into planner-facing perception."""

    estimated_structures = tuple(
        observation.estimated_structure
        for observation in observations
    )

    localisation_errors = np.asarray(
        [
            observation.localisation_error
            for observation in observations
        ],
        dtype=float,
    )

    return PerceptionResult(
        estimated_structures=(
            estimated_structures
        ),
        localisation_errors=(
            localisation_errors
        ),
    )


def _mean_localisation_error(
    observations: tuple[
        StructureObservation,
        ...,
    ],
) -> float:
    """Return mean realised localisation error."""

    if len(observations) == 0:
        return 0.0

    return float(
        np.mean(
            [
                observation.localisation_error
                for observation in observations
            ]
        )
    )


def _mean_predicted_sigma(
    observations: tuple[
        StructureObservation,
        ...,
    ],
) -> float:
    """Return mean predicted positional sigma."""

    if len(observations) == 0:
        return 0.0

    return float(
        np.mean(
            [
                observation
                .quality
                .localisation_sigma
                for observation in observations
            ]
        )
    )


def _visible_fraction(
    observations: tuple[
        StructureObservation,
        ...,
    ],
) -> float:
    """Return fraction of observed structures visible."""

    if len(observations) == 0:
        return 0.0

    return float(
        np.mean(
            [
                observation.quality.visible
                for observation in observations
            ]
        )
    )


def _occluded_fraction(
    observations: tuple[
        StructureObservation,
        ...,
    ],
) -> float:
    """Return fraction of observed structures occluded."""

    if len(observations) == 0:
        return 0.0

    return float(
        np.mean(
            [
                observation.quality.occluded
                for observation in observations
            ]
        )
    )


def _observe_final_pose(
    *,
    strategy: PerceptionStrategy,
    observation_model: ViewpointObservationModel,
    selected_pose: CameraPose,
    initial_pose: CameraPose,
    true_structures: tuple[
        SphericalStructure,
        ...,
    ],
    occluders: tuple[
        SphericalStructure,
        ...,
    ],
    seed: int,
    candidate_count: int,
    task_relevance: float | None = None,
    task_alignment: float | None = None,
) -> StrategyPerceptionResult:
    """Generate one final simulator observation after viewpoint selection."""

    if len(
        true_structures
    ) == 0:
        raise ValueError(
            "true_structures must not be empty."
        )

    observations = (
        observation_model
        .observe_structures(
            camera_pose=selected_pose,
            structures=true_structures,
            rng=np.random.default_rng(
                seed
            ),
            occluders=occluders,
        )
    )

    return StrategyPerceptionResult(
        strategy=strategy,
        selected_pose=selected_pose,
        perception_result=(
            _observations_to_perception_result(
                observations
            )
        ),
        mean_localisation_error=(
            _mean_localisation_error(
                observations
            )
        ),
        mean_predicted_sigma=(
            _mean_predicted_sigma(
                observations
            )
        ),
        camera_movement=float(
            viewpoint_displacement(
                current_pose=initial_pose,
                candidate_pose=selected_pose,
            )
        ),
        visible_fraction=(
            _visible_fraction(
                observations
            )
        ),
        occluded_fraction=(
            _occluded_fraction(
                observations
            )
        ),
        candidate_count=int(
            candidate_count
        ),
        task_relevance=(
            None
            if task_relevance is None
            else float(
                task_relevance
            )
        ),
        task_alignment=(
            None
            if task_alignment is None
            else float(
                task_alignment
            )
        ),
    )


def select_fair_scene_viewpoints(
    *,
    observation_model: ViewpointObservationModel,
    initial_pose: CameraPose,
    candidates: tuple[
        CandidateViewpoint,
        ...,
    ],
    initial_perception: PerceptionResult,
    task_trajectory: np.ndarray,
    config: FairSceneScoringConfig
    | None = None,
) -> FairSceneViewpointSelections:
    """Select Generic and Task-Aware poses from identical estimated anatomy.

    This public selection function deliberately has no argument for:

    - true anatomical structures;
    - true target position;
    - true occluder geometry.

    The selection layer therefore cannot directly access simulator truth.
    """

    if len(
        candidates
    ) == 0:
        raise ValueError(
            "candidates must not be empty."
        )

    scorer = FairSceneViewpointScorer(
        observation_model=(
            observation_model
        ),
        task_trajectory=(
            task_trajectory
        ),
        config=config,
    )

    generic = scorer.select_generic(
        current_pose=initial_pose,
        candidates=candidates,
        initial_perception=(
            initial_perception
        ),
    )

    task_aware = (
        scorer.select_task_aware(
            current_pose=initial_pose,
            candidates=candidates,
            initial_perception=(
                initial_perception
            ),
        )
    )

    return FairSceneViewpointSelections(
        generic=generic,
        task_aware=task_aware,
    )


def run_fair_scene_perception(
    *,
    observation_model: ViewpointObservationModel,
    initial_pose: CameraPose,
    candidates: tuple[
        CandidateViewpoint,
        ...,
    ],
    task_trajectory: np.ndarray,
    true_structures: tuple[
        SphericalStructure,
        ...,
    ],
    initial_seed: int,
    final_seed: int,
    occluders: tuple[
        SphericalStructure,
        ...,
    ] = (),
    config: FairSceneScoringConfig
    | None = None,
) -> FairScenePerceptionResult:
    """Run one matched Generic vs Task-Aware perception comparison.

    Ground truth is first used by the simulator to generate one shared noisy
    initial observation.

    Both strategies then receive the exact same planner-facing
    ``PerceptionResult`` object for viewpoint selection.

    Ground truth is used again only after selection to simulate the resulting
    final camera observations.
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

    initial_perception = (
        run_fixed_perception(
            observation_model=(
                observation_model
            ),
            initial_pose=(
                initial_pose
            ),
            true_structures=(
                true_structures
            ),
            seed=int(
                initial_seed
            ),
            occluders=(
                occluders
            ),
        )
    )

    selections = (
        select_fair_scene_viewpoints(
            observation_model=(
                observation_model
            ),
            initial_pose=(
                initial_pose
            ),
            candidates=candidates,
            initial_perception=(
                initial_perception
                .perception_result
            ),
            task_trajectory=(
                task_trajectory
            ),
            config=config,
        )
    )

    generic_selected = (
        selections
        .generic
        .selected
    )

    task_selected = (
        selections
        .task_aware
        .selected
    )

    generic_perception = (
        _observe_final_pose(
            strategy=(
                PerceptionStrategy
                .GENERIC_ACTIVE
            ),
            observation_model=(
                observation_model
            ),
            selected_pose=(
                generic_selected
                .candidate
                .pose
            ),
            initial_pose=(
                initial_pose
            ),
            true_structures=(
                true_structures
            ),
            occluders=(
                occluders
            ),
            seed=int(
                final_seed
            ),
            candidate_count=len(
                candidates
            ),
        )
    )

    relevance = np.asarray(
        task_selected
        .relevance_weights,
        dtype=float,
    )

    task_perception = (
        _observe_final_pose(
            strategy=(
                PerceptionStrategy
                .TASK_AWARE_ACTIVE
            ),
            observation_model=(
                observation_model
            ),
            selected_pose=(
                task_selected
                .candidate
                .pose
            ),
            initial_pose=(
                initial_pose
            ),
            true_structures=(
                true_structures
            ),
            occluders=(
                occluders
            ),
            seed=int(
                final_seed
            ),
            candidate_count=len(
                candidates
            ),
            task_relevance=float(
                np.mean(
                    relevance
                )
            ),
            task_alignment=(
                task_selected
                .task_alignment
            ),
        )
    )

    return FairScenePerceptionResult(
        initial_perception=(
            initial_perception
        ),
        selections=selections,
        generic_perception=(
            generic_perception
        ),
        task_aware_perception=(
            task_perception
        ),
        initial_seed=int(
            initial_seed
        ),
        final_seed=int(
            final_seed
        ),
    )