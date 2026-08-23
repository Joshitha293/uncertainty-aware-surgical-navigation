"""Unified perception interface for the final three-strategy experiment.

This module places fixed-view perception, generic active perception and
task-aware active perception behind a common interface.  Each strategy
observes the same ground-truth anatomy under matched random conditions and
returns planner-compatible perceived anatomy.

Ground-truth structures are used only by the simulated observation model.
The resulting planner-facing state contains noisy anatomical estimates and
explicit localisation uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.geometry.workspace import SphericalStructure
from src.perception.active_perception import GenericActivePerception
from src.perception.camera import CameraPose
from src.perception.observation import (
    StructureObservation,
    ViewpointObservationModel,
)
from src.perception.perception import PerceptionResult
from src.perception.task_aware_active_perception import (
    TaskAwareActivePerception,
)
from src.perception.task_relevance import SurgicalTask
from src.perception.viewpoints import (
    CandidateViewpoint,
    viewpoint_displacement,
)


class PerceptionStrategy(str, Enum):
    """Perception strategies evaluated by the final experiment."""

    FIXED = "fixed"
    GENERIC_ACTIVE = "generic_active"
    TASK_AWARE_ACTIVE = "task_aware_active"


@dataclass(frozen=True)
class StrategyPerceptionResult:
    """Common output returned by every perception strategy."""

    strategy: PerceptionStrategy
    selected_pose: CameraPose

    perception_result: PerceptionResult

    mean_localisation_error: float
    mean_predicted_sigma: float

    camera_movement: float

    visible_fraction: float
    occluded_fraction: float

    candidate_count: int

    task_relevance: float | None
    task_alignment: float | None


def _observations_to_perception_result(
    observations: tuple[StructureObservation, ...],
) -> PerceptionResult:
    """Convert viewpoint observations into the planner-facing perception type."""

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
        estimated_structures=estimated_structures,
        localisation_errors=localisation_errors,
    )


def _mean_localisation_error(
    observations: tuple[StructureObservation, ...],
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
    observations: tuple[StructureObservation, ...],
) -> float:
    """Return mean predicted positional standard deviation."""

    if len(observations) == 0:
        return 0.0

    return float(
        np.mean(
            [
                observation.quality.localisation_sigma
                for observation in observations
            ]
        )
    )


def _visible_fraction(
    observations: tuple[StructureObservation, ...],
) -> float:
    """Return fraction of structures visible from the selected pose."""

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
    observations: tuple[StructureObservation, ...],
) -> float:
    """Return fraction of structures geometrically occluded."""

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


def _observe_from_pose(
    *,
    strategy: PerceptionStrategy,
    observation_model: ViewpointObservationModel,
    selected_pose: CameraPose,
    initial_pose: CameraPose,
    true_structures: tuple[SphericalStructure, ...],
    rng: np.random.Generator,
    occluders: tuple[SphericalStructure, ...],
    candidate_count: int,
    task_relevance: float | None = None,
    task_alignment: float | None = None,
) -> StrategyPerceptionResult:
    """Observe all planning structures from one selected camera pose."""

    if len(true_structures) == 0:
        raise ValueError(
            "true_structures must contain at least one structure."
        )

    observations = observation_model.observe_structures(
        camera_pose=selected_pose,
        structures=true_structures,
        rng=rng,
        occluders=occluders,
    )

    perception_result = (
        _observations_to_perception_result(
            observations
        )
    )

    return StrategyPerceptionResult(
        strategy=strategy,
        selected_pose=selected_pose,
        perception_result=perception_result,
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
            else float(task_relevance)
        ),
        task_alignment=(
            None
            if task_alignment is None
            else float(task_alignment)
        ),
    )


def run_fixed_perception(
    *,
    observation_model: ViewpointObservationModel,
    initial_pose: CameraPose,
    true_structures: tuple[SphericalStructure, ...],
    seed: int,
    occluders: tuple[SphericalStructure, ...] = (),
) -> StrategyPerceptionResult:
    """Run the fixed-view perception baseline."""

    return _observe_from_pose(
        strategy=PerceptionStrategy.FIXED,
        observation_model=observation_model,
        selected_pose=initial_pose,
        initial_pose=initial_pose,
        true_structures=true_structures,
        rng=np.random.default_rng(seed),
        occluders=occluders,
        candidate_count=0,
    )


def run_generic_active_perception(
    *,
    controller: GenericActivePerception,
    observation_model: ViewpointObservationModel,
    initial_pose: CameraPose,
    candidates: tuple[CandidateViewpoint, ...],
    target: SphericalStructure,
    true_structures: tuple[SphericalStructure, ...],
    seed: int,
    occluders: tuple[SphericalStructure, ...] = (),
) -> StrategyPerceptionResult:
    """Run generic task-agnostic active perception."""

    if len(candidates) == 0:
        raise ValueError(
            "candidates must not be empty."
        )

    selection = controller.select_viewpoint(
        current_pose=initial_pose,
        candidates=candidates,
        target=target,
        occluders=occluders,
    )

    selected_pose = (
        selection.selected_viewpoint.pose
    )

    return _observe_from_pose(
        strategy=(
            PerceptionStrategy.GENERIC_ACTIVE
        ),
        observation_model=observation_model,
        selected_pose=selected_pose,
        initial_pose=initial_pose,
        true_structures=true_structures,
        rng=np.random.default_rng(seed),
        occluders=occluders,
        candidate_count=len(candidates),
    )


def run_task_aware_active_perception(
    *,
    controller: TaskAwareActivePerception,
    observation_model: ViewpointObservationModel,
    initial_pose: CameraPose,
    candidates: tuple[CandidateViewpoint, ...],
    target: SphericalStructure,
    task: SurgicalTask,
    true_structures: tuple[SphericalStructure, ...],
    seed: int,
    occluders: tuple[SphericalStructure, ...] = (),
) -> StrategyPerceptionResult:
    """Run task-aware active perception."""

    if len(candidates) == 0:
        raise ValueError(
            "candidates must not be empty."
        )

    if controller.task is not task:
        if not (
            np.array_equal(
                controller.task.trajectory,
                task.trajectory,
            )
            and np.array_equal(
                controller.task.safety_critical_points,
                task.safety_critical_points,
            )
        ):
            raise ValueError(
                "controller task does not match supplied task."
            )

    selection = controller.select_viewpoint(
        current_pose=initial_pose,
        candidates=candidates,
        target=target,
        occluders=occluders,
    )

    selected_pose = (
        selection.selected_viewpoint.pose
    )

    return _observe_from_pose(
        strategy=(
            PerceptionStrategy.TASK_AWARE_ACTIVE
        ),
        observation_model=observation_model,
        selected_pose=selected_pose,
        initial_pose=initial_pose,
        true_structures=true_structures,
        rng=np.random.default_rng(seed),
        occluders=occluders,
        candidate_count=len(candidates),
        task_relevance=(
            selection.task_relevance
        ),
        task_alignment=(
            selection.task_alignment
        ),
    )