"""Closed-loop generic active perception for surgical navigation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.geometry.workspace import SphericalStructure
from src.perception.active_perception import (
    GenericActivePerception,
    ViewpointSelectionResult,
)
from src.perception.camera import CameraPose
from src.perception.observation import (
    ObservationQuality,
    ViewpointObservationModel,
)
from src.perception.viewpoints import CandidateViewpoint


@dataclass(frozen=True)
class ActivePerceptionCycleResult:
    """Result of one select-and-observe active-perception cycle."""

    selection: ViewpointSelectionResult
    selected_pose: CameraPose
    observation: Any
    post_observation_quality: ObservationQuality
    localisation_error: float
    random_seed: int


class ClosedLoopActivePerception:
    """Execute generic active perception as a closed-loop cycle.

    The controller first evaluates candidate viewpoints, selects one using
    generic perception utility, then acquires a simulated observation from
    that viewpoint.

    No task trajectory or task-specific risk is supplied to this controller.
    """

    def __init__(
        self,
        observation_model: ViewpointObservationModel,
        active_perception: GenericActivePerception | None = None,
    ) -> None:
        self.observation_model = observation_model

        if active_perception is None:
            active_perception = GenericActivePerception(
                observation_model=observation_model
            )

        self.active_perception = active_perception

    def execute_cycle(
        self,
        current_pose: CameraPose,
        candidates: tuple[
            CandidateViewpoint,
            ...,
        ],
        target: SphericalStructure,
        rng: np.random.Generator,
        occluders: tuple[
            SphericalStructure,
            ...,
        ] = (),
    ) -> ActivePerceptionCycleResult:
        """Select a viewpoint and acquire a new observation."""

        if len(candidates) == 0:
            raise ValueError(
                "candidates must not be empty."
            )

        selection = (
            self.active_perception.select_viewpoint(
                current_pose=current_pose,
                candidates=candidates,
                target=target,
                occluders=occluders,
            )
        )

        selected_pose = (
            selection.selected_viewpoint.pose
        )

        observation = (
            self.observation_model.observe_structure(
                camera_pose=selected_pose,
                structure=target,
                rng=rng,
                occluders=occluders,
            )
        )

        post_observation_quality = (
            self.observation_model.observation_quality(
                camera_pose=selected_pose,
                structure=target,
                occluders=occluders,
            )
        )

        return ActivePerceptionCycleResult(
            selection=selection,
            selected_pose=selected_pose,
            observation=observation,
            post_observation_quality=(
                post_observation_quality
            ),
            localisation_error=float(
                observation.localisation_error
            ),
            random_seed=0,
        )


def execute_reproducible_cycle(
    controller: ClosedLoopActivePerception,
    current_pose: CameraPose,
    candidates: tuple[
        CandidateViewpoint,
        ...,
    ],
    target: SphericalStructure,
    seed: int,
    occluders: tuple[
        SphericalStructure,
        ...,
    ] = (),
) -> ActivePerceptionCycleResult:
    """Execute a deterministic active-perception cycle."""

    rng = np.random.default_rng(seed)

    result = controller.execute_cycle(
        current_pose=current_pose,
        candidates=candidates,
        target=target,
        rng=rng,
        occluders=occluders,
    )

    return ActivePerceptionCycleResult(
        selection=result.selection,
        selected_pose=result.selected_pose,
        observation=result.observation,
        post_observation_quality=(
            result.post_observation_quality
        ),
        localisation_error=(
            result.localisation_error
        ),
        random_seed=seed,
    )


def uncertainty_improvement(
    result: ActivePerceptionCycleResult,
) -> float:
    """Return predicted uncertainty reduction after viewpoint selection."""

    return float(
        result.selection.current_quality.localisation_sigma
        - result.post_observation_quality.localisation_sigma
    )


def uncertainty_reduction_fraction(
    result: ActivePerceptionCycleResult,
) -> float:
    """Return fractional predicted uncertainty reduction."""

    initial_sigma = (
        result.selection.current_quality.localisation_sigma
    )

    if initial_sigma <= 0.0:
        raise ValueError(
            "Initial localisation uncertainty must be positive."
        )

    return float(
        uncertainty_improvement(result)
        / initial_sigma
    )


def selected_viewpoint_position(
    result: ActivePerceptionCycleResult,
) -> np.ndarray:
    """Return the selected camera position."""

    return np.asarray(
        result.selected_pose.position,
        dtype=float,
    ).copy()