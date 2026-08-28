"""Ground-truth-isolated candidate generation for Phase 1.

The historical simulation generated active camera candidates around the exact
simulator target position. Although every strategy received the same
candidate set, that candidate set implicitly encoded privileged ground-truth
geometry.

This module removes that information path.

Protocol
--------
1. A fixed nominal workspace prior defines the initial camera search geometry.
2. The simulator generates one noisy initial anatomical observation.
3. The estimated target centre from that observation defines the active-view
   candidate set.
4. Generic and Task-Aware strategies receive exactly the same candidates.

Simulator ground truth is therefore used only to generate the simulated
initial observation. It is not used to centre the active candidate set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry.workspace import (
    SphericalStructure,
)
from src.perception.camera import (
    CameraPose,
)
from src.perception.observation import (
    ViewpointObservationModel,
)
from src.perception.perception import (
    PerceptionResult,
)
from src.perception.viewpoints import (
    CandidateViewpoint,
    generate_candidate_viewpoints,
)
from src.simulation.three_strategy_perception import (
    StrategyPerceptionResult,
    run_fixed_perception,
)


# Fixed prior used for every Phase 1 scenario.
#
# This corresponds to the nominal workspace target location before procedural
# scenario perturbations. Crucially, it does not change with the hidden
# scenario translation and therefore does not reveal the actual target.
NOMINAL_TARGET_PRIOR = np.asarray(
    [
        0.14,
        0.04,
        0.00,
    ],
    dtype=float,
)


@dataclass(frozen=True)
class FairCandidateContext:
    """Shared perception and candidate state for one matched trial."""

    initial_pose: CameraPose

    initial_perception_result: StrategyPerceptionResult

    planner_facing_perception: PerceptionResult

    estimated_target_position: np.ndarray

    candidates: tuple[
        CandidateViewpoint,
        ...,
    ]

    initial_seed: int


def nominal_initial_candidates() -> tuple[
    CandidateViewpoint,
    ...,
]:
    """Generate the scenario-independent initial camera candidate set."""

    candidates = generate_candidate_viewpoints(
        target_position=(
            NOMINAL_TARGET_PRIOR
        )
    )

    if len(
        candidates
    ) == 0:
        raise RuntimeError(
            "Nominal initial candidate generation returned no viewpoints."
        )

    return candidates


def make_nominal_initial_pose(
    *,
    initial_view_index: int,
) -> CameraPose:
    """Return an initial camera pose independent of simulator truth."""

    if initial_view_index < 0:
        raise ValueError(
            "initial_view_index must be non-negative."
        )

    candidates = (
        nominal_initial_candidates()
    )

    return candidates[
        initial_view_index
        % len(
            candidates
        )
    ].pose


def estimated_target_position(
    perception_result: PerceptionResult,
) -> np.ndarray:
    """Return target position from planner-facing noisy perception only."""

    estimates = (
        perception_result
        .estimated_structures
    )

    if len(
        estimates
    ) == 0:
        raise ValueError(
            "perception_result contains no estimated structures."
        )

    position = np.asarray(
        estimates[
            0
        ].estimated_centre,
        dtype=float,
    )

    if position.shape != (3,):
        raise ValueError(
            "Estimated target centre must have shape (3,)."
        )

    if not np.all(
        np.isfinite(
            position
        )
    ):
        raise ValueError(
            "Estimated target centre must be finite."
        )

    return position.copy()


def generate_estimate_centred_candidates(
    *,
    perception_result: PerceptionResult,
) -> tuple[
    CandidateViewpoint,
    ...,
]:
    """Generate active candidates around the noisy estimated target."""

    target_position = (
        estimated_target_position(
            perception_result
        )
    )

    candidates = (
        generate_candidate_viewpoints(
            target_position=(
                target_position
            )
        )
    )

    if len(
        candidates
    ) == 0:
        raise RuntimeError(
            "Estimated-target candidate generation returned no viewpoints."
        )

    return candidates


def build_fair_candidate_context(
    *,
    observation_model: ViewpointObservationModel,
    true_structures: tuple[
        SphericalStructure,
        ...,
    ],
    occluders: tuple[
        SphericalStructure,
        ...,
    ] = (),
    initial_view_index: int,
    initial_seed: int,
) -> FairCandidateContext:
    """Build a shared candidate set without true-target centring.

    Ground truth enters only ``run_fixed_perception`` to simulate the initial
    measurement.

    The resulting active candidate set is generated solely from the noisy
    planner-facing target estimate.
    """

    if len(
        true_structures
    ) == 0:
        raise ValueError(
            "true_structures must not be empty."
        )

    initial_pose = (
        make_nominal_initial_pose(
            initial_view_index=(
                initial_view_index
            )
        )
    )

    initial_result = (
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

    planner_facing = (
        initial_result
        .perception_result
    )

    estimated_target = (
        estimated_target_position(
            planner_facing
        )
    )

    candidates = (
        generate_estimate_centred_candidates(
            perception_result=(
                planner_facing
            )
        )
    )

    return FairCandidateContext(
        initial_pose=(
            initial_pose
        ),
        initial_perception_result=(
            initial_result
        ),
        planner_facing_perception=(
            planner_facing
        ),
        estimated_target_position=(
            estimated_target
        ),
        candidates=candidates,
        initial_seed=int(
            initial_seed
        ),
    )