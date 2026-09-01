"""Robot-aware integration layer for active viewpoint selection.

Phase 1 viewpoint scorers operate on virtual candidate camera poses.

Phase 2 introduces a pre-selection feasibility gate:

    candidates
        -> robot reachability
        -> existing perception scorer
        -> selected robot configuration

The established Phase 1 scoring implementations are deliberately not
modified. This preserves the frozen Phase 1 evidence while allowing later
experiments to use only viewpoints executable by the simulated
RCM-constrained endoscope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.geometry.workspace import (
    SphericalStructure,
)
from src.perception.camera import (
    CameraPose,
)
from src.perception.perception import (
    PerceptionResult,
)
from src.perception.robot_reachability import (
    ReachableViewpointSet,
    ViewpointReachabilityConfig,
    ViewpointReachabilityResult,
    filter_reachable_viewpoints,
)
from src.perception.viewpoints import (
    CandidateViewpoint,
)
from src.robotics.instrument import (
    SurgicalInstrument,
)


class NoReachableViewpointError(
    RuntimeError
):
    """Raised when no candidate viewpoint can be executed by the robot."""


@dataclass(frozen=True)
class RobotAwareSceneSelection:
    """Robot-aware active-perception result.

    Attributes
    ----------
    base_selection:
        Selection object returned by the existing Generic or Task-Aware
        scene scorer.

    reachability:
        Complete candidate-level robot-feasibility diagnostics.

    selected_viewpoint:
        Candidate chosen by the existing perception scorer after robot
        filtering.

    selected_configuration:
        Joint configuration that realises the selected camera viewpoint.

    selected_evaluation:
        Complete reachability diagnostics for the selected viewpoint.
    """

    base_selection: Any

    reachability: ReachableViewpointSet

    selected_viewpoint: CandidateViewpoint

    selected_configuration: np.ndarray

    selected_evaluation: ViewpointReachabilityResult


def _selected_candidate_from_scene_selection(
    selection: Any,
) -> CandidateViewpoint:
    """Extract the candidate from a Phase 1 scene-selection result."""

    if not hasattr(
        selection,
        "selected",
    ):
        raise TypeError(
            "Scene scorer result must expose a 'selected' attribute."
        )

    selected_score = (
        selection.selected
    )

    if not hasattr(
        selected_score,
        "candidate",
    ):
        raise TypeError(
            "Selected scene score must expose a 'candidate' attribute."
        )

    candidate = (
        selected_score.candidate
    )

    if not isinstance(
        candidate,
        CandidateViewpoint,
    ):
        raise TypeError(
            "Selected candidate must be a CandidateViewpoint."
        )

    return candidate


def _find_selected_evaluation(
    reachability: ReachableViewpointSet,
    selected_viewpoint: CandidateViewpoint,
) -> ViewpointReachabilityResult:
    """Find robot diagnostics corresponding to the selected candidate.

    Reachable candidates are passed directly from the feasibility result
    into the existing scorer, so object identity should be preserved.
    """

    for evaluation in (
        reachability.evaluations
    ):
        if (
            evaluation.candidate
            is selected_viewpoint
        ):
            return evaluation

    raise RuntimeError(
        "The active-perception scorer selected a viewpoint that was "
        "not supplied by the robot-feasibility layer."
    )


def select_robot_aware_scene_viewpoint(
    *,
    instrument: SurgicalInstrument,
    scorer: Any,
    current_pose: CameraPose,
    candidates: tuple[
        CandidateViewpoint,
        ...,
    ],
    initial_perception: PerceptionResult,
    planning_structures: tuple[
        SphericalStructure,
        ...,
    ] = (),
    reachability_config: (
        ViewpointReachabilityConfig
        | None
    ) = None,
) -> RobotAwareSceneSelection:
    """Select a robot-executable active-perception viewpoint.

    The function is compatible with both:

    - GenericSceneViewpointScorer
    - TaskAwareSceneViewpointScorer

    because both expose the same keyword-based ``select_viewpoint`` API.

    Parameters
    ----------
    planning_structures:
        Anatomical geometry available to the planner.

        In uncertainty experiments this must be planner-visible estimated
        anatomy rather than hidden simulator ground truth. The caller is
        responsible for preserving that information boundary.
    """

    if len(
        candidates
    ) == 0:
        raise ValueError(
            "candidates must not be empty."
        )

    reachability = (
        filter_reachable_viewpoints(
            instrument=instrument,
            candidates=candidates,
            structures=(
                planning_structures
            ),
            config=(
                reachability_config
            ),
        )
    )

    if (
        reachability.reachable_count
        == 0
    ):
        raise NoReachableViewpointError(
            "No candidate viewpoint satisfies the configured "
            "robot kinematic and safety constraints."
        )

    selection = (
        scorer.select_viewpoint(
            current_pose=current_pose,
            candidates=(
                reachability
                .reachable_candidates
            ),
            initial_perception=(
                initial_perception
            ),
        )
    )

    selected_viewpoint = (
        _selected_candidate_from_scene_selection(
            selection
        )
    )

    selected_evaluation = (
        _find_selected_evaluation(
            reachability,
            selected_viewpoint,
        )
    )

    if (
        not selected_evaluation.reachable
        or selected_evaluation.configuration
        is None
    ):
        raise RuntimeError(
            "Selected candidate does not have a valid robot configuration."
        )

    return RobotAwareSceneSelection(
        base_selection=selection,
        reachability=reachability,
        selected_viewpoint=(
            selected_viewpoint
        ),
        selected_configuration=np.array(
            selected_evaluation.configuration,
            dtype=float,
            copy=True,
        ),
        selected_evaluation=(
            selected_evaluation
        ),
    )