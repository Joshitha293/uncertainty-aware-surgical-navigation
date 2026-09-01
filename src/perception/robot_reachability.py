"""Robot-aware reachability filtering for active camera viewpoints.

Phase 1 generated geometrically useful virtual camera viewpoints.

Phase 2 introduces a physical-feasibility layer for a simplified straight
RCM-constrained robotic endoscope.

Model assumption
----------------
The camera is mounted at the distal end of the RCM instrument.

The instrument tool frame uses:

    +x = shaft direction

The camera frame uses:

    +z = optical / forward direction

For a straight zero-degree endoscope:

    camera +z == tool +x

Axial instrument roll is selected to best reproduce the requested image-plane
orientation after the required tip position has been solved analytically.

A candidate is considered robot-reachable only when it satisfies all enabled
kinematic and safety constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry.workspace import SphericalStructure
from src.perception.camera import CameraPose
from src.perception.viewpoints import CandidateViewpoint
from src.robotics.advanced_kinematics import (
    KinematicState,
    evaluate_kinematics,
    tool_rotation_matrix,
)
from src.robotics.instrument import SurgicalInstrument
from src.robotics.safety import (
    SafetyEvaluation,
    evaluate_instrument_safety,
)


# Camera axes expressed in the instrument tool frame.
#
# camera right   (+x_c) = tool +y
# camera up      (+y_c) = tool +z
# camera forward (+z_c) = tool +x
_TOOL_TO_CAMERA = np.asarray(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class ViewpointReachabilityConfig:
    """Robot-feasibility requirements for candidate viewpoints."""

    instrument_radius: float = 0.006

    proximal_length: float = 0.10

    maximum_orientation_error: float = np.deg2rad(
        5.0
    )

    rcm_tolerance: float = 1e-8

    minimum_joint_limit_margin: float = 0.0

    maximum_condition_number: float = 10.0

    minimum_manipulability: float = 0.0

    require_safety_margin: bool = True

    def __post_init__(self) -> None:
        if (
            not np.isfinite(
                self.instrument_radius
            )
            or self.instrument_radius <= 0.0
        ):
            raise ValueError(
                "instrument_radius must be finite and positive."
            )

        if (
            not np.isfinite(
                self.proximal_length
            )
            or self.proximal_length <= 0.0
        ):
            raise ValueError(
                "proximal_length must be finite and positive."
            )

        if (
            not np.isfinite(
                self.maximum_orientation_error
            )
            or self.maximum_orientation_error < 0.0
        ):
            raise ValueError(
                "maximum_orientation_error must be finite "
                "and non-negative."
            )

        if (
            not np.isfinite(
                self.rcm_tolerance
            )
            or self.rcm_tolerance < 0.0
        ):
            raise ValueError(
                "rcm_tolerance must be finite and non-negative."
            )

        if not (
            0.0
            <= self.minimum_joint_limit_margin
            <= 0.5
        ):
            raise ValueError(
                "minimum_joint_limit_margin must lie between 0 and 0.5."
            )

        if (
            not np.isfinite(
                self.maximum_condition_number
            )
            or self.maximum_condition_number <= 0.0
        ):
            raise ValueError(
                "maximum_condition_number must be finite and positive."
            )

        if (
            not np.isfinite(
                self.minimum_manipulability
            )
            or self.minimum_manipulability < 0.0
        ):
            raise ValueError(
                "minimum_manipulability must be finite "
                "and non-negative."
            )


@dataclass(frozen=True)
class ViewpointReachabilityResult:
    """Feasibility diagnostics for one candidate."""

    candidate: CandidateViewpoint

    reachable: bool

    rejection_reasons: tuple[str, ...]

    configuration: np.ndarray | None

    reconstructed_camera_pose: CameraPose | None

    orientation_error: float | None

    kinematics: KinematicState | None

    safety: SafetyEvaluation | None


@dataclass(frozen=True)
class ReachableViewpointSet:
    """Result of filtering a candidate viewpoint collection."""

    reachable_candidates: tuple[
        CandidateViewpoint,
        ...
    ]

    evaluations: tuple[
        ViewpointReachabilityResult,
        ...
    ]

    total_count: int

    reachable_count: int

    rejected_count: int


def endoscope_camera_rotation(
    instrument: SurgicalInstrument,
    q: np.ndarray,
) -> np.ndarray:
    """Return camera orientation produced by an endoscope configuration."""

    tool_rotation = tool_rotation_matrix(
        instrument,
        q,
    )

    return (
        tool_rotation
        @ _TOOL_TO_CAMERA
    )


def endoscope_camera_pose(
    instrument: SurgicalInstrument,
    q: np.ndarray,
) -> CameraPose:
    """Return the camera pose generated by one robot configuration."""

    q = np.asarray(
        q,
        dtype=float,
    )

    instrument.validate_configuration(
        q
    )

    return CameraPose(
        position=instrument.forward_position(
            q
        ),
        rotation=endoscope_camera_rotation(
            instrument,
            q,
        ),
    )


def rotation_distance(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Return the geodesic angular distance between rotations."""

    first = np.asarray(
        first,
        dtype=float,
    )

    second = np.asarray(
        second,
        dtype=float,
    )

    if (
        first.shape != (
            3,
            3,
        )
        or second.shape != (
            3,
            3,
        )
    ):
        raise ValueError(
            "Both rotations must have shape (3, 3)."
        )

    relative = (
        first.T
        @ second
    )

    cosine_angle = (
        np.trace(
            relative
        )
        - 1.0
    ) / 2.0

    cosine_angle = float(
        np.clip(
            cosine_angle,
            -1.0,
            1.0,
        )
    )

    return float(
        np.arccos(
            cosine_angle
        )
    )


def _wrap_to_pi(
    angle: float,
) -> float:
    """Wrap an angle to [-pi, pi]."""

    wrapped = (
        angle
        + np.pi
    ) % (
        2.0
        * np.pi
    ) - np.pi

    return float(
        wrapped
    )


def solve_endoscope_viewpoint_configuration(
    instrument: SurgicalInstrument,
    pose: CameraPose,
) -> tuple[
    np.ndarray,
    CameraPose,
    float,
]:
    """Solve the robot configuration best matching a camera pose.

    Position is solved analytically.

    Roll is then chosen to align the requested camera-right direction with
    the endoscope image plane as closely as possible.

    Returns
    -------
    tuple
        configuration, reconstructed camera pose, orientation error
    """

    base_q = instrument.inverse_position(
        pose.position,
        roll=0.0,
    )

    base_tool_rotation = (
        tool_rotation_matrix(
            instrument,
            base_q,
        )
    )

    base_tool_y = (
        base_tool_rotation[
            :,
            1,
        ]
    )

    base_tool_z = (
        base_tool_rotation[
            :,
            2,
        ]
    )

    requested_right = (
        pose.right
    )

    cosine_component = float(
        np.dot(
            requested_right,
            base_tool_y,
        )
    )

    sine_component = float(
        np.dot(
            requested_right,
            base_tool_z,
        )
    )

    roll = _wrap_to_pi(
        np.arctan2(
            sine_component,
            cosine_component,
        )
    )

    q = np.array(
        base_q,
        dtype=float,
        copy=True,
    )

    q[
        3
    ] = roll

    instrument.validate_configuration(
        q
    )

    reconstructed = (
        endoscope_camera_pose(
            instrument,
            q,
        )
    )

    orientation_error = (
        rotation_distance(
            reconstructed.rotation,
            pose.rotation,
        )
    )

    return (
        q,
        reconstructed,
        orientation_error,
    )


def evaluate_viewpoint_reachability(
    instrument: SurgicalInstrument,
    candidate: CandidateViewpoint,
    structures: tuple[
        SphericalStructure,
        ...,
    ] = (),
    config: ViewpointReachabilityConfig | None = None,
) -> ViewpointReachabilityResult:
    """Evaluate whether a virtual camera viewpoint is robot executable."""

    if config is None:
        config = (
            ViewpointReachabilityConfig()
        )

    try:
        (
            q,
            reconstructed_pose,
            orientation_error,
        ) = solve_endoscope_viewpoint_configuration(
            instrument,
            candidate.pose,
        )

    except ValueError:
        return ViewpointReachabilityResult(
            candidate=candidate,
            reachable=False,
            rejection_reasons=(
                "inverse_kinematics_unreachable",
            ),
            configuration=None,
            reconstructed_camera_pose=None,
            orientation_error=None,
            kinematics=None,
            safety=None,
        )

    kinematics = evaluate_kinematics(
        instrument,
        q,
    )

    shaft_start, shaft_end = (
        instrument.shaft_segment(
            q,
            proximal_length=(
                config.proximal_length
            ),
        )
    )

    safety = evaluate_instrument_safety(
        shaft_start=shaft_start,
        shaft_end=shaft_end,
        structures=structures,
        instrument_radius=(
            config.instrument_radius
        ),
    )

    reasons: list[str] = []

    if (
        orientation_error
        > config.maximum_orientation_error
    ):
        reasons.append(
            "orientation_incompatible"
        )

    if (
        kinematics.rcm_error
        > config.rcm_tolerance
    ):
        reasons.append(
            "rcm_constraint_violation"
        )

    if (
        kinematics.normalised_joint_limit_margin
        < config.minimum_joint_limit_margin
    ):
        reasons.append(
            "joint_limit_margin"
        )

    if (
        not np.isfinite(
            kinematics.condition_number
        )
        or kinematics.condition_number
        > config.maximum_condition_number
    ):
        reasons.append(
            "poor_conditioning"
        )

    if (
        kinematics.manipulability
        < config.minimum_manipulability
    ):
        reasons.append(
            "low_manipulability"
        )

    if safety.collision:
        reasons.append(
            "collision"
        )

    if (
        config.require_safety_margin
        and safety.safety_margin_violation
    ):
        reasons.append(
            "safety_margin_violation"
        )

    return ViewpointReachabilityResult(
        candidate=candidate,
        reachable=(
            len(
                reasons
            )
            == 0
        ),
        rejection_reasons=tuple(
            reasons
        ),
        configuration=np.array(
            q,
            dtype=float,
            copy=True,
        ),
        reconstructed_camera_pose=(
            reconstructed_pose
        ),
        orientation_error=float(
            orientation_error
        ),
        kinematics=kinematics,
        safety=safety,
    )


def filter_reachable_viewpoints(
    instrument: SurgicalInstrument,
    candidates: tuple[
        CandidateViewpoint,
        ...,
    ],
    structures: tuple[
        SphericalStructure,
        ...,
    ] = (),
    config: ViewpointReachabilityConfig | None = None,
) -> ReachableViewpointSet:
    """Filter candidate viewpoints using robot and safety constraints."""

    evaluations = tuple(
        evaluate_viewpoint_reachability(
            instrument,
            candidate,
            structures=structures,
            config=config,
        )
        for candidate
        in candidates
    )

    reachable = tuple(
        evaluation.candidate
        for evaluation
        in evaluations
        if evaluation.reachable
    )

    return ReachableViewpointSet(
        reachable_candidates=(
            reachable
        ),
        evaluations=(
            evaluations
        ),
        total_count=len(
            candidates
        ),
        reachable_count=len(
            reachable
        ),
        rejected_count=(
            len(
                candidates
            )
            - len(
                reachable
            )
        ),
    )