"""Tests for Phase 2 robot-aware viewpoint reachability."""

import numpy as np
import pytest

from src.geometry.workspace import (
    SphericalStructure,
)
from src.perception.camera import (
    CameraPose,
)
from src.perception.robot_reachability import (
    ViewpointReachabilityConfig,
    endoscope_camera_pose,
    evaluate_viewpoint_reachability,
    filter_reachable_viewpoints,
    rotation_distance,
    solve_endoscope_viewpoint_configuration,
)
from src.perception.viewpoints import (
    CandidateViewpoint,
    ViewpointSamplingConfig,
    generate_candidate_viewpoints,
)
from src.robotics.instrument import (
    SurgicalInstrument,
)
from src.robotics.workspace_analysis import (
    characterise_kinematic_workspace,
)


def make_instrument() -> SurgicalInstrument:
    """Create a deterministic RCM instrument."""

    return SurgicalInstrument(
        rcm_position=np.zeros(
            3,
            dtype=float,
        )
    )


def candidate_from_configuration(
    instrument: SurgicalInstrument,
    q: np.ndarray,
) -> CandidateViewpoint:
    """Construct an exactly robot-achievable camera candidate."""

    return CandidateViewpoint(
        pose=endoscope_camera_pose(
            instrument,
            q,
        ),
        radius=0.10,
        azimuth=0.0,
        elevation=0.0,
    )


def interior_configuration() -> np.ndarray:
    """Return a well-conditioned internal configuration."""

    return np.asarray(
        [
            0.20,
            -0.10,
            0.18,
            0.35,
        ],
        dtype=float,
    )


def test_endoscope_camera_forward_matches_shaft():
    """Straight endoscope optical axis must coincide with shaft."""

    instrument = (
        make_instrument()
    )

    q = (
        interior_configuration()
    )

    pose = (
        endoscope_camera_pose(
            instrument,
            q,
        )
    )

    expected = (
        instrument.shaft_direction(
            q[
                0
            ],
            q[
                1
            ],
        )
    )

    assert np.allclose(
        pose.forward,
        expected,
        atol=1e-12,
    )


def test_endoscope_camera_position_matches_tip():
    """Camera centre must coincide with distal tool-tip position."""

    instrument = (
        make_instrument()
    )

    q = (
        interior_configuration()
    )

    pose = (
        endoscope_camera_pose(
            instrument,
            q,
        )
    )

    assert np.allclose(
        pose.position,
        instrument.forward_position(
            q
        ),
        atol=1e-12,
    )


def test_exact_robot_pose_round_trip():
    """An exactly achievable camera pose must recover its robot state."""

    instrument = (
        make_instrument()
    )

    original_q = (
        interior_configuration()
    )

    pose = (
        endoscope_camera_pose(
            instrument,
            original_q,
        )
    )

    (
        recovered_q,
        reconstructed,
        orientation_error,
    ) = solve_endoscope_viewpoint_configuration(
        instrument,
        pose,
    )

    assert np.allclose(
        recovered_q,
        original_q,
        atol=1e-10,
    )

    assert np.allclose(
        reconstructed.position,
        pose.position,
        atol=1e-12,
    )

    assert (
        orientation_error
        <= 1e-10
    )


def test_rotation_distance_zero_for_identical_rotation():
    """Rotation distance must vanish for identical frames."""

    identity = np.eye(
        3
    )

    assert (
        rotation_distance(
            identity,
            identity,
        )
        == pytest.approx(
            0.0
        )
    )


def test_exact_candidate_is_reachable():
    """An interior collision-free robot pose must pass filtering."""

    instrument = (
        make_instrument()
    )

    candidate = (
        candidate_from_configuration(
            instrument,
            interior_configuration(),
        )
    )

    result = (
        evaluate_viewpoint_reachability(
            instrument,
            candidate,
        )
    )

    assert result.reachable

    assert (
        result.rejection_reasons
        == ()
    )

    assert (
        result.configuration
        is not None
    )

    assert (
        result.kinematics
        is not None
    )

    assert (
        result.safety
        is not None
    )


def test_candidate_beyond_insertion_limit_is_rejected():
    """Camera positions outside insertion range must fail IK."""

    instrument = (
        make_instrument()
    )

    rotation = (
        endoscope_camera_pose(
            instrument,
            np.asarray(
                [
                    0.0,
                    0.0,
                    0.20,
                    0.0,
                ]
            ),
        )
        .rotation
    )

    candidate = CandidateViewpoint(
        pose=CameraPose(
            position=np.asarray(
                [
                    0.50,
                    0.0,
                    0.0,
                ],
                dtype=float,
            ),
            rotation=rotation,
        ),
        radius=0.10,
        azimuth=0.0,
        elevation=0.0,
    )

    result = (
        evaluate_viewpoint_reachability(
            instrument,
            candidate,
        )
    )

    assert not result.reachable

    assert (
        "inverse_kinematics_unreachable"
        in result.rejection_reasons
    )


def test_orientation_incompatible_candidate_is_rejected():
    """A virtual camera looking backwards cannot be a straight endoscope."""

    instrument = (
        make_instrument()
    )

    q = (
        interior_configuration()
    )

    achievable = (
        endoscope_camera_pose(
            instrument,
            q,
        )
    )

    flip = np.asarray(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )

    candidate = CandidateViewpoint(
        pose=CameraPose(
            position=(
                achievable.position
            ),
            rotation=(
                achievable.rotation
                @ flip
            ),
        ),
        radius=0.10,
        azimuth=0.0,
        elevation=0.0,
    )

    result = (
        evaluate_viewpoint_reachability(
            instrument,
            candidate,
        )
    )

    assert not result.reachable

    assert (
        "orientation_incompatible"
        in result.rejection_reasons
    )


def test_joint_limit_margin_can_reject_boundary_pose():
    """Reachable poses may still be rejected near a joint boundary."""

    instrument = (
        make_instrument()
    )

    limits = (
        instrument.joint_limits
    )

    q = np.asarray(
        [
            limits.yaw_max,
            0.0,
            0.18,
            0.0,
        ],
        dtype=float,
    )

    candidate = (
        candidate_from_configuration(
            instrument,
            q,
        )
    )

    result = (
        evaluate_viewpoint_reachability(
            instrument,
            candidate,
            config=(
                ViewpointReachabilityConfig(
                    minimum_joint_limit_margin=0.01,
                )
            ),
        )
    )

    assert not result.reachable

    assert (
        "joint_limit_margin"
        in result.rejection_reasons
    )


def test_conditioning_threshold_can_reject_pose():
    """Kinematic conditioning can be enforced as a feasibility criterion."""

    instrument = (
        make_instrument()
    )

    candidate = (
        candidate_from_configuration(
            instrument,
            interior_configuration(),
        )
    )

    result = (
        evaluate_viewpoint_reachability(
            instrument,
            candidate,
            config=(
                ViewpointReachabilityConfig(
                    maximum_condition_number=1.01,
                )
            ),
        )
    )

    assert not result.reachable

    assert (
        "poor_conditioning"
        in result.rejection_reasons
    )


def test_manipulability_threshold_can_reject_pose():
    """Low-manipulability configurations can be filtered."""

    instrument = (
        make_instrument()
    )

    candidate = (
        candidate_from_configuration(
            instrument,
            interior_configuration(),
        )
    )

    result = (
        evaluate_viewpoint_reachability(
            instrument,
            candidate,
            config=(
                ViewpointReachabilityConfig(
                    minimum_manipulability=1.0,
                )
            ),
        )
    )

    assert not result.reachable

    assert (
        "low_manipulability"
        in result.rejection_reasons
    )


def test_anatomical_collision_rejects_candidate():
    """A shaft intersecting anatomy must be rejected."""

    instrument = (
        make_instrument()
    )

    q = np.asarray(
        [
            0.0,
            0.0,
            0.15,
            0.0,
        ],
        dtype=float,
    )

    candidate = (
        candidate_from_configuration(
            instrument,
            q,
        )
    )

    structure = SphericalStructure(
        centre=np.asarray(
            [
                0.08,
                0.0,
                0.0,
            ],
            dtype=float,
        ),
        physical_radius=0.01,
        safety_margin=0.0,
    )

    result = (
        evaluate_viewpoint_reachability(
            instrument,
            candidate,
            structures=(
                structure,
            ),
        )
    )

    assert not result.reachable

    assert (
        "collision"
        in result.rejection_reasons
    )


def test_safety_margin_violation_rejects_candidate_without_collision():
    """Protected anatomy margin must be enforced independently of collision."""

    instrument = (
        make_instrument()
    )

    q = np.asarray(
        [
            0.0,
            0.0,
            0.15,
            0.0,
        ],
        dtype=float,
    )

    candidate = (
        candidate_from_configuration(
            instrument,
            q,
        )
    )

    structure = SphericalStructure(
        centre=np.asarray(
            [
                0.08,
                0.020,
                0.0,
            ],
            dtype=float,
        ),
        physical_radius=0.005,
        safety_margin=0.015,
    )

    result = (
        evaluate_viewpoint_reachability(
            instrument,
            candidate,
            structures=(
                structure,
            ),
            config=(
                ViewpointReachabilityConfig(
                    instrument_radius=0.004,
                )
            ),
        )
    )

    assert result.safety is not None

    assert not (
        result.safety.collision
    )

    assert (
        result.safety
        .safety_margin_violation
    )

    assert not result.reachable

    assert (
        "safety_margin_violation"
        in result.rejection_reasons
    )


def test_filter_preserves_only_reachable_candidates():
    """Candidate-set filtering must preserve only executable viewpoints."""

    instrument = (
        make_instrument()
    )

    reachable = (
        candidate_from_configuration(
            instrument,
            interior_configuration(),
        )
    )

    impossible = CandidateViewpoint(
        pose=CameraPose(
            position=np.asarray(
                [
                    0.50,
                    0.0,
                    0.0,
                ],
                dtype=float,
            ),
            rotation=(
                reachable.pose.rotation
            ),
        ),
        radius=0.10,
        azimuth=0.0,
        elevation=0.0,
    )

    result = (
        filter_reachable_viewpoints(
            instrument,
            (
                reachable,
                impossible,
            ),
        )
    )

    assert (
        result.total_count
        == 2
    )

    assert (
        result.reachable_count
        == 1
    )

    assert (
        result.rejected_count
        == 1
    )

    assert (
        result.reachable_candidates
        == (
            reachable,
        )
    )


def test_phase1_style_generated_candidates_are_robot_filtered():
    """Virtual spherical viewpoints must now face robot feasibility checks."""

    instrument = (
        make_instrument()
    )

    target = np.asarray(
        [
            0.25,
            0.0,
            0.0,
        ],
        dtype=float,
    )

    candidates = (
        generate_candidate_viewpoints(
            target,
            ViewpointSamplingConfig(
                radii=(
                    0.10,
                ),
                azimuth_count=2,
                elevation_angles=(
                    0.0,
                ),
            ),
        )
    )

    result = (
        filter_reachable_viewpoints(
            instrument,
            candidates,
        )
    )

    assert (
        result.total_count
        == 2
    )

    # Candidate at x=0.15 looks toward +x and is compatible with
    # a straight RCM endoscope. Candidate at x=0.35 exceeds the
    # configured insertion range.
    assert (
        result.reachable_count
        == 1
    )

    reachable_pose = (
        result
        .reachable_candidates[
            0
        ]
        .pose
    )

    assert np.allclose(
        reachable_pose.position,
        np.asarray(
            [
                0.15,
                0.0,
                0.0,
            ]
        ),
        atol=1e-12,
    )


def test_reachable_candidate_preserves_rcm_constraint():
    """Robot-approved viewpoints must preserve the fixed trocar RCM."""

    instrument = (
        make_instrument()
    )

    candidate = (
        candidate_from_configuration(
            instrument,
            interior_configuration(),
        )
    )

    result = (
        evaluate_viewpoint_reachability(
            instrument,
            candidate,
        )
    )

    assert result.reachable

    assert (
        result.kinematics
        is not None
    )

    assert (
        result.kinematics.rcm_error
        <= 1e-8
    )


def test_workspace_characterisation_covers_requested_grid():
    """Workspace analysis must evaluate the complete configured grid."""

    instrument = (
        make_instrument()
    )

    summary = (
        characterise_kinematic_workspace(
            instrument,
            yaw_samples=5,
            pitch_samples=5,
            insertion_samples=5,
        )
    )

    assert (
        summary.sample_count
        == 125
    )

    assert (
        summary.tip_minimum.shape
        == (
            3,
        )
    )

    assert (
        summary.tip_maximum.shape
        == (
            3,
        )
    )

    assert (
        summary.maximum_manipulability
        >= summary.minimum_manipulability
        > 0.0
    )

    assert (
        summary.maximum_condition_number
        >= summary.minimum_condition_number
        >= 1.0
    )


def test_admissible_workspace_avoids_true_position_singularity():
    """Configured pitch/insertion limits should keep positional rank nonzero."""

    instrument = (
        make_instrument()
    )

    summary = (
        characterise_kinematic_workspace(
            instrument,
            yaw_samples=7,
            pitch_samples=7,
            insertion_samples=7,
            singular_value_threshold=1e-6,
        )
    )

    assert (
        summary.minimum_singular_value
        > 1e-6
    )

    assert (
        summary.near_singular_count
        == 0
    )

    assert (
        summary.near_singular_fraction
        == pytest.approx(
            0.0
        )
    )