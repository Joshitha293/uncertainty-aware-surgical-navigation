"""Tests for Phase 2 robot-aware active-perception integration."""

from dataclasses import dataclass

import numpy as np
import pytest

from src.perception.camera import (
    CameraPose,
)
from src.perception.robot_aware_selection import (
    NoReachableViewpointError,
    select_robot_aware_scene_viewpoint,
)
from src.perception.robot_reachability import (
    endoscope_camera_pose,
)
from src.perception.viewpoints import (
    CandidateViewpoint,
)
from src.robotics.instrument import (
    SurgicalInstrument,
)


@dataclass(frozen=True)
class FakeCandidateScore:
    """Minimal score compatible with scene-selection API."""

    candidate: CandidateViewpoint


@dataclass(frozen=True)
class FakeSceneSelection:
    """Minimal selection compatible with scene-selection API."""

    selected: FakeCandidateScore


class RecordingSceneScorer:
    """Test scorer recording the candidates it receives."""

    def __init__(
        self,
        *,
        choose_last: bool = False,
    ) -> None:
        self.choose_last = (
            choose_last
        )

        self.received_candidates: tuple[
            CandidateViewpoint,
            ...,
        ] | None = None

    def select_viewpoint(
        self,
        *,
        current_pose,
        candidates,
        initial_perception,
    ):
        del current_pose
        del initial_perception

        self.received_candidates = (
            candidates
        )

        if self.choose_last:
            candidate = (
                candidates[
                    -1
                ]
            )

        else:
            candidate = (
                candidates[
                    0
                ]
            )

        return FakeSceneSelection(
            selected=FakeCandidateScore(
                candidate=candidate
            )
        )


def make_instrument() -> SurgicalInstrument:
    """Create deterministic RCM instrument."""

    return SurgicalInstrument(
        rcm_position=np.zeros(
            3,
            dtype=float,
        )
    )


def make_reachable_candidate(
    instrument: SurgicalInstrument,
    q: np.ndarray,
) -> CandidateViewpoint:
    """Create one exactly executable endoscope viewpoint."""

    return CandidateViewpoint(
        pose=endoscope_camera_pose(
            instrument,
            q,
        ),
        radius=0.10,
        azimuth=0.0,
        elevation=0.0,
    )


def first_configuration() -> np.ndarray:
    return np.asarray(
        [
            0.10,
            -0.10,
            0.15,
            0.20,
        ],
        dtype=float,
    )


def second_configuration() -> np.ndarray:
    return np.asarray(
        [
            -0.20,
            0.15,
            0.20,
            -0.30,
        ],
        dtype=float,
    )


def make_impossible_candidate(
    reference: CandidateViewpoint,
) -> CandidateViewpoint:
    """Create candidate outside configured insertion range."""

    return CandidateViewpoint(
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
                reference.pose.rotation
            ),
        ),
        radius=0.10,
        azimuth=0.0,
        elevation=0.0,
    )


def test_unreachable_candidates_are_removed_before_scoring():
    """Perception scorer must never receive robot-impossible candidates."""

    instrument = (
        make_instrument()
    )

    reachable = (
        make_reachable_candidate(
            instrument,
            first_configuration(),
        )
    )

    impossible = (
        make_impossible_candidate(
            reachable
        )
    )

    scorer = (
        RecordingSceneScorer()
    )

    result = (
        select_robot_aware_scene_viewpoint(
            instrument=instrument,
            scorer=scorer,
            current_pose=(
                reachable.pose
            ),
            candidates=(
                impossible,
                reachable,
            ),
            initial_perception=object(),
        )
    )

    assert (
        result.reachability.total_count
        == 2
    )

    assert (
        result.reachability.reachable_count
        == 1
    )

    assert (
        result.reachability.rejected_count
        == 1
    )

    assert (
        scorer.received_candidates
        == (
            reachable,
        )
    )


def test_selected_viewpoint_is_robot_reachable():
    """Returned active viewpoint must have passed feasibility filtering."""

    instrument = (
        make_instrument()
    )

    candidate = (
        make_reachable_candidate(
            instrument,
            first_configuration(),
        )
    )

    result = (
        select_robot_aware_scene_viewpoint(
            instrument=instrument,
            scorer=RecordingSceneScorer(),
            current_pose=(
                candidate.pose
            ),
            candidates=(
                candidate,
            ),
            initial_perception=object(),
        )
    )

    assert (
        result.selected_evaluation
        .reachable
    )

    assert (
        result.selected_evaluation
        .rejection_reasons
        == ()
    )


def test_selected_configuration_reconstructs_camera_position():
    """Returned robot configuration must realise selected camera centre."""

    instrument = (
        make_instrument()
    )

    candidate = (
        make_reachable_candidate(
            instrument,
            first_configuration(),
        )
    )

    result = (
        select_robot_aware_scene_viewpoint(
            instrument=instrument,
            scorer=RecordingSceneScorer(),
            current_pose=(
                candidate.pose
            ),
            candidates=(
                candidate,
            ),
            initial_perception=object(),
        )
    )

    reconstructed_position = (
        instrument.forward_position(
            result.selected_configuration
        )
    )

    np.testing.assert_allclose(
        reconstructed_position,
        candidate.pose.position,
        atol=1e-10,
    )


def test_scorer_can_choose_between_multiple_reachable_candidates():
    """Existing scorer retains control after feasibility filtering."""

    instrument = (
        make_instrument()
    )

    first = (
        make_reachable_candidate(
            instrument,
            first_configuration(),
        )
    )

    second = (
        make_reachable_candidate(
            instrument,
            second_configuration(),
        )
    )

    scorer = (
        RecordingSceneScorer(
            choose_last=True
        )
    )

    result = (
        select_robot_aware_scene_viewpoint(
            instrument=instrument,
            scorer=scorer,
            current_pose=(
                first.pose
            ),
            candidates=(
                first,
                second,
            ),
            initial_perception=object(),
        )
    )

    assert (
        result.selected_viewpoint
        is second
    )

    np.testing.assert_allclose(
        result.selected_configuration,
        second_configuration(),
        atol=1e-10,
    )


def test_no_reachable_candidate_raises_explicit_failure():
    """The system must fail explicitly rather than select impossible motion."""

    instrument = (
        make_instrument()
    )

    reference = (
        make_reachable_candidate(
            instrument,
            first_configuration(),
        )
    )

    impossible = (
        make_impossible_candidate(
            reference
        )
    )

    with pytest.raises(
        NoReachableViewpointError,
        match="No candidate viewpoint",
    ):
        select_robot_aware_scene_viewpoint(
            instrument=instrument,
            scorer=RecordingSceneScorer(),
            current_pose=(
                reference.pose
            ),
            candidates=(
                impossible,
            ),
            initial_perception=object(),
        )


def test_empty_candidate_set_is_rejected():
    """Empty viewpoint collections must fail before scoring."""

    instrument = (
        make_instrument()
    )

    reference = (
        make_reachable_candidate(
            instrument,
            first_configuration(),
        )
    )

    with pytest.raises(
        ValueError,
        match="candidates must not be empty",
    ):
        select_robot_aware_scene_viewpoint(
            instrument=instrument,
            scorer=RecordingSceneScorer(),
            current_pose=(
                reference.pose
            ),
            candidates=(),
            initial_perception=object(),
        )