"""Test the Phase 8 ROS 2 coordinate-frame contract."""

from surgical_navigation_ros.frame_contract import (
    ALL_FRAMES,
    CAMERA_FRAME,
    frame_parent_map,
    FRAME_RELATIONSHIPS,
    INSTRUMENT_FRAME,
    INSTRUMENT_TIP_FRAME,
    PATIENT_REFERENCE_FRAME,
    PROTECTED_STRUCTURE_FRAME,
    ROBOT_BASE_FRAME,
    TARGET_FRAME,
    validate_frame_contract,
    WORLD_FRAME,
)


def test_frame_contract_is_valid() -> None:
    """Confirm that the frozen TF contract passes validation."""
    validate_frame_contract()


def test_world_is_root() -> None:
    """Confirm that world has no parent frame."""
    parent_map = frame_parent_map()

    assert WORLD_FRAME not in parent_map


def test_robot_chain_is_correct() -> None:
    """Confirm the base-to-instrument robot-frame hierarchy."""
    parent_map = frame_parent_map()

    assert (
        parent_map[ROBOT_BASE_FRAME]
        == WORLD_FRAME
    )

    assert (
        parent_map[INSTRUMENT_FRAME]
        == ROBOT_BASE_FRAME
    )

    assert (
        parent_map[INSTRUMENT_TIP_FRAME]
        == INSTRUMENT_FRAME
    )


def test_camera_is_world_referenced() -> None:
    """Confirm that the camera branch originates from world."""
    parent_map = frame_parent_map()

    assert (
        parent_map[CAMERA_FRAME]
        == WORLD_FRAME
    )


def test_patient_reference_is_world_referenced() -> None:
    """Confirm that the patient reference is registered to world."""
    parent_map = frame_parent_map()

    assert (
        parent_map[PATIENT_REFERENCE_FRAME]
        == WORLD_FRAME
    )


def test_navigation_targets_use_patient_reference() -> None:
    """Confirm that targets use patient-reference coordinates."""
    parent_map = frame_parent_map()

    assert (
        parent_map[TARGET_FRAME]
        == PATIENT_REFERENCE_FRAME
    )

    assert (
        parent_map[PROTECTED_STRUCTURE_FRAME]
        == PATIENT_REFERENCE_FRAME
    )


def test_every_non_root_frame_has_one_parent() -> None:
    """Confirm that each non-root frame has one incoming TF edge."""
    children = [
        relationship.child
        for relationship in FRAME_RELATIONSHIPS
    ]

    assert len(children) == len(set(children))

    assert set(children) == (
        ALL_FRAMES
        - {
            WORLD_FRAME
        }
    )


def test_frame_names_are_unique() -> None:
    """Confirm that semantic frame identifiers are globally unique."""
    assert len(ALL_FRAMES) == 8
