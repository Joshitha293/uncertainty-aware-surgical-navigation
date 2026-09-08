"""Define the ROS 2 surgical-navigation coordinate-frame contract."""

from __future__ import annotations

from dataclasses import dataclass


WORLD_FRAME = 'world'

ROBOT_BASE_FRAME = 'robot_base'
INSTRUMENT_FRAME = 'instrument'
INSTRUMENT_TIP_FRAME = 'instrument_tip'

CAMERA_FRAME = 'camera'

PATIENT_REFERENCE_FRAME = 'patient_reference'
TARGET_FRAME = 'target'
PROTECTED_STRUCTURE_FRAME = 'protected_structure'


ALL_FRAMES = frozenset(
    {
        WORLD_FRAME,
        ROBOT_BASE_FRAME,
        INSTRUMENT_FRAME,
        INSTRUMENT_TIP_FRAME,
        CAMERA_FRAME,
        PATIENT_REFERENCE_FRAME,
        TARGET_FRAME,
        PROTECTED_STRUCTURE_FRAME,
    }
)


@dataclass(frozen=True)
class FrameRelationship:
    """Represent one directed TF parent-child relationship."""

    parent: str
    child: str
    owner: str

    def __post_init__(self) -> None:
        """Validate the declared frame relationship."""
        if not self.parent:
            raise ValueError(
                'parent frame must be non-empty.'
            )

        if not self.child:
            raise ValueError(
                'child frame must be non-empty.'
            )

        if self.parent == self.child:
            raise ValueError(
                'A TF frame cannot be its own parent.'
            )

        if not self.owner:
            raise ValueError(
                'owner must be non-empty.'
            )


FRAME_RELATIONSHIPS = (
    FrameRelationship(
        parent=WORLD_FRAME,
        child=ROBOT_BASE_FRAME,
        owner='robot_description_or_simulator',
    ),
    FrameRelationship(
        parent=ROBOT_BASE_FRAME,
        child=INSTRUMENT_FRAME,
        owner='robot_state_publisher',
    ),
    FrameRelationship(
        parent=INSTRUMENT_FRAME,
        child=INSTRUMENT_TIP_FRAME,
        owner='robot_state_publisher',
    ),
    FrameRelationship(
        parent=WORLD_FRAME,
        child=CAMERA_FRAME,
        owner='camera_calibration_or_simulator',
    ),
    FrameRelationship(
        parent=WORLD_FRAME,
        child=PATIENT_REFERENCE_FRAME,
        owner='registration_or_simulator',
    ),
    FrameRelationship(
        parent=PATIENT_REFERENCE_FRAME,
        child=TARGET_FRAME,
        owner='navigation_system',
    ),
    FrameRelationship(
        parent=PATIENT_REFERENCE_FRAME,
        child=PROTECTED_STRUCTURE_FRAME,
        owner='navigation_system',
    ),
)


def frame_parent_map() -> dict[str, str]:
    """Return the child-to-parent mapping for the frame contract."""
    return {
        relationship.child: relationship.parent
        for relationship in FRAME_RELATIONSHIPS
    }


def validate_frame_contract() -> None:
    """Validate uniqueness and connectivity of the TF frame contract."""
    parent_map = frame_parent_map()

    children = [
        relationship.child
        for relationship in FRAME_RELATIONSHIPS
    ]

    if len(children) != len(set(children)):
        raise ValueError(
            'Each TF child frame must have exactly one parent.'
        )

    if WORLD_FRAME in parent_map:
        raise ValueError(
            'The world frame must remain the root frame.'
        )

    referenced_frames = {
        WORLD_FRAME
    }

    for relationship in FRAME_RELATIONSHIPS:
        referenced_frames.add(
            relationship.parent
        )
        referenced_frames.add(
            relationship.child
        )

    if referenced_frames != ALL_FRAMES:
        raise ValueError(
            'Frame contract does not reference exactly the declared frames.'
        )

    for frame in ALL_FRAMES:
        if frame == WORLD_FRAME:
            continue

        visited: set[str] = set()
        current = frame

        while current != WORLD_FRAME:
            if current in visited:
                raise ValueError(
                    f'Cycle detected in TF frame contract at {current!r}.'
                )

            visited.add(
                current
            )

            if current not in parent_map:
                raise ValueError(
                    f'Frame {current!r} is disconnected from world.'
                )

            current = parent_map[
                current
            ]


validate_frame_contract()
