"""Kinematic model of a simplified laparoscopic surgical instrument."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


@dataclass(frozen=True)
class JointLimits:
    """Joint limits for the simplified surgical instrument."""

    yaw_min: float = np.deg2rad(-60.0)
    yaw_max: float = np.deg2rad(60.0)

    pitch_min: float = np.deg2rad(-45.0)
    pitch_max: float = np.deg2rad(45.0)

    insertion_min: float = 0.05
    insertion_max: float = 0.30

    roll_min: float = np.deg2rad(-180.0)
    roll_max: float = np.deg2rad(180.0)


class SurgicalInstrument:
    """Simplified 4-DOF laparoscopic surgical instrument.

    Degrees of freedom
    ------------------
    q1 : yaw about the remote centre of motion
    q2 : pitch about the remote centre of motion
    q3 : insertion along the shaft axis
    q4 : roll about the shaft axis
    """

    def __init__(
        self,
        rcm_position: Array,
        joint_limits: JointLimits | None = None,
    ) -> None:
        self.rcm_position = np.asarray(rcm_position, dtype=float)

        if self.rcm_position.shape != (3,):
            raise ValueError(
                "rcm_position must have shape (3,)."
            )

        self.joint_limits = joint_limits or JointLimits()

    def validate_configuration(
        self,
        q: Array,
    ) -> None:
        """Validate a joint configuration against shape and joint limits."""
        q = np.asarray(q, dtype=float)

        if q.shape != (4,):
            raise ValueError(
                "Joint configuration q must have shape (4,)."
            )

        yaw, pitch, insertion, roll = q
        limits = self.joint_limits

        if not limits.yaw_min <= yaw <= limits.yaw_max:
            raise ValueError("Yaw joint is outside configured limits.")

        if not limits.pitch_min <= pitch <= limits.pitch_max:
            raise ValueError("Pitch joint is outside configured limits.")

        if not limits.insertion_min <= insertion <= limits.insertion_max:
            raise ValueError("Insertion joint is outside configured limits.")

        if not limits.roll_min <= roll <= limits.roll_max:
            raise ValueError("Roll joint is outside configured limits.")

    @staticmethod
    def shaft_direction(
        yaw: float,
        pitch: float,
    ) -> Array:
        """Return the unit shaft direction for yaw and pitch.

        The direction vector is:

            [cos(pitch) cos(yaw),
             cos(pitch) sin(yaw),
             sin(pitch)]
        """
        direction = np.array(
            [
                np.cos(pitch) * np.cos(yaw),
                np.cos(pitch) * np.sin(yaw),
                np.sin(pitch),
            ],
            dtype=float,
        )

        return direction

    def forward_position(
        self,
        q: Array,
    ) -> Array:
        """Compute the tool-tip position from joint configuration.

        Parameters
        ----------
        q:
            Joint vector [yaw, pitch, insertion, roll].

        Returns
        -------
        numpy.ndarray
            Tool-tip Cartesian position in the same frame as the RCM.
        """
        q = np.asarray(q, dtype=float)

        self.validate_configuration(q)

        yaw, pitch, insertion, _roll = q

        direction = self.shaft_direction(
            yaw,
            pitch,
        )

        tip_position = (
            self.rcm_position
            + insertion * direction
        )

        return tip_position
