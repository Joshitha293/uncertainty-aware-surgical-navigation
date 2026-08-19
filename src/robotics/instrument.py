"""Kinematic model of an RCM-constrained laparoscopic surgical instrument."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointLimits:
    """Simulation joint limits for the simplified instrument."""

    yaw_min: float = np.deg2rad(-60.0)
    yaw_max: float = np.deg2rad(60.0)

    pitch_min: float = np.deg2rad(-45.0)
    pitch_max: float = np.deg2rad(45.0)

    insertion_min: float = 0.05
    insertion_max: float = 0.30

    roll_min: float = np.deg2rad(-180.0)
    roll_max: float = np.deg2rad(180.0)


class SurgicalInstrument:
    """Simplified 4-DOF laparoscopic instrument.

    Joint vector:

        q = [yaw, pitch, insertion, roll]

    The instrument shaft is constrained to pass through a fixed
    remote centre of motion (RCM).
    """

    def __init__(
        self,
        rcm_position: np.ndarray,
        joint_limits: JointLimits | None = None,
    ) -> None:
        """Create an RCM-constrained surgical instrument."""
        self.rcm_position = np.asarray(
            rcm_position,
            dtype=float,
        )

        if self.rcm_position.shape != (3,):
            raise ValueError(
                "rcm_position must have shape (3,)."
            )

        self.joint_limits = (
            joint_limits
            if joint_limits is not None
            else JointLimits()
        )

    def validate_configuration(
        self,
        q: np.ndarray,
    ) -> None:
        """Validate a 4-DOF joint configuration."""
        q = np.asarray(q, dtype=float)

        if q.shape != (4,):
            raise ValueError(
                "Joint configuration q must have shape (4,)."
            )

        if not np.all(np.isfinite(q)):
            raise ValueError(
                "Joint configuration must contain finite values."
            )

        yaw, pitch, insertion, roll = q
        limits = self.joint_limits

        if not limits.yaw_min <= yaw <= limits.yaw_max:
            raise ValueError(
                "Yaw joint is outside configured limits."
            )

        if not limits.pitch_min <= pitch <= limits.pitch_max:
            raise ValueError(
                "Pitch joint is outside configured limits."
            )

        if not (
            limits.insertion_min
            <= insertion
            <= limits.insertion_max
        ):
            raise ValueError(
                "Insertion joint is outside configured limits."
            )

        if not limits.roll_min <= roll <= limits.roll_max:
            raise ValueError(
                "Roll joint is outside configured limits."
            )

    @staticmethod
    def shaft_direction(
        yaw: float,
        pitch: float,
    ) -> np.ndarray:
        """Return the unit direction of the instrument shaft."""
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
        q: np.ndarray,
    ) -> np.ndarray:
        """Calculate the tool-tip position."""
        q = np.asarray(q, dtype=float)

        self.validate_configuration(q)

        yaw, pitch, insertion, _roll = q

        direction = self.shaft_direction(
            yaw,
            pitch,
        )

        return (
            self.rcm_position
            + insertion * direction
        )

    def inverse_position(
        self,
        target_position: np.ndarray,
        roll: float = 0.0,
    ) -> np.ndarray:
        """Calculate joint values required to reach a target."""
        target_position = np.asarray(
            target_position,
            dtype=float,
        )

        if target_position.shape != (3,):
            raise ValueError(
                "target_position must have shape (3,)."
            )

        if not np.all(np.isfinite(target_position)):
            raise ValueError(
                "target_position must contain finite values."
            )

        displacement = (
            target_position
            - self.rcm_position
        )

        insertion = np.linalg.norm(
            displacement
        )

        if np.isclose(insertion, 0.0):
            raise ValueError(
                "Target cannot coincide with the RCM."
            )

        dx, dy, dz = displacement

        yaw = np.arctan2(
            dy,
            dx,
        )

        horizontal_distance = np.hypot(
            dx,
            dy,
        )

        pitch = np.arctan2(
            dz,
            horizontal_distance,
        )

        q = np.array(
            [
                yaw,
                pitch,
                insertion,
                roll,
            ],
            dtype=float,
        )

        self.validate_configuration(q)

        return q

    def shaft_segment(
        self,
        q: np.ndarray,
        proximal_length: float = 0.10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return two points defining the instrument shaft.

        The segment extends from a point outside the body,
        through the RCM, to the tool tip.

        Parameters
        ----------
        q:
            Joint vector [yaw, pitch, insertion, roll].

        proximal_length:
            Length of shaft represented on the proximal side
            of the RCM.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Proximal shaft point and tool-tip point.
        """
        q = np.asarray(q, dtype=float)

        self.validate_configuration(q)

        if proximal_length <= 0.0:
            raise ValueError(
                "proximal_length must be positive."
            )

        yaw, pitch, _insertion, _roll = q

        direction = self.shaft_direction(
            yaw,
            pitch,
        )

        proximal_point = (
            self.rcm_position
            - proximal_length * direction
        )

        tip_point = self.forward_position(q)

        return proximal_point, tip_point

    @staticmethod
    def point_to_line_distance(
        point: np.ndarray,
        line_start: np.ndarray,
        line_end: np.ndarray,
    ) -> float:
        """Calculate perpendicular distance from a point to a 3-D line."""
        point = np.asarray(point, dtype=float)
        line_start = np.asarray(line_start, dtype=float)
        line_end = np.asarray(line_end, dtype=float)

        if (
            point.shape != (3,)
            or line_start.shape != (3,)
            or line_end.shape != (3,)
        ):
            raise ValueError(
                "All points must have shape (3,)."
            )

        line_vector = line_end - line_start
        line_length = np.linalg.norm(line_vector)

        if np.isclose(line_length, 0.0):
            raise ValueError(
                "Line start and end cannot coincide."
            )

        displacement = point - line_start

        cross_product = np.cross(
            displacement,
            line_vector,
        )

        distance = (
            np.linalg.norm(cross_product)
            / line_length
        )

        return float(distance)

    def rcm_error(
        self,
        q: np.ndarray,
        proximal_length: float = 0.10,
    ) -> float:
        """Calculate the geometric RCM constraint error.

        RCM error is defined as the shortest perpendicular distance
        between the fixed RCM and the current instrument shaft axis.

        For an ideal RCM-constrained configuration:

            rcm_error = 0
        """
        proximal_point, tip_point = (
            self.shaft_segment(
                q,
                proximal_length=proximal_length,
            )
        )

        return self.point_to_line_distance(
            self.rcm_position,
            proximal_point,
            tip_point,
        )

    def satisfies_rcm_constraint(
        self,
        q: np.ndarray,
        tolerance: float = 1e-8,
    ) -> bool:
        """Check whether the shaft satisfies the RCM constraint."""
        if tolerance < 0.0:
            raise ValueError(
                "tolerance must be non-negative."
            )

        error = self.rcm_error(q)

        return bool(error <= tolerance)