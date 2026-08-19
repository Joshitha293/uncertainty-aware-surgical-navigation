"""Geometric representation of the simulated surgical workspace."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SphericalStructure:
    """Simplified safety-critical anatomical structure.

    Parameters
    ----------
    centre:
        Ground-truth centre of the structure in world coordinates.

    physical_radius:
        Radius representing the physical structure.

    safety_margin:
        Additional protected distance surrounding the physical structure.
    """

    centre: np.ndarray
    physical_radius: float
    safety_margin: float

    def __post_init__(self) -> None:
        centre = np.asarray(self.centre, dtype=float)

        if centre.shape != (3,):
            raise ValueError(
                "centre must have shape (3,)."
            )

        if not np.all(np.isfinite(centre)):
            raise ValueError(
                "centre must contain finite values."
            )

        if self.physical_radius <= 0.0:
            raise ValueError(
                "physical_radius must be positive."
            )

        if self.safety_margin < 0.0:
            raise ValueError(
                "safety_margin must be non-negative."
            )

        object.__setattr__(
            self,
            "centre",
            centre,
        )

    @property
    def safety_radius(self) -> float:
        """Return the total protected radius."""
        return (
            self.physical_radius
            + self.safety_margin
        )

    def signed_surface_clearance(
        self,
        point: np.ndarray,
    ) -> float:
        """Return signed distance from a point to the physical surface.

        Positive:
            Point lies outside the structure.

        Zero:
            Point lies on the surface.

        Negative:
            Point lies inside the structure.
        """
        point = np.asarray(point, dtype=float)

        if point.shape != (3,):
            raise ValueError(
                "point must have shape (3,)."
            )

        distance_to_centre = np.linalg.norm(
            point - self.centre
        )

        return float(
            distance_to_centre
            - self.physical_radius
        )

    def signed_safety_clearance(
        self,
        point: np.ndarray,
    ) -> float:
        """Return signed distance from a point to the safety boundary."""
        point = np.asarray(point, dtype=float)

        if point.shape != (3,):
            raise ValueError(
                "point must have shape (3,)."
            )

        distance_to_centre = np.linalg.norm(
            point - self.centre
        )

        return float(
            distance_to_centre
            - self.safety_radius
        )

    def contains_point(
        self,
        point: np.ndarray,
    ) -> bool:
        """Check whether a point lies inside the physical structure."""
        return bool(
            self.signed_surface_clearance(point) <= 0.0
        )

    def violates_safety_margin(
        self,
        point: np.ndarray,
    ) -> bool:
        """Check whether a point enters the protected safety region."""
        return bool(
            self.signed_safety_clearance(point) <= 0.0
        )


@dataclass(frozen=True)
class SurgicalWorkspace:
    """Container for a simplified surgical planning environment."""

    target_position: np.ndarray
    structures: tuple[SphericalStructure, ...]

    def __post_init__(self) -> None:
        target = np.asarray(
            self.target_position,
            dtype=float,
        )

        if target.shape != (3,):
            raise ValueError(
                "target_position must have shape (3,)."
            )

        if not np.all(np.isfinite(target)):
            raise ValueError(
                "target_position must contain finite values."
            )

        object.__setattr__(
            self,
            "target_position",
            target,
        )

    def nearest_surface_clearance(
        self,
        point: np.ndarray,
    ) -> float:
        """Return clearance to the nearest physical structure."""
        if len(self.structures) == 0:
            return float("inf")

        clearances = [
            structure.signed_surface_clearance(point)
            for structure in self.structures
        ]

        return float(min(clearances))

    def nearest_safety_clearance(
        self,
        point: np.ndarray,
    ) -> float:
        """Return clearance to the nearest protected safety boundary."""
        if len(self.structures) == 0:
            return float("inf")

        clearances = [
            structure.signed_safety_clearance(point)
            for structure in self.structures
        ]

        return float(min(clearances))

    def point_in_collision(
        self,
        point: np.ndarray,
    ) -> bool:
        """Return True if a point lies inside any physical structure."""
        return any(
            structure.contains_point(point)
            for structure in self.structures
        )

    def point_violates_safety_margin(
        self,
        point: np.ndarray,
    ) -> bool:
        """Return True if a point enters any protected region."""
        return any(
            structure.violates_safety_margin(point)
            for structure in self.structures
        )