"""Safety evaluation for the simulated surgical instrument."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry.workspace import SphericalStructure


@dataclass(frozen=True)
class SafetyEvaluation:
    """Safety metrics for one instrument configuration."""

    minimum_surface_clearance: float
    minimum_safety_clearance: float
    collision: bool
    safety_margin_violation: bool


def point_to_segment_distance(
    point: np.ndarray,
    segment_start: np.ndarray,
    segment_end: np.ndarray,
) -> float:
    """Return the shortest distance from a point to a finite 3-D segment."""
    point = np.asarray(point, dtype=float)
    segment_start = np.asarray(segment_start, dtype=float)
    segment_end = np.asarray(segment_end, dtype=float)

    for name, value in (
        ("point", point),
        ("segment_start", segment_start),
        ("segment_end", segment_end),
    ):
        if value.shape != (3,):
            raise ValueError(
                f"{name} must have shape (3,)."
            )

        if not np.all(np.isfinite(value)):
            raise ValueError(
                f"{name} must contain finite values."
            )

    segment_vector = segment_end - segment_start
    segment_length_squared = float(
        np.dot(segment_vector, segment_vector)
    )

    if np.isclose(segment_length_squared, 0.0):
        raise ValueError(
            "segment_start and segment_end cannot coincide."
        )

    projection = float(
        np.dot(
            point - segment_start,
            segment_vector,
        )
        / segment_length_squared
    )

    projection_clamped = np.clip(
        projection,
        0.0,
        1.0,
    )

    closest_point = (
        segment_start
        + projection_clamped * segment_vector
    )

    return float(
        np.linalg.norm(point - closest_point)
    )


def structure_clearance_to_instrument(
    structure: SphericalStructure,
    shaft_start: np.ndarray,
    shaft_end: np.ndarray,
    instrument_radius: float,
) -> tuple[float, float]:
    """Return physical and safety clearance to a spherical structure.

    The instrument is treated as a finite cylindrical shaft represented
    geometrically by its centreline segment plus an instrument radius.

    Returns
    -------
    tuple[float, float]
        Signed physical-surface clearance and signed safety-boundary clearance.
    """
    if instrument_radius <= 0.0:
        raise ValueError(
            "instrument_radius must be positive."
        )

    centreline_distance = point_to_segment_distance(
        structure.centre,
        shaft_start,
        shaft_end,
    )

    surface_clearance = (
        centreline_distance
        - structure.physical_radius
        - instrument_radius
    )

    safety_clearance = (
        centreline_distance
        - structure.safety_radius
        - instrument_radius
    )

    return (
        float(surface_clearance),
        float(safety_clearance),
    )


def evaluate_instrument_safety(
    shaft_start: np.ndarray,
    shaft_end: np.ndarray,
    structures: tuple[SphericalStructure, ...],
    instrument_radius: float,
) -> SafetyEvaluation:
    """Evaluate collision and safety-margin status for the instrument shaft."""
    if instrument_radius <= 0.0:
        raise ValueError(
            "instrument_radius must be positive."
        )

    if len(structures) == 0:
        return SafetyEvaluation(
            minimum_surface_clearance=float("inf"),
            minimum_safety_clearance=float("inf"),
            collision=False,
            safety_margin_violation=False,
        )

    surface_clearances: list[float] = []
    safety_clearances: list[float] = []

    for structure in structures:
        surface_clearance, safety_clearance = (
            structure_clearance_to_instrument(
                structure=structure,
                shaft_start=shaft_start,
                shaft_end=shaft_end,
                instrument_radius=instrument_radius,
            )
        )

        surface_clearances.append(
            surface_clearance
        )

        safety_clearances.append(
            safety_clearance
        )

    minimum_surface_clearance = float(
        min(surface_clearances)
    )

    minimum_safety_clearance = float(
        min(safety_clearances)
    )

    return SafetyEvaluation(
        minimum_surface_clearance=minimum_surface_clearance,
        minimum_safety_clearance=minimum_safety_clearance,
        collision=(
            minimum_surface_clearance <= 0.0
        ),
        safety_margin_violation=(
            minimum_safety_clearance <= 0.0
        ),
    )