"""Convert perceived anatomy into structures used by motion planning."""

from __future__ import annotations

from dataclasses import dataclass

from src.geometry.workspace import SphericalStructure
from src.perception.perception import PerceptionResult
from src.perception.uncertainty import inflate_estimated_structures


@dataclass(frozen=True)
class PlanningPerception:
    """Planner-facing representation of perceived anatomy."""

    structures: tuple[SphericalStructure, ...]
    sigma_multiplier: float
    uncertainty_aware: bool


def deterministic_planning_structures(
    perception_result: PerceptionResult,
) -> PlanningPerception:
    """Build planner geometry from noisy estimates without uncertainty inflation.

    The planner uses each estimated anatomical centre and the original base
    safety margin, but does not compensate for localisation uncertainty.

    This provides the deterministic baseline for later comparison.
    """

    structures = tuple(
        SphericalStructure(
            centre=estimate.estimated_centre.copy(),
            physical_radius=estimate.physical_radius,
            safety_margin=estimate.base_safety_margin,
        )
        for estimate in perception_result.estimated_structures
    )

    return PlanningPerception(
        structures=structures,
        sigma_multiplier=0.0,
        uncertainty_aware=False,
    )


def uncertainty_aware_planning_structures(
    perception_result: PerceptionResult,
    sigma_multiplier: float,
) -> PlanningPerception:
    """Build uncertainty-inflated planner geometry.

    For each perceived structure, the planning safety margin becomes:

        base safety margin
        + sigma_multiplier * principal positional standard deviation

    Parameters
    ----------
    perception_result:
        Noisy anatomical perception result.

    sigma_multiplier:
        Number of standard deviations used to inflate the planning margin.

    Returns
    -------
    PlanningPerception
        Structures that can be supplied directly to the motion planner.
    """

    if sigma_multiplier < 0.0:
        raise ValueError(
            "sigma_multiplier must be non-negative."
        )

    structures = inflate_estimated_structures(
        estimated_structures=(
            perception_result.estimated_structures
        ),
        sigma_multiplier=sigma_multiplier,
    )

    return PlanningPerception(
        structures=structures,
        sigma_multiplier=float(
            sigma_multiplier
        ),
        uncertainty_aware=True,
    )


def maximum_planning_safety_radius(
    planning_perception: PlanningPerception,
) -> float:
    """Return the largest protected radius used by the planner."""

    if len(planning_perception.structures) == 0:
        return 0.0

    return float(
        max(
            structure.safety_radius
            for structure in planning_perception.structures
        )
    )


def mean_planning_safety_margin(
    planning_perception: PlanningPerception,
) -> float:
    """Return the mean safety margin used across perceived structures."""

    if len(planning_perception.structures) == 0:
        return 0.0

    return float(
        sum(
            structure.safety_margin
            for structure in planning_perception.structures
        )
        / len(planning_perception.structures)
    )