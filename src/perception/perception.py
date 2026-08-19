"""Simulated anatomical perception under localisation uncertainty."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry.workspace import SphericalStructure
from src.perception.uncertainty import (
    EstimatedStructure,
    PositionUncertainty,
    make_estimated_structure,
)


@dataclass(frozen=True)
class PerceptionResult:
    """Output of one simulated anatomical perception step."""

    estimated_structures: tuple[EstimatedStructure, ...]
    localisation_errors: np.ndarray


def perceive_structures(
    true_structures: tuple[SphericalStructure, ...],
    uncertainty: PositionUncertainty,
    rng: np.random.Generator,
) -> PerceptionResult:
    """Generate noisy anatomical structure estimates.

    Each ground-truth structure is perturbed independently using the
    specified positional uncertainty model.

    Parameters
    ----------
    true_structures:
        Ground-truth simulated anatomical structures.

    uncertainty:
        Positional uncertainty model applied to each structure.

    rng:
        NumPy random generator used for reproducibility.

    Returns
    -------
    PerceptionResult
        Estimated structures and Euclidean localisation errors.
    """

    estimates: list[EstimatedStructure] = []
    errors: list[float] = []

    for true_structure in true_structures:
        estimate = make_estimated_structure(
            true_structure=true_structure,
            uncertainty=uncertainty,
            rng=rng,
        )

        localisation_error = float(
            np.linalg.norm(
                estimate.estimated_centre
                - true_structure.centre
            )
        )

        estimates.append(
            estimate
        )

        errors.append(
            localisation_error
        )

    return PerceptionResult(
        estimated_structures=tuple(estimates),
        localisation_errors=np.asarray(
            errors,
            dtype=float,
        ),
    )


def mean_localisation_error(
    result: PerceptionResult,
) -> float:
    """Return mean Euclidean localisation error across structures."""

    if len(result.localisation_errors) == 0:
        return 0.0

    return float(
        np.mean(
            result.localisation_errors
        )
    )


def maximum_localisation_error(
    result: PerceptionResult,
) -> float:
    """Return maximum Euclidean localisation error across structures."""

    if len(result.localisation_errors) == 0:
        return 0.0

    return float(
        np.max(
            result.localisation_errors
        )
    )