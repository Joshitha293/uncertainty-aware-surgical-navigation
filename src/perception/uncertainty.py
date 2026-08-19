"""Uncertainty models for simulated anatomical localisation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry.workspace import SphericalStructure


@dataclass(frozen=True)
class PositionUncertainty:
    """Gaussian positional uncertainty for a 3-D anatomical estimate.

    Parameters
    ----------
    covariance:
        3 x 3 covariance matrix describing uncertainty in metres squared.
    """

    covariance: np.ndarray

    def __post_init__(self) -> None:
        covariance = np.asarray(
            self.covariance,
            dtype=float,
        )

        if covariance.shape != (3, 3):
            raise ValueError(
                "covariance must have shape (3, 3)."
            )

        if not np.all(np.isfinite(covariance)):
            raise ValueError(
                "covariance must contain finite values."
            )

        if not np.allclose(
            covariance,
            covariance.T,
            atol=1e-12,
        ):
            raise ValueError(
                "covariance must be symmetric."
            )

        eigenvalues = np.linalg.eigvalsh(
            covariance
        )

        if np.any(eigenvalues < -1e-12):
            raise ValueError(
                "covariance must be positive semi-definite."
            )

        object.__setattr__(
            self,
            "covariance",
            covariance,
        )

    @classmethod
    def isotropic(
        cls,
        sigma: float,
    ) -> "PositionUncertainty":
        """Create isotropic positional uncertainty.

        Parameters
        ----------
        sigma:
            Standard deviation in metres.
        """

        if sigma < 0.0:
            raise ValueError(
                "sigma must be non-negative."
            )

        covariance = (
            sigma**2
        ) * np.eye(
            3,
            dtype=float,
        )

        return cls(
            covariance=covariance
        )

    @property
    def principal_sigma(self) -> float:
        """Return the largest positional standard deviation."""

        largest_variance = float(
            np.max(
                np.linalg.eigvalsh(
                    self.covariance
                )
            )
        )

        return float(
            np.sqrt(
                max(
                    largest_variance,
                    0.0,
                )
            )
        )


@dataclass(frozen=True)
class EstimatedStructure:
    """Perceived anatomical structure with localisation uncertainty."""

    estimated_centre: np.ndarray
    physical_radius: float
    base_safety_margin: float
    uncertainty: PositionUncertainty

    def __post_init__(self) -> None:
        estimated_centre = np.asarray(
            self.estimated_centre,
            dtype=float,
        )

        if estimated_centre.shape != (3,):
            raise ValueError(
                "estimated_centre must have shape (3,)."
            )

        if not np.all(
            np.isfinite(
                estimated_centre
            )
        ):
            raise ValueError(
                "estimated_centre must contain finite values."
            )

        if self.physical_radius <= 0.0:
            raise ValueError(
                "physical_radius must be positive."
            )

        if self.base_safety_margin < 0.0:
            raise ValueError(
                "base_safety_margin must be non-negative."
            )

        object.__setattr__(
            self,
            "estimated_centre",
            estimated_centre,
        )

    def uncertainty_aware_structure(
        self,
        sigma_multiplier: float,
    ) -> SphericalStructure:
        """Convert the estimate into an uncertainty-inflated structure.

        The protected margin is:

            base_margin + sigma_multiplier * principal_sigma

        Parameters
        ----------
        sigma_multiplier:
            Number of standard deviations used to inflate the margin.
        """

        if sigma_multiplier < 0.0:
            raise ValueError(
                "sigma_multiplier must be non-negative."
            )

        uncertainty_margin = (
            sigma_multiplier
            * self.uncertainty.principal_sigma
        )

        total_margin = (
            self.base_safety_margin
            + uncertainty_margin
        )

        return SphericalStructure(
            centre=self.estimated_centre.copy(),
            physical_radius=self.physical_radius,
            safety_margin=total_margin,
        )


def sample_noisy_centre(
    true_centre: np.ndarray,
    uncertainty: PositionUncertainty,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a noisy anatomical position estimate."""

    true_centre = np.asarray(
        true_centre,
        dtype=float,
    )

    if true_centre.shape != (3,):
        raise ValueError(
            "true_centre must have shape (3,)."
        )

    if not np.all(
        np.isfinite(
            true_centre
        )
    ):
        raise ValueError(
            "true_centre must contain finite values."
        )

    error = rng.multivariate_normal(
        mean=np.zeros(
            3,
            dtype=float,
        ),
        cov=uncertainty.covariance,
    )

    return (
        true_centre
        + error
    )


def make_estimated_structure(
    true_structure: SphericalStructure,
    uncertainty: PositionUncertainty,
    rng: np.random.Generator,
) -> EstimatedStructure:
    """Generate one noisy perception of a true anatomical structure."""

    estimated_centre = sample_noisy_centre(
        true_centre=true_structure.centre,
        uncertainty=uncertainty,
        rng=rng,
    )

    return EstimatedStructure(
        estimated_centre=estimated_centre,
        physical_radius=(
            true_structure.physical_radius
        ),
        base_safety_margin=(
            true_structure.safety_margin
        ),
        uncertainty=uncertainty,
    )


def inflate_estimated_structures(
    estimated_structures: tuple[
        EstimatedStructure,
        ...
    ],
    sigma_multiplier: float,
) -> tuple[SphericalStructure, ...]:
    """Create planner structures with uncertainty-aware safety margins."""

    return tuple(
        structure.uncertainty_aware_structure(
            sigma_multiplier
        )
        for structure in estimated_structures
    )