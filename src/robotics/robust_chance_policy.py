"""Robustification policy for Phase 6 chance-constrained planning.

The nominal Phase 6 planner assumes that positional uncertainty is adequately
described by a zero-mean Gaussian covariance.

This module introduces explicit robustness allowances for three model failures:

1. covariance underestimation;
2. bounded systematic localisation bias;
3. epsilon-contamination by arbitrary non-Gaussian errors.

The policy retains covariance orientation and therefore does not reduce
anisotropic uncertainty to a scalar principal-sigma margin.

Scientific boundary
-------------------

These parameters are modelling safeguards. They are not clinical safety
factors and do not establish medical-device risk guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.perception.uncertainty import (
    EstimatedStructure,
    PositionUncertainty,
)
from src.robotics.risk_aware_planning import (
    ChanceConstraintConfig,
)


@dataclass(frozen=True)
class RobustChancePolicy:
    """Robustness assumptions applied to the nominal chance model.

    Parameters
    ----------
    covariance_std_scale:
        Multiplicative factor applied to positional standard deviation.
        Covariance is therefore multiplied by the square of this value.

    bias_bound:
        Additional isotropic protected distance in metres used to represent
        a bounded systematic localisation bias.

    contamination_probability:
        Fraction of the error distribution allowed to come from an arbitrary
        contaminating distribution.

    target_violation_probability:
        Desired overall pointwise violation-probability limit.
    """

    covariance_std_scale: float = 1.0

    bias_bound: float = 0.0

    contamination_probability: float = 0.0

    target_violation_probability: float = 0.05

    def __post_init__(
        self,
    ) -> None:
        if (
            not np.isfinite(
                self.covariance_std_scale
            )
            or self.covariance_std_scale
            < 1.0
        ):
            raise ValueError(
                "covariance_std_scale must be finite and at least 1."
            )

        if (
            not np.isfinite(
                self.bias_bound
            )
            or self.bias_bound < 0.0
        ):
            raise ValueError(
                "bias_bound must be finite and non-negative."
            )

        if (
            not np.isfinite(
                self.target_violation_probability
            )
            or not (
                0.0
                < self.target_violation_probability
                < 1.0
            )
        ):
            raise ValueError(
                "target_violation_probability must lie strictly in (0, 1)."
            )

        if (
            not np.isfinite(
                self.contamination_probability
            )
            or self.contamination_probability < 0.0
            or self.contamination_probability
            >= self.target_violation_probability
        ):
            raise ValueError(
                "contamination_probability must be non-negative "
                "and smaller than target_violation_probability."
            )

    @property
    def effective_gaussian_threshold(
        self,
    ) -> float:
        """Return Gaussian risk budget after epsilon contamination.

        For

            P = (1 - epsilon) P_G + epsilon Q

        with arbitrary contaminating distribution Q,

            P(violation)
            <= epsilon + (1 - epsilon) p_G.

        Requiring this upper bound to remain below alpha gives

            p_G <= (alpha - epsilon) / (1 - epsilon).
        """

        alpha = float(
            self.target_violation_probability
        )

        epsilon = float(
            self.contamination_probability
        )

        return float(
            (
                alpha
                - epsilon
            )
            / (
                1.0
                - epsilon
            )
        )


def robustify_estimated_structure(
    estimate: EstimatedStructure,
    policy: RobustChancePolicy,
) -> EstimatedStructure:
    """Apply covariance and bounded-bias robustness allowances."""

    if not isinstance(
        estimate,
        EstimatedStructure,
    ):
        raise TypeError(
            "estimate must be an EstimatedStructure."
        )

    if not isinstance(
        policy,
        RobustChancePolicy,
    ):
        raise TypeError(
            "policy must be a RobustChancePolicy."
        )

    covariance_scale = (
        policy.covariance_std_scale
        ** 2
    )

    robust_covariance = (
        covariance_scale
        * np.asarray(
            estimate.uncertainty.covariance,
            dtype=float,
        )
    )

    return EstimatedStructure(
        estimated_centre=np.array(
            estimate.estimated_centre,
            dtype=float,
            copy=True,
        ),
        physical_radius=float(
            estimate.physical_radius
        ),
        base_safety_margin=float(
            estimate.base_safety_margin
            + policy.bias_bound
        ),
        uncertainty=PositionUncertainty(
            covariance=robust_covariance
        ),
    )


def robustify_estimated_structures(
    estimates: Sequence[
        EstimatedStructure
    ],
    policy: RobustChancePolicy,
) -> tuple[
    EstimatedStructure,
    ...,
]:
    """Apply a robustness policy to all perceived structures."""

    return tuple(
        robustify_estimated_structure(
            estimate,
            policy,
        )
        for estimate in estimates
    )


def robust_chance_constraint_config(
    policy: RobustChancePolicy,
    *,
    sample_spacing: float = 0.003,
    sigma_epsilon: float = 1e-12,
    geometry_epsilon: float = 1e-12,
) -> ChanceConstraintConfig:
    """Create the corresponding chance-constraint configuration."""

    return ChanceConstraintConfig(
        max_point_violation_probability=(
            policy.effective_gaussian_threshold
        ),
        max_path_union_bound_probability=None,
        sample_spacing=float(
            sample_spacing
        ),
        sigma_epsilon=float(
            sigma_epsilon
        ),
        geometry_epsilon=float(
            geometry_epsilon
        ),
    )