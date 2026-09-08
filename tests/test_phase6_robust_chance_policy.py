"""Tests for the Phase 6 robust chance policy."""

from __future__ import annotations

import numpy as np
import pytest

from src.perception.uncertainty import (
    EstimatedStructure,
    PositionUncertainty,
)
from src.robotics.robust_chance_policy import (
    RobustChancePolicy,
    robust_chance_constraint_config,
    robustify_estimated_structure,
    robustify_estimated_structures,
)


def _estimate() -> EstimatedStructure:
    return EstimatedStructure(
        estimated_centre=np.array(
            [
                0.1,
                -0.02,
                0.03,
            ],
            dtype=float,
        ),
        physical_radius=0.015,
        base_safety_margin=0.008,
        uncertainty=PositionUncertainty(
            covariance=np.diag(
                [
                    1e-6,
                    4e-6,
                    100e-6,
                ]
            )
        ),
    )


def test_default_policy_preserves_covariance() -> None:
    estimate = _estimate()

    robust = (
        robustify_estimated_structure(
            estimate,
            RobustChancePolicy(),
        )
    )

    assert np.allclose(
        robust.uncertainty.covariance,
        estimate.uncertainty.covariance,
    )


def test_covariance_standard_deviation_scale_is_squared() -> None:
    estimate = _estimate()

    robust = (
        robustify_estimated_structure(
            estimate,
            RobustChancePolicy(
                covariance_std_scale=2.0
            ),
        )
    )

    assert np.allclose(
        robust.uncertainty.covariance,
        4.0
        * estimate.uncertainty.covariance,
    )


def test_covariance_scaling_preserves_eigenvectors() -> None:
    covariance = np.array(
        [
            [
                5e-6,
                2e-6,
                0.0,
            ],
            [
                2e-6,
                5e-6,
                0.0,
            ],
            [
                0.0,
                0.0,
                1e-6,
            ],
        ],
        dtype=float,
    )

    estimate = EstimatedStructure(
        estimated_centre=np.zeros(
            3
        ),
        physical_radius=0.01,
        base_safety_margin=0.005,
        uncertainty=PositionUncertainty(
            covariance=covariance
        ),
    )

    robust = (
        robustify_estimated_structure(
            estimate,
            RobustChancePolicy(
                covariance_std_scale=1.5
            ),
        )
    )

    original_values, original_vectors = (
        np.linalg.eigh(
            covariance
        )
    )

    robust_values, robust_vectors = (
        np.linalg.eigh(
            robust.uncertainty.covariance
        )
    )

    assert np.allclose(
        robust_values,
        original_values
        * 1.5**2,
    )

    # Eigenvector signs are arbitrary, so compare absolute alignment.
    alignment = np.abs(
        original_vectors.T
        @ robust_vectors
    )

    assert np.allclose(
        alignment,
        np.eye(
            3
        ),
        atol=1e-8,
    )


def test_bias_bound_increases_base_safety_margin() -> None:
    estimate = _estimate()

    robust = (
        robustify_estimated_structure(
            estimate,
            RobustChancePolicy(
                bias_bound=0.004
            ),
        )
    )

    assert np.isclose(
        robust.base_safety_margin,
        estimate.base_safety_margin
        + 0.004,
    )


def test_robustification_does_not_move_estimated_centre() -> None:
    estimate = _estimate()

    robust = (
        robustify_estimated_structure(
            estimate,
            RobustChancePolicy(
                covariance_std_scale=1.5,
                bias_bound=0.003,
            ),
        )
    )

    assert np.allclose(
        robust.estimated_centre,
        estimate.estimated_centre,
    )


def test_effective_threshold_matches_epsilon_contamination_bound() -> None:
    policy = RobustChancePolicy(
        contamination_probability=0.01,
        target_violation_probability=0.05,
    )

    expected = (
        0.05
        - 0.01
    ) / (
        1.0
        - 0.01
    )

    assert np.isclose(
        policy.effective_gaussian_threshold,
        expected,
    )


def test_contamination_tightens_gaussian_threshold() -> None:
    nominal = RobustChancePolicy(
        contamination_probability=0.0,
        target_violation_probability=0.05,
    )

    contaminated = RobustChancePolicy(
        contamination_probability=0.02,
        target_violation_probability=0.05,
    )

    assert (
        contaminated.effective_gaussian_threshold
        < nominal.effective_gaussian_threshold
    )


def test_generated_chance_config_uses_effective_threshold() -> None:
    policy = RobustChancePolicy(
        contamination_probability=0.01,
        target_violation_probability=0.05,
    )

    config = (
        robust_chance_constraint_config(
            policy,
            sample_spacing=0.004,
        )
    )

    assert np.isclose(
        config.max_point_violation_probability,
        policy.effective_gaussian_threshold,
    )

    assert np.isclose(
        config.sample_spacing,
        0.004,
    )


def test_multiple_estimates_are_robustified() -> None:
    estimates = (
        _estimate(),
        _estimate(),
    )

    robust = (
        robustify_estimated_structures(
            estimates,
            RobustChancePolicy(
                covariance_std_scale=1.25
            ),
        )
    )

    assert len(
        robust
    ) == 2

    for result, original in zip(
        robust,
        estimates,
    ):
        assert np.allclose(
            result.uncertainty.covariance,
            original.uncertainty.covariance
            * 1.25**2,
        )


def test_covariance_scale_below_one_is_rejected() -> None:
    with pytest.raises(
        ValueError
    ):
        RobustChancePolicy(
            covariance_std_scale=0.9
        )


def test_negative_bias_bound_is_rejected() -> None:
    with pytest.raises(
        ValueError
    ):
        RobustChancePolicy(
            bias_bound=-0.001
        )


def test_contamination_must_be_below_target_probability() -> None:
    with pytest.raises(
        ValueError
    ):
        RobustChancePolicy(
            contamination_probability=0.05,
            target_violation_probability=0.05,
        )