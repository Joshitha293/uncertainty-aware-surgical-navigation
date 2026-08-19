import numpy as np
import pytest

from src.geometry.workspace import SphericalStructure
from src.perception.uncertainty import (
    EstimatedStructure,
    PositionUncertainty,
    inflate_estimated_structures,
    make_estimated_structure,
    sample_noisy_centre,
)


def test_isotropic_uncertainty_has_expected_covariance():
    uncertainty = PositionUncertainty.isotropic(
        sigma=0.005
    )

    expected = (
        0.005**2
    ) * np.eye(3)

    np.testing.assert_allclose(
        uncertainty.covariance,
        expected,
        atol=1e-12,
    )


def test_principal_sigma_matches_isotropic_sigma():
    uncertainty = PositionUncertainty.isotropic(
        sigma=0.007
    )

    assert uncertainty.principal_sigma == pytest.approx(
        0.007
    )


def test_negative_sigma_is_rejected():
    with pytest.raises(ValueError):
        PositionUncertainty.isotropic(
            sigma=-0.001
        )


def test_non_symmetric_covariance_is_rejected():
    covariance = np.array(
        [
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    with pytest.raises(ValueError):
        PositionUncertainty(
            covariance=covariance
        )


def test_negative_variance_covariance_is_rejected():
    covariance = np.diag(
        [
            1.0,
            1.0,
            -1.0,
        ]
    )

    with pytest.raises(ValueError):
        PositionUncertainty(
            covariance=covariance
        )


def test_noisy_centre_is_reproducible_for_fixed_seed():
    uncertainty = PositionUncertainty.isotropic(
        sigma=0.005
    )

    true_centre = np.array(
        [0.10, 0.02, -0.01]
    )

    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    sample_a = sample_noisy_centre(
        true_centre=true_centre,
        uncertainty=uncertainty,
        rng=rng_a,
    )

    sample_b = sample_noisy_centre(
        true_centre=true_centre,
        uncertainty=uncertainty,
        rng=rng_b,
    )

    np.testing.assert_allclose(
        sample_a,
        sample_b,
        atol=1e-12,
    )


def test_zero_uncertainty_returns_true_centre():
    uncertainty = PositionUncertainty.isotropic(
        sigma=0.0
    )

    true_centre = np.array(
        [0.10, 0.02, -0.01]
    )

    rng = np.random.default_rng(123)

    sample = sample_noisy_centre(
        true_centre=true_centre,
        uncertainty=uncertainty,
        rng=rng,
    )

    np.testing.assert_allclose(
        sample,
        true_centre,
        atol=1e-12,
    )


def test_estimated_structure_preserves_geometry():
    true_structure = SphericalStructure(
        centre=np.array(
            [0.14, 0.04, 0.0]
        ),
        physical_radius=0.025,
        safety_margin=0.015,
    )

    uncertainty = PositionUncertainty.isotropic(
        sigma=0.003
    )

    rng = np.random.default_rng(7)

    estimate = make_estimated_structure(
        true_structure=true_structure,
        uncertainty=uncertainty,
        rng=rng,
    )

    assert estimate.physical_radius == pytest.approx(
        0.025
    )

    assert estimate.base_safety_margin == pytest.approx(
        0.015
    )


def test_uncertainty_aware_margin_is_inflated_correctly():
    uncertainty = PositionUncertainty.isotropic(
        sigma=0.004
    )

    estimate = EstimatedStructure(
        estimated_centre=np.zeros(3),
        physical_radius=0.025,
        base_safety_margin=0.015,
        uncertainty=uncertainty,
    )

    structure = estimate.uncertainty_aware_structure(
        sigma_multiplier=2.0
    )

    expected_margin = (
        0.015
        + 2.0 * 0.004
    )

    assert structure.safety_margin == pytest.approx(
        expected_margin
    )


def test_larger_uncertainty_produces_larger_margin():
    low_uncertainty = PositionUncertainty.isotropic(
        sigma=0.002
    )

    high_uncertainty = PositionUncertainty.isotropic(
        sigma=0.008
    )

    low_estimate = EstimatedStructure(
        estimated_centre=np.zeros(3),
        physical_radius=0.025,
        base_safety_margin=0.015,
        uncertainty=low_uncertainty,
    )

    high_estimate = EstimatedStructure(
        estimated_centre=np.zeros(3),
        physical_radius=0.025,
        base_safety_margin=0.015,
        uncertainty=high_uncertainty,
    )

    low_structure = (
        low_estimate.uncertainty_aware_structure(
            sigma_multiplier=2.0
        )
    )

    high_structure = (
        high_estimate.uncertainty_aware_structure(
            sigma_multiplier=2.0
        )
    )

    assert (
        high_structure.safety_margin
        > low_structure.safety_margin
    )


def test_inflate_estimated_structures_preserves_count():
    uncertainty = PositionUncertainty.isotropic(
        sigma=0.003
    )

    estimates = (
        EstimatedStructure(
            estimated_centre=np.array(
                [0.10, 0.0, 0.0]
            ),
            physical_radius=0.02,
            base_safety_margin=0.01,
            uncertainty=uncertainty,
        ),
        EstimatedStructure(
            estimated_centre=np.array(
                [0.20, 0.0, 0.0]
            ),
            physical_radius=0.02,
            base_safety_margin=0.01,
            uncertainty=uncertainty,
        ),
    )

    inflated = inflate_estimated_structures(
        estimated_structures=estimates,
        sigma_multiplier=3.0,
    )

    assert len(inflated) == 2


def test_negative_sigma_multiplier_is_rejected():
    uncertainty = PositionUncertainty.isotropic(
        sigma=0.003
    )

    estimate = EstimatedStructure(
        estimated_centre=np.zeros(3),
        physical_radius=0.02,
        base_safety_margin=0.01,
        uncertainty=uncertainty,
    )

    with pytest.raises(ValueError):
        estimate.uncertainty_aware_structure(
            sigma_multiplier=-1.0
        )