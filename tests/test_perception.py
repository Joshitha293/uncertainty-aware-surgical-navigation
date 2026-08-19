import numpy as np
import pytest

from src.geometry.workspace import SphericalStructure
from src.perception.perception import (
    maximum_localisation_error,
    mean_localisation_error,
    perceive_structures,
)
from src.perception.uncertainty import PositionUncertainty


def make_true_structures() -> tuple[SphericalStructure, ...]:
    return (
        SphericalStructure(
            centre=np.array([0.14, 0.04, 0.00]),
            physical_radius=0.025,
            safety_margin=0.015,
        ),
        SphericalStructure(
            centre=np.array([0.18, -0.06, 0.02]),
            physical_radius=0.025,
            safety_margin=0.015,
        ),
    )


def test_zero_uncertainty_recovers_true_centres():
    true_structures = make_true_structures()

    uncertainty = PositionUncertainty.isotropic(
        sigma=0.0
    )

    rng = np.random.default_rng(123)

    result = perceive_structures(
        true_structures=true_structures,
        uncertainty=uncertainty,
        rng=rng,
    )

    for true_structure, estimate in zip(
        true_structures,
        result.estimated_structures,
    ):
        np.testing.assert_allclose(
            estimate.estimated_centre,
            true_structure.centre,
            atol=1e-12,
        )


def test_localisation_errors_are_zero_with_zero_uncertainty():
    result = perceive_structures(
        true_structures=make_true_structures(),
        uncertainty=PositionUncertainty.isotropic(
            sigma=0.0
        ),
        rng=np.random.default_rng(123),
    )

    np.testing.assert_allclose(
        result.localisation_errors,
        np.zeros(2),
        atol=1e-12,
    )


def test_perception_is_reproducible_for_fixed_seed():
    structures = make_true_structures()

    uncertainty = PositionUncertainty.isotropic(
        sigma=0.005
    )

    result_a = perceive_structures(
        true_structures=structures,
        uncertainty=uncertainty,
        rng=np.random.default_rng(99),
    )

    result_b = perceive_structures(
        true_structures=structures,
        uncertainty=uncertainty,
        rng=np.random.default_rng(99),
    )

    for estimate_a, estimate_b in zip(
        result_a.estimated_structures,
        result_b.estimated_structures,
    ):
        np.testing.assert_allclose(
            estimate_a.estimated_centre,
            estimate_b.estimated_centre,
            atol=1e-12,
        )

    np.testing.assert_allclose(
        result_a.localisation_errors,
        result_b.localisation_errors,
        atol=1e-12,
    )


def test_different_seeds_can_produce_different_estimates():
    structures = make_true_structures()

    uncertainty = PositionUncertainty.isotropic(
        sigma=0.005
    )

    result_a = perceive_structures(
        true_structures=structures,
        uncertainty=uncertainty,
        rng=np.random.default_rng(1),
    )

    result_b = perceive_structures(
        true_structures=structures,
        uncertainty=uncertainty,
        rng=np.random.default_rng(2),
    )

    centres_a = np.vstack(
        [
            estimate.estimated_centre
            for estimate in result_a.estimated_structures
        ]
    )

    centres_b = np.vstack(
        [
            estimate.estimated_centre
            for estimate in result_b.estimated_structures
        ]
    )

    assert not np.allclose(
        centres_a,
        centres_b,
    )


def test_localisation_error_is_non_negative():
    result = perceive_structures(
        true_structures=make_true_structures(),
        uncertainty=PositionUncertainty.isotropic(
            sigma=0.005
        ),
        rng=np.random.default_rng(7),
    )

    assert np.all(
        result.localisation_errors >= 0.0
    )


def test_result_preserves_number_of_structures():
    structures = make_true_structures()

    result = perceive_structures(
        true_structures=structures,
        uncertainty=PositionUncertainty.isotropic(
            sigma=0.005
        ),
        rng=np.random.default_rng(7),
    )

    assert len(result.estimated_structures) == len(
        structures
    )

    assert len(result.localisation_errors) == len(
        structures
    )


def test_estimated_structure_preserves_physical_radius():
    structures = make_true_structures()

    result = perceive_structures(
        true_structures=structures,
        uncertainty=PositionUncertainty.isotropic(
            sigma=0.005
        ),
        rng=np.random.default_rng(7),
    )

    for true_structure, estimate in zip(
        structures,
        result.estimated_structures,
    ):
        assert estimate.physical_radius == pytest.approx(
            true_structure.physical_radius
        )


def test_estimated_structure_preserves_base_safety_margin():
    structures = make_true_structures()

    result = perceive_structures(
        true_structures=structures,
        uncertainty=PositionUncertainty.isotropic(
            sigma=0.005
        ),
        rng=np.random.default_rng(7),
    )

    for true_structure, estimate in zip(
        structures,
        result.estimated_structures,
    ):
        assert estimate.base_safety_margin == pytest.approx(
            true_structure.safety_margin
        )


def test_mean_localisation_error_matches_manual_mean():
    result = perceive_structures(
        true_structures=make_true_structures(),
        uncertainty=PositionUncertainty.isotropic(
            sigma=0.005
        ),
        rng=np.random.default_rng(13),
    )

    expected = float(
        np.mean(
            result.localisation_errors
        )
    )

    assert mean_localisation_error(
        result
    ) == pytest.approx(
        expected
    )


def test_maximum_localisation_error_matches_manual_maximum():
    result = perceive_structures(
        true_structures=make_true_structures(),
        uncertainty=PositionUncertainty.isotropic(
            sigma=0.005
        ),
        rng=np.random.default_rng(13),
    )

    expected = float(
        np.max(
            result.localisation_errors
        )
    )

    assert maximum_localisation_error(
        result
    ) == pytest.approx(
        expected
    )


def test_empty_structure_set_returns_empty_result():
    result = perceive_structures(
        true_structures=(),
        uncertainty=PositionUncertainty.isotropic(
            sigma=0.005
        ),
        rng=np.random.default_rng(13),
    )

    assert len(
        result.estimated_structures
    ) == 0

    assert result.localisation_errors.shape == (0,)


def test_empty_result_metrics_are_zero():
    result = perceive_structures(
        true_structures=(),
        uncertainty=PositionUncertainty.isotropic(
            sigma=0.005
        ),
        rng=np.random.default_rng(13),
    )

    assert mean_localisation_error(
        result
    ) == 0.0

    assert maximum_localisation_error(
        result
    ) == 0.0