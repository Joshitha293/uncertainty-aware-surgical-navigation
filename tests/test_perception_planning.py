import numpy as np
import pytest

from src.geometry.workspace import SphericalStructure
from src.perception.perception import perceive_structures
from src.perception.planning import (
    deterministic_planning_structures,
    maximum_planning_safety_radius,
    mean_planning_safety_margin,
    uncertainty_aware_planning_structures,
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


def make_perception_result():
    return perceive_structures(
        true_structures=make_true_structures(),
        uncertainty=PositionUncertainty.isotropic(
            sigma=0.005
        ),
        rng=np.random.default_rng(123),
    )


def test_deterministic_planning_preserves_estimated_centres():
    perception = make_perception_result()

    planning = deterministic_planning_structures(
        perception
    )

    for estimate, structure in zip(
        perception.estimated_structures,
        planning.structures,
    ):
        np.testing.assert_allclose(
            structure.centre,
            estimate.estimated_centre,
            atol=1e-12,
        )


def test_deterministic_planning_preserves_base_margin():
    perception = make_perception_result()

    planning = deterministic_planning_structures(
        perception
    )

    for estimate, structure in zip(
        perception.estimated_structures,
        planning.structures,
    ):
        assert structure.safety_margin == pytest.approx(
            estimate.base_safety_margin
        )


def test_deterministic_planning_is_not_uncertainty_aware():
    planning = deterministic_planning_structures(
        make_perception_result()
    )

    assert not planning.uncertainty_aware
    assert planning.sigma_multiplier == 0.0


def test_uncertainty_aware_planning_inflates_margin():
    perception = make_perception_result()

    planning = uncertainty_aware_planning_structures(
        perception_result=perception,
        sigma_multiplier=2.0,
    )

    for estimate, structure in zip(
        perception.estimated_structures,
        planning.structures,
    ):
        assert (
            structure.safety_margin
            > estimate.base_safety_margin
        )


def test_uncertainty_aware_margin_matches_expected_value():
    sigma = 0.005

    perception = perceive_structures(
        true_structures=make_true_structures(),
        uncertainty=PositionUncertainty.isotropic(
            sigma=sigma
        ),
        rng=np.random.default_rng(123),
    )

    multiplier = 2.5

    planning = uncertainty_aware_planning_structures(
        perception_result=perception,
        sigma_multiplier=multiplier,
    )

    expected_margin = (
        0.015
        + multiplier * sigma
    )

    for structure in planning.structures:
        assert structure.safety_margin == pytest.approx(
            expected_margin
        )


def test_uncertainty_aware_planning_metadata_is_correct():
    planning = uncertainty_aware_planning_structures(
        perception_result=make_perception_result(),
        sigma_multiplier=3.0,
    )

    assert planning.uncertainty_aware
    assert planning.sigma_multiplier == pytest.approx(
        3.0
    )


def test_zero_multiplier_matches_deterministic_margin():
    perception = make_perception_result()

    deterministic = deterministic_planning_structures(
        perception
    )

    uncertainty_aware = (
        uncertainty_aware_planning_structures(
            perception_result=perception,
            sigma_multiplier=0.0,
        )
    )

    for deterministic_structure, aware_structure in zip(
        deterministic.structures,
        uncertainty_aware.structures,
    ):
        assert (
            aware_structure.safety_margin
            == pytest.approx(
                deterministic_structure.safety_margin
            )
        )


def test_larger_multiplier_produces_larger_margin():
    perception = make_perception_result()

    low = uncertainty_aware_planning_structures(
        perception_result=perception,
        sigma_multiplier=1.0,
    )

    high = uncertainty_aware_planning_structures(
        perception_result=perception,
        sigma_multiplier=3.0,
    )

    for low_structure, high_structure in zip(
        low.structures,
        high.structures,
    ):
        assert (
            high_structure.safety_margin
            > low_structure.safety_margin
        )


def test_negative_multiplier_is_rejected():
    with pytest.raises(ValueError):
        uncertainty_aware_planning_structures(
            perception_result=make_perception_result(),
            sigma_multiplier=-1.0,
        )


def test_mean_planning_margin_is_correct():
    perception = make_perception_result()

    planning = uncertainty_aware_planning_structures(
        perception_result=perception,
        sigma_multiplier=2.0,
    )

    expected = float(
        np.mean(
            [
                structure.safety_margin
                for structure in planning.structures
            ]
        )
    )

    assert mean_planning_safety_margin(
        planning
    ) == pytest.approx(
        expected
    )


def test_maximum_planning_safety_radius_is_correct():
    perception = make_perception_result()

    planning = uncertainty_aware_planning_structures(
        perception_result=perception,
        sigma_multiplier=2.0,
    )

    expected = max(
        structure.safety_radius
        for structure in planning.structures
    )

    assert maximum_planning_safety_radius(
        planning
    ) == pytest.approx(
        expected
    )


def test_empty_perception_returns_zero_summary_metrics():
    empty_perception = perceive_structures(
        true_structures=(),
        uncertainty=PositionUncertainty.isotropic(
            sigma=0.005
        ),
        rng=np.random.default_rng(123),
    )

    planning = uncertainty_aware_planning_structures(
        perception_result=empty_perception,
        sigma_multiplier=2.0,
    )

    assert len(planning.structures) == 0
    assert mean_planning_safety_margin(planning) == 0.0
    assert maximum_planning_safety_radius(planning) == 0.0