"""Tests for the controlled Phase 6 chance-constraint benchmark."""

from __future__ import annotations

import numpy as np

from src.robotics.planner import (
    edge_is_safe,
)
from src.simulation.phase6_chance_constraint_benchmark import (
    SCALAR_SIGMA_MULTIPLIER,
    build_orientation_conditions,
    direct_edge_decisions,
    make_instrument,
    select_disagreement_clearance,
    start_configuration,
    goal_configuration,
    _scalar_structure,
    INSTRUMENT_RADIUS,
    PROXIMAL_LENGTH,
    DIRECT_EDGE_RESOLUTION,
)


def test_orientation_conditions_have_same_estimated_centre() -> None:
    tangential, radial = (
        build_orientation_conditions(
            0.060
        )
    )

    assert np.allclose(
        tangential
        .estimate
        .estimated_centre,
        radial
        .estimate
        .estimated_centre,
    )


def test_orientation_conditions_have_identical_covariance_eigenvalues() -> None:
    tangential, radial = (
        build_orientation_conditions(
            0.060
        )
    )

    tangent_eigenvalues = np.linalg.eigvalsh(
        tangential
        .estimate
        .uncertainty
        .covariance
    )

    radial_eigenvalues = np.linalg.eigvalsh(
        radial
        .estimate
        .uncertainty
        .covariance
    )

    assert np.allclose(
        tangent_eigenvalues,
        radial_eigenvalues,
        atol=1e-14,
    )


def test_orientation_conditions_have_same_principal_sigma() -> None:
    tangential, radial = (
        build_orientation_conditions(
            0.060
        )
    )

    assert np.isclose(
        tangential
        .estimate
        .uncertainty
        .principal_sigma,
        radial
        .estimate
        .uncertainty
        .principal_sigma,
    )


def test_scalar_inflation_is_identical_between_orientations() -> None:
    tangential, radial = (
        build_orientation_conditions(
            0.060
        )
    )

    tangent_structure = (
        _scalar_structure(
            tangential.estimate
        )
    )

    radial_structure = (
        _scalar_structure(
            radial.estimate
        )
    )

    assert np.isclose(
        tangent_structure.safety_radius,
        radial_structure.safety_radius,
    )

    assert np.isclose(
        tangent_structure.safety_margin,
        radial_structure.safety_margin,
    )


def test_controlled_geometry_exposes_expected_strategy_disagreement() -> None:
    (
        selected_clearance,
        conditions,
    ) = (
        select_disagreement_clearance()
    )

    assert (
        0.050
        <= selected_clearance
        <= 0.071
    )

    tangential, radial = conditions

    tangent_decisions = (
        direct_edge_decisions(
            tangential
        )
    )

    radial_decisions = (
        direct_edge_decisions(
            radial
        )
    )

    assert tangent_decisions[
        "deterministic"
    ] is True

    assert radial_decisions[
        "deterministic"
    ] is True

    assert tangent_decisions[
        "scalar_principal_sigma"
    ] is False

    assert radial_decisions[
        "scalar_principal_sigma"
    ] is False

    assert tangent_decisions[
        "anisotropic_chance"
    ] is True

    assert radial_decisions[
        "anisotropic_chance"
    ] is False


def test_scalar_planner_geometry_cannot_distinguish_orientation() -> None:
    tangential, radial = (
        build_orientation_conditions(
            0.060
        )
    )

    instrument = make_instrument()

    tangent_safe = edge_is_safe(
        instrument=instrument,
        q_start=start_configuration(),
        q_goal=goal_configuration(),
        structures=(
            _scalar_structure(
                tangential.estimate
            ),
        ),
        instrument_radius=(
            INSTRUMENT_RADIUS
        ),
        proximal_length=(
            PROXIMAL_LENGTH
        ),
        resolution=(
            DIRECT_EDGE_RESOLUTION
        ),
    )

    radial_safe = edge_is_safe(
        instrument=instrument,
        q_start=start_configuration(),
        q_goal=goal_configuration(),
        structures=(
            _scalar_structure(
                radial.estimate
            ),
        ),
        instrument_radius=(
            INSTRUMENT_RADIUS
        ),
        proximal_length=(
            PROXIMAL_LENGTH
        ),
        resolution=(
            DIRECT_EDGE_RESOLUTION
        ),
    )

    assert (
        tangent_safe
        == radial_safe
    )


def test_scalar_multiplier_matches_one_sided_five_percent_threshold() -> None:
    assert np.isclose(
        SCALAR_SIGMA_MULTIPLIER,
        1.6448536269514722,
        atol=1e-12,
    )