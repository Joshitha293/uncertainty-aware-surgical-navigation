"""Tests for the final Phase 6 closed-loop benchmark."""

from __future__ import annotations

import numpy as np
import pytest

from src.simulation.phase6_closed_loop_benchmark import (
    EXECUTION_SAMPLES,
    FROZEN_SECONDARY_CENTRE,
    MAX_EXECUTION_STEPS,
    SECONDARY_PHYSICAL_RADIUS,
    SECONDARY_POSITION_SIGMA,
    SECONDARY_SAFETY_MARGIN,
    STRUCTURE_UPDATE_ONSET_INDEX,
    _estimated_structures_for_step,
    _reported_std_scale,
    _secondary_structure_estimate,
    _truth_centre,
    _truth_structures_for_step,
    make_environment,
    resample_joint_path,
    scenario_names,
)
from src.simulation.phase6_chance_constraint_benchmark import (
    goal_configuration,
    start_configuration,
)


def test_required_final_scenarios_exist() -> None:
    """Final benchmark must contain all four frozen scenarios."""

    assert scenario_names() == (
        "stable",
        "recoverable_structure_update",
        "transient_uncertainty_spike",
        "severe_perception_loss",
    )


def test_base_hidden_anatomy_is_stationary() -> None:
    """The final benchmark no longer moves the original structure."""

    for scenario in scenario_names():
        first = _truth_centre(
            scenario,
            0,
        )

        last = _truth_centre(
            scenario,
            MAX_EXECUTION_STEPS - 1,
        )

        assert np.allclose(
            first,
            last,
        )


def test_frozen_secondary_structure_parameters() -> None:
    """Recoverable structure geometry must remain frozen."""

    assert (
        STRUCTURE_UPDATE_ONSET_INDEX
        == 11
    )

    assert np.allclose(
        FROZEN_SECONDARY_CENTRE,
        np.asarray(
            [
                0.0420764,
                0.00799174,
                0.02476202,
            ],
            dtype=float,
        ),
    )

    assert np.isclose(
        SECONDARY_PHYSICAL_RADIUS,
        0.005,
    )

    assert np.isclose(
        SECONDARY_SAFETY_MARGIN,
        0.004,
    )

    assert np.isclose(
        SECONDARY_POSITION_SIGMA,
        0.002,
    )


def test_secondary_runtime_structure_not_visible_before_update() -> None:
    """Runtime perception must not receive secondary structure too early."""

    environment = (
        make_environment(
            "recoverable_structure_update",
            123,
        )
    )

    before = (
        _estimated_structures_for_step(
            environment,
            STRUCTURE_UPDATE_ONSET_INDEX - 1,
        )
    )

    assert len(
        before
    ) == 1


def test_secondary_runtime_structure_appears_at_update() -> None:
    """Runtime perception receives secondary structure at frozen onset."""

    environment = (
        make_environment(
            "recoverable_structure_update",
            123,
        )
    )

    at_update = (
        _estimated_structures_for_step(
            environment,
            STRUCTURE_UPDATE_ONSET_INDEX,
        )
    )

    assert len(
        at_update
    ) == 2

    secondary = (
        at_update[
            1
        ]
    )

    assert np.allclose(
        secondary.estimated_centre,
        FROZEN_SECONDARY_CENTRE,
    )


def test_secondary_structure_exists_in_hidden_truth_before_discovery() -> None:
    """Hidden truth contains the secondary object before runtime sees it."""

    environment = (
        make_environment(
            "recoverable_structure_update",
            321,
        )
    )

    hidden_before = (
        _truth_structures_for_step(
            environment,
            STRUCTURE_UPDATE_ONSET_INDEX - 1,
        )
    )

    runtime_before = (
        _estimated_structures_for_step(
            environment,
            STRUCTURE_UPDATE_ONSET_INDEX - 1,
        )
    )

    assert len(
        hidden_before
    ) == 2

    assert len(
        runtime_before
    ) == 1


def test_secondary_estimate_has_frozen_isotropic_sigma() -> None:
    """Secondary covariance must encode 2 mm isotropic positional sigma."""

    secondary = (
        _secondary_structure_estimate()
    )

    covariance = (
        secondary
        .uncertainty
        .covariance
    )

    assert np.allclose(
        covariance,
        np.eye(
            3,
            dtype=float,
        )
        * SECONDARY_POSITION_SIGMA**2,
    )

    principal_sigma = float(
        np.sqrt(
            np.max(
                np.linalg.eigvalsh(
                    covariance
                )
            )
        )
    )

    assert np.isclose(
        principal_sigma,
        0.002,
    )


def test_transient_uncertainty_spike_recovers() -> None:
    """Transient uncertainty must increase and then return to nominal."""

    before = (
        _reported_std_scale(
            "transient_uncertainty_spike",
            5,
        )
    )

    spike = (
        _reported_std_scale(
            "transient_uncertainty_spike",
            12,
        )
    )

    after = (
        _reported_std_scale(
            "transient_uncertainty_spike",
            20,
        )
    )

    assert np.isclose(
        before,
        1.0,
    )

    assert spike > before

    assert np.isclose(
        after,
        1.0,
    )


def test_severe_perception_loss_exceeds_stop_scale() -> None:
    """Severe degradation must create a large uncertainty increase."""

    early = (
        _reported_std_scale(
            "severe_perception_loss",
            5,
        )
    )

    late = (
        _reported_std_scale(
            "severe_perception_loss",
            20,
        )
    )

    assert np.isclose(
        early,
        1.0,
    )

    assert late > 3.0


def test_environment_has_expected_horizon() -> None:
    """Generated episode must contain complete simulation horizon."""

    environment = (
        make_environment(
            "stable",
            123,
        )
    )

    assert (
        environment
        .truth_centres
        .shape
        == (
            MAX_EXECUTION_STEPS,
            3,
        )
    )

    assert (
        len(
            environment
            .reported_estimates
        )
        == MAX_EXECUTION_STEPS
    )


def test_environment_is_deterministic_for_fixed_seed() -> None:
    """Identical seeds must reproduce identical simulated perception."""

    first = (
        make_environment(
            "transient_uncertainty_spike",
            456,
        )
    )

    second = (
        make_environment(
            "transient_uncertainty_spike",
            456,
        )
    )

    assert np.allclose(
        first.truth_centres,
        second.truth_centres,
    )

    for (
        first_estimate,
        second_estimate,
    ) in zip(
        first.reported_estimates,
        second.reported_estimates,
    ):
        assert np.allclose(
            first_estimate
            .estimated_centre,
            second_estimate
            .estimated_centre,
        )

        assert np.allclose(
            first_estimate
            .uncertainty
            .covariance,
            second_estimate
            .uncertainty
            .covariance,
        )


def test_initial_perception_uses_controlled_nominal_mean() -> None:
    """Initial mean is controlled while covariance remains non-zero."""

    environment = (
        make_environment(
            "stable",
            789,
        )
    )

    initial_estimate = (
        environment
        .reported_estimates[
            0
        ]
    )

    initial_truth = (
        environment
        .truth_centres[
            0
        ]
    )

    assert np.allclose(
        initial_estimate
        .estimated_centre,
        initial_truth,
    )

    covariance = (
        initial_estimate
        .uncertainty
        .covariance
    )

    assert np.max(
        np.linalg.eigvalsh(
            covariance
        )
    ) > 0.0


def test_stable_perception_after_initialisation_is_noisy() -> None:
    """Stable scenario should still contain routine localisation noise."""

    environment = (
        make_environment(
            "stable",
            789,
        )
    )

    estimate = (
        environment
        .reported_estimates[
            1
        ]
        .estimated_centre
    )

    truth = (
        environment
        .truth_centres[
            1
        ]
    )

    assert (
        np.linalg.norm(
            estimate
            - truth
        )
        > 0.0
    )


def test_recoverable_structure_update_isolates_secondary_discovery() -> None:
    """Base estimate remains controlled in structure-update scenario."""

    environment = (
        make_environment(
            "recoverable_structure_update",
            789,
        )
    )

    for step_index in (
        0,
        5,
        STRUCTURE_UPDATE_ONSET_INDEX,
        20,
    ):
        assert np.allclose(
            environment
            .reported_estimates[
                step_index
            ]
            .estimated_centre,
            environment
            .truth_centres[
                step_index
            ],
        )


def test_path_resampling_preserves_endpoints() -> None:
    """Resampling must preserve original trajectory endpoints."""

    path = np.vstack(
        [
            start_configuration(),
            goal_configuration(),
        ]
    )

    resampled = (
        resample_joint_path(
            path,
            EXECUTION_SAMPLES,
        )
    )

    assert resampled.shape == (
        EXECUTION_SAMPLES,
        4,
    )

    assert np.allclose(
        resampled[
            0
        ],
        start_configuration(),
    )

    assert np.allclose(
        resampled[
            -1
        ],
        goal_configuration(),
    )


def test_path_resampling_rejects_invalid_sample_count() -> None:
    """At least two resampled configurations are required."""

    path = np.vstack(
        [
            start_configuration(),
            goal_configuration(),
        ]
    )

    with pytest.raises(
        ValueError
    ):
        resample_joint_path(
            path,
            1,
        )


def test_unknown_scenario_is_rejected() -> None:
    """Undefined scenario names must be rejected."""

    with pytest.raises(
        ValueError
    ):
        make_environment(
            "unknown",
            123,
        )