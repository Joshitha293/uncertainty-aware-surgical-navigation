"""Tests for Phase 5 temporal Kalman state estimation."""

import numpy as np
import pytest

from src.perception.state_estimation import (
    ConstantVelocityKalmanFilter,
    KalmanFilterConfig,
    TrackingState,
    process_noise_covariance,
    tracked_estimated_structure,
    transition_matrix,
)
from src.perception.uncertainty import (
    PositionUncertainty,
    inflate_estimated_structures,
)


def measurement_covariance(
    sigma: float = 0.003,
) -> np.ndarray:
    return (
        sigma**2
        * np.eye(
            3,
            dtype=float,
        )
    )


def test_transition_matrix_propagates_constant_velocity():
    transition = (
        transition_matrix(
            2.0
        )
    )

    state = np.asarray(
        [
            1.0,
            2.0,
            3.0,
            0.5,
            -0.25,
            1.0,
        ]
    )

    propagated = (
        transition
        @ state
    )

    np.testing.assert_allclose(
        propagated[
            :3
        ],
        np.asarray(
            [
                2.0,
                1.5,
                5.0,
            ]
        ),
        atol=1e-12,
    )


def test_process_noise_has_expected_shape_and_is_psd():
    covariance = (
        process_noise_covariance(
            dt=0.1,
            acceleration_sigma=0.02,
        )
    )

    assert covariance.shape == (
        6,
        6,
    )

    np.testing.assert_allclose(
        covariance,
        covariance.T,
        atol=1e-12,
    )

    assert np.all(
        np.linalg.eigvalsh(
            covariance
        )
        >= -1e-12
    )


def test_zero_dt_produces_zero_process_noise():
    covariance = (
        process_noise_covariance(
            dt=0.0,
            acceleration_sigma=0.02,
        )
    )

    np.testing.assert_allclose(
        covariance,
        np.zeros(
            (
                6,
                6,
            )
        ),
        atol=1e-12,
    )


def test_filter_initialises_position_from_measurement():
    tracker = (
        ConstantVelocityKalmanFilter()
    )

    measurement = np.asarray(
        [
            0.10,
            -0.02,
            0.15,
        ]
    )

    state = tracker.initialise(
        measurement=measurement,
        measurement_covariance=(
            measurement_covariance()
        ),
        timestamp=0.0,
    )

    np.testing.assert_allclose(
        state.position,
        measurement,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        state.velocity,
        np.zeros(
            3
        ),
        atol=1e-12,
    )


def test_initial_position_covariance_matches_measurement():
    tracker = (
        ConstantVelocityKalmanFilter()
    )

    covariance = (
        measurement_covariance(
            0.004
        )
    )

    state = tracker.initialise(
        measurement=np.zeros(
            3
        ),
        measurement_covariance=(
            covariance
        ),
        timestamp=0.0,
    )

    np.testing.assert_allclose(
        state.position_covariance,
        covariance,
        atol=1e-12,
    )


def test_prediction_moves_position_using_velocity():
    tracker = (
        ConstantVelocityKalmanFilter(
            KalmanFilterConfig(
                acceleration_sigma=0.0,
            )
        )
    )

    state = TrackingState(
        mean=np.asarray(
            [
                0.0,
                0.0,
                0.0,
                0.1,
                -0.2,
                0.3,
            ]
        ),
        covariance=np.eye(
            6
        )
        * 0.001,
        timestamp=0.0,
        accepted_measurements=1,
    )

    predicted = tracker.predict(
        state,
        timestamp=0.5,
    )

    np.testing.assert_allclose(
        predicted.position,
        np.asarray(
            [
                0.05,
                -0.10,
                0.15,
            ]
        ),
        atol=1e-12,
    )


def test_prediction_rejects_backward_time():
    tracker = (
        ConstantVelocityKalmanFilter()
    )

    state = tracker.initialise(
        measurement=np.zeros(
            3
        ),
        measurement_covariance=(
            measurement_covariance()
        ),
        timestamp=1.0,
    )

    with pytest.raises(
        ValueError,
        match="cannot precede",
    ):
        tracker.predict(
            state,
            timestamp=0.5,
        )


def test_update_moves_prediction_toward_measurement():
    tracker = (
        ConstantVelocityKalmanFilter()
    )

    initial = tracker.initialise(
        measurement=np.zeros(
            3
        ),
        measurement_covariance=(
            measurement_covariance()
        ),
        timestamp=0.0,
    )

    predicted = tracker.predict(
        initial,
        timestamp=1.0,
    )

    measurement = np.asarray(
        [
            0.020,
            0.0,
            0.0,
        ]
    )

    result = tracker.update(
        predicted,
        measurement=measurement,
        measurement_covariance=(
            measurement_covariance()
        ),
    )

    assert result.accepted

    assert (
        result.state.position[
            0
        ]
        > predicted.position[
            0
        ]
    )

    assert (
        result.state.position[
            0
        ]
        < measurement[
            0
        ]
        + 1e-12
    )


def test_update_reduces_position_covariance():
    tracker = (
        ConstantVelocityKalmanFilter()
    )

    initial = tracker.initialise(
        measurement=np.zeros(
            3
        ),
        measurement_covariance=(
            measurement_covariance()
        ),
        timestamp=0.0,
    )

    predicted = tracker.predict(
        initial,
        timestamp=0.2,
    )

    result = tracker.update(
        predicted,
        measurement=np.zeros(
            3
        ),
        measurement_covariance=(
            measurement_covariance()
        ),
    )

    assert (
        np.trace(
            result
            .state
            .position_covariance
        )
        < np.trace(
            predicted
            .position_covariance
        )
    )


def test_lower_noise_measurement_has_stronger_update():
    tracker = (
        ConstantVelocityKalmanFilter()
    )

    initial = tracker.initialise(
        measurement=np.zeros(
            3
        ),
        measurement_covariance=(
            measurement_covariance(
                0.01
            )
        ),
        timestamp=0.0,
    )

    predicted = tracker.predict(
        initial,
        timestamp=0.5,
    )

    measurement = np.asarray(
        [
            0.020,
            0.0,
            0.0,
        ]
    )

    low_noise = tracker.update(
        predicted,
        measurement=measurement,
        measurement_covariance=(
            measurement_covariance(
                0.001
            )
        ),
    )

    high_noise = tracker.update(
        predicted,
        measurement=measurement,
        measurement_covariance=(
            measurement_covariance(
                0.020
            )
        ),
    )

    assert (
        low_noise
        .state
        .position[
            0
        ]
        > high_noise
        .state
        .position[
            0
        ]
    )


def test_outlier_measurement_can_be_rejected_by_mahalanobis_gate():
    tracker = (
        ConstantVelocityKalmanFilter(
            KalmanFilterConfig(
                innovation_gate_threshold=(
                    7.814727903251179
                )
            )
        )
    )

    state = tracker.initialise(
        measurement=np.zeros(
            3
        ),
        measurement_covariance=(
            measurement_covariance(
                0.001
            )
        ),
        timestamp=0.0,
    )

    predicted = tracker.predict(
        state,
        timestamp=0.1,
    )

    result = tracker.update(
        predicted,
        measurement=np.asarray(
            [
                1.0,
                1.0,
                1.0,
            ]
        ),
        measurement_covariance=(
            measurement_covariance(
                0.001
            )
        ),
    )

    assert not result.accepted

    np.testing.assert_allclose(
        result.state.mean,
        predicted.mean,
        atol=1e-12,
    )


def test_valid_measurement_passes_reasonable_gate():
    tracker = (
        ConstantVelocityKalmanFilter(
            KalmanFilterConfig(
                innovation_gate_threshold=(
                    7.814727903251179
                )
            )
        )
    )

    state = tracker.initialise(
        measurement=np.zeros(
            3
        ),
        measurement_covariance=(
            measurement_covariance(
                0.003
            )
        ),
        timestamp=0.0,
    )

    predicted = tracker.predict(
        state,
        timestamp=0.1,
    )

    result = tracker.update(
        predicted,
        measurement=np.asarray(
            [
                0.001,
                -0.001,
                0.001,
            ]
        ),
        measurement_covariance=(
            measurement_covariance(
                0.003
            )
        ),
    )

    assert result.accepted


def test_tracking_state_exposes_existing_position_uncertainty():
    tracker = (
        ConstantVelocityKalmanFilter()
    )

    state = tracker.initialise(
        measurement=np.zeros(
            3
        ),
        measurement_covariance=(
            measurement_covariance()
        ),
        timestamp=0.0,
    )

    uncertainty = (
        state.position_uncertainty
    )

    assert isinstance(
        uncertainty,
        PositionUncertainty,
    )

    np.testing.assert_allclose(
        uncertainty.covariance,
        state.position_covariance,
        atol=1e-12,
    )


def test_tracking_state_converts_to_estimated_structure():
    tracker = (
        ConstantVelocityKalmanFilter()
    )

    state = tracker.initialise(
        measurement=np.asarray(
            [
                0.10,
                0.02,
                0.15,
            ]
        ),
        measurement_covariance=(
            measurement_covariance()
        ),
        timestamp=0.0,
    )

    estimate = (
        tracked_estimated_structure(
            state,
            physical_radius=0.025,
            base_safety_margin=0.015,
        )
    )

    np.testing.assert_allclose(
        estimate.estimated_centre,
        state.position,
        atol=1e-12,
    )

    assert isinstance(
        estimate.uncertainty,
        PositionUncertainty,
    )


def test_tracked_uncertainty_can_inflate_planning_margin():
    tracker = (
        ConstantVelocityKalmanFilter()
    )

    state = tracker.initialise(
        measurement=np.asarray(
            [
                0.10,
                0.02,
                0.15,
            ]
        ),
        measurement_covariance=(
            measurement_covariance()
        ),
        timestamp=0.0,
    )

    estimate = (
        tracked_estimated_structure(
            state,
            physical_radius=0.025,
            base_safety_margin=0.015,
        )
    )

    inflated = (
        inflate_estimated_structures(
            estimated_structures=(
                estimate,
            ),
            sigma_multiplier=2.0,
        )
    )

    assert (
        inflated[
            0
        ].safety_margin
        > estimate.base_safety_margin
    )


def test_repeated_measurements_estimate_constant_velocity():
    tracker = (
        ConstantVelocityKalmanFilter(
            KalmanFilterConfig(
                acceleration_sigma=0.005,
                initial_velocity_sigma=0.1,
            )
        )
    )

    velocity = np.asarray(
        [
            0.010,
            -0.005,
            0.003,
        ]
    )

    state = tracker.initialise(
        measurement=np.zeros(
            3
        ),
        measurement_covariance=(
            measurement_covariance(
                0.001
            )
        ),
        timestamp=0.0,
    )

    for index in range(
        1,
        21,
    ):
        timestamp = (
            0.1
            * index
        )

        true_position = (
            velocity
            * timestamp
        )

        result = tracker.step(
            state,
            measurement=(
                true_position
            ),
            measurement_covariance=(
                measurement_covariance(
                    0.001
                )
            ),
            timestamp=timestamp,
        )

        state = (
            result.state
        )

    np.testing.assert_allclose(
        state.velocity,
        velocity,
        atol=0.0015,
    )


def test_state_covariance_remains_symmetric_psd_after_many_steps():
    tracker = (
        ConstantVelocityKalmanFilter()
    )

    rng = np.random.default_rng(
        5301
    )

    state = tracker.initialise(
        measurement=np.zeros(
            3
        ),
        measurement_covariance=(
            measurement_covariance()
        ),
        timestamp=0.0,
    )

    for index in range(
        1,
        30,
    ):
        timestamp = (
            0.05
            * index
        )

        measurement = rng.normal(
            0.0,
            0.002,
            size=3,
        )

        state = (
            tracker.step(
                state,
                measurement=measurement,
                measurement_covariance=(
                    measurement_covariance()
                ),
                timestamp=timestamp,
            )
            .state
        )

    np.testing.assert_allclose(
        state.covariance,
        state.covariance.T,
        atol=1e-10,
    )

    assert np.all(
        np.linalg.eigvalsh(
            state.covariance
        )
        >= -1e-10
    )