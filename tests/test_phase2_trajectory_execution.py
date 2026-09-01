"""Tests for Phase 2 trajectory execution and timing."""

import numpy as np

from src.robotics.instrument import (
    SurgicalInstrument,
)
from src.robotics.trajectory_execution import (
    TimeParameterisationConfig,
    evaluate_execution_metrics,
    time_parameterise_path,
)


def make_instrument() -> SurgicalInstrument:
    return SurgicalInstrument(
        rcm_position=np.zeros(
            3,
            dtype=float,
        )
    )


def make_path() -> np.ndarray:
    return np.asarray(
        [
            [
                -0.25,
                -0.15,
                0.12,
                0.00,
            ],
            [
                0.00,
                0.00,
                0.18,
                0.20,
            ],
            [
                0.30,
                0.20,
                0.24,
                0.50,
            ],
        ],
        dtype=float,
    )


def test_timed_trajectory_preserves_start_and_goal():
    instrument = (
        make_instrument()
    )

    path = make_path()

    trajectory = (
        time_parameterise_path(
            instrument,
            path,
        )
    )

    np.testing.assert_allclose(
        trajectory.positions[0],
        path[0],
        atol=1e-12,
    )

    np.testing.assert_allclose(
        trajectory.positions[-1],
        path[-1],
        atol=1e-12,
    )


def test_trajectory_time_is_strictly_increasing():
    trajectory = (
        time_parameterise_path(
            make_instrument(),
            make_path(),
        )
    )

    assert np.all(
        np.diff(
            trajectory.times
        )
        > 0.0
    )


def test_velocity_limits_are_respected():
    config = (
        TimeParameterisationConfig(
            maximum_joint_velocities=(
                0.5,
                0.5,
                0.05,
                1.0,
            ),
            sample_period=0.01,
        )
    )

    trajectory = (
        time_parameterise_path(
            make_instrument(),
            make_path(),
            config=config,
        )
    )

    limits = np.asarray(
        config.maximum_joint_velocities
    )

    assert np.all(
        np.abs(
            trajectory.velocities
        )
        <= limits[None, :]
        + 1e-10
    )


def test_velocity_shape_matches_sample_count():
    trajectory = (
        time_parameterise_path(
            make_instrument(),
            make_path(),
        )
    )

    assert (
        trajectory.velocities.shape
        == (
            trajectory.positions.shape[0]
            - 1,
            4,
        )
    )


def test_acceleration_shape_is_consistent():
    trajectory = (
        time_parameterise_path(
            make_instrument(),
            make_path(),
        )
    )

    assert (
        trajectory.accelerations.shape
        == (
            max(
                trajectory.velocities.shape[0]
                - 1,
                0,
            ),
            4,
        )
    )


def test_execution_metrics_are_finite():
    instrument = (
        make_instrument()
    )

    trajectory = (
        time_parameterise_path(
            instrument,
            make_path(),
        )
    )

    metrics = (
        evaluate_execution_metrics(
            instrument,
            trajectory,
        )
    )

    assert metrics.duration > 0.0

    assert metrics.sample_count >= 2

    assert (
        metrics.scaled_joint_path_cost
        > 0.0
    )

    assert (
        metrics.cartesian_tip_distance
        > 0.0
    )

    assert np.all(
        np.isfinite(
            metrics.peak_absolute_velocity
        )
    )

    assert np.all(
        np.isfinite(
            metrics.peak_absolute_acceleration
        )
    )

    assert np.isfinite(
        metrics.minimum_manipulability
    )

    assert np.isfinite(
        metrics.maximum_condition_number
    )

    assert np.isfinite(
        metrics.maximum_rcm_error
    )


def test_rcm_remains_satisfied_during_execution():
    instrument = (
        make_instrument()
    )

    trajectory = (
        time_parameterise_path(
            instrument,
            make_path(),
        )
    )

    metrics = (
        evaluate_execution_metrics(
            instrument,
            trajectory,
        )
    )

    assert (
        metrics.maximum_rcm_error
        <= 1e-8
    )


def test_execution_cost_changes_for_longer_path():
    instrument = (
        make_instrument()
    )

    short_path = np.asarray(
        [
            make_path()[0],
            make_path()[-1],
        ]
    )

    long_path = make_path()

    short_metrics = (
        evaluate_execution_metrics(
            instrument,
            time_parameterise_path(
                instrument,
                short_path,
            ),
        )
    )

    long_metrics = (
        evaluate_execution_metrics(
            instrument,
            time_parameterise_path(
                instrument,
                long_path,
            ),
        )
    )

    assert (
        long_metrics.scaled_joint_path_cost
        >= short_metrics.scaled_joint_path_cost
        - 1e-12
    )