"""Phase 5 temporal tracking and state-estimation benchmark.

The experiment compares raw frame-by-frame 3-D measurements with a
constant-velocity Kalman tracker under:

- variable measurement uncertainty;
- temporary measurement dropout;
- deliberately injected gross outliers;
- smooth simulated anatomical motion.

Metrics include:

- raw measurement position RMSE;
- tracked position RMSE;
- tracked position RMSE during dropout;
- raw finite-difference velocity RMSE;
- tracked velocity RMSE;
- 95% positional covariance coverage;
- innovation-gate outlier rejection precision and recall.

Ground-truth motion is used only for evaluation.

This is simulation-only engineering validation and does not constitute
clinical anatomical-motion tracking validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.perception.state_estimation import (
    ConstantVelocityKalmanFilter,
    KalmanFilterConfig,
)


OUTPUT_PATH = Path(
    "results/phase5/phase5_tracking_benchmark.json"
)

RANDOM_SEED = 5701

TRIALS = 20

STEPS = 120

DT = 0.05

DROPOUT_PROBABILITY = 0.10

OUTLIER_PROBABILITY = 0.06

MINIMUM_MEASUREMENT_SIGMA = 0.0015

MAXIMUM_MEASUREMENT_SIGMA = 0.0040

OUTLIER_MINIMUM_MAGNITUDE = 0.025

OUTLIER_MAXIMUM_MAGNITUDE = 0.050

# 99% chi-square threshold with 3 degrees of freedom.
INNOVATION_GATE_THRESHOLD = 11.344866730144373

# 95% chi-square threshold with 3 degrees of freedom.
POSITION_COVERAGE_THRESHOLD = 7.814727903251179


def true_motion(
    timestamp: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return simulated anatomical position and velocity."""

    initial_position = np.asarray(
        [
            0.060,
            -0.020,
            0.180,
        ],
        dtype=float,
    )

    base_velocity = np.asarray(
        [
            0.0040,
            0.0020,
            -0.0010,
        ],
        dtype=float,
    )

    angular_frequency = (
        2.0
        * np.pi
        * 0.35
    )

    oscillation_amplitude = np.asarray(
        [
            0.0025,
            0.0018,
            0.0012,
        ],
        dtype=float,
    )

    phase = np.asarray(
        [
            0.0,
            0.8,
            1.4,
        ],
        dtype=float,
    )

    position = (
        initial_position
        + base_velocity
        * timestamp
        + oscillation_amplitude
        * np.sin(
            angular_frequency
            * timestamp
            + phase
        )
    )

    velocity = (
        base_velocity
        + oscillation_amplitude
        * angular_frequency
        * np.cos(
            angular_frequency
            * timestamp
            + phase
        )
    )

    return (
        np.asarray(
            position,
            dtype=float,
        ),
        np.asarray(
            velocity,
            dtype=float,
        ),
    )


def measurement_covariance(
    sigma: float,
) -> np.ndarray:
    """Return isotropic measurement covariance."""

    return (
        float(sigma) ** 2
    ) * np.eye(
        3,
        dtype=float,
    )


def random_outlier_offset(
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a random gross 3-D outlier displacement."""

    direction = rng.normal(
        0.0,
        1.0,
        size=3,
    )

    norm = float(
        np.linalg.norm(
            direction
        )
    )

    if norm <= 1e-12:
        direction = np.asarray(
            [
                1.0,
                0.0,
                0.0,
            ],
            dtype=float,
        )
    else:
        direction = (
            direction
            / norm
        )

    magnitude = float(
        rng.uniform(
            OUTLIER_MINIMUM_MAGNITUDE,
            OUTLIER_MAXIMUM_MAGNITUDE,
        )
    )

    return (
        magnitude
        * direction
    )


def rms_vector_error(
    errors: list[np.ndarray],
) -> float | None:
    """Return RMS Euclidean vector error."""

    if not errors:
        return None

    array = np.vstack(
        errors
    )

    return float(
        np.sqrt(
            np.mean(
                np.sum(
                    array**2,
                    axis=1,
                )
            )
        )
    )


def run_trial(
    *,
    seed: int,
) -> dict:
    """Run one deterministic moving-target tracking trial."""

    rng = np.random.default_rng(
        seed
    )

    tracker = (
        ConstantVelocityKalmanFilter(
            KalmanFilterConfig(
                acceleration_sigma=0.012,
                initial_velocity_sigma=0.030,
                innovation_gate_threshold=(
                    INNOVATION_GATE_THRESHOLD
                ),
            )
        )
    )

    timestamps = (
        np.arange(
            STEPS,
            dtype=float,
        )
        * DT
    )

    truth_positions = []
    truth_velocities = []

    measurements: list[
        np.ndarray | None
    ] = []

    measurement_covariances: list[
        np.ndarray | None
    ] = []

    dropout_flags = []
    outlier_flags = []

    for index, timestamp in enumerate(
        timestamps
    ):
        true_position, true_velocity = (
            true_motion(
                float(timestamp)
            )
        )

        truth_positions.append(
            true_position
        )

        truth_velocities.append(
            true_velocity
        )

        # First measurement is always available.
        dropout = bool(
            index > 0
            and rng.random()
            < DROPOUT_PROBABILITY
        )

        dropout_flags.append(
            dropout
        )

        if dropout:
            measurements.append(
                None
            )

            measurement_covariances.append(
                None
            )

            outlier_flags.append(
                False
            )

            continue

        sigma = float(
            rng.uniform(
                MINIMUM_MEASUREMENT_SIGMA,
                MAXIMUM_MEASUREMENT_SIGMA,
            )
        )

        covariance = (
            measurement_covariance(
                sigma
            )
        )

        measurement = (
            true_position
            + rng.normal(
                loc=0.0,
                scale=sigma,
                size=3,
            )
        )

        outlier = bool(
            index > 0
            and rng.random()
            < OUTLIER_PROBABILITY
        )

        if outlier:
            measurement = (
                measurement
                + random_outlier_offset(
                    rng
                )
            )

        measurements.append(
            np.asarray(
                measurement,
                dtype=float,
            )
        )

        measurement_covariances.append(
            covariance
        )

        outlier_flags.append(
            outlier
        )

    first_measurement = (
        measurements[0]
    )

    first_covariance = (
        measurement_covariances[0]
    )

    if (
        first_measurement is None
        or first_covariance is None
    ):
        raise RuntimeError(
            "First measurement must be available."
        )

    state = tracker.initialise(
        measurement=first_measurement,
        measurement_covariance=(
            first_covariance
        ),
        timestamp=float(
            timestamps[0]
        ),
    )

    raw_position_errors: list[
        np.ndarray
    ] = []

    raw_non_outlier_position_errors: list[
        np.ndarray
    ] = []

    tracked_position_errors: list[
        np.ndarray
    ] = []

    tracked_measurement_step_errors: list[
        np.ndarray
    ] = []

    tracked_dropout_errors: list[
        np.ndarray
    ] = []

    tracked_velocity_errors: list[
        np.ndarray
    ] = []

    raw_velocity_errors: list[
        np.ndarray
    ] = []

    coverage_values: list[
        bool
    ] = []

    rejection_truth: list[
        bool
    ] = []

    rejection_prediction: list[
        bool
    ] = []

    accepted_measurements = 1
    rejected_measurements = 0

    previous_raw_measurement = (
        first_measurement.copy()
    )

    previous_raw_timestamp = float(
        timestamps[0]
    )

    initial_truth_position = (
        truth_positions[0]
    )

    initial_position_error = (
        state.position
        - initial_truth_position
    )

    tracked_position_errors.append(
        initial_position_error
    )

    tracked_measurement_step_errors.append(
        initial_position_error
    )

    raw_position_errors.append(
        first_measurement
        - initial_truth_position
    )

    raw_non_outlier_position_errors.append(
        first_measurement
        - initial_truth_position
    )

    for index in range(
        1,
        STEPS
    ):
        timestamp = float(
            timestamps[index]
        )

        truth_position = (
            truth_positions[index]
        )

        truth_velocity = (
            truth_velocities[index]
        )

        measurement = (
            measurements[index]
        )

        covariance = (
            measurement_covariances[index]
        )

        if measurement is None:
            state = tracker.predict(
                state,
                timestamp=timestamp,
            )

            tracked_dropout_errors.append(
                state.position
                - truth_position
            )

        else:
            raw_error = (
                measurement
                - truth_position
            )

            raw_position_errors.append(
                raw_error
            )

            if not outlier_flags[index]:
                raw_non_outlier_position_errors.append(
                    raw_error
                )

            raw_dt = (
                timestamp
                - previous_raw_timestamp
            )

            if raw_dt > 0.0:
                raw_velocity = (
                    measurement
                    - previous_raw_measurement
                ) / raw_dt

                raw_velocity_errors.append(
                    raw_velocity
                    - truth_velocity
                )

            previous_raw_measurement = (
                measurement.copy()
            )

            previous_raw_timestamp = (
                timestamp
            )

            if covariance is None:
                raise RuntimeError(
                    "Measurement covariance is missing."
                )

            result = tracker.step(
                state,
                measurement=measurement,
                measurement_covariance=(
                    covariance
                ),
                timestamp=timestamp,
            )

            rejected = (
                not result.accepted
            )

            rejection_truth.append(
                bool(
                    outlier_flags[index]
                )
            )

            rejection_prediction.append(
                rejected
            )

            if result.accepted:
                accepted_measurements += 1
            else:
                rejected_measurements += 1

            state = (
                result.state
            )

            tracked_measurement_step_errors.append(
                state.position
                - truth_position
            )

        position_error = (
            state.position
            - truth_position
        )

        velocity_error = (
            state.velocity
            - truth_velocity
        )

        tracked_position_errors.append(
            position_error
        )

        tracked_velocity_errors.append(
            velocity_error
        )

        position_covariance = (
            state.position_covariance
        )

        inverse_covariance = (
            np.linalg.pinv(
                position_covariance,
                hermitian=True,
            )
        )

        mahalanobis_squared = float(
            position_error.T
            @ inverse_covariance
            @ position_error
        )

        coverage_values.append(
            bool(
                mahalanobis_squared
                <= POSITION_COVERAGE_THRESHOLD
            )
        )

    true_outliers = np.asarray(
        rejection_truth,
        dtype=bool,
    )

    predicted_rejections = np.asarray(
        rejection_prediction,
        dtype=bool,
    )

    true_positive = int(
        np.count_nonzero(
            true_outliers
            & predicted_rejections
        )
    )

    false_positive = int(
        np.count_nonzero(
            ~true_outliers
            & predicted_rejections
        )
    )

    false_negative = int(
        np.count_nonzero(
            true_outliers
            & ~predicted_rejections
        )
    )

    rejection_precision = float(
        true_positive
        / max(
            true_positive
            + false_positive,
            1,
        )
    )

    rejection_recall = float(
        true_positive
        / max(
            true_positive
            + false_negative,
            1,
        )
    )

    raw_rmse = (
        rms_vector_error(
            raw_position_errors
        )
    )

    clean_raw_rmse = (
        rms_vector_error(
            raw_non_outlier_position_errors
        )
    )

    tracked_rmse = (
        rms_vector_error(
            tracked_position_errors
        )
    )

    tracked_measurement_rmse = (
        rms_vector_error(
            tracked_measurement_step_errors
        )
    )

    dropout_rmse = (
        rms_vector_error(
            tracked_dropout_errors
        )
    )

    tracked_velocity_rmse = (
        rms_vector_error(
            tracked_velocity_errors
        )
    )

    raw_velocity_rmse = (
        rms_vector_error(
            raw_velocity_errors
        )
    )

    if (
        raw_rmse is None
        or clean_raw_rmse is None
        or tracked_rmse is None
        or tracked_measurement_rmse is None
        or tracked_velocity_rmse is None
        or raw_velocity_rmse is None
    ):
        raise RuntimeError(
            "Insufficient samples for tracking metrics."
        )

    return {
        "seed": int(seed),
        "steps": int(STEPS),
        "duration_seconds": float(
            timestamps[-1]
        ),
        "measurement_count": int(
            np.count_nonzero(
                [
                    value is not None
                    for value in measurements
                ]
            )
        ),
        "dropout_count": int(
            np.count_nonzero(
                dropout_flags
            )
        ),
        "outlier_count": int(
            np.count_nonzero(
                outlier_flags
            )
        ),
        "accepted_measurements": int(
            accepted_measurements
        ),
        "rejected_measurements": int(
            rejected_measurements
        ),
        "raw_position_rmse_metres": float(
            raw_rmse
        ),
        "raw_non_outlier_position_rmse_metres": float(
            clean_raw_rmse
        ),
        "tracked_position_rmse_metres": float(
            tracked_rmse
        ),
        "tracked_position_rmse_measurement_steps_metres": float(
            tracked_measurement_rmse
        ),
        "tracked_position_rmse_dropout_steps_metres": (
            None
            if dropout_rmse is None
            else float(
                dropout_rmse
            )
        ),
        "raw_finite_difference_velocity_rmse_metres_per_second": float(
            raw_velocity_rmse
        ),
        "tracked_velocity_rmse_metres_per_second": float(
            tracked_velocity_rmse
        ),
        "position_covariance_95_coverage": float(
            np.mean(
                coverage_values
            )
        ),
        "outlier_rejection_precision": float(
            rejection_precision
        ),
        "outlier_rejection_recall": float(
            rejection_recall
        ),
    }


def aggregate(
    records: list[dict],
    key: str,
) -> dict:
    """Aggregate one metric across trials."""

    values = [
        float(
            record[key]
        )
        for record in records
        if record[key] is not None
    ]

    if not values:
        return {
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
        }

    array = np.asarray(
        values,
        dtype=float,
    )

    return {
        "mean": float(
            np.mean(array)
        ),
        "median": float(
            np.median(array)
        ),
        "minimum": float(
            np.min(array)
        ),
        "maximum": float(
            np.max(array)
        ),
    }


def run_benchmark() -> dict:
    """Run all temporal tracking trials."""

    records = [
        run_trial(
            seed=(
                RANDOM_SEED
                + index
            )
        )
        for index in range(
            TRIALS
        )
    ]

    raw_position = aggregate(
        records,
        "raw_position_rmse_metres",
    )

    clean_raw_position = aggregate(
        records,
        "raw_non_outlier_position_rmse_metres",
    )

    tracked_position = aggregate(
        records,
        "tracked_position_rmse_metres",
    )

    tracked_measurement_position = (
        aggregate(
            records,
            "tracked_position_rmse_measurement_steps_metres",
        )
    )

    tracked_dropout_position = (
        aggregate(
            records,
            "tracked_position_rmse_dropout_steps_metres",
        )
    )

    raw_velocity = aggregate(
        records,
        "raw_finite_difference_velocity_rmse_metres_per_second",
    )

    tracked_velocity = aggregate(
        records,
        "tracked_velocity_rmse_metres_per_second",
    )

    coverage = aggregate(
        records,
        "position_covariance_95_coverage",
    )

    rejection_precision = aggregate(
        records,
        "outlier_rejection_precision",
    )

    rejection_recall = aggregate(
        records,
        "outlier_rejection_recall",
    )

    tracker_better_position_fraction = float(
        np.mean(
            [
                record[
                    "tracked_position_rmse_measurement_steps_metres"
                ]
                < record[
                    "raw_position_rmse_metres"
                ]
                for record in records
            ]
        )
    )

    tracker_better_velocity_fraction = float(
        np.mean(
            [
                record[
                    "tracked_velocity_rmse_metres_per_second"
                ]
                < record[
                    "raw_finite_difference_velocity_rmse_metres_per_second"
                ]
                for record in records
            ]
        )
    )

    evidence = {
        "phase": 5,
        "experiment": (
            "temporal_state_estimation_with_uncertain_measurements"
        ),
        "scope": "simulation_only",
        "random_seed": int(
            RANDOM_SEED
        ),
        "trials": int(
            TRIALS
        ),
        "steps_per_trial": int(
            STEPS
        ),
        "dt_seconds": float(
            DT
        ),
        "dropout_probability": float(
            DROPOUT_PROBABILITY
        ),
        "outlier_probability": float(
            OUTLIER_PROBABILITY
        ),
        "innovation_gate_threshold": float(
            INNOVATION_GATE_THRESHOLD
        ),
        "summary": {
            "raw_position_rmse_metres": (
                raw_position
            ),
            "raw_non_outlier_position_rmse_metres": (
                clean_raw_position
            ),
            "tracked_position_rmse_metres": (
                tracked_position
            ),
            "tracked_position_rmse_measurement_steps_metres": (
                tracked_measurement_position
            ),
            "tracked_position_rmse_dropout_steps_metres": (
                tracked_dropout_position
            ),
            "raw_velocity_rmse_metres_per_second": (
                raw_velocity
            ),
            "tracked_velocity_rmse_metres_per_second": (
                tracked_velocity
            ),
            "position_covariance_95_coverage": (
                coverage
            ),
            "outlier_rejection_precision": (
                rejection_precision
            ),
            "outlier_rejection_recall": (
                rejection_recall
            ),
            "tracker_lower_position_rmse_fraction": (
                tracker_better_position_fraction
            ),
            "tracker_lower_velocity_rmse_fraction": (
                tracker_better_velocity_fraction
            ),
        },
        "trial_records": records,
        "claims_note": (
            "Results evaluate a simulated constant-velocity Kalman tracker "
            "under synthetic smooth motion, measurement noise, dropouts and "
            "gross outliers. Ground truth is evaluation-only. Results do "
            "not establish clinical anatomical-motion tracking accuracy."
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            evidence,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "Phase 5 Temporal Tracking + State Estimation Benchmark"
    )
    print(
        "=" * 52
    )
    print(
        f"Trials: {TRIALS}"
    )
    print(
        f"Steps/trial: {STEPS}"
    )
    print(
        f"Dropout probability: {100.0 * DROPOUT_PROBABILITY:.1f}%"
    )
    print(
        f"Outlier probability: {100.0 * OUTLIER_PROBABILITY:.1f}%"
    )
    print()

    print(
        "Raw position RMSE: "
        f"{1000.0 * raw_position['mean']:.3f} mm"
    )

    print(
        "Raw non-outlier position RMSE: "
        f"{1000.0 * clean_raw_position['mean']:.3f} mm"
    )

    print(
        "Tracked position RMSE: "
        f"{1000.0 * tracked_position['mean']:.3f} mm"
    )

    print(
        "Tracked RMSE on measurement steps: "
        f"{1000.0 * tracked_measurement_position['mean']:.3f} mm"
    )

    print(
        "Tracked RMSE during dropout: "
        f"{1000.0 * tracked_dropout_position['mean']:.3f} mm"
    )

    print()

    print(
        "Raw finite-difference velocity RMSE: "
        f"{1000.0 * raw_velocity['mean']:.3f} mm/s"
    )

    print(
        "Tracked velocity RMSE: "
        f"{1000.0 * tracked_velocity['mean']:.3f} mm/s"
    )

    print()

    print(
        "95% position covariance coverage: "
        f"{100.0 * coverage['mean']:.1f}%"
    )

    print(
        "Outlier rejection precision: "
        f"{100.0 * rejection_precision['mean']:.1f}%"
    )

    print(
        "Outlier rejection recall: "
        f"{100.0 * rejection_recall['mean']:.1f}%"
    )

    print()

    print(
        "Tracker lower position RMSE: "
        f"{100.0 * tracker_better_position_fraction:.1f}% of trials"
    )

    print(
        "Tracker lower velocity RMSE: "
        f"{100.0 * tracker_better_velocity_fraction:.1f}% of trials"
    )

    print()

    print(
        "Evidence written to:"
    )

    print(
        OUTPUT_PATH
    )

    return evidence


def main() -> None:
    run_benchmark()


if __name__ == "__main__":
    main()