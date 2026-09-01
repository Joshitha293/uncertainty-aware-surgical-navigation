"""Phase 5 held-out state-estimation calibration benchmark.

The previous tracking benchmark demonstrated substantially lower position
and velocity error than raw measurements, but nominal 95% positional
covariance coverage was below 95%, indicating an overconfident state
estimate.

This experiment tunes only the constant-velocity model's acceleration
process-noise standard deviation using validation trials.

Procedure
---------
1. Evaluate several acceleration_sigma candidates on validation seeds.
2. Select the candidate whose empirical 95% covariance coverage is closest
   to 95%, using tracked position RMSE as a secondary criterion.
3. Freeze that value.
4. Evaluate it once on a disjoint held-out test seed set.

The held-out seeds are not used for hyperparameter selection.

Ground truth is used only for simulation evaluation.

This experiment does not constitute clinical anatomical-motion tracking
validation or clinical uncertainty calibration.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.perception.state_estimation import (
    ConstantVelocityKalmanFilter,
    KalmanFilterConfig,
)

from src.simulation.phase5_tracking_benchmark import (
    DROPOUT_PROBABILITY,
    DT,
    INNOVATION_GATE_THRESHOLD,
    MAXIMUM_MEASUREMENT_SIGMA,
    MINIMUM_MEASUREMENT_SIGMA,
    OUTLIER_MAXIMUM_MAGNITUDE,
    OUTLIER_MINIMUM_MAGNITUDE,
    OUTLIER_PROBABILITY,
    POSITION_COVERAGE_THRESHOLD,
    STEPS,
    true_motion,
)


OUTPUT_PATH = Path(
    "results/phase5/phase5_tracking_calibration_benchmark.json"
)

VALIDATION_SEED_START = 5901
VALIDATION_TRIALS = 12

HELDOUT_SEED_START = 6901
HELDOUT_TRIALS = 20

CANDIDATE_ACCELERATION_SIGMAS = (
    0.008,
    0.012,
    0.018,
    0.025,
    0.035,
    0.050,
)


def measurement_covariance(
    sigma: float,
) -> np.ndarray:
    """Return isotropic 3-D measurement covariance."""

    return (
        float(sigma) ** 2
        * np.eye(
            3,
            dtype=float,
        )
    )


def random_outlier_offset(
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one gross random measurement displacement."""

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
) -> float:
    """Return RMS Euclidean vector error."""

    if not errors:
        raise RuntimeError(
            "No vector errors were supplied."
        )

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


def generate_measurements(
    *,
    seed: int,
) -> dict:
    """Generate one deterministic simulated tracking sequence."""

    rng = np.random.default_rng(
        seed
    )

    timestamps = (
        np.arange(
            STEPS,
            dtype=float,
        )
        * DT
    )

    truth_positions: list[np.ndarray] = []
    truth_velocities: list[np.ndarray] = []

    measurements: list[
        np.ndarray | None
    ] = []

    measurement_covariances: list[
        np.ndarray | None
    ] = []

    dropout_flags: list[bool] = []
    outlier_flags: list[bool] = []

    for index, timestamp in enumerate(
        timestamps
    ):
        position, velocity = (
            true_motion(
                float(timestamp)
            )
        )

        truth_positions.append(
            position
        )

        truth_velocities.append(
            velocity
        )

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
            position
            + rng.normal(
                0.0,
                sigma,
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

    return {
        "timestamps": timestamps,
        "truth_positions": truth_positions,
        "truth_velocities": truth_velocities,
        "measurements": measurements,
        "measurement_covariances": (
            measurement_covariances
        ),
        "dropout_flags": dropout_flags,
        "outlier_flags": outlier_flags,
    }


def run_trial(
    *,
    seed: int,
    acceleration_sigma: float,
) -> dict:
    """Run one sequence using one frozen process-noise setting."""

    sequence = generate_measurements(
        seed=seed
    )

    timestamps = sequence[
        "timestamps"
    ]

    truth_positions = sequence[
        "truth_positions"
    ]

    truth_velocities = sequence[
        "truth_velocities"
    ]

    measurements = sequence[
        "measurements"
    ]

    measurement_covariances = sequence[
        "measurement_covariances"
    ]

    dropout_flags = sequence[
        "dropout_flags"
    ]

    outlier_flags = sequence[
        "outlier_flags"
    ]

    tracker = (
        ConstantVelocityKalmanFilter(
            KalmanFilterConfig(
                acceleration_sigma=(
                    acceleration_sigma
                ),
                initial_velocity_sigma=0.030,
                innovation_gate_threshold=(
                    INNOVATION_GATE_THRESHOLD
                ),
            )
        )
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

    raw_errors: list[np.ndarray] = []

    raw_clean_errors: list[np.ndarray] = []

    tracked_errors: list[np.ndarray] = []

    tracked_dropout_errors: list[
        np.ndarray
    ] = []

    tracked_velocity_errors: list[
        np.ndarray
    ] = []

    coverage: list[bool] = []

    outlier_truth: list[bool] = []

    rejected_prediction: list[bool] = []

    initial_error = (
        state.position
        - truth_positions[0]
    )

    tracked_errors.append(
        initial_error
    )

    raw_errors.append(
        first_measurement
        - truth_positions[0]
    )

    raw_clean_errors.append(
        first_measurement
        - truth_positions[0]
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

            raw_errors.append(
                raw_error
            )

            if not outlier_flags[index]:
                raw_clean_errors.append(
                    raw_error
                )

            if covariance is None:
                raise RuntimeError(
                    "Measurement covariance missing."
                )

            result = tracker.step(
                state,
                measurement=measurement,
                measurement_covariance=(
                    covariance
                ),
                timestamp=timestamp,
            )

            outlier_truth.append(
                bool(
                    outlier_flags[index]
                )
            )

            rejected_prediction.append(
                not result.accepted
            )

            state = result.state

        position_error = (
            state.position
            - truth_position
        )

        velocity_error = (
            state.velocity
            - truth_velocity
        )

        tracked_errors.append(
            position_error
        )

        tracked_velocity_errors.append(
            velocity_error
        )

        covariance = (
            state.position_covariance
        )

        inverse_covariance = (
            np.linalg.pinv(
                covariance,
                hermitian=True,
            )
        )

        mahalanobis_squared = float(
            position_error.T
            @ inverse_covariance
            @ position_error
        )

        coverage.append(
            mahalanobis_squared
            <= POSITION_COVERAGE_THRESHOLD
        )

    truth_array = np.asarray(
        outlier_truth,
        dtype=bool,
    )

    rejection_array = np.asarray(
        rejected_prediction,
        dtype=bool,
    )

    true_positive = int(
        np.count_nonzero(
            truth_array
            & rejection_array
        )
    )

    false_positive = int(
        np.count_nonzero(
            ~truth_array
            & rejection_array
        )
    )

    false_negative = int(
        np.count_nonzero(
            truth_array
            & ~rejection_array
        )
    )

    precision = float(
        true_positive
        / max(
            true_positive
            + false_positive,
            1,
        )
    )

    recall = float(
        true_positive
        / max(
            true_positive
            + false_negative,
            1,
        )
    )

    dropout_rmse = (
        None
        if not tracked_dropout_errors
        else rms_vector_error(
            tracked_dropout_errors
        )
    )

    return {
        "seed": int(seed),
        "acceleration_sigma": float(
            acceleration_sigma
        ),
        "raw_position_rmse_metres": (
            rms_vector_error(
                raw_errors
            )
        ),
        "raw_clean_position_rmse_metres": (
            rms_vector_error(
                raw_clean_errors
            )
        ),
        "tracked_position_rmse_metres": (
            rms_vector_error(
                tracked_errors
            )
        ),
        "tracked_dropout_rmse_metres": (
            dropout_rmse
        ),
        "tracked_velocity_rmse_metres_per_second": (
            rms_vector_error(
                tracked_velocity_errors
            )
        ),
        "position_covariance_95_coverage": float(
            np.mean(
                coverage
            )
        ),
        "outlier_rejection_precision": (
            precision
        ),
        "outlier_rejection_recall": (
            recall
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
    }


def aggregate(
    records: list[dict],
    key: str,
) -> dict:
    """Aggregate one scalar metric."""

    values = [
        float(
            record[key]
        )
        for record in records
        if record[key] is not None
    ]

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


def summarise_trials(
    records: list[dict],
) -> dict:
    """Summarise repeated tracking trials."""

    return {
        "raw_position_rmse_metres": aggregate(
            records,
            "raw_position_rmse_metres",
        ),
        "raw_clean_position_rmse_metres": aggregate(
            records,
            "raw_clean_position_rmse_metres",
        ),
        "tracked_position_rmse_metres": aggregate(
            records,
            "tracked_position_rmse_metres",
        ),
        "tracked_dropout_rmse_metres": aggregate(
            records,
            "tracked_dropout_rmse_metres",
        ),
        "tracked_velocity_rmse_metres_per_second": aggregate(
            records,
            "tracked_velocity_rmse_metres_per_second",
        ),
        "position_covariance_95_coverage": aggregate(
            records,
            "position_covariance_95_coverage",
        ),
        "outlier_rejection_precision": aggregate(
            records,
            "outlier_rejection_precision",
        ),
        "outlier_rejection_recall": aggregate(
            records,
            "outlier_rejection_recall",
        ),
    }


def validation_score(
    summary: dict,
) -> tuple[
    float,
    float,
]:
    """Return validation-only candidate-selection score.

    Primary objective:
        absolute difference from nominal 95% covariance coverage.

    Secondary objective:
        tracked position RMSE.
    """

    coverage = float(
        summary[
            "position_covariance_95_coverage"
        ][
            "mean"
        ]
    )

    position_rmse = float(
        summary[
            "tracked_position_rmse_metres"
        ][
            "mean"
        ]
    )

    return (
        abs(
            coverage
            - 0.95
        ),
        position_rmse,
    )


def run_validation() -> tuple[
    float,
    list[dict],
]:
    """Select process noise using validation data only."""

    candidate_results = []

    validation_seeds = [
        VALIDATION_SEED_START
        + index
        for index in range(
            VALIDATION_TRIALS
        )
    ]

    for sigma in (
        CANDIDATE_ACCELERATION_SIGMAS
    ):
        records = [
            run_trial(
                seed=seed,
                acceleration_sigma=(
                    sigma
                ),
            )
            for seed in validation_seeds
        ]

        summary = summarise_trials(
            records
        )

        score = validation_score(
            summary
        )

        candidate_results.append(
            {
                "acceleration_sigma": float(
                    sigma
                ),
                "coverage_distance_from_0_95": float(
                    score[0]
                ),
                "summary": summary,
            }
        )

    selected = min(
        candidate_results,
        key=lambda record: (
            record[
                "coverage_distance_from_0_95"
            ],
            record[
                "summary"
            ][
                "tracked_position_rmse_metres"
            ][
                "mean"
            ],
            record[
                "acceleration_sigma"
            ],
        ),
    )

    return (
        float(
            selected[
                "acceleration_sigma"
            ]
        ),
        candidate_results,
    )


def run_heldout(
    *,
    acceleration_sigma: float,
) -> tuple[
    list[dict],
    dict,
]:
    """Evaluate frozen parameter on unseen held-out seeds."""

    heldout_seeds = [
        HELDOUT_SEED_START
        + index
        for index in range(
            HELDOUT_TRIALS
        )
    ]

    records = [
        run_trial(
            seed=seed,
            acceleration_sigma=(
                acceleration_sigma
            ),
        )
        for seed in heldout_seeds
    ]

    return (
        records,
        summarise_trials(
            records
        ),
    )


def run_benchmark() -> dict:
    """Run validation tuning followed by held-out evaluation."""

    selected_sigma, validation_results = (
        run_validation()
    )

    (
        heldout_records,
        heldout_summary,
    ) = run_heldout(
        acceleration_sigma=(
            selected_sigma
        )
    )

    validation_seed_set = set(
        VALIDATION_SEED_START
        + index
        for index in range(
            VALIDATION_TRIALS
        )
    )

    heldout_seed_set = set(
        HELDOUT_SEED_START
        + index
        for index in range(
            HELDOUT_TRIALS
        )
    )

    if not validation_seed_set.isdisjoint(
        heldout_seed_set
    ):
        raise RuntimeError(
            "Validation and held-out seed sets overlap."
        )

    evidence = {
        "phase": 5,
        "experiment": (
            "validation_tuned_tracking_uncertainty_calibration"
        ),
        "scope": "simulation_only",
        "selection_rule": (
            "Minimise validation absolute deviation from 0.95 positional "
            "covariance coverage; break ties using tracked position RMSE."
        ),
        "candidate_acceleration_sigmas": [
            float(value)
            for value in (
                CANDIDATE_ACCELERATION_SIGMAS
            )
        ],
        "validation_seed_start": int(
            VALIDATION_SEED_START
        ),
        "validation_trials": int(
            VALIDATION_TRIALS
        ),
        "heldout_seed_start": int(
            HELDOUT_SEED_START
        ),
        "heldout_trials": int(
            HELDOUT_TRIALS
        ),
        "seed_sets_disjoint": True,
        "selected_acceleration_sigma": float(
            selected_sigma
        ),
        "validation_candidates": (
            validation_results
        ),
        "heldout_summary": (
            heldout_summary
        ),
        "heldout_trial_records": (
            heldout_records
        ),
        "claims_note": (
            "Process noise was selected using validation trials before "
            "evaluation on disjoint held-out simulation seeds. The result "
            "assesses calibration only for this synthetic motion/noise "
            "model and does not establish clinical uncertainty calibration."
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
        "Phase 5 Tracking Calibration Benchmark"
    )

    print(
        "=" * 38
    )

    print()

    print(
        "VALIDATION CANDIDATES"
    )

    for candidate in (
        validation_results
    ):
        sigma = candidate[
            "acceleration_sigma"
        ]

        summary = candidate[
            "summary"
        ]

        coverage = (
            summary[
                "position_covariance_95_coverage"
            ][
                "mean"
            ]
        )

        position_rmse = (
            summary[
                "tracked_position_rmse_metres"
            ][
                "mean"
            ]
        )

        velocity_rmse = (
            summary[
                "tracked_velocity_rmse_metres_per_second"
            ][
                "mean"
            ]
        )

        print(
            f"sigma={sigma:.3f} | "
            f"coverage={100.0 * coverage:.1f}% | "
            f"position RMSE={1000.0 * position_rmse:.3f} mm | "
            f"velocity RMSE={1000.0 * velocity_rmse:.3f} mm/s"
        )

    print()

    print(
        "Selected acceleration sigma: "
        f"{selected_sigma:.3f} m/s^2"
    )

    print()

    print(
        "HELD-OUT EVALUATION"
    )

    heldout_coverage = (
        heldout_summary[
            "position_covariance_95_coverage"
        ][
            "mean"
        ]
    )

    heldout_position = (
        heldout_summary[
            "tracked_position_rmse_metres"
        ][
            "mean"
        ]
    )

    heldout_dropout = (
        heldout_summary[
            "tracked_dropout_rmse_metres"
        ][
            "mean"
        ]
    )

    heldout_velocity = (
        heldout_summary[
            "tracked_velocity_rmse_metres_per_second"
        ][
            "mean"
        ]
    )

    heldout_precision = (
        heldout_summary[
            "outlier_rejection_precision"
        ][
            "mean"
        ]
    )

    heldout_recall = (
        heldout_summary[
            "outlier_rejection_recall"
        ][
            "mean"
        ]
    )

    raw_position = (
        heldout_summary[
            "raw_position_rmse_metres"
        ][
            "mean"
        ]
    )

    clean_raw = (
        heldout_summary[
            "raw_clean_position_rmse_metres"
        ][
            "mean"
        ]
    )

    print(
        "Raw position RMSE: "
        f"{1000.0 * raw_position:.3f} mm"
    )

    print(
        "Raw non-outlier position RMSE: "
        f"{1000.0 * clean_raw:.3f} mm"
    )

    print(
        "Tracked position RMSE: "
        f"{1000.0 * heldout_position:.3f} mm"
    )

    print(
        "Tracked dropout RMSE: "
        f"{1000.0 * heldout_dropout:.3f} mm"
    )

    print(
        "Tracked velocity RMSE: "
        f"{1000.0 * heldout_velocity:.3f} mm/s"
    )

    print(
        "Nominal 95% covariance coverage: "
        f"{100.0 * heldout_coverage:.1f}%"
    )

    print(
        "Outlier rejection precision: "
        f"{100.0 * heldout_precision:.1f}%"
    )

    print(
        "Outlier rejection recall: "
        f"{100.0 * heldout_recall:.1f}%"
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