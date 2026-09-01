"""Final Phase 5 registration-tracking-navigation integration benchmark.

This benchmark evaluates the complete Phase 5 pathway:

    uncertain 3-D perception-frame observation
        ->
    estimated rigid registration
        ->
    propagated registration + measurement covariance
        ->
    temporally fused Kalman state estimate
        ->
    EstimatedStructure
        ->
    uncertainty-aware planning margin

Registration is estimated from noisy fiducial correspondences containing
gross outliers using RANSAC. The true transform is retained only for
evaluation.

Sequential measurements contain:

- heteroscedastic position noise;
- temporary observation dropout;
- gross measurement outliers.

The Kalman process-noise parameter is frozen at the value selected in the
previous validation-only calibration experiment:

    acceleration_sigma = 0.025 m/s^2

Metrics include:

- registration transform error;
- frame-by-frame registered position RMSE;
- temporally tracked position RMSE;
- dropout tracking RMSE;
- raw and tracked covariance coverage;
- measurement-outlier rejection;
- planner safety-margin propagation.

This is simulation-only engineering validation. It does not establish
clinical registration, anatomical tracking, navigation accuracy, or
medical-device safety.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from src.geometry.registration import (
    transform_error,
    transform_points,
)
from src.geometry.robust_registration import (
    ransac_rigid_registration,
)
from src.geometry.transforms import (
    invert_transform,
    make_transform,
    transform_point,
)
from src.perception.perception import (
    PerceptionResult,
)
from src.perception.planning import (
    uncertainty_aware_planning_structures,
)
from src.perception.state_estimation import (
    ConstantVelocityKalmanFilter,
    KalmanFilterConfig,
)
from src.perception.tracked_navigation import (
    integrate_learned_stereo_observation,
    register_position_measurement,
    tracking_state_to_perception_result,
)
from src.perception.uncertainty import (
    EstimatedStructure,
    PositionUncertainty,
)
from src.simulation.phase5_tracking_benchmark import (
    true_motion,
)


OUTPUT_PATH = Path(
    "results/phase5/phase5_final_integration_benchmark.json"
)

RANDOM_SEED = 7501

TRIALS = 20

STEPS = 120

DT = 0.05

DROPOUT_PROBABILITY = 0.10

OUTLIER_PROBABILITY = 0.06

MINIMUM_MEASUREMENT_SIGMA = 0.0015

MAXIMUM_MEASUREMENT_SIGMA = 0.0040

OUTLIER_MINIMUM_MAGNITUDE = 0.025

OUTLIER_MAXIMUM_MAGNITUDE = 0.050

REGISTRATION_LANDMARKS = 20

REGISTRATION_OUTLIERS = 4

REGISTRATION_NOISE_SIGMA = 0.0005

RANSAC_INLIER_THRESHOLD = 0.0025

RANSAC_ITERATIONS = 400

CALIBRATED_ACCELERATION_SIGMA = 0.025

INITIAL_VELOCITY_SIGMA = 0.030

INNOVATION_GATE_THRESHOLD = 11.344866730144373

POSITION_COVERAGE_THRESHOLD = 7.814727903251179

PHYSICAL_RADIUS = 0.010

BASE_SAFETY_MARGIN = 0.005

PLANNING_SIGMA_MULTIPLIER = 2.0


@dataclass(frozen=True)
class SyntheticTriangulation:
    """Minimal stereo triangulation interface used by the integration API."""

    point_world: np.ndarray


@dataclass(frozen=True)
class SyntheticStereoObservation:
    """Minimal uncertain stereo result required by tracked navigation."""

    triangulation: SyntheticTriangulation

    world_position_covariance: np.ndarray


def rotation_xyz(
    ax: float,
    ay: float,
    az: float,
) -> np.ndarray:
    """Construct an XYZ rotation matrix."""

    cx = np.cos(ax)
    sx = np.sin(ax)

    cy = np.cos(ay)
    sy = np.sin(ay)

    cz = np.cos(az)
    sz = np.sin(az)

    rx = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, cx, -sx],
            [0.0, sx, cx],
        ],
        dtype=float,
    )

    ry = np.asarray(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ],
        dtype=float,
    )

    rz = np.asarray(
        [
            [cz, -sz, 0.0],
            [sz, cz, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    return (
        rz
        @ ry
        @ rx
    )


def make_true_registration_transform(
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate true perception-to-navigation transform."""

    angles = np.deg2rad(
        rng.uniform(
            low=np.asarray(
                [
                    -8.0,
                    -8.0,
                    -12.0,
                ]
            ),
            high=np.asarray(
                [
                    8.0,
                    8.0,
                    12.0,
                ]
            ),
        )
    )

    translation = rng.uniform(
        low=-0.015,
        high=0.015,
        size=3,
    )

    return make_transform(
        rotation=rotation_xyz(
            float(angles[0]),
            float(angles[1]),
            float(angles[2]),
        ),
        translation=np.asarray(
            translation,
            dtype=float,
        ),
    )


def make_registration_landmarks(
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate non-degenerate perception-frame fiducials."""

    centre = np.asarray(
        [
            0.055,
            -0.020,
            0.180,
        ],
        dtype=float,
    )

    spread = np.asarray(
        [
            0.050,
            0.045,
            0.040,
        ],
        dtype=float,
    )

    points = (
        centre
        + rng.uniform(
            -1.0,
            1.0,
            size=(
                REGISTRATION_LANDMARKS,
                3,
            ),
        )
        * spread
    )

    points[:, 2] += (
        0.12
        * (
            points[:, 0]
            - centre[0]
        )
        - 0.08
        * (
            points[:, 1]
            - centre[1]
        )
    )

    return np.asarray(
        points,
        dtype=float,
    )


def estimate_registration(
    *,
    rng: np.random.Generator,
    true_transform: np.ndarray,
    ransac_seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict,
]:
    """Estimate registration and derive approximate pose covariance.

    Registration-pose covariance is estimated only from observable
    registration residuals and landmark geometry. Ground-truth transform
    error is not used to construct the runtime covariance.
    """

    source_landmarks = (
        make_registration_landmarks(
            rng
        )
    )

    exact_target_landmarks = (
        transform_points(
            true_transform,
            source_landmarks,
        )
    )

    observed_target_landmarks = (
        exact_target_landmarks
        + rng.normal(
            loc=0.0,
            scale=REGISTRATION_NOISE_SIGMA,
            size=exact_target_landmarks.shape,
        )
    )

    outlier_indices = rng.choice(
        source_landmarks.shape[0],
        size=REGISTRATION_OUTLIERS,
        replace=False,
    )

    observed_target_landmarks[
        outlier_indices
    ] += rng.uniform(
        low=-0.050,
        high=0.050,
        size=(
            REGISTRATION_OUTLIERS,
            3,
        ),
    )

    robust = (
        ransac_rigid_registration(
            source_landmarks,
            observed_target_landmarks,
            iterations=RANSAC_ITERATIONS,
            inlier_threshold=(
                RANSAC_INLIER_THRESHOLD
            ),
            seed=ransac_seed,
        )
    )

    estimated_transform = (
        robust
        .registration
        .transform
    )

    inlier_source = source_landmarks[
        robust.inlier_mask
    ]

    centred_inliers = (
        inlier_source
        - np.mean(
            inlier_source,
            axis=0,
        )
    )

    characteristic_radius = float(
        np.sqrt(
            np.mean(
                np.sum(
                    centred_inliers**2,
                    axis=1,
                )
            )
        )
    )

    # Conservative residual-derived translational scale.
    translation_sigma = max(
        float(
            robust
            .registration
            .fre_rms
        ),
        0.00025,
    )

    # Approximate small-angle rotational uncertainty derived from
    # landmark residual scale divided by landmark spatial extent.
    rotation_sigma = (
        translation_sigma
        / max(
            characteristic_radius,
            0.010,
        )
    )

    registration_pose_covariance = np.diag(
        [
            translation_sigma**2,
            translation_sigma**2,
            translation_sigma**2,
            rotation_sigma**2,
            rotation_sigma**2,
            rotation_sigma**2,
        ]
    )

    transform_difference = (
        transform_error(
            estimated_transform,
            true_transform,
        )
    )

    diagnostics = {
        "ransac_inlier_count": int(
            robust.inlier_count
        ),
        "ransac_inlier_fraction": float(
            robust.inlier_fraction
        ),
        "registration_fre_rms_metres": float(
            robust
            .registration
            .fre_rms
        ),
        "estimated_translation_sigma_metres": float(
            translation_sigma
        ),
        "estimated_rotation_sigma_radians": float(
            rotation_sigma
        ),
        "translation_error_metres": float(
            transform_difference
            .translation_error
        ),
        "rotation_error_degrees": float(
            transform_difference
            .rotation_angle_error_degrees
        ),
    }

    return (
        np.asarray(
            estimated_transform,
            dtype=float,
        ),
        np.asarray(
            registration_pose_covariance,
            dtype=float,
        ),
        diagnostics,
    )


def make_measurement_covariance(
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate heteroscedastic anisotropic stereo-position covariance."""

    sigma = float(
        rng.uniform(
            MINIMUM_MEASUREMENT_SIGMA,
            MAXIMUM_MEASUREMENT_SIGMA,
        )
    )

    axis_scale = rng.uniform(
        0.85,
        1.20,
        size=3,
    )

    standard_deviations = (
        sigma
        * axis_scale
    )

    return np.diag(
        standard_deviations**2
    )


def sample_gaussian_measurement_noise(
    *,
    rng: np.random.Generator,
    covariance: np.ndarray,
) -> np.ndarray:
    """Sample one position perturbation from supplied covariance."""

    return rng.multivariate_normal(
        mean=np.zeros(
            3,
            dtype=float,
        ),
        cov=covariance,
    )


def random_outlier_offset(
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one gross perception outlier."""

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


def covariance_contains_truth(
    *,
    error: np.ndarray,
    covariance: np.ndarray,
) -> bool:
    """Check membership in nominal 95% 3-D Gaussian ellipsoid."""

    inverse_covariance = np.linalg.pinv(
        covariance,
        hermitian=True,
    )

    mahalanobis_squared = float(
        error.T
        @ inverse_covariance
        @ error
    )

    return bool(
        mahalanobis_squared
        <= POSITION_COVERAGE_THRESHOLD
    )


def rms_error(
    errors: list[np.ndarray],
) -> float | None:
    """Return RMS Euclidean error."""

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


def raw_planning_margin(
    *,
    position: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """Pass one frame-by-frame estimate through existing planner interface."""

    estimate = EstimatedStructure(
        estimated_centre=np.asarray(
            position,
            dtype=float,
        ),
        physical_radius=(
            PHYSICAL_RADIUS
        ),
        base_safety_margin=(
            BASE_SAFETY_MARGIN
        ),
        uncertainty=PositionUncertainty(
            covariance=np.asarray(
                covariance,
                dtype=float,
            )
        ),
    )

    perception = PerceptionResult(
        estimated_structures=(
            estimate,
        ),
        localisation_errors=np.empty(
            (
                0,
            ),
            dtype=float,
        ),
    )

    planning = (
        uncertainty_aware_planning_structures(
            perception_result=perception,
            sigma_multiplier=(
                PLANNING_SIGMA_MULTIPLIER
            ),
        )
    )

    return float(
        planning
        .structures[0]
        .safety_margin
    )


def run_trial(
    *,
    seed: int,
) -> dict:
    """Run one complete registration-to-navigation trial."""

    rng = np.random.default_rng(
        seed
    )

    true_transform = (
        make_true_registration_transform(
            rng
        )
    )

    (
        estimated_transform,
        registration_pose_covariance,
        registration_diagnostics,
    ) = estimate_registration(
        rng=rng,
        true_transform=true_transform,
        ransac_seed=(
            seed
            + 100000
        ),
    )

    navigation_to_perception = (
        invert_transform(
            true_transform
        )
    )

    tracker = (
        ConstantVelocityKalmanFilter(
            KalmanFilterConfig(
                acceleration_sigma=(
                    CALIBRATED_ACCELERATION_SIGMA
                ),
                initial_velocity_sigma=(
                    INITIAL_VELOCITY_SIGMA
                ),
                innovation_gate_threshold=(
                    INNOVATION_GATE_THRESHOLD
                ),
            )
        )
    )

    state = None

    raw_errors: list[np.ndarray] = []

    raw_clean_errors: list[np.ndarray] = []

    tracked_errors: list[np.ndarray] = []

    dropout_errors: list[np.ndarray] = []

    raw_coverage: list[bool] = []

    raw_clean_coverage: list[bool] = []

    tracked_coverage: list[bool] = []

    raw_margins: list[float] = []

    tracked_margins: list[float] = []

    tracked_measurement_margins: list[
        float
    ] = []

    tracked_dropout_margins: list[
        float
    ] = []

    rejection_truth: list[bool] = []

    rejection_prediction: list[bool] = []

    dropout_count = 0

    outlier_count = 0

    accepted_measurements = 0

    rejected_measurements = 0

    for index in range(
        STEPS
    ):
        timestamp = (
            index
            * DT
        )

        true_navigation_position, _ = (
            true_motion(
                timestamp
            )
        )

        true_perception_position = (
            transform_point(
                navigation_to_perception,
                true_navigation_position,
            )
        )

        dropout = bool(
            index > 0
            and rng.random()
            < DROPOUT_PROBABILITY
        )

        if dropout:
            dropout_count += 1

            if state is None:
                raise RuntimeError(
                    "Tracker cannot drop first observation."
                )

            state = tracker.predict(
                state,
                timestamp=timestamp,
            )

            tracked_error = (
                state.position
                - true_navigation_position
            )

            tracked_errors.append(
                tracked_error
            )

            dropout_errors.append(
                tracked_error
            )

            tracked_coverage.append(
                covariance_contains_truth(
                    error=tracked_error,
                    covariance=(
                        state
                        .position_covariance
                    ),
                )
            )

            perception = (
                tracking_state_to_perception_result(
                    state=state,
                    physical_radius=(
                        PHYSICAL_RADIUS
                    ),
                    base_safety_margin=(
                        BASE_SAFETY_MARGIN
                    ),
                )
            )

            planning = (
                uncertainty_aware_planning_structures(
                    perception_result=(
                        perception
                    ),
                    sigma_multiplier=(
                        PLANNING_SIGMA_MULTIPLIER
                    ),
                )
            )

            margin = float(
                planning
                .structures[0]
                .safety_margin
            )

            tracked_margins.append(
                margin
            )

            tracked_dropout_margins.append(
                margin
            )

            continue

        measurement_covariance = (
            make_measurement_covariance(
                rng=rng
            )
        )

        measured_perception_position = (
            true_perception_position
            + sample_gaussian_measurement_noise(
                rng=rng,
                covariance=(
                    measurement_covariance
                ),
            )
        )

        is_outlier = bool(
            index > 0
            and rng.random()
            < OUTLIER_PROBABILITY
        )

        if is_outlier:
            outlier_count += 1

            measured_perception_position = (
                measured_perception_position
                + random_outlier_offset(
                    rng
                )
            )

        stereo_observation = (
            SyntheticStereoObservation(
                triangulation=(
                    SyntheticTriangulation(
                        point_world=np.asarray(
                            measured_perception_position,
                            dtype=float,
                        )
                    )
                ),
                world_position_covariance=np.asarray(
                    measurement_covariance,
                    dtype=float,
                ),
            )
        )

        registered_raw = (
            register_position_measurement(
                position=(
                    measured_perception_position
                ),
                covariance=(
                    measurement_covariance
                ),
                perception_to_navigation_transform=(
                    estimated_transform
                ),
                registration_pose_covariance=(
                    registration_pose_covariance
                ),
            )
        )

        raw_error = (
            registered_raw.position
            - true_navigation_position
        )

        raw_errors.append(
            raw_error
        )

        raw_is_covered = (
            covariance_contains_truth(
                error=raw_error,
                covariance=(
                    registered_raw
                    .covariance
                ),
            )
        )

        raw_coverage.append(
            raw_is_covered
        )

        if not is_outlier:
            raw_clean_errors.append(
                raw_error
            )

            raw_clean_coverage.append(
                raw_is_covered
            )

        raw_margins.append(
            raw_planning_margin(
                position=(
                    registered_raw.position
                ),
                covariance=(
                    registered_raw.covariance
                ),
            )
        )

        integrated = (
            integrate_learned_stereo_observation(
                stereo_result=(
                    stereo_observation
                ),
                perception_to_navigation_transform=(
                    estimated_transform
                ),
                tracker=tracker,
                previous_state=state,
                timestamp=timestamp,
                physical_radius=(
                    PHYSICAL_RADIUS
                ),
                base_safety_margin=(
                    BASE_SAFETY_MARGIN
                ),
                sigma_multiplier=(
                    PLANNING_SIGMA_MULTIPLIER
                ),
                registration_pose_covariance=(
                    registration_pose_covariance
                ),
            )
        )

        state = (
            integrated.state
        )

        if index == 0:
            # First measurement initialises the tracker and is forced clean.
            accepted_measurements += 1

        else:
            rejection_truth.append(
                is_outlier
            )

            rejection_prediction.append(
                not integrated
                .measurement_accepted
            )

            if (
                integrated
                .measurement_accepted
            ):
                accepted_measurements += 1

            else:
                rejected_measurements += 1

        tracked_error = (
            state.position
            - true_navigation_position
        )

        tracked_errors.append(
            tracked_error
        )

        tracked_coverage.append(
            covariance_contains_truth(
                error=tracked_error,
                covariance=(
                    state
                    .position_covariance
                ),
            )
        )

        tracked_margin = float(
            integrated
            .planning_perception
            .structures[0]
            .safety_margin
        )

        tracked_margins.append(
            tracked_margin
        )

        tracked_measurement_margins.append(
            tracked_margin
        )

    truth = np.asarray(
        rejection_truth,
        dtype=bool,
    )

    prediction = np.asarray(
        rejection_prediction,
        dtype=bool,
    )

    true_positive = int(
        np.count_nonzero(
            truth
            & prediction
        )
    )

    false_positive = int(
        np.count_nonzero(
            ~truth
            & prediction
        )
    )

    false_negative = int(
        np.count_nonzero(
            truth
            & ~prediction
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

    raw_rmse = rms_error(
        raw_errors
    )

    clean_raw_rmse = rms_error(
        raw_clean_errors
    )

    tracked_rmse = rms_error(
        tracked_errors
    )

    dropout_rmse = rms_error(
        dropout_errors
    )

    if (
        raw_rmse is None
        or clean_raw_rmse is None
        or tracked_rmse is None
    ):
        raise RuntimeError(
            "Insufficient error samples."
        )

    return {
        "seed": int(seed),
        "registration": (
            registration_diagnostics
        ),
        "measurement_count": int(
            len(
                raw_errors
            )
        ),
        "dropout_count": int(
            dropout_count
        ),
        "outlier_count": int(
            outlier_count
        ),
        "accepted_measurements": int(
            accepted_measurements
        ),
        "rejected_measurements": int(
            rejected_measurements
        ),
        "raw_registered_position_rmse_metres": float(
            raw_rmse
        ),
        "raw_clean_registered_position_rmse_metres": float(
            clean_raw_rmse
        ),
        "tracked_position_rmse_metres": float(
            tracked_rmse
        ),
        "tracked_dropout_rmse_metres": (
            None
            if dropout_rmse is None
            else float(
                dropout_rmse
            )
        ),
        "raw_covariance_95_coverage": float(
            np.mean(
                raw_coverage
            )
        ),
        "raw_clean_covariance_95_coverage": float(
            np.mean(
                raw_clean_coverage
            )
        ),
        "tracked_covariance_95_coverage": float(
            np.mean(
                tracked_coverage
            )
        ),
        "outlier_rejection_precision": float(
            precision
        ),
        "outlier_rejection_recall": float(
            recall
        ),
        "mean_raw_planner_margin_metres": float(
            np.mean(
                raw_margins
            )
        ),
        "mean_tracked_planner_margin_metres": float(
            np.mean(
                tracked_margins
            )
        ),
        "mean_tracked_measurement_step_margin_metres": float(
            np.mean(
                tracked_measurement_margins
            )
        ),
        "mean_tracked_dropout_margin_metres": (
            None
            if not tracked_dropout_margins
            else float(
                np.mean(
                    tracked_dropout_margins
                )
            )
        ),
    }


def aggregate(
    records: list[dict],
    path: tuple[str, ...],
) -> dict:
    """Aggregate one scalar across repeated trials."""

    values = []

    for record in records:
        value = record

        for key in path:
            value = value[
                key
            ]

        if value is not None:
            values.append(
                float(value)
            )

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
    """Run final Phase 5 integration benchmark."""

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

    registration_translation = aggregate(
        records,
        (
            "registration",
            "translation_error_metres",
        ),
    )

    registration_rotation = aggregate(
        records,
        (
            "registration",
            "rotation_error_degrees",
        ),
    )

    raw_position = aggregate(
        records,
        (
            "raw_registered_position_rmse_metres",
        ),
    )

    clean_raw_position = aggregate(
        records,
        (
            "raw_clean_registered_position_rmse_metres",
        ),
    )

    tracked_position = aggregate(
        records,
        (
            "tracked_position_rmse_metres",
        ),
    )

    dropout_position = aggregate(
        records,
        (
            "tracked_dropout_rmse_metres",
        ),
    )

    raw_coverage = aggregate(
        records,
        (
            "raw_covariance_95_coverage",
        ),
    )

    clean_raw_coverage = aggregate(
        records,
        (
            "raw_clean_covariance_95_coverage",
        ),
    )

    tracked_coverage = aggregate(
        records,
        (
            "tracked_covariance_95_coverage",
        ),
    )

    rejection_precision = aggregate(
        records,
        (
            "outlier_rejection_precision",
        ),
    )

    rejection_recall = aggregate(
        records,
        (
            "outlier_rejection_recall",
        ),
    )

    raw_margin = aggregate(
        records,
        (
            "mean_raw_planner_margin_metres",
        ),
    )

    tracked_margin = aggregate(
        records,
        (
            "mean_tracked_planner_margin_metres",
        ),
    )

    dropout_margin = aggregate(
        records,
        (
            "mean_tracked_dropout_margin_metres",
        ),
    )

    tracked_better_fraction = float(
        np.mean(
            [
                record[
                    "tracked_position_rmse_metres"
                ]
                < record[
                    "raw_registered_position_rmse_metres"
                ]
                for record in records
            ]
        )
    )

    evidence = {
        "phase": 5,
        "experiment": (
            "final_registration_tracking_navigation_integration"
        ),
        "scope": "simulation_only",
        "trials": int(
            TRIALS
        ),
        "steps_per_trial": int(
            STEPS
        ),
        "calibrated_acceleration_sigma": float(
            CALIBRATED_ACCELERATION_SIGMA
        ),
        "registration_method": (
            "RANSAC corresponding-landmark registration"
        ),
        "registration_pose_uncertainty_method": (
            "Residual-derived approximate translational and "
            "small-angle rotational covariance; no ground truth is used "
            "to construct runtime registration covariance."
        ),
        "summary": {
            "registration_translation_error_metres": (
                registration_translation
            ),
            "registration_rotation_error_degrees": (
                registration_rotation
            ),
            "raw_registered_position_rmse_metres": (
                raw_position
            ),
            "raw_clean_registered_position_rmse_metres": (
                clean_raw_position
            ),
            "tracked_position_rmse_metres": (
                tracked_position
            ),
            "tracked_dropout_rmse_metres": (
                dropout_position
            ),
            "raw_covariance_95_coverage": (
                raw_coverage
            ),
            "raw_clean_covariance_95_coverage": (
                clean_raw_coverage
            ),
            "tracked_covariance_95_coverage": (
                tracked_coverage
            ),
            "outlier_rejection_precision": (
                rejection_precision
            ),
            "outlier_rejection_recall": (
                rejection_recall
            ),
            "mean_raw_planner_margin_metres": (
                raw_margin
            ),
            "mean_tracked_planner_margin_metres": (
                tracked_margin
            ),
            "mean_tracked_dropout_margin_metres": (
                dropout_margin
            ),
            "tracker_lower_position_rmse_fraction": float(
                tracked_better_fraction
            ),
        },
        "trial_records": records,
        "claims_note": (
            "The benchmark evaluates synthetic rigid registration, "
            "uncertain 3-D observations, temporal state estimation and "
            "planner-margin propagation. The stereo observation object "
            "matches the learned-stereo navigation interface but the "
            "measurements in this Phase 5 benchmark are synthetic. "
            "The actual learned image-to-stereo pathway is evaluated "
            "separately in Phase 4. Results do not establish clinical "
            "registration, tracking, navigation performance or safety."
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
        "Phase 5 Final Registration + Tracking + Navigation Benchmark"
    )

    print(
        "=" * 59
    )

    print(
        f"Trials: {TRIALS}"
    )

    print(
        f"Steps/trial: {STEPS}"
    )

    print(
        "Frozen acceleration sigma: "
        f"{CALIBRATED_ACCELERATION_SIGMA:.3f} m/s^2"
    )

    print()

    print(
        "REGISTRATION"
    )

    print(
        "Mean translation error: "
        f"{1000.0 * registration_translation['mean']:.3f} mm"
    )

    print(
        "Mean rotation error: "
        f"{registration_rotation['mean']:.3f} deg"
    )

    print()

    print(
        "POSITION ESTIMATION"
    )

    print(
        "Raw registered RMSE: "
        f"{1000.0 * raw_position['mean']:.3f} mm"
    )

    print(
        "Raw clean registered RMSE: "
        f"{1000.0 * clean_raw_position['mean']:.3f} mm"
    )

    print(
        "Tracked RMSE: "
        f"{1000.0 * tracked_position['mean']:.3f} mm"
    )

    print(
        "Tracked dropout RMSE: "
        f"{1000.0 * dropout_position['mean']:.3f} mm"
    )

    print(
        "Tracker lower RMSE: "
        f"{100.0 * tracked_better_fraction:.1f}% of trials"
    )

    print()

    print(
        "UNCERTAINTY"
    )

    print(
        "Raw all-measurement 95% coverage: "
        f"{100.0 * raw_coverage['mean']:.1f}%"
    )

    print(
        "Raw clean-measurement 95% coverage: "
        f"{100.0 * clean_raw_coverage['mean']:.1f}%"
    )

    print(
        "Tracked 95% coverage: "
        f"{100.0 * tracked_coverage['mean']:.1f}%"
    )

    print()

    print(
        "OUTLIER GATING"
    )

    print(
        "Precision: "
        f"{100.0 * rejection_precision['mean']:.1f}%"
    )

    print(
        "Recall: "
        f"{100.0 * rejection_recall['mean']:.1f}%"
    )

    print()

    print(
        "PLANNER UNCERTAINTY PROPAGATION"
    )

    print(
        "Mean raw safety margin: "
        f"{1000.0 * raw_margin['mean']:.3f} mm"
    )

    print(
        "Mean tracked safety margin: "
        f"{1000.0 * tracked_margin['mean']:.3f} mm"
    )

    print(
        "Mean tracked dropout safety margin: "
        f"{1000.0 * dropout_margin['mean']:.3f} mm"
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