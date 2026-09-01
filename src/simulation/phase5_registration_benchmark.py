"""Phase 5 robust registration and ICP benchmark.

This benchmark evaluates two registration problems.

Experiment A
------------
Corresponding fiducial landmarks are available, but some target
correspondences are gross outliers.

    naive Kabsch
        vs
    RANSAC + Kabsch refinement

Experiment B
------------
Point-cloud correspondence is unknown.

    trimmed iterative closest point

Metrics include:

- translation error;
- rotation-angle error;
- fiducial registration error (FRE);
- target registration error (TRE) on independent evaluation points;
- RANSAC inlier recovery;
- ICP residual and convergence.

Ground-truth transforms are used only for evaluation.

This is simulation-only engineering validation and does not constitute
clinical image-to-patient registration validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.geometry.registration import (
    register_corresponding_points,
    registration_error_at_targets,
    transform_error,
    transform_points,
)
from src.geometry.robust_registration import (
    iterative_closest_point,
    ransac_rigid_registration,
)
from src.geometry.transforms import (
    make_transform,
)


OUTPUT_PATH = Path(
    "results/phase5/phase5_registration_benchmark.json"
)

RANDOM_SEED = 5501

ROBUST_TRIALS = 20

ICP_TRIALS = 15


def rotation_xyz(
    ax: float,
    ay: float,
    az: float,
) -> np.ndarray:
    """Construct a proper XYZ rotation."""

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


def random_reference_transform(
    rng: np.random.Generator,
    *,
    maximum_rotation_degrees: float,
    maximum_translation_metres: float,
) -> np.ndarray:
    """Generate one random rigid transform."""

    angles = np.deg2rad(
        rng.uniform(
            -maximum_rotation_degrees,
            maximum_rotation_degrees,
            size=3,
        )
    )

    translation = rng.uniform(
        -maximum_translation_metres,
        maximum_translation_metres,
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


def make_landmarks(
    *,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate an asymmetric 3-D landmark configuration."""

    points = rng.uniform(
        low=np.asarray(
            [
                -0.055,
                -0.045,
                -0.035,
            ],
            dtype=float,
        ),
        high=np.asarray(
            [
                0.060,
                0.050,
                0.050,
            ],
            dtype=float,
        ),
        size=(
            count,
            3,
        ),
    )

    points[:, 2] += (
        0.18 * points[:, 0]
        - 0.09 * points[:, 1]
    )

    return np.asarray(
        points,
        dtype=float,
    )


def make_evaluation_targets(
    *,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate independent source-frame TRE evaluation points."""

    return rng.uniform(
        low=np.asarray(
            [
                -0.070,
                -0.060,
                -0.045,
            ],
            dtype=float,
        ),
        high=np.asarray(
            [
                0.075,
                0.065,
                0.070,
            ],
            dtype=float,
        ),
        size=(
            count,
            3,
        ),
    )


def transform_metrics(
    *,
    estimated_transform: np.ndarray,
    reference_transform: np.ndarray,
    evaluation_targets: np.ndarray,
) -> dict:
    """Return transform and TRE metrics."""

    error = transform_error(
        estimated_transform,
        reference_transform,
    )

    tre = registration_error_at_targets(
        estimated_transform,
        reference_transform,
        evaluation_targets,
    )

    return {
        "translation_error_metres": float(
            error.translation_error
        ),
        "rotation_error_degrees": float(
            error.rotation_angle_error_degrees
        ),
        "mean_tre_metres": float(
            np.mean(tre)
        ),
        "rms_tre_metres": float(
            np.sqrt(
                np.mean(
                    tre**2
                )
            )
        ),
        "maximum_tre_metres": float(
            np.max(tre)
        ),
    }


def run_robust_trial(
    *,
    seed: int,
) -> dict:
    """Compare naive and RANSAC registration under correspondence outliers."""

    rng = np.random.default_rng(
        seed
    )

    source = make_landmarks(
        count=20,
        rng=rng,
    )

    reference = random_reference_transform(
        rng,
        maximum_rotation_degrees=18.0,
        maximum_translation_metres=0.025,
    )

    clean_target = transform_points(
        reference,
        source,
    )

    noisy_target = (
        clean_target
        + rng.normal(
            loc=0.0,
            scale=0.0006,
            size=clean_target.shape,
        )
    )

    outlier_count = 4

    outlier_indices = rng.choice(
        source.shape[0],
        size=outlier_count,
        replace=False,
    )

    contaminated_target = (
        noisy_target.copy()
    )

    contaminated_target[
        outlier_indices
    ] += rng.uniform(
        low=-0.060,
        high=0.060,
        size=(
            outlier_count,
            3,
        ),
    )

    evaluation_targets = (
        make_evaluation_targets(
            count=12,
            rng=rng,
        )
    )

    naive = (
        register_corresponding_points(
            source,
            contaminated_target,
        )
    )

    robust = (
        ransac_rigid_registration(
            source,
            contaminated_target,
            iterations=400,
            inlier_threshold=0.003,
            seed=seed + 10000,
        )
    )

    naive_metrics = (
        transform_metrics(
            estimated_transform=(
                naive.transform
            ),
            reference_transform=(
                reference
            ),
            evaluation_targets=(
                evaluation_targets
            ),
        )
    )

    robust_metrics = (
        transform_metrics(
            estimated_transform=(
                robust.registration.transform
            ),
            reference_transform=(
                reference
            ),
            evaluation_targets=(
                evaluation_targets
            ),
        )
    )

    true_inlier_mask = np.ones(
        source.shape[0],
        dtype=bool,
    )

    true_inlier_mask[
        outlier_indices
    ] = False

    detected_inlier_mask = (
        robust.inlier_mask
    )

    true_positive = int(
        np.count_nonzero(
            detected_inlier_mask
            & true_inlier_mask
        )
    )

    false_positive = int(
        np.count_nonzero(
            detected_inlier_mask
            & ~true_inlier_mask
        )
    )

    false_negative = int(
        np.count_nonzero(
            ~detected_inlier_mask
            & true_inlier_mask
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

    return {
        "seed": int(seed),
        "landmark_count": int(
            source.shape[0]
        ),
        "outlier_count": int(
            outlier_count
        ),
        "outlier_fraction": float(
            outlier_count
            / source.shape[0]
        ),
        "naive": {
            **naive_metrics,
            "fre_rms_metres": float(
                naive.fre_rms
            ),
        },
        "ransac": {
            **robust_metrics,
            "fre_rms_inliers_metres": float(
                robust.registration.fre_rms
            ),
            "detected_inlier_count": int(
                robust.inlier_count
            ),
            "detected_inlier_fraction": float(
                robust.inlier_fraction
            ),
            "inlier_precision": precision,
            "inlier_recall": recall,
        },
    }


def run_icp_trial(
    *,
    seed: int,
) -> dict:
    """Evaluate ICP with unknown point correspondence."""

    rng = np.random.default_rng(
        seed
    )

    source = make_landmarks(
        count=60,
        rng=rng,
    )

    reference = random_reference_transform(
        rng,
        maximum_rotation_degrees=7.0,
        maximum_translation_metres=0.008,
    )

    target = transform_points(
        reference,
        source,
    )

    target += rng.normal(
        loc=0.0,
        scale=0.00025,
        size=target.shape,
    )

    target = target[
        rng.permutation(
            target.shape[0]
        )
    ]

    # Add target clutter that has no direct source correspondence.
    target_clutter = rng.uniform(
        low=-0.090,
        high=0.090,
        size=(
            5,
            3,
        ),
    )

    target_with_clutter = np.vstack(
        (
            target,
            target_clutter,
        )
    )

    evaluation_targets = (
        make_evaluation_targets(
            count=12,
            rng=rng,
        )
    )

    result = (
        iterative_closest_point(
            source,
            target_with_clutter,
            max_iterations=80,
            tolerance=1e-9,
            trim_fraction=0.90,
            maximum_correspondence_distance=0.030,
        )
    )

    metrics = transform_metrics(
        estimated_transform=(
            result.transform
        ),
        reference_transform=(
            reference
        ),
        evaluation_targets=(
            evaluation_targets
        ),
    )

    return {
        "seed": int(seed),
        "source_points": int(
            source.shape[0]
        ),
        "target_points": int(
            target_with_clutter.shape[0]
        ),
        "target_clutter_points": int(
            target_clutter.shape[0]
        ),
        **metrics,
        "icp_rms_metres": float(
            result.rms_error
        ),
        "iterations": int(
            result.iterations
        ),
        "converged": bool(
            result.converged
        ),
        "used_correspondence_count": int(
            result.used_correspondence_count
        ),
    }


def aggregate_metric(
    records: list[dict],
    *,
    path: tuple[str, ...],
) -> dict:
    """Aggregate one scalar metric across records."""

    values = []

    for record in records:
        value = record

        for key in path:
            value = value[
                key
            ]

        values.append(
            float(value)
        )

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
        "maximum": float(
            np.max(array)
        ),
        "minimum": float(
            np.min(array)
        ),
    }


def run_benchmark() -> dict:
    """Run complete Phase 5 registration benchmark."""

    robust_trials = [
        run_robust_trial(
            seed=(
                RANDOM_SEED
                + index
            )
        )
        for index in range(
            ROBUST_TRIALS
        )
    ]

    icp_trials = [
        run_icp_trial(
            seed=(
                RANDOM_SEED
                + 1000
                + index
            )
        )
        for index in range(
            ICP_TRIALS
        )
    ]

    naive_translation = aggregate_metric(
        robust_trials,
        path=(
            "naive",
            "translation_error_metres",
        ),
    )

    ransac_translation = aggregate_metric(
        robust_trials,
        path=(
            "ransac",
            "translation_error_metres",
        ),
    )

    naive_rotation = aggregate_metric(
        robust_trials,
        path=(
            "naive",
            "rotation_error_degrees",
        ),
    )

    ransac_rotation = aggregate_metric(
        robust_trials,
        path=(
            "ransac",
            "rotation_error_degrees",
        ),
    )

    naive_tre = aggregate_metric(
        robust_trials,
        path=(
            "naive",
            "rms_tre_metres",
        ),
    )

    ransac_tre = aggregate_metric(
        robust_trials,
        path=(
            "ransac",
            "rms_tre_metres",
        ),
    )

    inlier_precision = (
        aggregate_metric(
            robust_trials,
            path=(
                "ransac",
                "inlier_precision",
            ),
        )
    )

    inlier_recall = (
        aggregate_metric(
            robust_trials,
            path=(
                "ransac",
                "inlier_recall",
            ),
        )
    )

    icp_translation = (
        aggregate_metric(
            icp_trials,
            path=(
                "translation_error_metres",
            ),
        )
    )

    icp_rotation = (
        aggregate_metric(
            icp_trials,
            path=(
                "rotation_error_degrees",
            ),
        )
    )

    icp_tre = aggregate_metric(
        icp_trials,
        path=(
            "rms_tre_metres",
        ),
    )

    icp_rms = aggregate_metric(
        icp_trials,
        path=(
            "icp_rms_metres",
        ),
    )

    icp_convergence_rate = float(
        np.mean(
            [
                trial[
                    "converged"
                ]
                for trial in icp_trials
            ]
        )
    )

    ransac_better_tre_fraction = float(
        np.mean(
            [
                trial[
                    "ransac"
                ][
                    "rms_tre_metres"
                ]
                < trial[
                    "naive"
                ][
                    "rms_tre_metres"
                ]
                for trial in robust_trials
            ]
        )
    )

    evidence = {
        "phase": 5,
        "experiment": (
            "robust_landmark_registration_and_icp"
        ),
        "scope": (
            "simulation_only"
        ),
        "random_seed": int(
            RANDOM_SEED
        ),
        "robust_registration_trials": int(
            ROBUST_TRIALS
        ),
        "icp_trials": int(
            ICP_TRIALS
        ),
        "robust_registration_summary": {
            "naive_translation_error_metres": (
                naive_translation
            ),
            "ransac_translation_error_metres": (
                ransac_translation
            ),
            "naive_rotation_error_degrees": (
                naive_rotation
            ),
            "ransac_rotation_error_degrees": (
                ransac_rotation
            ),
            "naive_rms_tre_metres": (
                naive_tre
            ),
            "ransac_rms_tre_metres": (
                ransac_tre
            ),
            "ransac_inlier_precision": (
                inlier_precision
            ),
            "ransac_inlier_recall": (
                inlier_recall
            ),
            "ransac_lower_tre_fraction": (
                ransac_better_tre_fraction
            ),
        },
        "icp_summary": {
            "translation_error_metres": (
                icp_translation
            ),
            "rotation_error_degrees": (
                icp_rotation
            ),
            "rms_tre_metres": (
                icp_tre
            ),
            "icp_rms_metres": (
                icp_rms
            ),
            "convergence_rate": (
                icp_convergence_rate
            ),
        },
        "robust_registration_trial_records": (
            robust_trials
        ),
        "icp_trial_records": (
            icp_trials
        ),
        "claims_note": (
            "The experiment evaluates synthetic rigid registration. "
            "RANSAC assumes row-wise correspondences containing outliers; "
            "ICP evaluates unknown correspondence from a nearby initial "
            "alignment. Results do not establish clinical registration "
            "accuracy."
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
        "Phase 5 Robust Registration + ICP Benchmark"
    )

    print(
        "="
        * 43
    )

    print()

    print(
        f"RANSAC trials: {ROBUST_TRIALS}"
    )

    print(
        "Naive mean translation error: "
        f"{1000.0 * naive_translation['mean']:.3f} mm"
    )

    print(
        "RANSAC mean translation error: "
        f"{1000.0 * ransac_translation['mean']:.3f} mm"
    )

    print(
        "Naive mean rotation error: "
        f"{naive_rotation['mean']:.3f} deg"
    )

    print(
        "RANSAC mean rotation error: "
        f"{ransac_rotation['mean']:.3f} deg"
    )

    print(
        "Naive mean RMS TRE: "
        f"{1000.0 * naive_tre['mean']:.3f} mm"
    )

    print(
        "RANSAC mean RMS TRE: "
        f"{1000.0 * ransac_tre['mean']:.3f} mm"
    )

    print(
        "RANSAC mean inlier precision: "
        f"{100.0 * inlier_precision['mean']:.1f}%"
    )

    print(
        "RANSAC mean inlier recall: "
        f"{100.0 * inlier_recall['mean']:.1f}%"
    )

    print(
        "RANSAC lower TRE than naive: "
        f"{100.0 * ransac_better_tre_fraction:.1f}% of trials"
    )

    print()

    print(
        f"ICP trials: {ICP_TRIALS}"
    )

    print(
        "ICP mean translation error: "
        f"{1000.0 * icp_translation['mean']:.3f} mm"
    )

    print(
        "ICP mean rotation error: "
        f"{icp_rotation['mean']:.3f} deg"
    )

    print(
        "ICP mean RMS TRE: "
        f"{1000.0 * icp_tre['mean']:.3f} mm"
    )

    print(
        "ICP mean residual RMS: "
        f"{1000.0 * icp_rms['mean']:.3f} mm"
    )

    print(
        "ICP convergence rate: "
        f"{100.0 * icp_convergence_rate:.1f}%"
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