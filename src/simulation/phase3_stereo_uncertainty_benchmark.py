"""Final Phase 3 stereo-perception and uncertainty benchmark.

This benchmark evaluates the Phase 3 image-driven perception pipeline.

It contains two complementary experiments:

1. Image-driven demonstration
   Synthetic stereo images -> OpenCV segmentation -> stereo triangulation ->
   EstimatedStructure -> PositionUncertainty -> uncertainty-aware margin.

2. Monte-Carlo stereo uncertainty experiment
   Controlled pixel measurement noise is applied to calibrated stereo
   correspondences while camera baseline is varied. Actual 3-D localisation
   errors are compared with first-order propagated covariance.

Simulator ground truth is used only for evaluation.

The experiment does not constitute clinical imaging validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.perception.camera import CameraPose
from src.perception.image_driven_perception import (
    evaluate_image_perception_error,
    perceive_structure_from_stereo_images,
)
from src.perception.image_geometry import (
    PixelCameraIntrinsics,
    project_world_point,
)
from src.perception.image_processing import (
    render_synthetic_marker_frame,
)
from src.perception.stereo_geometry import (
    triangulate_world_point_from_pixels,
)
from src.perception.stereo_uncertainty import (
    propagate_stereo_pixel_uncertainty,
)
from src.perception.uncertainty import (
    inflate_estimated_structures,
)


OUTPUT_PATH = Path(
    "results/phase3/phase3_stereo_uncertainty_benchmark.json"
)

BASELINES_METRES = (
    0.01,
    0.02,
    0.04,
)

PIXEL_SIGMAS = (
    0.25,
    0.50,
    1.00,
)

REPETITIONS = 500

RANDOM_SEED = 3107

CHI_SQUARE_3D_95 = 7.814727903251179


def make_intrinsics() -> PixelCameraIntrinsics:
    """Return calibrated synthetic camera intrinsics."""

    return PixelCameraIntrinsics(
        width=320,
        height=240,
        fx=250.0,
        fy=250.0,
        cx=159.5,
        cy=119.5,
    )


def left_pose() -> CameraPose:
    """Return left stereo-camera pose."""

    return CameraPose(
        position=np.zeros(
            3,
            dtype=float,
        ),
        rotation=np.eye(
            3,
            dtype=float,
        ),
    )


def right_pose(
    baseline: float,
) -> CameraPose:
    """Return rectified right-camera pose."""

    return CameraPose(
        position=np.asarray(
            [
                baseline,
                0.0,
                0.0,
            ],
            dtype=float,
        ),
        rotation=np.eye(
            3,
            dtype=float,
        ),
    )


def target_position() -> np.ndarray:
    """Return hidden simulation target used only for rendering/evaluation."""

    return np.asarray(
        [
            0.020,
            -0.010,
            0.250,
        ],
        dtype=float,
    )


def _serialise_matrix(
    matrix: np.ndarray,
) -> list[list[float]]:
    """Convert a matrix into JSON-safe floats."""

    return [
        [
            float(
                value
            )
            for value in row
        ]
        for row in np.asarray(
            matrix,
            dtype=float,
        )
    ]


def run_image_driven_demonstration() -> dict:
    """Run one complete image -> uncertainty-aware estimate demonstration."""

    intrinsics = (
        make_intrinsics()
    )

    target = (
        target_position()
    )

    baseline = 0.02

    left = left_pose()
    right = right_pose(
        baseline
    )

    left_projection = (
        project_world_point(
            intrinsics,
            left,
            target,
        )
    )

    right_projection = (
        project_world_point(
            intrinsics,
            right,
            target,
        )
    )

    left_image = (
        render_synthetic_marker_frame(
            width=intrinsics.width,
            height=intrinsics.height,
            marker_pixel=(
                left_projection.pixel
            ),
            marker_radius=14,
            seed=1001,
        )
    )

    right_image = (
        render_synthetic_marker_frame(
            width=intrinsics.width,
            height=intrinsics.height,
            marker_pixel=(
                right_projection.pixel
            ),
            marker_radius=14,
            seed=1002,
        )
    )

    result = (
        perceive_structure_from_stereo_images(
            left_image=left_image,
            right_image=right_image,
            intrinsics=intrinsics,
            left_pose=left,
            right_pose=right,
            physical_radius=0.025,
            base_safety_margin=0.015,
            pixel_sigma=0.50,
        )
    )

    estimate = (
        result
        .perception_result
        .estimated_structures[
            0
        ]
    )

    localisation_error = (
        evaluate_image_perception_error(
            result,
            target,
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

    return {
        "baseline_metres": baseline,
        "pixel_sigma": 0.50,
        "true_position_metres": [
            float(value)
            for value in target
        ],
        "estimated_position_metres": [
            float(value)
            for value in (
                estimate.estimated_centre
            )
        ],
        "localisation_error_metres": float(
            localisation_error
        ),
        "principal_sigma_metres": float(
            estimate
            .uncertainty
            .principal_sigma
        ),
        "position_covariance_metres_squared": (
            _serialise_matrix(
                estimate
                .uncertainty
                .covariance
            )
        ),
        "base_safety_margin_metres": float(
            estimate.base_safety_margin
        ),
        "two_sigma_inflated_margin_metres": float(
            inflated[
                0
            ].safety_margin
        ),
        "runtime_ground_truth_error_count": int(
            result
            .perception_result
            .localisation_errors
            .size
        ),
        "left_detected": bool(
            result
            .left_segmentation
            .detection
            .found
        ),
        "right_detected": bool(
            result
            .right_segmentation
            .detection
            .found
        ),
        "closest_ray_gap_metres": float(
            result
            .triangulation
            .closest_ray_gap
        ),
    }


def run_uncertainty_condition(
    *,
    baseline: float,
    pixel_sigma: float,
    seed: int,
) -> dict:
    """Run one Monte-Carlo stereo uncertainty condition."""

    intrinsics = (
        make_intrinsics()
    )

    left = left_pose()

    right = right_pose(
        baseline
    )

    truth = (
        target_position()
    )

    left_exact = (
        project_world_point(
            intrinsics,
            left,
            truth,
        )
        .pixel
    )

    right_exact = (
        project_world_point(
            intrinsics,
            right,
            truth,
        )
        .pixel
    )

    exact_disparity = float(
        left_exact[
            0
        ]
        - right_exact[
            0
        ]
    )

    rng = np.random.default_rng(
        seed
    )

    position_errors: list[
        np.ndarray
    ] = []

    predicted_covariances: list[
        np.ndarray
    ] = []

    predicted_principal_sigmas: list[
        float
    ] = []

    mahalanobis_squared: list[
        float
    ] = []

    failures = 0

    for _ in range(
        REPETITIONS
    ):
        measurement_noise = (
            rng.normal(
                loc=0.0,
                scale=pixel_sigma,
                size=4,
            )
        )

        measured_left = (
            left_exact
            + measurement_noise[
                :2
            ]
        )

        measured_right = (
            right_exact
            + measurement_noise[
                2:
            ]
        )

        try:
            triangulation = (
                triangulate_world_point_from_pixels(
                    intrinsics=intrinsics,
                    left_pose=left,
                    right_pose=right,
                    left_pixel=(
                        measured_left
                    ),
                    right_pixel=(
                        measured_right
                    ),
                )
            )

            uncertainty = (
                propagate_stereo_pixel_uncertainty(
                    intrinsics=intrinsics,
                    left_pose=left,
                    right_pose=right,
                    left_pixel=(
                        measured_left
                    ),
                    right_pixel=(
                        measured_right
                    ),
                    pixel_sigma=(
                        pixel_sigma
                    ),
                )
            )

        except ValueError:
            failures += 1
            continue

        error = (
            triangulation.point_world
            - truth
        )

        covariance = np.asarray(
            uncertainty.covariance,
            dtype=float,
        )

        inverse_covariance = (
            np.linalg.pinv(
                covariance,
                hermitian=True,
            )
        )

        mahalanobis = float(
            error.T
            @ inverse_covariance
            @ error
        )

        position_errors.append(
            np.asarray(
                error,
                dtype=float,
            )
        )

        predicted_covariances.append(
            covariance
        )

        predicted_principal_sigmas.append(
            float(
                uncertainty
                .principal_sigma
            )
        )

        mahalanobis_squared.append(
            mahalanobis
        )

    if len(
        position_errors
    ) < 2:
        raise RuntimeError(
            "Insufficient successful stereo reconstructions."
        )

    errors = np.vstack(
        position_errors
    )

    covariance_stack = np.stack(
        predicted_covariances,
        axis=0,
    )

    error_norms = np.linalg.norm(
        errors,
        axis=1,
    )

    empirical_covariance = (
        np.cov(
            errors,
            rowvar=False,
            ddof=1,
        )
    )

    mean_predicted_covariance = (
        np.mean(
            covariance_stack,
            axis=0,
        )
    )

    empirical_eigenvalues = (
        np.linalg.eigvalsh(
            empirical_covariance
        )
    )

    predicted_eigenvalues = (
        np.linalg.eigvalsh(
            mean_predicted_covariance
        )
    )

    empirical_principal_sigma = float(
        np.sqrt(
            max(
                float(
                    np.max(
                        empirical_eigenvalues
                    )
                ),
                0.0,
            )
        )
    )

    mean_covariance_principal_sigma = float(
        np.sqrt(
            max(
                float(
                    np.max(
                        predicted_eigenvalues
                    )
                ),
                0.0,
            )
        )
    )

    actual_rms_position_error = float(
        np.sqrt(
            np.mean(
                np.sum(
                    errors**2,
                    axis=1,
                )
            )
        )
    )

    predicted_rms_position_sigma = float(
        np.sqrt(
            max(
                float(
                    np.trace(
                        mean_predicted_covariance
                    )
                ),
                0.0,
            )
        )
    )

    coverage_95 = float(
        np.mean(
            np.asarray(
                mahalanobis_squared,
                dtype=float,
            )
            <= CHI_SQUARE_3D_95
        )
    )

    covariance_relative_error = float(
        np.linalg.norm(
            empirical_covariance
            - mean_predicted_covariance
        )
        / max(
            np.linalg.norm(
                empirical_covariance
            ),
            1e-15,
        )
    )

    bias = np.mean(
        errors,
        axis=0,
    )

    return {
        "baseline_metres": float(
            baseline
        ),
        "pixel_sigma": float(
            pixel_sigma
        ),
        "exact_disparity_pixels": float(
            exact_disparity
        ),
        "attempted_repetitions": int(
            REPETITIONS
        ),
        "successful_reconstructions": int(
            len(
                errors
            )
        ),
        "failed_reconstructions": int(
            failures
        ),
        "failure_fraction": float(
            failures
            / REPETITIONS
        ),
        "mean_error_metres": float(
            np.mean(
                error_norms
            )
        ),
        "median_error_metres": float(
            np.median(
                error_norms
            )
        ),
        "p95_error_metres": float(
            np.quantile(
                error_norms,
                0.95,
            )
        ),
        "rms_error_metres": (
            actual_rms_position_error
        ),
        "bias_vector_metres": [
            float(value)
            for value in bias
        ],
        "bias_norm_metres": float(
            np.linalg.norm(
                bias
            )
        ),
        "empirical_covariance_metres_squared": (
            _serialise_matrix(
                empirical_covariance
            )
        ),
        "mean_predicted_covariance_metres_squared": (
            _serialise_matrix(
                mean_predicted_covariance
            )
        ),
        "empirical_principal_sigma_metres": (
            empirical_principal_sigma
        ),
        "mean_predicted_principal_sigma_metres": float(
            np.mean(
                predicted_principal_sigmas
            )
        ),
        "principal_sigma_from_mean_covariance_metres": (
            mean_covariance_principal_sigma
        ),
        "predicted_rms_position_sigma_metres": (
            predicted_rms_position_sigma
        ),
        "actual_to_predicted_rms_ratio": float(
            actual_rms_position_error
            / max(
                predicted_rms_position_sigma,
                1e-15,
            )
        ),
        "mahalanobis_95_coverage": (
            coverage_95
        ),
        "covariance_relative_frobenius_error": (
            covariance_relative_error
        ),
    }


def run_monte_carlo_experiment() -> list[dict]:
    """Evaluate all baseline and pixel-noise conditions."""

    records: list[
        dict
    ] = []

    condition_index = 0

    for baseline in (
        BASELINES_METRES
    ):
        for pixel_sigma in (
            PIXEL_SIGMAS
        ):
            records.append(
                run_uncertainty_condition(
                    baseline=baseline,
                    pixel_sigma=(
                        pixel_sigma
                    ),
                    seed=(
                        RANDOM_SEED
                        + condition_index
                    ),
                )
            )

            condition_index += 1

    return records


def evaluate_expected_trends(
    records: list[dict],
) -> dict:
    """Evaluate expected qualitative stereo-uncertainty trends."""

    noise_trends: dict[
        str,
        bool
    ] = {}

    for baseline in (
        BASELINES_METRES
    ):
        matching = sorted(
            [
                record
                for record in records
                if np.isclose(
                    record[
                        "baseline_metres"
                    ],
                    baseline,
                )
            ],
            key=lambda item:
            item[
                "pixel_sigma"
            ],
        )

        sigmas = [
            record[
                "mean_predicted_principal_sigma_metres"
            ]
            for record in matching
        ]

        noise_trends[
            f"{baseline:.3f}_m_baseline"
        ] = bool(
            all(
                later
                > earlier
                for earlier, later
                in zip(
                    sigmas[
                        :-1
                    ],
                    sigmas[
                        1:
                    ],
                )
            )
        )

    baseline_trends: dict[
        str,
        bool
    ] = {}

    for pixel_sigma in (
        PIXEL_SIGMAS
    ):
        matching = sorted(
            [
                record
                for record in records
                if np.isclose(
                    record[
                        "pixel_sigma"
                    ],
                    pixel_sigma,
                )
            ],
            key=lambda item:
            item[
                "baseline_metres"
            ],
        )

        sigmas = [
            record[
                "mean_predicted_principal_sigma_metres"
            ]
            for record in matching
        ]

        baseline_trends[
            f"{pixel_sigma:.2f}_pixel_sigma"
        ] = bool(
            all(
                later
                < earlier
                for earlier, later
                in zip(
                    sigmas[
                        :-1
                    ],
                    sigmas[
                        1:
                    ],
                )
            )
        )

    return {
        "uncertainty_increases_with_pixel_noise": (
            noise_trends
        ),
        "uncertainty_decreases_with_larger_baseline": (
            baseline_trends
        ),
        "all_expected_noise_trends_hold": bool(
            all(
                noise_trends.values()
            )
        ),
        "all_expected_baseline_trends_hold": bool(
            all(
                baseline_trends.values()
            )
        ),
    }


def run_benchmark() -> dict:
    """Run final Phase 3 evidence generation."""

    image_demonstration = (
        run_image_driven_demonstration()
    )

    monte_carlo = (
        run_monte_carlo_experiment()
    )

    trends = (
        evaluate_expected_trends(
            monte_carlo
        )
    )

    return {
        "phase": 3,
        "benchmark": (
            "image_driven_stereo_perception_and_uncertainty"
        ),
        "scope": (
            "simulation_only"
        ),
        "claims_note": (
            "Results validate synthetic image processing, calibrated "
            "stereo geometry and first-order uncertainty propagation. "
            "They do not constitute clinical imaging, anatomical "
            "segmentation or physical camera validation."
        ),
        "monte_carlo_repetitions_per_condition": int(
            REPETITIONS
        ),
        "baseline_values_metres": [
            float(value)
            for value in (
                BASELINES_METRES
            )
        ],
        "pixel_sigma_values": [
            float(value)
            for value in (
                PIXEL_SIGMAS
            )
        ],
        "image_driven_demonstration": (
            image_demonstration
        ),
        "monte_carlo_conditions": (
            monte_carlo
        ),
        "expected_trends": trends,
    }


def write_results(
    evidence: dict,
) -> None:
    """Write evidence to the Phase 3 results directory."""

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


def print_summary(
    evidence: dict,
) -> None:
    """Print concise Phase 3 benchmark results."""

    print(
        "Phase 3 Stereo Perception + Uncertainty Benchmark"
    )

    print(
        "="
        * 49
    )

    image = evidence[
        "image_driven_demonstration"
    ]

    print(
        "Image-driven stereo demonstration"
    )

    print(
        "  Localisation error: "
        f"{1000.0 * image['localisation_error_metres']:.3f} mm"
    )

    print(
        "  Principal sigma: "
        f"{1000.0 * image['principal_sigma_metres']:.3f} mm"
    )

    print(
        "  Runtime truth-error entries: "
        f"{image['runtime_ground_truth_error_count']}"
    )

    print()

    print(
        "Monte-Carlo conditions"
    )

    for record in evidence[
        "monte_carlo_conditions"
    ]:
        print(
            "  "
            f"baseline={1000.0 * record['baseline_metres']:.0f} mm, "
            f"pixel sigma={record['pixel_sigma']:.2f} px"
        )

        print(
            "    RMS error: "
            f"{1000.0 * record['rms_error_metres']:.3f} mm"
        )

        print(
            "    predicted RMS sigma: "
            f"{1000.0 * record['predicted_rms_position_sigma_metres']:.3f} mm"
        )

        print(
            "    RMS ratio actual/predicted: "
            f"{record['actual_to_predicted_rms_ratio']:.3f}"
        )

        print(
            "    95% covariance coverage: "
            f"{100.0 * record['mahalanobis_95_coverage']:.1f}%"
        )

    print()

    trends = evidence[
        "expected_trends"
    ]

    print(
        "Uncertainty increases with pixel noise: "
        f"{trends['all_expected_noise_trends_hold']}"
    )

    print(
        "Uncertainty decreases with larger baseline: "
        f"{trends['all_expected_baseline_trends_hold']}"
    )

    print()

    print(
        "Evidence written to:"
    )

    print(
        OUTPUT_PATH
    )


def main() -> None:
    """Execute Phase 3 benchmark."""

    evidence = (
        run_benchmark()
    )

    write_results(
        evidence
    )

    print_summary(
        evidence
    )


if __name__ == "__main__":
    main()