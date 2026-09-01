"""Final Phase 4 learned-perception integration benchmark.

The validation-selected Tiny U-Net is used without retraining.

Pipeline:

    synthetic stereo images
        -> learned segmentation
        -> perturbation ensemble
        -> 2-D centroid covariance
        -> stereo triangulation
        -> 3-D world covariance
        -> PositionUncertainty
        -> EstimatedStructure
        -> uncertainty-inflated planning geometry

The benchmark evaluates several hidden simulated target positions under:

- clean imagery;
- moderate degradation;
- colour-shift OOD.

Simulator truth is used only after perception for evaluation.

The experiment is simulation-only and does not establish clinical,
anatomical or physical-camera performance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.perception.camera import CameraPose
from src.perception.image_geometry import (
    PixelCameraIntrinsics,
    project_world_point,
)
from src.perception.ml_dataset import (
    _render_scenario_frame,
)
from src.perception.ml_segmentation import (
    TinyUNet,
)
from src.perception.ml_stereo_perception import (
    LearnedStereoPerceptionFailure,
    perceive_structure_from_learned_stereo,
)
from src.perception.uncertainty import (
    inflate_estimated_structures,
)
from src.simulation.phase4_uncertainty_robustness_benchmark import (
    degrade_image,
)


MODEL_PATH = Path(
    "results/phase4/tiny_unet_best.pt"
)

OUTPUT_PATH = Path(
    "results/phase4/phase4_final_integration_benchmark.json"
)

RANDOM_SEED = 4981

BASELINE_METRES = 0.020

ENSEMBLE_SAMPLES = 10

MINIMUM_PIXEL_SIGMA = 0.50

SIGMA_MULTIPLIER = 2.0

CHI_SQUARE_3D_95 = 7.814727903251179


def make_intrinsics() -> PixelCameraIntrinsics:
    """Return synthetic stereo intrinsics."""

    return PixelCameraIntrinsics(
        width=96,
        height=96,
        fx=180.0,
        fy=180.0,
        cx=47.5,
        cy=47.5,
    )


def left_pose() -> CameraPose:
    """Return left camera pose."""

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


def right_pose() -> CameraPose:
    """Return right camera pose."""

    return CameraPose(
        position=np.asarray(
            [
                BASELINE_METRES,
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


def target_positions() -> tuple[
    np.ndarray,
    ...
]:
    """Return hidden target positions spanning the stereo workspace."""

    return (
        np.asarray(
            [
                -0.010,
                -0.008,
                0.220,
            ],
            dtype=float,
        ),
        np.asarray(
            [
                0.000,
                0.008,
                0.220,
            ],
            dtype=float,
        ),
        np.asarray(
            [
                0.012,
                -0.006,
                0.250,
            ],
            dtype=float,
        ),
        np.asarray(
            [
                -0.012,
                0.005,
                0.250,
            ],
            dtype=float,
        ),
        np.asarray(
            [
                0.005,
                0.000,
                0.280,
            ],
            dtype=float,
        ),
        np.asarray(
            [
                0.015,
                0.009,
                0.280,
            ],
            dtype=float,
        ),
    )


def load_model() -> TinyUNet:
    """Load the validation-selected trained model."""

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model = TinyUNet(
        base_channels=int(
            checkpoint[
                "base_channels"
            ]
        )
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    return model


def render_stereo_pair(
    *,
    target: np.ndarray,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Render a paired stereo observation of one target."""

    intrinsics = (
        make_intrinsics()
    )

    left_pixel = (
        project_world_point(
            intrinsics,
            left_pose(),
            target,
        )
        .pixel
    )

    right_pixel = (
        project_world_point(
            intrinsics,
            right_pose(),
            target,
        )
        .pixel
    )

    background = np.asarray(
        [
            65,
            72,
            135,
        ],
        dtype=np.uint8,
    )

    left_image, _left_mask = (
        _render_scenario_frame(
            width=96,
            height=96,
            marker_pixel=left_pixel,
            marker_radius=9,
            background_bgr=background,
            illumination_scale=1.0,
            frame_seed=seed,
        )
    )

    right_image, _right_mask = (
        _render_scenario_frame(
            width=96,
            height=96,
            marker_pixel=right_pixel,
            marker_radius=9,
            background_bgr=background,
            illumination_scale=1.0,
            frame_seed=seed + 1,
        )
    )

    return (
        left_image,
        right_image,
    )


def apply_condition(
    image: np.ndarray,
    *,
    condition: str,
    seed: int,
) -> np.ndarray:
    """Apply one final integration condition."""

    if condition == "clean":
        return image.copy()

    return degrade_image(
        image,
        condition=condition,
        seed=seed,
    )


def run_case(
    *,
    model: TinyUNet,
    target: np.ndarray,
    condition: str,
    case_index: int,
) -> dict:
    """Run one complete learned stereo-navigation perception case."""

    seed = (
        RANDOM_SEED
        + 100
        * case_index
    )

    left_image, right_image = (
        render_stereo_pair(
            target=target,
            seed=seed,
        )
    )

    left_image = (
        apply_condition(
            left_image,
            condition=condition,
            seed=seed + 10,
        )
    )

    right_image = (
        apply_condition(
            right_image,
            condition=condition,
            seed=seed + 20,
        )
    )

    try:
        result = (
            perceive_structure_from_learned_stereo(
                model=model,
                left_image=left_image,
                right_image=right_image,
                intrinsics=(
                    make_intrinsics()
                ),
                left_pose=(
                    left_pose()
                ),
                right_pose=(
                    right_pose()
                ),
                physical_radius=0.025,
                base_safety_margin=0.015,
                device=torch.device(
                    "cpu"
                ),
                ensemble_samples=(
                    ENSEMBLE_SAMPLES
                ),
                minimum_pixel_sigma=(
                    MINIMUM_PIXEL_SIGMA
                ),
                noise_sigma=4.0,
                brightness_sigma=0.04,
                seed=seed + 30,
            )
        )

    except (
        LearnedStereoPerceptionFailure,
        ValueError,
    ) as error:
        return {
            "success": False,
            "condition": condition,
            "target_position_metres": [
                float(
                    value
                )
                for value in target
            ],
            "failure_reason": str(
                error
            ),
        }

    estimate = (
        result
        .perception_result
        .estimated_structures[
            0
        ]
    )

    error_vector = (
        estimate.estimated_centre
        - target
    )

    localisation_error = float(
        np.linalg.norm(
            error_vector
        )
    )

    covariance = np.asarray(
        estimate
        .uncertainty
        .covariance,
        dtype=float,
    )

    inverse_covariance = (
        np.linalg.pinv(
            covariance,
            hermitian=True,
        )
    )

    mahalanobis_squared = float(
        error_vector.T
        @ inverse_covariance
        @ error_vector
    )

    inflated = (
        inflate_estimated_structures(
            estimated_structures=(
                estimate,
            ),
            sigma_multiplier=(
                SIGMA_MULTIPLIER
            ),
        )
    )

    return {
        "success": True,
        "condition": condition,
        "target_position_metres": [
            float(
                value
            )
            for value in target
        ],
        "estimated_position_metres": [
            float(
                value
            )
            for value in (
                estimate
                .estimated_centre
            )
        ],
        "localisation_error_metres": (
            localisation_error
        ),
        "principal_sigma_metres": float(
            estimate
            .uncertainty
            .principal_sigma
        ),
        "mahalanobis_squared": (
            mahalanobis_squared
        ),
        "inside_95_percent_covariance_ellipsoid": bool(
            mahalanobis_squared
            <= CHI_SQUARE_3D_95
        ),
        "left_centroid_success_rate": float(
            result
            .left_prediction
            .centroid_success_rate
        ),
        "right_centroid_success_rate": float(
            result
            .right_prediction
            .centroid_success_rate
        ),
        "left_foreground_entropy": float(
            result
            .left_prediction
            .foreground_entropy
        ),
        "right_foreground_entropy": float(
            result
            .right_prediction
            .foreground_entropy
        ),
        "base_safety_margin_metres": float(
            estimate.base_safety_margin
        ),
        "two_sigma_planner_margin_metres": float(
            inflated[
                0
            ]
            .safety_margin
        ),
        "runtime_ground_truth_error_entries": int(
            result
            .perception_result
            .localisation_errors
            .size
        ),
        "world_covariance_metres_squared": (
            np.asarray(
                covariance,
                dtype=float,
            )
            .tolist()
        ),
    }


def summarise_condition(
    records: list[
        dict
    ],
    condition: str,
) -> dict:
    """Aggregate final integration results for one condition."""

    matching = [
        record
        for record in records
        if record[
            "condition"
        ] == condition
    ]

    successful = [
        record
        for record in matching
        if record[
            "success"
        ]
    ]

    success_rate = float(
        len(
            successful
        )
        / len(
            matching
        )
    )

    if not successful:
        return {
            "condition": condition,
            "attempts": int(
                len(
                    matching
                )
            ),
            "successful": 0,
            "success_rate": (
                success_rate
            ),
            "mean_localisation_error_metres": None,
            "mean_principal_sigma_metres": None,
            "covariance_95_coverage": None,
            "mean_two_sigma_planner_margin_metres": None,
        }

    localisation_errors = np.asarray(
        [
            record[
                "localisation_error_metres"
            ]
            for record in successful
        ],
        dtype=float,
    )

    principal_sigmas = np.asarray(
        [
            record[
                "principal_sigma_metres"
            ]
            for record in successful
        ],
        dtype=float,
    )

    coverage = np.asarray(
        [
            record[
                "inside_95_percent_covariance_ellipsoid"
            ]
            for record in successful
        ],
        dtype=float,
    )

    planner_margins = np.asarray(
        [
            record[
                "two_sigma_planner_margin_metres"
            ]
            for record in successful
        ],
        dtype=float,
    )

    return {
        "condition": condition,
        "attempts": int(
            len(
                matching
            )
        ),
        "successful": int(
            len(
                successful
            )
        ),
        "success_rate": (
            success_rate
        ),
        "mean_localisation_error_metres": float(
            np.mean(
                localisation_errors
            )
        ),
        "maximum_localisation_error_metres": float(
            np.max(
                localisation_errors
            )
        ),
        "mean_principal_sigma_metres": float(
            np.mean(
                principal_sigmas
            )
        ),
        "covariance_95_coverage": float(
            np.mean(
                coverage
            )
        ),
        "mean_two_sigma_planner_margin_metres": float(
            np.mean(
                planner_margins
            )
        ),
    }


def run_benchmark() -> dict:
    """Run the final Phase 4 integration experiment."""

    model = (
        load_model()
    )

    targets = (
        target_positions()
    )

    conditions = (
        "clean",
        "moderate",
        "colour_shift_ood",
    )

    records: list[
        dict
    ] = []

    print(
        "Phase 4 Final Learned Stereo Integration Benchmark"
    )

    print(
        "="
        * 52
    )

    print(
        f"Targets: {len(targets)}"
    )

    print(
        f"Conditions: {len(conditions)}"
    )

    print(
        f"Stereo baseline: {1000.0 * BASELINE_METRES:.1f} mm"
    )

    print(
        f"Ensemble samples/view: {ENSEMBLE_SAMPLES}"
    )

    print()

    case_index = 0

    for condition in conditions:
        for target in targets:
            record = (
                run_case(
                    model=model,
                    target=target,
                    condition=condition,
                    case_index=case_index,
                )
            )

            records.append(
                record
            )

            case_index += 1

    summaries = [
        summarise_condition(
            records,
            condition,
        )
        for condition in conditions
    ]

    for summary in summaries:
        print(
            summary[
                "condition"
            ]
        )

        print(
            "  Success: "
            f"{100.0 * summary['success_rate']:.1f}%"
        )

        if (
            summary[
                "mean_localisation_error_metres"
            ]
            is not None
        ):
            print(
                "  Mean localisation error: "
                f"{1000.0 * summary['mean_localisation_error_metres']:.3f} mm"
            )

            print(
                "  Max localisation error: "
                f"{1000.0 * summary['maximum_localisation_error_metres']:.3f} mm"
            )

            print(
                "  Mean principal sigma: "
                f"{1000.0 * summary['mean_principal_sigma_metres']:.3f} mm"
            )

            print(
                "  95% covariance coverage: "
                f"{100.0 * summary['covariance_95_coverage']:.1f}%"
            )

            print(
                "  Mean 2-sigma planner margin: "
                f"{1000.0 * summary['mean_two_sigma_planner_margin_metres']:.3f} mm"
            )

        print()

    clean_summary = summaries[
        0
    ]

    ood_summary = summaries[
        2
    ]

    evidence = {
        "phase": 4,
        "experiment": (
            "final_learned_stereo_navigation_integration"
        ),
        "scope": (
            "simulation_only"
        ),
        "trained_checkpoint": str(
            MODEL_PATH
        ),
        "stereo_baseline_metres": float(
            BASELINE_METRES
        ),
        "ensemble_samples_per_view": int(
            ENSEMBLE_SAMPLES
        ),
        "minimum_pixel_sigma": float(
            MINIMUM_PIXEL_SIGMA
        ),
        "sigma_multiplier": float(
            SIGMA_MULTIPLIER
        ),
        "target_count": int(
            len(
                targets
            )
        ),
        "conditions": list(
            conditions
        ),
        "condition_summaries": (
            summaries
        ),
        "cases": records,
        "integration_checks": {
            "clean_runtime_truth_separation": bool(
                all(
                    (
                        not record[
                            "success"
                        ]
                    )
                    or (
                        record[
                            "runtime_ground_truth_error_entries"
                        ]
                        == 0
                    )
                    for record in records
                    if record[
                        "condition"
                    ]
                    == "clean"
                )
            ),
            "clean_pipeline_successful": bool(
                clean_summary[
                    "success_rate"
                ]
                > 0.0
            ),
            "ood_pipeline_success_rate": float(
                ood_summary[
                    "success_rate"
                ]
            ),
        },
        "claims_note": (
            "The benchmark demonstrates learned 2-D perception feeding "
            "stereo 3-D localisation, covariance propagation and the "
            "existing uncertainty-aware navigation representation. "
            "OOD failures and covariance miscalibration are retained as "
            "limitations rather than interpreted as clinical safety."
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