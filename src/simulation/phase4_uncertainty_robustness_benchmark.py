"""Phase 4 learned-perception robustness and uncertainty benchmark.

A separately generated synthetic evaluation set is exposed to controlled
image degradation and colour-shift distribution shift.

For each condition the benchmark measures:

- Dice and IoU;
- target detection rate;
- centroid localisation error;
- Brier score;
- expected calibration error;
- predictive entropy;
- perturbation-ensemble probability variance;
- empirical centroid uncertainty.

The trained model is frozen. No retraining or model selection is performed
using this evaluation set.

These experiments use synthetic engineering imagery and do not constitute
clinical image or tissue-segmentation validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from src.perception.ml_dataset import (
    SyntheticDatasetConfig,
    generate_synthetic_segmentation_dataset,
)
from src.perception.ml_segmentation import (
    TinyUNet,
    binary_mask_centroid,
    binary_segmentation_metrics,
)
from src.perception.ml_uncertainty import (
    binary_brier_score,
    expected_calibration_error,
    predict_with_perturbation_ensemble,
)


MODEL_PATH = Path(
    "results/phase4/tiny_unet_best.pt"
)

OUTPUT_PATH = Path(
    "results/phase4/phase4_uncertainty_robustness.json"
)

ROBUSTNESS_SEED = 4811

SAMPLE_COUNT = 40

ENSEMBLE_SAMPLES = 10


def load_trained_model() -> TinyUNet:
    """Load the validation-selected Phase 4 U-Net."""

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


def make_robustness_dataset():
    """Generate scenarios independent of the training experiment."""

    config = SyntheticDatasetConfig(
        width=96,
        height=96,
        scenario_count=SAMPLE_COUNT,
        frames_per_scenario=1,
        train_fraction=0.70,
        validation_fraction=0.15,
        minimum_marker_radius=6,
        maximum_marker_radius=14,
        seed=ROBUSTNESS_SEED,
    )

    return (
        generate_synthetic_segmentation_dataset(
            config
        )
    )


def degrade_image(
    image: np.ndarray,
    *,
    condition: str,
    seed: int,
) -> np.ndarray:
    """Apply one deterministic robustness condition."""

    rng = np.random.default_rng(
        seed
    )

    image = np.asarray(
        image,
        dtype=np.uint8,
    )

    if condition == "clean":
        return image.copy()

    if condition == "mild":
        noise = rng.normal(
            0.0,
            8.0,
            size=image.shape,
        )

        return np.clip(
            image.astype(float)
            + noise,
            0.0,
            255.0,
        ).astype(
            np.uint8
        )

    if condition == "moderate":
        blurred = cv2.GaussianBlur(
            image,
            (3, 3),
            sigmaX=0.0,
        )

        noise = rng.normal(
            0.0,
            18.0,
            size=image.shape,
        )

        return np.clip(
            blurred.astype(float)
            * 0.82
            + noise,
            0.0,
            255.0,
        ).astype(
            np.uint8
        )

    if condition == "severe":
        blurred = cv2.GaussianBlur(
            image,
            (5, 5),
            sigmaX=0.0,
        )

        noise = rng.normal(
            0.0,
            32.0,
            size=image.shape,
        )

        return np.clip(
            blurred.astype(float)
            * 0.62
            + noise,
            0.0,
            255.0,
        ).astype(
            np.uint8
        )

    if condition == "colour_shift_ood":
        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV,
        )

        hsv = hsv.copy()

        hue = (
            hsv[
                :,
                :,
                0,
            ].astype(
                np.int16
            )
            + 45
        ) % 180

        hsv[
            :,
            :,
            0,
        ] = hue.astype(
            np.uint8
        )

        return cv2.cvtColor(
            hsv,
            cv2.COLOR_HSV2BGR,
        )

    raise ValueError(
        f"Unknown degradation condition: {condition}"
    )


def predict_probability(
    *,
    model: TinyUNet,
    image_bgr: np.ndarray,
) -> np.ndarray:
    """Run deterministic single-image segmentation inference."""

    rgb = (
        image_bgr[
            :,
            :,
            ::-1
        ]
        .copy()
    )

    tensor = (
        torch.from_numpy(
            rgb
        )
        .permute(
            2,
            0,
            1,
        )
        .float()
        .unsqueeze(
            0
        )
        / 255.0
    )

    with torch.no_grad():
        logits = model(
            tensor
        )

        probability = (
            torch.sigmoid(
                logits
            )[
                0,
                0,
            ]
            .cpu()
            .numpy()
        )

    return np.asarray(
        probability,
        dtype=float,
    )


def evaluate_condition(
    *,
    model: TinyUNet,
    dataset,
    condition: str,
    condition_index: int,
) -> dict:
    """Evaluate one degradation condition."""

    dice_values = []
    iou_values = []
    centroid_errors = []
    brier_values = []
    calibration_values = []
    entropy_values = []
    foreground_entropy_values = []
    variance_values = []
    centroid_sigma_values = []

    detections = 0

    for index in range(
        dataset.sample_count
    ):
        image = degrade_image(
            dataset.images[
                index
            ],
            condition=condition,
            seed=(
                ROBUSTNESS_SEED
                + 1000
                * condition_index
                + index
            ),
        )

        truth = (
            dataset.masks[
                index
            ]
            > 0
        )

        probability = (
            predict_probability(
                model=model,
                image_bgr=image,
            )
        )

        prediction = (
            probability
            >= 0.5
        )

        dice, iou, _accuracy = (
            binary_segmentation_metrics(
                prediction,
                truth,
            )
        )

        dice_values.append(
            dice
        )

        iou_values.append(
            iou
        )

        brier_values.append(
            binary_brier_score(
                probability,
                truth.astype(float),
            )
        )

        calibration_values.append(
            expected_calibration_error(
                probability,
                truth.astype(float),
                bin_count=15,
            )
        )

        centroid = (
            binary_mask_centroid(
                prediction
            )
        )

        if centroid is not None:
            detections += 1

            centroid_errors.append(
                float(
                    np.linalg.norm(
                        centroid
                        - dataset.marker_pixels[
                            index
                        ]
                    )
                )
            )

        ensemble = (
            predict_with_perturbation_ensemble(
                model=model,
                image_bgr=image,
                device=torch.device(
                    "cpu"
                ),
                sample_count=(
                    ENSEMBLE_SAMPLES
                ),
                threshold=0.5,
                noise_sigma=5.0,
                brightness_sigma=0.05,
                seed=(
                    ROBUSTNESS_SEED
                    + 100000
                    + 1000
                    * condition_index
                    + index
                ),
            )
        )

        entropy_values.append(
            ensemble.mean_entropy
        )

        foreground_entropy_values.append(
            ensemble.foreground_entropy
        )

        variance_values.append(
            float(
                np.mean(
                    ensemble
                    .probability_variance
                )
            )
        )

        if (
            ensemble.centroid_covariance
            is not None
        ):
            eigenvalues = (
                np.linalg.eigvalsh(
                    ensemble
                    .centroid_covariance
                )
            )

            centroid_sigma_values.append(
                float(
                    np.sqrt(
                        max(
                            float(
                                np.max(
                                    eigenvalues
                                )
                            ),
                            0.0,
                        )
                    )
                )
            )

    return {
        "condition": condition,
        "samples": int(
            dataset.sample_count
        ),
        "dice": float(
            np.mean(
                dice_values
            )
        ),
        "iou": float(
            np.mean(
                iou_values
            )
        ),
        "detection_rate": float(
            detections
            / dataset.sample_count
        ),
        "mean_centroid_error_pixels": (
            None
            if not centroid_errors
            else float(
                np.mean(
                    centroid_errors
                )
            )
        ),
        "brier_score": float(
            np.mean(
                brier_values
            )
        ),
        "expected_calibration_error": float(
            np.mean(
                calibration_values
            )
        ),
        "mean_predictive_entropy": float(
            np.mean(
                entropy_values
            )
        ),
        "mean_foreground_entropy": float(
            np.mean(
                foreground_entropy_values
            )
        ),
        "mean_probability_variance": float(
            np.mean(
                variance_values
            )
        ),
        "mean_centroid_principal_sigma_pixels": (
            None
            if not centroid_sigma_values
            else float(
                np.mean(
                    centroid_sigma_values
                )
            )
        ),
    }


def analyse_degradation_trends(
    records: list[dict],
) -> dict:
    """Summarise robustness behaviour without forcing a desired outcome."""

    lookup = {
        record[
            "condition"
        ]: record
        for record in records
    }

    ordered_names = [
        "clean",
        "mild",
        "moderate",
        "severe",
    ]

    ordered = [
        lookup[
            name
        ]
        for name in ordered_names
    ]

    dice = np.asarray(
        [
            item[
                "dice"
            ]
            for item in ordered
        ],
        dtype=float,
    )

    entropy = np.asarray(
        [
            item[
                "mean_foreground_entropy"
            ]
            for item in ordered
        ],
        dtype=float,
    )

    brier = np.asarray(
        [
            item[
                "brier_score"
            ]
            for item in ordered
        ],
        dtype=float,
    )

    severity = np.arange(
        len(
            ordered
        ),
        dtype=float,
    )

    return {
        "dice_clean_to_severe_change": float(
            dice[
                -1
            ]
            - dice[
                0
            ]
        ),
        "foreground_entropy_clean_to_severe_change": float(
            entropy[
                -1
            ]
            - entropy[
                0
            ]
        ),
        "brier_clean_to_severe_change": float(
            brier[
                -1
            ]
            - brier[
                0
            ]
        ),
        "severity_vs_dice_correlation": float(
            np.corrcoef(
                severity,
                dice,
            )[
                0,
                1
            ]
        ),
        "severity_vs_entropy_correlation": float(
            np.corrcoef(
                severity,
                entropy,
            )[
                0,
                1
            ]
        ),
        "colour_shift_ood": (
            lookup[
                "colour_shift_ood"
            ]
        ),
        "interpretation_note": (
            "Uncertainty is not required to increase monotonically. "
            "Failure to increase under performance degradation would "
            "indicate overconfident behaviour under distribution shift."
        ),
    }


def run_benchmark() -> dict:
    """Run the complete Phase 4 robustness benchmark."""

    model = (
        load_trained_model()
    )

    dataset = (
        make_robustness_dataset()
    )

    conditions = (
        "clean",
        "mild",
        "moderate",
        "severe",
        "colour_shift_ood",
    )

    records = []

    print(
        "Phase 4 Learned Perception Robustness Benchmark"
    )

    print(
        "="
        * 47
    )

    print(
        f"Independent scenarios: {dataset.scenario_count}"
    )

    print(
        f"Ensemble samples/image: {ENSEMBLE_SAMPLES}"
    )

    print()

    for index, condition in enumerate(
        conditions
    ):
        result = (
            evaluate_condition(
                model=model,
                dataset=dataset,
                condition=condition,
                condition_index=index,
            )
        )

        records.append(
            result
        )

        print(
            condition
        )

        print(
            f"  Dice: {result['dice']:.4f}"
        )

        print(
            f"  IoU: {result['iou']:.4f}"
        )

        print(
            "  Detection: "
            f"{100.0 * result['detection_rate']:.1f}%"
        )

        print(
            "  Centroid error: "
            f"{result['mean_centroid_error_pixels']} px"
        )

        print(
            f"  Brier: {result['brier_score']:.5f}"
        )

        print(
            "  ECE: "
            f"{result['expected_calibration_error']:.5f}"
        )

        print(
            "  Foreground entropy: "
            f"{result['mean_foreground_entropy']:.5f}"
        )

        print(
            "  Mean probability variance: "
            f"{result['mean_probability_variance']:.8f}"
        )

        print()

    trends = (
        analyse_degradation_trends(
            records
        )
    )

    evidence = {
        "phase": 4,
        "experiment": (
            "learned_perception_uncertainty_and_robustness"
        ),
        "scope": (
            "independent_synthetic_engineering_evaluation"
        ),
        "robustness_seed": int(
            ROBUSTNESS_SEED
        ),
        "scenario_count": int(
            dataset.scenario_count
        ),
        "ensemble_samples_per_image": int(
            ENSEMBLE_SAMPLES
        ),
        "conditions": records,
        "trend_analysis": trends,
        "claims_note": (
            "This experiment evaluates synthetic segmentation robustness "
            "and perturbation-based predictive uncertainty. It does not "
            "establish clinical calibration, anatomical generalisation "
            "or medical-device performance."
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
        "Trend summary"
    )

    print(
        "-------------"
    )

    print(
        "Dice clean -> severe: "
        f"{trends['dice_clean_to_severe_change']:+.4f}"
    )

    print(
        "Entropy clean -> severe: "
        f"{trends['foreground_entropy_clean_to_severe_change']:+.5f}"
    )

    print(
        "Brier clean -> severe: "
        f"{trends['brier_clean_to_severe_change']:+.5f}"
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