"""Phase 4 learned-segmentation training and held-out evaluation.

Protocol
--------
1. Generate synthetic scenarios.
2. Split by latent scenario before training.
3. Train Tiny U-Net using TRAIN only.
4. Select the best epoch using VALIDATION Dice only.
5. Evaluate the selected model once on TEST.
6. Compare against the frozen classical HSV/OpenCV baseline on the same
   untouched test images.

This is synthetic engineering validation, not clinical segmentation
validation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.perception.image_processing import (
    SegmentationConfig,
    segment_target_bgr,
)
from src.perception.ml_dataset import (
    SyntheticDatasetConfig,
    SyntheticSegmentationDataset,
    build_phase4_dataset,
)
from src.perception.ml_segmentation import (
    SegmentationMetrics,
    TinyUNet,
    TorchSegmentationDataset,
    binary_mask_centroid,
    binary_segmentation_metrics,
    combined_segmentation_loss,
    evaluate_segmentation_model,
)


RESULT_PATH = Path(
    "results/phase4/phase4_segmentation_training.json"
)

MODEL_PATH = Path(
    "results/phase4/tiny_unet_best.pt"
)

RANDOM_SEED = 4401

EPOCHS = 12

BATCH_SIZE = 16

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5


def make_dataset_config() -> SyntheticDatasetConfig:
    """Return the frozen Phase 4 training dataset configuration."""

    return SyntheticDatasetConfig(
        width=96,
        height=96,
        scenario_count=100,
        frames_per_scenario=2,
        train_fraction=0.70,
        validation_fraction=0.15,
        minimum_marker_radius=6,
        maximum_marker_radius=14,
        seed=RANDOM_SEED,
    )


def set_deterministic_seeds(
    seed: int,
) -> None:
    """Seed Python, NumPy and PyTorch."""

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )


def train_one_epoch(
    *,
    model: TinyUNet,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one optimisation epoch."""

    model.train()

    losses: list[
        float
    ] = []

    for images, masks in loader:
        images = images.to(
            device
        )

        masks = masks.to(
            device
        )

        optimiser.zero_grad(
            set_to_none=True
        )

        logits = model(
            images
        )

        loss = (
            combined_segmentation_loss(
                logits,
                masks,
                positive_weight=8.0,
            )
        )

        loss.backward()

        optimiser.step()

        losses.append(
            float(
                loss.detach().cpu()
            )
        )

    return float(
        np.mean(
            losses
        )
    )


def classical_segmentation_metrics(
    dataset: SyntheticSegmentationDataset,
) -> SegmentationMetrics:
    """Evaluate fixed classical OpenCV segmentation on a dataset."""

    dice_values: list[
        float
    ] = []

    iou_values: list[
        float
    ] = []

    accuracy_values: list[
        float
    ] = []

    centroid_errors: list[
        float
    ] = []

    detections = 0

    config = SegmentationConfig()

    for index in range(
        dataset.sample_count
    ):
        result = segment_target_bgr(
            dataset.images[
                index
            ],
            config=config,
        )

        prediction = (
            result.cleaned_mask
            > 0
        )

        truth = (
            dataset.masks[
                index
            ]
            > 0
        )

        (
            dice,
            iou,
            accuracy,
        ) = binary_segmentation_metrics(
            prediction,
            truth,
        )

        dice_values.append(
            dice
        )

        iou_values.append(
            iou
        )

        accuracy_values.append(
            accuracy
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

    if centroid_errors:
        mean_centroid_error = float(
            np.mean(
                centroid_errors
            )
        )

    else:
        mean_centroid_error = float(
            "inf"
        )

    return SegmentationMetrics(
        dice=float(
            np.mean(
                dice_values
            )
        ),
        iou=float(
            np.mean(
                iou_values
            )
        ),
        pixel_accuracy=float(
            np.mean(
                accuracy_values
            )
        ),
        detection_rate=float(
            detections
            / dataset.sample_count
        ),
        mean_centroid_error_pixels=(
            mean_centroid_error
        ),
    )


def metrics_to_dict(
    metrics: SegmentationMetrics,
) -> dict:
    """Convert metrics to JSON-safe values."""

    centroid_error = (
        metrics.mean_centroid_error_pixels
    )

    return {
        "dice": float(
            metrics.dice
        ),
        "iou": float(
            metrics.iou
        ),
        "pixel_accuracy": float(
            metrics.pixel_accuracy
        ),
        "detection_rate": float(
            metrics.detection_rate
        ),
        "mean_centroid_error_pixels": (
            None
            if not np.isfinite(
                centroid_error
            )
            else float(
                centroid_error
            )
        ),
    }


def run_training() -> dict:
    """Train, validate and perform one held-out test evaluation."""

    set_deterministic_seeds(
        RANDOM_SEED
    )

    device = torch.device(
        "cpu"
    )

    dataset_config = (
        make_dataset_config()
    )

    splits = (
        build_phase4_dataset(
            dataset_config
        )
    )

    train_scenarios = set(
        splits.train
        .scenario_ids
        .tolist()
    )

    validation_scenarios = set(
        splits.validation
        .scenario_ids
        .tolist()
    )

    test_scenarios = set(
        splits.test
        .scenario_ids
        .tolist()
    )

    if not (
        train_scenarios.isdisjoint(
            validation_scenarios
        )
        and train_scenarios.isdisjoint(
            test_scenarios
        )
        and validation_scenarios.isdisjoint(
            test_scenarios
        )
    ):
        raise RuntimeError(
            "Scenario leakage detected between dataset splits."
        )

    generator = torch.Generator()

    generator.manual_seed(
        RANDOM_SEED
    )

    train_loader = DataLoader(
        TorchSegmentationDataset(
            splits.train
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    model = TinyUNet(
        base_channels=8
    ).to(
        device
    )

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_validation_dice = float(
        "-inf"
    )

    best_epoch = -1

    best_state = None

    history: list[
        dict
    ] = []

    print(
        "Phase 4 Tiny U-Net Training"
    )

    print(
        "="
        * 32
    )

    print(
        f"Train samples: {splits.train.sample_count}"
    )

    print(
        f"Validation samples: {splits.validation.sample_count}"
    )

    print(
        f"Test samples: {splits.test.sample_count}"
    )

    print(
        f"Device: {device}"
    )

    print()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        training_loss = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                optimiser=optimiser,
                device=device,
            )
        )

        validation = (
            evaluate_segmentation_model(
                model=model,
                dataset=(
                    splits.validation
                ),
                device=device,
                batch_size=BATCH_SIZE,
                threshold=0.5,
            )
        )

        history.append(
            {
                "epoch": int(
                    epoch
                ),
                "training_loss": float(
                    training_loss
                ),
                "validation": (
                    metrics_to_dict(
                        validation
                    )
                ),
            }
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS}: "
            f"loss={training_loss:.4f}, "
            f"val Dice={validation.dice:.4f}, "
            f"val IoU={validation.iou:.4f}, "
            f"centroid={validation.mean_centroid_error_pixels:.3f}px"
        )

        if (
            validation.dice
            > best_validation_dice
        ):
            best_validation_dice = float(
                validation.dice
            )

            best_epoch = int(
                epoch
            )

            best_state = deepcopy(
                model.state_dict()
            )

    if best_state is None:
        raise RuntimeError(
            "No valid model checkpoint was produced."
        )

    model.load_state_dict(
        best_state
    )

    print()

    print(
        f"Best epoch selected using validation only: {best_epoch}"
    )

    print(
        f"Best validation Dice: {best_validation_dice:.4f}"
    )

    print()

    # ---------------------------------------------------------------
    # Untouched test set is evaluated only after model selection.
    # ---------------------------------------------------------------

    learned_test = (
        evaluate_segmentation_model(
            model=model,
            dataset=splits.test,
            device=device,
            batch_size=BATCH_SIZE,
            threshold=0.5,
        )
    )

    classical_test = (
        classical_segmentation_metrics(
            splits.test
        )
    )

    print(
        "Held-out TEST results"
    )

    print(
        "---------------------"
    )

    print(
        "Tiny U-Net"
    )

    print(
        f"  Dice: {learned_test.dice:.4f}"
    )

    print(
        f"  IoU: {learned_test.iou:.4f}"
    )

    print(
        f"  Detection: {100.0 * learned_test.detection_rate:.1f}%"
    )

    print(
        "  Mean centroid error: "
        f"{learned_test.mean_centroid_error_pixels:.3f} px"
    )

    print()

    print(
        "Classical HSV/OpenCV"
    )

    print(
        f"  Dice: {classical_test.dice:.4f}"
    )

    print(
        f"  IoU: {classical_test.iou:.4f}"
    )

    print(
        f"  Detection: {100.0 * classical_test.detection_rate:.1f}%"
    )

    print(
        "  Mean centroid error: "
        f"{classical_test.mean_centroid_error_pixels:.3f} px"
    )

    model_better_dice = bool(
        learned_test.dice
        > classical_test.dice
    )

    result = {
        "phase": 4,
        "experiment": (
            "tiny_unet_vs_classical_segmentation"
        ),
        "scope": (
            "synthetic_engineering_images_only"
        ),
        "claims_note": (
            "Results quantify segmentation of synthetic navigation markers. "
            "They do not constitute clinical tissue segmentation validation."
        ),
        "random_seed": int(
            RANDOM_SEED
        ),
        "device": str(
            device
        ),
        "epochs": int(
            EPOCHS
        ),
        "batch_size": int(
            BATCH_SIZE
        ),
        "learning_rate": float(
            LEARNING_RATE
        ),
        "weight_decay": float(
            WEIGHT_DECAY
        ),
        "dataset_config": asdict(
            dataset_config
        ),
        "split_summary": {
            "train_samples": int(
                splits.train.sample_count
            ),
            "validation_samples": int(
                splits.validation.sample_count
            ),
            "test_samples": int(
                splits.test.sample_count
            ),
            "train_scenarios": int(
                splits.train.scenario_count
            ),
            "validation_scenarios": int(
                splits.validation.scenario_count
            ),
            "test_scenarios": int(
                splits.test.scenario_count
            ),
            "scenario_leakage": False,
        },
        "best_epoch": int(
            best_epoch
        ),
        "best_validation_dice": float(
            best_validation_dice
        ),
        "training_history": history,
        "held_out_test": {
            "tiny_unet": (
                metrics_to_dict(
                    learned_test
                )
            ),
            "classical_hsv_opencv": (
                metrics_to_dict(
                    classical_test
                )
            ),
            "tiny_unet_higher_dice": (
                model_better_dice
            ),
            "dice_difference_unet_minus_classical": float(
                learned_test.dice
                - classical_test.dice
            ),
        },
    }

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_PATH.write_text(
        json.dumps(
            result,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    torch.save(
        {
            "model_state_dict": (
                best_state
            ),
            "base_channels": 8,
            "threshold": 0.5,
            "dataset_config": asdict(
                dataset_config
            ),
            "best_epoch": int(
                best_epoch
            ),
        },
        MODEL_PATH,
    )

    print()

    print(
        "Evidence written to:"
    )

    print(
        RESULT_PATH
    )

    print(
        "Best model written to:"
    )

    print(
        MODEL_PATH
    )

    return result


def main() -> None:
    run_training()


if __name__ == "__main__":
    main()