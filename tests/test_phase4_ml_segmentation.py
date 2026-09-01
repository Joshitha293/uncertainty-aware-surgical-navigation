"""Tests for Phase 4 learned segmentation components."""

import numpy as np
import pytest
import torch

from src.perception.ml_dataset import (
    SyntheticDatasetConfig,
    generate_synthetic_segmentation_dataset,
)
from src.perception.ml_segmentation import (
    TinyUNet,
    TorchSegmentationDataset,
    binary_mask_centroid,
    binary_segmentation_metrics,
    combined_segmentation_loss,
    evaluate_segmentation_model,
    soft_dice_loss,
)


def make_dataset():
    return (
        generate_synthetic_segmentation_dataset(
            SyntheticDatasetConfig(
                width=64,
                height=64,
                scenario_count=4,
                frames_per_scenario=1,
                seed=91,
            )
        )
    )


def test_unet_preserves_spatial_resolution():
    model = TinyUNet(
        base_channels=4
    )

    x = torch.zeros(
        (
            2,
            3,
            64,
            64,
        )
    )

    output = model(
        x
    )

    assert output.shape == (
        2,
        1,
        64,
        64,
    )


def test_unet_output_is_finite():
    model = TinyUNet(
        base_channels=4
    )

    x = torch.rand(
        (
            1,
            3,
            64,
            64,
        )
    )

    output = model(
        x
    )

    assert torch.all(
        torch.isfinite(
            output
        )
    )


def test_invalid_unet_input_channels_are_rejected():
    model = TinyUNet(
        base_channels=4
    )

    with pytest.raises(
        ValueError,
        match="shape",
    ):
        model(
            torch.zeros(
                (
                    1,
                    1,
                    64,
                    64,
                )
            )
        )


def test_torch_dataset_returns_normalised_image_and_mask():
    dataset = (
        TorchSegmentationDataset(
            make_dataset()
        )
    )

    image, mask = dataset[
        0
    ]

    assert image.shape == (
        3,
        64,
        64,
    )

    assert mask.shape == (
        1,
        64,
        64,
    )

    assert (
        image.dtype
        == torch.float32
    )

    assert (
        mask.dtype
        == torch.float32
    )

    assert (
        torch.min(
            image
        )
        >= 0.0
    )

    assert (
        torch.max(
            image
        )
        <= 1.0
    )


def test_perfect_binary_prediction_has_unit_metrics():
    mask = np.zeros(
        (
            20,
            20,
        ),
        dtype=bool,
    )

    mask[
        5:10,
        7:12
    ] = True

    dice, iou, accuracy = (
        binary_segmentation_metrics(
            mask,
            mask,
        )
    )

    assert dice == pytest.approx(
        1.0
    )

    assert iou == pytest.approx(
        1.0
    )

    assert accuracy == pytest.approx(
        1.0
    )


def test_disjoint_masks_have_zero_dice_and_iou():
    first = np.zeros(
        (
            20,
            20,
        ),
        dtype=bool,
    )

    second = np.zeros_like(
        first
    )

    first[
        2:5,
        2:5
    ] = True

    second[
        10:13,
        10:13
    ] = True

    dice, iou, _accuracy = (
        binary_segmentation_metrics(
            first,
            second,
        )
    )

    assert dice == pytest.approx(
        0.0
    )

    assert iou == pytest.approx(
        0.0
    )


def test_binary_centroid_is_correct():
    mask = np.zeros(
        (
            10,
            10,
        ),
        dtype=bool,
    )

    mask[
        4,
        6
    ] = True

    centroid = (
        binary_mask_centroid(
            mask
        )
    )

    np.testing.assert_allclose(
        centroid,
        np.asarray(
            [
                6.0,
                4.0,
            ]
        ),
        atol=1e-12,
    )


def test_empty_mask_has_no_centroid():
    centroid = (
        binary_mask_centroid(
            np.zeros(
                (
                    10,
                    10,
                ),
                dtype=bool,
            )
        )
    )

    assert centroid is None


def test_soft_dice_loss_is_finite():
    logits = torch.zeros(
        (
            2,
            1,
            16,
            16,
        )
    )

    targets = torch.zeros_like(
        logits
    )

    loss = soft_dice_loss(
        logits,
        targets,
    )

    assert torch.isfinite(
        loss
    )


def test_combined_loss_is_positive_and_finite():
    logits = torch.zeros(
        (
            2,
            1,
            16,
            16,
        )
    )

    targets = torch.zeros_like(
        logits
    )

    targets[
        :,
        :,
        5:10,
        5:10,
    ] = 1.0

    loss = (
        combined_segmentation_loss(
            logits,
            targets,
        )
    )

    assert torch.isfinite(
        loss
    )

    assert float(
        loss
    ) > 0.0


def test_model_evaluation_returns_finite_metrics():
    dataset = (
        make_dataset()
    )

    model = TinyUNet(
        base_channels=4
    )

    metrics = (
        evaluate_segmentation_model(
            model=model,
            dataset=dataset,
            device=torch.device(
                "cpu"
            ),
            batch_size=2,
        )
    )

    assert np.isfinite(
        metrics.dice
    )

    assert np.isfinite(
        metrics.iou
    )

    assert np.isfinite(
        metrics.pixel_accuracy
    )

    assert (
        0.0
        <= metrics.detection_rate
        <= 1.0
    )