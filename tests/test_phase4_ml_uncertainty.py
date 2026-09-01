"""Tests for Phase 4 predictive uncertainty utilities."""

import numpy as np
import pytest
import torch
from torch import nn

from src.perception.ml_uncertainty import (
    bernoulli_predictive_entropy,
    binary_brier_score,
    expected_calibration_error,
    predict_with_perturbation_ensemble,
)


class FixedBlobModel(
    nn.Module
):
    """Deterministic test model containing one foreground blob."""

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x.shape[
            0
        ]

        height = x.shape[
            2
        ]

        width = x.shape[
            3
        ]

        logits = torch.full(
            (
                batch_size,
                1,
                height,
                width,
            ),
            -8.0,
            dtype=x.dtype,
            device=x.device,
        )

        logits[
            :,
            :,
            20:30,
            25:35,
        ] = 8.0

        return logits


def make_image() -> np.ndarray:
    return np.full(
        (
            64,
            64,
            3,
        ),
        100,
        dtype=np.uint8,
    )


def test_entropy_is_small_for_confident_probabilities():
    probability = np.asarray(
        [
            0.001,
            0.999,
        ]
    )

    entropy = (
        bernoulli_predictive_entropy(
            probability
        )
    )

    assert np.all(
        entropy < 0.01
    )


def test_entropy_is_maximal_near_half_probability():
    entropy = (
        bernoulli_predictive_entropy(
            np.asarray(
                [
                    0.1,
                    0.5,
                    0.9,
                ]
            )
        )
    )

    assert entropy[
        1
    ] > entropy[
        0
    ]

    assert entropy[
        1
    ] > entropy[
        2
    ]


def test_perfect_predictions_have_zero_brier_score():
    probability = np.asarray(
        [
            0.0,
            1.0,
            0.0,
            1.0,
        ]
    )

    truth = np.asarray(
        [
            0.0,
            1.0,
            0.0,
            1.0,
        ]
    )

    assert binary_brier_score(
        probability,
        truth,
    ) == pytest.approx(
        0.0
    )


def test_perfect_probabilities_have_zero_calibration_error():
    probability = np.asarray(
        [
            0.0,
            1.0,
            0.0,
            1.0,
        ]
    )

    truth = probability.copy()

    assert (
        expected_calibration_error(
            probability,
            truth,
            bin_count=5,
        )
        == pytest.approx(
            0.0
        )
    )


def test_calibration_rejects_mismatched_shapes():
    with pytest.raises(
        ValueError,
        match="same number",
    ):
        expected_calibration_error(
            np.zeros(
                4
            ),
            np.zeros(
                3
            ),
        )


def test_ensemble_result_has_expected_shapes():
    result = (
        predict_with_perturbation_ensemble(
            model=FixedBlobModel(),
            image_bgr=make_image(),
            device=torch.device(
                "cpu"
            ),
            sample_count=5,
        )
    )

    assert result.mean_probability.shape == (
        64,
        64,
    )

    assert result.probability_variance.shape == (
        64,
        64,
    )

    assert result.predictive_entropy.shape == (
        64,
        64,
    )

    assert result.binary_mask.shape == (
        64,
        64,
    )


def test_ensemble_probabilities_remain_valid():
    result = (
        predict_with_perturbation_ensemble(
            model=FixedBlobModel(),
            image_bgr=make_image(),
            device=torch.device(
                "cpu"
            ),
            sample_count=5,
        )
    )

    assert np.all(
        result.mean_probability
        >= 0.0
    )

    assert np.all(
        result.mean_probability
        <= 1.0
    )

    assert np.all(
        result.probability_variance
        >= 0.0
    )


def test_fixed_blob_has_detectable_centroid():
    result = (
        predict_with_perturbation_ensemble(
            model=FixedBlobModel(),
            image_bgr=make_image(),
            device=torch.device(
                "cpu"
            ),
            sample_count=5,
        )
    )

    assert result.centroid_mean is not None

    assert result.centroid_mean.shape == (
        2,
    )


def test_fixed_blob_produces_two_dimensional_centroid_covariance():
    result = (
        predict_with_perturbation_ensemble(
            model=FixedBlobModel(),
            image_bgr=make_image(),
            device=torch.device(
                "cpu"
            ),
            sample_count=5,
        )
    )

    assert (
        result.centroid_covariance
        is not None
    )

    assert (
        result.centroid_covariance.shape
        == (
            2,
            2,
        )
    )


def test_centroid_success_rate_is_valid_probability():
    result = (
        predict_with_perturbation_ensemble(
            model=FixedBlobModel(),
            image_bgr=make_image(),
            device=torch.device(
                "cpu"
            ),
            sample_count=5,
        )
    )

    assert (
        0.0
        <= result.centroid_success_rate
        <= 1.0
    )


def test_zero_perturbation_produces_zero_predictive_variance():
    result = (
        predict_with_perturbation_ensemble(
            model=FixedBlobModel(),
            image_bgr=make_image(),
            device=torch.device(
                "cpu"
            ),
            sample_count=5,
            noise_sigma=0.0,
            brightness_sigma=0.0,
        )
    )

    assert float(
        np.max(
            result.probability_variance
        )
    ) < 1e-12


def test_ensemble_is_reproducible_for_fixed_seed():
    first = (
        predict_with_perturbation_ensemble(
            model=FixedBlobModel(),
            image_bgr=make_image(),
            device=torch.device(
                "cpu"
            ),
            sample_count=5,
            seed=123,
        )
    )

    second = (
        predict_with_perturbation_ensemble(
            model=FixedBlobModel(),
            image_bgr=make_image(),
            device=torch.device(
                "cpu"
            ),
            sample_count=5,
            seed=123,
        )
    )

    np.testing.assert_allclose(
        first.mean_probability,
        second.mean_probability,
        atol=1e-12,
    )


def test_too_few_ensemble_samples_are_rejected():
    with pytest.raises(
        ValueError,
        match="at least 2",
    ):
        predict_with_perturbation_ensemble(
            model=FixedBlobModel(),
            image_bgr=make_image(),
            device=torch.device(
                "cpu"
            ),
            sample_count=1,
        )