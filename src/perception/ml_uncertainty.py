"""Predictive uncertainty utilities for learned Phase 4 perception.

The module quantifies sensitivity of a trained segmentation network to
plausible image-measurement perturbations.

Repeated inference is performed on independently perturbed versions of the
same BGR image. The resulting distribution provides:

- mean foreground probability;
- probability variance;
- Bernoulli predictive entropy;
- segmentation-mask estimate;
- centroid distribution;
- empirical 2-D centroid covariance.

This is perturbation-based predictive uncertainty. It should not be
interpreted as a complete Bayesian posterior or clinical confidence score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.perception.ml_segmentation import (
    binary_mask_centroid,
)


@dataclass(frozen=True)
class PredictiveSegmentationUncertainty:
    """Predictive segmentation statistics from repeated inference."""

    mean_probability: np.ndarray

    probability_variance: np.ndarray

    predictive_entropy: np.ndarray

    binary_mask: np.ndarray

    mean_entropy: float

    foreground_entropy: float

    centroid_mean: np.ndarray | None

    centroid_covariance: np.ndarray | None

    centroid_success_rate: float

    sample_count: int


def _validate_probability_array(
    probabilities: np.ndarray,
) -> np.ndarray:
    """Validate and return floating-point probabilities."""

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if not np.all(
        np.isfinite(
            probabilities
        )
    ):
        raise ValueError(
            "probabilities must contain finite values."
        )

    if (
        np.any(
            probabilities < 0.0
        )
        or np.any(
            probabilities > 1.0
        )
    ):
        raise ValueError(
            "probabilities must lie in [0, 1]."
        )

    return probabilities


def bernoulli_predictive_entropy(
    probabilities: np.ndarray,
    *,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Return Bernoulli entropy for foreground probabilities."""

    probabilities = (
        _validate_probability_array(
            probabilities
        )
    )

    if epsilon <= 0.0:
        raise ValueError(
            "epsilon must be positive."
        )

    clipped = np.clip(
        probabilities,
        epsilon,
        1.0 - epsilon,
    )

    return -(
        clipped
        * np.log(
            clipped
        )
        + (
            1.0
            - clipped
        )
        * np.log(
            1.0
            - clipped
        )
    )


def binary_brier_score(
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Return mean binary Brier score."""

    probabilities = (
        _validate_probability_array(
            probabilities
        )
    )

    targets = np.asarray(
        targets,
        dtype=float,
    )

    if targets.shape != probabilities.shape:
        raise ValueError(
            "targets and probabilities must have identical shapes."
        )

    if not np.all(
        np.isin(
            targets,
            (
                0.0,
                1.0,
            ),
        )
    ):
        raise ValueError(
            "targets must contain only 0 and 1."
        )

    return float(
        np.mean(
            (
                probabilities
                - targets
            )
            ** 2
        )
    )


def expected_calibration_error(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    bin_count: int = 15,
) -> float:
    """Compute pixel-level expected calibration error."""

    probabilities = (
        _validate_probability_array(
            probabilities
        )
        .reshape(
            -1
        )
    )

    targets = np.asarray(
        targets,
        dtype=float,
    ).reshape(
        -1
    )

    if targets.shape != probabilities.shape:
        raise ValueError(
            "targets and probabilities must contain the same number "
            "of elements."
        )

    if bin_count < 2:
        raise ValueError(
            "bin_count must be at least 2."
        )

    if not np.all(
        np.isin(
            targets,
            (
                0.0,
                1.0,
            ),
        )
    ):
        raise ValueError(
            "targets must contain only 0 and 1."
        )

    boundaries = np.linspace(
        0.0,
        1.0,
        bin_count + 1,
    )

    calibration_error = 0.0

    sample_count = int(
        probabilities.size
    )

    for index in range(
        bin_count
    ):
        lower = boundaries[
            index
        ]

        upper = boundaries[
            index + 1
        ]

        if index == (
            bin_count - 1
        ):
            membership = (
                (
                    probabilities
                    >= lower
                )
                & (
                    probabilities
                    <= upper
                )
            )
        else:
            membership = (
                (
                    probabilities
                    >= lower
                )
                & (
                    probabilities
                    < upper
                )
            )

        count = int(
            np.count_nonzero(
                membership
            )
        )

        if count == 0:
            continue

        confidence = float(
            np.mean(
                probabilities[
                    membership
                ]
            )
        )

        accuracy = float(
            np.mean(
                targets[
                    membership
                ]
            )
        )

        calibration_error += (
            count
            / sample_count
        ) * abs(
            confidence
            - accuracy
        )

    return float(
        calibration_error
    )


def _image_to_tensor(
    image_bgr: np.ndarray,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Convert one uint8 BGR image into an NCHW RGB tensor."""

    image = np.asarray(
        image_bgr
    )

    if (
        image.ndim != 3
        or image.shape[
            2
        ] != 3
    ):
        raise ValueError(
            "image_bgr must have shape (H, W, 3)."
        )

    if image.dtype != np.uint8:
        raise ValueError(
            "image_bgr must use uint8 values."
        )

    rgb = (
        image[
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
        / 255.0
    )

    return (
        tensor
        .unsqueeze(
            0
        )
        .to(
            device
        )
    )


def _perturb_image(
    image_bgr: np.ndarray,
    *,
    rng: np.random.Generator,
    noise_sigma: float,
    brightness_sigma: float,
) -> np.ndarray:
    """Apply one controlled image-measurement perturbation."""

    if noise_sigma < 0.0:
        raise ValueError(
            "noise_sigma must be non-negative."
        )

    if brightness_sigma < 0.0:
        raise ValueError(
            "brightness_sigma must be non-negative."
        )

    image = np.asarray(
        image_bgr,
        dtype=float,
    )

    brightness = float(
        rng.normal(
            loc=1.0,
            scale=brightness_sigma,
        )
    )

    noise = rng.normal(
        loc=0.0,
        scale=noise_sigma,
        size=image.shape,
    )

    perturbed = (
        brightness
        * image
        + noise
    )

    return np.clip(
        perturbed,
        0.0,
        255.0,
    ).astype(
        np.uint8
    )


def predict_with_perturbation_ensemble(
    *,
    model: nn.Module,
    image_bgr: np.ndarray,
    device: torch.device,
    sample_count: int = 20,
    threshold: float = 0.5,
    noise_sigma: float = 4.0,
    brightness_sigma: float = 0.04,
    seed: int = 4701,
) -> PredictiveSegmentationUncertainty:
    """Estimate predictive uncertainty using repeated perturbed inference."""

    if sample_count < 2:
        raise ValueError(
            "sample_count must be at least 2."
        )

    if not (
        0.0
        < threshold
        < 1.0
    ):
        raise ValueError(
            "threshold must lie between 0 and 1."
        )

    if noise_sigma < 0.0:
        raise ValueError(
            "noise_sigma must be non-negative."
        )

    if brightness_sigma < 0.0:
        raise ValueError(
            "brightness_sigma must be non-negative."
        )

    rng = np.random.default_rng(
        seed
    )

    model.eval()

    probability_samples: list[
        np.ndarray
    ] = []

    centroids: list[
        np.ndarray
    ] = []

    with torch.no_grad():
        for sample_index in range(
            sample_count
        ):
            # Include the original observation once.
            if sample_index == 0:
                image = np.asarray(
                    image_bgr
                ).copy()
            else:
                image = (
                    _perturb_image(
                        image_bgr,
                        rng=rng,
                        noise_sigma=(
                            noise_sigma
                        ),
                        brightness_sigma=(
                            brightness_sigma
                        ),
                    )
                )

            tensor = (
                _image_to_tensor(
                    image,
                    device=device,
                )
            )

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
                .astype(
                    float
                )
            )

            probability_samples.append(
                probability
            )

            centroid = (
                binary_mask_centroid(
                    probability
                    >= threshold
                )
            )

            if centroid is not None:
                centroids.append(
                    centroid
                )

    probability_stack = np.stack(
        probability_samples,
        axis=0,
    )

    mean_probability = np.mean(
        probability_stack,
        axis=0,
    )

    probability_variance = np.var(
        probability_stack,
        axis=0,
        ddof=1,
    )

    entropy = (
        bernoulli_predictive_entropy(
            mean_probability
        )
    )

    binary_mask = (
        mean_probability
        >= threshold
    )

    mean_entropy = float(
        np.mean(
            entropy
        )
    )

    uncertainty_region = (
        mean_probability
        >= 0.10
    )

    if np.any(
        uncertainty_region
    ):
        foreground_entropy = float(
            np.mean(
                entropy[
                    uncertainty_region
                ]
            )
        )

    else:
        foreground_entropy = (
            mean_entropy
        )

    if centroids:
        centroid_array = np.vstack(
            centroids
        )

        centroid_mean = np.mean(
            centroid_array,
            axis=0,
        )

        if centroid_array.shape[
            0
        ] >= 2:
            centroid_covariance = (
                np.cov(
                    centroid_array,
                    rowvar=False,
                    ddof=1,
                )
            )

            centroid_covariance = (
                0.5
                * (
                    centroid_covariance
                    + centroid_covariance.T
                )
            )

        else:
            centroid_covariance = None

    else:
        centroid_mean = None
        centroid_covariance = None

    return PredictiveSegmentationUncertainty(
        mean_probability=np.asarray(
            mean_probability,
            dtype=float,
        ),
        probability_variance=np.asarray(
            probability_variance,
            dtype=float,
        ),
        predictive_entropy=np.asarray(
            entropy,
            dtype=float,
        ),
        binary_mask=np.asarray(
            binary_mask,
            dtype=bool,
        ),
        mean_entropy=mean_entropy,
        foreground_entropy=(
            foreground_entropy
        ),
        centroid_mean=(
            None
            if centroid_mean is None
            else np.asarray(
                centroid_mean,
                dtype=float,
            )
        ),
        centroid_covariance=(
            None
            if centroid_covariance is None
            else np.asarray(
                centroid_covariance,
                dtype=float,
            )
        ),
        centroid_success_rate=float(
            len(
                centroids
            )
            / sample_count
        ),
        sample_count=int(
            sample_count
        ),
    )