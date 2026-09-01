"""Synthetic segmentation dataset for Phase 4 learned perception.

The dataset is designed specifically to prevent train/validation/test
information leakage.

Each latent scenario defines a marker location, marker radius, background
appearance and illumination configuration. Multiple noisy image frames may
be generated from the same scenario.

Dataset splitting is therefore performed by scenario identifier rather than
by individual image. Frames derived from one latent scene can never appear
in more than one dataset split.

Ground-truth segmentation masks are generated directly from rendering
geometry rather than from the classical OpenCV segmentation algorithm.

The images are synthetic engineering validation data and are not intended
to represent clinical endoscopic-image distributions.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SyntheticDatasetConfig:
    """Configuration for the Phase 4 synthetic segmentation dataset."""

    width: int = 128

    height: int = 128

    scenario_count: int = 120

    frames_per_scenario: int = 3

    train_fraction: float = 0.70

    validation_fraction: float = 0.15

    minimum_marker_radius: int = 6

    maximum_marker_radius: int = 14

    seed: int = 4401

    def __post_init__(self) -> None:
        if self.width < 64:
            raise ValueError(
                "width must be at least 64."
            )

        if self.height < 64:
            raise ValueError(
                "height must be at least 64."
            )

        if self.scenario_count < 3:
            raise ValueError(
                "scenario_count must be at least 3."
            )

        if self.frames_per_scenario < 1:
            raise ValueError(
                "frames_per_scenario must be at least 1."
            )

        if not (
            0.0
            < self.train_fraction
            < 1.0
        ):
            raise ValueError(
                "train_fraction must lie between 0 and 1."
            )

        if not (
            0.0
            < self.validation_fraction
            < 1.0
        ):
            raise ValueError(
                "validation_fraction must lie between 0 and 1."
            )

        if (
            self.train_fraction
            + self.validation_fraction
            >= 1.0
        ):
            raise ValueError(
                "train_fraction + validation_fraction must be less than 1."
            )

        if self.minimum_marker_radius < 2:
            raise ValueError(
                "minimum_marker_radius must be at least 2."
            )

        if (
            self.maximum_marker_radius
            < self.minimum_marker_radius
        ):
            raise ValueError(
                "maximum_marker_radius must not be smaller "
                "than minimum_marker_radius."
            )


@dataclass(frozen=True)
class SyntheticSegmentationDataset:
    """In-memory segmentation dataset."""

    images: np.ndarray

    masks: np.ndarray

    scenario_ids: np.ndarray

    frame_ids: np.ndarray

    marker_pixels: np.ndarray

    marker_radii: np.ndarray

    def __post_init__(self) -> None:
        images = np.asarray(
            self.images
        )

        masks = np.asarray(
            self.masks
        )

        scenario_ids = np.asarray(
            self.scenario_ids
        )

        frame_ids = np.asarray(
            self.frame_ids
        )

        marker_pixels = np.asarray(
            self.marker_pixels,
            dtype=float,
        )

        marker_radii = np.asarray(
            self.marker_radii
        )

        if (
            images.ndim != 4
            or images.shape[-1] != 3
        ):
            raise ValueError(
                "images must have shape (N, H, W, 3)."
            )

        if masks.ndim != 3:
            raise ValueError(
                "masks must have shape (N, H, W)."
            )

        sample_count = images.shape[
            0
        ]

        if masks.shape[0] != sample_count:
            raise ValueError(
                "images and masks must contain the same number of samples."
            )

        if (
            images.shape[1]
            != masks.shape[1]
            or images.shape[2]
            != masks.shape[2]
        ):
            raise ValueError(
                "image and mask dimensions must match."
            )

        if images.dtype != np.uint8:
            raise ValueError(
                "images must use uint8 values."
            )

        if masks.dtype != np.uint8:
            raise ValueError(
                "masks must use uint8 values."
            )

        if scenario_ids.shape != (
            sample_count,
        ):
            raise ValueError(
                "scenario_ids must have shape (N,)."
            )

        if frame_ids.shape != (
            sample_count,
        ):
            raise ValueError(
                "frame_ids must have shape (N,)."
            )

        if marker_pixels.shape != (
            sample_count,
            2,
        ):
            raise ValueError(
                "marker_pixels must have shape (N, 2)."
            )

        if marker_radii.shape != (
            sample_count,
        ):
            raise ValueError(
                "marker_radii must have shape (N,)."
            )

        object.__setattr__(
            self,
            "images",
            images,
        )

        object.__setattr__(
            self,
            "masks",
            masks,
        )

        object.__setattr__(
            self,
            "scenario_ids",
            scenario_ids,
        )

        object.__setattr__(
            self,
            "frame_ids",
            frame_ids,
        )

        object.__setattr__(
            self,
            "marker_pixels",
            marker_pixels,
        )

        object.__setattr__(
            self,
            "marker_radii",
            marker_radii,
        )

    @property
    def sample_count(
        self,
    ) -> int:
        """Return number of images."""

        return int(
            self.images.shape[
                0
            ]
        )

    @property
    def scenario_count(
        self,
    ) -> int:
        """Return number of distinct latent scenarios."""

        return int(
            len(
                np.unique(
                    self.scenario_ids
                )
            )
        )


@dataclass(frozen=True)
class SegmentationDatasetSplits:
    """Leakage-safe train, validation and test splits."""

    train: SyntheticSegmentationDataset

    validation: SyntheticSegmentationDataset

    test: SyntheticSegmentationDataset


def _render_scenario_frame(
    *,
    width: int,
    height: int,
    marker_pixel: np.ndarray,
    marker_radius: int,
    background_bgr: np.ndarray,
    illumination_scale: float,
    frame_seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Render one image and its geometry-derived ground-truth mask."""

    rng = np.random.default_rng(
        frame_seed
    )

    marker_pixel = np.asarray(
        marker_pixel,
        dtype=float,
    )

    marker_x = int(
        round(
            float(
                marker_pixel[
                    0
                ]
            )
        )
    )

    marker_y = int(
        round(
            float(
                marker_pixel[
                    1
                ]
            )
        )
    )

    image = np.empty(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    scaled_background = np.clip(
        background_bgr.astype(
            float
        )
        * illumination_scale,
        0.0,
        255.0,
    ).astype(
        np.uint8
    )

    image[:] = (
        scaled_background
    )

    # Endoscopic-style circular illumination field.
    field_mask = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.uint8,
    )

    cv2.circle(
        field_mask,
        (
            width // 2,
            height // 2,
        ),
        int(
            0.48
            * min(
                width,
                height,
            )
        ),
        255,
        thickness=-1,
    )

    darker = np.clip(
        image.astype(
            float
        )
        * 0.55,
        0.0,
        255.0,
    ).astype(
        np.uint8
    )

    image = np.where(
        field_mask[
            :,
            :,
            None,
        ]
        > 0,
        image,
        darker,
    ).astype(
        np.uint8
    )

    mask = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.uint8,
    )

    cv2.circle(
        mask,
        (
            marker_x,
            marker_y,
        ),
        marker_radius,
        255,
        thickness=-1,
        lineType=cv2.LINE_8,
    )

    green_value = int(
        rng.integers(
            175,
            241,
        )
    )

    marker_colour = (
        int(
            rng.integers(
                20,
                65,
            )
        ),
        green_value,
        int(
            rng.integers(
                20,
                70,
            )
        ),
    )

    cv2.circle(
        image,
        (
            marker_x,
            marker_y,
        ),
        marker_radius,
        marker_colour,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    # Add non-target reddish structures.
    distractor_count = int(
        rng.integers(
            1,
            5,
        )
    )

    for _ in range(
        distractor_count
    ):
        centre = (
            int(
                rng.integers(
                    8,
                    width - 8,
                )
            ),
            int(
                rng.integers(
                    8,
                    height - 8,
                )
            ),
        )

        radius = int(
            rng.integers(
                3,
                10,
            )
        )

        colour = (
            int(
                rng.integers(
                    40,
                    100,
                )
            ),
            int(
                rng.integers(
                    40,
                    100,
                )
            ),
            int(
                rng.integers(
                    100,
                    180,
                )
            ),
        )

        cv2.circle(
            image,
            centre,
            radius,
            colour,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    # Specular highlights vary between frames.
    highlight_count = int(
        rng.integers(
            1,
            4,
        )
    )

    for _ in range(
        highlight_count
    ):
        centre = (
            int(
                rng.integers(
                    5,
                    width - 5,
                )
            ),
            int(
                rng.integers(
                    5,
                    height - 5,
                )
            ),
        )

        radius = int(
            rng.integers(
                2,
                6,
            )
        )

        intensity = int(
            rng.integers(
                220,
                256,
            )
        )

        cv2.circle(
            image,
            centre,
            radius,
            (
                intensity,
                intensity,
                intensity,
            ),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    # Mild blur variability.
    if rng.random() < 0.50:
        image = cv2.GaussianBlur(
            image,
            (
                3,
                3,
            ),
            sigmaX=0.0,
        )

    noise_sigma = float(
        rng.uniform(
            1.0,
            10.0,
        )
    )

    noise = rng.normal(
        loc=0.0,
        scale=noise_sigma,
        size=image.shape,
    )

    image = np.clip(
        image.astype(
            float
        )
        + noise,
        0.0,
        255.0,
    ).astype(
        np.uint8
    )

    return (
        image,
        mask,
    )


def generate_synthetic_segmentation_dataset(
    config: SyntheticDatasetConfig
    | None = None,
) -> SyntheticSegmentationDataset:
    """Generate all synthetic segmentation samples."""

    if config is None:
        config = (
            SyntheticDatasetConfig()
        )

    rng = np.random.default_rng(
        config.seed
    )

    images: list[
        np.ndarray
    ] = []

    masks: list[
        np.ndarray
    ] = []

    scenario_ids: list[
        int
    ] = []

    frame_ids: list[
        int
    ] = []

    marker_pixels: list[
        np.ndarray
    ] = []

    marker_radii: list[
        int
    ] = []

    for scenario_id in range(
        config.scenario_count
    ):
        radius = int(
            rng.integers(
                config.minimum_marker_radius,
                config.maximum_marker_radius
                + 1,
            )
        )

        border = (
            radius
            + 5
        )

        marker_pixel = np.asarray(
            [
                rng.uniform(
                    border,
                    config.width
                    - border,
                ),
                rng.uniform(
                    border,
                    config.height
                    - border,
                ),
            ],
            dtype=float,
        )

        background_bgr = np.asarray(
            [
                rng.integers(
                    45,
                    95,
                ),
                rng.integers(
                    45,
                    100,
                ),
                rng.integers(
                    105,
                    165,
                ),
            ],
            dtype=np.uint8,
        )

        illumination_scale = float(
            rng.uniform(
                0.70,
                1.30,
            )
        )

        for frame_id in range(
            config.frames_per_scenario
        ):
            frame_seed = int(
                config.seed
                + 10000
                * scenario_id
                + frame_id
            )

            image, mask = (
                _render_scenario_frame(
                    width=config.width,
                    height=config.height,
                    marker_pixel=(
                        marker_pixel
                    ),
                    marker_radius=radius,
                    background_bgr=(
                        background_bgr
                    ),
                    illumination_scale=(
                        illumination_scale
                    ),
                    frame_seed=(
                        frame_seed
                    ),
                )
            )

            images.append(
                image
            )

            masks.append(
                mask
            )

            scenario_ids.append(
                scenario_id
            )

            frame_ids.append(
                frame_id
            )

            marker_pixels.append(
                marker_pixel.copy()
            )

            marker_radii.append(
                radius
            )

    return SyntheticSegmentationDataset(
        images=np.stack(
            images,
            axis=0,
        ),
        masks=np.stack(
            masks,
            axis=0,
        ),
        scenario_ids=np.asarray(
            scenario_ids,
            dtype=int,
        ),
        frame_ids=np.asarray(
            frame_ids,
            dtype=int,
        ),
        marker_pixels=np.vstack(
            marker_pixels
        ),
        marker_radii=np.asarray(
            marker_radii,
            dtype=int,
        ),
    )


def _subset_dataset(
    dataset: SyntheticSegmentationDataset,
    scenario_ids: np.ndarray,
) -> SyntheticSegmentationDataset:
    """Extract all frames associated with selected scenarios."""

    scenario_ids = np.asarray(
        scenario_ids,
        dtype=int,
    )

    keep = np.isin(
        dataset.scenario_ids,
        scenario_ids,
    )

    return SyntheticSegmentationDataset(
        images=dataset.images[
            keep
        ].copy(),
        masks=dataset.masks[
            keep
        ].copy(),
        scenario_ids=dataset.scenario_ids[
            keep
        ].copy(),
        frame_ids=dataset.frame_ids[
            keep
        ].copy(),
        marker_pixels=dataset.marker_pixels[
            keep
        ].copy(),
        marker_radii=dataset.marker_radii[
            keep
        ].copy(),
    )


def split_dataset_by_scenario(
    dataset: SyntheticSegmentationDataset,
    *,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> SegmentationDatasetSplits:
    """Split dataset while keeping every latent scenario in one split."""

    if not (
        0.0
        < train_fraction
        < 1.0
    ):
        raise ValueError(
            "train_fraction must lie between 0 and 1."
        )

    if not (
        0.0
        < validation_fraction
        < 1.0
    ):
        raise ValueError(
            "validation_fraction must lie between 0 and 1."
        )

    if (
        train_fraction
        + validation_fraction
        >= 1.0
    ):
        raise ValueError(
            "train_fraction + validation_fraction must be less than 1."
        )

    unique_scenarios = np.unique(
        dataset.scenario_ids
    )

    if unique_scenarios.size < 3:
        raise ValueError(
            "At least three scenarios are required for splitting."
        )

    rng = np.random.default_rng(
        seed
    )

    shuffled = (
        unique_scenarios.copy()
    )

    rng.shuffle(
        shuffled
    )

    scenario_count = int(
        shuffled.size
    )

    train_count = max(
        1,
        int(
            np.floor(
                scenario_count
                * train_fraction
            )
        ),
    )

    validation_count = max(
        1,
        int(
            np.floor(
                scenario_count
                * validation_fraction
            )
        ),
    )

    if (
        train_count
        + validation_count
        >= scenario_count
    ):
        raise ValueError(
            "Split fractions leave no scenarios for the test set."
        )

    train_scenarios = shuffled[
        :train_count
    ]

    validation_scenarios = shuffled[
        train_count:
        train_count
        + validation_count
    ]

    test_scenarios = shuffled[
        train_count
        + validation_count:
    ]

    return SegmentationDatasetSplits(
        train=_subset_dataset(
            dataset,
            train_scenarios,
        ),
        validation=_subset_dataset(
            dataset,
            validation_scenarios,
        ),
        test=_subset_dataset(
            dataset,
            test_scenarios,
        ),
    )


def build_phase4_dataset(
    config: SyntheticDatasetConfig
    | None = None,
) -> SegmentationDatasetSplits:
    """Generate and split the Phase 4 dataset."""

    if config is None:
        config = (
            SyntheticDatasetConfig()
        )

    dataset = (
        generate_synthetic_segmentation_dataset(
            config
        )
    )

    return split_dataset_by_scenario(
        dataset,
        train_fraction=(
            config.train_fraction
        ),
        validation_fraction=(
            config.validation_fraction
        ),
        seed=(
            config.seed
            + 1
        ),
    )