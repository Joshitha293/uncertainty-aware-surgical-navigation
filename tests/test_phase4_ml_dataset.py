"""Tests for Phase 4 leakage-safe synthetic segmentation dataset."""

import numpy as np
import pytest

from src.perception.ml_dataset import (
    SyntheticDatasetConfig,
    build_phase4_dataset,
    generate_synthetic_segmentation_dataset,
)


def small_config(
    seed: int = 4401,
) -> SyntheticDatasetConfig:
    return SyntheticDatasetConfig(
        width=96,
        height=96,
        scenario_count=20,
        frames_per_scenario=2,
        train_fraction=0.60,
        validation_fraction=0.20,
        seed=seed,
    )


def test_generated_dataset_has_expected_sample_count():
    config = small_config()

    dataset = (
        generate_synthetic_segmentation_dataset(
            config
        )
    )

    assert (
        dataset.sample_count
        == (
            config.scenario_count
            * config.frames_per_scenario
        )
    )

    assert (
        dataset.scenario_count
        == config.scenario_count
    )


def test_image_and_mask_shapes_are_correct():
    config = small_config()

    dataset = (
        generate_synthetic_segmentation_dataset(
            config
        )
    )

    assert dataset.images.shape == (
        40,
        96,
        96,
        3,
    )

    assert dataset.masks.shape == (
        40,
        96,
        96,
    )


def test_ground_truth_masks_are_binary_and_nonempty():
    dataset = (
        generate_synthetic_segmentation_dataset(
            small_config()
        )
    )

    values = set(
        np.unique(
            dataset.masks
        ).tolist()
    )

    assert values == {
        0,
        255,
    }

    foreground_counts = (
        np.count_nonzero(
            dataset.masks,
            axis=(
                1,
                2,
            ),
        )
    )

    assert np.all(
        foreground_counts
        > 0
    )


def test_marker_centres_lie_inside_ground_truth_masks():
    dataset = (
        generate_synthetic_segmentation_dataset(
            small_config()
        )
    )

    for index in range(
        dataset.sample_count
    ):
        x = int(
            round(
                dataset.marker_pixels[
                    index,
                    0,
                ]
            )
        )

        y = int(
            round(
                dataset.marker_pixels[
                    index,
                    1,
                ]
            )
        )

        assert (
            dataset.masks[
                index,
                y,
                x,
            ]
            == 255
        )


def test_generation_is_reproducible_for_fixed_seed():
    first = (
        generate_synthetic_segmentation_dataset(
            small_config(
                seed=101
            )
        )
    )

    second = (
        generate_synthetic_segmentation_dataset(
            small_config(
                seed=101
            )
        )
    )

    np.testing.assert_array_equal(
        first.images,
        second.images,
    )

    np.testing.assert_array_equal(
        first.masks,
        second.masks,
    )

    np.testing.assert_array_equal(
        first.scenario_ids,
        second.scenario_ids,
    )


def test_different_seed_changes_dataset():
    first = (
        generate_synthetic_segmentation_dataset(
            small_config(
                seed=101
            )
        )
    )

    second = (
        generate_synthetic_segmentation_dataset(
            small_config(
                seed=102
            )
        )
    )

    assert not np.array_equal(
        first.images,
        second.images,
    )


def test_scenario_sets_are_disjoint_between_splits():
    splits = (
        build_phase4_dataset(
            small_config()
        )
    )

    train = set(
        splits.train
        .scenario_ids
        .tolist()
    )

    validation = set(
        splits.validation
        .scenario_ids
        .tolist()
    )

    test = set(
        splits.test
        .scenario_ids
        .tolist()
    )

    assert train.isdisjoint(
        validation
    )

    assert train.isdisjoint(
        test
    )

    assert validation.isdisjoint(
        test
    )


def test_every_scenario_appears_in_exactly_one_split():
    config = small_config()

    splits = (
        build_phase4_dataset(
            config
        )
    )

    combined = (
        set(
            splits.train
            .scenario_ids
            .tolist()
        )
        | set(
            splits.validation
            .scenario_ids
            .tolist()
        )
        | set(
            splits.test
            .scenario_ids
            .tolist()
        )
    )

    assert combined == set(
        range(
            config.scenario_count
        )
    )


def test_all_frames_from_one_scenario_remain_together():
    splits = (
        build_phase4_dataset(
            small_config()
        )
    )

    ownership: dict[
        int,
        str,
    ] = {}

    for name, subset in (
        (
            "train",
            splits.train,
        ),
        (
            "validation",
            splits.validation,
        ),
        (
            "test",
            splits.test,
        ),
    ):
        for scenario_id in np.unique(
            subset.scenario_ids
        ):
            scenario_id = int(
                scenario_id
            )

            assert (
                scenario_id
                not in ownership
            )

            ownership[
                scenario_id
            ] = name


def test_invalid_split_fractions_are_rejected():
    with pytest.raises(
        ValueError,
        match="less than 1",
    ):
        SyntheticDatasetConfig(
            train_fraction=0.80,
            validation_fraction=0.25,
        )