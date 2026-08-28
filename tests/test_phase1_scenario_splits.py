"""Tests for frozen Phase 1 scenario splits."""

import json

from src.simulation.phase1_scenario_splits import (
    Phase1SplitConfig,
    build_phase1_splits,
    manifest_digest,
    save_manifest,
    scenario_signature,
)
from src.simulation.three_strategy_robustness_benchmark import (
    default_scenarios,
)


def test_default_split_counts():
    """Frozen split sizes must match the experimental protocol."""

    config = (
        Phase1SplitConfig()
    )

    splits = (
        build_phase1_splits(
            config
        )
    )

    assert len(
        splits.development
    ) == 10

    assert len(
        splits.validation
    ) == 20

    assert len(
        splits.held_out
    ) == 30


def test_development_split_is_historical_set():
    """The ten already-seen scenarios remain development-only."""

    splits = (
        build_phase1_splits()
    )

    expected = tuple(
        default_scenarios()
    )

    assert len(
        splits.development
    ) == len(
        expected
    )

    for actual, reference in zip(
        splits.development,
        expected,
    ):
        assert (
            scenario_signature(
                actual
            )
            == scenario_signature(
                reference
            )
        )


def test_generation_is_deterministic():
    """Same frozen seeds must reproduce exactly the same scenes."""

    first = (
        build_phase1_splits()
    )

    second = (
        build_phase1_splits()
    )

    assert [
        scenario_signature(
            scenario
        )
        for scenario
        in first.validation
    ] == [
        scenario_signature(
            scenario
        )
        for scenario
        in second.validation
    ]

    assert [
        scenario_signature(
            scenario
        )
        for scenario
        in first.held_out
    ] == [
        scenario_signature(
            scenario
        )
        for scenario
        in second.held_out
    ]


def test_all_three_splits_are_physically_disjoint():
    """No exact scene definition may appear in more than one split."""

    splits = (
        build_phase1_splits()
    )

    development = {
        scenario_signature(
            scenario
        )
        for scenario
        in splits.development
    }

    validation = {
        scenario_signature(
            scenario
        )
        for scenario
        in splits.validation
    }

    held_out = {
        scenario_signature(
            scenario
        )
        for scenario
        in splits.held_out
    }

    assert development.isdisjoint(
        validation
    )

    assert development.isdisjoint(
        held_out
    )

    assert validation.isdisjoint(
        held_out
    )


def test_ids_are_unique_across_splits():
    """Scenario identifiers must not collide."""

    splits = (
        build_phase1_splits()
    )

    ids = [
        scenario.scenario_id
        for scenario
        in (
            splits.development
            + splits.validation
            + splits.held_out
        )
    ]

    assert len(
        ids
    ) == len(
        set(
            ids
        )
    )


def test_generated_parameters_stay_within_frozen_bounds():
    """Procedural validation/test scenes must obey protocol bounds."""

    config = (
        Phase1SplitConfig()
    )

    splits = (
        build_phase1_splits(
            config
        )
    )

    generated = (
        splits.validation
        + splits.held_out
    )

    for scenario in generated:
        x, y, z = (
            scenario.translation
        )

        assert (
            config.translation_x_min
            <= x
            <= config.translation_x_max
        )

        assert (
            config.translation_y_min
            <= y
            <= config.translation_y_max
        )

        assert (
            config.translation_z_min
            <= z
            <= config.translation_z_max
        )

        assert (
            config.radius_scale_min
            <= scenario.radius_scale
            <= config.radius_scale_max
        )

        assert (
            config.safety_margin_scale_min
            <= scenario.safety_margin_scale
            <= config.safety_margin_scale_max
        )

        assert (
            config.initial_view_index_min
            <= scenario.initial_view_index
            < config.initial_view_index_max_exclusive
        )

        assert (
            scenario.occluder_radius
            == 0.0
            or (
                config.occluder_radius_min
                <= scenario.occluder_radius
                <= config.occluder_radius_max
            )
        )


def test_manifest_digest_is_reproducible():
    """Frozen scenario definitions must produce a stable digest."""

    config = (
        Phase1SplitConfig()
    )

    first = (
        build_phase1_splits(
            config
        )
    )

    second = (
        build_phase1_splits(
            config
        )
    )

    assert (
        manifest_digest(
            splits=first,
            config=config,
        )
        == manifest_digest(
            splits=second,
            config=config,
        )
    )


def test_manifest_save_contains_digest(
    tmp_path,
):
    """Saved protocol evidence must contain its SHA-256 digest."""

    config = (
        Phase1SplitConfig()
    )

    splits = (
        build_phase1_splits(
            config
        )
    )

    path, digest = (
        save_manifest(
            splits=splits,
            config=config,
            output_path=(
                tmp_path
                / "manifest.json"
            ),
        )
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(
            handle
        )

    assert (
        payload[
            "sha256"
        ]
        == digest
    )

    assert (
        len(
            payload[
                "validation"
            ]
        )
        == config.validation_count
    )

    assert (
        len(
            payload[
                "held_out"
            ]
        )
        == config.held_out_count
    )