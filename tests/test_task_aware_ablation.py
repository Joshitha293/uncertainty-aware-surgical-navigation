"""Tests for task-aware ablation experiments."""

import numpy as np
import pytest

from src.simulation.task_aware_ablation import (
    AblationResult,
    AblationStudy,
    AblationVariant,
    build_scorer,
    make_variants,
    run_ablation,
    run_variant,
)


def test_four_ablation_variants_exist() -> None:
    variants = make_variants()

    assert len(variants) == 4


def test_variant_names_are_unique() -> None:
    variants = make_variants()

    names = [
        variant.name
        for variant in variants
    ]

    assert len(names) == len(
        set(names)
    )


def test_generic_variant_disables_both_components() -> None:
    variant = make_variants()[0]

    assert not variant.use_alignment
    assert not variant.use_uncertainty


def test_alignment_variant_enables_only_alignment() -> None:
    variant = make_variants()[1]

    assert variant.use_alignment
    assert not variant.use_uncertainty


def test_uncertainty_variant_enables_only_uncertainty() -> None:
    variant = make_variants()[2]

    assert not variant.use_alignment
    assert variant.use_uncertainty


def test_full_variant_enables_both_components() -> None:
    variant = make_variants()[3]

    assert variant.use_alignment
    assert variant.use_uncertainty


def test_run_variant_requires_positive_trial_count() -> None:
    variant = make_variants()[0]

    with pytest.raises(ValueError):
        run_variant(
            variant=variant,
            trial_count=0,
        )


def test_run_variant_returns_result() -> None:
    result = run_variant(
        variant=make_variants()[0],
        trial_count=1,
    )

    assert isinstance(
        result,
        AblationResult,
    )

    assert result.trial_count == 1


def test_result_metrics_are_finite() -> None:
    result = run_variant(
        variant=make_variants()[3],
        trial_count=1,
    )

    assert np.isfinite(
        result.mean_selected_uncertainty
    )

    assert np.isfinite(
        result.mean_task_alignment
    )

    assert np.isfinite(
        result.mean_generic_score
    )

    assert np.isfinite(
        result.mean_final_score
    )


def test_alignment_is_bounded() -> None:
    result = run_variant(
        variant=make_variants()[3],
        trial_count=1,
    )

    assert (
        0.0
        <= result.mean_task_alignment
        <= 1.0
    )


def test_selection_difference_rate_is_bounded() -> None:
    result = run_variant(
        variant=make_variants()[3],
        trial_count=1,
    )

    assert (
        0.0
        <= result.selection_difference_rate_percent
        <= 100.0
    )


def test_ablation_returns_complete_study() -> None:
    study = run_ablation(
        trial_count=1
    )

    assert isinstance(
        study,
        AblationStudy,
    )

    assert len(
        study.results
    ) == 4


def test_ablation_order_matches_variants() -> None:
    study = run_ablation(
        trial_count=1
    )

    expected = [
        variant.name
        for variant in make_variants()
    ]

    actual = [
        result.variant
        for result in study.results
    ]

    assert actual == expected