"""Tests for the normalised task-aware ablation."""

import numpy as np
import pytest

from src.simulation.normalised_ablation import (
    NormalisedAblationVariant,
    candidate_uncertainties,
    normalise_uncertainty,
    make_observation_model,
    make_target,
    make_task,
    make_variants,
    run_study,
    run_variant,
)


def test_four_variants_exist() -> None:
    variants = make_variants()

    assert len(variants) == 4


def test_uncertainty_normalisation_rewards_lower_uncertainty() -> None:
    values = np.array(
        [1.0, 2.0, 3.0],
        dtype=float,
    )

    result = normalise_uncertainty(
        values
    )

    np.testing.assert_allclose(
        result,
        np.array(
            [1.0, 0.5, 0.0]
        ),
    )


def test_uncertainty_normalisation_is_bounded() -> None:
    values = np.array(
        [0.2, 0.5, 1.0],
        dtype=float,
    )

    result = normalise_uncertainty(
        values
    )

    assert np.all(
        result >= 0.0
    )

    assert np.all(
        result <= 1.0
    )


def test_constant_uncertainty_returns_ones() -> None:
    values = np.array(
        [2.0, 2.0, 2.0],
        dtype=float,
    )

    result = normalise_uncertainty(
        values
    )

    np.testing.assert_allclose(
        result,
        np.ones(3),
    )


def test_empty_uncertainty_is_rejected() -> None:
    with pytest.raises(ValueError):
        normalise_uncertainty(
            np.array([])
        )


def test_non_finite_uncertainty_is_rejected() -> None:
    with pytest.raises(ValueError):
        normalise_uncertainty(
            np.array(
                [1.0, np.nan]
            )
        )


def test_candidate_uncertainties_are_finite() -> None:
    model = make_observation_model()
    target = make_target()

    from src.perception.viewpoints import (
        generate_candidate_viewpoints,
    )

    candidates = (
        generate_candidate_viewpoints(
            target_position=target.centre
        )
    )

    values = candidate_uncertainties(
        model=model,
        candidates=candidates,
        target=target,
    )

    assert values.shape == (
        len(candidates),
    )

    assert np.all(
        np.isfinite(values)
    )


def test_variants_have_unique_names() -> None:
    variants = make_variants()

    names = [
        variant.name
        for variant in variants
    ]

    assert len(names) == len(
        set(names)
    )


def test_invalid_trial_count_is_rejected() -> None:
    with pytest.raises(ValueError):
        run_variant(
            variant=make_variants()[0],
            trial_count=0,
        )


def test_variant_result_is_valid() -> None:
    result = run_variant(
        variant=make_variants()[3],
        trial_count=1,
    )

    assert result.trial_count == 1

    assert np.isfinite(
        result.mean_selected_uncertainty
    )

    assert np.isfinite(
        result.mean_selected_alignment
    )

    assert np.isfinite(
        result.mean_generic_score
    )

    assert np.isfinite(
        result.mean_final_score
    )


def test_study_contains_four_results() -> None:
    study = run_study(
        trial_count=1
    )

    assert len(
        study.results
    ) == 4