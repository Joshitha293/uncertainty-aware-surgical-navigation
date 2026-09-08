"""Tests for Phase 6 robustness stress benchmark."""

from __future__ import annotations

import numpy as np
import pytest

from src.simulation.phase6_robustness_stress_benchmark import (
    DEFAULT_SEED,
    MAX_DRIFT,
    OVER_ESTIMATED_STD_SCALE,
    ROBUST_POLICY,
    TEMPORAL_CORRELATION,
    UNDER_ESTIMATED_STD_SCALE,
    _orientation_mismatch_covariance,
    _reported_covariance,
    generate_stress_errors,
    nominal_decision,
    planner_decisions,
    robust_decision,
    stress_condition_names,
)


def test_all_required_stress_conditions_exist() -> None:
    assert stress_condition_names() == (
        "calibrated_gaussian",
        "covariance_underestimated",
        "covariance_overestimated",
        "heavy_tailed_contamination",
        "orientation_mismatch",
        "temporally_correlated_drift",
    )


def test_nominal_candidate_is_released() -> None:
    decision = (
        nominal_decision()
    )

    assert decision.accepted is True


def test_robust_policy_is_more_conservative_for_candidate() -> None:
    nominal, robust = (
        planner_decisions()
    )

    assert nominal.accepted is True

    assert robust.accepted is False


def test_robust_probability_threshold_is_tighter_than_nominal() -> None:
    decision = (
        robust_decision()
    )

    assert (
        decision.reported_threshold
        < 0.05
    )

    assert np.isclose(
        decision.reported_threshold,
        ROBUST_POLICY
        .effective_gaussian_threshold,
    )


def test_underestimated_covariance_has_larger_empirical_variance() -> None:
    nominal_rng = (
        np.random.default_rng(
            DEFAULT_SEED
        )
    )

    underestimated_rng = (
        np.random.default_rng(
            DEFAULT_SEED
        )
    )

    nominal = (
        generate_stress_errors(
            "calibrated_gaussian",
            5000,
            nominal_rng,
        )
    )

    underestimated = (
        generate_stress_errors(
            "covariance_underestimated",
            5000,
            underestimated_rng,
        )
    )

    nominal_trace = float(
        np.trace(
            np.cov(
                nominal,
                rowvar=False,
            )
        )
    )

    underestimated_trace = float(
        np.trace(
            np.cov(
                underestimated,
                rowvar=False,
            )
        )
    )

    assert (
        underestimated_trace
        > nominal_trace
    )

    expected_ratio = (
        UNDER_ESTIMATED_STD_SCALE
        ** 2
    )

    empirical_ratio = (
        underestimated_trace
        / nominal_trace
    )

    assert np.isclose(
        empirical_ratio,
        expected_ratio,
        rtol=0.15,
    )


def test_overestimated_covariance_has_smaller_empirical_variance() -> None:
    nominal_rng = (
        np.random.default_rng(
            DEFAULT_SEED
        )
    )

    overestimated_rng = (
        np.random.default_rng(
            DEFAULT_SEED
        )
    )

    nominal = (
        generate_stress_errors(
            "calibrated_gaussian",
            5000,
            nominal_rng,
        )
    )

    overestimated = (
        generate_stress_errors(
            "covariance_overestimated",
            5000,
            overestimated_rng,
        )
    )

    nominal_trace = float(
        np.trace(
            np.cov(
                nominal,
                rowvar=False,
            )
        )
    )

    overestimated_trace = float(
        np.trace(
            np.cov(
                overestimated,
                rowvar=False,
            )
        )
    )

    assert (
        overestimated_trace
        < nominal_trace
    )

    expected_ratio = (
        OVER_ESTIMATED_STD_SCALE
        ** 2
    )

    empirical_ratio = (
        overestimated_trace
        / nominal_trace
    )

    assert np.isclose(
        empirical_ratio,
        expected_ratio,
        rtol=0.15,
    )


def test_orientation_mismatch_preserves_covariance_eigenvalues() -> None:
    reported = (
        _reported_covariance()
    )

    mismatched = (
        _orientation_mismatch_covariance()
    )

    assert np.allclose(
        np.linalg.eigvalsh(
            reported
        ),
        np.linalg.eigvalsh(
            mismatched
        ),
        atol=1e-14,
    )

    assert not np.allclose(
        reported,
        mismatched,
    )


def test_temporally_correlated_errors_show_positive_lag_one_correlation() -> None:
    errors = (
        generate_stress_errors(
            "temporally_correlated_drift",
            1000,
            np.random.default_rng(
                DEFAULT_SEED
            ),
        )
    )

    first = errors[
        :-1,
        0
    ]

    second = errors[
        1:,
        0
    ]

    correlation = float(
        np.corrcoef(
            first,
            second,
        )[
            0,
            1
        ]
    )

    assert correlation > 0.5

    assert (
        TEMPORAL_CORRELATION
        > 0.5
    )


def test_temporal_drift_changes_mean_error_across_sequence() -> None:
    errors = (
        generate_stress_errors(
            "temporally_correlated_drift",
            1000,
            np.random.default_rng(
                DEFAULT_SEED
            ),
        )
    )

    first_mean = np.mean(
        errors[
            :200
        ],
        axis=0,
    )

    last_mean = np.mean(
        errors[
            -200:
        ],
        axis=0,
    )

    difference = float(
        np.linalg.norm(
            last_mean
            - first_mean
        )
    )

    assert difference > 0.001

    assert MAX_DRIFT > 0.0


def test_heavy_tailed_errors_have_more_extreme_samples_than_gaussian() -> None:
    gaussian = (
        generate_stress_errors(
            "calibrated_gaussian",
            10000,
            np.random.default_rng(
                DEFAULT_SEED
            ),
        )
    )

    heavy = (
        generate_stress_errors(
            "heavy_tailed_contamination",
            10000,
            np.random.default_rng(
                DEFAULT_SEED
                + 1
            ),
        )
    )

    gaussian_norm = np.linalg.norm(
        gaussian,
        axis=1,
    )

    heavy_norm = np.linalg.norm(
        heavy,
        axis=1,
    )

    assert (
        np.quantile(
            heavy_norm,
            0.995,
        )
        >
        np.quantile(
            gaussian_norm,
            0.995,
        )
    )


def test_unknown_stress_condition_is_rejected() -> None:
    with pytest.raises(
        ValueError
    ):
        generate_stress_errors(
            "not_a_real_condition",
            10,
            np.random.default_rng(
                DEFAULT_SEED
            ),
        )