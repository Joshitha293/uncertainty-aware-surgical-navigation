"""Integrity tests for the saved Day 7 statistical results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_PATH = (
    Path("results")
    / "day7_statistical_trials.csv"
)


def load_results() -> pd.DataFrame:
    """Load the saved Day 7 results."""

    assert RESULTS_PATH.exists(), (
        "Day 7 statistical results CSV "
        "does not exist."
    )

    data = pd.read_csv(
        RESULTS_PATH
    )

    return data


def test_exactly_100_trials_are_saved() -> None:
    data = load_results()

    assert len(data) == 100


def test_trial_ids_are_unique_and_complete() -> None:
    data = load_results()

    trials = data["trial"].to_numpy()

    assert len(np.unique(trials)) == 100

    assert set(trials) == set(
        range(100)
    )


def test_clearance_values_are_finite() -> None:
    data = load_results()

    generic = data[
        "generic_minimum_safety_clearance_m"
    ].to_numpy()

    task_aware = data[
        "task_aware_minimum_safety_clearance_m"
    ].to_numpy()

    assert np.all(
        np.isfinite(generic)
    )

    assert np.all(
        np.isfinite(task_aware)
    )


def test_paired_difference_is_correct() -> None:
    data = load_results()

    generic = data[
        "generic_minimum_safety_clearance_m"
    ].to_numpy()

    task_aware = data[
        "task_aware_minimum_safety_clearance_m"
    ].to_numpy()

    recorded = data[
        "clearance_difference_m"
    ].to_numpy()

    expected = (
        task_aware
        - generic
    )

    assert np.allclose(
        recorded,
        expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_expected_columns_are_present() -> None:
    data = load_results()

    required = {
        "trial",
        "generic_planning_success",
        "generic_safe",
        "generic_collision",
        "generic_safety_margin_violation",
        "generic_minimum_safety_clearance_m",
        "generic_path_cost",
        "task_aware_planning_success",
        "task_aware_safe",
        "task_aware_collision",
        "task_aware_safety_margin_violation",
        "task_aware_minimum_safety_clearance_m",
        "task_aware_path_cost",
        "clearance_difference_m",
    }

    assert required.issubset(
        set(data.columns)
    )


def test_both_methods_have_100_trials() -> None:
    data = load_results()

    assert len(
        data[
            "generic_minimum_safety_clearance_m"
        ]
    ) == 100

    assert len(
        data[
            "task_aware_minimum_safety_clearance_m"
        ]
    ) == 100


def test_safe_rates_match_validated_result() -> None:
    data = load_results()

    generic_safe_rate = (
        100.0
        * data["generic_safe"]
        .astype(bool)
        .mean()
    )

    task_aware_safe_rate = (
        100.0
        * data["task_aware_safe"]
        .astype(bool)
        .mean()
    )

    assert generic_safe_rate == 38.0

    assert task_aware_safe_rate == 98.0


def test_mean_clearance_difference_matches_result() -> None:
    data = load_results()

    mean_difference_mm = (
        data["clearance_difference_m"]
        .mean()
        * 1000.0
    )

    assert np.isclose(
        mean_difference_mm,
        9.863,
        atol=0.001,
    )