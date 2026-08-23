"""Tests for the supplementary uncertainty-calibration benchmark."""

import json

import numpy as np
import pytest

from src.simulation.three_strategy_robustness_benchmark import (
    default_scenarios,
)
from src.simulation.uncertainty_calibration_benchmark import (
    CHI_SQUARE_3_THRESHOLDS,
    CalibrationConfig,
    EXPECTED_MEAN_NEES,
    EXPECTED_MEAN_RADIAL_ERROR_OVER_SIGMA,
    _wilson_interval,
    analyse_calibration,
    run_calibration_trials,
    run_uncertainty_calibration,
    save_calibration_outputs,
)


def small_config():
    return CalibrationConfig(
        samples_per_viewpoint=5,
        candidate_viewpoints_per_scenario=2,
        bootstrap_samples=30,
        bootstrap_seed=123,
        calibration_bin_count=3,
    )


def small_scenarios():
    scenarios = (
        default_scenarios()
    )

    return (
        scenarios[0],
        scenarios[1],
    )


def test_calibration_constants_match_three_dimensional_model():
    assert (
        EXPECTED_MEAN_NEES
        == pytest.approx(
            3.0
        )
    )

    assert (
        EXPECTED_MEAN_RADIAL_ERROR_OVER_SIGMA
        == pytest.approx(
            1.5957691216,
            rel=1e-6,
        )
    )


def test_chi_square_thresholds_are_ordered():
    assert (
        CHI_SQUARE_3_THRESHOLDS[
            0.50
        ]
        < CHI_SQUARE_3_THRESHOLDS[
            0.90
        ]
        < CHI_SQUARE_3_THRESHOLDS[
            0.95
        ]
        < CHI_SQUARE_3_THRESHOLDS[
            0.99
        ]
    )


def test_invalid_configuration_is_rejected():
    with pytest.raises(
        ValueError,
        match="samples_per_viewpoint",
    ):
        CalibrationConfig(
            samples_per_viewpoint=0
        )

    with pytest.raises(
        ValueError,
        match="candidate_viewpoints",
    ):
        CalibrationConfig(
            candidate_viewpoints_per_scenario=0
        )


def test_wilson_interval_contains_observed_fraction():
    low, high = (
        _wilson_interval(
            successes=95,
            total=100,
        )
    )

    assert (
        0.0
        <= low
        <= 0.95
        <= high
        <= 1.0
    )


def test_small_calibration_run_produces_finite_statistics():
    config = (
        small_config()
    )

    records = (
        run_calibration_trials(
            config,
            scenarios=small_scenarios(),
        )
    )

    expected_conditions = (
        len(
            small_scenarios()
        )
        * 3
    )

    assert len(
        records
    ) == (
        expected_conditions
        * config.samples_per_viewpoint
    )

    for record in records:
        assert (
            record.predicted_sigma
            > 0.0
        )

        assert np.isfinite(
            record.normalized_squared_error
        )

        assert (
            record.normalized_squared_error
            >= 0.0
        )


def test_calibration_analysis_builds_bins_and_summary():
    config = (
        small_config()
    )

    records = (
        run_calibration_trials(
            config,
            scenarios=small_scenarios(),
        )
    )

    bins, summary = (
        analyse_calibration(
            records=records,
            config=config,
        )
    )

    assert len(
        bins
    ) == 3

    assert (
        summary.record_count
        == len(
            records
        )
    )

    assert (
        summary.mean_nees.mean
        > 0.0
    )

    assert (
        0.0
        <= summary.coverage_95.mean
        <= 1.0
    )

    assert np.isfinite(
        summary.calibration_slope
    )


def test_complete_calibration_outputs_are_saved(
    tmp_path,
):
    result = (
        run_uncertainty_calibration(
            small_config(),
            scenarios=small_scenarios(),
        )
    )

    raw_path, summary_path = (
        save_calibration_outputs(
            result,
            output_directory=tmp_path,
        )
    )

    assert raw_path.exists()
    assert summary_path.exists()

    payload = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        payload[
            "chi_square_df"
        ]
        == 3
    )

    assert (
        payload[
            "summary"
        ][
            "record_count"
        ]
        == len(
            result.records
        )
    )

    assert len(
        payload[
            "calibration_bins"
        ]
    ) == 3