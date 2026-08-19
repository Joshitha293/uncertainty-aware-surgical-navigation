"""Tests for the fixed-view versus generic active-perception benchmark."""

import numpy as np
import pytest

from src.simulation.active_perception_benchmark import (
    BenchmarkConfig,
    BenchmarkSummary,
    compare_strategies,
    make_initial_pose,
    make_observation_model,
    make_occluder,
    make_target,
    run_active_view_benchmark,
    run_fixed_view_benchmark,
)


def test_benchmark_config_defaults_are_valid() -> None:
    config = BenchmarkConfig()

    assert config.trial_count > 0
    assert config.movement_weight >= 0.0
    assert config.uncertainty_weight >= 0.0
    assert config.occlusion_penalty >= 0.0
    assert config.invisibility_penalty >= 0.0


def test_fixed_benchmark_returns_correct_trial_count() -> None:
    config = BenchmarkConfig(
        trial_count=10
    )

    model = make_observation_model()
    target = make_target()
    pose = make_initial_pose(target)
    occluders = (
        make_occluder(),
    )

    result = run_fixed_view_benchmark(
        model=model,
        target=target,
        initial_pose=pose,
        occluders=occluders,
        config=config,
    )

    assert result.trials == 10
    assert result.strategy == "Fixed view"


def test_active_benchmark_returns_correct_trial_count() -> None:
    config = BenchmarkConfig(
        trial_count=10
    )

    model = make_observation_model()
    target = make_target()
    pose = make_initial_pose(target)
    occluders = (
        make_occluder(),
    )

    result = run_active_view_benchmark(
        model=model,
        target=target,
        initial_pose=pose,
        occluders=occluders,
        config=config,
    )

    assert result.trials == 10
    assert (
        result.strategy
        == "Generic active perception"
    )


def test_fixed_view_has_zero_camera_movement() -> None:
    config = BenchmarkConfig(
        trial_count=10
    )

    model = make_observation_model()
    target = make_target()
    pose = make_initial_pose(target)
    occluders = (
        make_occluder(),
    )

    result = run_fixed_view_benchmark(
        model=model,
        target=target,
        initial_pose=pose,
        occluders=occluders,
        config=config,
    )

    assert result.mean_movement == pytest.approx(
        0.0
    )


def test_active_perception_moves_camera() -> None:
    config = BenchmarkConfig(
        trial_count=10
    )

    model = make_observation_model()
    target = make_target()
    pose = make_initial_pose(target)
    occluders = (
        make_occluder(),
    )

    result = run_active_view_benchmark(
        model=model,
        target=target,
        initial_pose=pose,
        occluders=occluders,
        config=config,
    )

    assert result.mean_movement > 0.0


def test_fixed_view_detects_initial_occlusion() -> None:
    config = BenchmarkConfig(
        trial_count=10
    )

    model = make_observation_model()
    target = make_target()
    pose = make_initial_pose(target)
    occluders = (
        make_occluder(),
    )

    result = run_fixed_view_benchmark(
        model=model,
        target=target,
        initial_pose=pose,
        occluders=occluders,
        config=config,
    )

    assert result.occlusion_rate == pytest.approx(
        100.0
    )


def test_active_perception_removes_initial_occlusion() -> None:
    config = BenchmarkConfig(
        trial_count=10
    )

    model = make_observation_model()
    target = make_target()
    pose = make_initial_pose(target)
    occluders = (
        make_occluder(),
    )

    result = run_active_view_benchmark(
        model=model,
        target=target,
        initial_pose=pose,
        occluders=occluders,
        config=config,
    )

    assert result.occlusion_rate == pytest.approx(
        0.0
    )


def test_benchmark_metrics_are_finite() -> None:
    config = BenchmarkConfig(
        trial_count=10
    )

    model = make_observation_model()
    target = make_target()
    pose = make_initial_pose(target)
    occluders = (
        make_occluder(),
    )

    fixed = run_fixed_view_benchmark(
        model=model,
        target=target,
        initial_pose=pose,
        occluders=occluders,
        config=config,
    )

    active = run_active_view_benchmark(
        model=model,
        target=target,
        initial_pose=pose,
        occluders=occluders,
        config=config,
    )

    for result in (
        fixed,
        active,
    ):
        assert np.isfinite(
            result.mean_localisation_error
        )

        assert np.isfinite(
            result.median_localisation_error
        )

        assert np.isfinite(
            result.mean_predicted_sigma
        )

        assert np.isfinite(
            result.mean_movement
        )


def test_active_perception_reduces_localisation_error() -> None:
    config = BenchmarkConfig(
        trial_count=100
    )

    model = make_observation_model()
    target = make_target()
    pose = make_initial_pose(target)
    occluders = (
        make_occluder(),
    )

    fixed = run_fixed_view_benchmark(
        model=model,
        target=target,
        initial_pose=pose,
        occluders=occluders,
        config=config,
    )

    active = run_active_view_benchmark(
        model=model,
        target=target,
        initial_pose=pose,
        occluders=occluders,
        config=config,
    )

    assert (
        active.mean_localisation_error
        < fixed.mean_localisation_error
    )


def test_active_perception_reduces_predicted_uncertainty() -> None:
    config = BenchmarkConfig(
        trial_count=100
    )

    model = make_observation_model()
    target = make_target()
    pose = make_initial_pose(target)
    occluders = (
        make_occluder(),
    )

    fixed = run_fixed_view_benchmark(
        model=model,
        target=target,
        initial_pose=pose,
        occluders=occluders,
        config=config,
    )

    active = run_active_view_benchmark(
        model=model,
        target=target,
        initial_pose=pose,
        occluders=occluders,
        config=config,
    )

    assert (
        active.mean_predicted_sigma
        < fixed.mean_predicted_sigma
    )


def test_compare_strategies_calculates_reduction() -> None:
    fixed = BenchmarkSummary(
        strategy="Fixed view",
        trials=100,
        mean_localisation_error=0.050,
        median_localisation_error=0.048,
        mean_predicted_sigma=0.030,
        mean_movement=0.0,
        occlusion_rate=100.0,
        visibility_rate=100.0,
    )

    active = BenchmarkSummary(
        strategy="Generic active perception",
        trials=100,
        mean_localisation_error=0.005,
        median_localisation_error=0.004,
        mean_predicted_sigma=0.003,
        mean_movement=0.08,
        occlusion_rate=0.0,
        visibility_rate=100.0,
    )

    comparison = compare_strategies(
        fixed=fixed,
        active=active,
    )

    assert comparison.localisation_error_reduction_percent == pytest.approx(
        90.0
    )

    assert comparison.sigma_reduction_percent == pytest.approx(
        90.0
    )

    assert comparison.occlusion_reduction_percentage_points == pytest.approx(
        100.0
    )


def test_compare_strategies_rejects_zero_fixed_error() -> None:
    fixed = BenchmarkSummary(
        strategy="Fixed view",
        trials=10,
        mean_localisation_error=0.0,
        median_localisation_error=0.0,
        mean_predicted_sigma=0.01,
        mean_movement=0.0,
        occlusion_rate=0.0,
        visibility_rate=100.0,
    )

    active = BenchmarkSummary(
        strategy="Generic active perception",
        trials=10,
        mean_localisation_error=0.001,
        median_localisation_error=0.001,
        mean_predicted_sigma=0.001,
        mean_movement=0.05,
        occlusion_rate=0.0,
        visibility_rate=100.0,
    )

    with pytest.raises(ValueError):
        compare_strategies(
            fixed=fixed,
            active=active,
        )


def test_compare_strategies_rejects_zero_fixed_sigma() -> None:
    fixed = BenchmarkSummary(
        strategy="Fixed view",
        trials=10,
        mean_localisation_error=0.01,
        median_localisation_error=0.01,
        mean_predicted_sigma=0.0,
        mean_movement=0.0,
        occlusion_rate=0.0,
        visibility_rate=100.0,
    )

    active = BenchmarkSummary(
        strategy="Generic active perception",
        trials=10,
        mean_localisation_error=0.001,
        median_localisation_error=0.001,
        mean_predicted_sigma=0.001,
        mean_movement=0.05,
        occlusion_rate=0.0,
        visibility_rate=100.0,
    )

    with pytest.raises(ValueError):
        compare_strategies(
            fixed=fixed,
            active=active,
        )