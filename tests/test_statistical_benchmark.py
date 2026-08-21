"""Tests for the statistical benchmark integration."""

from src.simulation.statistical_benchmark import (
    convert_trial_result,
)


def test_failed_trial_is_marked_unsafe() -> None:
    from src.simulation.uncertainty_benchmark import (
        TrialResult,
    )

    result = TrialResult(
        trial=0,
        method="Test",
        planning_success=False,
        planning_time=0.0,
        iterations=0,
        path_cost=float("inf"),
        minimum_true_safety_clearance=float("-inf"),
        collision_against_truth=False,
        safety_violation_against_truth=False,
        maximum_rcm_error=float("nan"),
    )

    converted = convert_trial_result(
        result
    )

    assert converted.planning_success is False
    assert converted.safe is False


def test_safe_trial_is_marked_safe() -> None:
    from src.simulation.uncertainty_benchmark import (
        TrialResult,
    )

    result = TrialResult(
        trial=0,
        method="Test",
        planning_success=True,
        planning_time=0.1,
        iterations=100,
        path_cost=1.0,
        minimum_true_safety_clearance=0.01,
        collision_against_truth=False,
        safety_violation_against_truth=False,
        maximum_rcm_error=0.0,
    )

    converted = convert_trial_result(
        result
    )

    assert converted.planning_success is True
    assert converted.safe is True


def test_collision_is_marked_unsafe() -> None:
    from src.simulation.uncertainty_benchmark import (
        TrialResult,
    )

    result = TrialResult(
        trial=0,
        method="Test",
        planning_success=True,
        planning_time=0.1,
        iterations=100,
        path_cost=1.0,
        minimum_true_safety_clearance=-0.001,
        collision_against_truth=True,
        safety_violation_against_truth=False,
        maximum_rcm_error=0.0,
    )

    converted = convert_trial_result(
        result
    )

    assert converted.safe is False


def test_safety_violation_is_marked_unsafe() -> None:
    from src.simulation.uncertainty_benchmark import (
        TrialResult,
    )

    result = TrialResult(
        trial=0,
        method="Test",
        planning_success=True,
        planning_time=0.1,
        iterations=100,
        path_cost=1.0,
        minimum_true_safety_clearance=0.001,
        collision_against_truth=False,
        safety_violation_against_truth=True,
        maximum_rcm_error=0.0,
    )

    converted = convert_trial_result(
        result
    )

    assert converted.safe is False