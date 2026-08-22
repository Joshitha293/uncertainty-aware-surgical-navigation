"""Tests for the surgical-navigation safety gate."""

from surgical_navigation_ros.safety_gate_node import (
    validate_navigation_command,
)


def make_command(uncertainty: float) -> dict:
    """Create a valid navigation command for testing."""

    return {
        "command": "MOVE_CAMERA_TO_VIEWPOINT",
        "position": [0.12, 0.10, 0.12],
        "candidate_count": 6,
        "task_relevance": 0.95,
        "task_alignment": 0.99,
        "task_uncertainty_m": uncertainty,
        "task_aware_score": 100.0,
        "generic_score": -1.0,
    }


def test_low_uncertainty_is_accepted() -> None:
    """A 1.904 mm uncertainty should be accepted."""

    command = make_command(0.001904)

    is_safe, reason = validate_navigation_command(command)

    assert is_safe is True
    assert reason == "Navigation command is safe."


def test_high_uncertainty_is_rejected() -> None:
    """A 50 mm uncertainty must be rejected."""

    command = make_command(0.050)

    is_safe, reason = validate_navigation_command(command)

    assert is_safe is False
    assert reason == "Perception uncertainty exceeds 0.030 m."


def test_threshold_value_is_accepted() -> None:
    """A value exactly at the 30 mm threshold is accepted."""

    command = make_command(0.030)

    is_safe, reason = validate_navigation_command(command)

    assert is_safe is True
    assert reason == "Navigation command is safe."


def test_position_must_have_three_values() -> None:
    """Invalid position dimensionality must be rejected."""

    command = make_command(0.001904)
    command["position"] = [0.12, 0.10]

    is_safe, reason = validate_navigation_command(command)

    assert is_safe is False
    assert reason == "Position must contain three values."


def test_non_finite_uncertainty_is_rejected() -> None:
    """NaN uncertainty must be rejected."""

    command = make_command(float("nan"))

    is_safe, reason = validate_navigation_command(command)

    assert is_safe is False
    assert reason == "Uncertainty is non-finite."


def test_unsupported_command_is_rejected() -> None:
    """Unsupported navigation commands must be rejected."""

    command = make_command(0.001904)
    command["command"] = "INVALID_COMMAND"

    is_safe, reason = validate_navigation_command(command)

    assert is_safe is False
    assert reason == "Unsupported navigation command."