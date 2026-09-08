"""Tests for Phase 7 runtime safety monitors."""

from __future__ import annotations

import numpy as np
import pytest

from src.robotics.runtime_safety_monitors import (
    RuntimeSafetyConfig,
    RuntimeSafetySnapshot,
    check_clearance,
    check_execution_timeout,
    check_joint_limits,
    check_perception_freshness,
    check_tracking_error,
    check_uncertainty,
    evaluate_runtime_safety,
    highest_severity_event,
)
from src.robotics.safety_state_machine import (
    SafetyHazard,
    SafetySeverity,
)


def _safe_snapshot(
    **overrides,
) -> RuntimeSafetySnapshot:
    """Return a nominally safe runtime snapshot."""

    values = {
        "step_index": 10,
        "latest_perception_step": 10,
        "maximum_principal_sigma": 0.005,
        "predicted_clearance": 0.010,
        "joint_positions": np.asarray(
            [
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=float,
        ),
        "joint_lower_limits": np.asarray(
            [
                -1.0,
                -1.0,
                -1.0,
                -1.0,
            ],
            dtype=float,
        ),
        "joint_upper_limits": np.asarray(
            [
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype=float,
        ),
        "tracking_error": 0.001,
        "execution_steps": 10,
    }

    values.update(
        overrides
    )

    return RuntimeSafetySnapshot(
        **values
    )


def test_safe_snapshot_produces_no_events() -> None:
    """Nominal runtime state should produce no safety event."""

    events = (
        evaluate_runtime_safety(
            _safe_snapshot()
        )
    )

    assert events == ()


def test_stale_perception_detected() -> None:
    """Old perception should request recovery."""

    snapshot = (
        _safe_snapshot(
            step_index=10,
            latest_perception_step=6,
        )
    )

    event = (
        check_perception_freshness(
            snapshot,
            RuntimeSafetyConfig(),
        )
    )

    assert event is not None

    assert (
        event.hazard
        == SafetyHazard.STALE_PERCEPTION
    )

    assert (
        event.severity
        == SafetySeverity.RECOVERABLE
    )


def test_perception_age_at_limit_is_accepted() -> None:
    """Freshness threshold is exceeded only beyond configured age."""

    config = (
        RuntimeSafetyConfig(
            max_perception_age_steps=3
        )
    )

    snapshot = (
        _safe_snapshot(
            step_index=10,
            latest_perception_step=7,
        )
    )

    assert (
        check_perception_freshness(
            snapshot,
            config,
        )
        is None
    )


def test_uncertainty_recovery_threshold() -> None:
    """Moderately high uncertainty should request reacquisition."""

    snapshot = (
        _safe_snapshot(
            maximum_principal_sigma=0.015,
        )
    )

    event = (
        check_uncertainty(
            snapshot,
            RuntimeSafetyConfig(),
        )
    )

    assert event is not None

    assert (
        event.hazard
        == SafetyHazard.EXCESSIVE_UNCERTAINTY
    )

    assert (
        event.severity
        == SafetySeverity.RECOVERABLE
    )


def test_uncertainty_stop_threshold() -> None:
    """Extreme uncertainty must be critical."""

    snapshot = (
        _safe_snapshot(
            maximum_principal_sigma=0.030,
        )
    )

    event = (
        check_uncertainty(
            snapshot,
            RuntimeSafetyConfig(),
        )
    )

    assert event is not None

    assert (
        event.severity
        == SafetySeverity.CRITICAL
    )


def test_clearance_recovery_region() -> None:
    """Small positive clearance should request recovery."""

    snapshot = (
        _safe_snapshot(
            predicted_clearance=0.001,
        )
    )

    event = (
        check_clearance(
            snapshot,
            RuntimeSafetyConfig(),
        )
    )

    assert event is not None

    assert (
        event.severity
        == SafetySeverity.RECOVERABLE
    )


def test_clearance_violation_is_critical() -> None:
    """Zero or negative protected clearance must be critical."""

    snapshot = (
        _safe_snapshot(
            predicted_clearance=-0.001,
        )
    )

    event = (
        check_clearance(
            snapshot,
            RuntimeSafetyConfig(),
        )
    )

    assert event is not None

    assert (
        event.hazard
        == SafetyHazard.CLEARANCE_VIOLATION
    )

    assert (
        event.severity
        == SafetySeverity.CRITICAL
    )


def test_joint_limit_approach_detected() -> None:
    """Near-limit configuration should request recovery."""

    snapshot = (
        _safe_snapshot(
            joint_positions=np.asarray(
                [
                    0.97,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=float,
            )
        )
    )

    event = (
        check_joint_limits(
            snapshot,
            RuntimeSafetyConfig(),
        )
    )

    assert event is not None

    assert (
        event.hazard
        == SafetyHazard.JOINT_LIMIT_APPROACH
    )

    assert (
        event.severity
        == SafetySeverity.RECOVERABLE
    )


def test_joint_limit_violation_is_critical() -> None:
    """Joint position outside hard limits must be critical."""

    snapshot = (
        _safe_snapshot(
            joint_positions=np.asarray(
                [
                    1.01,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=float,
            )
        )
    )

    event = (
        check_joint_limits(
            snapshot,
            RuntimeSafetyConfig(),
        )
    )

    assert event is not None

    assert (
        event.severity
        == SafetySeverity.CRITICAL
    )


def test_tracking_error_recovery_threshold() -> None:
    """Moderate tracking deviation should request recovery."""

    snapshot = (
        _safe_snapshot(
            tracking_error=0.005,
        )
    )

    event = (
        check_tracking_error(
            snapshot,
            RuntimeSafetyConfig(),
        )
    )

    assert event is not None

    assert (
        event.severity
        == SafetySeverity.RECOVERABLE
    )


def test_tracking_error_stop_threshold() -> None:
    """Large tracking error must be critical."""

    snapshot = (
        _safe_snapshot(
            tracking_error=0.015,
        )
    )

    event = (
        check_tracking_error(
            snapshot,
            RuntimeSafetyConfig(),
        )
    )

    assert event is not None

    assert (
        event.severity
        == SafetySeverity.CRITICAL
    )


def test_execution_timeout_detected() -> None:
    """Maximum execution-step budget must be enforced."""

    snapshot = (
        _safe_snapshot(
            execution_steps=60,
        )
    )

    event = (
        check_execution_timeout(
            snapshot,
            RuntimeSafetyConfig(),
        )
    )

    assert event is not None

    assert (
        event.hazard
        == SafetyHazard.EXECUTION_TIMEOUT
    )

    assert (
        event.severity
        == SafetySeverity.CRITICAL
    )


def test_invalid_nan_data_is_critical() -> None:
    """NaN runtime data must fail safely."""

    snapshot = (
        _safe_snapshot(
            tracking_error=float(
                "nan"
            )
        )
    )

    events = (
        evaluate_runtime_safety(
            snapshot
        )
    )

    assert len(
        events
    ) == 1

    assert (
        events[
            0
        ].hazard
        == SafetyHazard.INVALID_DATA
    )

    assert (
        events[
            0
        ].severity
        == SafetySeverity.CRITICAL
    )


def test_invalid_joint_limits_are_detected() -> None:
    """Lower limit >= upper limit must be rejected at runtime."""

    snapshot = (
        _safe_snapshot(
            joint_lower_limits=np.asarray(
                [
                    -1.0,
                    1.0,
                    -1.0,
                    -1.0,
                ],
                dtype=float,
            ),
            joint_upper_limits=np.asarray(
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                dtype=float,
            ),
        )
    )

    events = (
        evaluate_runtime_safety(
            snapshot
        )
    )

    assert len(
        events
    ) == 1

    assert (
        events[
            0
        ].hazard
        == SafetyHazard.INVALID_DATA
    )


def test_negative_uncertainty_is_invalid() -> None:
    """Negative standard deviation is invalid data."""

    snapshot = (
        _safe_snapshot(
            maximum_principal_sigma=-0.001,
        )
    )

    events = (
        evaluate_runtime_safety(
            snapshot
        )
    )

    assert len(
        events
    ) == 1

    assert (
        events[
            0
        ].hazard
        == SafetyHazard.INVALID_DATA
    )


def test_multiple_recoverable_events_are_preserved() -> None:
    """Independent monitors should return all simultaneous hazards."""

    snapshot = (
        _safe_snapshot(
            step_index=10,
            latest_perception_step=5,
            maximum_principal_sigma=0.020,
            tracking_error=0.007,
        )
    )

    events = (
        evaluate_runtime_safety(
            snapshot
        )
    )

    hazards = {
        event.hazard
        for event in events
    }

    assert (
        SafetyHazard.STALE_PERCEPTION
        in hazards
    )

    assert (
        SafetyHazard.EXCESSIVE_UNCERTAINTY
        in hazards
    )

    assert (
        SafetyHazard.TRACKING_ERROR
        in hazards
    )


def test_highest_severity_event_prefers_critical() -> None:
    """Critical hazard must dominate recoverable hazards."""

    snapshot = (
        _safe_snapshot(
            step_index=10,
            latest_perception_step=5,
            tracking_error=0.020,
        )
    )

    events = (
        evaluate_runtime_safety(
            snapshot
        )
    )

    event = (
        highest_severity_event(
            events
        )
    )

    assert event is not None

    assert (
        event.severity
        == SafetySeverity.CRITICAL
    )

    assert (
        event.hazard
        == SafetyHazard.TRACKING_ERROR
    )


def test_empty_event_collection_has_no_highest_event() -> None:
    """No active hazards should return None."""

    assert (
        highest_severity_event(
            ()
        )
        is None
    )


def test_future_perception_timestamp_rejected() -> None:
    """Perception timestamps cannot come from a future execution step."""

    with pytest.raises(
        ValueError
    ):
        _safe_snapshot(
            step_index=5,
            latest_perception_step=6,
        )


def test_joint_array_shapes_must_match() -> None:
    """Joint-state and joint-limit vectors require identical shapes."""

    with pytest.raises(
        ValueError
    ):
        _safe_snapshot(
            joint_lower_limits=np.asarray(
                [
                    -1.0,
                    -1.0,
                ],
                dtype=float,
            )
        )


def test_configuration_requires_ordered_uncertainty_thresholds() -> None:
    """Recovery uncertainty threshold must precede stop threshold."""

    with pytest.raises(
        ValueError
    ):
        RuntimeSafetyConfig(
            uncertainty_reacquire_sigma=0.030,
            uncertainty_stop_sigma=0.020,
        )


def test_configuration_requires_ordered_tracking_thresholds() -> None:
    """Tracking recovery threshold must precede stop threshold."""

    with pytest.raises(
        ValueError
    ):
        RuntimeSafetyConfig(
            tracking_error_recovery_threshold=0.020,
            tracking_error_stop_threshold=0.010,
        )