"""Tests for the final Phase 7 autonomous safety benchmark."""

from __future__ import annotations

from pathlib import Path

from src.robotics.autonomous_execution_controller import (
    AutonomousControllerAction,
)
from src.robotics.safety_state_machine import (
    SafetyState,
)
from src.simulation.phase7_autonomous_safety_benchmark import (
    benchmark_scenarios,
    run_benchmark,
    run_episode,
)


def _scenario(
    name: str,
):
    """Return one frozen scenario by name."""

    matches = [
        scenario
        for scenario in benchmark_scenarios()
        if scenario.name == name
    ]

    assert len(
        matches
    ) == 1

    return matches[
        0
    ]


def test_final_benchmark_contains_nominal_and_fault_cases() -> None:
    """Final benchmark must contain nominal plus diverse fault cases."""

    names = {
        scenario.name
        for scenario in benchmark_scenarios()
    }

    assert "nominal" in names
    assert "temporary_stale_perception" in names
    assert "critical_uncertainty" in names
    assert "recoverable_clearance" in names
    assert "joint_limit_violation" in names
    assert "recoverable_tracking_error" in names
    assert "invalid_runtime_data" in names
    assert "execution_timeout" in names
    assert "replanning_failure" in names


def test_nominal_episode_completes_without_false_alarm() -> None:
    """Nominal operation must not produce unnecessary intervention."""

    outcome = run_episode(
        _scenario(
            "nominal"
        ),
        repetition=0,
        seed=123,
    )

    assert outcome.completed
    assert not outcome.false_alarm
    assert not outcome.unsafe_continuation

    assert (
        outcome.terminal_state
        == SafetyState.COMPLETED
    )


def test_temporary_stale_perception_recovers() -> None:
    """Temporary stale perception should reacquire and finish."""

    outcome = run_episode(
        _scenario(
            "temporary_stale_perception"
        ),
        repetition=0,
        seed=123,
    )

    assert outcome.detected
    assert outcome.correct_response
    assert outcome.completed
    assert outcome.recovery_success

    assert (
        outcome.first_fault_action
        == AutonomousControllerAction.REACQUIRE
    )


def test_persistent_perception_failure_stops() -> None:
    """Unsuccessful perception recovery must stop."""

    outcome = run_episode(
        _scenario(
            "persistent_perception_recovery_failure"
        ),
        repetition=0,
        seed=123,
    )

    assert outcome.detected
    assert outcome.correct_response

    assert (
        outcome.terminal_state
        == SafetyState.STOPPED
    )

    assert not outcome.unsafe_continuation


def test_recoverable_clearance_replans_and_completes() -> None:
    """Recoverable clearance concern should replan."""

    outcome = run_episode(
        _scenario(
            "recoverable_clearance"
        ),
        repetition=0,
        seed=456,
    )

    assert outcome.detected
    assert outcome.correct_response
    assert outcome.completed

    assert (
        outcome.first_fault_action
        == AutonomousControllerAction.REPLAN
    )


def test_recoverable_tracking_error_recovers_and_completes() -> None:
    """Recoverable tracking fault should recover."""

    outcome = run_episode(
        _scenario(
            "recoverable_tracking_error"
        ),
        repetition=0,
        seed=456,
    )

    assert outcome.detected
    assert outcome.correct_response
    assert outcome.completed

    assert (
        outcome.first_fault_action
        == AutonomousControllerAction.RECOVER
    )


def test_critical_uncertainty_stops() -> None:
    """Critical uncertainty must terminate execution."""

    outcome = run_episode(
        _scenario(
            "critical_uncertainty"
        ),
        repetition=0,
        seed=789,
    )

    assert outcome.detected
    assert outcome.correct_response

    assert (
        outcome.terminal_state
        == SafetyState.STOPPED
    )

    assert not outcome.unsafe_continuation


def test_joint_limit_violation_stops() -> None:
    """Hard joint-limit fault must terminate execution."""

    outcome = run_episode(
        _scenario(
            "joint_limit_violation"
        ),
        repetition=0,
        seed=789,
    )

    assert outcome.detected
    assert outcome.correct_response

    assert (
        outcome.terminal_state
        == SafetyState.STOPPED
    )


def test_invalid_runtime_data_stops() -> None:
    """NaN runtime state must terminate execution."""

    outcome = run_episode(
        _scenario(
            "invalid_runtime_data"
        ),
        repetition=0,
        seed=789,
    )

    assert outcome.detected

    assert (
        outcome.terminal_state
        == SafetyState.STOPPED
    )


def test_replanning_failure_enters_failed() -> None:
    """Planner failure during recovery must enter FAILED."""

    outcome = run_episode(
        _scenario(
            "replanning_failure"
        ),
        repetition=0,
        seed=222,
    )

    assert outcome.detected
    assert outcome.correct_response

    assert (
        outcome.terminal_state
        == SafetyState.FAILED
    )


def test_fault_response_latency_is_zero_at_threshold_crossing() -> None:
    """Controller should respond in the same cycle that fault is presented."""

    outcome = run_episode(
        _scenario(
            "critical_tracking_error"
        ),
        repetition=0,
        seed=333,
    )

    assert (
        outcome.response_latency_steps
        == 0
    )


def test_episode_is_deterministic_for_fixed_seed() -> None:
    """Same frozen scenario and seed must reproduce outcome."""

    scenario = _scenario(
        "temporary_stale_perception"
    )

    first = run_episode(
        scenario,
        repetition=0,
        seed=999,
    )

    second = run_episode(
        scenario,
        repetition=0,
        seed=999,
    )

    assert (
        first
        == second
    )


def test_small_benchmark_writes_artifacts(
    tmp_path: Path,
) -> None:
    """Benchmark must generate machine-readable CSV and JSON evidence."""

    summary = run_benchmark(
        repetitions=2,
        seed=1234,
        output_directory=tmp_path,
    )

    assert (
        summary[
            "repetitions_per_scenario"
        ]
        == 2
    )

    assert (
        summary[
            "total_episodes"
        ]
        == (
            2
            * len(
                benchmark_scenarios()
            )
        )
    )

    assert (
        tmp_path
        / "phase7_autonomous_safety_trials.csv"
    ).exists()

    assert (
        tmp_path
        / "phase7_autonomous_safety_summary.json"
    ).exists()


def test_small_benchmark_nominal_false_alarm_rate_is_zero(
    tmp_path: Path,
) -> None:
    """Nominal episodes should remain free from false alarms."""

    summary = run_benchmark(
        repetitions=2,
        seed=5678,
        output_directory=tmp_path,
    )

    rate = (
        summary[
            "scenarios"
        ][
            "nominal"
        ][
            "false_alarm"
        ][
            "rate"
        ]
    )

    assert (
        rate
        == 0.0
    )


def test_small_benchmark_has_no_unsafe_continuation(
    tmp_path: Path,
) -> None:
    """Frozen verification scenarios should not silently continue faults."""

    summary = run_benchmark(
        repetitions=2,
        seed=9012,
        output_directory=tmp_path,
    )

    for (
        name,
        metrics,
    ) in (
        summary[
            "scenarios"
        ].items()
    ):
        assert (
            metrics[
                "unsafe_continuation"
            ][
                "rate"
            ]
            == 0.0
        ), name