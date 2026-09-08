"""Final Phase 7 autonomous safety and fault-response benchmark.

This benchmark evaluates the deterministic Phase 7 safety supervisor under
matched synthetic runtime fault scenarios.

Measured outcomes include:

- fault detection;
- correct supervisory response;
- response latency in controller update cycles;
- successful recovery;
- correct fail-safe termination;
- false alarms during nominal operation;
- unsafe continuation;
- autonomous task completion;
- replanning, reacquisition and recovery counts.

The benchmark evaluates software behaviour in engineered simulation cases.
It does not estimate clinical hazard probabilities or establish surgical,
medical-device, or patient safety.
"""

from __future__ import annotations

import csv
import json

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.robotics.autonomous_execution_controller import (
    AutonomousControllerAction,
    AutonomousExecutionController,
)
from src.robotics.runtime_safety_monitors import (
    RuntimeSafetyConfig,
)
from src.robotics.safety_state_machine import (
    SafetyState,
)
from src.simulation.phase7_fault_injection import (
    FaultType,
    inject_fault,
    nominal_runtime_snapshot,
)


DEFAULT_REPETITIONS = 30
DEFAULT_SEED = 71001

EPISODE_STEPS = 30

RESULT_DIRECTORY = Path(
    "results/phase7_autonomous_safety"
)


@dataclass(frozen=True)
class SafetyBenchmarkScenario:
    """Definition of one frozen Phase 7 verification scenario."""

    name: str

    fault: FaultType | None

    expected_action: AutonomousControllerAction

    expected_terminal_state: SafetyState

    should_complete: bool

    reacquisition_succeeds: bool = True

    recovery_succeeds: bool = True

    planner_fails_after_initialisation: bool = False


@dataclass(frozen=True)
class SafetyEpisodeOutcome:
    """Outcome from one simulated autonomous safety episode."""

    scenario: str

    repetition: int

    seed: int

    fault_onset_step: int | None

    detected: bool

    correct_response: bool

    response_latency_steps: int | None

    completed: bool

    terminal_state: SafetyState

    recovery_success: bool

    appropriate_terminal_response: bool

    false_alarm: bool

    unsafe_continuation: bool

    first_fault_action: AutonomousControllerAction | None

    replans: int

    reacquisitions: int

    recoveries: int

    stops: int

    failures: int


def benchmark_scenarios() -> tuple[
    SafetyBenchmarkScenario,
    ...,
]:
    """Return frozen Phase 7 final verification scenarios."""

    return (
        SafetyBenchmarkScenario(
            name="nominal",
            fault=None,
            expected_action=(
                AutonomousControllerAction.CONTINUE
            ),
            expected_terminal_state=(
                SafetyState.COMPLETED
            ),
            should_complete=True,
        ),

        SafetyBenchmarkScenario(
            name="temporary_stale_perception",
            fault=(
                FaultType.STALE_PERCEPTION
            ),
            expected_action=(
                AutonomousControllerAction.REACQUIRE
            ),
            expected_terminal_state=(
                SafetyState.COMPLETED
            ),
            should_complete=True,
        ),

        SafetyBenchmarkScenario(
            name="persistent_perception_recovery_failure",
            fault=(
                FaultType.STALE_PERCEPTION
            ),
            expected_action=(
                AutonomousControllerAction.STOP
            ),
            expected_terminal_state=(
                SafetyState.STOPPED
            ),
            should_complete=False,
            reacquisition_succeeds=False,
        ),

        SafetyBenchmarkScenario(
            name="recoverable_uncertainty",
            fault=(
                FaultType.UNCERTAINTY_RECOVERABLE
            ),
            expected_action=(
                AutonomousControllerAction.REACQUIRE
            ),
            expected_terminal_state=(
                SafetyState.COMPLETED
            ),
            should_complete=True,
        ),

        SafetyBenchmarkScenario(
            name="critical_uncertainty",
            fault=(
                FaultType.UNCERTAINTY_CRITICAL
            ),
            expected_action=(
                AutonomousControllerAction.STOP
            ),
            expected_terminal_state=(
                SafetyState.STOPPED
            ),
            should_complete=False,
        ),

        SafetyBenchmarkScenario(
            name="recoverable_clearance",
            fault=(
                FaultType.CLEARANCE_RECOVERABLE
            ),
            expected_action=(
                AutonomousControllerAction.REPLAN
            ),
            expected_terminal_state=(
                SafetyState.COMPLETED
            ),
            should_complete=True,
        ),

        SafetyBenchmarkScenario(
            name="critical_clearance",
            fault=(
                FaultType.CLEARANCE_CRITICAL
            ),
            expected_action=(
                AutonomousControllerAction.STOP
            ),
            expected_terminal_state=(
                SafetyState.STOPPED
            ),
            should_complete=False,
        ),

        SafetyBenchmarkScenario(
            name="recoverable_joint_limit_approach",
            fault=(
                FaultType.JOINT_LIMIT_APPROACH
            ),
            expected_action=(
                AutonomousControllerAction.RECOVER
            ),
            expected_terminal_state=(
                SafetyState.COMPLETED
            ),
            should_complete=True,
        ),

        SafetyBenchmarkScenario(
            name="joint_limit_violation",
            fault=(
                FaultType.JOINT_LIMIT_VIOLATION
            ),
            expected_action=(
                AutonomousControllerAction.STOP
            ),
            expected_terminal_state=(
                SafetyState.STOPPED
            ),
            should_complete=False,
        ),

        SafetyBenchmarkScenario(
            name="recoverable_tracking_error",
            fault=(
                FaultType.TRACKING_ERROR_RECOVERABLE
            ),
            expected_action=(
                AutonomousControllerAction.RECOVER
            ),
            expected_terminal_state=(
                SafetyState.COMPLETED
            ),
            should_complete=True,
        ),

        SafetyBenchmarkScenario(
            name="critical_tracking_error",
            fault=(
                FaultType.TRACKING_ERROR_CRITICAL
            ),
            expected_action=(
                AutonomousControllerAction.STOP
            ),
            expected_terminal_state=(
                SafetyState.STOPPED
            ),
            should_complete=False,
        ),

        SafetyBenchmarkScenario(
            name="invalid_runtime_data",
            fault=(
                FaultType.INVALID_NUMERICAL_DATA
            ),
            expected_action=(
                AutonomousControllerAction.STOP
            ),
            expected_terminal_state=(
                SafetyState.STOPPED
            ),
            should_complete=False,
        ),

        SafetyBenchmarkScenario(
            name="execution_timeout",
            fault=(
                FaultType.EXECUTION_TIMEOUT
            ),
            expected_action=(
                AutonomousControllerAction.STOP
            ),
            expected_terminal_state=(
                SafetyState.STOPPED
            ),
            should_complete=False,
        ),

        SafetyBenchmarkScenario(
            name="replanning_failure",
            fault=(
                FaultType.CLEARANCE_RECOVERABLE
            ),
            expected_action=(
                AutonomousControllerAction.FAIL
            ),
            expected_terminal_state=(
                SafetyState.FAILED
            ),
            should_complete=False,
            planner_fails_after_initialisation=True,
        ),

        SafetyBenchmarkScenario(
            name="persistent_execution_recovery_failure",
            fault=(
                FaultType.TRACKING_ERROR_RECOVERABLE
            ),
            expected_action=(
                AutonomousControllerAction.STOP
            ),
            expected_terminal_state=(
                SafetyState.STOPPED
            ),
            should_complete=False,
            recovery_succeeds=False,
        ),
    )


def _make_controller(
    scenario: SafetyBenchmarkScenario,
) -> AutonomousExecutionController:
    """Create deterministic controller callbacks for one scenario."""

    planner_calls = {
        "count": 0
    }

    def planner_callback(
        snapshot,
    ) -> bool:
        planner_calls[
            "count"
        ] += 1

        if (
            scenario
            .planner_fails_after_initialisation
            and planner_calls[
                "count"
            ] >= 2
        ):
            return False

        return True

    def reacquisition_callback(
        snapshot,
    ) -> bool:
        return bool(
            scenario.reacquisition_succeeds
        )

    def recovery_callback(
        snapshot,
    ) -> bool:
        return bool(
            scenario.recovery_succeeds
        )

    return AutonomousExecutionController(
        planner_callback=(
            planner_callback
        ),
        reacquisition_callback=(
            reacquisition_callback
        ),
        recovery_callback=(
            recovery_callback
        ),
    )


def run_episode(
    scenario: SafetyBenchmarkScenario,
    *,
    repetition: int,
    seed: int,
    safety_config: RuntimeSafetyConfig | None = None,
) -> SafetyEpisodeOutcome:
    """Run one autonomous controller verification episode."""

    if repetition < 0:
        raise ValueError(
            "repetition must be non-negative."
        )

    if safety_config is None:
        safety_config = (
            RuntimeSafetyConfig()
        )

    rng = np.random.default_rng(
        seed
    )

    if scenario.fault is None:
        fault_onset_step: int | None = None

    else:
        fault_onset_step = int(
            rng.integers(
                6,
                19,
            )
        )

    controller = (
        _make_controller(
            scenario
        )
    )

    controller.start(
        nominal_runtime_snapshot(
            step_index=0
        )
    )

    detected = False

    first_fault_action: (
        AutonomousControllerAction
        | None
    ) = None

    first_response_step: (
        int
        | None
    ) = None

    false_alarm = False

    for step_index in range(
        1,
        EPISODE_STEPS,
    ):
        snapshot = (
            nominal_runtime_snapshot(
                step_index=step_index
            )
        )

        is_fault_step = bool(
            scenario.fault is not None
            and step_index
            == fault_onset_step
        )

        if is_fault_step:
            snapshot = (
                inject_fault(
                    snapshot,
                    scenario.fault,
                    config=safety_config,
                )
                .snapshot
            )

        decision = (
            controller.process_snapshot(
                snapshot
            )
        )

        if scenario.fault is None:
            if (
                decision.action
                != AutonomousControllerAction.CONTINUE
            ):
                false_alarm = True

        elif is_fault_step:
            first_fault_action = (
                decision.action
            )

            detected = bool(
                decision.action
                != AutonomousControllerAction.CONTINUE
            )

            if detected:
                first_response_step = (
                    step_index
                )

        if controller.state in {
            SafetyState.STOPPED,
            SafetyState.FAILED,
            SafetyState.COMPLETED,
        }:
            break

    if (
        controller.state
        == SafetyState.EXECUTING
    ):
        controller.complete(
            step_index=(
                EPISODE_STEPS
                - 1
            )
        )

    completed = bool(
        controller.state
        == SafetyState.COMPLETED
    )

    if scenario.fault is None:
        correct_response = bool(
            not false_alarm
            and completed
        )

        response_latency = None

        recovery_success = True

    else:
        correct_response = bool(
            first_fault_action
            == scenario.expected_action
        )

        if (
            detected
            and first_response_step
            is not None
            and fault_onset_step
            is not None
        ):
            response_latency = int(
                first_response_step
                - fault_onset_step
            )

        else:
            response_latency = None

        recovery_success = bool(
            scenario.should_complete
            and completed
            and correct_response
        )

    appropriate_terminal_response = bool(
        controller.state
        == scenario.expected_terminal_state
    )

    if scenario.fault is None:
        unsafe_continuation = False

    elif scenario.should_complete:
        # For a recoverable fault, continuing without the intended
        # supervisory intervention is treated as a missed safety response.
        unsafe_continuation = bool(
            not detected
        )

    else:
        # For a terminal fault, completing or remaining in execution would
        # represent unsafe continuation.
        unsafe_continuation = bool(
            controller.state
            not in {
                SafetyState.STOPPED,
                SafetyState.FAILED,
            }
        )

    return SafetyEpisodeOutcome(
        scenario=(
            scenario.name
        ),
        repetition=int(
            repetition
        ),
        seed=int(
            seed
        ),
        fault_onset_step=(
            fault_onset_step
        ),
        detected=bool(
            detected
        ),
        correct_response=bool(
            correct_response
        ),
        response_latency_steps=(
            response_latency
        ),
        completed=bool(
            completed
        ),
        terminal_state=(
            controller.state
        ),
        recovery_success=bool(
            recovery_success
        ),
        appropriate_terminal_response=bool(
            appropriate_terminal_response
        ),
        false_alarm=bool(
            false_alarm
        ),
        unsafe_continuation=bool(
            unsafe_continuation
        ),
        first_fault_action=(
            first_fault_action
        ),
        replans=int(
            controller.replan_count
        ),
        reacquisitions=int(
            controller.reacquisition_count
        ),
        recoveries=int(
            controller.recovery_count
        ),
        stops=int(
            controller.stop_count
        ),
        failures=int(
            controller.failure_count
        ),
    )


def _wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[
    float,
    float,
]:
    """Return a Wilson 95% confidence interval."""

    if total <= 0:
        return (
            0.0,
            0.0,
        )

    n = float(
        total
    )

    p = float(
        successes
        / total
    )

    z2 = (
        z**2
    )

    denominator = (
        1.0
        + z2
        / n
    )

    centre = (
        p
        + z2
        / (
            2.0
            * n
        )
    ) / denominator

    half_width = (
        z
        / denominator
        * np.sqrt(
            (
                p
                * (
                    1.0
                    - p
                )
                / n
            )
            + z2
            / (
                4.0
                * n**2
            )
        )
    )

    return (
        float(
            max(
                0.0,
                centre
                - half_width,
            )
        ),
        float(
            min(
                1.0,
                centre
                + half_width,
            )
        ),
    )


def _rate_summary(
    values: list[
        bool
    ],
) -> dict:
    """Return rate, count and Wilson interval."""

    total = len(
        values
    )

    successes = int(
        sum(
            values
        )
    )

    lower, upper = (
        _wilson_interval(
            successes,
            total,
        )
    )

    return {
        "count": int(
            successes
        ),
        "total": int(
            total
        ),
        "rate": float(
            successes
            / total
            if total > 0
            else 0.0
        ),
        "95_ci": [
            float(
                lower
            ),
            float(
                upper
            ),
        ],
    }


def _summarise_scenario(
    outcomes: list[
        SafetyEpisodeOutcome
    ],
    scenario: SafetyBenchmarkScenario,
) -> dict:
    """Aggregate one final benchmark scenario."""

    if len(
        outcomes
    ) == 0:
        raise ValueError(
            "At least one outcome is required."
        )

    latency_values = [
        outcome.response_latency_steps
        for outcome in outcomes
        if outcome.response_latency_steps
        is not None
    ]

    if len(
        latency_values
    ) == 0:
        mean_latency = None
        maximum_latency = None

    else:
        mean_latency = float(
            np.mean(
                latency_values
            )
        )

        maximum_latency = int(
            np.max(
                latency_values
            )
        )

    return {
        "fault": (
            None
            if scenario.fault is None
            else scenario.fault.value
        ),
        "expected_action": (
            scenario
            .expected_action
            .value
        ),
        "expected_terminal_state": (
            scenario
            .expected_terminal_state
            .value
        ),
        "should_complete": bool(
            scenario.should_complete
        ),
        "detection": (
            _rate_summary(
                [
                    outcome.detected
                    for outcome in outcomes
                ]
            )
            if scenario.fault
            is not None
            else None
        ),
        "correct_response": (
            _rate_summary(
                [
                    outcome.correct_response
                    for outcome in outcomes
                ]
            )
        ),
        "completion": (
            _rate_summary(
                [
                    outcome.completed
                    for outcome in outcomes
                ]
            )
        ),
        "recovery_success": (
            _rate_summary(
                [
                    outcome.recovery_success
                    for outcome in outcomes
                ]
            )
            if scenario.should_complete
            and scenario.fault
            is not None
            else None
        ),
        "appropriate_terminal_response": (
            _rate_summary(
                [
                    outcome
                    .appropriate_terminal_response
                    for outcome in outcomes
                ]
            )
        ),
        "false_alarm": (
            _rate_summary(
                [
                    outcome.false_alarm
                    for outcome in outcomes
                ]
            )
        ),
        "unsafe_continuation": (
            _rate_summary(
                [
                    outcome.unsafe_continuation
                    for outcome in outcomes
                ]
            )
        ),
        "mean_response_latency_steps": (
            mean_latency
        ),
        "maximum_response_latency_steps": (
            maximum_latency
        ),
        "mean_replans": float(
            np.mean(
                [
                    outcome.replans
                    for outcome in outcomes
                ]
            )
        ),
        "mean_reacquisitions": float(
            np.mean(
                [
                    outcome.reacquisitions
                    for outcome in outcomes
                ]
            )
        ),
        "mean_recoveries": float(
            np.mean(
                [
                    outcome.recoveries
                    for outcome in outcomes
                ]
            )
        ),
        "mean_stops": float(
            np.mean(
                [
                    outcome.stops
                    for outcome in outcomes
                ]
            )
        ),
        "mean_failures": float(
            np.mean(
                [
                    outcome.failures
                    for outcome in outcomes
                ]
            )
        ),
    }


def run_benchmark(
    repetitions: int = DEFAULT_REPETITIONS,
    seed: int = DEFAULT_SEED,
    output_directory: Path = RESULT_DIRECTORY,
) -> dict:
    """Run the final Phase 7 autonomous safety benchmark."""

    if repetitions < 1:
        raise ValueError(
            "repetitions must be positive."
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    scenarios = (
        benchmark_scenarios()
    )

    rows: list[
        dict
    ] = []

    grouped: dict[
        str,
        list[
            SafetyEpisodeOutcome
        ],
    ] = {
        scenario.name: []
        for scenario in scenarios
    }

    for (
        scenario_index,
        scenario,
    ) in enumerate(
        scenarios
    ):
        for repetition in range(
            repetitions
        ):
            episode_seed = int(
                seed
                + scenario_index
                * 100000
                + repetition
                * 101
            )

            outcome = run_episode(
                scenario,
                repetition=repetition,
                seed=episode_seed,
            )

            grouped[
                scenario.name
            ].append(
                outcome
            )

            rows.append(
                {
                    "scenario": outcome.scenario,
                    "repetition": outcome.repetition,
                    "seed": outcome.seed,
                    "fault_onset_step": (
                        ""
                        if outcome.fault_onset_step
                        is None
                        else outcome.fault_onset_step
                    ),
                    "detected": outcome.detected,
                    "correct_response": (
                        outcome.correct_response
                    ),
                    "response_latency_steps": (
                        ""
                        if outcome.response_latency_steps
                        is None
                        else outcome.response_latency_steps
                    ),
                    "completed": outcome.completed,
                    "terminal_state": (
                        outcome.terminal_state.value
                    ),
                    "recovery_success": (
                        outcome.recovery_success
                    ),
                    "appropriate_terminal_response": (
                        outcome
                        .appropriate_terminal_response
                    ),
                    "false_alarm": (
                        outcome.false_alarm
                    ),
                    "unsafe_continuation": (
                        outcome.unsafe_continuation
                    ),
                    "first_fault_action": (
                        ""
                        if outcome.first_fault_action
                        is None
                        else outcome
                        .first_fault_action
                        .value
                    ),
                    "replans": outcome.replans,
                    "reacquisitions": (
                        outcome.reacquisitions
                    ),
                    "recoveries": (
                        outcome.recoveries
                    ),
                    "stops": outcome.stops,
                    "failures": outcome.failures,
                }
            )

    summary = {
        "benchmark": (
            "phase7_autonomous_safety"
        ),
        "repetitions_per_scenario": int(
            repetitions
        ),
        "base_seed": int(
            seed
        ),
        "episode_steps": int(
            EPISODE_STEPS
        ),
        "scenario_count": int(
            len(
                scenarios
            )
        ),
        "total_episodes": int(
            len(
                scenarios
            )
            * repetitions
        ),
        "scenarios": {},
        "interpretation_boundary": (
            "Software fault-response verification in engineered "
            "simulation scenarios only. These rates are not clinical "
            "hazard probabilities and do not establish surgical or "
            "medical-device safety."
        ),
    }

    for scenario in scenarios:
        summary[
            "scenarios"
        ][
            scenario.name
        ] = (
            _summarise_scenario(
                grouped[
                    scenario.name
                ],
                scenario,
            )
        )

    csv_path = (
        output_directory
        / "phase7_autonomous_safety_trials.csv"
    )

    json_path = (
        output_directory
        / "phase7_autonomous_safety_summary.json"
    )

    fieldnames = [
        "scenario",
        "repetition",
        "seed",
        "fault_onset_step",
        "detected",
        "correct_response",
        "response_latency_steps",
        "completed",
        "terminal_state",
        "recovery_success",
        "appropriate_terminal_response",
        "false_alarm",
        "unsafe_continuation",
        "first_fault_action",
        "replans",
        "reacquisitions",
        "recoveries",
        "stops",
        "failures",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    summary[
        "trial_csv"
    ] = str(
        csv_path
    )

    summary[
        "summary_json"
    ] = str(
        json_path
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    return summary


def print_summary(
    summary: dict,
) -> None:
    """Print compact Phase 7 benchmark evidence."""

    print(
        "\n"
        "=== Phase 7 Autonomous Safety Benchmark ==="
    )

    print(
        "Scenarios: "
        f"{summary['scenario_count']}"
    )

    print(
        "Repetitions per scenario: "
        f"{summary['repetitions_per_scenario']}"
    )

    print(
        "Total episodes: "
        f"{summary['total_episodes']}"
    )

    for (
        name,
        metrics,
    ) in (
        summary[
            "scenarios"
        ].items()
    ):
        detection = (
            metrics[
                "detection"
            ]
        )

        detection_text = (
            "N/A"
            if detection is None
            else (
                f"{100.0 * detection['rate']:.1f}%"
            )
        )

        print(
            "\n"
            f"{name}: "
            f"detect={detection_text} | "
            f"correct="
            f"{100.0 * metrics['correct_response']['rate']:.1f}% | "
            f"complete="
            f"{100.0 * metrics['completion']['rate']:.1f}% | "
            f"terminal="
            f"{100.0 * metrics['appropriate_terminal_response']['rate']:.1f}% | "
            f"false_alarm="
            f"{100.0 * metrics['false_alarm']['rate']:.1f}% | "
            f"unsafe_continue="
            f"{100.0 * metrics['unsafe_continuation']['rate']:.1f}%"
        )

    print(
        "\nArtifacts:"
    )

    print(
        summary[
            "trial_csv"
        ]
    )

    print(
        summary[
            "summary_json"
        ]
    )

    print(
        "\nInterpretation: engineered simulation "
        "fault-response evidence only."
    )


def main() -> None:
    """Run final benchmark."""

    summary = (
        run_benchmark()
    )

    print_summary(
        summary
    )


if __name__ == "__main__":
    main()