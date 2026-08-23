"""Supplementary three-strategy planning-efficiency benchmark.

This benchmark closes the final computation/path-cost comparison for:

- Fixed View
- Generic Active Perception
- Task-Aware Active Perception

The existing planner already reports:

- RRT planning time;
- RRT iterations;
- smoothed-path cost;
- planning success;
- hidden-ground-truth safety outcomes.

No planner behaviour is changed here.

Metric interpretation
---------------------
Planning time and iteration count are reported over ALL planning attempts,
including failures, because failed planning still consumes computation.

Path cost is reported only conditional on successful planning because the
existing planner represents failed-plan path cost as +infinity.

For paired path-cost comparisons, Task-Aware is compared with a baseline only
on matched trials where BOTH strategies successfully produced a path.

Because planning time is measured using wall-clock time, strategy execution
order is cyclically balanced across matched trials to reduce systematic
warm-up/order bias.

This remains a simulation benchmark and does not measure real-time performance
on a physical surgical robot.
"""

from __future__ import annotations

import argparse
import csv
import json

from dataclasses import (
    asdict,
    dataclass,
)
from pathlib import Path

import numpy as np

from src.perception.active_perception import (
    GenericActivePerception,
)
from src.perception.task_aware_active_perception import (
    TaskAwareActivePerception,
)
from src.perception.task_aware_scoring import (
    TaskAwareScoringConfig,
    TaskAwareViewpointScorer,
)
from src.perception.viewpoint_scoring import (
    GenericViewpointScorer,
)
from src.simulation.three_strategy_navigation import (
    StrategyNavigationResult,
    run_navigation_from_perception,
)
from src.simulation.three_strategy_perception import (
    StrategyPerceptionResult,
    run_fixed_perception,
    run_generic_active_perception,
    run_task_aware_active_perception,
)
from src.simulation.three_strategy_robustness_benchmark import (
    RobustnessScenario,
    ScenarioInputs,
    build_scenario_inputs,
    default_scenarios,
)


STRATEGIES = (
    "fixed",
    "generic_active",
    "task_aware_active",
)


@dataclass(frozen=True)
class EfficiencyBenchmarkConfig:
    """Configuration for the supplementary efficiency benchmark."""

    repetitions_per_scenario: int = 5

    perception_seed_base: int = 20261020

    planner_seed_base: int = 13000

    sigma_multiplier: float = 2.0

    instrument_radius: float = 0.006

    proximal_length: float = 0.10

    task_weight: float = 2.0

    alignment_weight: float = 1.0

    uncertainty_weight: float = 1.0

    bootstrap_samples: int = 1000

    bootstrap_seed: int = 20261021

    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if self.repetitions_per_scenario <= 0:
            raise ValueError(
                "repetitions_per_scenario must be positive."
            )

        if self.sigma_multiplier < 0.0:
            raise ValueError(
                "sigma_multiplier must be non-negative."
            )

        if self.instrument_radius <= 0.0:
            raise ValueError(
                "instrument_radius must be positive."
            )

        if self.proximal_length <= 0.0:
            raise ValueError(
                "proximal_length must be positive."
            )

        if self.task_weight < 0.0:
            raise ValueError(
                "task_weight must be non-negative."
            )

        if self.alignment_weight < 0.0:
            raise ValueError(
                "alignment_weight must be non-negative."
            )

        if self.uncertainty_weight < 0.0:
            raise ValueError(
                "uncertainty_weight must be non-negative."
            )

        if self.bootstrap_samples <= 0:
            raise ValueError(
                "bootstrap_samples must be positive."
            )

        if not (
            0.0
            < self.confidence_level
            < 1.0
        ):
            raise ValueError(
                "confidence_level must lie between 0 and 1."
            )


@dataclass(frozen=True)
class EfficiencyRecord:
    """One strategy result in one matched planning unit."""

    scenario_id: int

    scenario_name: str

    repetition: int

    strategy: str

    execution_position: int

    perception_seed: int

    planner_seed: int

    camera_movement: float

    planning_success: bool

    safe_navigation_success: bool

    collision_against_truth: bool

    safety_violation_against_truth: bool

    planning_time_seconds: float

    iterations: int

    path_cost: float


@dataclass(frozen=True)
class MetricEstimate:
    """Mean and confidence interval."""

    n: int

    mean: float

    ci_low: float

    ci_high: float


@dataclass(frozen=True)
class EfficiencyStrategySummary:
    """Efficiency results for one strategy."""

    strategy: str

    scenario_count: int

    record_count: int

    planning_success_rate: MetricEstimate

    safe_navigation_success_rate: MetricEstimate

    camera_movement: MetricEstimate

    planning_time_all_attempts: MetricEstimate

    iterations_all_attempts: MetricEstimate

    planning_time_given_success: MetricEstimate

    iterations_given_success: MetricEstimate

    path_cost_given_success: MetricEstimate


@dataclass(frozen=True)
class EfficiencyPairedComparison:
    """Task-Aware minus comparator efficiency effects."""

    comparator: str

    task_strategy: str

    matched_count: int

    both_success_count: int

    planning_time_difference_all_attempts: MetricEstimate

    iterations_difference_all_attempts: MetricEstimate

    camera_movement_difference: MetricEstimate

    planning_success_difference: MetricEstimate

    safe_navigation_success_difference: MetricEstimate

    path_cost_difference_both_successful: MetricEstimate


@dataclass(frozen=True)
class EfficiencyBenchmarkResult:
    """Complete supplementary efficiency benchmark."""

    config: EfficiencyBenchmarkConfig

    scenarios: tuple[
        RobustnessScenario,
        ...,
    ]

    records: tuple[
        EfficiencyRecord,
        ...,
    ]

    summaries: tuple[
        EfficiencyStrategySummary,
        ...,
    ]

    paired_comparisons: tuple[
        EfficiencyPairedComparison,
        ...,
    ]


def balanced_strategy_order(
    unit_index: int,
) -> tuple[
    str,
    ...,
]:
    """Return a cyclically balanced strategy execution order."""

    if unit_index < 0:
        raise ValueError(
            "unit_index must be non-negative."
        )

    offset = (
        unit_index
        % len(
            STRATEGIES
        )
    )

    return (
        STRATEGIES[
            offset:
        ]
        + STRATEGIES[
            :offset
        ]
    )


def _build_controllers(
    *,
    inputs: ScenarioInputs,
    config: EfficiencyBenchmarkConfig,
) -> tuple[
    GenericActivePerception,
    TaskAwareActivePerception,
]:
    """Build controllers for one scenario."""

    generic_scorer = (
        GenericViewpointScorer(
            observation_model=(
                inputs.observation_model
            )
        )
    )

    generic_controller = (
        GenericActivePerception(
            observation_model=(
                inputs.observation_model
            ),
            scorer=generic_scorer,
        )
    )

    task_generic_scorer = (
        GenericViewpointScorer(
            observation_model=(
                inputs.observation_model
            )
        )
    )

    task_scorer = (
        TaskAwareViewpointScorer(
            generic_scorer=(
                task_generic_scorer
            ),
            task=inputs.task,
            task_config=(
                TaskAwareScoringConfig(
                    task_weight=(
                        config.task_weight
                    ),
                    alignment_weight=(
                        config.alignment_weight
                    ),
                    uncertainty_weight=(
                        config.uncertainty_weight
                    ),
                )
            ),
        )
    )

    task_controller = (
        TaskAwareActivePerception(
            scorer=task_scorer
        )
    )

    return (
        generic_controller,
        task_controller,
    )


def _run_perceptions(
    *,
    inputs: ScenarioInputs,
    config: EfficiencyBenchmarkConfig,
    perception_seed: int,
) -> dict[
    str,
    StrategyPerceptionResult,
]:
    """Run all three perception strategies with matched randomness."""

    (
        generic_controller,
        task_controller,
    ) = _build_controllers(
        inputs=inputs,
        config=config,
    )

    fixed = (
        run_fixed_perception(
            observation_model=(
                inputs.observation_model
            ),
            initial_pose=(
                inputs.initial_pose
            ),
            true_structures=(
                inputs.true_structures
            ),
            seed=perception_seed,
            occluders=(
                inputs.occluders
            ),
        )
    )

    generic = (
        run_generic_active_perception(
            controller=(
                generic_controller
            ),
            observation_model=(
                inputs.observation_model
            ),
            initial_pose=(
                inputs.initial_pose
            ),
            candidates=(
                inputs.candidates
            ),
            target=(
                inputs.target
            ),
            true_structures=(
                inputs.true_structures
            ),
            seed=perception_seed,
            occluders=(
                inputs.occluders
            ),
        )
    )

    task_aware = (
        run_task_aware_active_perception(
            controller=(
                task_controller
            ),
            observation_model=(
                inputs.observation_model
            ),
            initial_pose=(
                inputs.initial_pose
            ),
            candidates=(
                inputs.candidates
            ),
            target=(
                inputs.target
            ),
            task=(
                inputs.task
            ),
            true_structures=(
                inputs.true_structures
            ),
            seed=perception_seed,
            occluders=(
                inputs.occluders
            ),
        )
    )

    return {
        "fixed": fixed,
        "generic_active": generic,
        "task_aware_active": task_aware,
    }


def _navigation_to_record(
    *,
    scenario: RobustnessScenario,
    repetition: int,
    execution_position: int,
    perception_seed: int,
    planner_seed: int,
    navigation: StrategyNavigationResult,
) -> EfficiencyRecord:
    """Convert one navigation result to an efficiency record."""

    planner = (
        navigation.planner_result
    )

    planning_success = bool(
        planner.planning_success
    )

    collision = bool(
        planner.collision_against_truth
    )

    violation = bool(
        planner.safety_violation_against_truth
    )

    safe_navigation_success = (
        planning_success
        and not collision
        and not violation
    )

    planning_time = float(
        planner.planning_time
    )

    if (
        not np.isfinite(
            planning_time
        )
        or planning_time < 0.0
    ):
        raise ValueError(
            "planning_time must be finite and non-negative."
        )

    iterations = int(
        planner.iterations
    )

    if iterations < 0:
        raise ValueError(
            "iterations must be non-negative."
        )

    path_cost = float(
        planner.path_cost
    )

    if planning_success:
        if (
            not np.isfinite(
                path_cost
            )
            or path_cost < 0.0
        ):
            raise ValueError(
                "Successful planning must have finite "
                "non-negative path cost."
            )

    else:
        if not np.isinf(
            path_cost
        ):
            raise ValueError(
                "Failed planning is expected to have infinite path cost."
            )

    return EfficiencyRecord(
        scenario_id=(
            scenario.scenario_id
        ),
        scenario_name=(
            scenario.name
        ),
        repetition=int(
            repetition
        ),
        strategy=(
            navigation.strategy
        ),
        execution_position=int(
            execution_position
        ),
        perception_seed=int(
            perception_seed
        ),
        planner_seed=int(
            planner_seed
        ),
        camera_movement=float(
            navigation
            .perception
            .camera_movement
        ),
        planning_success=(
            planning_success
        ),
        safe_navigation_success=(
            safe_navigation_success
        ),
        collision_against_truth=(
            collision
        ),
        safety_violation_against_truth=(
            violation
        ),
        planning_time_seconds=(
            planning_time
        ),
        iterations=(
            iterations
        ),
        path_cost=(
            path_cost
        ),
    )


def run_efficiency_trials(
    config: EfficiencyBenchmarkConfig,
    *,
    scenarios: tuple[
        RobustnessScenario,
        ...,
    ] | None = None,
    show_progress: bool = False,
) -> tuple[
    EfficiencyRecord,
    ...,
]:
    """Run the complete matched efficiency experiment."""

    if scenarios is None:
        scenarios = (
            default_scenarios()
        )

    if len(
        scenarios
    ) == 0:
        raise ValueError(
            "At least one scenario is required."
        )

    records: list[
        EfficiencyRecord
    ] = []

    total_units = (
        len(
            scenarios
        )
        * config
        .repetitions_per_scenario
    )

    completed = 0

    for scenario in scenarios:
        inputs = (
            build_scenario_inputs(
                scenario
            )
        )

        for repetition in range(
            config.repetitions_per_scenario
        ):
            seed_offset = (
                scenario.scenario_id
                * 1000
                + repetition
            )

            perception_seed = (
                config
                .perception_seed_base
                + seed_offset
            )

            planner_seed = (
                config
                .planner_seed_base
                + seed_offset
            )

            perceptions = (
                _run_perceptions(
                    inputs=inputs,
                    config=config,
                    perception_seed=(
                        perception_seed
                    ),
                )
            )

            unit_index = (
                scenario.scenario_id
                * config
                .repetitions_per_scenario
                + repetition
            )

            order = (
                balanced_strategy_order(
                    unit_index
                )
            )

            for (
                execution_position,
                strategy,
            ) in enumerate(
                order
            ):
                navigation = (
                    run_navigation_from_perception(
                        trial=repetition,
                        perception=(
                            perceptions[
                                strategy
                            ]
                        ),
                        instrument=(
                            inputs.instrument
                        ),
                        start_q=(
                            inputs.start_q
                        ),
                        goal_q=(
                            inputs.goal_q
                        ),
                        true_structures=(
                            inputs
                            .true_structures
                        ),
                        sigma_multiplier=(
                            config
                            .sigma_multiplier
                        ),
                        instrument_radius=(
                            config
                            .instrument_radius
                        ),
                        proximal_length=(
                            config
                            .proximal_length
                        ),
                        planner_seed=(
                            planner_seed
                        ),
                    )
                )

                records.append(
                    _navigation_to_record(
                        scenario=scenario,
                        repetition=(
                            repetition
                        ),
                        execution_position=(
                            execution_position
                        ),
                        perception_seed=(
                            perception_seed
                        ),
                        planner_seed=(
                            planner_seed
                        ),
                        navigation=(
                            navigation
                        ),
                    )
                )

            completed += 1

            if (
                show_progress
                and (
                    completed % 10
                    == 0
                    or completed
                    == total_units
                )
            ):
                print(
                    f"Completed "
                    f"{completed}/"
                    f"{total_units} "
                    "matched efficiency units."
                )

    return tuple(
        records
    )


def _clustered_estimate(
    values: np.ndarray,
    scenario_ids: np.ndarray,
    *,
    config: EfficiencyBenchmarkConfig,
    seed_offset: int,
) -> MetricEstimate:
    """Return a scenario-clustered bootstrap estimate."""

    values = np.asarray(
        values,
        dtype=float,
    )

    scenario_ids = np.asarray(
        scenario_ids,
        dtype=int,
    )

    finite = np.isfinite(
        values
    )

    values = (
        values[
            finite
        ]
    )

    scenario_ids = (
        scenario_ids[
            finite
        ]
    )

    if values.size == 0:
        return MetricEstimate(
            n=0,
            mean=float(
                "nan"
            ),
            ci_low=float(
                "nan"
            ),
            ci_high=float(
                "nan"
            ),
        )

    mean = float(
        np.mean(
            values
        )
    )

    unique_scenarios = (
        np.unique(
            scenario_ids
        )
    )

    if unique_scenarios.size == 1:
        return MetricEstimate(
            n=int(
                values.size
            ),
            mean=mean,
            ci_low=mean,
            ci_high=mean,
        )

    rng = (
        np.random.default_rng(
            config.bootstrap_seed
            + seed_offset
        )
    )

    bootstrap_means = np.empty(
        config.bootstrap_samples,
        dtype=float,
    )

    for index in range(
        config.bootstrap_samples
    ):
        sampled_scenarios = (
            rng.choice(
                unique_scenarios,
                size=(
                    unique_scenarios
                    .size
                ),
                replace=True,
            )
        )

        sampled_values = (
            np.concatenate(
                [
                    values[
                        scenario_ids
                        == scenario_id
                    ]
                    for scenario_id
                    in sampled_scenarios
                ]
            )
        )

        bootstrap_means[
            index
        ] = float(
            np.mean(
                sampled_values
            )
        )

    alpha = (
        1.0
        - config.confidence_level
    )

    return MetricEstimate(
        n=int(
            values.size
        ),
        mean=mean,
        ci_low=float(
            np.quantile(
                bootstrap_means,
                alpha / 2.0,
            )
        ),
        ci_high=float(
            np.quantile(
                bootstrap_means,
                1.0
                - alpha / 2.0,
            )
        ),
    )


def _strategy_records(
    records: tuple[
        EfficiencyRecord,
        ...,
    ],
    strategy: str,
) -> tuple[
    EfficiencyRecord,
    ...,
]:
    """Return sorted records for one strategy."""

    return tuple(
        sorted(
            (
                record
                for record
                in records
                if record.strategy
                == strategy
            ),
            key=lambda record: (
                record.scenario_id,
                record.repetition,
            ),
        )
    )


def _strategy_summary(
    *,
    records: tuple[
        EfficiencyRecord,
        ...,
    ],
    strategy: str,
    config: EfficiencyBenchmarkConfig,
    seed_offset: int,
) -> EfficiencyStrategySummary:
    """Calculate one strategy's efficiency statistics."""

    selected = (
        _strategy_records(
            records,
            strategy,
        )
    )

    if len(
        selected
    ) == 0:
        raise ValueError(
            f"No records found for strategy {strategy}."
        )

    scenario_ids = (
        np.asarray(
            [
                record.scenario_id
                for record
                in selected
            ],
            dtype=int,
        )
    )

    def values(
        attribute: str,
    ) -> np.ndarray:
        return np.asarray(
            [
                getattr(
                    record,
                    attribute,
                )
                for record
                in selected
            ],
            dtype=float,
        )

    def estimate(
        attribute: str,
        offset: int,
        mask: np.ndarray
        | None = None,
    ) -> MetricEstimate:
        metric_values = (
            values(
                attribute
            )
        )

        metric_scenarios = (
            scenario_ids
        )

        if mask is not None:
            metric_values = (
                metric_values[
                    mask
                ]
            )

            metric_scenarios = (
                metric_scenarios[
                    mask
                ]
            )

        return _clustered_estimate(
            metric_values,
            metric_scenarios,
            config=config,
            seed_offset=(
                seed_offset
                + offset
            ),
        )

    success_mask = (
        values(
            "planning_success"
        )
        > 0.5
    )

    return EfficiencyStrategySummary(
        strategy=strategy,
        scenario_count=int(
            np.unique(
                scenario_ids
            ).size
        ),
        record_count=len(
            selected
        ),
        planning_success_rate=(
            estimate(
                "planning_success",
                1,
            )
        ),
        safe_navigation_success_rate=(
            estimate(
                "safe_navigation_success",
                2,
            )
        ),
        camera_movement=(
            estimate(
                "camera_movement",
                3,
            )
        ),
        planning_time_all_attempts=(
            estimate(
                "planning_time_seconds",
                4,
            )
        ),
        iterations_all_attempts=(
            estimate(
                "iterations",
                5,
            )
        ),
        planning_time_given_success=(
            estimate(
                "planning_time_seconds",
                6,
                success_mask,
            )
        ),
        iterations_given_success=(
            estimate(
                "iterations",
                7,
                success_mask,
            )
        ),
        path_cost_given_success=(
            estimate(
                "path_cost",
                8,
                success_mask,
            )
        ),
    )


def _paired_comparison(
    *,
    records: tuple[
        EfficiencyRecord,
        ...,
    ],
    comparator: str,
    config: EfficiencyBenchmarkConfig,
    seed_offset: int,
) -> EfficiencyPairedComparison:
    """Calculate Task-Aware minus comparator paired effects."""

    task_records = (
        _strategy_records(
            records,
            "task_aware_active",
        )
    )

    comparator_records = (
        _strategy_records(
            records,
            comparator,
        )
    )

    if (
        len(
            task_records
        )
        != len(
            comparator_records
        )
    ):
        raise ValueError(
            "Matched strategy record counts differ."
        )

    task_keys = [
        (
            record.scenario_id,
            record.repetition,
        )
        for record
        in task_records
    ]

    comparator_keys = [
        (
            record.scenario_id,
            record.repetition,
        )
        for record
        in comparator_records
    ]

    if (
        task_keys
        != comparator_keys
    ):
        raise ValueError(
            "Strategy records are not correctly matched."
        )

    scenario_ids = (
        np.asarray(
            [
                record.scenario_id
                for record
                in task_records
            ],
            dtype=int,
        )
    )

    def task_values(
        attribute: str,
    ) -> np.ndarray:
        return np.asarray(
            [
                getattr(
                    record,
                    attribute,
                )
                for record
                in task_records
            ],
            dtype=float,
        )

    def comparator_values(
        attribute: str,
    ) -> np.ndarray:
        return np.asarray(
            [
                getattr(
                    record,
                    attribute,
                )
                for record
                in comparator_records
            ],
            dtype=float,
        )

    def paired_all(
        attribute: str,
        offset: int,
    ) -> MetricEstimate:
        differences = (
            task_values(
                attribute
            )
            - comparator_values(
                attribute
            )
        )

        return _clustered_estimate(
            differences,
            scenario_ids,
            config=config,
            seed_offset=(
                seed_offset
                + offset
            ),
        )

    task_success = (
        task_values(
            "planning_success"
        )
        > 0.5
    )

    comparator_success = (
        comparator_values(
            "planning_success"
        )
        > 0.5
    )

    both_success = (
        task_success
        & comparator_success
    )

    path_differences = (
        task_values(
            "path_cost"
        )[
            both_success
        ]
        - comparator_values(
            "path_cost"
        )[
            both_success
        ]
    )

    path_scenarios = (
        scenario_ids[
            both_success
        ]
    )

    path_cost_difference = (
        _clustered_estimate(
            path_differences,
            path_scenarios,
            config=config,
            seed_offset=(
                seed_offset
                + 6
            ),
        )
    )

    return EfficiencyPairedComparison(
        comparator=(
            comparator
        ),
        task_strategy=(
            "task_aware_active"
        ),
        matched_count=len(
            task_records
        ),
        both_success_count=int(
            np.count_nonzero(
                both_success
            )
        ),
        planning_time_difference_all_attempts=(
            paired_all(
                "planning_time_seconds",
                1,
            )
        ),
        iterations_difference_all_attempts=(
            paired_all(
                "iterations",
                2,
            )
        ),
        camera_movement_difference=(
            paired_all(
                "camera_movement",
                3,
            )
        ),
        planning_success_difference=(
            paired_all(
                "planning_success",
                4,
            )
        ),
        safe_navigation_success_difference=(
            paired_all(
                "safe_navigation_success",
                5,
            )
        ),
        path_cost_difference_both_successful=(
            path_cost_difference
        ),
    )


def analyse_efficiency_records(
    *,
    records: tuple[
        EfficiencyRecord,
        ...,
    ],
    scenarios: tuple[
        RobustnessScenario,
        ...,
    ],
    config: EfficiencyBenchmarkConfig,
) -> EfficiencyBenchmarkResult:
    """Aggregate the complete efficiency experiment."""

    expected = (
        len(
            scenarios
        )
        * config
        .repetitions_per_scenario
        * len(
            STRATEGIES
        )
    )

    if (
        len(
            records
        )
        != expected
    ):
        raise ValueError(
            f"Expected {expected} records, "
            f"received {len(records)}."
        )

    summaries = tuple(
        _strategy_summary(
            records=records,
            strategy=strategy,
            config=config,
            seed_offset=(
                index
                * 100
            ),
        )
        for index, strategy
        in enumerate(
            STRATEGIES
        )
    )

    comparisons = (
        _paired_comparison(
            records=records,
            comparator="fixed",
            config=config,
            seed_offset=1000,
        ),
        _paired_comparison(
            records=records,
            comparator=(
                "generic_active"
            ),
            config=config,
            seed_offset=2000,
        ),
    )

    return EfficiencyBenchmarkResult(
        config=config,
        scenarios=scenarios,
        records=records,
        summaries=summaries,
        paired_comparisons=(
            comparisons
        ),
    )


def run_efficiency_benchmark(
    config: EfficiencyBenchmarkConfig
    | None = None,
    *,
    scenarios: tuple[
        RobustnessScenario,
        ...,
    ] | None = None,
    show_progress: bool = False,
) -> EfficiencyBenchmarkResult:
    """Run and analyse the complete benchmark."""

    if config is None:
        config = (
            EfficiencyBenchmarkConfig()
        )

    if scenarios is None:
        scenarios = (
            default_scenarios()
        )

    records = (
        run_efficiency_trials(
            config,
            scenarios=scenarios,
            show_progress=(
                show_progress
            ),
        )
    )

    return analyse_efficiency_records(
        records=records,
        scenarios=scenarios,
        config=config,
    )


def save_efficiency_outputs(
    result: EfficiencyBenchmarkResult,
    *,
    output_directory: str
    | Path = (
        "results/"
        "supplementary_efficiency"
    ),
) -> tuple[
    Path,
    Path,
]:
    """Save raw efficiency records and summary."""

    if len(
        result.records
    ) == 0:
        raise ValueError(
            "Cannot save an empty efficiency result."
        )

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        output_directory
        / "three_strategy_efficiency_trials.csv"
    )

    summary_path = (
        output_directory
        / "three_strategy_efficiency_summary.json"
    )

    with raw_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                asdict(
                    result.records[
                        0
                    ]
                ).keys()
            ),
        )

        writer.writeheader()

        for record in (
            result.records
        ):
            writer.writerow(
                asdict(
                    record
                )
            )

    payload = {
        "metric_definitions": {
            "planning_time_all_attempts": (
                "RRT search wall-clock time over all "
                "planning attempts, including failures."
            ),
            "iterations_all_attempts": (
                "RRT iterations over all attempts, "
                "including failures."
            ),
            "path_cost_given_success": (
                "Smoothed joint-space path cost "
                "conditional on planning success."
            ),
            "paired_path_cost": (
                "Task-Aware minus comparator path cost "
                "only on matched trials where both "
                "strategies successfully planned."
            ),
            "execution_order": (
                "Strategy execution order is cyclically "
                "balanced to reduce systematic timing-order bias."
            ),
        },
        "config": asdict(
            result.config
        ),
        "scenarios": [
            asdict(
                scenario
            )
            for scenario
            in result.scenarios
        ],
        "summaries": [
            asdict(
                summary
            )
            for summary
            in result.summaries
        ],
        "paired_comparisons": [
            asdict(
                comparison
            )
            for comparison
            in result.paired_comparisons
        ],
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            allow_nan=True,
        )

    return (
        raw_path,
        summary_path,
    )


def _format_rate(
    metric: MetricEstimate,
) -> str:
    """Format a probability metric."""

    if metric.n == 0:
        return "N/A"

    return (
        f"{metric.mean * 100.0:.1f}% "
        f"["
        f"{metric.ci_low * 100.0:.1f}, "
        f"{metric.ci_high * 100.0:.1f}"
        f"]"
    )


def _format_seconds(
    metric: MetricEstimate,
) -> str:
    """Format planning time."""

    if metric.n == 0:
        return "N/A"

    return (
        f"{metric.mean:.4f} "
        f"["
        f"{metric.ci_low:.4f}, "
        f"{metric.ci_high:.4f}"
        f"] s"
    )


def _format_number(
    metric: MetricEstimate,
) -> str:
    """Format a generic numerical metric."""

    if metric.n == 0:
        return "N/A"

    return (
        f"{metric.mean:.3f} "
        f"["
        f"{metric.ci_low:.3f}, "
        f"{metric.ci_high:.3f}"
        f"]"
    )


def _format_mm(
    metric: MetricEstimate,
) -> str:
    """Format camera movement."""

    if metric.n == 0:
        return "N/A"

    return (
        f"{metric.mean * 1000.0:.3f} "
        f"["
        f"{metric.ci_low * 1000.0:.3f}, "
        f"{metric.ci_high * 1000.0:.3f}"
        f"] mm"
    )


def print_efficiency_summary(
    result: EfficiencyBenchmarkResult,
) -> None:
    """Print a compact research-oriented summary."""

    print()

    print(
        "Three-Strategy Planning-Efficiency Benchmark"
    )

    print(
        "============================================"
    )

    print(
        f"Scenarios: "
        f"{len(result.scenarios)}"
    )

    print(
        "Repetitions per scenario: "
        f"{result.config.repetitions_per_scenario}"
    )

    print(
        "Matched planning units: "
        f"{len(result.scenarios) * result.config.repetitions_per_scenario}"
    )

    print()

    names = {
        "fixed": (
            "Fixed View"
        ),
        "generic_active": (
            "Generic Active"
        ),
        "task_aware_active": (
            "Task-Aware Active"
        ),
    }

    for summary in (
        result.summaries
    ):
        print(
            names[
                summary.strategy
            ]
        )

        print(
            "  Planning success:              "
            + _format_rate(
                summary
                .planning_success_rate
            )
        )

        print(
            "  SAFE navigation success:       "
            + _format_rate(
                summary
                .safe_navigation_success_rate
            )
        )

        print(
            "  Camera movement:               "
            + _format_mm(
                summary
                .camera_movement
            )
        )

        print(
            "  Planning time | all attempts:  "
            + _format_seconds(
                summary
                .planning_time_all_attempts
            )
        )

        print(
            "  RRT iterations | all attempts: "
            + _format_number(
                summary
                .iterations_all_attempts
            )
        )

        print(
            "  Planning time | success:       "
            + _format_seconds(
                summary
                .planning_time_given_success
            )
        )

        print(
            "  RRT iterations | success:      "
            + _format_number(
                summary
                .iterations_given_success
            )
        )

        print(
            "  Path cost | success:           "
            + _format_number(
                summary
                .path_cost_given_success
            )
        )

        print()

    print(
        "Paired Task-Aware minus Comparator"
    )

    print(
        "----------------------------------"
    )

    for comparison in (
        result.paired_comparisons
    ):
        print(
            "Comparator: "
            + names[
                comparison.comparator
            ]
        )

        print(
            "  Matched trials: "
            f"{comparison.matched_count}"
        )

        print(
            "  Both successful: "
            f"{comparison.both_success_count}"
        )

        print(
            "  Planning-time difference: "
            + _format_seconds(
                comparison
                .planning_time_difference_all_attempts
            )
        )

        print(
            "  Iteration difference:     "
            + _format_number(
                comparison
                .iterations_difference_all_attempts
            )
        )

        print(
            "  Camera-movement difference: "
            + _format_mm(
                comparison
                .camera_movement_difference
            )
        )

        print(
            "  Planning-success difference: "
            + _format_rate(
                comparison
                .planning_success_difference
            )
        )

        print(
            "  Safe-navigation difference:  "
            + _format_rate(
                comparison
                .safe_navigation_success_difference
            )
        )

        print(
            "  Path-cost difference | both successful: "
            + _format_number(
                comparison
                .path_cost_difference_both_successful
            )
        )

        print()


def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the supplementary Fixed/Generic/"
            "Task-Aware planning-efficiency benchmark."
        )
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "results/"
            "supplementary_efficiency"
        ),
    )

    args = (
        parser.parse_args()
    )

    config = (
        EfficiencyBenchmarkConfig(
            repetitions_per_scenario=(
                args.repetitions
            ),
            bootstrap_samples=(
                args.bootstrap_samples
            ),
        )
    )

    result = (
        run_efficiency_benchmark(
            config,
            show_progress=True,
        )
    )

    print_efficiency_summary(
        result
    )

    (
        raw_path,
        summary_path,
    ) = (
        save_efficiency_outputs(
            result,
            output_directory=(
                args.output
            ),
        )
    )

    print(
        "Raw efficiency data saved to: "
        f"{raw_path}"
    )

    print(
        "Efficiency summary saved to: "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()