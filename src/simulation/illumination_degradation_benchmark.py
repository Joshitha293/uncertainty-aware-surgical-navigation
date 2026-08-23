"""Supplementary simulated illumination-degradation benchmark.

This experiment evaluates Fixed View, Generic Active Perception and
Task-Aware Active Perception under controlled visual-quality degradation.

The existing observation model does not simulate photons, exposure, image
formation or a laparoscopic sensor. Therefore illumination is represented
conservatively as a synthetic image-quality factor L in (0, 1]:

    sigma_degraded = sigma_nominal / sqrt(L)

This is analogous to reduced signal quality increasing localisation noise.

The experiment scales:
- base localisation sigma;
- occluded localisation sigma;
- invisible localisation sigma.

The geometric distance/angle observation model, camera geometry and
occlusion logic are otherwise unchanged.

This must therefore be described as a synthetic illumination/visual-quality
stress test, not as a photometric or clinically validated laparoscopic
illumination model.
"""

from __future__ import annotations

import argparse
import csv
import json

from dataclasses import (
    asdict,
    dataclass,
    replace,
)
from pathlib import Path

import numpy as np

from src.perception.active_perception import (
    GenericActivePerception,
)
from src.perception.observation import (
    ViewpointObservationModel,
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
    run_navigation_from_perception,
)
from src.simulation.three_strategy_perception import (
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


@dataclass(frozen=True)
class IlluminationCondition:
    """One controlled simulated visual-quality condition."""

    name: str

    quality: float

    def __post_init__(self) -> None:
        if not np.isfinite(
            self.quality
        ):
            raise ValueError(
                "quality must be finite."
            )

        if (
            self.quality <= 0.0
            or self.quality > 1.0
        ):
            raise ValueError(
                "quality must lie in (0, 1]."
            )

    @property
    def sigma_multiplier(
        self,
    ) -> float:
        """Return observation-noise multiplier."""

        return float(
            1.0
            / np.sqrt(
                self.quality
            )
        )


@dataclass(frozen=True)
class IlluminationBenchmarkConfig:
    """Configuration for the illumination-degradation benchmark."""

    repetitions_per_condition: int = 2

    perception_seed_base: int = 20261010

    planner_seed_base: int = 11000

    sigma_multiplier: float = 2.0

    instrument_radius: float = 0.006

    proximal_length: float = 0.10

    task_weight: float = 2.0

    alignment_weight: float = 1.0

    uncertainty_weight: float = 1.0

    bootstrap_samples: int = 1000

    bootstrap_seed: int = 20261011

    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if (
            self.repetitions_per_condition
            <= 0
        ):
            raise ValueError(
                "repetitions_per_condition must be positive."
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
class IlluminationRecord:
    """One end-to-end illumination-strategy result."""

    scenario_id: int

    scenario_name: str

    illumination_name: str

    illumination_quality: float

    observation_sigma_multiplier: float

    repetition: int

    strategy: str

    perception_seed: int

    planner_seed: int

    mean_localisation_error: float

    mean_predicted_sigma: float

    camera_movement: float

    planning_success: bool

    safe_navigation_success: bool

    collision_against_truth: bool

    safety_violation_against_truth: bool

    minimum_true_safety_clearance: float


@dataclass(frozen=True)
class MetricEstimate:
    """Mean and confidence interval."""

    n: int

    mean: float

    ci_low: float

    ci_high: float


@dataclass(frozen=True)
class IlluminationStrategySummary:
    """Aggregate result for a strategy/illumination pair."""

    strategy: str

    illumination_name: str

    illumination_quality: float

    record_count: int

    localisation_error: MetricEstimate

    predicted_sigma: MetricEstimate

    camera_movement: MetricEstimate

    planning_success_rate: MetricEstimate

    safe_navigation_success_rate: MetricEstimate

    collision_rate: MetricEstimate

    safety_violation_rate: MetricEstimate

    true_clearance_given_plan: MetricEstimate


@dataclass(frozen=True)
class IlluminationResult:
    """Complete illumination experiment."""

    config: IlluminationBenchmarkConfig

    conditions: tuple[
        IlluminationCondition,
        ...,
    ]

    scenarios: tuple[
        RobustnessScenario,
        ...,
    ]

    records: tuple[
        IlluminationRecord,
        ...,
    ]

    summaries: tuple[
        IlluminationStrategySummary,
        ...,
    ]


def default_illumination_conditions() -> tuple[
    IlluminationCondition,
    ...,
]:
    """Return controlled illumination-quality levels."""

    return (
        IlluminationCondition(
            name="normal",
            quality=1.00,
        ),
        IlluminationCondition(
            name="mild_degradation",
            quality=0.75,
        ),
        IlluminationCondition(
            name="moderate_degradation",
            quality=0.50,
        ),
        IlluminationCondition(
            name="severe_degradation",
            quality=0.25,
        ),
    )


def apply_illumination_condition(
    *,
    inputs: ScenarioInputs,
    condition: IlluminationCondition,
) -> ScenarioInputs:
    """Apply synthetic illumination degradation.

    The observation-noise terms are multiplied by

        1 / sqrt(quality)

    while geometry, anatomy, candidate viewpoints and task remain
    unchanged.
    """

    base_model = (
        inputs.observation_model
    )

    base_config = (
        base_model.config
    )

    factor = (
        condition.sigma_multiplier
    )

    modified_config = replace(
        base_config,
        base_sigma=(
            float(
                base_config.base_sigma
            )
            * factor
        ),
        occluded_sigma=(
            float(
                base_config.occluded_sigma
            )
            * factor
        ),
        invisible_sigma=(
            float(
                base_config.invisible_sigma
            )
            * factor
        ),
    )

    modified_model = (
        ViewpointObservationModel(
            camera=base_model.camera,
            config=modified_config,
        )
    )

    return replace(
        inputs,
        observation_model=modified_model,
    )


def _build_controllers(
    *,
    inputs: ScenarioInputs,
    config: IlluminationBenchmarkConfig,
) -> tuple[
    GenericActivePerception,
    TaskAwareActivePerception,
]:
    """Build generic and task-aware controllers."""

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


def _run_matched_unit(
    *,
    inputs: ScenarioInputs,
    config: IlluminationBenchmarkConfig,
    scenario: RobustnessScenario,
    condition: IlluminationCondition,
    repetition: int,
    perception_seed: int,
    planner_seed: int,
) -> tuple[
    IlluminationRecord,
    ...,
]:
    """Run all three strategies under one matched condition."""

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

    perception_results = (
        fixed,
        generic,
        task_aware,
    )

    records: list[
        IlluminationRecord
    ] = []

    for perception in (
        perception_results
    ):
        navigation = (
            run_navigation_from_perception(
                trial=repetition,
                perception=perception,
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
                    inputs.true_structures
                ),
                sigma_multiplier=(
                    config.sigma_multiplier
                ),
                instrument_radius=(
                    config.instrument_radius
                ),
                proximal_length=(
                    config.proximal_length
                ),
                planner_seed=(
                    planner_seed
                ),
            )
        )

        planning_success = bool(
            navigation.planning_success
        )

        collision = bool(
            navigation
            .collision_against_truth
        )

        violation = bool(
            navigation
            .safety_violation_against_truth
        )

        safe_navigation_success = (
            planning_success
            and not collision
            and not violation
        )

        clearance = float(
            navigation
            .minimum_true_safety_clearance
        )

        records.append(
            IlluminationRecord(
                scenario_id=(
                    scenario.scenario_id
                ),
                scenario_name=(
                    scenario.name
                ),
                illumination_name=(
                    condition.name
                ),
                illumination_quality=(
                    condition.quality
                ),
                observation_sigma_multiplier=(
                    condition
                    .sigma_multiplier
                ),
                repetition=int(
                    repetition
                ),
                strategy=(
                    navigation.strategy
                ),
                perception_seed=int(
                    perception_seed
                ),
                planner_seed=int(
                    planner_seed
                ),
                mean_localisation_error=float(
                    perception
                    .mean_localisation_error
                ),
                mean_predicted_sigma=float(
                    perception
                    .mean_predicted_sigma
                ),
                camera_movement=float(
                    perception
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
                minimum_true_safety_clearance=(
                    clearance
                ),
            )
        )

    return tuple(
        records
    )


def run_illumination_trials(
    config: IlluminationBenchmarkConfig,
    *,
    scenarios: tuple[
        RobustnessScenario,
        ...,
    ] | None = None,
    conditions: tuple[
        IlluminationCondition,
        ...,
    ] | None = None,
    show_progress: bool = False,
) -> tuple[
    IlluminationRecord,
    ...,
]:
    """Run matched illumination-degradation trials."""

    if scenarios is None:
        scenarios = (
            default_scenarios()
        )

    if conditions is None:
        conditions = (
            default_illumination_conditions()
        )

    if len(
        scenarios
    ) == 0:
        raise ValueError(
            "At least one scenario is required."
        )

    if len(
        conditions
    ) == 0:
        raise ValueError(
            "At least one illumination condition is required."
        )

    records: list[
        IlluminationRecord
    ] = []

    total_units = (
        len(
            scenarios
        )
        * len(
            conditions
        )
        * config
        .repetitions_per_condition
    )

    completed = 0

    for scenario in (
        scenarios
    ):
        base_inputs = (
            build_scenario_inputs(
                scenario
            )
        )

        for (
            condition_index,
            condition,
        ) in enumerate(
            conditions
        ):
            inputs = (
                apply_illumination_condition(
                    inputs=base_inputs,
                    condition=condition,
                )
            )

            for repetition in range(
                config
                .repetitions_per_condition
            ):
                seed_offset = (
                    scenario.scenario_id
                    * 10_000
                    + condition_index
                    * 100
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

                records.extend(
                    _run_matched_unit(
                        inputs=inputs,
                        config=config,
                        scenario=scenario,
                        condition=condition,
                        repetition=(
                            repetition
                        ),
                        perception_seed=(
                            perception_seed
                        ),
                        planner_seed=(
                            planner_seed
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
                        "matched illumination units."
                    )

    return tuple(
        records
    )


def _clustered_estimate(
    values: np.ndarray,
    scenario_ids: np.ndarray,
    *,
    config: IlluminationBenchmarkConfig,
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

    if (
        unique_scenarios.size
        == 1
    ):
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

    bootstrap_means = (
        np.empty(
            config.bootstrap_samples,
            dtype=float,
        )
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

        combined = (
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
                combined
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


def analyse_illumination_records(
    *,
    records: tuple[
        IlluminationRecord,
        ...,
    ],
    config: IlluminationBenchmarkConfig,
    conditions: tuple[
        IlluminationCondition,
        ...,
    ],
) -> tuple[
    IlluminationStrategySummary,
    ...,
]:
    """Aggregate all strategy/illumination combinations."""

    if len(
        records
    ) == 0:
        raise ValueError(
            "records must not be empty."
        )

    strategies = (
        "fixed",
        "generic_active",
        "task_aware_active",
    )

    summaries: list[
        IlluminationStrategySummary
    ] = []

    seed_counter = 0

    for condition in (
        conditions
    ):
        for strategy in (
            strategies
        ):
            selected = [
                record
                for record
                in records
                if (
                    record.strategy
                    == strategy
                    and record
                    .illumination_name
                    == condition.name
                )
            ]

            if len(
                selected
            ) == 0:
                raise ValueError(
                    "Missing records for "
                    f"{condition.name}/"
                    f"{strategy}."
                )

            scenario_ids = (
                np.asarray(
                    [
                        record
                        .scenario_id
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
                mask: (
                    np.ndarray
                    | None
                ) = None,
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

                return (
                    _clustered_estimate(
                        metric_values,
                        metric_scenarios,
                        config=config,
                        seed_offset=(
                            seed_counter
                            + offset
                        ),
                    )
                )

            planning_mask = (
                values(
                    "planning_success"
                )
                > 0.5
            )

            summaries.append(
                IlluminationStrategySummary(
                    strategy=(
                        strategy
                    ),
                    illumination_name=(
                        condition.name
                    ),
                    illumination_quality=(
                        condition.quality
                    ),
                    record_count=len(
                        selected
                    ),
                    localisation_error=(
                        estimate(
                            "mean_localisation_error",
                            1,
                        )
                    ),
                    predicted_sigma=(
                        estimate(
                            "mean_predicted_sigma",
                            2,
                        )
                    ),
                    camera_movement=(
                        estimate(
                            "camera_movement",
                            3,
                        )
                    ),
                    planning_success_rate=(
                        estimate(
                            "planning_success",
                            4,
                        )
                    ),
                    safe_navigation_success_rate=(
                        estimate(
                            "safe_navigation_success",
                            5,
                        )
                    ),
                    collision_rate=(
                        estimate(
                            "collision_against_truth",
                            6,
                        )
                    ),
                    safety_violation_rate=(
                        estimate(
                            "safety_violation_against_truth",
                            7,
                        )
                    ),
                    true_clearance_given_plan=(
                        estimate(
                            "minimum_true_safety_clearance",
                            8,
                            planning_mask,
                        )
                    ),
                )
            )

            seed_counter += 100

    return tuple(
        summaries
    )


def run_illumination_benchmark(
    config: (
        IlluminationBenchmarkConfig
        | None
    ) = None,
    *,
    scenarios: tuple[
        RobustnessScenario,
        ...,
    ] | None = None,
    conditions: tuple[
        IlluminationCondition,
        ...,
    ] | None = None,
    show_progress: bool = False,
) -> IlluminationResult:
    """Run and analyse the illumination benchmark."""

    if config is None:
        config = (
            IlluminationBenchmarkConfig()
        )

    if scenarios is None:
        scenarios = (
            default_scenarios()
        )

    if conditions is None:
        conditions = (
            default_illumination_conditions()
        )

    records = (
        run_illumination_trials(
            config,
            scenarios=scenarios,
            conditions=conditions,
            show_progress=(
                show_progress
            ),
        )
    )

    summaries = (
        analyse_illumination_records(
            records=records,
            config=config,
            conditions=conditions,
        )
    )

    return IlluminationResult(
        config=config,
        conditions=conditions,
        scenarios=scenarios,
        records=records,
        summaries=summaries,
    )


def save_illumination_outputs(
    result: IlluminationResult,
    *,
    output_directory: (
        str
        | Path
    ) = (
        "results/"
        "supplementary_illumination"
    ),
) -> tuple[
    Path,
    Path,
]:
    """Save raw and summary illumination results."""

    if len(
        result.records
    ) == 0:
        raise ValueError(
            "Cannot save an empty result."
        )

    output_directory = (
        Path(
            output_directory
        )
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        output_directory
        / "illumination_trials.csv"
    )

    summary_path = (
        output_directory
        / "illumination_summary.json"
    )

    with raw_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = (
            csv.DictWriter(
                handle,
                fieldnames=list(
                    asdict(
                        result
                        .records[
                            0
                        ]
                    ).keys()
                ),
            )
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
        "model": {
            "description": (
                "Synthetic illumination/"
                "visual-quality degradation "
                "represented by increased "
                "localisation noise."
            ),
            "equation": (
                "sigma_degraded = "
                "sigma_nominal / sqrt(quality)"
            ),
            "physical_camera_model": False,
            "clinical_validation": False,
        },
        "config": asdict(
            result.config
        ),
        "conditions": [
            {
                **asdict(
                    condition
                ),
                "sigma_multiplier": (
                    condition
                    .sigma_multiplier
                ),
            }
            for condition
            in result.conditions
        ],
        "summaries": [
            asdict(
                summary
            )
            for summary
            in result.summaries
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


def _format_mm(
    metric: MetricEstimate,
) -> str:
    """Format a distance metric in millimetres."""

    if metric.n == 0:
        return "N/A"

    return (
        f"{metric.mean * 1000.0:.3f} "
        f"["
        f"{metric.ci_low * 1000.0:.3f}, "
        f"{metric.ci_high * 1000.0:.3f}"
        f"]"
    )


def _format_rate(
    metric: MetricEstimate,
) -> str:
    """Format a probability as a percentage."""

    if metric.n == 0:
        return "N/A"

    return (
        f"{metric.mean * 100.0:.1f}% "
        f"["
        f"{metric.ci_low * 100.0:.1f}, "
        f"{metric.ci_high * 100.0:.1f}"
        f"]"
    )


def print_illumination_summary(
    result: IlluminationResult,
) -> None:
    """Print research-oriented illumination results."""

    print()

    print(
        "Synthetic Illumination-Degradation Benchmark"
    )

    print(
        "============================================"
    )

    print(
        f"Scenarios: "
        f"{len(result.scenarios)}"
    )

    print(
        "Repetitions per condition: "
        f"{result.config.repetitions_per_condition}"
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

    for condition in (
        result.conditions
    ):
        print(
            f"{condition.name} "
            f"(quality="
            f"{condition.quality:.2f}, "
            "sigma multiplier="
            f"{condition.sigma_multiplier:.3f})"
        )

        print(
            "-" * 72
        )

        for summary in (
            result.summaries
        ):
            if (
                summary
                .illumination_name
                != condition.name
            ):
                continue

            print(
                names[
                    summary.strategy
                ]
            )

            print(
                "  Localisation error (mm): "
                + _format_mm(
                    summary
                    .localisation_error
                )
            )

            print(
                "  Predicted sigma (mm):    "
                + _format_mm(
                    summary
                    .predicted_sigma
                )
            )

            print(
                "  Camera movement (mm):    "
                + _format_mm(
                    summary
                    .camera_movement
                )
            )

            print(
                "  Planning success:        "
                + _format_rate(
                    summary
                    .planning_success_rate
                )
            )

            print(
                "  SAFE navigation success: "
                + _format_rate(
                    summary
                    .safe_navigation_success_rate
                )
            )

            print(
                "  Collision rate:          "
                + _format_rate(
                    summary
                    .collision_rate
                )
            )

            print(
                "  Safety violation rate:   "
                + _format_rate(
                    summary
                    .safety_violation_rate
                )
            )

            print(
                "  Clearance | plan (mm):   "
                + _format_mm(
                    summary
                    .true_clearance_given_plan
                )
            )

            print()

        print()


def main() -> None:
    """Command-line entry point."""

    parser = (
        argparse.ArgumentParser(
            description=(
                "Run the supplementary synthetic "
                "illumination-degradation benchmark."
            )
        )
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=2,
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
            "supplementary_illumination"
        ),
    )

    args = (
        parser.parse_args()
    )

    config = (
        IlluminationBenchmarkConfig(
            repetitions_per_condition=(
                args.repetitions
            ),
            bootstrap_samples=(
                args.bootstrap_samples
            ),
        )
    )

    result = (
        run_illumination_benchmark(
            config,
            show_progress=True,
        )
    )

    print_illumination_summary(
        result
    )

    (
        raw_path,
        summary_path,
    ) = (
        save_illumination_outputs(
            result,
            output_directory=(
                args.output
            ),
        )
    )

    print(
        "Raw illumination data saved to: "
        f"{raw_path}"
    )

    print(
        "Illumination summary saved to: "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()