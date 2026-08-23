"""Multi-scenario robustness benchmark for three-strategy navigation.

E4 established stochastic performance within one fixed surgical scene.
E5 tests whether those findings generalise across variations in:

- anatomical position;
- anatomical scale;
- required safety margin;
- initial camera viewpoint;
- visual occlusion.

Each scenario is repeated under matched stochastic seeds for:

    Fixed View
    Generic Active Perception
    Task-Aware Active Perception

The benchmark additionally introduces ``safe_navigation_success``:

    planning succeeded
    AND no hidden-ground-truth collision
    AND no hidden-ground-truth safety violation

This avoids treating planner failure as evidence of safety.

Confidence intervals use scenario-clustered bootstrap resampling so repeated
noise seeds from the same anatomy are not treated as independent scenarios.
"""

from __future__ import annotations

import argparse
import csv
import json

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from src.geometry.workspace import SphericalStructure
from src.perception.active_perception import GenericActivePerception
from src.perception.camera import CameraPose
from src.perception.observation import ViewpointObservationModel
from src.perception.task_aware_active_perception import TaskAwareActivePerception
from src.perception.task_relevance import SurgicalTask
from src.perception.viewpoints import (
    CandidateViewpoint,
    generate_candidate_viewpoints,
)
from src.robotics.instrument import SurgicalInstrument
from src.simulation.statistical_benchmark import (
    make_goal_configuration,
    make_instrument,
    make_start_configuration,
    make_true_structures,
)
from src.simulation.task_aware_benchmark import (
    TaskAwareBenchmarkConfig,
    make_generic_controller,
    make_observation_model,
    make_task_aware_controller,
)
from src.simulation.three_strategy_navigation import (
    StrategyNavigationResult,
    run_navigation_from_perception,
)
from src.simulation.three_strategy_perception import (
    run_fixed_perception,
    run_generic_active_perception,
    run_task_aware_active_perception,
)
from src.simulation.three_strategy_trial import make_trial_task


@dataclass(frozen=True)
class RobustnessScenario:
    """One deterministic surgical-scene variation."""

    scenario_id: int
    name: str

    translation: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )

    radius_scale: float = 1.0
    safety_margin_scale: float = 1.0

    initial_view_index: int = 0

    occluder_radius: float = 0.0

    def __post_init__(self) -> None:
        if self.scenario_id < 0:
            raise ValueError(
                "scenario_id must be non-negative."
            )

        translation = np.asarray(
            self.translation,
            dtype=float,
        )

        if translation.shape != (3,):
            raise ValueError(
                "translation must have shape (3,)."
            )

        if not np.all(
            np.isfinite(translation)
        ):
            raise ValueError(
                "translation must be finite."
            )

        if self.radius_scale <= 0.0:
            raise ValueError(
                "radius_scale must be positive."
            )

        if self.safety_margin_scale <= 0.0:
            raise ValueError(
                "safety_margin_scale must be positive."
            )

        if self.initial_view_index < 0:
            raise ValueError(
                "initial_view_index must be non-negative."
            )

        if self.occluder_radius < 0.0:
            raise ValueError(
                "occluder_radius must be non-negative."
            )


@dataclass(frozen=True)
class RobustnessConfig:
    """Configuration for the E5 robustness experiment."""

    repetitions_per_scenario: int = 10

    perception_seed_base: int = 20260901
    planner_seed_base: int = 5000

    sigma_multiplier: float = 2.0

    instrument_radius: float = 0.006
    proximal_length: float = 0.10

    task_weight: float = 2.0
    alignment_weight: float = 1.0

    confidence_level: float = 0.95

    bootstrap_samples: int = 2000
    bootstrap_seed: int = 20260902

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

        if not (
            0.0
            < self.confidence_level
            < 1.0
        ):
            raise ValueError(
                "confidence_level must lie between 0 and 1."
            )

        if self.bootstrap_samples <= 0:
            raise ValueError(
                "bootstrap_samples must be positive."
            )


@dataclass(frozen=True)
class ScenarioInputs:
    """Physical and perception state for one surgical scenario."""

    scenario: RobustnessScenario

    instrument: SurgicalInstrument

    start_q: np.ndarray
    goal_q: np.ndarray

    true_structures: tuple[
        SphericalStructure,
        ...,
    ]

    target: SphericalStructure

    observation_model: ViewpointObservationModel

    initial_pose: CameraPose

    candidates: tuple[
        CandidateViewpoint,
        ...,
    ]

    task: SurgicalTask

    occluders: tuple[
        SphericalStructure,
        ...,
    ]


@dataclass(frozen=True)
class RobustnessRecord:
    """One strategy result from one matched scenario repetition."""

    scenario_id: int
    scenario_name: str

    repetition: int

    strategy: str

    perception_seed: int
    planner_seed: int

    planning_success: bool

    collision_against_truth: bool

    safety_violation_against_truth: bool

    safe_navigation_success: bool

    mean_localisation_error: float

    mean_predicted_sigma: float

    camera_movement: float

    minimum_true_safety_clearance: float

    mean_planning_safety_margin: float

    maximum_planning_safety_radius: float


@dataclass(frozen=True)
class MetricEstimate:
    """Mean and scenario-clustered confidence interval."""

    n: int

    mean: float

    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class RobustnessStrategySummary:
    """Aggregate robustness statistics for one strategy."""

    strategy: str

    scenario_count: int
    record_count: int

    localisation_error: MetricEstimate

    predicted_sigma: MetricEstimate

    camera_movement: MetricEstimate

    planning_success_rate: MetricEstimate

    safe_navigation_success_rate: MetricEstimate

    collision_rate_all_trials: MetricEstimate

    safety_violation_rate_all_trials: MetricEstimate

    collision_rate_given_plan: MetricEstimate

    safety_violation_rate_given_plan: MetricEstimate

    true_clearance_given_plan: MetricEstimate

    worst_scenario_safe_navigation_rate: float


@dataclass(frozen=True)
class RobustnessPairedComparison:
    """Task-aware minus comparator matched effect."""

    comparator: str

    task_strategy: str

    localisation_error_difference: MetricEstimate

    predicted_sigma_difference: MetricEstimate

    camera_movement_difference: MetricEstimate

    planning_success_difference: MetricEstimate

    safe_navigation_success_difference: MetricEstimate

    collision_difference: MetricEstimate

    safety_violation_difference: MetricEstimate


@dataclass(frozen=True)
class RobustnessResult:
    """Complete E5 benchmark result."""

    config: RobustnessConfig

    scenarios: tuple[
        RobustnessScenario,
        ...,
    ]

    records: tuple[
        RobustnessRecord,
        ...,
    ]

    summaries: tuple[
        RobustnessStrategySummary,
        ...,
    ]

    paired_comparisons: tuple[
        RobustnessPairedComparison,
        ...,
    ]


def default_scenarios() -> tuple[
    RobustnessScenario,
    ...,
]:
    """Return ten deterministic surgical-scene variations."""

    return (
        RobustnessScenario(
            scenario_id=0,
            name="baseline",
            initial_view_index=0,
        ),
        RobustnessScenario(
            scenario_id=1,
            name="lateral_positive",
            translation=(
                0.0,
                0.008,
                0.0,
            ),
            initial_view_index=3,
        ),
        RobustnessScenario(
            scenario_id=2,
            name="lateral_negative",
            translation=(
                0.0,
                -0.008,
                0.0,
            ),
            initial_view_index=6,
        ),
        RobustnessScenario(
            scenario_id=3,
            name="superior_shift",
            translation=(
                0.0,
                0.0,
                0.006,
            ),
            initial_view_index=9,
        ),
        RobustnessScenario(
            scenario_id=4,
            name="larger_anatomy",
            radius_scale=1.10,
            initial_view_index=12,
        ),
        RobustnessScenario(
            scenario_id=5,
            name="smaller_anatomy",
            radius_scale=0.90,
            initial_view_index=15,
        ),
        RobustnessScenario(
            scenario_id=6,
            name="larger_safety_margin",
            safety_margin_scale=1.20,
            initial_view_index=18,
        ),
        RobustnessScenario(
            scenario_id=7,
            name="partial_occlusion",
            initial_view_index=4,
            occluder_radius=0.008,
        ),
        RobustnessScenario(
            scenario_id=8,
            name="strong_occlusion",
            translation=(
                0.003,
                -0.004,
                0.0,
            ),
            initial_view_index=8,
            occluder_radius=0.012,
        ),
        RobustnessScenario(
            scenario_id=9,
            name="combined_variation",
            translation=(
                -0.004,
                0.005,
                0.003,
            ),
            radius_scale=1.05,
            safety_margin_scale=1.10,
            initial_view_index=13,
            occluder_radius=0.006,
        ),
    )


def _transform_structures(
    *,
    structures: tuple[
        SphericalStructure,
        ...,
    ],
    scenario: RobustnessScenario,
) -> tuple[
    SphericalStructure,
    ...,
]:
    """Apply anatomical translation and scale variation."""

    translation = np.asarray(
        scenario.translation,
        dtype=float,
    )

    transformed = tuple(
        replace(
            structure,
            centre=(
                np.asarray(
                    structure.centre,
                    dtype=float,
                )
                + translation
            ),
            physical_radius=(
                float(
                    structure.physical_radius
                )
                * scenario.radius_scale
            ),
            safety_margin=(
                float(
                    structure.safety_margin
                )
                * scenario.safety_margin_scale
            ),
        )
        for structure in structures
    )

    return transformed


def _make_occluders(
    *,
    scenario: RobustnessScenario,
    target: SphericalStructure,
    initial_pose: CameraPose,
    template: SphericalStructure,
) -> tuple[
    SphericalStructure,
    ...,
]:
    """Create a visual occluder between camera and target."""

    if scenario.occluder_radius <= 0.0:
        return ()

    camera_position = np.asarray(
        initial_pose.position,
        dtype=float,
    )

    target_position = np.asarray(
        target.centre,
        dtype=float,
    )

    centre = (
        0.55 * camera_position
        + 0.45 * target_position
    )

    occluder = replace(
        template,
        centre=centre,
        physical_radius=float(
            scenario.occluder_radius
        ),
        safety_margin=0.0,
    )

    return (
        occluder,
    )


def build_scenario_inputs(
    scenario: RobustnessScenario,
) -> ScenarioInputs:
    """Build one varied surgical scenario."""

    instrument = make_instrument()

    start_q = make_start_configuration()

    goal_q = make_goal_configuration()

    base_structures = (
        make_true_structures()
    )

    true_structures = (
        _transform_structures(
            structures=base_structures,
            scenario=scenario,
        )
    )

    if len(true_structures) == 0:
        raise RuntimeError(
            "Scenario contains no anatomical structures."
        )

    target = true_structures[0]

    observation_model = (
        make_observation_model()
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre
    )

    if len(candidates) == 0:
        raise RuntimeError(
            "No candidate viewpoints generated."
        )

    initial_pose = (
        candidates[
            scenario.initial_view_index
            % len(candidates)
        ].pose
    )

    task = make_trial_task(
        instrument=instrument,
        start_q=start_q,
        goal_q=goal_q,
        true_structures=true_structures,
    )

    occluders = _make_occluders(
        scenario=scenario,
        target=target,
        initial_pose=initial_pose,
        template=true_structures[-1],
    )

    return ScenarioInputs(
        scenario=scenario,
        instrument=instrument,
        start_q=start_q,
        goal_q=goal_q,
        true_structures=true_structures,
        target=target,
        observation_model=observation_model,
        initial_pose=initial_pose,
        candidates=candidates,
        task=task,
        occluders=occluders,
    )


def _make_controllers(
    *,
    inputs: ScenarioInputs,
    config: RobustnessConfig,
) -> tuple[
    GenericActivePerception,
    TaskAwareActivePerception,
]:
    """Build generic and task-aware controllers for one scenario."""

    generic = make_generic_controller(
        inputs.observation_model
    )

    task_config = (
        TaskAwareBenchmarkConfig(
            trial_count=1,
            random_seed=(
                config.perception_seed_base
            ),
            task_weight=(
                config.task_weight
            ),
            alignment_weight=(
                config.alignment_weight
            ),
        )
    )

    task_aware = (
        make_task_aware_controller(
            model=inputs.observation_model,
            task=inputs.task,
            config=task_config,
        )
    )

    return (
        generic,
        task_aware,
    )


def _run_scenario_repetition(
    *,
    inputs: ScenarioInputs,
    config: RobustnessConfig,
    repetition: int,
    perception_seed: int,
    planner_seed: int,
) -> tuple[
    StrategyNavigationResult,
    StrategyNavigationResult,
    StrategyNavigationResult,
]:
    """Run all three strategies for one matched scenario repetition."""

    generic_controller, task_controller = (
        _make_controllers(
            inputs=inputs,
            config=config,
        )
    )

    fixed_perception = (
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
            occluders=inputs.occluders,
        )
    )

    generic_perception = (
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
            candidates=inputs.candidates,
            target=inputs.target,
            true_structures=(
                inputs.true_structures
            ),
            seed=perception_seed,
            occluders=inputs.occluders,
        )
    )

    task_perception = (
        run_task_aware_active_perception(
            controller=task_controller,
            observation_model=(
                inputs.observation_model
            ),
            initial_pose=(
                inputs.initial_pose
            ),
            candidates=inputs.candidates,
            target=inputs.target,
            task=inputs.task,
            true_structures=(
                inputs.true_structures
            ),
            seed=perception_seed,
            occluders=inputs.occluders,
        )
    )

    common = {
        "trial": repetition,
        "instrument": inputs.instrument,
        "start_q": inputs.start_q,
        "goal_q": inputs.goal_q,
        "true_structures": (
            inputs.true_structures
        ),
        "sigma_multiplier": (
            config.sigma_multiplier
        ),
        "instrument_radius": (
            config.instrument_radius
        ),
        "proximal_length": (
            config.proximal_length
        ),
        "planner_seed": planner_seed,
    }

    fixed = (
        run_navigation_from_perception(
            perception=fixed_perception,
            **common,
        )
    )

    generic = (
        run_navigation_from_perception(
            perception=generic_perception,
            **common,
        )
    )

    task = (
        run_navigation_from_perception(
            perception=task_perception,
            **common,
        )
    )

    return (
        fixed,
        generic,
        task,
    )


def _record_from_navigation(
    *,
    scenario: RobustnessScenario,
    repetition: int,
    perception_seed: int,
    planner_seed: int,
    navigation: StrategyNavigationResult,
) -> RobustnessRecord:
    """Convert a navigation result into one robustness record."""

    planning_success = bool(
        navigation.planning_success
    )

    collision = bool(
        navigation.collision_against_truth
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

    return RobustnessRecord(
        scenario_id=(
            scenario.scenario_id
        ),
        scenario_name=scenario.name,
        repetition=int(repetition),
        strategy=navigation.strategy,
        perception_seed=int(
            perception_seed
        ),
        planner_seed=int(
            planner_seed
        ),
        planning_success=(
            planning_success
        ),
        collision_against_truth=(
            collision
        ),
        safety_violation_against_truth=(
            violation
        ),
        safe_navigation_success=(
            safe_navigation_success
        ),
        mean_localisation_error=float(
            navigation
            .perception
            .mean_localisation_error
        ),
        mean_predicted_sigma=float(
            navigation
            .perception
            .mean_predicted_sigma
        ),
        camera_movement=float(
            navigation
            .perception
            .camera_movement
        ),
        minimum_true_safety_clearance=float(
            navigation
            .minimum_true_safety_clearance
        ),
        mean_planning_safety_margin=float(
            navigation
            .mean_planning_safety_margin
        ),
        maximum_planning_safety_radius=float(
            navigation
            .maximum_planning_safety_radius
        ),
    )


def run_robustness_trials(
    config: RobustnessConfig,
    *,
    scenarios: tuple[
        RobustnessScenario,
        ...,
    ] | None = None,
    show_progress: bool = False,
) -> tuple[
    RobustnessRecord,
    ...,
]:
    """Run all matched scenario repetitions."""

    if scenarios is None:
        scenarios = default_scenarios()

    if len(scenarios) == 0:
        raise ValueError(
            "At least one scenario is required."
        )

    records: list[
        RobustnessRecord
    ] = []

    total = (
        len(scenarios)
        * config.repetitions_per_scenario
    )

    completed = 0

    for scenario in scenarios:
        inputs = build_scenario_inputs(
            scenario
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
                config.perception_seed_base
                + seed_offset
            )

            planner_seed = (
                config.planner_seed_base
                + seed_offset
            )

            results = (
                _run_scenario_repetition(
                    inputs=inputs,
                    config=config,
                    repetition=repetition,
                    perception_seed=(
                        perception_seed
                    ),
                    planner_seed=(
                        planner_seed
                    ),
                )
            )

            for navigation in results:
                records.append(
                    _record_from_navigation(
                        scenario=scenario,
                        repetition=repetition,
                        perception_seed=(
                            perception_seed
                        ),
                        planner_seed=(
                            planner_seed
                        ),
                        navigation=navigation,
                    )
                )

            completed += 1

            if (
                show_progress
                and (
                    completed % 10 == 0
                    or completed == total
                )
            ):
                print(
                    f"Completed {completed}/{total} "
                    "matched scenario repetitions."
                )

    return tuple(
        records
    )


def _clustered_metric_estimate(
    values: np.ndarray,
    scenario_ids: np.ndarray,
    *,
    confidence_level: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> MetricEstimate:
    """Calculate a scenario-clustered bootstrap estimate."""

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

    values = values[finite]

    scenario_ids = (
        scenario_ids[finite]
    )

    if values.size == 0:
        return MetricEstimate(
            n=0,
            mean=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
        )

    mean = float(
        np.mean(values)
    )

    unique_scenarios = np.unique(
        scenario_ids
    )

    if unique_scenarios.size == 1:
        return MetricEstimate(
            n=int(values.size),
            mean=mean,
            ci_low=mean,
            ci_high=mean,
        )

    rng = np.random.default_rng(
        bootstrap_seed
    )

    bootstrap_means = np.empty(
        bootstrap_samples,
        dtype=float,
    )

    for index in range(
        bootstrap_samples
    ):
        sampled_scenarios = (
            rng.choice(
                unique_scenarios,
                size=(
                    unique_scenarios.size
                ),
                replace=True,
            )
        )

        sampled_values: list[
            np.ndarray
        ] = []

        for scenario_id in (
            sampled_scenarios
        ):
            sampled_values.append(
                values[
                    scenario_ids
                    == scenario_id
                ]
            )

        combined = np.concatenate(
            sampled_values
        )

        bootstrap_means[index] = (
            np.mean(combined)
        )

    alpha = (
        1.0
        - confidence_level
    )

    return MetricEstimate(
        n=int(values.size),
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
                1.0 - alpha / 2.0,
            )
        ),
    )


def _strategy_records(
    records: tuple[
        RobustnessRecord,
        ...,
    ],
    strategy: str,
) -> tuple[
    RobustnessRecord,
        ...,
]:
    """Return records for one strategy."""

    return tuple(
        sorted(
            (
                record
                for record in records
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
        RobustnessRecord,
        ...,
    ],
    strategy: str,
    config: RobustnessConfig,
    seed_offset: int,
) -> RobustnessStrategySummary:
    """Calculate one strategy's multi-scenario statistics."""

    selected = _strategy_records(
        records,
        strategy,
    )

    if len(selected) == 0:
        raise ValueError(
            f"No records for strategy {strategy}."
        )

    scenario_ids = np.asarray(
        [
            record.scenario_id
            for record in selected
        ],
        dtype=int,
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
                for record in selected
            ],
            dtype=float,
        )

    def estimate(
        attribute: str,
        offset: int,
        mask: np.ndarray | None = None,
    ) -> MetricEstimate:
        metric_values = values(
            attribute
        )

        metric_scenarios = (
            scenario_ids
        )

        if mask is not None:
            metric_values = (
                metric_values[mask]
            )

            metric_scenarios = (
                metric_scenarios[mask]
            )

        return (
            _clustered_metric_estimate(
                metric_values,
                metric_scenarios,
                confidence_level=(
                    config.confidence_level
                ),
                bootstrap_samples=(
                    config.bootstrap_samples
                ),
                bootstrap_seed=(
                    config.bootstrap_seed
                    + seed_offset
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

    scenario_rates = []

    for scenario_id in np.unique(
        scenario_ids
    ):
        scenario_mask = (
            scenario_ids
            == scenario_id
        )

        scenario_rates.append(
            float(
                np.mean(
                    values(
                        "safe_navigation_success"
                    )[scenario_mask]
                )
            )
        )

    return RobustnessStrategySummary(
        strategy=strategy,
        scenario_count=int(
            np.unique(
                scenario_ids
            ).size
        ),
        record_count=len(selected),
        localisation_error=estimate(
            "mean_localisation_error",
            1,
        ),
        predicted_sigma=estimate(
            "mean_predicted_sigma",
            2,
        ),
        camera_movement=estimate(
            "camera_movement",
            3,
        ),
        planning_success_rate=estimate(
            "planning_success",
            4,
        ),
        safe_navigation_success_rate=estimate(
            "safe_navigation_success",
            5,
        ),
        collision_rate_all_trials=estimate(
            "collision_against_truth",
            6,
        ),
        safety_violation_rate_all_trials=estimate(
            "safety_violation_against_truth",
            7,
        ),
        collision_rate_given_plan=estimate(
            "collision_against_truth",
            8,
            planning_mask,
        ),
        safety_violation_rate_given_plan=estimate(
            "safety_violation_against_truth",
            9,
            planning_mask,
        ),
        true_clearance_given_plan=estimate(
            "minimum_true_safety_clearance",
            10,
            planning_mask,
        ),
        worst_scenario_safe_navigation_rate=float(
            np.min(
                scenario_rates
            )
        ),
    )


def _paired_comparison(
    *,
    records: tuple[
        RobustnessRecord,
        ...,
    ],
    comparator: str,
    config: RobustnessConfig,
    seed_offset: int,
) -> RobustnessPairedComparison:
    """Calculate Task-Aware minus Comparator paired effects."""

    task_records = _strategy_records(
        records,
        "task_aware_active",
    )

    comparator_records = (
        _strategy_records(
            records,
            comparator,
        )
    )

    if (
        len(task_records)
        != len(comparator_records)
    ):
        raise ValueError(
            "Matched strategy record counts differ."
        )

    task_keys = [
        (
            record.scenario_id,
            record.repetition,
        )
        for record in task_records
    ]

    comparator_keys = [
        (
            record.scenario_id,
            record.repetition,
        )
        for record in comparator_records
    ]

    if task_keys != comparator_keys:
        raise ValueError(
            "Strategy records are not correctly matched."
        )

    scenario_ids = np.asarray(
        [
            record.scenario_id
            for record in task_records
        ],
        dtype=int,
    )

    def paired(
        attribute: str,
        offset: int,
    ) -> MetricEstimate:
        task_values = np.asarray(
            [
                getattr(
                    record,
                    attribute,
                )
                for record in task_records
            ],
            dtype=float,
        )

        comparator_values = np.asarray(
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

        differences = (
            task_values
            - comparator_values
        )

        return (
            _clustered_metric_estimate(
                differences,
                scenario_ids,
                confidence_level=(
                    config.confidence_level
                ),
                bootstrap_samples=(
                    config.bootstrap_samples
                ),
                bootstrap_seed=(
                    config.bootstrap_seed
                    + seed_offset
                    + offset
                ),
            )
        )

    return RobustnessPairedComparison(
        comparator=comparator,
        task_strategy=(
            "task_aware_active"
        ),
        localisation_error_difference=paired(
            "mean_localisation_error",
            1,
        ),
        predicted_sigma_difference=paired(
            "mean_predicted_sigma",
            2,
        ),
        camera_movement_difference=paired(
            "camera_movement",
            3,
        ),
        planning_success_difference=paired(
            "planning_success",
            4,
        ),
        safe_navigation_success_difference=paired(
            "safe_navigation_success",
            5,
        ),
        collision_difference=paired(
            "collision_against_truth",
            6,
        ),
        safety_violation_difference=paired(
            "safety_violation_against_truth",
            7,
        ),
    )


def analyse_robustness_records(
    *,
    records: tuple[
        RobustnessRecord,
        ...,
    ],
    scenarios: tuple[
        RobustnessScenario,
        ...,
    ],
    config: RobustnessConfig,
) -> RobustnessResult:
    """Aggregate the complete E5 robustness experiment."""

    expected = (
        len(scenarios)
        * config.repetitions_per_scenario
        * 3
    )

    if len(records) != expected:
        raise ValueError(
            f"Expected {expected} records, "
            f"received {len(records)}."
        )

    strategies = (
        "fixed",
        "generic_active",
        "task_aware_active",
    )

    summaries = tuple(
        _strategy_summary(
            records=records,
            strategy=strategy,
            config=config,
            seed_offset=(
                index * 100
            ),
        )
        for index, strategy
        in enumerate(
            strategies
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
            comparator="generic_active",
            config=config,
            seed_offset=2000,
        ),
    )

    return RobustnessResult(
        config=config,
        scenarios=scenarios,
        records=records,
        summaries=summaries,
        paired_comparisons=(
            comparisons
        ),
    )


def run_robustness_benchmark(
    config: RobustnessConfig
    | None = None,
    *,
    scenarios: tuple[
        RobustnessScenario,
        ...,
    ] | None = None,
    show_progress: bool = False,
) -> RobustnessResult:
    """Run and analyse the complete E5 benchmark."""

    if config is None:
        config = RobustnessConfig()

    if scenarios is None:
        scenarios = default_scenarios()

    records = run_robustness_trials(
        config,
        scenarios=scenarios,
        show_progress=show_progress,
    )

    return analyse_robustness_records(
        records=records,
        scenarios=scenarios,
        config=config,
    )


def save_robustness_outputs(
    result: RobustnessResult,
    *,
    output_directory: str
    | Path = "results/e5_robustness",
) -> tuple[
    Path,
    Path,
]:
    """Save raw E5 records and statistical results."""

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        output_directory
        / "three_strategy_robustness_trials.csv"
    )

    summary_path = (
        output_directory
        / "three_strategy_robustness_summary.json"
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
                    result.records[0]
                ).keys()
            ),
        )

        writer.writeheader()

        for record in result.records:
            writer.writerow(
                asdict(record)
            )

    payload = {
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
    return (
        f"{metric.mean * 100.0:.1f}% "
        f"[{metric.ci_low * 100.0:.1f}, "
        f"{metric.ci_high * 100.0:.1f}]"
    )


def _format_mm(
    metric: MetricEstimate,
) -> str:
    return (
        f"{metric.mean * 1000.0:.3f} "
        f"[{metric.ci_low * 1000.0:.3f}, "
        f"{metric.ci_high * 1000.0:.3f}]"
    )


def print_robustness_summary(
    result: RobustnessResult,
) -> None:
    """Print a compact research-oriented E5 summary."""

    print()
    print(
        "Three-Strategy Multi-Scenario Robustness Benchmark"
    )

    print(
        "=================================================="
    )

    print(
        f"Scenarios: {len(result.scenarios)}"
    )

    print(
        "Repetitions per scenario: "
        f"{result.config.repetitions_per_scenario}"
    )

    print(
        "Matched scenario repetitions: "
        f"{len(result.scenarios) * result.config.repetitions_per_scenario}"
    )

    print()

    names = {
        "fixed": "Fixed View",
        "generic_active": "Generic Active",
        "task_aware_active": "Task-Aware Active",
    }

    for summary in result.summaries:
        print(
            names[
                summary.strategy
            ]
        )

        print(
            "  Localisation error (mm):      "
            + _format_mm(
                summary.localisation_error
            )
        )

        print(
            "  Predicted sigma (mm):         "
            + _format_mm(
                summary.predicted_sigma
            )
        )

        print(
            "  Camera movement (mm):         "
            + _format_mm(
                summary.camera_movement
            )
        )

        print(
            "  Planning success:             "
            + _format_rate(
                summary.planning_success_rate
            )
        )

        print(
            "  SAFE navigation success:      "
            + _format_rate(
                summary
                .safe_navigation_success_rate
            )
        )

        print(
            "  Collision | successful plan:  "
            + _format_rate(
                summary
                .collision_rate_given_plan
            )
        )

        print(
            "  Violation | successful plan:  "
            + _format_rate(
                summary
                .safety_violation_rate_given_plan
            )
        )

        print(
            "  True clearance | plan (mm):   "
            + _format_mm(
                summary
                .true_clearance_given_plan
            )
        )

        print(
            "  Worst-scenario safe success:  "
            f"{summary.worst_scenario_safe_navigation_rate * 100:.1f}%"
        )

        print()

    print(
        "Paired Task-Aware minus Comparator Effects"
    )

    print(
        "------------------------------------------"
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
            "  Localisation error Δ (mm): "
            + _format_mm(
                comparison
                .localisation_error_difference
            )
        )

        print(
            "  Predicted sigma Δ (mm):    "
            + _format_mm(
                comparison
                .predicted_sigma_difference
            )
        )

        print(
            "  Camera movement Δ (mm):    "
            + _format_mm(
                comparison
                .camera_movement_difference
            )
        )

        print(
            "  Planning success Δ:        "
            + _format_rate(
                comparison
                .planning_success_difference
            )
        )

        print(
            "  SAFE navigation Δ:         "
            + _format_rate(
                comparison
                .safe_navigation_success_difference
            )
        )

        print()


def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(
        description=(
            "Run E5 multi-scenario "
            "three-strategy robustness validation."
        )
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/e5_robustness",
    )

    args = parser.parse_args()

    config = RobustnessConfig(
        repetitions_per_scenario=(
            args.repetitions
        ),
        bootstrap_samples=(
            args.bootstrap_samples
        ),
    )

    result = run_robustness_benchmark(
        config,
        show_progress=True,
    )

    print_robustness_summary(
        result
    )

    raw_path, summary_path = (
        save_robustness_outputs(
            result,
            output_directory=args.output,
        )
    )

    print(
        f"Raw robustness data saved to: {raw_path}"
    )

    print(
        "Robustness summary saved to: "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()