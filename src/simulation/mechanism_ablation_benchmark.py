"""End-to-end mechanism ablation for task-aware active perception.

E6 identifies which components of the task-aware viewpoint scorer are
responsible for the multi-scenario navigation improvements observed in E5.

Four matched variants are evaluated:

    Generic baseline
    Alignment-only
    Uncertainty-only
    Full task-aware

The experiment uses the same multi-scenario surgical environments, hidden
ground-truth evaluation and corrected safe-navigation endpoint introduced
by the E5 robustness benchmark.

Within every scenario repetition, all variants receive identical perception
and planner seeds. Statistical comparisons therefore use paired,
scenario-clustered bootstrap confidence intervals.
"""

from __future__ import annotations

import argparse
import csv
import json

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

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
    run_task_aware_active_perception,
)
from src.simulation.three_strategy_robustness_benchmark import (
    RobustnessScenario,
    ScenarioInputs,
    build_scenario_inputs,
    default_scenarios,
)


@dataclass(frozen=True)
class MechanismVariant:
    """One isolated task-aware scoring condition."""

    name: str

    alignment_weight: float
    uncertainty_weight: float

    def __post_init__(self) -> None:
        if self.alignment_weight < 0.0:
            raise ValueError(
                "alignment_weight must be non-negative."
            )

        if self.uncertainty_weight < 0.0:
            raise ValueError(
                "uncertainty_weight must be non-negative."
            )


@dataclass(frozen=True)
class MechanismAblationConfig:
    """Configuration for the E6 multi-scenario ablation."""

    repetitions_per_scenario: int = 10

    perception_seed_base: int = 20260910
    planner_seed_base: int = 7000

    sigma_multiplier: float = 2.0

    instrument_radius: float = 0.006
    proximal_length: float = 0.10

    task_weight: float = 2.0

    confidence_level: float = 0.95

    bootstrap_samples: int = 2000
    bootstrap_seed: int = 20260911

    def __post_init__(self) -> None:
        if self.repetitions_per_scenario <= 0:
            raise ValueError(
                "repetitions_per_scenario must be positive."
            )

        if not np.isfinite(
            self.sigma_multiplier
        ):
            raise ValueError(
                "sigma_multiplier must be finite."
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
class MechanismRecord:
    """One ablation result from one matched scenario repetition."""

    scenario_id: int
    scenario_name: str

    repetition: int

    variant: str

    perception_seed: int
    planner_seed: int

    selected_candidate_index: int

    differs_from_generic: bool

    mean_localisation_error: float
    mean_predicted_sigma: float

    camera_movement: float

    planning_success: bool

    collision_against_truth: bool
    safety_violation_against_truth: bool

    safe_navigation_success: bool

    minimum_true_safety_clearance: float


@dataclass(frozen=True)
class MetricEstimate:
    """Mean with scenario-clustered bootstrap confidence interval."""

    n: int

    mean: float

    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class MechanismSummary:
    """Aggregate statistics for one ablation condition."""

    variant: str

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

    selection_difference_from_generic_rate: MetricEstimate

    worst_scenario_safe_navigation_rate: float


@dataclass(frozen=True)
class FullVsAblatedComparison:
    """Full task-aware minus one ablated variant."""

    comparator: str

    localisation_error_difference: MetricEstimate

    predicted_sigma_difference: MetricEstimate

    camera_movement_difference: MetricEstimate

    planning_success_difference: MetricEstimate

    safe_navigation_success_difference: MetricEstimate

    collision_difference: MetricEstimate

    safety_violation_difference: MetricEstimate

    viewpoint_difference_rate: MetricEstimate


@dataclass(frozen=True)
class MechanismAblationResult:
    """Complete E6 experimental result."""

    config: MechanismAblationConfig

    scenarios: tuple[
        RobustnessScenario,
        ...,
    ]

    variants: tuple[
        MechanismVariant,
        ...,
    ]

    records: tuple[
        MechanismRecord,
        ...,
    ]

    summaries: tuple[
        MechanismSummary,
        ...,
    ]

    full_comparisons: tuple[
        FullVsAblatedComparison,
        ...,
    ]


def make_variants() -> tuple[
    MechanismVariant,
    ...,
]:
    """Return the four scientifically valid scorer ablations."""

    return (
        MechanismVariant(
            name="generic_baseline",
            alignment_weight=0.0,
            uncertainty_weight=0.0,
        ),
        MechanismVariant(
            name="alignment_only",
            alignment_weight=1.0,
            uncertainty_weight=0.0,
        ),
        MechanismVariant(
            name="uncertainty_only",
            alignment_weight=0.0,
            uncertainty_weight=1.0,
        ),
        MechanismVariant(
            name="full_task_aware",
            alignment_weight=1.0,
            uncertainty_weight=1.0,
        ),
    )


def build_variant_controller(
    *,
    inputs: ScenarioInputs,
    variant: MechanismVariant,
    config: MechanismAblationConfig,
) -> TaskAwareActivePerception:
    """Build a controller containing only the requested scoring terms."""

    generic_scorer = (
        GenericViewpointScorer(
            observation_model=(
                inputs.observation_model
            )
        )
    )

    task_scorer = (
        TaskAwareViewpointScorer(
            generic_scorer=generic_scorer,
            task=inputs.task,
            task_config=(
                TaskAwareScoringConfig(
                    task_weight=(
                        config.task_weight
                    ),
                    alignment_weight=(
                        variant.alignment_weight
                    ),
                    uncertainty_weight=(
                        variant.uncertainty_weight
                    ),
                )
            ),
        )
    )

    return TaskAwareActivePerception(
        scorer=task_scorer
    )


def _candidate_index(
    *,
    inputs: ScenarioInputs,
    selected_position: np.ndarray,
) -> int:
    """Return the candidate index matching a selected pose."""

    selected_position = np.asarray(
        selected_position,
        dtype=float,
    )

    for index, candidate in enumerate(
        inputs.candidates
    ):
        if np.allclose(
            candidate.pose.position,
            selected_position,
            atol=1e-10,
            rtol=0.0,
        ):
            return int(index)

    raise RuntimeError(
        "Selected viewpoint does not match any candidate."
    )


def _run_variant(
    *,
    inputs: ScenarioInputs,
    variant: MechanismVariant,
    config: MechanismAblationConfig,
    repetition: int,
    perception_seed: int,
    planner_seed: int,
) -> tuple[
    StrategyNavigationResult,
    int,
]:
    """Run one isolated mechanism through perception and navigation."""

    controller = build_variant_controller(
        inputs=inputs,
        variant=variant,
        config=config,
    )

    perception = (
        run_task_aware_active_perception(
            controller=controller,
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

    selected_index = _candidate_index(
        inputs=inputs,
        selected_position=(
            perception.selected_pose.position
        ),
    )

    navigation = (
        run_navigation_from_perception(
            trial=repetition,
            perception=perception,
            instrument=inputs.instrument,
            start_q=inputs.start_q,
            goal_q=inputs.goal_q,
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
            planner_seed=planner_seed,
        )
    )

    return (
        navigation,
        selected_index,
    )


def _make_record(
    *,
    scenario: RobustnessScenario,
    repetition: int,
    variant: MechanismVariant,
    perception_seed: int,
    planner_seed: int,
    navigation: StrategyNavigationResult,
    selected_candidate_index: int,
    generic_candidate_index: int,
) -> MechanismRecord:
    """Convert a matched result into one E6 record."""

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

    return MechanismRecord(
        scenario_id=(
            scenario.scenario_id
        ),
        scenario_name=(
            scenario.name
        ),
        repetition=int(
            repetition
        ),
        variant=variant.name,
        perception_seed=int(
            perception_seed
        ),
        planner_seed=int(
            planner_seed
        ),
        selected_candidate_index=int(
            selected_candidate_index
        ),
        differs_from_generic=bool(
            selected_candidate_index
            != generic_candidate_index
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
        minimum_true_safety_clearance=float(
            navigation
            .minimum_true_safety_clearance
        ),
    )


def run_mechanism_trials(
    config: MechanismAblationConfig,
    *,
    scenarios: tuple[
        RobustnessScenario,
        ...,
    ] | None = None,
    show_progress: bool = False,
) -> tuple[
    MechanismRecord,
    ...,
]:
    """Run all variants under matched multi-scenario conditions."""

    if scenarios is None:
        scenarios = default_scenarios()

    if len(scenarios) == 0:
        raise ValueError(
            "At least one scenario is required."
        )

    variants = make_variants()

    records: list[
        MechanismRecord
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

            variant_results: dict[
                str,
                tuple[
                    StrategyNavigationResult,
                    int,
                ],
            ] = {}

            for variant in variants:
                variant_results[
                    variant.name
                ] = _run_variant(
                    inputs=inputs,
                    variant=variant,
                    config=config,
                    repetition=repetition,
                    perception_seed=(
                        perception_seed
                    ),
                    planner_seed=(
                        planner_seed
                    ),
                )

            generic_candidate_index = (
                variant_results[
                    "generic_baseline"
                ][1]
            )

            for variant in variants:
                (
                    navigation,
                    selected_index,
                ) = variant_results[
                    variant.name
                ]

                records.append(
                    _make_record(
                        scenario=scenario,
                        repetition=repetition,
                        variant=variant,
                        perception_seed=(
                            perception_seed
                        ),
                        planner_seed=(
                            planner_seed
                        ),
                        navigation=navigation,
                        selected_candidate_index=(
                            selected_index
                        ),
                        generic_candidate_index=(
                            generic_candidate_index
                        ),
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
                    "matched mechanism repetitions."
                )

    return tuple(
        records
    )


def _clustered_estimate(
    values: np.ndarray,
    scenario_ids: np.ndarray,
    *,
    config: MechanismAblationConfig,
    bootstrap_seed: int,
) -> MetricEstimate:
    """Return scenario-clustered percentile-bootstrap inference."""

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

    values = values[
        finite
    ]

    scenario_ids = scenario_ids[
        finite
    ]

    n = int(
        values.size
    )

    if n == 0:
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
            n=n,
            mean=mean,
            ci_low=mean,
            ci_high=mean,
        )

    rng = np.random.default_rng(
        bootstrap_seed
    )

    bootstrap_means = np.empty(
        config.bootstrap_samples,
        dtype=float,
    )

    for sample_index in range(
        config.bootstrap_samples
    ):
        sampled_scenarios = rng.choice(
            unique_scenarios,
            size=unique_scenarios.size,
            replace=True,
        )

        sampled_values = np.concatenate(
            [
                values[
                    scenario_ids
                    == scenario_id
                ]
                for scenario_id
                in sampled_scenarios
            ]
        )

        bootstrap_means[
            sample_index
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
        n=n,
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


def _records_for_variant(
    records: tuple[
        MechanismRecord,
        ...,
    ],
    variant: str,
) -> tuple[
    MechanismRecord,
    ...,
]:
    """Return one variant ordered by matched experimental unit."""

    return tuple(
        sorted(
            (
                record
                for record in records
                if record.variant
                == variant
            ),
            key=lambda record: (
                record.scenario_id,
                record.repetition,
            ),
        )
    )


def _summary_for_variant(
    *,
    records: tuple[
        MechanismRecord,
        ...,
    ],
    variant: str,
    config: MechanismAblationConfig,
    seed_offset: int,
) -> MechanismSummary:
    """Calculate aggregate mechanism statistics."""

    selected = (
        _records_for_variant(
            records,
            variant,
        )
    )

    if len(selected) == 0:
        raise ValueError(
            f"No records found for {variant}."
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
            bootstrap_seed=(
                config.bootstrap_seed
                + seed_offset
                + offset
            ),
        )

    planning_mask = (
        values(
            "planning_success"
        )
        > 0.5
    )

    safe_values = values(
        "safe_navigation_success"
    )

    scenario_safe_rates = []

    for scenario_id in np.unique(
        scenario_ids
    ):
        mask = (
            scenario_ids
            == scenario_id
        )

        scenario_safe_rates.append(
            float(
                np.mean(
                    safe_values[
                        mask
                    ]
                )
            )
        )

    return MechanismSummary(
        variant=variant,
        scenario_count=int(
            np.unique(
                scenario_ids
            ).size
        ),
        record_count=len(
            selected
        ),
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
        selection_difference_from_generic_rate=estimate(
            "differs_from_generic",
            11,
        ),
        worst_scenario_safe_navigation_rate=float(
            np.min(
                scenario_safe_rates
            )
        ),
    )


def _full_vs_variant(
    *,
    records: tuple[
        MechanismRecord,
        ...,
    ],
    comparator: str,
    config: MechanismAblationConfig,
    seed_offset: int,
) -> FullVsAblatedComparison:
    """Calculate Full Task-Aware minus ablated-variant effects."""

    full = _records_for_variant(
        records,
        "full_task_aware",
    )

    comparison = (
        _records_for_variant(
            records,
            comparator,
        )
    )

    if len(full) != len(comparison):
        raise ValueError(
            "Matched variant record counts differ."
        )

    full_keys = [
        (
            record.scenario_id,
            record.repetition,
        )
        for record in full
    ]

    comparator_keys = [
        (
            record.scenario_id,
            record.repetition,
        )
        for record in comparison
    ]

    if full_keys != comparator_keys:
        raise ValueError(
            "Variant records are not matched."
        )

    scenario_ids = np.asarray(
        [
            record.scenario_id
            for record in full
        ],
        dtype=int,
    )

    def paired(
        attribute: str,
        offset: int,
    ) -> MetricEstimate:
        full_values = np.asarray(
            [
                getattr(
                    record,
                    attribute,
                )
                for record in full
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
                in comparison
            ],
            dtype=float,
        )

        return _clustered_estimate(
            (
                full_values
                - comparator_values
            ),
            scenario_ids,
            config=config,
            bootstrap_seed=(
                config.bootstrap_seed
                + seed_offset
                + offset
            ),
        )

    viewpoint_difference = np.asarray(
        [
            full_record
            .selected_candidate_index
            != comparator_record
            .selected_candidate_index
            for (
                full_record,
                comparator_record,
            )
            in zip(
                full,
                comparison,
            )
        ],
        dtype=float,
    )

    viewpoint_difference_rate = (
        _clustered_estimate(
            viewpoint_difference,
            scenario_ids,
            config=config,
            bootstrap_seed=(
                config.bootstrap_seed
                + seed_offset
                + 100
            ),
        )
    )

    return FullVsAblatedComparison(
        comparator=comparator,
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
        viewpoint_difference_rate=(
            viewpoint_difference_rate
        ),
    )


def analyse_mechanism_records(
    *,
    records: tuple[
        MechanismRecord,
        ...,
    ],
    scenarios: tuple[
        RobustnessScenario,
        ...,
    ],
    config: MechanismAblationConfig,
) -> MechanismAblationResult:
    """Analyse all E6 multi-scenario ablation records."""

    variants = make_variants()

    expected = (
        len(scenarios)
        * config.repetitions_per_scenario
        * len(variants)
    )

    if len(records) != expected:
        raise ValueError(
            f"Expected {expected} records, "
            f"received {len(records)}."
        )

    summaries = tuple(
        _summary_for_variant(
            records=records,
            variant=variant.name,
            config=config,
            seed_offset=(
                index
                * 100
            ),
        )
        for index, variant
        in enumerate(
            variants
        )
    )

    comparisons = tuple(
        _full_vs_variant(
            records=records,
            comparator=comparator,
            config=config,
            seed_offset=(
                1000
                + index
                * 500
            ),
        )
        for index, comparator
        in enumerate(
            (
                "generic_baseline",
                "alignment_only",
                "uncertainty_only",
            )
        )
    )

    return MechanismAblationResult(
        config=config,
        scenarios=scenarios,
        variants=variants,
        records=records,
        summaries=summaries,
        full_comparisons=(
            comparisons
        ),
    )


def run_mechanism_ablation(
    config: MechanismAblationConfig
    | None = None,
    *,
    scenarios: tuple[
        RobustnessScenario,
        ...,
    ] | None = None,
    show_progress: bool = False,
) -> MechanismAblationResult:
    """Execute the complete E6 experiment."""

    if config is None:
        config = (
            MechanismAblationConfig()
        )

    if scenarios is None:
        scenarios = (
            default_scenarios()
        )

    records = run_mechanism_trials(
        config,
        scenarios=scenarios,
        show_progress=show_progress,
    )

    return analyse_mechanism_records(
        records=records,
        scenarios=scenarios,
        config=config,
    )


def save_mechanism_outputs(
    result: MechanismAblationResult,
    *,
    output_directory: str
    | Path = "results/e6_mechanism",
) -> tuple[
    Path,
    Path,
]:
    """Save raw ablation records and aggregate results."""

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        output_directory
        / "mechanism_ablation_trials.csv"
    )

    summary_path = (
        output_directory
        / "mechanism_ablation_summary.json"
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
                asdict(
                    record
                )
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
        "variants": [
            asdict(
                variant
            )
            for variant
            in result.variants
        ],
        "summaries": [
            asdict(
                summary
            )
            for summary
            in result.summaries
        ],
        "full_comparisons": [
            asdict(
                comparison
            )
            for comparison
            in result.full_comparisons
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
    """Format probability estimate."""

    return (
        f"{metric.mean * 100.0:.1f}% "
        f"[{metric.ci_low * 100.0:.1f}, "
        f"{metric.ci_high * 100.0:.1f}]"
    )


def _format_mm(
    metric: MetricEstimate,
) -> str:
    """Format metre-valued estimate as millimetres."""

    return (
        f"{metric.mean * 1000.0:.3f} "
        f"[{metric.ci_low * 1000.0:.3f}, "
        f"{metric.ci_high * 1000.0:.3f}]"
    )


def print_mechanism_summary(
    result: MechanismAblationResult,
) -> None:
    """Print research-oriented mechanism results."""

    print()
    print(
        "End-to-End Task-Aware Mechanism Ablation"
    )
    print(
        "========================================"
    )

    print(
        f"Scenarios: {len(result.scenarios)}"
    )

    print(
        "Repetitions per scenario: "
        f"{result.config.repetitions_per_scenario}"
    )

    print()

    names = {
        "generic_baseline": (
            "Generic Baseline"
        ),
        "alignment_only": (
            "Alignment Only"
        ),
        "uncertainty_only": (
            "Uncertainty Only"
        ),
        "full_task_aware": (
            "Full Task-Aware"
        ),
    }

    for summary in result.summaries:
        print(
            names[
                summary.variant
            ]
        )

        print(
            "  Localisation error (mm):      "
            + _format_mm(
                summary
                .localisation_error
            )
        )

        print(
            "  Predicted sigma (mm):         "
            + _format_mm(
                summary
                .predicted_sigma
            )
        )

        print(
            "  Camera movement (mm):         "
            + _format_mm(
                summary
                .camera_movement
            )
        )

        print(
            "  Planning success:             "
            + _format_rate(
                summary
                .planning_success_rate
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
            "  Viewpoint change vs generic:  "
            + _format_rate(
                summary
                .selection_difference_from_generic_rate
            )
        )

        print(
            "  Worst-scenario safe success:  "
            f"{summary.worst_scenario_safe_navigation_rate * 100.0:.1f}%"
        )

        print()

    print(
        "Full Task-Aware minus Ablated Variant"
    )

    print(
        "-------------------------------------"
    )

    for comparison in (
        result.full_comparisons
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

        print(
            "  Different viewpoint rate:  "
            + _format_rate(
                comparison
                .viewpoint_difference_rate
            )
        )

        print()


def main() -> None:
    """Run E6 from the command line."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the multi-scenario end-to-end "
            "task-aware mechanism ablation."
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
        default="results/e6_mechanism",
    )

    args = parser.parse_args()

    config = (
        MechanismAblationConfig(
            repetitions_per_scenario=(
                args.repetitions
            ),
            bootstrap_samples=(
                args.bootstrap_samples
            ),
        )
    )

    result = run_mechanism_ablation(
        config,
        show_progress=True,
    )

    print_mechanism_summary(
        result
    )

    raw_path, summary_path = (
        save_mechanism_outputs(
            result,
            output_directory=(
                args.output
            ),
        )
    )

    print(
        "Raw mechanism data saved to: "
        f"{raw_path}"
    )

    print(
        "Mechanism summary saved to: "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()