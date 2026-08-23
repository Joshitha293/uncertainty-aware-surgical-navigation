"""Statistical benchmark for the end-to-end three-strategy experiment.

This module repeatedly evaluates:

    Fixed View
    Generic Active Perception
    Task-Aware Active Perception

under matched trial conditions.

Within each trial, all strategies receive the same perception seed and
planner seed. This permits paired statistical comparison while preserving
the complete perception -> uncertainty-aware planning -> hidden-ground-truth
evaluation pipeline established by the three-strategy trial experiment.
"""

from __future__ import annotations

import argparse
import csv
import json

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.simulation.three_strategy_trial import (
    ThreeStrategyTrialConfig,
    run_three_strategy_trial,
)


@dataclass(frozen=True)
class ThreeStrategyStatisticalConfig:
    """Configuration for repeated matched three-strategy trials."""

    trial_count: int = 100

    perception_seed_base: int = 20260823
    planner_seed_base: int = 2000

    sigma_multiplier: float = 2.0

    instrument_radius: float = 0.006
    proximal_length: float = 0.10

    task_weight: float = 2.0
    alignment_weight: float = 1.0

    confidence_level: float = 0.95

    bootstrap_samples: int = 2000
    bootstrap_seed: int = 20260824

    def __post_init__(self) -> None:
        if self.trial_count <= 0:
            raise ValueError(
                "trial_count must be positive."
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
class StrategyTrialRecord:
    """Raw measurements from one strategy in one matched trial."""

    trial: int
    strategy: str

    perception_seed: int
    planner_seed: int

    mean_localisation_error: float
    mean_predicted_sigma: float

    camera_movement: float

    planning_success: bool

    collision_against_truth: bool
    safety_violation_against_truth: bool

    minimum_true_safety_clearance: float

    mean_planning_safety_margin: float
    maximum_planning_safety_radius: float


@dataclass(frozen=True)
class MetricEstimate:
    """Mean metric value with a bootstrap confidence interval."""

    n: int

    mean: float

    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class StrategySummary:
    """Aggregate statistics for one perception strategy."""

    strategy: str
    trial_count: int

    localisation_error: MetricEstimate
    predicted_sigma: MetricEstimate

    camera_movement: MetricEstimate

    planning_success_rate: MetricEstimate

    collision_rate: MetricEstimate
    safety_violation_rate: MetricEstimate

    minimum_true_safety_clearance: MetricEstimate

    planning_safety_margin: MetricEstimate
    maximum_planning_safety_radius: MetricEstimate


@dataclass(frozen=True)
class PairedComparison:
    """Task-aware minus comparator paired effect estimates."""

    comparator: str
    task_strategy: str

    localisation_error_difference: MetricEstimate

    predicted_sigma_difference: MetricEstimate

    camera_movement_difference: MetricEstimate

    planning_success_rate_difference: MetricEstimate

    collision_rate_difference: MetricEstimate

    safety_violation_rate_difference: MetricEstimate

    minimum_true_safety_clearance_difference: MetricEstimate


@dataclass(frozen=True)
class ThreeStrategyStatisticalResult:
    """Complete output of the repeated statistical benchmark."""

    config: ThreeStrategyStatisticalConfig

    records: tuple[
        StrategyTrialRecord,
        ...,
    ]

    summaries: tuple[
        StrategySummary,
        ...,
    ]

    paired_comparisons: tuple[
        PairedComparison,
        ...,
    ]


def _metric_estimate(
    values: np.ndarray,
    *,
    confidence_level: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> MetricEstimate:
    """Estimate a mean and percentile bootstrap confidence interval."""

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
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

    if n == 1:
        return MetricEstimate(
            n=1,
            mean=mean,
            ci_low=mean,
            ci_high=mean,
        )

    rng = np.random.default_rng(
        bootstrap_seed
    )

    sampled_indices = rng.integers(
        low=0,
        high=n,
        size=(
            bootstrap_samples,
            n,
        ),
    )

    bootstrap_means = np.mean(
        values[
            sampled_indices
        ],
        axis=1,
    )

    alpha = (
        1.0
        - confidence_level
    )

    ci_low = float(
        np.quantile(
            bootstrap_means,
            alpha / 2.0,
        )
    )

    ci_high = float(
        np.quantile(
            bootstrap_means,
            1.0 - alpha / 2.0,
        )
    )

    return MetricEstimate(
        n=n,
        mean=mean,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def _paired_metric_estimate(
    task_values: np.ndarray,
    comparator_values: np.ndarray,
    *,
    confidence_level: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> MetricEstimate:
    """Estimate Task-Aware minus Comparator paired mean difference."""

    task_values = np.asarray(
        task_values,
        dtype=float,
    )

    comparator_values = np.asarray(
        comparator_values,
        dtype=float,
    )

    if (
        task_values.shape
        != comparator_values.shape
    ):
        raise ValueError(
            "Paired arrays must have identical shapes."
        )

    finite = (
        np.isfinite(task_values)
        & np.isfinite(
            comparator_values
        )
    )

    differences = (
        task_values[finite]
        - comparator_values[finite]
    )

    return _metric_estimate(
        differences,
        confidence_level=confidence_level,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )


def _record_from_navigation(
    *,
    trial: int,
    perception_seed: int,
    planner_seed: int,
    navigation,
) -> StrategyTrialRecord:
    """Convert one navigation result into a statistical record."""

    return StrategyTrialRecord(
        trial=int(trial),
        strategy=navigation.strategy,
        perception_seed=int(
            perception_seed
        ),
        planner_seed=int(
            planner_seed
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
        planning_success=bool(
            navigation.planning_success
        ),
        collision_against_truth=bool(
            navigation
            .collision_against_truth
        ),
        safety_violation_against_truth=bool(
            navigation
            .safety_violation_against_truth
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


def run_matched_trials(
    config: ThreeStrategyStatisticalConfig,
    *,
    show_progress: bool = False,
) -> tuple[
    StrategyTrialRecord,
    ...,
]:
    """Run all matched end-to-end trials."""

    records: list[
        StrategyTrialRecord
    ] = []

    for trial in range(
        config.trial_count
    ):
        perception_seed = (
            config.perception_seed_base
            + trial
        )

        planner_seed = (
            config.planner_seed_base
            + trial
        )

        trial_config = (
            ThreeStrategyTrialConfig(
                trial=trial,
                perception_seed=(
                    perception_seed
                ),
                planner_seed=(
                    planner_seed
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
                task_weight=(
                    config.task_weight
                ),
                alignment_weight=(
                    config.alignment_weight
                ),
            )
        )

        result = (
            run_three_strategy_trial(
                trial_config
            )
        )

        for navigation in result.results:
            records.append(
                _record_from_navigation(
                    trial=trial,
                    perception_seed=(
                        perception_seed
                    ),
                    planner_seed=(
                        planner_seed
                    ),
                    navigation=navigation,
                )
            )

        if (
            show_progress
            and (
                (trial + 1) % 10 == 0
                or trial + 1
                == config.trial_count
            )
        ):
            print(
                "Completed "
                f"{trial + 1}/"
                f"{config.trial_count} "
                "matched trials."
            )

    return tuple(
        records
    )


def _records_for_strategy(
    records: tuple[
        StrategyTrialRecord,
        ...,
    ],
    strategy: str,
) -> tuple[
    StrategyTrialRecord,
    ...,
]:
    """Return records for one strategy ordered by trial."""

    selected = tuple(
        sorted(
            (
                record
                for record in records
                if record.strategy
                == strategy
            ),
            key=lambda record: (
                record.trial
            ),
        )
    )

    return selected


def _summary_for_strategy(
    *,
    records: tuple[
        StrategyTrialRecord,
        ...,
    ],
    strategy: str,
    config: ThreeStrategyStatisticalConfig,
    seed_offset: int,
) -> StrategySummary:
    """Compute aggregate statistics for one strategy."""

    selected = (
        _records_for_strategy(
            records,
            strategy,
        )
    )

    if len(selected) == 0:
        raise ValueError(
            f"No records found for {strategy}."
        )

    localisation_error = np.asarray(
        [
            record.mean_localisation_error
            for record in selected
        ],
        dtype=float,
    )

    predicted_sigma = np.asarray(
        [
            record.mean_predicted_sigma
            for record in selected
        ],
        dtype=float,
    )

    camera_movement = np.asarray(
        [
            record.camera_movement
            for record in selected
        ],
        dtype=float,
    )

    planning_success = np.asarray(
        [
            record.planning_success
            for record in selected
        ],
        dtype=float,
    )

    collision = np.asarray(
        [
            record.collision_against_truth
            for record in selected
        ],
        dtype=float,
    )

    safety_violation = np.asarray(
        [
            record
            .safety_violation_against_truth
            for record in selected
        ],
        dtype=float,
    )

    clearance = np.asarray(
        [
            record
            .minimum_true_safety_clearance
            for record in selected
        ],
        dtype=float,
    )

    planning_margin = np.asarray(
        [
            record.mean_planning_safety_margin
            for record in selected
        ],
        dtype=float,
    )

    maximum_radius = np.asarray(
        [
            record.maximum_planning_safety_radius
            for record in selected
        ],
        dtype=float,
    )

    def estimate(
        values: np.ndarray,
        metric_offset: int,
    ) -> MetricEstimate:
        return _metric_estimate(
            values,
            confidence_level=(
                config.confidence_level
            ),
            bootstrap_samples=(
                config.bootstrap_samples
            ),
            bootstrap_seed=(
                config.bootstrap_seed
                + seed_offset
                + metric_offset
            ),
        )

    return StrategySummary(
        strategy=strategy,
        trial_count=len(selected),
        localisation_error=estimate(
            localisation_error,
            1,
        ),
        predicted_sigma=estimate(
            predicted_sigma,
            2,
        ),
        camera_movement=estimate(
            camera_movement,
            3,
        ),
        planning_success_rate=estimate(
            planning_success,
            4,
        ),
        collision_rate=estimate(
            collision,
            5,
        ),
        safety_violation_rate=estimate(
            safety_violation,
            6,
        ),
        minimum_true_safety_clearance=estimate(
            clearance,
            7,
        ),
        planning_safety_margin=estimate(
            planning_margin,
            8,
        ),
        maximum_planning_safety_radius=estimate(
            maximum_radius,
            9,
        ),
    )


def _paired_comparison(
    *,
    records: tuple[
        StrategyTrialRecord,
        ...,
    ],
    comparator: str,
    config: ThreeStrategyStatisticalConfig,
    seed_offset: int,
) -> PairedComparison:
    """Compare Task-Aware Active against one comparator."""

    task_records = (
        _records_for_strategy(
            records,
            "task_aware_active",
        )
    )

    comparator_records = (
        _records_for_strategy(
            records,
            comparator,
        )
    )

    if (
        len(task_records)
        != len(comparator_records)
    ):
        raise ValueError(
            "Matched strategies have different trial counts."
        )

    task_trials = [
        record.trial
        for record in task_records
    ]

    comparator_trials = [
        record.trial
        for record in comparator_records
    ]

    if (
        task_trials
        != comparator_trials
    ):
        raise ValueError(
            "Strategy records are not matched by trial."
        )

    def values(
        selected,
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

    def paired(
        attribute: str,
        metric_offset: int,
    ) -> MetricEstimate:
        return _paired_metric_estimate(
            values(
                task_records,
                attribute,
            ),
            values(
                comparator_records,
                attribute,
            ),
            confidence_level=(
                config.confidence_level
            ),
            bootstrap_samples=(
                config.bootstrap_samples
            ),
            bootstrap_seed=(
                config.bootstrap_seed
                + seed_offset
                + metric_offset
            ),
        )

    return PairedComparison(
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
        planning_success_rate_difference=paired(
            "planning_success",
            4,
        ),
        collision_rate_difference=paired(
            "collision_against_truth",
            5,
        ),
        safety_violation_rate_difference=paired(
            "safety_violation_against_truth",
            6,
        ),
        minimum_true_safety_clearance_difference=paired(
            "minimum_true_safety_clearance",
            7,
        ),
    )


def analyse_records(
    *,
    records: tuple[
        StrategyTrialRecord,
        ...,
    ],
    config: ThreeStrategyStatisticalConfig,
) -> ThreeStrategyStatisticalResult:
    """Aggregate raw matched trials and calculate paired effects."""

    expected_count = (
        config.trial_count
        * 3
    )

    if len(records) != expected_count:
        raise ValueError(
            "Expected "
            f"{expected_count} records, "
            f"received {len(records)}."
        )

    strategies = (
        "fixed",
        "generic_active",
        "task_aware_active",
    )

    summaries = tuple(
        _summary_for_strategy(
            records=records,
            strategy=strategy,
            config=config,
            seed_offset=(
                index * 100
            ),
        )
        for index, strategy
        in enumerate(strategies)
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

    return ThreeStrategyStatisticalResult(
        config=config,
        records=records,
        summaries=summaries,
        paired_comparisons=(
            comparisons
        ),
    )


def run_statistical_benchmark(
    config: ThreeStrategyStatisticalConfig
    | None = None,
    *,
    show_progress: bool = False,
) -> ThreeStrategyStatisticalResult:
    """Execute and analyse the complete statistical benchmark."""

    if config is None:
        config = (
            ThreeStrategyStatisticalConfig()
        )

    records = run_matched_trials(
        config,
        show_progress=show_progress,
    )

    return analyse_records(
        records=records,
        config=config,
    )


def save_benchmark_outputs(
    result: ThreeStrategyStatisticalResult,
    *,
    output_directory: str
    | Path = "results",
) -> tuple[
    Path,
    Path,
]:
    """Save raw trial data and statistical summaries."""

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        output_directory
        / "three_strategy_trials.csv"
    )

    summary_path = (
        output_directory
        / "three_strategy_statistical_summary.json"
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

    summary_payload = {
        "config": asdict(
            result.config
        ),
        "summaries": [
            asdict(summary)
            for summary
            in result.summaries
        ],
        "paired_comparisons": [
            asdict(comparison)
            for comparison
            in result.paired_comparisons
        ],
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary_payload,
            handle,
            indent=2,
            allow_nan=True,
        )

    return (
        raw_path,
        summary_path,
    )


def _format_metric_mm(
    metric: MetricEstimate,
) -> str:
    """Format a metric stored in metres as millimetres."""

    return (
        f"{metric.mean * 1000.0:.3f} "
        f"[{metric.ci_low * 1000.0:.3f}, "
        f"{metric.ci_high * 1000.0:.3f}]"
    )


def _format_rate(
    metric: MetricEstimate,
) -> str:
    """Format probability estimates as percentages."""

    return (
        f"{metric.mean * 100.0:.1f}% "
        f"[{metric.ci_low * 100.0:.1f}, "
        f"{metric.ci_high * 100.0:.1f}]"
    )


def print_statistical_summary(
    result: ThreeStrategyStatisticalResult,
) -> None:
    """Print publication-oriented aggregate results."""

    print()
    print(
        "Three-Strategy Statistical Benchmark"
    )
    print(
        "===================================="
    )

    print(
        f"Matched trials: "
        f"{result.config.trial_count}"
    )

    print(
        "Confidence level: "
        f"{result.config.confidence_level * 100:.1f}%"
    )

    print()

    display_names = {
        "fixed": "Fixed View",
        "generic_active": (
            "Generic Active"
        ),
        "task_aware_active": (
            "Task-Aware Active"
        ),
    }

    for summary in result.summaries:
        print(
            display_names[
                summary.strategy
            ]
        )

        print(
            "  Localisation error (mm): "
            + _format_metric_mm(
                summary.localisation_error
            )
        )

        print(
            "  Predicted sigma (mm):    "
            + _format_metric_mm(
                summary.predicted_sigma
            )
        )

        print(
            "  Camera movement (mm):    "
            + _format_metric_mm(
                summary.camera_movement
            )
        )

        print(
            "  Planning success:        "
            + _format_rate(
                summary.planning_success_rate
            )
        )

        print(
            "  Collision rate:          "
            + _format_rate(
                summary.collision_rate
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
            "  True clearance (mm):     "
            + _format_metric_mm(
                summary
                .minimum_true_safety_clearance
            )
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
            + display_names[
                comparison.comparator
            ]
        )

        print(
            "  Localisation error Δ (mm): "
            + _format_metric_mm(
                comparison
                .localisation_error_difference
            )
        )

        print(
            "  Predicted sigma Δ (mm):    "
            + _format_metric_mm(
                comparison
                .predicted_sigma_difference
            )
        )

        print(
            "  Camera movement Δ (mm):    "
            + _format_metric_mm(
                comparison
                .camera_movement_difference
            )
        )

        print(
            "  Planning success Δ:        "
            + _format_rate(
                comparison
                .planning_success_rate_difference
            )
        )

        print(
            "  Collision-rate Δ:          "
            + _format_rate(
                comparison
                .collision_rate_difference
            )
        )

        print(
            "  Safety-violation Δ:        "
            + _format_rate(
                comparison
                .safety_violation_rate_difference
            )
        )

        print(
            "  True-clearance Δ (mm):     "
            + _format_metric_mm(
                comparison
                .minimum_true_safety_clearance_difference
            )
        )

        print()


def main() -> None:
    """Run the benchmark from the command line."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the matched three-strategy "
            "statistical surgical-navigation benchmark."
        )
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help=(
            "Number of matched trials "
            "(default: 100)."
        ),
    )

    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help=(
            "Bootstrap resamples used for confidence "
            "intervals (default: 2000)."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help=(
            "Directory used for CSV and JSON outputs."
        ),
    )

    args = parser.parse_args()

    config = (
        ThreeStrategyStatisticalConfig(
            trial_count=args.trials,
            bootstrap_samples=(
                args.bootstrap_samples
            ),
        )
    )

    result = run_statistical_benchmark(
        config,
        show_progress=True,
    )

    print_statistical_summary(
        result
    )

    raw_path, summary_path = (
        save_benchmark_outputs(
            result,
            output_directory=args.output,
        )
    )

    print(
        f"Raw trial data saved to: "
        f"{raw_path}"
    )

    print(
        f"Statistical summary saved to: "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()