"""Frozen statistical analysis for the final Phase 1 experiment.

This module implements the statistical plan declared BEFORE held-out
execution in:

    results/phase1_protocol/frozen_phase1_protocol.json

Frozen protocol SHA-256:

    dc6537d3ff75832ccbe48c9b2690c8e966711b31806076cfbc2721488dbf6361

Primary comparison
------------------
Full Task-Aware
versus
Movement-Budget-Matched Generic

Primary endpoint
----------------
Safe-navigation success rate.

Inferential unit
----------------
Held-out scenario.

The 10 stochastic repetitions nested within each scenario are first
aggregated to a scenario-level success proportion. They are NOT treated
as 300 independent experimental scenes.

Primary inference
-----------------
- effect:
      Task-Aware minus Budget-Matched Generic
      scenario-level safe-navigation success rate

- two-sided paired scenario-level sign-flip permutation test

- 95% scenario-cluster bootstrap confidence interval

- superiority only when:
      p < 0.05
      AND
      95% CI lies entirely above zero

The implementation is deterministic under the seeds frozen in the
Phase 1 protocol.

Post-held-out implementation note
---------------------------------
The original raw-record validator incorrectly required path_cost to be
finite for failed planning trials. That conflicted with the already
frozen protocol, under which path cost is only analysed for paired trials
where BOTH strategies successfully plan.

A failed planner may legitimately retain path_cost = +inf. The validator
therefore now requires:

- movement/perception metrics to always be finite;
- successful planning -> finite path_cost;
- failed planning -> path_cost may be +inf, but not NaN.

This correction does not alter any frozen endpoint, comparison, seed,
weight, scenario, statistical test, superiority criterion, or raw result.
"""

from __future__ import annotations

import argparse
import csv
import json

from collections import defaultdict

from dataclasses import (
    asdict,
    dataclass,
)
from pathlib import Path
from typing import Iterable

import numpy as np

from src.simulation.phase1_frozen_protocol import (
    BOOTSTRAP_RESAMPLES,
    PERMUTATION_RESAMPLES,
    StrategyId,
    build_frozen_phase1_protocol,
)


FROZEN_PROTOCOL_SHA256 = (
    "dc6537d3ff75832ccbe48c9b2690c8e9"
    "66711b31806076cfbc2721488dbf6361"
)


PRIMARY_STRATEGY = (
    StrategyId.FULL_TASK_AWARE.value
)

PRIMARY_COMPARATOR = (
    StrategyId.BUDGET_MATCHED_GENERIC.value
)

PRIMARY_OUTCOME = (
    "safe_navigation_success_rate"
)


@dataclass(frozen=True)
class StatisticalRecord:
    """Minimal raw record required by the statistical engine."""

    scenario_id: int

    repetition: int

    strategy: str

    initial_perception_seed: int
    final_perception_seed: int
    random_viewpoint_seed: int
    planner_seed: int

    camera_movement: float

    mean_localisation_error: float
    mean_predicted_sigma: float

    planning_success: bool

    collision_against_truth: bool

    safety_violation_against_truth: bool

    safe_navigation_success: bool

    path_cost: float


@dataclass(frozen=True)
class ScenarioEffect:
    """One paired scenario-level effect."""

    scenario_id: int

    task_value: float

    comparator_value: float

    difference: float


@dataclass(frozen=True)
class PermutationResult:
    """Paired sign-flip permutation-test output."""

    observed_mean_difference: float

    p_value_two_sided: float

    resamples: int

    seed: int


@dataclass(frozen=True)
class BootstrapResult:
    """Scenario-cluster bootstrap interval."""

    observed_mean_difference: float

    confidence_level: float

    lower: float

    upper: float

    resamples: int

    seed: int


@dataclass(frozen=True)
class PrimaryAnalysis:
    """Frozen primary inferential result."""

    outcome: str

    task_strategy: str

    comparator_strategy: str

    scenario_count: int

    task_mean_success_rate: float

    comparator_mean_success_rate: float

    absolute_effect: float

    permutation_p_value: float

    confidence_interval_lower: float

    confidence_interval_upper: float

    alpha: float

    superiority_established: bool


@dataclass(frozen=True)
class MovementBudgetDiagnostic:
    """Held-out movement-budget diagnostic."""

    scenario_count: int

    task_mean_camera_movement: float

    comparator_mean_camera_movement: float

    absolute_gap: float

    relative_gap: float

    previous_validation_tolerance: float

    within_previous_validation_tolerance: bool


@dataclass(frozen=True)
class Phase1StatisticalAnalysis:
    """Complete first-stage Phase 1 analysis."""

    protocol_sha256: str

    primary: PrimaryAnalysis

    movement_budget: MovementBudgetDiagnostic


def _parse_bool(
    value: object,
) -> bool:
    """Parse a CSV boolean strictly."""

    if isinstance(
        value,
        bool,
    ):
        return value

    text = str(
        value
    ).strip().lower()

    if text in (
        "true",
        "1",
    ):
        return True

    if text in (
        "false",
        "0",
    ):
        return False

    raise ValueError(
        f"Invalid boolean value: {value!r}"
    )


def load_statistical_records(
    path: str | Path,
) -> tuple[
    StatisticalRecord,
    ...,
]:
    """Load final raw CSV records."""

    path = Path(
        path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Raw Phase 1 results not found: {path}"
        )

    records: list[
        StatisticalRecord
    ] = []

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        for row in reader:
            records.append(
                StatisticalRecord(
                    scenario_id=int(
                        row[
                            "scenario_id"
                        ]
                    ),
                    repetition=int(
                        row[
                            "repetition"
                        ]
                    ),
                    strategy=str(
                        row[
                            "strategy"
                        ]
                    ),
                    initial_perception_seed=int(
                        row[
                            "initial_perception_seed"
                        ]
                    ),
                    final_perception_seed=int(
                        row[
                            "final_perception_seed"
                        ]
                    ),
                    random_viewpoint_seed=int(
                        row[
                            "random_viewpoint_seed"
                        ]
                    ),
                    planner_seed=int(
                        row[
                            "planner_seed"
                        ]
                    ),
                    camera_movement=float(
                        row[
                            "camera_movement"
                        ]
                    ),
                    mean_localisation_error=float(
                        row[
                            "mean_localisation_error"
                        ]
                    ),
                    mean_predicted_sigma=float(
                        row[
                            "mean_predicted_sigma"
                        ]
                    ),
                    planning_success=(
                        _parse_bool(
                            row[
                                "planning_success"
                            ]
                        )
                    ),
                    collision_against_truth=(
                        _parse_bool(
                            row[
                                "collision_against_truth"
                            ]
                        )
                    ),
                    safety_violation_against_truth=(
                        _parse_bool(
                            row[
                                "safety_violation_against_truth"
                            ]
                        )
                    ),
                    safe_navigation_success=(
                        _parse_bool(
                            row[
                                "safe_navigation_success"
                            ]
                        )
                    ),
                    path_cost=float(
                        row[
                            "path_cost"
                        ]
                    ),
                )
            )

    if len(
        records
    ) == 0:
        raise ValueError(
            "Raw results file contains no records."
        )

    return tuple(
        records
    )


def validate_raw_records(
    records: tuple[
        StatisticalRecord,
        ...,
    ],
    *,
    require_complete_strategy_family: bool = True,
) -> None:
    """Verify paired-design integrity before statistical analysis.

    Important:
    path_cost is conditionally defined.

    A successful planner must return a finite path cost.

    A failed planner may legitimately preserve an undefined path cost as
    positive infinity. This is allowed because the frozen protocol compares
    path cost only for pairs where both strategies successfully planned.
    """

    if len(
        records
    ) == 0:
        raise ValueError(
            "records must not be empty."
        )

    expected_strategies = {
        strategy.value
        for strategy
        in StrategyId
    }

    grouped: dict[
        tuple[
            int,
            int,
        ],
        list[
            StatisticalRecord
        ],
    ] = defaultdict(
        list
    )

    seen_keys = set()

    for record in records:
        key = (
            record.scenario_id,
            record.repetition,
            record.strategy,
        )

        if key in seen_keys:
            raise ValueError(
                "Duplicate scenario/repetition/strategy "
                f"record detected: {key}"
            )

        seen_keys.add(
            key
        )

        grouped[
            (
                record.scenario_id,
                record.repetition,
            )
        ].append(
            record
        )

        # These quantities are defined for every strategy/trial and must
        # therefore always remain finite.
        always_finite_values = (
            record.camera_movement,
            record.mean_localisation_error,
            record.mean_predicted_sigma,
        )

        if not all(
            np.isfinite(
                value
            )
            for value
            in always_finite_values
        ):
            raise ValueError(
                "Raw statistical record contains non-finite "
                "movement or perception values."
            )

        # Path cost is only defined when planning succeeds.
        #
        # Failed planners may legitimately report +inf. We preserve that
        # value rather than inventing an arbitrary finite failure cost,
        # because the frozen statistical protocol explicitly compares path
        # costs only when BOTH paired strategies successfully plan.
        if record.planning_success:
            if not np.isfinite(
                record.path_cost
            ):
                raise ValueError(
                    "Successful planning record contains "
                    "a non-finite path cost."
                )

        else:
            if np.isnan(
                record.path_cost
            ):
                raise ValueError(
                    "Failed planning record contains NaN path cost."
                )

    for trial_key, trial_records in grouped.items():
        strategies = {
            record.strategy
            for record
            in trial_records
        }

        if (
            require_complete_strategy_family
            and strategies
            != expected_strategies
        ):
            raise ValueError(
                "Incomplete strategy family for "
                f"trial {trial_key}. "
                f"Observed={sorted(strategies)}"
            )

        initial_seeds = {
            record.initial_perception_seed
            for record
            in trial_records
        }

        final_seeds = {
            record.final_perception_seed
            for record
            in trial_records
        }

        planner_seeds = {
            record.planner_seed
            for record
            in trial_records
        }

        if len(
            initial_seeds
        ) != 1:
            raise ValueError(
                "Initial perception seed is not matched "
                f"within trial {trial_key}."
            )

        if len(
            final_seeds
        ) != 1:
            raise ValueError(
                "Final perception seed is not matched "
                f"within trial {trial_key}."
            )

        if len(
            planner_seeds
        ) != 1:
            raise ValueError(
                "Planner seed is not matched "
                f"within trial {trial_key}."
            )


def _records_for_strategy(
    records: Iterable[
        StatisticalRecord
    ],
    *,
    strategy: str,
) -> tuple[
    StatisticalRecord,
    ...,
]:
    """Filter records by strategy."""

    return tuple(
        record
        for record
        in records
        if record.strategy
        == strategy
    )


def scenario_success_rates(
    records: tuple[
        StatisticalRecord,
        ...,
    ],
    *,
    strategy: str,
) -> dict[
    int,
    float,
]:
    """Return safe-navigation success proportion per scenario."""

    strategy_records = (
        _records_for_strategy(
            records,
            strategy=strategy,
        )
    )

    grouped: dict[
        int,
        list[
            float
        ],
    ] = defaultdict(
        list
    )

    for record in strategy_records:
        grouped[
            record.scenario_id
        ].append(
            float(
                record
                .safe_navigation_success
            )
        )

    if len(
        grouped
    ) == 0:
        raise ValueError(
            f"No records found for strategy {strategy!r}."
        )

    return {
        scenario_id: float(
            np.mean(
                values
            )
        )
        for scenario_id, values
        in grouped.items()
    }


def scenario_planning_success_rates(
    records: tuple[
        StatisticalRecord,
        ...,
    ],
    *,
    strategy: str,
) -> dict[
    int,
    float,
]:
    """Return planning-success proportion per scenario."""

    strategy_records = (
        _records_for_strategy(
            records,
            strategy=strategy,
        )
    )

    grouped: dict[
        int,
        list[
            float
        ],
    ] = defaultdict(
        list
    )

    for record in strategy_records:
        grouped[
            record.scenario_id
        ].append(
            float(
                record
                .planning_success
            )
        )

    if len(
        grouped
    ) == 0:
        raise ValueError(
            f"No records found for strategy {strategy!r}."
        )

    return {
        scenario_id: float(
            np.mean(
                values
            )
        )
        for scenario_id, values
        in grouped.items()
    }


def scenario_mean_metric(
    records: tuple[
        StatisticalRecord,
        ...,
    ],
    *,
    strategy: str,
    attribute: str,
) -> dict[
    int,
    float,
]:
    """Aggregate one continuous outcome to scenario level."""

    allowed = {
        "camera_movement",
        "mean_localisation_error",
        "mean_predicted_sigma",
    }

    if attribute not in allowed:
        raise ValueError(
            f"Unsupported metric: {attribute}"
        )

    strategy_records = (
        _records_for_strategy(
            records,
            strategy=strategy,
        )
    )

    grouped: dict[
        int,
        list[
            float
        ],
    ] = defaultdict(
        list
    )

    for record in strategy_records:
        grouped[
            record.scenario_id
        ].append(
            float(
                getattr(
                    record,
                    attribute,
                )
            )
        )

    if len(
        grouped
    ) == 0:
        raise ValueError(
            f"No records found for strategy {strategy!r}."
        )

    return {
        scenario_id: float(
            np.mean(
                values
            )
        )
        for scenario_id, values
        in grouped.items()
    }


def paired_scenario_effects(
    *,
    task_values: dict[
        int,
        float,
    ],
    comparator_values: dict[
        int,
        float,
    ],
) -> tuple[
    ScenarioEffect,
    ...,
]:
    """Construct paired scenario-level Task minus Comparator effects."""

    task_ids = set(
        task_values
    )

    comparator_ids = set(
        comparator_values
    )

    if task_ids != comparator_ids:
        raise ValueError(
            "Task and comparator scenario sets differ."
        )

    if len(
        task_ids
    ) == 0:
        raise ValueError(
            "No paired scenarios available."
        )

    effects = []

    for scenario_id in sorted(
        task_ids
    ):
        task_value = float(
            task_values[
                scenario_id
            ]
        )

        comparator_value = float(
            comparator_values[
                scenario_id
            ]
        )

        effects.append(
            ScenarioEffect(
                scenario_id=int(
                    scenario_id
                ),
                task_value=(
                    task_value
                ),
                comparator_value=(
                    comparator_value
                ),
                difference=float(
                    task_value
                    - comparator_value
                ),
            )
        )

    return tuple(
        effects
    )


def sign_flip_permutation_test(
    differences: np.ndarray,
    *,
    resamples: int = (
        PERMUTATION_RESAMPLES
    ),
    seed: int,
) -> PermutationResult:
    """Two-sided paired sign-flip permutation test."""

    differences = np.asarray(
        differences,
        dtype=float,
    )

    if (
        differences.ndim != 1
        or differences.size == 0
    ):
        raise ValueError(
            "differences must be a non-empty "
            "one-dimensional array."
        )

    if not np.all(
        np.isfinite(
            differences
        )
    ):
        raise ValueError(
            "differences must be finite."
        )

    if resamples <= 0:
        raise ValueError(
            "resamples must be positive."
        )

    observed = float(
        np.mean(
            differences
        )
    )

    rng = np.random.default_rng(
        seed
    )

    exceedances = 0

    absolute_observed = abs(
        observed
    )

    # Chunking avoids allocating a large 100000 x N matrix.
    chunk_size = 10_000

    remaining = int(
        resamples
    )

    while remaining > 0:
        current = min(
            chunk_size,
            remaining,
        )

        signs = rng.choice(
            np.asarray(
                [
                    -1.0,
                    1.0,
                ],
                dtype=float,
            ),
            size=(
                current,
                differences.size,
            ),
            replace=True,
        )

        null_means = np.mean(
            signs
            * differences[
                None,
                :
            ],
            axis=1,
        )

        exceedances += int(
            np.sum(
                np.abs(
                    null_means
                )
                >= (
                    absolute_observed
                    - 1e-15
                )
            )
        )

        remaining -= current

    # +1 correction avoids a Monte-Carlo p-value of exactly zero.
    p_value = (
        exceedances
        + 1
    ) / (
        resamples
        + 1
    )

    return PermutationResult(
        observed_mean_difference=(
            observed
        ),
        p_value_two_sided=float(
            p_value
        ),
        resamples=int(
            resamples
        ),
        seed=int(
            seed
        ),
    )


def cluster_bootstrap_mean_difference(
    differences: np.ndarray,
    *,
    resamples: int = (
        BOOTSTRAP_RESAMPLES
    ),
    seed: int,
    confidence_level: float = 0.95,
) -> BootstrapResult:
    """Bootstrap paired scenario-level effects."""

    differences = np.asarray(
        differences,
        dtype=float,
    )

    if (
        differences.ndim != 1
        or differences.size == 0
    ):
        raise ValueError(
            "differences must be a non-empty "
            "one-dimensional array."
        )

    if not np.all(
        np.isfinite(
            differences
        )
    ):
        raise ValueError(
            "differences must be finite."
        )

    if resamples <= 0:
        raise ValueError(
            "resamples must be positive."
        )

    if not (
        0.0
        < confidence_level
        < 1.0
    ):
        raise ValueError(
            "confidence_level must be between 0 and 1."
        )

    rng = np.random.default_rng(
        seed
    )

    scenario_count = int(
        differences.size
    )

    indices = rng.integers(
        low=0,
        high=scenario_count,
        size=(
            resamples,
            scenario_count,
        ),
    )

    bootstrap_means = np.mean(
        differences[
            indices
        ],
        axis=1,
    )

    tail = (
        1.0
        - confidence_level
    ) / 2.0

    lower = float(
        np.quantile(
            bootstrap_means,
            tail,
        )
    )

    upper = float(
        np.quantile(
            bootstrap_means,
            1.0 - tail,
        )
    )

    return BootstrapResult(
        observed_mean_difference=float(
            np.mean(
                differences
            )
        ),
        confidence_level=float(
            confidence_level
        ),
        lower=lower,
        upper=upper,
        resamples=int(
            resamples
        ),
        seed=int(
            seed
        ),
    )


def holm_adjust(
    p_values: tuple[
        float,
        ...,
    ],
) -> tuple[
    float,
    ...,
]:
    """Return Holm-adjusted p-values in original order."""

    if len(
        p_values
    ) == 0:
        return tuple()

    values = np.asarray(
        p_values,
        dtype=float,
    )

    if (
        np.any(
            values < 0.0
        )
        or np.any(
            values > 1.0
        )
        or not np.all(
            np.isfinite(
                values
            )
        )
    ):
        raise ValueError(
            "p-values must be finite and between 0 and 1."
        )

    order = np.argsort(
        values
    )

    adjusted_sorted = np.zeros(
        len(
            values
        ),
        dtype=float,
    )

    running_max = 0.0

    m = len(
        values
    )

    for rank, original_index in enumerate(
        order
    ):
        multiplier = (
            m
            - rank
        )

        raw_adjusted = (
            multiplier
            * values[
                original_index
            ]
        )

        running_max = max(
            running_max,
            raw_adjusted,
        )

        adjusted_sorted[
            rank
        ] = min(
            1.0,
            running_max,
        )

    adjusted = np.zeros(
        len(
            values
        ),
        dtype=float,
    )

    for rank, original_index in enumerate(
        order
    ):
        adjusted[
            original_index
        ] = adjusted_sorted[
            rank
        ]

    return tuple(
        float(
            value
        )
        for value
        in adjusted
    )


def paired_success_path_cost_differences(
    records: tuple[
        StatisticalRecord,
        ...,
    ],
    *,
    task_strategy: str,
    comparator_strategy: str,
) -> dict[
    int,
    float,
]:
    """Scenario-level path-cost effects for jointly successful pairs only.

    A repetition contributes only when BOTH strategies produced a
    successful plan.

    Returned effect:

        Task path cost - Comparator path cost

    Lower values favour Task-Aware.
    """

    lookup: dict[
        tuple[
            int,
            int,
            str,
        ],
        StatisticalRecord,
    ] = {
        (
            record.scenario_id,
            record.repetition,
            record.strategy,
        ):
        record
        for record
        in records
    }

    scenario_repetitions = {
        (
            record.scenario_id,
            record.repetition,
        )
        for record
        in records
        if record.strategy
        in (
            task_strategy,
            comparator_strategy,
        )
    }

    grouped: dict[
        int,
        list[
            float
        ],
    ] = defaultdict(
        list
    )

    for scenario_id, repetition in sorted(
        scenario_repetitions
    ):
        task = lookup.get(
            (
                scenario_id,
                repetition,
                task_strategy,
            )
        )

        comparator = lookup.get(
            (
                scenario_id,
                repetition,
                comparator_strategy,
            )
        )

        if (
            task is None
            or comparator is None
        ):
            continue

        if (
            not task.planning_success
            or not comparator.planning_success
        ):
            continue

        # Both planning_success flags are True. validate_raw_records()
        # guarantees both path costs are therefore finite.
        grouped[
            scenario_id
        ].append(
            float(
                task.path_cost
                - comparator.path_cost
            )
        )

    return {
        scenario_id: float(
            np.mean(
                differences
            )
        )
        for scenario_id, differences
        in grouped.items()
        if len(
            differences
        ) > 0
    }


def analyse_primary(
    records: tuple[
        StatisticalRecord,
        ...,
    ],
) -> PrimaryAnalysis:
    """Run the frozen primary analysis."""

    protocol = (
        build_frozen_phase1_protocol()
    )

    task_rates = (
        scenario_success_rates(
            records,
            strategy=(
                PRIMARY_STRATEGY
            ),
        )
    )

    comparator_rates = (
        scenario_success_rates(
            records,
            strategy=(
                PRIMARY_COMPARATOR
            ),
        )
    )

    effects = (
        paired_scenario_effects(
            task_values=(
                task_rates
            ),
            comparator_values=(
                comparator_rates
            ),
        )
    )

    differences = np.asarray(
        [
            effect.difference
            for effect
            in effects
        ],
        dtype=float,
    )

    permutation = (
        sign_flip_permutation_test(
            differences,
            resamples=(
                protocol
                .statistics
                .permutation_resamples
            ),
            seed=(
                protocol
                .seeds
                .permutation_seed
            ),
        )
    )

    bootstrap = (
        cluster_bootstrap_mean_difference(
            differences,
            resamples=(
                protocol
                .statistics
                .bootstrap_resamples
            ),
            seed=(
                protocol
                .seeds
                .bootstrap_seed
            ),
            confidence_level=0.95,
        )
    )

    task_mean = float(
        np.mean(
            tuple(
                task_rates.values()
            )
        )
    )

    comparator_mean = float(
        np.mean(
            tuple(
                comparator_rates.values()
            )
        )
    )

    alpha = float(
        protocol
        .statistics
        .alpha
    )

    superiority = bool(
        permutation
        .p_value_two_sided
        < alpha
        and bootstrap.lower
        > 0.0
    )

    return PrimaryAnalysis(
        outcome=(
            PRIMARY_OUTCOME
        ),
        task_strategy=(
            PRIMARY_STRATEGY
        ),
        comparator_strategy=(
            PRIMARY_COMPARATOR
        ),
        scenario_count=len(
            effects
        ),
        task_mean_success_rate=(
            task_mean
        ),
        comparator_mean_success_rate=(
            comparator_mean
        ),
        absolute_effect=float(
            np.mean(
                differences
            )
        ),
        permutation_p_value=float(
            permutation
            .p_value_two_sided
        ),
        confidence_interval_lower=float(
            bootstrap.lower
        ),
        confidence_interval_upper=float(
            bootstrap.upper
        ),
        alpha=alpha,
        superiority_established=(
            superiority
        ),
    )


def analyse_movement_budget(
    records: tuple[
        StatisticalRecord,
        ...,
    ],
) -> MovementBudgetDiagnostic:
    """Report held-out camera-motion matching without retuning."""

    task = (
        scenario_mean_metric(
            records,
            strategy=(
                PRIMARY_STRATEGY
            ),
            attribute=(
                "camera_movement"
            ),
        )
    )

    comparator = (
        scenario_mean_metric(
            records,
            strategy=(
                PRIMARY_COMPARATOR
            ),
            attribute=(
                "camera_movement"
            ),
        )
    )

    effects = (
        paired_scenario_effects(
            task_values=task,
            comparator_values=(
                comparator
            ),
        )
    )

    task_mean = float(
        np.mean(
            [
                effect.task_value
                for effect
                in effects
            ]
        )
    )

    comparator_mean = float(
        np.mean(
            [
                effect.comparator_value
                for effect
                in effects
            ]
        )
    )

    if task_mean <= 1e-12:
        raise ValueError(
            "Task-Aware mean camera movement must be positive."
        )

    gap = abs(
        task_mean
        - comparator_mean
    )

    relative_gap = (
        gap
        / task_mean
    )

    tolerance = 0.10

    return MovementBudgetDiagnostic(
        scenario_count=len(
            effects
        ),
        task_mean_camera_movement=(
            task_mean
        ),
        comparator_mean_camera_movement=(
            comparator_mean
        ),
        absolute_gap=float(
            gap
        ),
        relative_gap=float(
            relative_gap
        ),
        previous_validation_tolerance=(
            tolerance
        ),
        within_previous_validation_tolerance=bool(
            relative_gap
            <= tolerance
        ),
    )


def run_statistical_analysis(
    records: tuple[
        StatisticalRecord,
        ...,
    ],
) -> Phase1StatisticalAnalysis:
    """Run frozen primary inference plus movement-budget diagnostic."""

    validate_raw_records(
        records
    )

    primary = (
        analyse_primary(
            records
        )
    )

    movement = (
        analyse_movement_budget(
            records
        )
    )

    return Phase1StatisticalAnalysis(
        protocol_sha256=(
            FROZEN_PROTOCOL_SHA256
        ),
        primary=primary,
        movement_budget=movement,
    )


def save_analysis(
    analysis: Phase1StatisticalAnalysis,
    *,
    output_path: str | Path,
    overwrite: bool = False,
) -> Path:
    """Save statistical analysis evidence."""

    output_path = Path(
        output_path
    )

    if (
        output_path.exists()
        and not overwrite
    ):
        raise FileExistsError(
            "Refusing to overwrite existing statistical evidence: "
            f"{output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            asdict(
                analysis
            ),
            handle,
            indent=2,
            allow_nan=False,
        )

    return output_path


def print_analysis(
    analysis: Phase1StatisticalAnalysis,
) -> None:
    """Print frozen primary analysis."""

    primary = (
        analysis.primary
    )

    movement = (
        analysis
        .movement_budget
    )

    print()

    print(
        "Phase 1 Frozen Statistical Analysis"
    )

    print(
        "==================================="
    )

    print()

    print(
        "PRIMARY COMPARISON"
    )

    print(
        "  Full Task-Aware"
    )

    print(
        "        versus"
    )

    print(
        "  Movement-Budget-Matched Generic"
    )

    print()

    print(
        f"Scenario count:                    "
        f"{primary.scenario_count}"
    )

    print(
        f"Task-Aware safe-navigation rate:   "
        f"{primary.task_mean_success_rate * 100.0:.2f}%"
    )

    print(
        f"Matched Generic rate:              "
        f"{primary.comparator_mean_success_rate * 100.0:.2f}%"
    )

    print(
        f"Absolute effect:                   "
        f"{primary.absolute_effect * 100.0:+.2f} percentage points"
    )

    print(
        f"95% bootstrap CI:                  "
        f"["
        f"{primary.confidence_interval_lower * 100.0:+.2f}, "
        f"{primary.confidence_interval_upper * 100.0:+.2f}"
        f"] pp"
    )

    print(
        f"Two-sided permutation p-value:     "
        f"{primary.permutation_p_value:.6f}"
    )

    print(
        f"Superiority established:           "
        f"{primary.superiority_established}"
    )

    print()

    print(
        "HELD-OUT MOVEMENT-BUDGET DIAGNOSTIC"
    )

    print(
        f"Task-Aware movement:               "
        f"{movement.task_mean_camera_movement * 1000.0:.3f} mm"
    )

    print(
        f"Matched Generic movement:          "
        f"{movement.comparator_mean_camera_movement * 1000.0:.3f} mm"
    )

    print(
        f"Relative mismatch:                 "
        f"{movement.relative_gap * 100.0:.2f}%"
    )

    print(
        f"Within previous 10% tolerance:     "
        f"{movement.within_previous_validation_tolerance}"
    )


def main() -> None:
    """Analyse the final frozen held-out evidence."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the pre-specified Phase 1 statistical analysis."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default=(
            "results/"
            "phase1_held_out/"
            "held_out_raw_records.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "results/"
            "phase1_held_out/"
            "held_out_primary_analysis.json"
        ),
    )

    args = parser.parse_args()

    records = (
        load_statistical_records(
            args.input
        )
    )

    analysis = (
        run_statistical_analysis(
            records
        )
    )

    print_analysis(
        analysis
    )

    output = (
        save_analysis(
            analysis,
            output_path=(
                args.output
            ),
            overwrite=False,
        )
    )

    print()

    print(
        f"Statistical evidence saved to: {output}"
    )


if __name__ == "__main__":
    main()