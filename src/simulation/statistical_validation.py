"""Statistical analysis utilities for matched trial outcomes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrialOutcome:
    """Outcome from one trial."""

    trial: int
    planning_success: bool
    safe: bool
    collision: bool
    safety_margin_violation: bool
    minimum_surface_clearance: float
    minimum_safety_clearance: float
    path_cost: float


@dataclass(frozen=True)
class StatisticalSummary:
    """Descriptive statistics for one strategy."""

    strategy: str
    trials: int
    planning_success_rate_percent: float
    safe_rate_percent: float
    collision_rate_percent: float
    safety_margin_violation_rate_percent: float
    mean_minimum_safety_clearance: float
    median_minimum_safety_clearance: float
    standard_deviation_minimum_safety_clearance: float
    mean_path_cost: float


@dataclass(frozen=True)
class PairedClearanceStatistics:
    """Paired statistics for minimum safety clearance."""

    mean_difference: float
    median_difference: float
    standard_deviation_difference: float
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    cohens_dz: float


@dataclass(frozen=True)
class PairedSafetyStatistics:
    """Paired statistics for binary safety."""

    generic_safe_rate_percent: float
    task_aware_safe_rate_percent: float
    discordant_generic_safe_task_unsafe: int
    discordant_generic_unsafe_task_safe: int
    exact_two_sided_p_value: float


@dataclass(frozen=True)
class StatisticalValidationResult:
    """Complete statistical analysis."""

    generic_summary: StatisticalSummary
    task_aware_summary: StatisticalSummary
    paired_clearance: PairedClearanceStatistics
    paired_safety: PairedSafetyStatistics


def exact_two_sided_binomial_p_value(
    successes: int,
    trials: int,
) -> float:
    """Exact two-sided binomial test with null probability 0.5."""

    if trials < 0:
        raise ValueError(
            "trials must be non-negative."
        )

    if successes < 0 or successes > trials:
        raise ValueError(
            "successes must be between 0 and trials."
        )

    if trials == 0:
        return 1.0

    probabilities = np.asarray(
        [
            math.comb(trials, k)
            / (2.0 ** trials)
            for k in range(trials + 1)
        ],
        dtype=float,
    )

    observed = probabilities[successes]

    p_value = float(
        np.sum(
            probabilities[
                probabilities
                <= observed + 1e-15
            ]
        )
    )

    return min(
        1.0,
        p_value,
    )


def bootstrap_mean_difference_ci(
    differences: np.ndarray,
    seed: int = 20260819,
    bootstrap_samples: int = 10000,
) -> tuple[float, float]:
    """Return percentile bootstrap 95% confidence interval."""

    values = np.asarray(
        differences,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return (
            float("nan"),
            float("nan"),
        )

    if bootstrap_samples <= 0:
        raise ValueError(
            "bootstrap_samples must be positive."
        )

    rng = np.random.default_rng(
        seed
    )

    samples = rng.choice(
        values,
        size=(
            bootstrap_samples,
            len(values),
        ),
        replace=True,
    )

    means = np.mean(
        samples,
        axis=1,
    )

    return (
        float(
            np.percentile(
                means,
                2.5,
            )
        ),
        float(
            np.percentile(
                means,
                97.5,
            )
        ),
    )


def descriptive_summary(
    strategy: str,
    outcomes: tuple[TrialOutcome, ...],
) -> StatisticalSummary:
    """Calculate descriptive statistics."""

    if len(outcomes) == 0:
        raise ValueError(
            "outcomes must not be empty."
        )

    trials = len(outcomes)

    successful = tuple(
        outcome
        for outcome in outcomes
        if outcome.planning_success
    )

    planning_success_rate = (
        100.0
        * len(successful)
        / trials
    )

    if len(successful) == 0:
        return StatisticalSummary(
            strategy=strategy,
            trials=trials,
            planning_success_rate_percent=(
                planning_success_rate
            ),
            safe_rate_percent=0.0,
            collision_rate_percent=0.0,
            safety_margin_violation_rate_percent=0.0,
            mean_minimum_safety_clearance=float(
                "nan"
            ),
            median_minimum_safety_clearance=float(
                "nan"
            ),
            standard_deviation_minimum_safety_clearance=float(
                "nan"
            ),
            mean_path_cost=float("nan"),
        )

    safe_rate = (
        100.0
        * sum(
            outcome.safe
            for outcome in successful
        )
        / len(successful)
    )

    collision_rate = (
        100.0
        * sum(
            outcome.collision
            for outcome in successful
        )
        / len(successful)
    )

    violation_rate = (
        100.0
        * sum(
            (
                outcome.safety_margin_violation
                or outcome.collision
            )
            for outcome in successful
        )
        / len(successful)
    )

    clearance = np.asarray(
        [
            outcome.minimum_safety_clearance
            for outcome in successful
            if np.isfinite(
                outcome.minimum_safety_clearance
            )
        ],
        dtype=float,
    )

    costs = np.asarray(
        [
            outcome.path_cost
            for outcome in successful
            if np.isfinite(
                outcome.path_cost
            )
        ],
        dtype=float,
    )

    if len(clearance) == 0:
        mean_clearance = float("nan")
        median_clearance = float("nan")
        standard_deviation = float("nan")
    else:
        mean_clearance = float(
            np.mean(clearance)
        )

        median_clearance = float(
            np.median(clearance)
        )

        standard_deviation = float(
            np.std(
                clearance,
                ddof=1,
            )
            if len(clearance) > 1
            else 0.0
        )

    mean_path_cost = (
        float(
            np.mean(costs)
        )
        if len(costs) > 0
        else float("nan")
    )

    return StatisticalSummary(
        strategy=strategy,
        trials=trials,
        planning_success_rate_percent=(
            planning_success_rate
        ),
        safe_rate_percent=safe_rate,
        collision_rate_percent=collision_rate,
        safety_margin_violation_rate_percent=(
            violation_rate
        ),
        mean_minimum_safety_clearance=(
            mean_clearance
        ),
        median_minimum_safety_clearance=(
            median_clearance
        ),
        standard_deviation_minimum_safety_clearance=(
            standard_deviation
        ),
        mean_path_cost=mean_path_cost,
    )


def paired_clearance_statistics(
    generic: tuple[TrialOutcome, ...],
    task_aware: tuple[TrialOutcome, ...],
) -> PairedClearanceStatistics:
    """Calculate paired clearance statistics."""

    if len(generic) != len(task_aware):
        raise ValueError(
            "Matched trial sets must have equal length."
        )

    differences: list[float] = []

    for generic_outcome, task_outcome in zip(
        generic,
        task_aware,
    ):
        if not (
            generic_outcome.planning_success
            and task_outcome.planning_success
        ):
            continue

        generic_value = (
            generic_outcome.minimum_safety_clearance
        )

        task_value = (
            task_outcome.minimum_safety_clearance
        )

        if not (
            np.isfinite(generic_value)
            and np.isfinite(task_value)
        ):
            continue

        differences.append(
            task_value - generic_value
        )

    values = np.asarray(
        differences,
        dtype=float,
    )

    if len(values) == 0:
        return PairedClearanceStatistics(
            mean_difference=float("nan"),
            median_difference=float("nan"),
            standard_deviation_difference=float("nan"),
            bootstrap_ci_lower=float("nan"),
            bootstrap_ci_upper=float("nan"),
            cohens_dz=float("nan"),
        )

    mean_difference = float(
        np.mean(values)
    )

    median_difference = float(
        np.median(values)
    )

    standard_deviation = float(
        np.std(
            values,
            ddof=1,
        )
        if len(values) > 1
        else 0.0
    )

    ci_lower, ci_upper = (
        bootstrap_mean_difference_ci(
            values
        )
    )

    if standard_deviation > 1e-12:
        cohens_dz = (
            mean_difference
            / standard_deviation
        )
    else:
        cohens_dz = float("inf")

    return PairedClearanceStatistics(
        mean_difference=mean_difference,
        median_difference=median_difference,
        standard_deviation_difference=(
            standard_deviation
        ),
        bootstrap_ci_lower=ci_lower,
        bootstrap_ci_upper=ci_upper,
        cohens_dz=float(cohens_dz),
    )


def paired_safety_statistics(
    generic: tuple[TrialOutcome, ...],
    task_aware: tuple[TrialOutcome, ...],
) -> PairedSafetyStatistics:
    """Calculate paired safety statistics."""

    if len(generic) != len(task_aware):
        raise ValueError(
            "Matched trial sets must have equal length."
        )

    generic_planned = 0
    task_aware_planned = 0

    generic_safe = 0
    task_aware_safe = 0

    generic_safe_task_unsafe = 0
    generic_unsafe_task_safe = 0

    for generic_outcome, task_outcome in zip(
        generic,
        task_aware,
    ):
        if generic_outcome.planning_success:
            generic_planned += 1

            if generic_outcome.safe:
                generic_safe += 1

        if task_outcome.planning_success:
            task_aware_planned += 1

            if task_outcome.safe:
                task_aware_safe += 1

        if not (
            generic_outcome.planning_success
            and task_outcome.planning_success
        ):
            continue

        if (
            generic_outcome.safe
            and not task_outcome.safe
        ):
            generic_safe_task_unsafe += 1

        elif (
            not generic_outcome.safe
            and task_outcome.safe
        ):
            generic_unsafe_task_safe += 1

    discordant = (
        generic_safe_task_unsafe
        + generic_unsafe_task_safe
    )

    if discordant == 0:
        p_value = 1.0
    else:
        p_value = (
            exact_two_sided_binomial_p_value(
                successes=(
                    generic_unsafe_task_safe
                ),
                trials=discordant,
            )
        )

    return PairedSafetyStatistics(
        generic_safe_rate_percent=(
            100.0
            * generic_safe
            / generic_planned
            if generic_planned > 0
            else 0.0
        ),
        task_aware_safe_rate_percent=(
            100.0
            * task_aware_safe
            / task_aware_planned
            if task_aware_planned > 0
            else 0.0
        ),
        discordant_generic_safe_task_unsafe=(
            generic_safe_task_unsafe
        ),
        discordant_generic_unsafe_task_safe=(
            generic_unsafe_task_safe
        ),
        exact_two_sided_p_value=p_value,
    )


def validate_matched_trials(
    generic: tuple[TrialOutcome, ...],
    task_aware: tuple[TrialOutcome, ...],
) -> None:
    """Ensure the two strategies contain the same trial IDs."""

    if len(generic) != len(task_aware):
        raise ValueError(
            "Matched strategies must have equal length."
        )

    for generic_outcome, task_outcome in zip(
        generic,
        task_aware,
    ):
        if generic_outcome.trial != task_outcome.trial:
            raise ValueError(
                "Trial identifiers must match."
            )


def build_validation_result(
    generic: tuple[TrialOutcome, ...],
    task_aware: tuple[TrialOutcome, ...],
) -> StatisticalValidationResult:
    """Build statistical results from matched trial outcomes."""

    validate_matched_trials(
        generic=generic,
        task_aware=task_aware,
    )

    return StatisticalValidationResult(
        generic_summary=descriptive_summary(
            strategy="Generic active perception",
            outcomes=generic,
        ),
        task_aware_summary=descriptive_summary(
            strategy="Task-aware active perception",
            outcomes=task_aware,
        ),
        paired_clearance=paired_clearance_statistics(
            generic=generic,
            task_aware=task_aware,
        ),
        paired_safety=paired_safety_statistics(
            generic=generic,
            task_aware=task_aware,
        ),
    )


if __name__ == "__main__":
    print(
        "Statistical validation utilities loaded successfully."
    )