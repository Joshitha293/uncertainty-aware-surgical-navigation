"""Statistical analysis for generic vs task-aware active perception."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StatisticalSummary:
    """Summary statistics for one experimental metric."""

    mean: float
    median: float
    standard_deviation: float
    standard_error: float
    confidence_interval_low: float
    confidence_interval_high: float


@dataclass(frozen=True)
class PairedComparison:
    """Statistical comparison between two matched conditions."""

    mean_difference: float
    standard_deviation_difference: float
    confidence_interval_low: float
    confidence_interval_high: float
    relative_change_percent: float
    cohens_dz: float


def _validate_array(
    values: np.ndarray,
    name: str,
) -> np.ndarray:
    """Validate and return a one-dimensional finite array."""

    values = np.asarray(
        values,
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional."
        )

    if len(values) < 2:
        raise ValueError(
            f"{name} must contain at least two observations."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"{name} must contain only finite values."
        )

    return values


def bootstrap_mean_confidence_interval(
    values: np.ndarray,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    """Estimate a percentile-bootstrap confidence interval for the mean."""

    values = _validate_array(
        values,
        "values",
    )

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level must lie between 0 and 1."
        )

    if bootstrap_samples < 100:
        raise ValueError(
            "bootstrap_samples must be at least 100."
        )

    rng = np.random.default_rng(seed)

    sample_indices = rng.integers(
        0,
        len(values),
        size=(
            bootstrap_samples,
            len(values),
        ),
    )

    bootstrap_means = np.mean(
        values[sample_indices],
        axis=1,
    )

    alpha = 1.0 - confidence_level

    lower = float(
        np.percentile(
            bootstrap_means,
            100.0 * alpha / 2.0,
        )
    )

    upper = float(
        np.percentile(
            bootstrap_means,
            100.0 * (1.0 - alpha / 2.0),
        )
    )

    return lower, upper


def summarise_metric(
    values: np.ndarray,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 10000,
    seed: int = 42,
) -> StatisticalSummary:
    """Calculate descriptive and bootstrap confidence statistics."""

    values = _validate_array(
        values,
        "values",
    )

    mean = float(
        np.mean(values)
    )

    median = float(
        np.median(values)
    )

    standard_deviation = float(
        np.std(
            values,
            ddof=1,
        )
    )

    standard_error = float(
        standard_deviation
        / np.sqrt(len(values))
    )

    confidence_interval_low, confidence_interval_high = (
        bootstrap_mean_confidence_interval(
            values=values,
            confidence_level=confidence_level,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
    )

    return StatisticalSummary(
        mean=mean,
        median=median,
        standard_deviation=standard_deviation,
        standard_error=standard_error,
        confidence_interval_low=confidence_interval_low,
        confidence_interval_high=confidence_interval_high,
    )


def paired_comparison(
    generic_values: np.ndarray,
    task_aware_values: np.ndarray,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 10000,
    seed: int = 42,
) -> PairedComparison:
    """Compare two matched experimental conditions."""

    generic_values = _validate_array(
        generic_values,
        "generic_values",
    )

    task_aware_values = _validate_array(
        task_aware_values,
        "task_aware_values",
    )

    if len(generic_values) != len(task_aware_values):
        raise ValueError(
            "Matched conditions must contain the same number "
            "of observations."
        )

    differences = (
        task_aware_values
        - generic_values
    )

    mean_difference = float(
        np.mean(differences)
    )

    standard_deviation_difference = float(
        np.std(
            differences,
            ddof=1,
        )
    )

    confidence_interval_low, confidence_interval_high = (
        bootstrap_mean_confidence_interval(
            values=differences,
            confidence_level=confidence_level,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
    )

    generic_mean = float(
        np.mean(generic_values)
    )

    if np.isclose(
        generic_mean,
        0.0,
    ):
        relative_change_percent = float("nan")
    else:
        relative_change_percent = float(
            100.0
            * (
                mean_difference
                / generic_mean
            )
        )

    if np.isclose(
        standard_deviation_difference,
        0.0,
    ):
        cohens_dz = 0.0
    else:
        cohens_dz = float(
            mean_difference
            / standard_deviation_difference
        )

    return PairedComparison(
        mean_difference=mean_difference,
        standard_deviation_difference=(
            standard_deviation_difference
        ),
        confidence_interval_low=(
            confidence_interval_low
        ),
        confidence_interval_high=(
            confidence_interval_high
        ),
        relative_change_percent=(
            relative_change_percent
        ),
        cohens_dz=cohens_dz,
    )


def print_metric_summary(
    name: str,
    summary: StatisticalSummary,
    unit: str = "",
) -> None:
    """Print one research-friendly statistical summary."""

    print(
        f"{name}"
    )

    print(
        f"  Mean: "
        f"{summary.mean:.6f} {unit}"
    )

    print(
        f"  Median: "
        f"{summary.median:.6f} {unit}"
    )

    print(
        f"  SD: "
        f"{summary.standard_deviation:.6f} {unit}"
    )

    print(
        f"  95% CI: "
        f"["
        f"{summary.confidence_interval_low:.6f}, "
        f"{summary.confidence_interval_high:.6f}"
        f"] {unit}"
    )


def print_paired_comparison(
    name: str,
    comparison: PairedComparison,
    unit: str = "",
) -> None:
    """Print a research-friendly paired comparison."""

    print(
        name
    )

    print(
        f"  Mean paired difference: "
        f"{comparison.mean_difference:.6f} {unit}"
    )

    print(
        f"  SD of paired differences: "
        f"{comparison.standard_deviation_difference:.6f} {unit}"
    )

    print(
        f"  95% CI of difference: "
        f"["
        f"{comparison.confidence_interval_low:.6f}, "
        f"{comparison.confidence_interval_high:.6f}"
        f"] {unit}"
    )

    print(
        f"  Relative change: "
        f"{comparison.relative_change_percent:.2f}%"
    )

    print(
        f"  Cohen's dz: "
        f"{comparison.cohens_dz:.3f}"
    )


def main() -> None:
    """Run a self-contained statistical-module sanity check."""

    generic = np.array(
        [
            0.0042,
            0.0040,
            0.0045,
            0.0041,
            0.0043,
        ],
        dtype=float,
    )

    task_aware = np.array(
        [
            0.0032,
            0.0030,
            0.0033,
            0.0031,
            0.0032,
        ],
        dtype=float,
    )

    generic_summary = summarise_metric(
        generic
    )

    task_aware_summary = summarise_metric(
        task_aware
    )

    comparison = paired_comparison(
        generic_values=generic,
        task_aware_values=task_aware,
    )

    print()
    print(
        "Active Perception Statistical Analysis"
    )
    print(
        "======================================="
    )
    print()

    print(
        "Generic localisation error"
    )

    print_metric_summary(
        "Summary:",
        generic_summary,
        unit="m",
    )

    print()

    print(
        "Task-aware localisation error"
    )

    print_metric_summary(
        "Summary:",
        task_aware_summary,
        unit="m",
    )

    print()

    print_paired_comparison(
        "Paired comparison:",
        comparison,
        unit="m",
    )


if __name__ == "__main__":
    main()