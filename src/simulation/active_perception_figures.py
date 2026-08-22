"""Publication-quality figures for active-perception experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.simulation.task_aware_benchmark import (
    TaskAwareBenchmarkConfig,
    run_benchmark,
)
from src.simulation.uncertainty_sensitivity import (
    run_sensitivity,
)


OUTPUT_DIRECTORY = (
    Path("results")
    / "active_perception_figures"
)


def prepare_output_directory() -> None:
    """Create the output directory."""

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


def save_figure(
    figure: plt.Figure,
    filename: str,
) -> None:
    """Save a publication-quality figure."""

    path = (
        OUTPUT_DIRECTORY
        / filename
    )

    figure.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Saved: {path}"
    )

    plt.close(
        figure
    )


def plot_localisation_error(
    comparison,
) -> None:
    """Plot the matched localisation-error distributions."""

    generic = (
        comparison
        .generic_localisation_errors
        * 1000.0
    )

    task_aware = (
        comparison
        .task_aware_localisation_errors
        * 1000.0
    )

    figure, axis = plt.subplots(
        figsize=(8, 5.5)
    )

    axis.boxplot(
        [
            generic,
            task_aware,
        ],
        tick_labels=[
            "Generic",
            "Task-aware",
        ],
        widths=0.55,
        showmeans=True,
    )

    axis.set_ylabel(
        "Localisation error (mm)"
    )

    axis.set_title(
        "Localisation Error Across 100 Matched Trials"
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    figure.tight_layout()

    save_figure(
        figure,
        "localisation_error_comparison.png",
    )


def plot_predicted_uncertainty(
    comparison,
) -> None:
    """Plot predicted localisation uncertainty."""

    generic = (
        comparison
        .generic_predicted_sigmas
        * 1000.0
    )

    task_aware = (
        comparison
        .task_aware_predicted_sigmas
        * 1000.0
    )

    figure, axis = plt.subplots(
        figsize=(8, 5.5)
    )

    means = [
        float(
            np.mean(
                generic
            )
        ),
        float(
            np.mean(
                task_aware
            )
        ),
    ]

    axis.bar(
        [
            "Generic",
            "Task-aware",
        ],
        means,
    )

    axis.set_ylabel(
        "Predicted localisation uncertainty (mm)"
    )

    axis.set_title(
        "Predicted Localisation Uncertainty"
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    for index, value in enumerate(
        means
    ):
        axis.text(
            index,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    figure.tight_layout()

    save_figure(
        figure,
        "predicted_uncertainty_comparison.png",
    )


def plot_uncertainty_sensitivity() -> None:
    """Plot the controlled uncertainty sensitivity experiment."""

    study = run_sensitivity()

    sigma_values = np.asarray(
        [
            result.sigma_m
            for result in study.results
        ],
        dtype=float,
    ) * 1000.0

    generic_values = np.asarray(
        [
            result
            .generic_selected_uncertainty_m
            for result in study.results
        ],
        dtype=float,
    ) * 1000.0

    task_aware_values = np.asarray(
        [
            result
            .task_aware_selected_uncertainty_m
            for result in study.results
        ],
        dtype=float,
    ) * 1000.0

    figure, axis = plt.subplots(
        figsize=(8, 5.5)
    )

    axis.plot(
        sigma_values,
        generic_values,
        marker="o",
        linewidth=2,
        label="Generic",
    )

    axis.plot(
        sigma_values,
        task_aware_values,
        marker="o",
        linewidth=2,
        label="Task-aware",
    )

    axis.set_xlabel(
        "Baseline perception uncertainty (mm)"
    )

    axis.set_ylabel(
        "Selected viewpoint uncertainty (mm)"
    )

    axis.set_title(
        "Sensitivity to Perception Uncertainty"
    )

    axis.legend()

    axis.grid(
        alpha=0.25,
    )

    figure.tight_layout()

    save_figure(
        figure,
        "uncertainty_sensitivity.png",
    )


def plot_summary_metrics(
    comparison,
) -> None:
    """Plot the principal benchmark outcomes."""

    localisation_reduction = (
        comparison
        .localisation_error_reduction_percent
    )

    sigma_reduction = (
        comparison
        .predicted_sigma_reduction_percent
    )

    movement_increase = (
        100.0
        * (
            comparison
            .task_aware
            .mean_camera_movement
            - comparison
            .generic
            .mean_camera_movement
        )
        / comparison
        .generic
        .mean_camera_movement
    )

    figure, axis = plt.subplots(
        figsize=(8, 5.5)
    )

    labels = [
        "Localisation\nerror reduction",
        "Predicted\nuncertainty reduction",
        "Camera movement\nincrease",
    ]

    values = [
        localisation_reduction,
        sigma_reduction,
        movement_increase,
    ]

    axis.bar(
        labels,
        values,
    )

    axis.axhline(
        0.0,
        linewidth=1.0,
    )

    axis.set_ylabel(
        "Relative change from generic baseline (%)"
    )

    axis.set_title(
        "Task-Aware Active Perception: "
        "Performance Trade-off"
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    for index, value in enumerate(
        values
    ):
        axis.text(
            index,
            value,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
        )

    figure.tight_layout()

    save_figure(
        figure,
        "performance_tradeoff.png",
    )


def main() -> None:
    """Generate all publication figures."""

    print()
    print(
        "Active Perception Figure Generation"
    )
    print(
        "==================================="
    )
    print()

    prepare_output_directory()

    print(
        "Running 100-trial benchmark..."
    )

    comparison = run_benchmark(
        TaskAwareBenchmarkConfig()
    )

    print(
        "Generating figures..."
    )

    plot_localisation_error(
        comparison
    )

    plot_predicted_uncertainty(
        comparison
    )

    plot_uncertainty_sensitivity()

    plot_summary_metrics(
        comparison
    )

    print()
    print(
        "Figure generation complete."
    )


if __name__ == "__main__":
    main()

