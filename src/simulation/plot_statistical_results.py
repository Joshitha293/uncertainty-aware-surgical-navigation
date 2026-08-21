"""Create publication-quality figures from Day 7 trial-level results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_PATH = (
    Path("results")
    / "day7_statistical_trials.csv"
)

OUTPUT_DIR = (
    Path("results")
    / "figures"
)


def load_results() -> pd.DataFrame:
    """Load trial-level statistical results."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_PATH}. "
            "Run statistical_benchmark first."
        )

    data = pd.read_csv(
        INPUT_PATH
    )

    if len(data) == 0:
        raise ValueError(
            "Trial-results file is empty."
        )

    required_columns = [
        "trial",
        "generic_minimum_safety_clearance_m",
        "task_aware_minimum_safety_clearance_m",
        "clearance_difference_m",
        "generic_safe",
        "task_aware_safe",
    ]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    return data


def make_clearance_distribution(
    data: pd.DataFrame,
) -> None:
    """Create the clearance distribution figure."""

    generic = (
        data[
            "generic_minimum_safety_clearance_m"
        ]
        * 1000.0
    )

    task_aware = (
        data[
            "task_aware_minimum_safety_clearance_m"
        ]
        * 1000.0
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.boxplot(
        [
            generic,
            task_aware,
        ],
        tick_labels=[
            "Generic",
            "Task-aware",
        ],
        showmeans=True,
    )

    plt.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    plt.ylabel(
        "Minimum safety clearance (mm)"
    )

    plt.title(
        "Ground-truth minimum safety clearance"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "figure_clearance_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def make_paired_improvement(
    data: pd.DataFrame,
) -> None:
    """Create the paired trial improvement figure."""

    difference = (
        data[
            "clearance_difference_m"
        ]
        * 1000.0
    )

    trials = (
        data["trial"]
        + 1
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        trials,
        difference,
        marker="o",
        markersize=3,
        linewidth=1.0,
    )

    plt.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    plt.xlabel(
        "Matched trial"
    )

    plt.ylabel(
        "Task-aware − generic "
        "clearance (mm)"
    )

    plt.title(
        "Paired improvement in "
        "minimum safety clearance"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "figure_paired_clearance_improvement.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def make_safe_rate_figure(
    data: pd.DataFrame,
) -> None:
    """Create the safe trajectory rate figure."""

    generic_rate = (
        100.0
        * data[
            "generic_safe"
        ].astype(bool).mean()
    )

    task_aware_rate = (
        100.0
        * data[
            "task_aware_safe"
        ].astype(bool).mean()
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.bar(
        [
            "Generic",
            "Task-aware",
        ],
        [
            generic_rate,
            task_aware_rate,
        ],
    )

    plt.ylim(
        0.0,
        100.0,
    )

    plt.ylabel(
        "Safe trajectories (%)"
    )

    plt.title(
        "Ground-truth safety outcome"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "figure_safe_trajectory_rate.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:
    """Generate all statistical figures."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_results()

    make_clearance_distribution(
        data
    )

    make_paired_improvement(
        data
    )

    make_safe_rate_figure(
        data
    )

    print()
    print(
        "Generated figures:"
    )

    print(
        OUTPUT_DIR
        / "figure_clearance_distribution.png"
    )

    print(
        OUTPUT_DIR
        / "figure_paired_clearance_improvement.png"
    )

    print(
        OUTPUT_DIR
        / "figure_safe_trajectory_rate.png"
    )

    print()
    print(
        f"Trials visualised: {len(data)}"
    )


if __name__ == "__main__":
    main()