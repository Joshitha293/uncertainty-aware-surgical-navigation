"""Statistical validation using the matched Monte Carlo benchmark."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.geometry.workspace import SphericalStructure
from src.perception.perception import perceive_structures
from src.perception.planning import (
    deterministic_planning_structures,
    uncertainty_aware_planning_structures,
)
from src.perception.uncertainty import PositionUncertainty
from src.robotics.instrument import SurgicalInstrument
from src.simulation.statistical_validation import (
    TrialOutcome,
    build_validation_result,
)
from src.simulation.uncertainty_benchmark import (
    TrialResult,
    run_planner_trial,
)


RESULTS_PATH = Path(
    "results"
) / "day7_statistical_trials.csv"


def make_start_configuration() -> np.ndarray:
    """Return the validated Day 6 start configuration."""

    return np.array(
        [
            np.deg2rad(-25.0),
            np.deg2rad(-15.0),
            0.16,
            0.0,
        ],
        dtype=float,
    )


def make_goal_configuration() -> np.ndarray:
    """Return the validated Day 6 goal configuration."""

    return np.array(
        [
            np.deg2rad(35.0),
            np.deg2rad(25.0),
            0.25,
            0.0,
        ],
        dtype=float,
    )


def make_true_structures() -> tuple[
    SphericalStructure,
    ...,
]:
    """Return the validated Day 6 ground-truth anatomy."""

    return (
        SphericalStructure(
            centre=np.array(
                [0.14, 0.04, 0.00],
                dtype=float,
            ),
            physical_radius=0.025,
            safety_margin=0.015,
        ),
        SphericalStructure(
            centre=np.array(
                [0.18, -0.06, 0.02],
                dtype=float,
            ),
            physical_radius=0.025,
            safety_margin=0.015,
        ),
    )


def make_instrument() -> SurgicalInstrument:
    """Return the validated Day 6 surgical instrument."""

    return SurgicalInstrument(
        rcm_position=np.zeros(
            3,
            dtype=float,
        )
    )


def convert_trial_result(
    result: TrialResult,
) -> TrialOutcome:
    """Convert a benchmark result to statistical format."""

    if result.planning_success:
        safe = not (
            result.collision_against_truth
            or result.safety_violation_against_truth
        )
    else:
        safe = False

    return TrialOutcome(
        trial=result.trial,
        planning_success=result.planning_success,
        safe=safe,
        collision=result.collision_against_truth,
        safety_margin_violation=(
            result.safety_violation_against_truth
        ),
        minimum_surface_clearance=float(
            "nan"
        ),
        minimum_safety_clearance=(
            result.minimum_true_safety_clearance
        ),
        path_cost=result.path_cost,
    )


def save_trial_results(
    generic: tuple[TrialOutcome, ...],
    task_aware: tuple[TrialOutcome, ...],
    path: Path = RESULTS_PATH,
) -> None:
    """Save matched trial-level results to CSV."""

    if len(generic) != len(task_aware):
        raise ValueError(
            "generic and task_aware must have equal length."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "trial",
                "generic_planning_success",
                "generic_safe",
                "generic_collision",
                "generic_safety_margin_violation",
                "generic_minimum_safety_clearance_m",
                "generic_path_cost",
                "task_aware_planning_success",
                "task_aware_safe",
                "task_aware_collision",
                "task_aware_safety_margin_violation",
                "task_aware_minimum_safety_clearance_m",
                "task_aware_path_cost",
                "clearance_difference_m",
            ]
        )

        for generic_result, task_result in zip(
            generic,
            task_aware,
        ):
            difference = (
                task_result.minimum_safety_clearance
                - generic_result.minimum_safety_clearance
            )

            writer.writerow(
                [
                    generic_result.trial,
                    generic_result.planning_success,
                    generic_result.safe,
                    generic_result.collision,
                    generic_result.safety_margin_violation,
                    generic_result.minimum_safety_clearance,
                    generic_result.path_cost,
                    task_result.planning_success,
                    task_result.safe,
                    task_result.collision,
                    task_result.safety_margin_violation,
                    task_result.minimum_safety_clearance,
                    task_result.path_cost,
                    difference,
                ]
            )

    print()
    print(
        f"Saved trial-level results to: {path}"
    )


def run_matched_benchmark(
    trial_count: int = 100,
) -> tuple[
    tuple[TrialOutcome, ...],
    tuple[TrialOutcome, ...],
]:
    """Run the matched Day 6 benchmark."""

    if trial_count <= 0:
        raise ValueError(
            "trial_count must be positive."
        )

    start_q = make_start_configuration()
    goal_q = make_goal_configuration()

    true_structures = make_true_structures()
    instrument = make_instrument()

    instrument_radius = 0.006
    proximal_length = 0.10

    localisation_sigma = 0.005
    sigma_multiplier = 2.0

    uncertainty = PositionUncertainty.isotropic(
        sigma=localisation_sigma
    )

    generic_results: list[
        TrialOutcome
    ] = []

    task_aware_results: list[
        TrialOutcome
    ] = []

    print()
    print(
        "Statistical Day 7 Benchmark"
    )
    print(
        "==========================="
    )

    print(
        f"Matched trials: {trial_count}"
    )

    for trial in range(
        trial_count
    ):
        perception_rng = (
            np.random.default_rng(
                1000 + trial
            )
        )

        perception = perceive_structures(
            true_structures=true_structures,
            uncertainty=uncertainty,
            rng=perception_rng,
        )

        deterministic = (
            deterministic_planning_structures(
                perception
            )
        )

        uncertainty_aware = (
            uncertainty_aware_planning_structures(
                perception_result=perception,
                sigma_multiplier=sigma_multiplier,
            )
        )

        planner_seed = 2000 + trial

        deterministic_result = (
            run_planner_trial(
                trial=trial,
                method="Deterministic RRT",
                instrument=instrument,
                start_q=start_q,
                goal_q=goal_q,
                planning_structures=(
                    deterministic.structures
                ),
                true_structures=true_structures,
                instrument_radius=instrument_radius,
                proximal_length=proximal_length,
                planner_seed=planner_seed,
            )
        )

        task_aware_result = (
            run_planner_trial(
                trial=trial,
                method="Uncertainty-aware RRT",
                instrument=instrument,
                start_q=start_q,
                goal_q=goal_q,
                planning_structures=(
                    uncertainty_aware.structures
                ),
                true_structures=true_structures,
                instrument_radius=instrument_radius,
                proximal_length=proximal_length,
                planner_seed=planner_seed,
            )
        )

        generic_results.append(
            convert_trial_result(
                deterministic_result
            )
        )

        task_aware_results.append(
            convert_trial_result(
                task_aware_result
            )
        )

        print(
            f"Trial {trial + 1:03d}/"
            f"{trial_count:03d} complete"
        )

    generic_tuple = tuple(
        generic_results
    )

    task_aware_tuple = tuple(
        task_aware_results
    )

    save_trial_results(
        generic=generic_tuple,
        task_aware=task_aware_tuple,
    )

    return (
        generic_tuple,
        task_aware_tuple,
    )


def print_statistical_summary(
    result,
) -> None:
    """Print the statistical summary."""

    generic = result.generic_summary
    task_aware = result.task_aware_summary
    clearance = result.paired_clearance
    safety = result.paired_safety

    print()
    print(
        "Statistical Validation"
    )
    print(
        "======================"
    )

    print()
    print(
        "Generic active perception"
    )
    print(
        "-------------------------"
    )

    print(
        f"Planning success: "
        f"{generic.planning_success_rate_percent:.2f}%"
    )

    print(
        f"Ground-truth safe: "
        f"{generic.safe_rate_percent:.2f}%"
    )

    print(
        "Mean minimum safety clearance: "
        f"{generic.mean_minimum_safety_clearance * 1000:.3f} mm"
    )

    print(
        "Median minimum safety clearance: "
        f"{generic.median_minimum_safety_clearance * 1000:.3f} mm"
    )

    print()
    print(
        "Task-aware active perception"
    )
    print(
        "----------------------------"
    )

    print(
        f"Planning success: "
        f"{task_aware.planning_success_rate_percent:.2f}%"
    )

    print(
        f"Ground-truth safe: "
        f"{task_aware.safe_rate_percent:.2f}%"
    )

    print(
        "Mean minimum safety clearance: "
        f"{task_aware.mean_minimum_safety_clearance * 1000:.3f} mm"
    )

    print(
        "Median minimum safety clearance: "
        f"{task_aware.median_minimum_safety_clearance * 1000:.3f} mm"
    )

    print()
    print(
        "Paired minimum-safety-clearance comparison"
    )
    print(
        "------------------------------------------"
    )

    print(
        "Mean difference: "
        f"{clearance.mean_difference * 1000:.3f} mm"
    )

    print(
        "Bootstrap 95% CI: "
        f"[{clearance.bootstrap_ci_lower * 1000:.3f}, "
        f"{clearance.bootstrap_ci_upper * 1000:.3f}] mm"
    )

    print(
        f"Cohen's dz: "
        f"{clearance.cohens_dz:.3f}"
    )

    print()
    print(
        "Paired safety comparison"
    )
    print(
        "------------------------"
    )

    print(
        f"Generic safe rate: "
        f"{safety.generic_safe_rate_percent:.2f}%"
    )

    print(
        f"Task-aware safe rate: "
        f"{safety.task_aware_safe_rate_percent:.2f}%"
    )

    print(
        "Generic unsafe -> task-aware safe: "
        f"{safety.discordant_generic_unsafe_task_safe}"
    )

    print(
        f"Exact paired p-value: "
        f"{safety.exact_two_sided_p_value:.6g}"
    )


def main() -> None:
    """Run and save the 100-trial experiment."""

    generic, task_aware = (
        run_matched_benchmark(
            trial_count=100
        )
    )

    result = build_validation_result(
        generic=generic,
        task_aware=task_aware,
    )

    print_statistical_summary(
        result
    )


if __name__ == "__main__":
    main()