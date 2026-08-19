"""Parameter sweep for uncertainty-aware surgical motion planning."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from src.geometry.workspace import SphericalStructure
from src.perception.perception import perceive_structures
from src.perception.planning import (
    deterministic_planning_structures,
    uncertainty_aware_planning_structures,
)
from src.perception.uncertainty import PositionUncertainty
from src.robotics.instrument import SurgicalInstrument
from src.robotics.planner import (
    path_cost,
    plan_rrt,
    shortcut_path,
)
from src.robotics.safety import evaluate_instrument_safety
from src.simulation.scene import densify_path


@dataclass(frozen=True)
class SweepTrialResult:
    """Result for one trial under one uncertainty configuration."""

    localisation_sigma: float
    sigma_multiplier: float
    trial: int
    method: str
    planning_success: bool
    planning_time: float
    path_cost: float
    minimum_true_safety_clearance: float
    collision: bool
    safety_violation: bool


def evaluate_against_truth(
    instrument: SurgicalInstrument,
    trajectory: np.ndarray,
    true_structures: tuple[SphericalStructure, ...],
    instrument_radius: float,
    proximal_length: float,
) -> tuple[float, bool, bool]:
    """Evaluate a trajectory against hidden ground-truth anatomy."""

    minimum_safety_clearance = float("inf")
    collision = False
    safety_violation = False

    for q in trajectory:
        proximal, tip = instrument.shaft_segment(
            q,
            proximal_length=proximal_length,
        )

        safety = evaluate_instrument_safety(
            shaft_start=proximal,
            shaft_end=tip,
            structures=true_structures,
            instrument_radius=instrument_radius,
        )

        minimum_safety_clearance = min(
            minimum_safety_clearance,
            safety.minimum_safety_clearance,
        )

        collision = (
            collision
            or safety.collision
        )

        safety_violation = (
            safety_violation
            or safety.safety_margin_violation
        )

    return (
        minimum_safety_clearance,
        collision,
        safety_violation,
    )


def run_method(
    *,
    localisation_sigma: float,
    sigma_multiplier: float,
    trial: int,
    method: str,
    instrument: SurgicalInstrument,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    planning_structures: tuple[SphericalStructure, ...],
    true_structures: tuple[SphericalStructure, ...],
    instrument_radius: float,
    proximal_length: float,
    planner_seed: int,
) -> SweepTrialResult:
    """Run one planner trial."""

    start_time = time.perf_counter()

    result = plan_rrt(
        instrument=instrument,
        start_q=start_q,
        goal_q=goal_q,
        structures=planning_structures,
        instrument_radius=instrument_radius,
        proximal_length=proximal_length,
        max_iterations=10000,
        step_size=0.08,
        goal_bias=0.20,
        edge_resolution=30,
        seed=planner_seed,
    )

    planning_time = (
        time.perf_counter()
        - start_time
    )

    if not result.success:
        return SweepTrialResult(
            localisation_sigma=localisation_sigma,
            sigma_multiplier=sigma_multiplier,
            trial=trial,
            method=method,
            planning_success=False,
            planning_time=planning_time,
            path_cost=float("inf"),
            minimum_true_safety_clearance=float("-inf"),
            collision=False,
            safety_violation=False,
        )

    smoothed_path = shortcut_path(
        instrument=instrument,
        path=result.path,
        structures=planning_structures,
        instrument_radius=instrument_radius,
        proximal_length=proximal_length,
        edge_resolution=40,
        attempts=500,
        seed=planner_seed + 1000,
    )

    trajectory = densify_path(
        smoothed_path,
        samples_per_edge=40,
    )

    (
        minimum_clearance,
        collision,
        safety_violation,
    ) = evaluate_against_truth(
        instrument=instrument,
        trajectory=trajectory,
        true_structures=true_structures,
        instrument_radius=instrument_radius,
        proximal_length=proximal_length,
    )

    return SweepTrialResult(
        localisation_sigma=localisation_sigma,
        sigma_multiplier=sigma_multiplier,
        trial=trial,
        method=method,
        planning_success=True,
        planning_time=planning_time,
        path_cost=path_cost(smoothed_path),
        minimum_true_safety_clearance=minimum_clearance,
        collision=collision,
        safety_violation=safety_violation,
    )


def summarise_condition(
    results: list[SweepTrialResult],
    localisation_sigma: float,
    sigma_multiplier: float,
    method: str,
) -> None:
    """Print summary statistics for one experimental condition."""

    selected = [
        result
        for result in results
        if (
            result.localisation_sigma
            == localisation_sigma
            and result.sigma_multiplier
            == sigma_multiplier
            and result.method
            == method
        )
    ]

    successful = [
        result
        for result in selected
        if result.planning_success
    ]

    success_rate = (
        100.0
        * len(successful)
        / len(selected)
    )

    if len(successful) == 0:
        print(
            f"{method:22s} "
            f"success={success_rate:5.1f}% "
            f"no successful plans"
        )
        return

    violation_rate = (
        100.0
        * np.mean(
            [
                result.safety_violation
                for result in successful
            ]
        )
    )

    collision_rate = (
        100.0
        * np.mean(
            [
                result.collision
                for result in successful
            ]
        )
    )

    clearances = np.array(
        [
            result.minimum_true_safety_clearance
            for result in successful
        ],
        dtype=float,
    )

    costs = np.array(
        [
            result.path_cost
            for result in successful
        ],
        dtype=float,
    )

    planning_times = np.array(
        [
            result.planning_time
            for result in successful
        ],
        dtype=float,
    )

    print(
        f"{method:22s} "
        f"success={success_rate:5.1f}% "
        f"violation={violation_rate:5.1f}% "
        f"collision={collision_rate:5.1f}% "
        f"mean_clearance={np.mean(clearances): .6f} m "
        f"min_clearance={np.min(clearances): .6f} m "
        f"cost={np.mean(costs):.4f} "
        f"time={np.mean(planning_times):.3f} s"
    )


def main() -> None:
    """Run an uncertainty-level and margin-multiplier sweep."""

    start_q = np.array(
        [
            np.deg2rad(-25.0),
            np.deg2rad(-15.0),
            0.16,
            0.0,
        ],
        dtype=float,
    )

    goal_q = np.array(
        [
            np.deg2rad(35.0),
            np.deg2rad(25.0),
            0.25,
            0.0,
        ],
        dtype=float,
    )

    true_structures = (
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

    instrument = SurgicalInstrument(
        rcm_position=np.zeros(
            3,
            dtype=float,
        )
    )

    instrument_radius = 0.006
    proximal_length = 0.10

    localisation_sigmas = (
        0.002,
        0.005,
        0.008,
    )

    sigma_multipliers = (
        0.0,
        1.0,
        2.0,
        3.0,
    )

    trials_per_condition = 10

    results: list[SweepTrialResult] = []

    print()
    print("Uncertainty parameter sweep")
    print("---------------------------")
    print(
        f"Trials per condition: "
        f"{trials_per_condition}"
    )

    for localisation_sigma in localisation_sigmas:
        uncertainty = PositionUncertainty.isotropic(
            sigma=localisation_sigma
        )

        for trial in range(
            trials_per_condition
        ):
            perception = perceive_structures(
                true_structures=true_structures,
                uncertainty=uncertainty,
                rng=np.random.default_rng(
                    10000
                    + int(
                        localisation_sigma
                        * 1_000_000
                    )
                    + trial
                ),
            )

            deterministic = (
                deterministic_planning_structures(
                    perception
                )
            )

            deterministic_result = run_method(
                localisation_sigma=localisation_sigma,
                sigma_multiplier=0.0,
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
                planner_seed=20000 + trial,
            )

            results.append(
                deterministic_result
            )

            for sigma_multiplier in sigma_multipliers[1:]:
                aware = (
                    uncertainty_aware_planning_structures(
                        perception_result=perception,
                        sigma_multiplier=sigma_multiplier,
                    )
                )

                aware_result = run_method(
                    localisation_sigma=localisation_sigma,
                    sigma_multiplier=sigma_multiplier,
                    trial=trial,
                    method="Uncertainty-aware RRT",
                    instrument=instrument,
                    start_q=start_q,
                    goal_q=goal_q,
                    planning_structures=(
                        aware.structures
                    ),
                    true_structures=true_structures,
                    instrument_radius=instrument_radius,
                    proximal_length=proximal_length,
                    planner_seed=20000 + trial,
                )

                results.append(
                    aware_result
                )

            print(
                f"sigma={localisation_sigma:.3f} m "
                f"trial {trial + 1:02d}/"
                f"{trials_per_condition} complete"
            )

    print()
    print("Sweep summary")
    print("-------------")

    for localisation_sigma in localisation_sigmas:
        print()
        print(
            f"Localisation sigma = "
            f"{localisation_sigma:.3f} m"
        )
        print(
            "-" * 36
        )

        summarise_condition(
            results=results,
            localisation_sigma=localisation_sigma,
            sigma_multiplier=0.0,
            method="Deterministic RRT",
        )

        for sigma_multiplier in sigma_multipliers[1:]:
            print(
                f"k = {sigma_multiplier:.1f}"
            )

            summarise_condition(
                results=results,
                localisation_sigma=localisation_sigma,
                sigma_multiplier=sigma_multiplier,
                method="Uncertainty-aware RRT",
            )


if __name__ == "__main__":
    main()