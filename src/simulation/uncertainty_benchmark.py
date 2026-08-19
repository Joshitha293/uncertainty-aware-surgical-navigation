"""Monte Carlo benchmark for deterministic versus uncertainty-aware RRT."""

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
class TrialResult:
    """Result for one planner in one Monte Carlo trial."""

    trial: int
    method: str
    planning_success: bool
    planning_time: float
    iterations: int
    path_cost: float
    minimum_true_safety_clearance: float
    collision_against_truth: bool
    safety_violation_against_truth: bool
    maximum_rcm_error: float


def evaluate_against_truth(
    instrument: SurgicalInstrument,
    trajectory: np.ndarray,
    true_structures: tuple[SphericalStructure, ...],
    instrument_radius: float,
    proximal_length: float,
) -> tuple[float, bool, bool, float]:
    """Evaluate one trajectory against hidden ground-truth anatomy."""

    minimum_true_safety_clearance = float("inf")
    collision = False
    safety_violation = False
    maximum_rcm_error = 0.0

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

        minimum_true_safety_clearance = min(
            minimum_true_safety_clearance,
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

        maximum_rcm_error = max(
            maximum_rcm_error,
            instrument.rcm_error(q),
        )

    return (
        minimum_true_safety_clearance,
        collision,
        safety_violation,
        maximum_rcm_error,
    )


def run_planner_trial(
    *,
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
) -> TrialResult:
    """Run one planner and evaluate its result against hidden truth."""

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
        return TrialResult(
            trial=trial,
            method=method,
            planning_success=False,
            planning_time=planning_time,
            iterations=result.iterations,
            path_cost=float("inf"),
            minimum_true_safety_clearance=float("-inf"),
            collision_against_truth=False,
            safety_violation_against_truth=False,
            maximum_rcm_error=float("nan"),
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
        minimum_true_safety_clearance,
        collision,
        safety_violation,
        maximum_rcm_error,
    ) = evaluate_against_truth(
        instrument=instrument,
        trajectory=trajectory,
        true_structures=true_structures,
        instrument_radius=instrument_radius,
        proximal_length=proximal_length,
    )

    return TrialResult(
        trial=trial,
        method=method,
        planning_success=True,
        planning_time=planning_time,
        iterations=result.iterations,
        path_cost=path_cost(smoothed_path),
        minimum_true_safety_clearance=minimum_true_safety_clearance,
        collision_against_truth=collision,
        safety_violation_against_truth=safety_violation,
        maximum_rcm_error=maximum_rcm_error,
    )


def summarise_results(
    results: list[TrialResult],
    method: str,
) -> None:
    """Print summary metrics for one method."""

    method_results = [
        result
        for result in results
        if result.method == method
    ]

    successful = [
        result
        for result in method_results
        if result.planning_success
    ]

    success_rate = (
        100.0
        * len(successful)
        / len(method_results)
    )

    print()
    print(method)
    print("-" * len(method))

    print(
        f"Planning success rate: "
        f"{success_rate:.1f}%"
    )

    if len(successful) == 0:
        return

    planning_times = np.array(
        [
            result.planning_time
            for result in successful
        ],
        dtype=float,
    )

    iterations = np.array(
        [
            result.iterations
            for result in successful
        ],
        dtype=float,
    )

    path_costs = np.array(
        [
            result.path_cost
            for result in successful
        ],
        dtype=float,
    )

    true_clearances = np.array(
        [
            result.minimum_true_safety_clearance
            for result in successful
        ],
        dtype=float,
    )

    collision_rate = (
        100.0
        * np.mean(
            [
                result.collision_against_truth
                for result in successful
            ]
        )
    )

    violation_rate = (
        100.0
        * np.mean(
            [
                result.safety_violation_against_truth
                for result in successful
            ]
        )
    )

    rcm_errors = np.array(
        [
            result.maximum_rcm_error
            for result in successful
        ],
        dtype=float,
    )

    print(
        "Mean planning time: "
        f"{np.mean(planning_times):.4f} s"
    )

    print(
        "Planning-time SD: "
        f"{np.std(planning_times):.4f} s"
    )

    print(
        "Mean iterations: "
        f"{np.mean(iterations):.1f}"
    )

    print(
        "Mean path cost: "
        f"{np.mean(path_costs):.4f}"
    )

    print(
        "Mean true safety clearance: "
        f"{np.mean(true_clearances):.6f} m"
    )

    print(
        "Minimum true safety clearance: "
        f"{np.min(true_clearances):.6f} m"
    )

    print(
        "Ground-truth collision rate: "
        f"{collision_rate:.1f}%"
    )

    print(
        "Ground-truth safety-violation rate: "
        f"{violation_rate:.1f}%"
    )

    print(
        "Maximum RCM error: "
        f"{np.max(rcm_errors):.12e} m"
    )


def main() -> None:
    """Run paired Monte Carlo comparison across noisy anatomical estimates."""

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

    localisation_sigma = 0.005
    sigma_multiplier = 2.0

    uncertainty = PositionUncertainty.isotropic(
        sigma=localisation_sigma
    )

    trial_count = 30

    results: list[TrialResult] = []

    print()
    print("Monte Carlo uncertainty benchmark")
    print("---------------------------------")
    print(
        f"Trials: {trial_count}"
    )
    print(
        f"Localisation sigma: "
        f"{localisation_sigma:.6f} m"
    )
    print(
        f"Uncertainty multiplier: "
        f"{sigma_multiplier:.2f}"
    )

    for trial in range(trial_count):
        perception_rng = np.random.default_rng(
            1000 + trial
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

        deterministic_result = run_planner_trial(
            trial=trial,
            method="Deterministic RRT",
            instrument=instrument,
            start_q=start_q,
            goal_q=goal_q,
            planning_structures=deterministic.structures,
            true_structures=true_structures,
            instrument_radius=instrument_radius,
            proximal_length=proximal_length,
            planner_seed=planner_seed,
        )

        uncertainty_result = run_planner_trial(
            trial=trial,
            method="Uncertainty-aware RRT",
            instrument=instrument,
            start_q=start_q,
            goal_q=goal_q,
            planning_structures=uncertainty_aware.structures,
            true_structures=true_structures,
            instrument_radius=instrument_radius,
            proximal_length=proximal_length,
            planner_seed=planner_seed,
        )

        results.append(
            deterministic_result
        )

        results.append(
            uncertainty_result
        )

        print(
            f"Trial {trial + 1:02d}/{trial_count} complete"
        )

    summarise_results(
        results,
        "Deterministic RRT",
    )

    summarise_results(
        results,
        "Uncertainty-aware RRT",
    )


if __name__ == "__main__":
    main()