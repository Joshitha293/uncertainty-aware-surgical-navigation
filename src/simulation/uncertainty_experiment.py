"""Controlled comparison of deterministic and uncertainty-aware RRT planning."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from src.geometry.workspace import SphericalStructure
from src.perception.perception import (
    maximum_localisation_error,
    mean_localisation_error,
    perceive_structures,
)
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
class MethodResult:
    """Quantitative result for one planning method."""

    name: str
    planning_success: bool
    iterations: int
    planning_time: float
    smoothing_time: float
    waypoints: int
    path_cost: float
    minimum_true_surface_clearance: float
    minimum_true_safety_clearance: float
    collision_against_truth: bool
    safety_violation_against_truth: bool
    maximum_rcm_error: float


def evaluate_against_ground_truth(
    instrument: SurgicalInstrument,
    trajectory: np.ndarray,
    true_structures: tuple[SphericalStructure, ...],
    instrument_radius: float,
    proximal_length: float,
) -> tuple[float, float, bool, bool, float]:
    """Evaluate a trajectory against hidden ground-truth anatomy."""

    minimum_surface_clearance = float("inf")
    minimum_safety_clearance = float("inf")

    collision = False
    safety_violation = False
    maximum_rcm_error = 0.0

    for q in trajectory:
        proximal, tip = instrument.shaft_segment(
            q,
            proximal_length=proximal_length,
        )

        evaluation = evaluate_instrument_safety(
            shaft_start=proximal,
            shaft_end=tip,
            structures=true_structures,
            instrument_radius=instrument_radius,
        )

        minimum_surface_clearance = min(
            minimum_surface_clearance,
            evaluation.minimum_surface_clearance,
        )

        minimum_safety_clearance = min(
            minimum_safety_clearance,
            evaluation.minimum_safety_clearance,
        )

        collision = (
            collision
            or evaluation.collision
        )

        safety_violation = (
            safety_violation
            or evaluation.safety_margin_violation
        )

        maximum_rcm_error = max(
            maximum_rcm_error,
            instrument.rcm_error(q),
        )

    return (
        minimum_surface_clearance,
        minimum_safety_clearance,
        collision,
        safety_violation,
        maximum_rcm_error,
    )


def run_method(
    name: str,
    instrument: SurgicalInstrument,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    planning_structures: tuple[SphericalStructure, ...],
    true_structures: tuple[SphericalStructure, ...],
    instrument_radius: float,
    proximal_length: float,
    planner_seed: int,
) -> MethodResult:
    """Plan using perceived anatomy, then evaluate against ground truth."""

    planning_start = time.perf_counter()

    planning_result = plan_rrt(
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
        - planning_start
    )

    if not planning_result.success:
        return MethodResult(
            name=name,
            planning_success=False,
            iterations=planning_result.iterations,
            planning_time=planning_time,
            smoothing_time=0.0,
            waypoints=0,
            path_cost=float("inf"),
            minimum_true_surface_clearance=float("-inf"),
            minimum_true_safety_clearance=float("-inf"),
            collision_against_truth=False,
            safety_violation_against_truth=False,
            maximum_rcm_error=float("nan"),
        )

    smoothing_start = time.perf_counter()

    smoothed_path = shortcut_path(
        instrument=instrument,
        path=planning_result.path,
        structures=planning_structures,
        instrument_radius=instrument_radius,
        proximal_length=proximal_length,
        edge_resolution=40,
        attempts=500,
        seed=planner_seed + 1000,
    )

    smoothing_time = (
        time.perf_counter()
        - smoothing_start
    )

    trajectory = densify_path(
        smoothed_path,
        samples_per_edge=40,
    )

    (
        minimum_surface_clearance,
        minimum_safety_clearance,
        collision,
        safety_violation,
        maximum_rcm_error,
    ) = evaluate_against_ground_truth(
        instrument=instrument,
        trajectory=trajectory,
        true_structures=true_structures,
        instrument_radius=instrument_radius,
        proximal_length=proximal_length,
    )

    return MethodResult(
        name=name,
        planning_success=True,
        iterations=planning_result.iterations,
        planning_time=planning_time,
        smoothing_time=smoothing_time,
        waypoints=len(smoothed_path),
        path_cost=path_cost(smoothed_path),
        minimum_true_surface_clearance=(
            minimum_surface_clearance
        ),
        minimum_true_safety_clearance=(
            minimum_safety_clearance
        ),
        collision_against_truth=collision,
        safety_violation_against_truth=safety_violation,
        maximum_rcm_error=maximum_rcm_error,
    )


def main() -> None:
    """Run one matched deterministic vs uncertainty-aware comparison."""

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

    uncertainty = PositionUncertainty.isotropic(
        sigma=localisation_sigma
    )

    perception_seed = 42
    planner_seed = 7

    perception = perceive_structures(
        true_structures=true_structures,
        uncertainty=uncertainty,
        rng=np.random.default_rng(
            perception_seed
        ),
    )

    deterministic_perception = (
        deterministic_planning_structures(
            perception
        )
    )

    uncertainty_aware_perception = (
        uncertainty_aware_planning_structures(
            perception_result=perception,
            sigma_multiplier=2.0,
        )
    )

    deterministic_result = run_method(
        name="Deterministic RRT",
        instrument=instrument,
        start_q=start_q,
        goal_q=goal_q,
        planning_structures=(
            deterministic_perception.structures
        ),
        true_structures=true_structures,
        instrument_radius=instrument_radius,
        proximal_length=proximal_length,
        planner_seed=planner_seed,
    )

    uncertainty_aware_result = run_method(
        name="Uncertainty-aware RRT",
        instrument=instrument,
        start_q=start_q,
        goal_q=goal_q,
        planning_structures=(
            uncertainty_aware_perception.structures
        ),
        true_structures=true_structures,
        instrument_radius=instrument_radius,
        proximal_length=proximal_length,
        planner_seed=planner_seed,
    )

    print()
    print("Matched uncertainty experiment")
    print("------------------------------")
    print(
        f"Localisation sigma: "
        f"{localisation_sigma:.6f} m"
    )
    print(
        f"Mean localisation error: "
        f"{mean_localisation_error(perception):.6f} m"
    )
    print(
        f"Maximum localisation error: "
        f"{maximum_localisation_error(perception):.6f} m"
    )

    for result in (
        deterministic_result,
        uncertainty_aware_result,
    ):
        print()
        print(result.name)
        print("-" * len(result.name))

        print(
            f"Planning success: "
            f"{result.planning_success}"
        )

        print(
            f"Iterations: "
            f"{result.iterations}"
        )

        print(
            f"Planning time: "
            f"{result.planning_time:.6f} s"
        )

        print(
            f"Smoothing time: "
            f"{result.smoothing_time:.6f} s"
        )

        print(
            f"Smoothed waypoints: "
            f"{result.waypoints}"
        )

        print(
            f"Path cost: "
            f"{result.path_cost:.6f}"
        )

        print(
            "Minimum true physical clearance: "
            f"{result.minimum_true_surface_clearance:.6f} m"
        )

        print(
            "Minimum true safety clearance: "
            f"{result.minimum_true_safety_clearance:.6f} m"
        )

        print(
            "Collision against ground truth: "
            f"{result.collision_against_truth}"
        )

        print(
            "Safety violation against ground truth: "
            f"{result.safety_violation_against_truth}"
        )

        print(
            "Maximum RCM error: "
            f"{result.maximum_rcm_error:.12e} m"
        )


if __name__ == "__main__":
    main()