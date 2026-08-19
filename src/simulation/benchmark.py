"""Multi-seed benchmark for RRT surgical motion planning."""

from __future__ import annotations

import time

import numpy as np

from src.simulation.scene import (
    SurgicalScene,
    densify_path,
    evaluate_trajectory,
)
from src.robotics.planner import (
    path_cost,
    plan_rrt,
    shortcut_path,
)


def main() -> None:
    """Benchmark raw and smoothed RRT planning across multiple seeds."""

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

    seeds = list(range(10))

    records: list[dict[str, float | int | bool]] = []

    scene = SurgicalScene(gui=False)

    for seed in seeds:
        planning_start = time.perf_counter()

        result = plan_rrt(
            instrument=scene.instrument_model,
            start_q=start_q,
            goal_q=goal_q,
            structures=scene.structures,
            instrument_radius=scene.instrument_radius,
            proximal_length=scene.instrument_proximal_length,
            max_iterations=10000,
            step_size=0.08,
            goal_bias=0.20,
            edge_resolution=30,
            seed=seed,
        )

        planning_time = (
            time.perf_counter()
            - planning_start
        )

        if not result.success:
            records.append(
                {
                    "seed": seed,
                    "success": False,
                    "planning_time": planning_time,
                    "iterations": result.iterations,
                }
            )

            continue

        raw_path = result.path

        smoothing_start = time.perf_counter()

        smoothed_path = shortcut_path(
            instrument=scene.instrument_model,
            path=raw_path,
            structures=scene.structures,
            instrument_radius=scene.instrument_radius,
            proximal_length=scene.instrument_proximal_length,
            edge_resolution=40,
            attempts=500,
            seed=seed + 1000,
        )

        smoothing_time = (
            time.perf_counter()
            - smoothing_start
        )

        raw_trajectory = densify_path(
            raw_path,
            samples_per_edge=40,
        )

        smoothed_trajectory = densify_path(
            smoothed_path,
            samples_per_edge=40,
        )

        raw_metrics = evaluate_trajectory(
            instrument=scene.instrument_model,
            trajectory=raw_trajectory,
            structures=scene.structures,
            instrument_radius=scene.instrument_radius,
            proximal_length=scene.instrument_proximal_length,
        )

        smoothed_metrics = evaluate_trajectory(
            instrument=scene.instrument_model,
            trajectory=smoothed_trajectory,
            structures=scene.structures,
            instrument_radius=scene.instrument_radius,
            proximal_length=scene.instrument_proximal_length,
        )

        records.append(
            {
                "seed": seed,
                "success": True,
                "planning_time": planning_time,
                "smoothing_time": smoothing_time,
                "iterations": result.iterations,
                "raw_waypoints": len(raw_path),
                "smoothed_waypoints": len(smoothed_path),
                "raw_cost": path_cost(raw_path),
                "smoothed_cost": path_cost(smoothed_path),
                "raw_min_clearance": (
                    raw_metrics.minimum_safety_clearance
                ),
                "smoothed_min_clearance": (
                    smoothed_metrics.minimum_safety_clearance
                ),
                "raw_collision": raw_metrics.collision,
                "smoothed_collision": smoothed_metrics.collision,
                "raw_violation": (
                    raw_metrics.safety_margin_violation
                ),
                "smoothed_violation": (
                    smoothed_metrics.safety_margin_violation
                ),
                "raw_rcm_error": (
                    raw_metrics.maximum_rcm_error
                ),
                "smoothed_rcm_error": (
                    smoothed_metrics.maximum_rcm_error
                ),
            }
        )

    scene.close()

    successful = [
        record
        for record in records
        if record["success"]
    ]

    print()
    print("Multi-seed RRT benchmark")
    print("------------------------")
    print(
        f"Seeds evaluated: {len(records)}"
    )
    print(
        f"Successful plans: {len(successful)}"
    )
    print(
        "Success rate: "
        f"{100.0 * len(successful) / len(records):.1f}%"
    )

    if len(successful) == 0:
        print(
            "No successful plans available for further analysis."
        )
        return

    planning_times = np.array(
        [
            float(record["planning_time"])
            for record in successful
        ]
    )

    iterations = np.array(
        [
            int(record["iterations"])
            for record in successful
        ]
    )

    raw_costs = np.array(
        [
            float(record["raw_cost"])
            for record in successful
        ]
    )

    smoothed_costs = np.array(
        [
            float(record["smoothed_cost"])
            for record in successful
        ]
    )

    raw_waypoints = np.array(
        [
            int(record["raw_waypoints"])
            for record in successful
        ]
    )

    smoothed_waypoints = np.array(
        [
            int(record["smoothed_waypoints"])
            for record in successful
        ]
    )

    smoothed_clearances = np.array(
        [
            float(record["smoothed_min_clearance"])
            for record in successful
        ]
    )

    smoothed_rcm_errors = np.array(
        [
            float(record["smoothed_rcm_error"])
            for record in successful
        ]
    )

    cost_reduction = (
        100.0
        * (
            raw_costs
            - smoothed_costs
        )
        / raw_costs
    )

    waypoint_reduction = (
        100.0
        * (
            raw_waypoints
            - smoothed_waypoints
        )
        / raw_waypoints
    )

    print()
    print("Planning performance")
    print("--------------------")
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

    print()
    print("Path optimisation")
    print("-----------------")
    print(
        "Mean raw waypoints: "
        f"{np.mean(raw_waypoints):.2f}"
    )
    print(
        "Mean smoothed waypoints: "
        f"{np.mean(smoothed_waypoints):.2f}"
    )
    print(
        "Mean waypoint reduction: "
        f"{np.mean(waypoint_reduction):.2f}%"
    )
    print(
        "Mean raw path cost: "
        f"{np.mean(raw_costs):.4f}"
    )
    print(
        "Mean smoothed path cost: "
        f"{np.mean(smoothed_costs):.4f}"
    )
    print(
        "Mean path-cost reduction: "
        f"{np.mean(cost_reduction):.2f}%"
    )

    print()
    print("Safety")
    print("------")
    print(
        "Minimum smoothed safety clearance "
        "across all successful runs: "
        f"{np.min(smoothed_clearances):.6f} m"
    )
    print(
        "Any smoothed collision: "
        f"{any(bool(record['smoothed_collision']) for record in successful)}"
    )
    print(
        "Any smoothed safety violation: "
        f"{any(bool(record['smoothed_violation']) for record in successful)}"
    )
    print(
        "Maximum RCM error across runs: "
        f"{np.max(smoothed_rcm_errors):.12e} m"
    )

    print()
    print("Per-seed results")
    print("----------------")

    for record in records:
        if not record["success"]:
            print(
                f"Seed {record['seed']:2d}: FAILED"
            )

            continue

        print(
            f"Seed {record['seed']:2d}: "
            f"iterations={record['iterations']:4d}, "
            f"raw_wp={record['raw_waypoints']:2d}, "
            f"smooth_wp={record['smoothed_waypoints']:2d}, "
            f"raw_cost={record['raw_cost']:.4f}, "
            f"smooth_cost={record['smoothed_cost']:.4f}, "
            f"clearance={record['smoothed_min_clearance']:.6f} m"
        )


if __name__ == "__main__":
    main()