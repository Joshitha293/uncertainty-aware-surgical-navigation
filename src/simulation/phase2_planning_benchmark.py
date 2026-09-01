"""Final Phase 2 robotics and motion-planning evidence benchmark.

This benchmark evaluates the new Phase 2 capabilities without modifying
the frozen Phase 1 experimental evidence.

Evidence generated:

- admissible RCM workspace characterisation;
- robot-reachable viewpoint filtering;
- RRT planning;
- RRT* planning;
- discretised configuration-space A*;
- collision-checked path shortcutting;
- timed joint-trajectory generation;
- execution metrics.

The benchmark is simulation-only. Timing and actuator limits are modelling
parameters and are not claims about a commercial surgical robot.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src.geometry.workspace import (
    SphericalStructure,
)
from src.perception.camera import (
    CameraPose,
)
from src.perception.robot_reachability import (
    endoscope_camera_pose,
    filter_reachable_viewpoints,
)
from src.perception.viewpoints import (
    CandidateViewpoint,
)
from src.robotics.advanced_planning import (
    AStarGridConfig,
    plan_astar_configuration_grid,
    plan_rrt_star,
)
from src.robotics.instrument import (
    SurgicalInstrument,
)
from src.robotics.planner import (
    PlanningResult,
    edge_is_safe,
    path_cost,
    plan_rrt,
    shortcut_path,
)
from src.robotics.trajectory_execution import (
    evaluate_execution_metrics,
    time_parameterise_path,
)
from src.robotics.workspace_analysis import (
    characterise_kinematic_workspace,
)


OUTPUT_PATH = Path(
    "results/phase2/phase2_planning_benchmark.json"
)

INSTRUMENT_RADIUS = 0.006
PROXIMAL_LENGTH = 0.10

SEEDS = (
    7,
    17,
    27,
)


def make_instrument() -> SurgicalInstrument:
    """Create the deterministic Phase 2 RCM instrument."""

    return SurgicalInstrument(
        rcm_position=np.zeros(
            3,
            dtype=float,
        )
    )


def make_structures() -> tuple[
    SphericalStructure,
    ...,
]:
    """Create the established obstacle environment."""

    return (
        SphericalStructure(
            centre=np.asarray(
                [
                    0.14,
                    0.04,
                    0.00,
                ],
                dtype=float,
            ),
            physical_radius=0.025,
            safety_margin=0.015,
        ),
        SphericalStructure(
            centre=np.asarray(
                [
                    0.18,
                    -0.06,
                    0.02,
                ],
                dtype=float,
            ),
            physical_radius=0.025,
            safety_margin=0.015,
        ),
    )


def start_configuration() -> np.ndarray:
    """Return benchmark start configuration."""

    return np.asarray(
        [
            np.deg2rad(
                -25.0
            ),
            np.deg2rad(
                -15.0
            ),
            0.16,
            0.0,
        ],
        dtype=float,
    )


def goal_configuration() -> np.ndarray:
    """Return benchmark goal configuration."""

    return np.asarray(
        [
            np.deg2rad(
                35.0
            ),
            np.deg2rad(
                25.0
            ),
            0.25,
            0.0,
        ],
        dtype=float,
    )


def _path_is_safe(
    instrument: SurgicalInstrument,
    path: np.ndarray,
    structures: tuple[
        SphericalStructure,
        ...,
    ],
) -> bool:
    """Verify every path edge against the established safety checker."""

    if path.shape[0] < 2:
        return False

    return all(
        edge_is_safe(
            instrument=instrument,
            q_start=path[index],
            q_goal=path[
                index + 1
            ],
            structures=structures,
            instrument_radius=(
                INSTRUMENT_RADIUS
            ),
            proximal_length=(
                PROXIMAL_LENGTH
            ),
            resolution=30,
        )
        for index in range(
            len(
                path
            )
            - 1
        )
    )


def _evaluate_planning_result(
    *,
    name: str,
    instrument: SurgicalInstrument,
    structures: tuple[
        SphericalStructure,
        ...,
    ],
    result: PlanningResult,
    planning_time_seconds: float,
    shortcut_seed: int,
) -> dict:
    """Convert one planner result into benchmark evidence."""

    record = {
        "planner": name,
        "success": bool(
            result.success
        ),
        "iterations": int(
            result.iterations
        ),
        "planning_time_seconds": float(
            planning_time_seconds
        ),
    }

    if not result.success:
        record.update(
            {
                "raw_waypoint_count": 0,
                "raw_path_cost": None,
                "raw_path_safe": False,
                "smoothed_waypoint_count": 0,
                "smoothed_path_cost": None,
                "smoothed_path_safe": False,
                "trajectory_duration_seconds": None,
                "trajectory_sample_count": 0,
                "cartesian_tip_distance_metres": None,
                "minimum_joint_limit_margin": None,
                "minimum_manipulability": None,
                "maximum_condition_number": None,
                "maximum_rcm_error_metres": None,
                "acceleration_energy": None,
            }
        )

        return record

    raw_path = np.asarray(
        result.path,
        dtype=float,
    )

    raw_safe = _path_is_safe(
        instrument,
        raw_path,
        structures,
    )

    smoothed_path = shortcut_path(
        instrument=instrument,
        path=raw_path,
        structures=structures,
        instrument_radius=(
            INSTRUMENT_RADIUS
        ),
        proximal_length=(
            PROXIMAL_LENGTH
        ),
        edge_resolution=30,
        attempts=200,
        seed=shortcut_seed,
    )

    smoothed_safe = _path_is_safe(
        instrument,
        smoothed_path,
        structures,
    )

    trajectory = time_parameterise_path(
        instrument,
        smoothed_path,
    )

    execution = (
        evaluate_execution_metrics(
            instrument,
            trajectory,
        )
    )

    record.update(
        {
            "raw_waypoint_count": int(
                raw_path.shape[
                    0
                ]
            ),
            "raw_path_cost": float(
                path_cost(
                    raw_path
                )
            ),
            "raw_path_safe": bool(
                raw_safe
            ),
            "smoothed_waypoint_count": int(
                smoothed_path.shape[
                    0
                ]
            ),
            "smoothed_path_cost": float(
                path_cost(
                    smoothed_path
                )
            ),
            "smoothed_path_safe": bool(
                smoothed_safe
            ),
            "trajectory_duration_seconds": float(
                execution.duration
            ),
            "trajectory_sample_count": int(
                execution.sample_count
            ),
            "cartesian_tip_distance_metres": float(
                execution.cartesian_tip_distance
            ),
            "minimum_joint_limit_margin": float(
                execution.minimum_joint_limit_margin
            ),
            "minimum_manipulability": float(
                execution.minimum_manipulability
            ),
            "maximum_condition_number": float(
                execution.maximum_condition_number
            ),
            "maximum_rcm_error_metres": float(
                execution.maximum_rcm_error
            ),
            "acceleration_energy": float(
                execution.acceleration_energy
            ),
        }
    )

    return record


def _run_rrt(
    instrument: SurgicalInstrument,
    structures: tuple[
        SphericalStructure,
        ...,
    ],
    seed: int,
) -> dict:
    """Run existing RRT baseline."""

    start_time = perf_counter()

    result = plan_rrt(
        instrument=instrument,
        start_q=start_configuration(),
        goal_q=goal_configuration(),
        structures=structures,
        instrument_radius=(
            INSTRUMENT_RADIUS
        ),
        proximal_length=(
            PROXIMAL_LENGTH
        ),
        max_iterations=5000,
        step_size=0.08,
        goal_bias=0.15,
        edge_resolution=20,
        seed=seed,
    )

    elapsed = (
        perf_counter()
        - start_time
    )

    return _evaluate_planning_result(
        name="RRT",
        instrument=instrument,
        structures=structures,
        result=result,
        planning_time_seconds=(
            elapsed
        ),
        shortcut_seed=(
            1000
            + seed
        ),
    )


def _run_rrt_star(
    instrument: SurgicalInstrument,
    structures: tuple[
        SphericalStructure,
        ...,
    ],
    seed: int,
) -> dict:
    """Run Phase 2 RRT*."""

    start_time = perf_counter()

    result = plan_rrt_star(
        instrument=instrument,
        start_q=start_configuration(),
        goal_q=goal_configuration(),
        structures=structures,
        instrument_radius=(
            INSTRUMENT_RADIUS
        ),
        proximal_length=(
            PROXIMAL_LENGTH
        ),
        max_iterations=500,
        step_size=0.08,
        goal_bias=0.15,
        edge_resolution=20,
        rewire_radius=0.60,
        seed=seed,
    )

    elapsed = (
        perf_counter()
        - start_time
    )

    return _evaluate_planning_result(
        name="RRT*",
        instrument=instrument,
        structures=structures,
        result=result,
        planning_time_seconds=(
            elapsed
        ),
        shortcut_seed=(
            2000
            + seed
        ),
    )


def _run_astar(
    instrument: SurgicalInstrument,
    structures: tuple[
        SphericalStructure,
        ...,
    ],
) -> dict:
    """Run discretised configuration-space A* baseline."""

    start_time = perf_counter()

    result = (
        plan_astar_configuration_grid(
            instrument=instrument,
            start_q=start_configuration(),
            goal_q=goal_configuration(),
            structures=structures,
            instrument_radius=(
                INSTRUMENT_RADIUS
            ),
            proximal_length=(
                PROXIMAL_LENGTH
            ),
            edge_resolution=12,
            grid_config=(
                AStarGridConfig(
                    yaw_samples=7,
                    pitch_samples=7,
                    insertion_samples=5,
                    roll_samples=3,
                    max_expansions=10000,
                )
            ),
        )
    )

    elapsed = (
        perf_counter()
        - start_time
    )

    return _evaluate_planning_result(
        name="A* configuration grid",
        instrument=instrument,
        structures=structures,
        result=result,
        planning_time_seconds=(
            elapsed
        ),
        shortcut_seed=3000,
    )


def _make_exact_candidate(
    instrument: SurgicalInstrument,
    q: np.ndarray,
) -> CandidateViewpoint:
    """Create one exactly robot-realizable camera viewpoint."""

    return CandidateViewpoint(
        pose=endoscope_camera_pose(
            instrument,
            q,
        ),
        radius=0.10,
        azimuth=0.0,
        elevation=0.0,
    )


def _reachability_evidence(
    instrument: SurgicalInstrument,
) -> dict:
    """Demonstrate robot-aware rejection of virtual viewpoints."""

    first = _make_exact_candidate(
        instrument,
        np.asarray(
            [
                0.10,
                -0.10,
                0.15,
                0.20,
            ],
            dtype=float,
        ),
    )

    second = _make_exact_candidate(
        instrument,
        np.asarray(
            [
                -0.20,
                0.15,
                0.20,
                -0.30,
            ],
            dtype=float,
        ),
    )

    impossible = CandidateViewpoint(
        pose=CameraPose(
            position=np.asarray(
                [
                    0.50,
                    0.0,
                    0.0,
                ],
                dtype=float,
            ),
            rotation=(
                first.pose.rotation
            ),
        ),
        radius=0.10,
        azimuth=0.0,
        elevation=0.0,
    )

    result = filter_reachable_viewpoints(
        instrument=instrument,
        candidates=(
            first,
            second,
            impossible,
        ),
    )

    rejection_reasons = [
        list(
            evaluation.rejection_reasons
        )
        for evaluation
        in result.evaluations
    ]

    return {
        "candidate_count": int(
            result.total_count
        ),
        "reachable_count": int(
            result.reachable_count
        ),
        "rejected_count": int(
            result.rejected_count
        ),
        "rejection_reasons": (
            rejection_reasons
        ),
    }


def _workspace_evidence(
    instrument: SurgicalInstrument,
) -> dict:
    """Generate quantitative workspace evidence."""

    result = (
        characterise_kinematic_workspace(
            instrument,
            yaw_samples=9,
            pitch_samples=7,
            insertion_samples=7,
            singular_value_threshold=1e-6,
        )
    )

    return {
        "sample_count": int(
            result.sample_count
        ),
        "tip_minimum_metres": [
            float(
                value
            )
            for value in (
                result.tip_minimum
            )
        ],
        "tip_maximum_metres": [
            float(
                value
            )
            for value in (
                result.tip_maximum
            )
        ],
        "minimum_manipulability": float(
            result.minimum_manipulability
        ),
        "maximum_manipulability": float(
            result.maximum_manipulability
        ),
        "mean_manipulability": float(
            result.mean_manipulability
        ),
        "minimum_condition_number": float(
            result.minimum_condition_number
        ),
        "maximum_condition_number": float(
            result.maximum_condition_number
        ),
        "mean_condition_number": float(
            result.mean_condition_number
        ),
        "minimum_singular_value": float(
            result.minimum_singular_value
        ),
        "maximum_singular_value": float(
            result.maximum_singular_value
        ),
        "near_singular_count": int(
            result.near_singular_count
        ),
        "near_singular_fraction": float(
            result.near_singular_fraction
        ),
    }


def _method_summary(
    records: list[dict],
) -> dict:
    """Summarise repeated planner trials."""

    successes = [
        record
        for record in records
        if record[
            "success"
        ]
    ]

    summary = {
        "trial_count": len(
            records
        ),
        "success_count": len(
            successes
        ),
        "success_rate": float(
            len(
                successes
            )
            / len(
                records
            )
        ),
    }

    if not successes:
        summary.update(
            {
                "mean_planning_time_seconds": None,
                "mean_raw_path_cost": None,
                "mean_smoothed_path_cost": None,
                "mean_trajectory_duration_seconds": None,
                "maximum_rcm_error_metres": None,
            }
        )

        return summary

    summary.update(
        {
            "mean_planning_time_seconds": float(
                np.mean(
                    [
                        record[
                            "planning_time_seconds"
                        ]
                        for record
                        in successes
                    ]
                )
            ),
            "mean_raw_path_cost": float(
                np.mean(
                    [
                        record[
                            "raw_path_cost"
                        ]
                        for record
                        in successes
                    ]
                )
            ),
            "mean_smoothed_path_cost": float(
                np.mean(
                    [
                        record[
                            "smoothed_path_cost"
                        ]
                        for record
                        in successes
                    ]
                )
            ),
            "mean_trajectory_duration_seconds": float(
                np.mean(
                    [
                        record[
                            "trajectory_duration_seconds"
                        ]
                        for record
                        in successes
                    ]
                )
            ),
            "maximum_rcm_error_metres": float(
                np.max(
                    [
                        record[
                            "maximum_rcm_error_metres"
                        ]
                        for record
                        in successes
                    ]
                )
            ),
        }
    )

    return summary


def run_benchmark() -> dict:
    """Execute the final Phase 2 evidence benchmark."""

    instrument = (
        make_instrument()
    )

    structures = (
        make_structures()
    )

    rrt_records = [
        _run_rrt(
            instrument,
            structures,
            seed,
        )
        for seed in SEEDS
    ]

    rrt_star_records = [
        _run_rrt_star(
            instrument,
            structures,
            seed,
        )
        for seed in SEEDS
    ]

    astar_record = _run_astar(
        instrument,
        structures,
    )

    all_successful_records = [
        record
        for record in (
            rrt_records
            + rrt_star_records
            + [
                astar_record
            ]
        )
        if record[
            "success"
        ]
    ]

    unsafe_successful = [
        record[
            "planner"
        ]
        for record
        in all_successful_records
        if (
            not record[
                "raw_path_safe"
            ]
            or not record[
                "smoothed_path_safe"
            ]
        )
    ]

    if unsafe_successful:
        raise RuntimeError(
            "A successful planner produced an unsafe path: "
            + ", ".join(
                unsafe_successful
            )
        )

    evidence = {
        "phase": 2,
        "benchmark": (
            "advanced_robot_kinematics_and_motion_planning"
        ),
        "scope": (
            "simulation_only"
        ),
        "claims_note": (
            "Results characterise the simplified simulated "
            "RCM-constrained instrument and do not constitute "
            "clinical or physical robot validation."
        ),
        "planner_seed_set": list(
            SEEDS
        ),
        "workspace": (
            _workspace_evidence(
                instrument
            )
        ),
        "viewpoint_reachability": (
            _reachability_evidence(
                instrument
            )
        ),
        "rrt_trials": rrt_records,
        "rrt_star_trials": (
            rrt_star_records
        ),
        "astar_trial": (
            astar_record
        ),
        "summary": {
            "RRT": _method_summary(
                rrt_records
            ),
            "RRT*": _method_summary(
                rrt_star_records
            ),
            "A* configuration grid": (
                _method_summary(
                    [
                        astar_record
                    ]
                )
            ),
        },
    }

    return evidence


def write_evidence(
    evidence: dict,
) -> None:
    """Write benchmark evidence to JSON."""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            evidence,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def print_summary(
    evidence: dict,
) -> None:
    """Print a concise human-readable benchmark summary."""

    print(
        "Phase 2 Advanced Robotics Benchmark"
    )

    print(
        "="
        * 36
    )

    workspace = evidence[
        "workspace"
    ]

    print(
        "Workspace samples: "
        f"{workspace['sample_count']}"
    )

    print(
        "Near-singular configurations: "
        f"{workspace['near_singular_count']}"
    )

    reachability = evidence[
        "viewpoint_reachability"
    ]

    print(
        "Reachable viewpoint demonstration: "
        f"{reachability['reachable_count']}/"
        f"{reachability['candidate_count']}"
    )

    print()

    for name, summary in (
        evidence[
            "summary"
        ].items()
    ):
        print(
            name
        )

        print(
            "  Success: "
            f"{summary['success_count']}/"
            f"{summary['trial_count']}"
        )

        if (
            summary[
                "mean_raw_path_cost"
            ]
            is not None
        ):
            print(
                "  Mean raw path cost: "
                f"{summary['mean_raw_path_cost']:.6f}"
            )

            print(
                "  Mean smoothed path cost: "
                f"{summary['mean_smoothed_path_cost']:.6f}"
            )

            print(
                "  Mean planning time: "
                f"{summary['mean_planning_time_seconds']:.6f} s"
            )

            print(
                "  Mean trajectory duration: "
                f"{summary['mean_trajectory_duration_seconds']:.6f} s"
            )

            print(
                "  Maximum RCM error: "
                f"{summary['maximum_rcm_error_metres']:.12e} m"
            )

        print()

    print(
        "Evidence written to:"
    )

    print(
        OUTPUT_PATH
    )


def main() -> None:
    """Run and save Phase 2 benchmark."""

    evidence = run_benchmark()

    write_evidence(
        evidence
    )

    print_summary(
        evidence
    )


if __name__ == "__main__":
    main()