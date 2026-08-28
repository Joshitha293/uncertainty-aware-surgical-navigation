"""Development sweep for fair Generic and Task-Aware viewpoint scoring.

This experiment examines the behaviour of the scale-controlled fair scoring
formulation over the established ten robustness scenarios.

Both Generic and Task-Aware strategies use exactly the same:

- candidate viewpoints;
- observation model;
- perception information term;
- movement-cost term;
- movement weight.

Task-Aware receives exactly one additional source of information:
task-relevance-weighted viewpoint alignment.

This is a development-stage diagnostic. It is not the final held-out
benchmark and must not be used to make final generalisation claims.
"""

from __future__ import annotations

import argparse
import csv
import json

from dataclasses import (
    asdict,
    dataclass,
)
from pathlib import Path

import numpy as np

from src.perception.fair_viewpoint_scoring import (
    FairScoringConfig,
    FairViewpointScorer,
)
from src.simulation.three_strategy_robustness_benchmark import (
    RobustnessScenario,
    build_scenario_inputs,
    default_scenarios,
)


DEFAULT_MOVEMENT_WEIGHTS = (
    0.000,
    0.025,
    0.050,
    0.075,
    0.100,
    0.150,
    0.200,
    0.250,
    0.500,
    1.000,
)


@dataclass(frozen=True)
class FairSweepRecord:
    """One scenario / movement-weight comparison."""

    scenario_id: int
    scenario_name: str

    movement_weight: float

    generic_candidate_index: int
    task_candidate_index: int

    generic_movement: float
    task_movement: float

    generic_sigma: float
    task_sigma: float

    generic_alignment: float
    task_alignment: float

    generic_information: float
    task_information: float

    generic_normalised_movement: float
    task_normalised_movement: float

    generic_score: float
    task_score: float

    different_candidate: bool


@dataclass(frozen=True)
class FairSweepSummary:
    """Aggregate metrics for one movement weight."""

    movement_weight: float
    scenario_count: int

    mean_generic_movement: float
    mean_task_movement: float
    mean_movement_difference: float

    mean_generic_sigma: float
    mean_task_sigma: float
    mean_sigma_difference: float

    mean_generic_alignment: float
    mean_task_alignment: float
    mean_alignment_difference: float

    different_candidate_rate: float


@dataclass(frozen=True)
class FairSweepResult:
    """Complete fair-scoring diagnostic."""

    movement_weights: tuple[
        float,
        ...,
    ]

    scenarios: tuple[
        RobustnessScenario,
        ...,
    ]

    records: tuple[
        FairSweepRecord,
        ...,
    ]

    summaries: tuple[
        FairSweepSummary,
        ...,
    ]


def _validate_weights(
    movement_weights: tuple[
        float,
        ...,
    ],
) -> None:
    """Validate requested movement weights."""

    if len(
        movement_weights
    ) == 0:
        raise ValueError(
            "movement_weights must not be empty."
        )

    if not all(
        np.isfinite(weight)
        for weight in movement_weights
    ):
        raise ValueError(
            "movement_weights must all be finite."
        )

    if any(
        weight < 0.0
        for weight in movement_weights
    ):
        raise ValueError(
            "movement_weights must be non-negative."
        )


def _candidate_index(
    candidates,
    selected_candidate,
) -> int:
    """Return candidate index using object identity.

    Candidate objects contain NumPy arrays, so equality-based membership is
    deliberately avoided.
    """

    for index, candidate in enumerate(
        candidates
    ):
        if selected_candidate is candidate:
            return index

    raise ValueError(
        "Selected candidate was not one of the supplied candidates."
    )


def run_fair_sweep_records(
    *,
    scenarios: tuple[
        RobustnessScenario,
        ...,
    ] | None = None,
    movement_weights: tuple[
        float,
        ...,
    ] = DEFAULT_MOVEMENT_WEIGHTS,
    perception_weight: float = 1.0,
    alignment_weight: float = 1.0,
) -> tuple[
    FairSweepRecord,
    ...,
]:
    """Run Generic and Task-Aware selection across scenarios and weights."""

    if scenarios is None:
        scenarios = (
            default_scenarios()
        )

    if len(
        scenarios
    ) == 0:
        raise ValueError(
            "At least one scenario is required."
        )

    _validate_weights(
        movement_weights
    )

    if (
        perception_weight < 0.0
        or alignment_weight < 0.0
    ):
        raise ValueError(
            "Scoring weights must be non-negative."
        )

    records: list[
        FairSweepRecord
    ] = []

    for scenario in scenarios:
        inputs = (
            build_scenario_inputs(
                scenario
            )
        )

        for movement_weight in (
            movement_weights
        ):
            scorer = (
                FairViewpointScorer(
                    observation_model=(
                        inputs.observation_model
                    ),
                    task=inputs.task,
                    config=(
                        FairScoringConfig(
                            perception_weight=(
                                perception_weight
                            ),
                            movement_weight=(
                                movement_weight
                            ),
                            alignment_weight=(
                                alignment_weight
                            ),
                        )
                    ),
                )
            )

            generic = (
                scorer.select_generic(
                    current_pose=(
                        inputs.initial_pose
                    ),
                    candidates=(
                        inputs.candidates
                    ),
                    target=(
                        inputs.target
                    ),
                    occluders=(
                        inputs.occluders
                    ),
                )
            )

            task = (
                scorer.select_task_aware(
                    current_pose=(
                        inputs.initial_pose
                    ),
                    candidates=(
                        inputs.candidates
                    ),
                    target=(
                        inputs.target
                    ),
                    occluders=(
                        inputs.occluders
                    ),
                )
            )

            generic_selected = (
                generic.selected
            )

            task_selected = (
                task.selected
            )

            generic_index = (
                _candidate_index(
                    inputs.candidates,
                    generic_selected.candidate,
                )
            )

            task_index = (
                _candidate_index(
                    inputs.candidates,
                    task_selected.candidate,
                )
            )

            records.append(
                FairSweepRecord(
                    scenario_id=(
                        scenario.scenario_id
                    ),
                    scenario_name=(
                        scenario.name
                    ),
                    movement_weight=float(
                        movement_weight
                    ),
                    generic_candidate_index=(
                        generic_index
                    ),
                    task_candidate_index=(
                        task_index
                    ),
                    generic_movement=float(
                        generic_selected
                        .movement_cost
                    ),
                    task_movement=float(
                        task_selected
                        .movement_cost
                    ),
                    generic_sigma=float(
                        generic_selected
                        .quality
                        .localisation_sigma
                    ),
                    task_sigma=float(
                        task_selected
                        .quality
                        .localisation_sigma
                    ),
                    generic_alignment=float(
                        generic_selected
                        .task_alignment
                    ),
                    task_alignment=float(
                        task_selected
                        .task_alignment
                    ),
                    generic_information=float(
                        generic_selected
                        .information_score
                    ),
                    task_information=float(
                        task_selected
                        .information_score
                    ),
                    generic_normalised_movement=float(
                        generic_selected
                        .normalised_movement_cost
                    ),
                    task_normalised_movement=float(
                        task_selected
                        .normalised_movement_cost
                    ),
                    generic_score=float(
                        generic_selected
                        .generic_score
                    ),
                    task_score=float(
                        task_selected
                        .task_aware_score
                    ),
                    different_candidate=bool(
                        generic_index
                        != task_index
                    ),
                )
            )

    return tuple(
        records
    )


def analyse_fair_sweep(
    *,
    records: tuple[
        FairSweepRecord,
        ...,
    ],
    movement_weights: tuple[
        float,
        ...,
    ],
) -> tuple[
    FairSweepSummary,
    ...,
]:
    """Aggregate comparison metrics by movement weight."""

    if len(
        records
    ) == 0:
        raise ValueError(
            "records must not be empty."
        )

    summaries: list[
        FairSweepSummary
    ] = []

    for movement_weight in (
        movement_weights
    ):
        selected = tuple(
            record
            for record in records
            if np.isclose(
                record.movement_weight,
                movement_weight,
            )
        )

        if len(
            selected
        ) == 0:
            raise ValueError(
                "No records found for movement weight "
                f"{movement_weight}."
            )

        generic_movement = np.asarray(
            [
                record.generic_movement
                for record in selected
            ],
            dtype=float,
        )

        task_movement = np.asarray(
            [
                record.task_movement
                for record in selected
            ],
            dtype=float,
        )

        generic_sigma = np.asarray(
            [
                record.generic_sigma
                for record in selected
            ],
            dtype=float,
        )

        task_sigma = np.asarray(
            [
                record.task_sigma
                for record in selected
            ],
            dtype=float,
        )

        generic_alignment = np.asarray(
            [
                record.generic_alignment
                for record in selected
            ],
            dtype=float,
        )

        task_alignment = np.asarray(
            [
                record.task_alignment
                for record in selected
            ],
            dtype=float,
        )

        different = np.asarray(
            [
                float(
                    record.different_candidate
                )
                for record in selected
            ],
            dtype=float,
        )

        summaries.append(
            FairSweepSummary(
                movement_weight=float(
                    movement_weight
                ),
                scenario_count=len(
                    selected
                ),
                mean_generic_movement=float(
                    np.mean(
                        generic_movement
                    )
                ),
                mean_task_movement=float(
                    np.mean(
                        task_movement
                    )
                ),
                mean_movement_difference=float(
                    np.mean(
                        task_movement
                        - generic_movement
                    )
                ),
                mean_generic_sigma=float(
                    np.mean(
                        generic_sigma
                    )
                ),
                mean_task_sigma=float(
                    np.mean(
                        task_sigma
                    )
                ),
                mean_sigma_difference=float(
                    np.mean(
                        task_sigma
                        - generic_sigma
                    )
                ),
                mean_generic_alignment=float(
                    np.mean(
                        generic_alignment
                    )
                ),
                mean_task_alignment=float(
                    np.mean(
                        task_alignment
                    )
                ),
                mean_alignment_difference=float(
                    np.mean(
                        task_alignment
                        - generic_alignment
                    )
                ),
                different_candidate_rate=float(
                    np.mean(
                        different
                    )
                ),
            )
        )

    return tuple(
        summaries
    )


def run_fair_scoring_sweep(
    *,
    scenarios: tuple[
        RobustnessScenario,
        ...,
    ] | None = None,
    movement_weights: tuple[
        float,
        ...,
    ] = DEFAULT_MOVEMENT_WEIGHTS,
) -> FairSweepResult:
    """Run the complete fair-scoring development diagnostic."""

    if scenarios is None:
        scenarios = (
            default_scenarios()
        )

    records = (
        run_fair_sweep_records(
            scenarios=scenarios,
            movement_weights=(
                movement_weights
            ),
        )
    )

    summaries = (
        analyse_fair_sweep(
            records=records,
            movement_weights=(
                movement_weights
            ),
        )
    )

    return FairSweepResult(
        movement_weights=(
            movement_weights
        ),
        scenarios=scenarios,
        records=records,
        summaries=summaries,
    )


def save_outputs(
    result: FairSweepResult,
    *,
    output_directory: str
    | Path = (
        "results/"
        "phase1_fair_scoring_sweep"
    ),
) -> tuple[
    Path,
    Path,
]:
    """Save raw CSV and aggregate JSON outputs."""

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        output_directory
        / "fair_scoring_sweep.csv"
    )

    summary_path = (
        output_directory
        / "fair_scoring_sweep_summary.json"
    )

    if len(
        result.records
    ) == 0:
        raise ValueError(
            "Cannot save an empty result."
        )

    with raw_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                asdict(
                    result.records[
                        0
                    ]
                ).keys()
            ),
        )

        writer.writeheader()

        for record in (
            result.records
        ):
            writer.writerow(
                asdict(
                    record
                )
            )

    payload = {
        "purpose": (
            "Development-stage comparison of fair "
            "Generic and Task-Aware scoring."
        ),
        "held_out_final_test": False,
        "movement_weights": list(
            result.movement_weights
        ),
        "scenario_count": len(
            result.scenarios
        ),
        "summaries": [
            asdict(
                summary
            )
            for summary in result.summaries
        ],
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
        )

    return (
        raw_path,
        summary_path,
    )


def print_summary(
    result: FairSweepResult,
) -> None:
    """Print aggregate sweep results."""

    print()

    print(
        "Fair Generic vs Task-Aware Scoring Sweep"
    )

    print(
        "========================================"
    )

    print(
        f"Scenarios: "
        f"{len(result.scenarios)}"
    )

    print()

    print(
        "Weight | Generic move | Task move | "
        "Generic align | Task align | Different"
    )

    print(
        "-" * 88
    )

    for summary in (
        result.summaries
    ):
        print(
            f"{summary.movement_weight:6.3f} | "
            f"{summary.mean_generic_movement * 1000.0:11.3f} mm | "
            f"{summary.mean_task_movement * 1000.0:9.3f} mm | "
            f"{summary.mean_generic_alignment:13.4f} | "
            f"{summary.mean_task_alignment:10.4f} | "
            f"{summary.different_candidate_rate * 100.0:8.1f}%"
        )

    print()

    print(
        "Weight | Generic sigma | Task sigma | "
        "Delta move | Delta alignment"
    )

    print(
        "-" * 76
    )

    for summary in (
        result.summaries
    ):
        print(
            f"{summary.movement_weight:6.3f} | "
            f"{summary.mean_generic_sigma * 1000.0:11.3f} mm | "
            f"{summary.mean_task_sigma * 1000.0:9.3f} mm | "
            f"{summary.mean_movement_difference * 1000.0:10.3f} mm | "
            f"{summary.mean_alignment_difference:15.4f}"
        )


def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the fair Generic versus "
            "Task-Aware movement-weight sweep."
        )
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "results/"
            "phase1_fair_scoring_sweep"
        ),
    )

    args = parser.parse_args()

    result = (
        run_fair_scoring_sweep()
    )

    print_summary(
        result
    )

    (
        raw_path,
        summary_path,
    ) = save_outputs(
        result,
        output_directory=args.output,
    )

    print()

    print(
        "Raw results saved to: "
        f"{raw_path}"
    )

    print(
        "Summary saved to: "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()