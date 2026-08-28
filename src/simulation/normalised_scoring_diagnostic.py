"""Diagnostic sweep for the normalised task-aware viewpoint scorer.

This module examines how the scale-controlled task-aware scorer behaves
across the ten established robustness scenarios as the camera-movement
weight is varied.

The goal is not to optimise performance on the final test benchmark.
Instead, this is a development-stage diagnostic used to verify that:

1. movement cost has meaningful numerical influence;
2. increasing movement penalty reduces selected camera displacement;
3. useful task-aware viewpoint behaviour is not destroyed immediately;
4. candidate selection is stable and interpretable.

Final benchmark hyperparameters will later be frozen before held-out testing.
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

from src.perception.normalised_task_aware_scoring import (
    NormalisedTaskAwareConfig,
    NormalisedTaskAwareViewpointScorer,
)
from src.perception.viewpoint_scoring import (
    GenericViewpointScorer,
    select_best_viewpoint,
)
from src.simulation.three_strategy_robustness_benchmark import (
    RobustnessScenario,
    build_scenario_inputs,
    default_scenarios,
)


DEFAULT_MOVEMENT_WEIGHTS = (
    0.0,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
)


@dataclass(frozen=True)
class DiagnosticRecord:
    """One scenario/weight viewpoint-selection result."""

    scenario_id: int

    scenario_name: str

    movement_weight: float

    selected_candidate_index: int

    camera_movement: float

    predicted_sigma: float

    task_relevance: float

    task_alignment: float

    normalised_generic_utility: float

    normalised_uncertainty_information: float

    normalised_movement_cost: float

    final_score: float

    generic_candidate_index: int

    generic_camera_movement: float

    generic_predicted_sigma: float

    differs_from_generic: bool


@dataclass(frozen=True)
class WeightSummary:
    """Aggregate diagnostic result for one movement weight."""

    movement_weight: float

    scenario_count: int

    mean_camera_movement: float

    median_camera_movement: float

    mean_predicted_sigma: float

    mean_task_alignment: float

    mean_final_score: float

    generic_difference_rate: float


@dataclass(frozen=True)
class DiagnosticResult:
    """Complete movement-weight diagnostic."""

    movement_weights: tuple[
        float,
        ...,
    ]

    scenarios: tuple[
        RobustnessScenario,
        ...,
    ]

    records: tuple[
        DiagnosticRecord,
        ...,
    ]

    summaries: tuple[
        WeightSummary,
        ...,
    ]


def _candidate_index(
    candidates,
    selected_candidate,
) -> int:
    """Return the index of a selected candidate."""

    for index, candidate in enumerate(
        candidates
    ):
        if candidate is selected_candidate:
            return index

    for index, candidate in enumerate(
        candidates
    ):
        if candidate == selected_candidate:
            return index

    raise ValueError(
        "Selected candidate was not found "
        "in the supplied candidate set."
    )


def _validate_weights(
    movement_weights: tuple[
        float,
        ...,
    ],
) -> None:
    """Validate diagnostic movement weights."""

    if len(
        movement_weights
    ) == 0:
        raise ValueError(
            "movement_weights must not be empty."
        )

    if not all(
        np.isfinite(
            weight
        )
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


def run_diagnostic_records(
    *,
    scenarios: tuple[
        RobustnessScenario,
        ...,
    ] | None = None,
    movement_weights: tuple[
        float,
        ...,
    ] = DEFAULT_MOVEMENT_WEIGHTS,
) -> tuple[
    DiagnosticRecord,
    ...,
]:
    """Run the normalised scoring sweep."""

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

    records: list[
        DiagnosticRecord
    ] = []

    for scenario in scenarios:
        inputs = (
            build_scenario_inputs(
                scenario
            )
        )

        generic_scorer = (
            GenericViewpointScorer(
                observation_model=(
                    inputs.observation_model
                )
            )
        )

        generic_scores = (
            generic_scorer
            .score_candidates(
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
            select_best_viewpoint(
                generic_scores
            )
        )

        generic_index = (
            _candidate_index(
                inputs.candidates,
                generic_selected.candidate,
            )
        )

        for movement_weight in (
            movement_weights
        ):
            scorer = (
                NormalisedTaskAwareViewpointScorer(
                    generic_scorer=(
                        GenericViewpointScorer(
                            observation_model=(
                                inputs
                                .observation_model
                            )
                        )
                    ),
                    task=inputs.task,
                    config=(
                        NormalisedTaskAwareConfig(
                            generic_weight=1.0,
                            alignment_weight=1.0,
                            uncertainty_weight=1.0,
                            movement_weight=(
                                movement_weight
                            ),
                        )
                    ),
                )
            )

            selected = (
                scorer.select_viewpoint(
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

            selected_index = (
                _candidate_index(
                    inputs.candidates,
                    selected.candidate,
                )
            )

            records.append(
                DiagnosticRecord(
                    scenario_id=(
                        scenario.scenario_id
                    ),
                    scenario_name=(
                        scenario.name
                    ),
                    movement_weight=float(
                        movement_weight
                    ),
                    selected_candidate_index=(
                        selected_index
                    ),
                    camera_movement=float(
                        selected.movement_cost
                    ),
                    predicted_sigma=float(
                        selected.predicted_sigma
                    ),
                    task_relevance=float(
                        selected.task_relevance
                    ),
                    task_alignment=float(
                        selected.task_alignment
                    ),
                    normalised_generic_utility=float(
                        selected
                        .normalised_generic_utility
                    ),
                    normalised_uncertainty_information=float(
                        selected
                        .normalised_uncertainty_information
                    ),
                    normalised_movement_cost=float(
                        selected
                        .normalised_movement_cost
                    ),
                    final_score=float(
                        selected.final_score
                    ),
                    generic_candidate_index=(
                        generic_index
                    ),
                    generic_camera_movement=float(
                        generic_selected
                        .movement_cost
                    ),
                    generic_predicted_sigma=float(
                        generic_selected
                        .quality
                        .localisation_sigma
                    ),
                    differs_from_generic=bool(
                        selected_index
                        != generic_index
                    ),
                )
            )

    return tuple(
        records
    )


def analyse_records(
    *,
    records: tuple[
        DiagnosticRecord,
        ...,
    ],
    movement_weights: tuple[
        float,
        ...,
    ],
) -> tuple[
    WeightSummary,
    ...,
]:
    """Aggregate diagnostic results by movement weight."""

    if len(
        records
    ) == 0:
        raise ValueError(
            "records must not be empty."
        )

    summaries: list[
        WeightSummary
    ] = []

    for movement_weight in (
        movement_weights
    ):
        selected = [
            record
            for record in records
            if np.isclose(
                record.movement_weight,
                movement_weight,
            )
        ]

        if len(
            selected
        ) == 0:
            raise ValueError(
                "Missing records for movement weight "
                f"{movement_weight}."
            )

        movement = np.asarray(
            [
                record.camera_movement
                for record in selected
            ],
            dtype=float,
        )

        sigma = np.asarray(
            [
                record.predicted_sigma
                for record in selected
            ],
            dtype=float,
        )

        alignment = np.asarray(
            [
                record.task_alignment
                for record in selected
            ],
            dtype=float,
        )

        scores = np.asarray(
            [
                record.final_score
                for record in selected
            ],
            dtype=float,
        )

        differs = np.asarray(
            [
                float(
                    record.differs_from_generic
                )
                for record in selected
            ],
            dtype=float,
        )

        summaries.append(
            WeightSummary(
                movement_weight=float(
                    movement_weight
                ),
                scenario_count=len(
                    selected
                ),
                mean_camera_movement=float(
                    np.mean(
                        movement
                    )
                ),
                median_camera_movement=float(
                    np.median(
                        movement
                    )
                ),
                mean_predicted_sigma=float(
                    np.mean(
                        sigma
                    )
                ),
                mean_task_alignment=float(
                    np.mean(
                        alignment
                    )
                ),
                mean_final_score=float(
                    np.mean(
                        scores
                    )
                ),
                generic_difference_rate=float(
                    np.mean(
                        differs
                    )
                ),
            )
        )

    return tuple(
        summaries
    )


def run_normalised_scoring_diagnostic(
    *,
    scenarios: tuple[
        RobustnessScenario,
        ...,
    ] | None = None,
    movement_weights: tuple[
        float,
        ...,
    ] = DEFAULT_MOVEMENT_WEIGHTS,
) -> DiagnosticResult:
    """Run and analyse the movement-weight sweep."""

    if scenarios is None:
        scenarios = (
            default_scenarios()
        )

    _validate_weights(
        movement_weights
    )

    records = (
        run_diagnostic_records(
            scenarios=scenarios,
            movement_weights=(
                movement_weights
            ),
        )
    )

    summaries = (
        analyse_records(
            records=records,
            movement_weights=(
                movement_weights
            ),
        )
    )

    return DiagnosticResult(
        movement_weights=(
            movement_weights
        ),
        scenarios=scenarios,
        records=records,
        summaries=summaries,
    )


def save_outputs(
    result: DiagnosticResult,
    *,
    output_directory: str
    | Path = (
        "results/"
        "phase1_normalised_scoring"
    ),
) -> tuple[
    Path,
    Path,
]:
    """Save raw and aggregate diagnostic outputs."""

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        output_directory
        / "movement_weight_diagnostic.csv"
    )

    summary_path = (
        output_directory
        / "movement_weight_summary.json"
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
            "Development-stage diagnostic of "
            "normalised task-aware movement weighting."
        ),
        "final_test_tuning": False,
        "movement_weights": list(
            result.movement_weights
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
    result: DiagnosticResult,
) -> None:
    """Print the movement-weight diagnostic."""

    print()

    print(
        "Normalised Task-Aware Scoring Diagnostic"
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
        "Weight | Mean move | Mean sigma | "
        "Alignment | Diff. from Generic"
    )

    print(
        "-" * 72
    )

    for summary in (
        result.summaries
    ):
        print(
            f"{summary.movement_weight:6.2f} | "
            f"{summary.mean_camera_movement * 1000.0:9.3f} mm | "
            f"{summary.mean_predicted_sigma * 1000.0:9.3f} mm | "
            f"{summary.mean_task_alignment:9.4f} | "
            f"{summary.generic_difference_rate * 100.0:8.1f}%"
        )


def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the development-stage normalised "
            "task-aware scoring diagnostic."
        )
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "results/"
            "phase1_normalised_scoring"
        ),
    )

    args = parser.parse_args()

    result = (
        run_normalised_scoring_diagnostic()
    )

    print_summary(
        result
    )

    (
        raw_path,
        summary_path,
    ) = save_outputs(
        result,
        output_directory=(
            args.output
        ),
    )

    print()

    print(
        f"Raw diagnostics saved to: "
        f"{raw_path}"
    )

    print(
        f"Summary saved to: "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()