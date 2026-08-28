"""Development-stage movement-budget matching for corrected Phase 1.

This module performs two strictly development-only operations:

1. Select the Task-Aware operating point using the pre-specified
   camera-movement versus predicted-uncertainty trade-off.

2. Independently tune the Generic movement weight so that Generic uses
   approximately the same mean camera movement as the selected Task-Aware
   operating point.

The underlying development runner uses the corrected information-isolation
pipeline:

- scenario-independent initial camera prior;
- shared noisy initial anatomical observation;
- active candidates centred on the estimated target;
- no simulator truth exposed to Generic or Task-Aware decision logic.

Realised localisation error and downstream planning outcomes do not enter
either hyperparameter-selection criterion.
"""

from __future__ import annotations

import argparse
import json

from dataclasses import (
    asdict,
    dataclass,
)
from pathlib import Path

import numpy as np

from src.simulation.fair_scene_development_sweep import (
    FairSceneDevelopmentSummary,
    analyse_development_records,
    run_development_records,
    run_fair_scene_development_sweep,
)
from src.simulation.three_strategy_robustness_benchmark import (
    RobustnessScenario,
    default_scenarios,
)


# The corrected candidate-generation pipeline shifted the Generic movement
# trade-off substantially relative to the historical truth-centred pipeline.
#
# This development-only grid deliberately brackets the new coarse operating
# region around approximately 0.075.
DEFAULT_GENERIC_MATCH_WEIGHTS = tuple(
    float(
        np.round(
            value,
            3,
        )
    )
    for value in np.arange(
        0.040,
        0.140 + 0.0001,
        0.002,
    )
)


@dataclass(frozen=True)
class TaskOperatingPoint:
    """Selected Task-Aware development operating point."""

    movement_weight: float

    mean_camera_movement: float
    mean_predicted_sigma: float

    normalised_camera_movement: float
    normalised_predicted_sigma: float

    utopia_distance: float


@dataclass(frozen=True)
class GenericBudgetCandidate:
    """One Generic development movement-budget candidate."""

    movement_weight: float

    mean_camera_movement: float

    absolute_budget_gap: float
    relative_budget_gap: float


@dataclass(frozen=True)
class MovementBudgetMatchResult:
    """Complete corrected development movement-budget result."""

    task_operating_point: TaskOperatingPoint

    generic_candidates: tuple[
        GenericBudgetCandidate,
        ...,
    ]

    selected_generic: GenericBudgetCandidate

    budget_match_within_five_percent: bool


def _normalise_lower_is_better(
    values: np.ndarray,
) -> np.ndarray:
    """Min-max normalise a lower-is-better development quantity."""

    values = np.asarray(
        values,
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError(
            "values must be one-dimensional."
        )

    if values.size == 0:
        raise ValueError(
            "values must not be empty."
        )

    if not np.all(
        np.isfinite(
            values
        )
    ):
        raise ValueError(
            "values must be finite."
        )

    minimum = float(
        np.min(
            values
        )
    )

    maximum = float(
        np.max(
            values
        )
    )

    span = (
        maximum
        - minimum
    )

    if span <= 1e-12:
        return np.zeros_like(
            values,
            dtype=float,
        )

    return (
        values
        - minimum
    ) / span


def select_task_operating_point(
    summaries: tuple[
        FairSceneDevelopmentSummary,
        ...,
    ],
) -> TaskOperatingPoint:
    """Select the Task-Aware movement/uncertainty compromise.

    The pre-specified criterion uses only:

    - mean Task-Aware camera movement;
    - mean Task-Aware predicted localisation uncertainty.

    Both are lower-is-better quantities.

    Each quantity is independently min-max normalised over the development
    operating points. The selected operating point minimises Euclidean
    distance to the ideal point:

        movement = 0
        uncertainty = 0

    Realised localisation error and downstream navigation performance are
    deliberately excluded from this decision.
    """

    if len(
        summaries
    ) < 2:
        raise ValueError(
            "At least two development operating points are required."
        )

    movement = np.asarray(
        [
            summary
            .mean_task_camera_movement
            for summary
            in summaries
        ],
        dtype=float,
    )

    sigma = np.asarray(
        [
            summary
            .mean_task_predicted_sigma
            for summary
            in summaries
        ],
        dtype=float,
    )

    movement_normalised = (
        _normalise_lower_is_better(
            movement
        )
    )

    sigma_normalised = (
        _normalise_lower_is_better(
            sigma
        )
    )

    distances = np.sqrt(
        movement_normalised
        * movement_normalised
        + sigma_normalised
        * sigma_normalised
    )

    best_index = min(
        range(
            len(
                summaries
            )
        ),
        key=lambda index: (
            float(
                distances[
                    index
                ]
            ),
            float(
                movement[
                    index
                ]
            ),
            float(
                summaries[
                    index
                ]
                .movement_weight
            ),
        ),
    )

    summary = (
        summaries[
            best_index
        ]
    )

    return TaskOperatingPoint(
        movement_weight=float(
            summary
            .movement_weight
        ),
        mean_camera_movement=float(
            movement[
                best_index
            ]
        ),
        mean_predicted_sigma=float(
            sigma[
                best_index
            ]
        ),
        normalised_camera_movement=float(
            movement_normalised[
                best_index
            ]
        ),
        normalised_predicted_sigma=float(
            sigma_normalised[
                best_index
            ]
        ),
        utopia_distance=float(
            distances[
                best_index
            ]
        ),
    )


def select_generic_budget_match(
    candidates: tuple[
        GenericBudgetCandidate,
        ...,
    ],
) -> GenericBudgetCandidate:
    """Return the Generic candidate closest to the Task-Aware budget."""

    if len(
        candidates
    ) == 0:
        raise ValueError(
            "candidates must not be empty."
        )

    return min(
        candidates,
        key=lambda candidate: (
            candidate
            .absolute_budget_gap,
            candidate
            .movement_weight,
        ),
    )


def match_generic_movement_budget(
    *,
    target_camera_movement: float,
    scenarios: tuple[
        RobustnessScenario,
        ...,
    ] | None = None,
    repetitions: int = 5,
    generic_weights: tuple[
        float,
        ...,
    ] = DEFAULT_GENERIC_MATCH_WEIGHTS,
) -> tuple[
    GenericBudgetCandidate,
    ...,
]:
    """Evaluate Generic weights against a frozen Task-Aware movement budget."""

    if (
        not np.isfinite(
            target_camera_movement
        )
        or target_camera_movement <= 0.0
    ):
        raise ValueError(
            "target_camera_movement must be positive and finite."
        )

    if len(
        generic_weights
    ) == 0:
        raise ValueError(
            "generic_weights must not be empty."
        )

    if any(
        (
            weight < 0.0
            or not np.isfinite(
                weight
            )
        )
        for weight
        in generic_weights
    ):
        raise ValueError(
            "generic_weights must be finite and non-negative."
        )

    if scenarios is None:
        scenarios = (
            default_scenarios()
        )

    # This calls the corrected development runner. Generic and Task-Aware
    # candidate sets are therefore generated around the noisy estimated
    # target rather than the simulator's hidden true target.
    records = (
        run_development_records(
            scenarios=scenarios,
            movement_weights=(
                generic_weights
            ),
            repetitions=(
                repetitions
            ),
        )
    )

    summaries = (
        analyse_development_records(
            records=records,
            movement_weights=(
                generic_weights
            ),
        )
    )

    candidates: list[
        GenericBudgetCandidate
    ] = []

    for summary in summaries:
        movement = float(
            summary
            .mean_generic_camera_movement
        )

        absolute_gap = abs(
            movement
            - target_camera_movement
        )

        relative_gap = (
            absolute_gap
            / target_camera_movement
        )

        candidates.append(
            GenericBudgetCandidate(
                movement_weight=float(
                    summary
                    .movement_weight
                ),
                mean_camera_movement=(
                    movement
                ),
                absolute_budget_gap=float(
                    absolute_gap
                ),
                relative_budget_gap=float(
                    relative_gap
                ),
            )
        )

    return tuple(
        candidates
    )


def run_movement_budget_matching(
    *,
    scenarios: tuple[
        RobustnessScenario,
        ...,
    ] | None = None,
    repetitions: int = 5,
    generic_weights: tuple[
        float,
        ...,
    ] = DEFAULT_GENERIC_MATCH_WEIGHTS,
) -> MovementBudgetMatchResult:
    """Run corrected Task-Aware selection and Generic budget matching."""

    if scenarios is None:
        scenarios = (
            default_scenarios()
        )

    development = (
        run_fair_scene_development_sweep(
            scenarios=scenarios,
            repetitions=(
                repetitions
            ),
        )
    )

    task_operating_point = (
        select_task_operating_point(
            development
            .summaries
        )
    )

    generic_candidates = (
        match_generic_movement_budget(
            target_camera_movement=(
                task_operating_point
                .mean_camera_movement
            ),
            scenarios=scenarios,
            repetitions=(
                repetitions
            ),
            generic_weights=(
                generic_weights
            ),
        )
    )

    selected_generic = (
        select_generic_budget_match(
            generic_candidates
        )
    )

    return MovementBudgetMatchResult(
        task_operating_point=(
            task_operating_point
        ),
        generic_candidates=(
            generic_candidates
        ),
        selected_generic=(
            selected_generic
        ),
        budget_match_within_five_percent=bool(
            selected_generic
            .relative_budget_gap
            <= 0.05
        ),
    )


def save_result(
    result: MovementBudgetMatchResult,
    *,
    output_path: str
    | Path = (
        "results/"
        "phase1_movement_budget/"
        "movement_budget_match_corrected.json"
    ),
) -> Path:
    """Save corrected development-stage matching evidence."""

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "experiment_role": (
            "development_hyperparameter_selection"
        ),
        "held_out_test": False,
        "candidate_generation": (
            "active candidates centred on shared noisy estimated target"
        ),
        "initial_camera_geometry": (
            "scenario-independent nominal workspace prior"
        ),
        "ground_truth_decision_access": False,
        "selection_rule": (
            "Task-Aware operating point minimises Euclidean distance "
            "to the normalised low-camera-movement / "
            "low-predicted-uncertainty ideal point. "
            "Generic weight then minimises absolute mean "
            "camera-movement mismatch."
        ),
        "outcomes_used_for_task_weight_selection": [
            "camera_movement",
            "predicted_sigma",
        ],
        "outcomes_used_for_generic_weight_selection": [
            "camera_movement",
        ],
        "outcomes_not_used_for_selection": [
            "realised_localisation_error",
            "planning_success",
            "safe_navigation_success",
            "collision",
            "safety_violation",
        ],
        "task_operating_point": asdict(
            result
            .task_operating_point
        ),
        "selected_generic": asdict(
            result
            .selected_generic
        ),
        "budget_match_within_five_percent": (
            result
            .budget_match_within_five_percent
        ),
        "generic_candidates": [
            asdict(
                candidate
            )
            for candidate
            in result
            .generic_candidates
        ],
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            allow_nan=False,
        )

    return output_path


def print_result(
    result: MovementBudgetMatchResult,
) -> None:
    """Print corrected movement-budget matching result."""

    task = (
        result
        .task_operating_point
    )

    generic = (
        result
        .selected_generic
    )

    print()

    print(
        "Phase 1 Corrected Movement-Budget Matching"
    )

    print(
        "=========================================="
    )

    print()

    print(
        "Task-Aware development operating point"
    )

    print(
        f"  Movement weight:       "
        f"{task.movement_weight:.3f}"
    )

    print(
        f"  Mean camera movement:  "
        f"{task.mean_camera_movement * 1000.0:.3f} mm"
    )

    print(
        f"  Mean predicted sigma:  "
        f"{task.mean_predicted_sigma * 1000.0:.3f} mm"
    )

    print(
        f"  Normalised movement:   "
        f"{task.normalised_camera_movement:.4f}"
    )

    print(
        f"  Normalised sigma:      "
        f"{task.normalised_predicted_sigma:.4f}"
    )

    print(
        f"  Utopia distance:       "
        f"{task.utopia_distance:.4f}"
    )

    print()

    print(
        "Matched Generic development operating point"
    )

    print(
        f"  Movement weight:       "
        f"{generic.movement_weight:.3f}"
    )

    print(
        f"  Mean camera movement:  "
        f"{generic.mean_camera_movement * 1000.0:.3f} mm"
    )

    print(
        f"  Absolute budget gap:   "
        f"{generic.absolute_budget_gap * 1000.0:.3f} mm"
    )

    print(
        f"  Relative budget gap:   "
        f"{generic.relative_budget_gap * 100.0:.2f}%"
    )

    print()

    print(
        "Within 5% development movement budget: "
        f"{result.budget_match_within_five_percent}"
    )


def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(
        description=(
            "Select the corrected Task-Aware development operating "
            "point and movement-budget-match Generic."
        )
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "results/"
            "phase1_movement_budget/"
            "movement_budget_match_corrected.json"
        ),
    )

    args = parser.parse_args()

    result = (
        run_movement_budget_matching(
            repetitions=(
                args.repetitions
            )
        )
    )

    print_result(
        result
    )

    output = (
        save_result(
            result,
            output_path=(
                args.output
            ),
        )
    )

    print()

    print(
        f"Corrected matching evidence saved to: "
        f"{output}"
    )


if __name__ == "__main__":
    main()