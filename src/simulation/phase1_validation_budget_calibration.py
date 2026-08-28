"""Validation-only calibration of the Generic camera-movement budget.

The development-selected Generic movement weight of 0.170 failed to preserve
the intended movement budget on the frozen validation split.

Development result:
    Generic movement weight: 0.170

Frozen Task-Aware movement weight:
    0.200

The purpose of this module is deliberately narrow:

    choose a Generic movement weight whose mean validation camera movement
    matches the frozen Task-Aware camera movement budget.

No localisation error, predicted uncertainty, planning result, collision
result or safe-navigation outcome enters the calibration objective.

The held-out split is never executed by this module.

After a satisfactory validation match is obtained, the selected Generic
weight is frozen before final held-out testing.
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

from src.perception.fair_scene_strategies import (
    GenericSceneScoringConfig,
    GenericSceneViewpointScorer,
    TaskAwareSceneScoringConfig,
    TaskAwareSceneViewpointScorer,
)
from src.simulation.phase1_validation_gate import (
    DEFAULT_REPETITIONS,
    FROZEN_MANIFEST_SHA256,
    FROZEN_TASK_MOVEMENT_WEIGHT,
    INITIAL_SEED_BASE,
    load_frozen_validation_scenarios,
)
from src.simulation.three_strategy_perception import (
    run_fixed_perception,
)
from src.simulation.three_strategy_robustness_benchmark import (
    RobustnessScenario,
    build_scenario_inputs,
)


DEVELOPMENT_GENERIC_MOVEMENT_WEIGHT = 0.170

FINAL_MATCH_TOLERANCE = 0.05


DEFAULT_GENERIC_WEIGHT_GRID = tuple(
    float(
        np.round(
            value,
            3,
        )
    )
    for value in np.arange(
        0.000,
        1.000 + 0.0001,
        0.005,
    )
)


@dataclass(frozen=True)
class ValidationBudgetTrial:
    """Task movement and Generic primitive scores for one validation trial."""

    scenario_id: int
    scenario_name: str

    repetition: int

    initial_seed: int

    task_camera_movement: float

    generic_scene_information: tuple[
        float,
        ...,
    ]

    generic_normalised_movement: tuple[
        float,
        ...,
    ]

    generic_camera_movements: tuple[
        float,
        ...,
    ]


@dataclass(frozen=True)
class GenericWeightCandidate:
    """Movement-budget result for one Generic movement weight."""

    movement_weight: float

    mean_generic_camera_movement: float

    target_task_camera_movement: float

    absolute_budget_gap: float

    relative_budget_gap: float


@dataclass(frozen=True)
class ValidationBudgetCalibrationResult:
    """Complete validation-only movement calibration."""

    manifest_sha256: str

    task_movement_weight: float

    development_generic_weight: float

    validation_trial_count: int

    target_task_camera_movement: float

    candidates: tuple[
        GenericWeightCandidate,
        ...,
    ]

    selected_generic: GenericWeightCandidate

    match_tolerance: float

    ready_to_freeze: bool


def _validate_weights(
    weights: tuple[
        float,
        ...,
    ],
) -> None:
    """Validate a Generic weight search grid."""

    if len(
        weights
    ) == 0:
        raise ValueError(
            "weights must not be empty."
        )

    if not all(
        np.isfinite(
            weight
        )
        for weight in weights
    ):
        raise ValueError(
            "weights must all be finite."
        )

    if any(
        weight < 0.0
        for weight in weights
    ):
        raise ValueError(
            "weights must be non-negative."
        )


def collect_validation_budget_trials(
    *,
    scenarios: tuple[
        RobustnessScenario,
        ...,
    ],
    repetitions: int = (
        DEFAULT_REPETITIONS
    ),
) -> tuple[
    ValidationBudgetTrial,
    ...,
]:
    """Collect task-independent Generic primitives on validation scenes.

    The Task-Aware movement budget is calculated using its already frozen
    movement weight of 0.200.

    Generic candidate primitives are computed once per trial. Arbitrarily
    many Generic movement weights can then be evaluated algebraically without
    resampling perception.
    """

    if len(
        scenarios
    ) == 0:
        raise ValueError(
            "scenarios must not be empty."
        )

    if repetitions <= 0:
        raise ValueError(
            "repetitions must be positive."
        )

    trials: list[
        ValidationBudgetTrial
    ] = []

    for scenario in scenarios:
        inputs = (
            build_scenario_inputs(
                scenario
            )
        )

        for repetition in range(
            repetitions
        ):
            initial_seed = (
                INITIAL_SEED_BASE
                + scenario.scenario_id
                * 100
                + repetition
            )

            shared_initial = (
                run_fixed_perception(
                    observation_model=(
                        inputs
                        .observation_model
                    ),
                    initial_pose=(
                        inputs.initial_pose
                    ),
                    true_structures=(
                        inputs.true_structures
                    ),
                    seed=int(
                        initial_seed
                    ),
                    occluders=(
                        inputs.occluders
                    ),
                )
            )

            initial_perception = (
                shared_initial
                .perception_result
            )

            generic = (
                GenericSceneViewpointScorer(
                    observation_model=(
                        inputs
                        .observation_model
                    ),
                    config=(
                        GenericSceneScoringConfig(
                            perception_weight=1.0,
                            movement_weight=0.0,
                        )
                    ),
                )
            )

            generic_scores = (
                generic.score_candidates(
                    current_pose=(
                        inputs.initial_pose
                    ),
                    candidates=(
                        inputs.candidates
                    ),
                    initial_perception=(
                        initial_perception
                    ),
                )
            )

            task = (
                TaskAwareSceneViewpointScorer(
                    observation_model=(
                        inputs
                        .observation_model
                    ),
                    task_trajectory=(
                        inputs
                        .task
                        .trajectory
                    ),
                    config=(
                        TaskAwareSceneScoringConfig(
                            perception_weight=1.0,
                            movement_weight=(
                                FROZEN_TASK_MOVEMENT_WEIGHT
                            ),
                            alignment_weight=1.0,
                        )
                    ),
                )
            )

            task_selection = (
                task.select_viewpoint(
                    current_pose=(
                        inputs.initial_pose
                    ),
                    candidates=(
                        inputs.candidates
                    ),
                    initial_perception=(
                        initial_perception
                    ),
                )
            )

            trials.append(
                ValidationBudgetTrial(
                    scenario_id=int(
                        scenario.scenario_id
                    ),
                    scenario_name=(
                        scenario.name
                    ),
                    repetition=int(
                        repetition
                    ),
                    initial_seed=int(
                        initial_seed
                    ),
                    task_camera_movement=float(
                        task_selection
                        .selected
                        .movement_cost
                    ),
                    generic_scene_information=tuple(
                        float(
                            score
                            .scene_information
                        )
                        for score
                        in generic_scores
                    ),
                    generic_normalised_movement=tuple(
                        float(
                            score
                            .normalised_movement_cost
                        )
                        for score
                        in generic_scores
                    ),
                    generic_camera_movements=tuple(
                        float(
                            score
                            .movement_cost
                        )
                        for score
                        in generic_scores
                    ),
                )
            )

    return tuple(
        trials
    )


def evaluate_generic_weight_grid(
    *,
    trials: tuple[
        ValidationBudgetTrial,
        ...,
    ],
    weights: tuple[
        float,
        ...,
    ] = DEFAULT_GENERIC_WEIGHT_GRID,
) -> tuple[
    GenericWeightCandidate,
    ...,
]:
    """Evaluate Generic movement weights using camera movement only."""

    if len(
        trials
    ) == 0:
        raise ValueError(
            "trials must not be empty."
        )

    _validate_weights(
        weights
    )

    target_task_movement = float(
        np.mean(
            [
                trial.task_camera_movement
                for trial
                in trials
            ]
        )
    )

    if target_task_movement <= 0.0:
        raise ValueError(
            "Mean Task-Aware movement must be positive."
        )

    candidates = []

    for movement_weight in (
        weights
    ):
        selected_movements = []

        for trial in trials:
            information = np.asarray(
                trial
                .generic_scene_information,
                dtype=float,
            )

            normalised_movement = (
                np.asarray(
                    trial
                    .generic_normalised_movement,
                    dtype=float,
                )
            )

            actual_movement = (
                np.asarray(
                    trial
                    .generic_camera_movements,
                    dtype=float,
                )
            )

            if not (
                information.shape
                == normalised_movement.shape
                == actual_movement.shape
            ):
                raise ValueError(
                    "Generic candidate arrays must "
                    "have identical shapes."
                )

            if information.size == 0:
                raise ValueError(
                    "Each trial must contain candidates."
                )

            generic_scores = (
                information
                - float(
                    movement_weight
                )
                * normalised_movement
            )

            selected_index = int(
                np.argmax(
                    generic_scores
                )
            )

            selected_movements.append(
                float(
                    actual_movement[
                        selected_index
                    ]
                )
            )

        mean_generic_movement = float(
            np.mean(
                selected_movements
            )
        )

        absolute_gap = abs(
            mean_generic_movement
            - target_task_movement
        )

        relative_gap = (
            absolute_gap
            / target_task_movement
        )

        candidates.append(
            GenericWeightCandidate(
                movement_weight=float(
                    movement_weight
                ),
                mean_generic_camera_movement=(
                    mean_generic_movement
                ),
                target_task_camera_movement=(
                    target_task_movement
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


def select_generic_weight(
    candidates: tuple[
        GenericWeightCandidate,
        ...,
    ],
) -> GenericWeightCandidate:
    """Select the closest Generic movement-budget match.

    The only optimisation target is absolute mean camera-movement mismatch.
    The lower movement weight wins exact ties.
    """

    if len(
        candidates
    ) == 0:
        raise ValueError(
            "candidates must not be empty."
        )

    return min(
        candidates,
        key=lambda candidate: (
            candidate.absolute_budget_gap,
            candidate.movement_weight,
        ),
    )


def run_validation_budget_calibration(
    *,
    manifest_path: str
    | Path = (
        "results/"
        "phase1_scenario_splits/"
        "scenario_manifest.json"
    ),
    repetitions: int = (
        DEFAULT_REPETITIONS
    ),
    weights: tuple[
        float,
        ...,
    ] = DEFAULT_GENERIC_WEIGHT_GRID,
) -> ValidationBudgetCalibrationResult:
    """Run validation-only movement-budget calibration."""

    scenarios = (
        load_frozen_validation_scenarios(
            manifest_path
        )
    )

    trials = (
        collect_validation_budget_trials(
            scenarios=scenarios,
            repetitions=repetitions,
        )
    )

    candidates = (
        evaluate_generic_weight_grid(
            trials=trials,
            weights=weights,
        )
    )

    selected = (
        select_generic_weight(
            candidates
        )
    )

    target = float(
        selected
        .target_task_camera_movement
    )

    return ValidationBudgetCalibrationResult(
        manifest_sha256=(
            FROZEN_MANIFEST_SHA256
        ),
        task_movement_weight=float(
            FROZEN_TASK_MOVEMENT_WEIGHT
        ),
        development_generic_weight=float(
            DEVELOPMENT_GENERIC_MOVEMENT_WEIGHT
        ),
        validation_trial_count=len(
            trials
        ),
        target_task_camera_movement=(
            target
        ),
        candidates=candidates,
        selected_generic=selected,
        match_tolerance=float(
            FINAL_MATCH_TOLERANCE
        ),
        ready_to_freeze=bool(
            selected
            .relative_budget_gap
            <= FINAL_MATCH_TOLERANCE
        ),
    )


def save_result(
    result: ValidationBudgetCalibrationResult,
    *,
    output_path: str
    | Path = (
        "results/"
        "phase1_validation/"
        "validation_budget_calibration.json"
    ),
) -> Path:
    """Save audit evidence for validation-only calibration."""

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "experiment_role": (
            "validation_only_budget_calibration"
        ),
        "held_out_executed": False,
        "manifest_sha256": (
            result.manifest_sha256
        ),
        "selection_information": (
            "camera_movement_only"
        ),
        "forbidden_for_weight_selection": [
            "localisation_error",
            "predicted_sigma",
            "planning_success",
            "safe_navigation_success",
            "collision",
            "safety_violation",
        ],
        "task_movement_weight": (
            result.task_movement_weight
        ),
        "development_generic_weight": (
            result
            .development_generic_weight
        ),
        "validation_trial_count": (
            result.validation_trial_count
        ),
        "target_task_camera_movement": (
            result
            .target_task_camera_movement
        ),
        "match_tolerance": (
            result.match_tolerance
        ),
        "selected_generic": asdict(
            result.selected_generic
        ),
        "ready_to_freeze": (
            result.ready_to_freeze
        ),
        "candidates": [
            asdict(
                candidate
            )
            for candidate
            in result.candidates
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
    result: ValidationBudgetCalibrationResult,
) -> None:
    """Print validation calibration result."""

    selected = (
        result.selected_generic
    )

    development_candidate = next(
        (
            candidate
            for candidate
            in result.candidates
            if np.isclose(
                candidate.movement_weight,
                DEVELOPMENT_GENERIC_MOVEMENT_WEIGHT,
            )
        ),
        None,
    )

    print()

    print(
        "Phase 1 Validation-Only Budget Calibration"
    )

    print(
        "=========================================="
    )

    print()

    print(
        f"Manifest SHA-256:"
    )

    print(
        f"  {result.manifest_sha256}"
    )

    print()

    print(
        f"Validation trials:                 "
        f"{result.validation_trial_count}"
    )

    print(
        f"Frozen Task-Aware weight:          "
        f"{result.task_movement_weight:.3f}"
    )

    print(
        f"Task-Aware target movement:        "
        f"{result.target_task_camera_movement * 1000.0:.3f} mm"
    )

    if development_candidate is not None:
        print()

        print(
            "Previous development-selected Generic"
        )

        print(
            f"  Weight:                          "
            f"{development_candidate.movement_weight:.3f}"
        )

        print(
            f"  Validation movement:             "
            f"{development_candidate.mean_generic_camera_movement * 1000.0:.3f} mm"
        )

        print(
            f"  Validation mismatch:             "
            f"{development_candidate.relative_budget_gap * 100.0:.2f}%"
        )

    print()

    print(
        "Best validation-calibrated Generic"
    )

    print(
        f"  Weight:                          "
        f"{selected.movement_weight:.3f}"
    )

    print(
        f"  Mean movement:                   "
        f"{selected.mean_generic_camera_movement * 1000.0:.3f} mm"
    )

    print(
        f"  Absolute movement gap:           "
        f"{selected.absolute_budget_gap * 1000.0:.3f} mm"
    )

    print(
        f"  Relative movement gap:           "
        f"{selected.relative_budget_gap * 100.0:.2f}%"
    )

    print(
        f"  Freeze threshold:                "
        f"{result.match_tolerance * 100.0:.1f}%"
    )

    print(
        f"  Ready to freeze:                 "
        f"{result.ready_to_freeze}"
    )


def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(
        description=(
            "Calibrate Generic camera movement against "
            "the frozen Task-Aware validation budget."
        )
    )

    parser.add_argument(
        "--manifest",
        type=str,
        default=(
            "results/"
            "phase1_scenario_splits/"
            "scenario_manifest.json"
        ),
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=(
            DEFAULT_REPETITIONS
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "results/"
            "phase1_validation/"
            "validation_budget_calibration.json"
        ),
    )

    args = parser.parse_args()

    result = (
        run_validation_budget_calibration(
            manifest_path=(
                args.manifest
            ),
            repetitions=(
                args.repetitions
            ),
        )
    )

    print_result(
        result
    )

    path = save_result(
        result,
        output_path=(
            args.output
        ),
    )

    print()

    print(
        f"Calibration evidence saved to: "
        f"{path}"
    )


if __name__ == "__main__":
    main()