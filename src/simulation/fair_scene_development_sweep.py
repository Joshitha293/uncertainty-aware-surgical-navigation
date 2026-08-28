"""Development sweep for the strengthened fair scene-wide comparison.

This experiment evaluates the Phase 1 fair active-perception formulation on
the ten established engineered DEVELOPMENT scenarios.

Important
---------
These scenarios have already been inspected during method development.
Results from this module may therefore be used for scorer design,
hyperparameter selection and movement-budget calibration, but they must not
be used for final held-out claims.

Strict information-isolation protocol
-------------------------------------
Each matched trial follows this sequence:

1. A scenario-independent nominal workspace prior defines the initial camera
   geometry.
2. Simulator truth is used only to generate one noisy initial observation.
3. The active-view candidate set is generated around the noisy ESTIMATED
   target position.
4. Generic and Task-Aware receive exactly the same:
       - noisy anatomical estimate;
       - candidate viewpoints;
       - observation model;
       - initial camera pose;
       - final observation seed.
5. Generic receives no task trajectory.
6. Task-Aware additionally receives the planned instrument trajectory.
7. Simulator truth is used again only to generate the final observation.

The active-view search space therefore does not encode the hidden true target
position.
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

from src.perception.fair_scene_strategies import (
    GenericSceneScoringConfig,
    GenericSceneViewpointScorer,
    TaskAwareSceneScoringConfig,
    TaskAwareSceneViewpointScorer,
)
from src.simulation.fair_scene_perception import (
    _observe_final_pose,
)
from src.simulation.phase1_fair_candidate_context import (
    build_fair_candidate_context,
)
from src.simulation.three_strategy_perception import (
    PerceptionStrategy,
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

DEFAULT_REPETITIONS = 5

INITIAL_SEED_BASE = 20261000
FINAL_SEED_BASE = 30261000


@dataclass(frozen=True)
class FairSceneDevelopmentRecord:
    """One matched Generic/Task-Aware development comparison."""

    scenario_id: int
    scenario_name: str

    repetition: int

    initial_seed: int
    final_seed: int

    movement_weight: float

    generic_candidate_index: int
    task_candidate_index: int

    different_candidate: bool

    generic_camera_movement: float
    task_camera_movement: float

    generic_localisation_error: float
    task_localisation_error: float

    generic_predicted_sigma: float
    task_predicted_sigma: float

    generic_scene_information: float
    task_generic_scene_information: float

    generic_task_scene_information: float
    task_task_scene_information: float

    generic_alignment: float
    task_alignment: float

    generic_score: float
    task_score: float


@dataclass(frozen=True)
class FairSceneDevelopmentSummary:
    """Aggregate development metrics for one movement weight."""

    movement_weight: float
    record_count: int

    mean_generic_camera_movement: float
    mean_task_camera_movement: float
    mean_camera_movement_difference: float

    mean_generic_localisation_error: float
    mean_task_localisation_error: float
    mean_localisation_error_difference: float

    mean_generic_predicted_sigma: float
    mean_task_predicted_sigma: float
    mean_predicted_sigma_difference: float

    mean_generic_alignment: float
    mean_task_alignment: float
    mean_alignment_difference: float

    mean_generic_scene_information: float
    mean_task_scene_information: float

    different_candidate_rate: float


@dataclass(frozen=True)
class FairSceneDevelopmentResult:
    """Complete development sweep."""

    movement_weights: tuple[
        float,
        ...,
    ]

    scenarios: tuple[
        RobustnessScenario,
        ...,
    ]

    repetitions: int

    records: tuple[
        FairSceneDevelopmentRecord,
        ...,
    ]

    summaries: tuple[
        FairSceneDevelopmentSummary,
        ...,
    ]


def _validate_configuration(
    *,
    movement_weights: tuple[
        float,
        ...,
    ],
    repetitions: int,
) -> None:
    """Validate development sweep inputs."""

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
        for weight
        in movement_weights
    ):
        raise ValueError(
            "movement_weights must all be finite."
        )

    if any(
        weight < 0.0
        for weight
        in movement_weights
    ):
        raise ValueError(
            "movement_weights must be non-negative."
        )

    if repetitions <= 0:
        raise ValueError(
            "repetitions must be positive."
        )


def _candidate_index(
    candidates,
    selected_candidate,
) -> int:
    """Return selected candidate index by object identity."""

    for index, candidate in enumerate(
        candidates
    ):
        if candidate is selected_candidate:
            return index

    raise ValueError(
        "Selected viewpoint was not in the supplied candidate set."
    )


def run_development_records(
    *,
    scenarios: tuple[
        RobustnessScenario,
        ...,
    ] | None = None,
    movement_weights: tuple[
        float,
        ...,
    ] = DEFAULT_MOVEMENT_WEIGHTS,
    repetitions: int = DEFAULT_REPETITIONS,
) -> tuple[
    FairSceneDevelopmentRecord,
    ...,
]:
    """Run matched development comparisons.

    Each trial follows the strict information-isolation protocol:

    1. Initial camera pose comes from the fixed nominal workspace prior.
    2. Simulator truth generates one shared noisy initial observation.
    3. The active candidate set is centred on the noisy estimated target.
    4. Generic and Task-Aware receive the same estimate and candidates.
    5. Neither selector receives the true target position or true anatomy.
    6. Simulator truth is used again only to generate final observations.

    Generic and Task-Aware use the same movement weight during this
    development sweep. Independent movement-budget matching occurs later.
    """

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

    _validate_configuration(
        movement_weights=(
            movement_weights
        ),
        repetitions=(
            repetitions
        ),
    )

    records: list[
        FairSceneDevelopmentRecord
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

            final_seed = (
                FINAL_SEED_BASE
                + scenario.scenario_id
                * 100
                + repetition
            )

            # Build the shared information-isolated trial context.
            #
            # Importantly:
            # - inputs.initial_pose is deliberately NOT used;
            # - inputs.candidates is deliberately NOT used.
            #
            # Those historical quantities were generated around the true
            # target. The new context instead uses:
            #
            # nominal prior -> noisy observation -> estimated target ->
            # estimate-centred active candidate set.
            context = (
                build_fair_candidate_context(
                    observation_model=(
                        inputs
                        .observation_model
                    ),
                    true_structures=(
                        inputs
                        .true_structures
                    ),
                    occluders=(
                        inputs
                        .occluders
                    ),
                    initial_view_index=(
                        scenario
                        .initial_view_index
                    ),
                    initial_seed=(
                        initial_seed
                    ),
                )
            )

            for movement_weight in (
                movement_weights
            ):
                generic_scorer = (
                    GenericSceneViewpointScorer(
                        observation_model=(
                            inputs
                            .observation_model
                        ),
                        config=(
                            GenericSceneScoringConfig(
                                perception_weight=1.0,
                                movement_weight=(
                                    movement_weight
                                ),
                            )
                        ),
                    )
                )

                task_scorer = (
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
                                    movement_weight
                                ),
                                alignment_weight=1.0,
                            )
                        ),
                    )
                )

                generic_scores = (
                    generic_scorer
                    .score_candidates(
                        current_pose=(
                            context
                            .initial_pose
                        ),
                        candidates=(
                            context
                            .candidates
                        ),
                        initial_perception=(
                            context
                            .planner_facing_perception
                        ),
                    )
                )

                task_scores = (
                    task_scorer
                    .score_candidates(
                        current_pose=(
                            context
                            .initial_pose
                        ),
                        candidates=(
                            context
                            .candidates
                        ),
                        initial_perception=(
                            context
                            .planner_facing_perception
                        ),
                    )
                )

                generic_selected = max(
                    generic_scores,
                    key=lambda item:
                    item.score,
                )

                task_selected = max(
                    task_scores,
                    key=lambda item:
                    item.score,
                )

                generic_index = (
                    _candidate_index(
                        context.candidates,
                        generic_selected
                        .candidate,
                    )
                )

                task_index = (
                    _candidate_index(
                        context.candidates,
                        task_selected
                        .candidate,
                    )
                )

                # Task-aware quantities evaluated at the viewpoint selected
                # by Generic.
                #
                # These are diagnostics only. They do not influence Generic
                # selection.
                generic_task_diagnostics = (
                    task_scores[
                        generic_index
                    ]
                )

                # Task-agnostic scene information evaluated at the viewpoint
                # selected by Task-Aware.
                #
                # Again this is post-selection diagnostic information only.
                task_generic_diagnostics = (
                    generic_scores[
                        task_index
                    ]
                )

                # Generate the planner-facing final observation for Generic.
                #
                # Ground truth is permitted here because this is the
                # simulator observation generator, not decision logic.
                generic_final = (
                    _observe_final_pose(
                        strategy=(
                            PerceptionStrategy
                            .GENERIC_ACTIVE
                        ),
                        observation_model=(
                            inputs
                            .observation_model
                        ),
                        selected_pose=(
                            generic_selected
                            .candidate
                            .pose
                        ),
                        initial_pose=(
                            context
                            .initial_pose
                        ),
                        true_structures=(
                            inputs
                            .true_structures
                        ),
                        occluders=(
                            inputs
                            .occluders
                        ),
                        seed=int(
                            final_seed
                        ),
                        candidate_count=len(
                            context
                            .candidates
                        ),
                    )
                )

                relevance = np.asarray(
                    task_selected
                    .relevance_weights,
                    dtype=float,
                )

                # Generate the planner-facing final observation for
                # Task-Aware using the same stochastic seed.
                task_final = (
                    _observe_final_pose(
                        strategy=(
                            PerceptionStrategy
                            .TASK_AWARE_ACTIVE
                        ),
                        observation_model=(
                            inputs
                            .observation_model
                        ),
                        selected_pose=(
                            task_selected
                            .candidate
                            .pose
                        ),
                        initial_pose=(
                            context
                            .initial_pose
                        ),
                        true_structures=(
                            inputs
                            .true_structures
                        ),
                        occluders=(
                            inputs
                            .occluders
                        ),
                        seed=int(
                            final_seed
                        ),
                        candidate_count=len(
                            context
                            .candidates
                        ),
                        task_relevance=float(
                            np.mean(
                                relevance
                            )
                        ),
                        task_alignment=float(
                            task_selected
                            .task_alignment
                        ),
                    )
                )

                records.append(
                    FairSceneDevelopmentRecord(
                        scenario_id=int(
                            scenario
                            .scenario_id
                        ),
                        scenario_name=(
                            scenario
                            .name
                        ),
                        repetition=int(
                            repetition
                        ),
                        initial_seed=int(
                            initial_seed
                        ),
                        final_seed=int(
                            final_seed
                        ),
                        movement_weight=float(
                            movement_weight
                        ),
                        generic_candidate_index=int(
                            generic_index
                        ),
                        task_candidate_index=int(
                            task_index
                        ),
                        different_candidate=bool(
                            generic_index
                            != task_index
                        ),
                        generic_camera_movement=float(
                            generic_final
                            .camera_movement
                        ),
                        task_camera_movement=float(
                            task_final
                            .camera_movement
                        ),
                        generic_localisation_error=float(
                            generic_final
                            .mean_localisation_error
                        ),
                        task_localisation_error=float(
                            task_final
                            .mean_localisation_error
                        ),
                        generic_predicted_sigma=float(
                            generic_final
                            .mean_predicted_sigma
                        ),
                        task_predicted_sigma=float(
                            task_final
                            .mean_predicted_sigma
                        ),
                        generic_scene_information=float(
                            generic_selected
                            .scene_information
                        ),
                        task_generic_scene_information=float(
                            task_generic_diagnostics
                            .scene_information
                        ),
                        generic_task_scene_information=float(
                            generic_task_diagnostics
                            .task_scene_information
                        ),
                        task_task_scene_information=float(
                            task_selected
                            .task_scene_information
                        ),
                        generic_alignment=float(
                            generic_task_diagnostics
                            .task_alignment
                        ),
                        task_alignment=float(
                            task_selected
                            .task_alignment
                        ),
                        generic_score=float(
                            generic_selected
                            .score
                        ),
                        task_score=float(
                            task_selected
                            .score
                        ),
                    )
                )

    return tuple(
        records
    )


def analyse_development_records(
    *,
    records: tuple[
        FairSceneDevelopmentRecord,
        ...,
    ],
    movement_weights: tuple[
        float,
        ...,
    ],
) -> tuple[
    FairSceneDevelopmentSummary,
    ...,
]:
    """Aggregate records by movement weight."""

    if len(
        records
    ) == 0:
        raise ValueError(
            "records must not be empty."
        )

    summaries: list[
        FairSceneDevelopmentSummary
    ] = []

    for movement_weight in (
        movement_weights
    ):
        selected = tuple(
            record
            for record
            in records
            if np.isclose(
                record
                .movement_weight,
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

        def values(
            attribute: str,
        ) -> np.ndarray:
            return np.asarray(
                [
                    getattr(
                        record,
                        attribute,
                    )
                    for record
                    in selected
                ],
                dtype=float,
            )

        generic_movement = values(
            "generic_camera_movement"
        )

        task_movement = values(
            "task_camera_movement"
        )

        generic_error = values(
            "generic_localisation_error"
        )

        task_error = values(
            "task_localisation_error"
        )

        generic_sigma = values(
            "generic_predicted_sigma"
        )

        task_sigma = values(
            "task_predicted_sigma"
        )

        generic_alignment = values(
            "generic_alignment"
        )

        task_alignment = values(
            "task_alignment"
        )

        different = values(
            "different_candidate"
        )

        summaries.append(
            FairSceneDevelopmentSummary(
                movement_weight=float(
                    movement_weight
                ),
                record_count=len(
                    selected
                ),
                mean_generic_camera_movement=float(
                    np.mean(
                        generic_movement
                    )
                ),
                mean_task_camera_movement=float(
                    np.mean(
                        task_movement
                    )
                ),
                mean_camera_movement_difference=float(
                    np.mean(
                        task_movement
                        - generic_movement
                    )
                ),
                mean_generic_localisation_error=float(
                    np.mean(
                        generic_error
                    )
                ),
                mean_task_localisation_error=float(
                    np.mean(
                        task_error
                    )
                ),
                mean_localisation_error_difference=float(
                    np.mean(
                        task_error
                        - generic_error
                    )
                ),
                mean_generic_predicted_sigma=float(
                    np.mean(
                        generic_sigma
                    )
                ),
                mean_task_predicted_sigma=float(
                    np.mean(
                        task_sigma
                    )
                ),
                mean_predicted_sigma_difference=float(
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
                mean_generic_scene_information=float(
                    np.mean(
                        values(
                            "generic_scene_information"
                        )
                    )
                ),
                mean_task_scene_information=float(
                    np.mean(
                        values(
                            "task_task_scene_information"
                        )
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


def run_fair_scene_development_sweep(
    *,
    scenarios: tuple[
        RobustnessScenario,
        ...,
    ] | None = None,
    movement_weights: tuple[
        float,
        ...,
    ] = DEFAULT_MOVEMENT_WEIGHTS,
    repetitions: int = DEFAULT_REPETITIONS,
) -> FairSceneDevelopmentResult:
    """Run the complete Phase 1 development sweep."""

    if scenarios is None:
        scenarios = (
            default_scenarios()
        )

    records = (
        run_development_records(
            scenarios=scenarios,
            movement_weights=(
                movement_weights
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
                movement_weights
            ),
        )
    )

    return FairSceneDevelopmentResult(
        movement_weights=(
            movement_weights
        ),
        scenarios=scenarios,
        repetitions=int(
            repetitions
        ),
        records=records,
        summaries=summaries,
    )


def save_outputs(
    result: FairSceneDevelopmentResult,
    *,
    output_directory: str
    | Path = (
        "results/"
        "phase1_fair_scene_development"
    ),
) -> tuple[
    Path,
    Path,
]:
    """Save raw and aggregate development results."""

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        output_directory
        / "fair_scene_development_trials.csv"
    )

    summary_path = (
        output_directory
        / "fair_scene_development_summary.json"
    )

    if len(
        result.records
    ) == 0:
        raise ValueError(
            "Cannot save empty results."
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
        "experiment_role": (
            "development_only"
        ),
        "held_out_test": False,
        "information_isolation": {
            "initial_pose": (
                "fixed nominal workspace prior"
            ),
            "candidate_generation": (
                "centred on noisy estimated target"
            ),
            "generic_task_information": False,
            "task_aware_task_information": (
                "planned instrument trajectory only"
            ),
            "ground_truth_use": (
                "simulated observation generation "
                "and evaluation only"
            ),
        },
        "scenario_count": len(
            result.scenarios
        ),
        "repetitions": (
            result.repetitions
        ),
        "movement_weights": list(
            result.movement_weights
        ),
        "summaries": [
            asdict(
                summary
            )
            for summary
            in result.summaries
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
            allow_nan=False,
        )

    return (
        raw_path,
        summary_path,
    )


def print_summary(
    result: FairSceneDevelopmentResult,
) -> None:
    """Print compact development results."""

    print()

    print(
        "Fair Scene-Wide Development Sweep"
    )

    print(
        "================================="
    )

    print(
        f"Scenarios: "
        f"{len(result.scenarios)}"
    )

    print(
        f"Repetitions per scenario: "
        f"{result.repetitions}"
    )

    print()

    print(
        "Weight | Generic move | Task move | "
        "Generic error | Task error | Different"
    )

    print(
        "-" * 92
    )

    for summary in (
        result.summaries
    ):
        print(
            f"{summary.movement_weight:6.3f} | "
            f"{summary.mean_generic_camera_movement * 1000.0:11.3f} mm | "
            f"{summary.mean_task_camera_movement * 1000.0:9.3f} mm | "
            f"{summary.mean_generic_localisation_error * 1000.0:12.3f} mm | "
            f"{summary.mean_task_localisation_error * 1000.0:10.3f} mm | "
            f"{summary.different_candidate_rate * 100.0:8.1f}%"
        )

    print()

    print(
        "Weight | Generic sigma | Task sigma | "
        "Generic align | Task align | Delta move"
    )

    print(
        "-" * 90
    )

    for summary in (
        result.summaries
    ):
        print(
            f"{summary.movement_weight:6.3f} | "
            f"{summary.mean_generic_predicted_sigma * 1000.0:11.3f} mm | "
            f"{summary.mean_task_predicted_sigma * 1000.0:9.3f} mm | "
            f"{summary.mean_generic_alignment:13.4f} | "
            f"{summary.mean_task_alignment:10.4f} | "
            f"{summary.mean_camera_movement_difference * 1000.0:10.3f} mm"
        )


def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 1 fair scene-wide development diagnostics "
            "with ground-truth-isolated candidate generation."
        )
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
            "phase1_fair_scene_development"
        ),
    )

    args = parser.parse_args()

    result = (
        run_fair_scene_development_sweep(
            repetitions=(
                args.repetitions
            )
        )
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
        "Raw development results saved to: "
        f"{raw_path}"
    )

    print(
        "Development summary saved to: "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()