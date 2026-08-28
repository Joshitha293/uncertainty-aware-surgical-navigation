"""Validation gate for the corrected Phase 1 fairness experiment.

Development-frozen parameters
-----------------------------
Generic movement weight:
    0.072

Task-Aware movement weight:
    0.200

Frozen scenario manifest SHA-256:
    490922fbc9f91743262257ff594a7440651016085207b3e8867b9e1bdce8ddd9

Corrected information flow
--------------------------
1. Initial camera geometry comes from a fixed nominal workspace prior.
2. Simulator truth generates one shared noisy initial observation.
3. Active candidates are centred on the noisy estimated target.
4. Generic and Task-Aware receive identical estimated anatomy and candidates.
5. Generic receives no task trajectory.
6. Task-Aware additionally receives the planned trajectory.
7. Ground truth is used again only to simulate final observations.

Only validation scenarios are executed by this module. Held-out scenarios are
reconstructed solely for manifest-integrity verification and are never run.
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
from src.simulation.fair_scene_perception import (
    _observe_final_pose,
)
from src.simulation.phase1_fair_candidate_context import (
    build_fair_candidate_context,
)
from src.simulation.phase1_scenario_splits import (
    Phase1ScenarioSplits,
    Phase1SplitConfig,
    manifest_digest,
)
from src.simulation.three_strategy_perception import (
    PerceptionStrategy,
)
from src.simulation.three_strategy_robustness_benchmark import (
    RobustnessScenario,
    build_scenario_inputs,
)


FROZEN_MANIFEST_SHA256 = (
    "490922fbc9f91743262257ff594a7440"
    "651016085207b3e8867b9e1bdce8ddd9"
)

FROZEN_GENERIC_MOVEMENT_WEIGHT = 0.072

FROZEN_TASK_MOVEMENT_WEIGHT = 0.200

VALIDATION_BUDGET_TOLERANCE = 0.10

DEFAULT_REPETITIONS = 5

INITIAL_SEED_BASE = 40262000

FINAL_SEED_BASE = 50262000


@dataclass(frozen=True)
class Phase1ValidationRecord:
    """One paired corrected validation trial."""

    scenario_id: int
    scenario_name: str

    repetition: int

    initial_seed: int
    final_seed: int

    generic_movement_weight: float
    task_movement_weight: float

    generic_candidate_index: int
    task_candidate_index: int

    different_candidate: bool

    generic_camera_movement: float
    task_camera_movement: float

    generic_localisation_error: float
    task_localisation_error: float

    generic_predicted_sigma: float
    task_predicted_sigma: float

    task_alignment: float


@dataclass(frozen=True)
class Phase1ValidationSummary:
    """Aggregate corrected validation result."""

    scenario_count: int
    record_count: int

    generic_movement_weight: float
    task_movement_weight: float

    mean_generic_camera_movement: float
    mean_task_camera_movement: float

    mean_paired_movement_difference: float

    absolute_budget_gap: float
    relative_budget_gap: float

    budget_tolerance: float
    budget_gate_passed: bool

    mean_generic_localisation_error: float
    mean_task_localisation_error: float

    mean_generic_predicted_sigma: float
    mean_task_predicted_sigma: float

    mean_task_alignment: float

    different_candidate_rate: float


@dataclass(frozen=True)
class Phase1ValidationResult:
    """Complete corrected validation experiment."""

    manifest_sha256: str

    records: tuple[
        Phase1ValidationRecord,
        ...,
    ]

    summary: Phase1ValidationSummary


def _scenario_from_dict(
    payload: dict,
) -> RobustnessScenario:
    """Reconstruct one frozen scenario from the manifest."""

    return RobustnessScenario(
        scenario_id=int(
            payload[
                "scenario_id"
            ]
        ),
        name=str(
            payload[
                "name"
            ]
        ),
        translation=tuple(
            float(value)
            for value
            in payload[
                "translation"
            ]
        ),
        radius_scale=float(
            payload[
                "radius_scale"
            ]
        ),
        safety_margin_scale=float(
            payload[
                "safety_margin_scale"
            ]
        ),
        initial_view_index=int(
            payload[
                "initial_view_index"
            ]
        ),
        occluder_radius=float(
            payload[
                "occluder_radius"
            ]
        ),
    )


def load_frozen_validation_scenarios(
    manifest_path: str | Path,
    *,
    expected_digest: str = (
        FROZEN_MANIFEST_SHA256
    ),
) -> tuple[
    RobustnessScenario,
    ...,
]:
    """Verify the frozen manifest and return validation scenarios only."""

    manifest_path = Path(
        manifest_path
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Scenario manifest not found: "
            f"{manifest_path}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(
            handle
        )

    stored_digest = payload.get(
        "sha256"
    )

    if stored_digest is None:
        raise ValueError(
            "Scenario manifest contains no SHA-256 digest."
        )

    if stored_digest != expected_digest:
        raise ValueError(
            "Stored scenario-manifest digest does not "
            "match the frozen Phase 1 digest."
        )

    config = Phase1SplitConfig(
        **payload[
            "config"
        ]
    )

    development = tuple(
        _scenario_from_dict(
            item
        )
        for item
        in payload[
            "development"
        ]
    )

    validation = tuple(
        _scenario_from_dict(
            item
        )
        for item
        in payload[
            "validation"
        ]
    )

    held_out = tuple(
        _scenario_from_dict(
            item
        )
        for item
        in payload[
            "held_out"
        ]
    )

    splits = Phase1ScenarioSplits(
        development=development,
        validation=validation,
        held_out=held_out,
    )

    recomputed_digest = (
        manifest_digest(
            splits=splits,
            config=config,
        )
    )

    if (
        recomputed_digest
        != stored_digest
    ):
        raise ValueError(
            "Scenario manifest integrity check failed."
        )

    if len(
        validation
    ) != config.validation_count:
        raise ValueError(
            "Validation scenario count does not match "
            "the frozen protocol."
        )

    return validation


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
        "Selected candidate was not in the common candidate set."
    )


def run_validation_records(
    *,
    scenarios: tuple[
        RobustnessScenario,
        ...,
    ],
    repetitions: int = DEFAULT_REPETITIONS,
    generic_movement_weight: float = (
        FROZEN_GENERIC_MOVEMENT_WEIGHT
    ),
    task_movement_weight: float = (
        FROZEN_TASK_MOVEMENT_WEIGHT
    ),
) -> tuple[
    Phase1ValidationRecord,
    ...,
]:
    """Run corrected Generic and Task-Aware strategies on validation."""

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

    if (
        not np.isfinite(
            generic_movement_weight
        )
        or not np.isfinite(
            task_movement_weight
        )
    ):
        raise ValueError(
            "Movement weights must be finite."
        )

    if (
        generic_movement_weight < 0.0
        or task_movement_weight < 0.0
    ):
        raise ValueError(
            "Movement weights must be non-negative."
        )

    records: list[
        Phase1ValidationRecord
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
                    initial_seed=int(
                        initial_seed
                    ),
                )
            )

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
                                generic_movement_weight
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
                                task_movement_weight
                            ),
                            alignment_weight=1.0,
                        )
                    ),
                )
            )

            generic_selection = (
                generic_scorer
                .select_viewpoint(
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

            task_selection = (
                task_scorer
                .select_viewpoint(
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

            generic_selected = (
                generic_selection
                .selected
            )

            task_selected = (
                task_selection
                .selected
            )

            generic_index = (
                _candidate_index(
                    context
                    .candidates,
                    generic_selected
                    .candidate,
                )
            )

            task_index = (
                _candidate_index(
                    context
                    .candidates,
                    task_selected
                    .candidate,
                )
            )

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
                Phase1ValidationRecord(
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
                    generic_movement_weight=float(
                        generic_movement_weight
                    ),
                    task_movement_weight=float(
                        task_movement_weight
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
                    task_alignment=float(
                        task_selected
                        .task_alignment
                    ),
                )
            )

    return tuple(
        records
    )


def analyse_validation_records(
    *,
    records: tuple[
        Phase1ValidationRecord,
        ...,
    ],
    budget_tolerance: float = (
        VALIDATION_BUDGET_TOLERANCE
    ),
) -> Phase1ValidationSummary:
    """Evaluate the pre-specified validation movement-budget gate."""

    if len(
        records
    ) == 0:
        raise ValueError(
            "records must not be empty."
        )

    if (
        not np.isfinite(
            budget_tolerance
        )
        or budget_tolerance < 0.0
    ):
        raise ValueError(
            "budget_tolerance must be finite and non-negative."
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
                in records
            ],
            dtype=float,
        )

    generic_movement = values(
        "generic_camera_movement"
    )

    task_movement = values(
        "task_camera_movement"
    )

    mean_generic = float(
        np.mean(
            generic_movement
        )
    )

    mean_task = float(
        np.mean(
            task_movement
        )
    )

    if mean_task <= 1e-12:
        raise ValueError(
            "Mean Task-Aware validation movement must be positive."
        )

    absolute_gap = abs(
        mean_task
        - mean_generic
    )

    relative_gap = (
        absolute_gap
        / mean_task
    )

    scenario_count = len(
        {
            record.scenario_id
            for record
            in records
        }
    )

    return Phase1ValidationSummary(
        scenario_count=int(
            scenario_count
        ),
        record_count=len(
            records
        ),
        generic_movement_weight=float(
            records[
                0
            ]
            .generic_movement_weight
        ),
        task_movement_weight=float(
            records[
                0
            ]
            .task_movement_weight
        ),
        mean_generic_camera_movement=(
            mean_generic
        ),
        mean_task_camera_movement=(
            mean_task
        ),
        mean_paired_movement_difference=float(
            np.mean(
                task_movement
                - generic_movement
            )
        ),
        absolute_budget_gap=float(
            absolute_gap
        ),
        relative_budget_gap=float(
            relative_gap
        ),
        budget_tolerance=float(
            budget_tolerance
        ),
        budget_gate_passed=bool(
            relative_gap
            <= budget_tolerance
        ),
        mean_generic_localisation_error=float(
            np.mean(
                values(
                    "generic_localisation_error"
                )
            )
        ),
        mean_task_localisation_error=float(
            np.mean(
                values(
                    "task_localisation_error"
                )
            )
        ),
        mean_generic_predicted_sigma=float(
            np.mean(
                values(
                    "generic_predicted_sigma"
                )
            )
        ),
        mean_task_predicted_sigma=float(
            np.mean(
                values(
                    "task_predicted_sigma"
                )
            )
        ),
        mean_task_alignment=float(
            np.mean(
                values(
                    "task_alignment"
                )
            )
        ),
        different_candidate_rate=float(
            np.mean(
                values(
                    "different_candidate"
                )
            )
        ),
    )


def run_validation_gate(
    *,
    manifest_path: str | Path = (
        "results/"
        "phase1_scenario_splits/"
        "scenario_manifest.json"
    ),
    repetitions: int = (
        DEFAULT_REPETITIONS
    ),
) -> Phase1ValidationResult:
    """Verify manifest and run validation scenarios only."""

    validation = (
        load_frozen_validation_scenarios(
            manifest_path
        )
    )

    records = (
        run_validation_records(
            scenarios=validation,
            repetitions=(
                repetitions
            ),
        )
    )

    summary = (
        analyse_validation_records(
            records=records
        )
    )

    return Phase1ValidationResult(
        manifest_sha256=(
            FROZEN_MANIFEST_SHA256
        ),
        records=records,
        summary=summary,
    )


def save_validation_result(
    result: Phase1ValidationResult,
    *,
    output_path: str | Path = (
        "results/"
        "phase1_validation/"
        "validation_gate_corrected.json"
    ),
) -> Path:
    """Save corrected validation evidence."""

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "experiment_role": (
            "validation_only_corrected_pipeline"
        ),
        "held_out_executed": False,
        "manifest_sha256": (
            result
            .manifest_sha256
        ),
        "candidate_generation": (
            "shared noisy estimated target"
        ),
        "initial_camera_geometry": (
            "fixed nominal workspace prior"
        ),
        "frozen_development_parameters": {
            "generic_movement_weight": (
                FROZEN_GENERIC_MOVEMENT_WEIGHT
            ),
            "task_movement_weight": (
                FROZEN_TASK_MOVEMENT_WEIGHT
            ),
            "validation_budget_tolerance": (
                VALIDATION_BUDGET_TOLERANCE
            ),
        },
        "summary": asdict(
            result.summary
        ),
        "records": [
            asdict(
                record
            )
            for record
            in result.records
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


def print_validation_summary(
    result: Phase1ValidationResult,
) -> None:
    """Print corrected validation results."""

    summary = (
        result.summary
    )

    print()

    print(
        "Phase 1 Corrected Validation Movement-Budget Gate"
    )

    print(
        "================================================"
    )

    print()

    print(
        "Frozen manifest:"
    )

    print(
        f"  {result.manifest_sha256}"
    )

    print()

    print(
        f"Validation scenarios:       "
        f"{summary.scenario_count}"
    )

    print(
        f"Records:                    "
        f"{summary.record_count}"
    )

    print()

    print(
        "Development-frozen movement weights"
    )

    print(
        f"  Generic:                  "
        f"{summary.generic_movement_weight:.3f}"
    )

    print(
        f"  Task-Aware:               "
        f"{summary.task_movement_weight:.3f}"
    )

    print()

    print(
        "Movement-budget validation"
    )

    print(
        f"  Generic movement:         "
        f"{summary.mean_generic_camera_movement * 1000.0:.3f} mm"
    )

    print(
        f"  Task-Aware movement:      "
        f"{summary.mean_task_camera_movement * 1000.0:.3f} mm"
    )

    print(
        f"  Absolute gap:             "
        f"{summary.absolute_budget_gap * 1000.0:.3f} mm"
    )

    print(
        f"  Relative gap:             "
        f"{summary.relative_budget_gap * 100.0:.2f}%"
    )

    print(
        f"  Pre-specified tolerance:  "
        f"{summary.budget_tolerance * 100.0:.1f}%"
    )

    print(
        f"  Budget gate passed:       "
        f"{summary.budget_gate_passed}"
    )

    print()

    print(
        "Diagnostic outcomes — NOT used for weight selection"
    )

    print(
        f"  Generic error:            "
        f"{summary.mean_generic_localisation_error * 1000.0:.3f} mm"
    )

    print(
        f"  Task-Aware error:         "
        f"{summary.mean_task_localisation_error * 1000.0:.3f} mm"
    )

    print(
        f"  Generic sigma:            "
        f"{summary.mean_generic_predicted_sigma * 1000.0:.3f} mm"
    )

    print(
        f"  Task-Aware sigma:         "
        f"{summary.mean_task_predicted_sigma * 1000.0:.3f} mm"
    )

    print(
        f"  Task alignment:           "
        f"{summary.mean_task_alignment:.4f}"
    )

    print(
        f"  Different viewpoint rate: "
        f"{summary.different_candidate_rate * 100.0:.1f}%"
    )


def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(
        description=(
            "Run corrected Phase 1 validation using "
            "development-frozen parameters."
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
            "validation_gate_corrected.json"
        ),
    )

    args = parser.parse_args()

    result = (
        run_validation_gate(
            manifest_path=(
                args.manifest
            ),
            repetitions=(
                args.repetitions
            ),
        )
    )

    print_validation_summary(
        result
    )

    output = (
        save_validation_result(
            result,
            output_path=(
                args.output
            ),
        )
    )

    print()

    print(
        "Corrected validation evidence saved to: "
        f"{output}"
    )


if __name__ == "__main__":
    main()