"""Final eight-strategy Phase 1 experiment runner.

IMPORTANT
---------
The final Phase 1 protocol was frozen before held-out execution.

Frozen protocol SHA-256:

    dc6537d3ff75832ccbe48c9b2690c8e966711b31806076cfbc2721488dbf6361

This module implements the strategy family declared in that protocol:

1. Fixed View
2. Random Active
3. Generic Active -- same movement coefficient as Task-Aware
4. Movement-Budget-Matched Generic
5. Alignment-Only
6. Task-Weighted Information-Only
7. Full Task-Aware
8. Privileged Oracle

The primary comparison remains:

    Full Task-Aware
            versus
    Movement-Budget-Matched Generic

This module is deliberately guarded so that development smoke testing can be
performed without accidentally executing the frozen held-out set.

Information isolation
---------------------
For every trial:

1. Initial camera geometry comes from the nominal workspace prior.
2. Simulator truth generates one shared noisy initial observation.
3. Active candidates are centred on the noisy estimated target.
4. Every active strategy receives the identical candidate objects.
5. Generic strategies receive no task trajectory.
6. Task-aware variants receive the planned instrument trajectory.
7. Only the Oracle may use true geometry during viewpoint selection.
8. Simulator truth is used for final observation generation and independent
   trajectory evaluation.

The same final-perception seed and planner seed are reused across strategies
within a scenario/repetition pair.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from src.perception.fair_scene_viewpoint_scoring import (
    FairSceneScoringConfig,
)
from src.perception.phase1_baselines import (
    select_oracle_viewpoint,
    select_random_viewpoint,
)
from src.simulation.fair_scene_perception import (
    _observe_final_pose,
)
from src.simulation.phase1_fair_candidate_context import (
    build_fair_candidate_context,
)
from src.simulation.phase1_frozen_protocol import (
    FROZEN_GENERIC_MOVEMENT_WEIGHT,
    FROZEN_SCENARIO_MANIFEST_SHA256,
    FROZEN_TASK_MOVEMENT_WEIGHT,
    StrategyId,
    build_frozen_phase1_protocol,
    held_out_trial_seeds,
)
from src.simulation.phase1_scenario_splits import (
    Phase1ScenarioSplits,
    Phase1SplitConfig,
    manifest_digest,
)
from src.simulation.three_strategy_navigation import (
    run_navigation_from_perception,
)
from src.simulation.three_strategy_perception import (
    PerceptionStrategy,
    StrategyPerceptionResult,
)
from src.simulation.three_strategy_robustness_benchmark import (
    RobustnessScenario,
    build_scenario_inputs,
    default_scenarios,
)


FROZEN_PROTOCOL_SHA256 = (
    "dc6537d3ff75832ccbe48c9b2690c8e9"
    "66711b31806076cfbc2721488dbf6361"
)


# Existing validated uncertainty-aware navigation configuration.
SIGMA_MULTIPLIER = 2.0

INSTRUMENT_RADIUS = 0.006

PROXIMAL_LENGTH = 0.10


HELD_OUT_ID_MIN = 2000
HELD_OUT_ID_MAX_EXCLUSIVE = 2030


@dataclass(frozen=True)
class StrategyPerceptionBundle:
    """One final perception result plus frozen strategy metadata."""

    strategy_id: StrategyId

    perception: StrategyPerceptionResult

    selected_candidate_index: int | None

    movement_weight: float | None

    selection_score: float | None


@dataclass(frozen=True)
class Phase1FinalRecord:
    """One strategy evaluation from one matched scenario repetition."""

    scenario_id: int
    scenario_name: str

    repetition: int

    strategy: str

    initial_perception_seed: int
    final_perception_seed: int
    random_viewpoint_seed: int
    planner_seed: int

    selected_candidate_index: int | None

    movement_weight: float | None

    camera_movement: float

    mean_localisation_error: float
    mean_predicted_sigma: float

    task_relevance: float | None
    task_alignment: float | None

    planning_success: bool

    collision_against_truth: bool

    safety_violation_against_truth: bool

    safe_navigation_success: bool

    minimum_true_safety_clearance: float

    mean_planning_safety_margin: float
    maximum_planning_safety_radius: float

    planner_iterations: int
    planning_time_seconds: float

    path_cost: float

    maximum_rcm_error: float


@dataclass(frozen=True)
class Phase1FinalTrial:
    """All eight strategy records from one matched trial."""

    scenario_id: int

    repetition: int

    records: tuple[
        Phase1FinalRecord,
        ...,
    ]


def _candidate_index(
    candidates,
    selected_candidate,
) -> int:
    """Return candidate index using object identity."""

    for index, candidate in enumerate(
        candidates
    ):
        if candidate is selected_candidate:
            return index

    raise ValueError(
        "Selected candidate was not in the shared candidate set."
    )


def _perception_enum(
    strategy_id: StrategyId,
) -> PerceptionStrategy:
    """Map the eight research strategies to the legacy perception type."""

    if strategy_id == StrategyId.FIXED:
        return PerceptionStrategy.FIXED

    if strategy_id in (
        StrategyId.RANDOM_ACTIVE,
        StrategyId.GENERIC_ACTIVE,
        StrategyId.BUDGET_MATCHED_GENERIC,
    ):
        return (
            PerceptionStrategy
            .GENERIC_ACTIVE
        )

    return (
        PerceptionStrategy
        .TASK_AWARE_ACTIVE
    )


def _mean_relevance(
    values,
) -> float:
    """Return mean task relevance."""

    array = np.asarray(
        values,
        dtype=float,
    )

    if array.size == 0:
        raise ValueError(
            "Task relevance array must not be empty."
        )

    return float(
        np.mean(
            array
        )
    )


def _observe_selected_pose(
    *,
    strategy_id: StrategyId,
    selected_pose,
    initial_pose,
    observation_model,
    true_structures,
    occluders,
    final_seed: int,
    candidate_count: int,
    task_relevance: float | None = None,
    task_alignment: float | None = None,
) -> StrategyPerceptionResult:
    """Generate one matched final observation."""

    return _observe_final_pose(
        strategy=(
            _perception_enum(
                strategy_id
            )
        ),
        observation_model=(
            observation_model
        ),
        selected_pose=(
            selected_pose
        ),
        initial_pose=(
            initial_pose
        ),
        true_structures=(
            true_structures
        ),
        occluders=(
            occluders
        ),
        seed=int(
            final_seed
        ),
        candidate_count=int(
            candidate_count
        ),
        task_relevance=(
            task_relevance
        ),
        task_alignment=(
            task_alignment
        ),
    )


def build_eight_strategy_perceptions(
    *,
    scenario: RobustnessScenario,
    repetition: int,
) -> tuple[
    StrategyPerceptionBundle,
    ...,
]:
    """Build the final matched perception result for all eight strategies."""

    if repetition < 0:
        raise ValueError(
            "repetition must be non-negative."
        )

    protocol = (
        build_frozen_phase1_protocol()
    )

    seeds = (
        held_out_trial_seeds(
            protocol=protocol,
            scenario_id=(
                scenario.scenario_id
            ),
            repetition=(
                repetition
            ),
        )
    )

    inputs = (
        build_scenario_inputs(
            scenario
        )
    )

    context = (
        build_fair_candidate_context(
            observation_model=(
                inputs.observation_model
            ),
            true_structures=(
                inputs.true_structures
            ),
            occluders=(
                inputs.occluders
            ),
            initial_view_index=(
                scenario.initial_view_index
            ),
            initial_seed=(
                seeds[
                    "initial_perception"
                ]
            ),
        )
    )

    candidates = (
        context.candidates
    )

    initial_pose = (
        context.initial_pose
    )

    initial_perception = (
        context
        .planner_facing_perception
    )

    final_seed = (
        seeds[
            "final_perception"
        ]
    )

    # --------------------------------------------------------------
    # Task-agnostic Generic: same coefficient as Full Task-Aware.
    # --------------------------------------------------------------

    generic_same_weight = (
        GenericSceneViewpointScorer(
            observation_model=(
                inputs.observation_model
            ),
            config=(
                GenericSceneScoringConfig(
                    perception_weight=1.0,
                    movement_weight=(
                        FROZEN_TASK_MOVEMENT_WEIGHT
                    ),
                )
            ),
        )
    )

    generic_same_scores = (
        generic_same_weight
        .score_candidates(
            current_pose=(
                initial_pose
            ),
            candidates=(
                candidates
            ),
            initial_perception=(
                initial_perception
            ),
        )
    )

    generic_same_selected = max(
        generic_same_scores,
        key=lambda item:
        item.score,
    )

    generic_same_index = (
        _candidate_index(
            candidates,
            generic_same_selected
            .candidate,
        )
    )

    # --------------------------------------------------------------
    # Primary comparator: movement-budget-matched Generic.
    # --------------------------------------------------------------

    budget_generic = (
        GenericSceneViewpointScorer(
            observation_model=(
                inputs.observation_model
            ),
            config=(
                GenericSceneScoringConfig(
                    perception_weight=1.0,
                    movement_weight=(
                        FROZEN_GENERIC_MOVEMENT_WEIGHT
                    ),
                )
            ),
        )
    )

    budget_scores = (
        budget_generic
        .score_candidates(
            current_pose=(
                initial_pose
            ),
            candidates=(
                candidates
            ),
            initial_perception=(
                initial_perception
            ),
        )
    )

    budget_selected = max(
        budget_scores,
        key=lambda item:
        item.score,
    )

    budget_index = (
        _candidate_index(
            candidates,
            budget_selected
            .candidate,
        )
    )

    # --------------------------------------------------------------
    # Full Task-Aware primitives.
    #
    # The same primitive quantities are reused to implement the two
    # mechanism ablations exactly as frozen in the protocol.
    # --------------------------------------------------------------

    task_scorer = (
        TaskAwareSceneViewpointScorer(
            observation_model=(
                inputs.observation_model
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

    task_scores = (
        task_scorer
        .score_candidates(
            current_pose=(
                initial_pose
            ),
            candidates=(
                candidates
            ),
            initial_perception=(
                initial_perception
            ),
        )
    )

    if len(
        generic_same_scores
    ) != len(
        task_scores
    ):
        raise RuntimeError(
            "Generic and Task-Aware candidate score counts differ."
        )

    for generic_score, task_score in zip(
        generic_same_scores,
        task_scores,
    ):
        if (
            generic_score.candidate
            is not task_score.candidate
        ):
            raise RuntimeError(
                "Generic and Task-Aware candidate ordering differs."
            )

    full_selected = max(
        task_scores,
        key=lambda item:
        item.score,
    )

    full_index = (
        _candidate_index(
            candidates,
            full_selected
            .candidate,
        )
    )

    # Alignment-Only:
    #
    # uniform scene information
    # + task alignment
    # - movement penalty
    alignment_scores = tuple(
        (
            generic_score
            .scene_information
            + task_score
            .task_alignment
            - FROZEN_TASK_MOVEMENT_WEIGHT
            * generic_score
            .normalised_movement_cost
        )
        for generic_score, task_score in zip(
            generic_same_scores,
            task_scores,
        )
    )

    alignment_index = int(
        np.argmax(
            np.asarray(
                alignment_scores,
                dtype=float,
            )
        )
    )

    alignment_generic = (
        generic_same_scores[
            alignment_index
        ]
    )

    alignment_task = (
        task_scores[
            alignment_index
        ]
    )

    # Information-Only:
    #
    # task-weighted predicted information
    # - movement penalty
    information_scores = tuple(
        (
            task_score
            .task_scene_information
            - FROZEN_TASK_MOVEMENT_WEIGHT
            * generic_score
            .normalised_movement_cost
        )
        for generic_score, task_score in zip(
            generic_same_scores,
            task_scores,
        )
    )

    information_index = int(
        np.argmax(
            np.asarray(
                information_scores,
                dtype=float,
            )
        )
    )

    information_task = (
        task_scores[
            information_index
        ]
    )

    # --------------------------------------------------------------
    # Random Active.
    # --------------------------------------------------------------

    random_selection = (
        select_random_viewpoint(
            current_pose=(
                initial_pose
            ),
            candidates=(
                candidates
            ),
            seed=(
                seeds[
                    "random_viewpoint"
                ]
            ),
        )
    )

    random_index = (
        random_selection
        .candidate_index
    )

    # --------------------------------------------------------------
    # Privileged Oracle.
    #
    # It MUST use the same estimate-centred candidates.
    # Only its scoring stage receives simulator truth.
    # --------------------------------------------------------------

    oracle = (
        select_oracle_viewpoint(
            observation_model=(
                inputs.observation_model
            ),
            current_pose=(
                initial_pose
            ),
            candidates=(
                candidates
            ),
            true_structures=(
                inputs.true_structures
            ),
            true_occluders=(
                inputs.occluders
            ),
            task_trajectory=(
                inputs
                .task
                .trajectory
            ),
            config=(
                FairSceneScoringConfig(
                    perception_weight=1.0,
                    movement_weight=(
                        FROZEN_TASK_MOVEMENT_WEIGHT
                    ),
                    alignment_weight=1.0,
                )
            ),
        )
    )

    oracle_selected = (
        oracle.selected
    )

    oracle_index = (
        _candidate_index(
            candidates,
            oracle_selected
            .candidate,
        )
    )

    # --------------------------------------------------------------
    # Final matched observations.
    # --------------------------------------------------------------

    fixed_perception = (
        _observe_selected_pose(
            strategy_id=(
                StrategyId.FIXED
            ),
            selected_pose=(
                initial_pose
            ),
            initial_pose=(
                initial_pose
            ),
            observation_model=(
                inputs.observation_model
            ),
            true_structures=(
                inputs.true_structures
            ),
            occluders=(
                inputs.occluders
            ),
            final_seed=(
                final_seed
            ),
            candidate_count=0,
        )
    )

    random_perception = (
        _observe_selected_pose(
            strategy_id=(
                StrategyId
                .RANDOM_ACTIVE
            ),
            selected_pose=(
                random_selection
                .candidate
                .pose
            ),
            initial_pose=(
                initial_pose
            ),
            observation_model=(
                inputs.observation_model
            ),
            true_structures=(
                inputs.true_structures
            ),
            occluders=(
                inputs.occluders
            ),
            final_seed=(
                final_seed
            ),
            candidate_count=len(
                candidates
            ),
        )
    )

    generic_perception = (
        _observe_selected_pose(
            strategy_id=(
                StrategyId
                .GENERIC_ACTIVE
            ),
            selected_pose=(
                generic_same_selected
                .candidate
                .pose
            ),
            initial_pose=(
                initial_pose
            ),
            observation_model=(
                inputs.observation_model
            ),
            true_structures=(
                inputs.true_structures
            ),
            occluders=(
                inputs.occluders
            ),
            final_seed=(
                final_seed
            ),
            candidate_count=len(
                candidates
            ),
        )
    )

    budget_perception = (
        _observe_selected_pose(
            strategy_id=(
                StrategyId
                .BUDGET_MATCHED_GENERIC
            ),
            selected_pose=(
                budget_selected
                .candidate
                .pose
            ),
            initial_pose=(
                initial_pose
            ),
            observation_model=(
                inputs.observation_model
            ),
            true_structures=(
                inputs.true_structures
            ),
            occluders=(
                inputs.occluders
            ),
            final_seed=(
                final_seed
            ),
            candidate_count=len(
                candidates
            ),
        )
    )

    alignment_perception = (
        _observe_selected_pose(
            strategy_id=(
                StrategyId
                .ALIGNMENT_ONLY
            ),
            selected_pose=(
                alignment_generic
                .candidate
                .pose
            ),
            initial_pose=(
                initial_pose
            ),
            observation_model=(
                inputs.observation_model
            ),
            true_structures=(
                inputs.true_structures
            ),
            occluders=(
                inputs.occluders
            ),
            final_seed=(
                final_seed
            ),
            candidate_count=len(
                candidates
            ),
            task_relevance=(
                _mean_relevance(
                    alignment_task
                    .relevance_weights
                )
            ),
            task_alignment=float(
                alignment_task
                .task_alignment
            ),
        )
    )

    information_perception = (
        _observe_selected_pose(
            strategy_id=(
                StrategyId
                .INFORMATION_ONLY
            ),
            selected_pose=(
                information_task
                .candidate
                .pose
            ),
            initial_pose=(
                initial_pose
            ),
            observation_model=(
                inputs.observation_model
            ),
            true_structures=(
                inputs.true_structures
            ),
            occluders=(
                inputs.occluders
            ),
            final_seed=(
                final_seed
            ),
            candidate_count=len(
                candidates
            ),
            task_relevance=(
                _mean_relevance(
                    information_task
                    .relevance_weights
                )
            ),
            task_alignment=float(
                information_task
                .task_alignment
            ),
        )
    )

    full_perception = (
        _observe_selected_pose(
            strategy_id=(
                StrategyId
                .FULL_TASK_AWARE
            ),
            selected_pose=(
                full_selected
                .candidate
                .pose
            ),
            initial_pose=(
                initial_pose
            ),
            observation_model=(
                inputs.observation_model
            ),
            true_structures=(
                inputs.true_structures
            ),
            occluders=(
                inputs.occluders
            ),
            final_seed=(
                final_seed
            ),
            candidate_count=len(
                candidates
            ),
            task_relevance=(
                _mean_relevance(
                    full_selected
                    .relevance_weights
                )
            ),
            task_alignment=float(
                full_selected
                .task_alignment
            ),
        )
    )

    oracle_perception = (
        _observe_selected_pose(
            strategy_id=(
                StrategyId.ORACLE
            ),
            selected_pose=(
                oracle_selected
                .candidate
                .pose
            ),
            initial_pose=(
                initial_pose
            ),
            observation_model=(
                inputs.observation_model
            ),
            true_structures=(
                inputs.true_structures
            ),
            occluders=(
                inputs.occluders
            ),
            final_seed=(
                final_seed
            ),
            candidate_count=len(
                candidates
            ),
            task_relevance=(
                _mean_relevance(
                    oracle_selected
                    .relevance_weights
                )
            ),
            task_alignment=float(
                oracle_selected
                .task_alignment
            ),
        )
    )

    return (
        StrategyPerceptionBundle(
            strategy_id=(
                StrategyId.FIXED
            ),
            perception=(
                fixed_perception
            ),
            selected_candidate_index=None,
            movement_weight=None,
            selection_score=None,
        ),
        StrategyPerceptionBundle(
            strategy_id=(
                StrategyId
                .RANDOM_ACTIVE
            ),
            perception=(
                random_perception
            ),
            selected_candidate_index=int(
                random_index
            ),
            movement_weight=None,
            selection_score=None,
        ),
        StrategyPerceptionBundle(
            strategy_id=(
                StrategyId
                .GENERIC_ACTIVE
            ),
            perception=(
                generic_perception
            ),
            selected_candidate_index=int(
                generic_same_index
            ),
            movement_weight=float(
                FROZEN_TASK_MOVEMENT_WEIGHT
            ),
            selection_score=float(
                generic_same_selected
                .score
            ),
        ),
        StrategyPerceptionBundle(
            strategy_id=(
                StrategyId
                .BUDGET_MATCHED_GENERIC
            ),
            perception=(
                budget_perception
            ),
            selected_candidate_index=int(
                budget_index
            ),
            movement_weight=float(
                FROZEN_GENERIC_MOVEMENT_WEIGHT
            ),
            selection_score=float(
                budget_selected
                .score
            ),
        ),
        StrategyPerceptionBundle(
            strategy_id=(
                StrategyId
                .ALIGNMENT_ONLY
            ),
            perception=(
                alignment_perception
            ),
            selected_candidate_index=int(
                alignment_index
            ),
            movement_weight=float(
                FROZEN_TASK_MOVEMENT_WEIGHT
            ),
            selection_score=float(
                alignment_scores[
                    alignment_index
                ]
            ),
        ),
        StrategyPerceptionBundle(
            strategy_id=(
                StrategyId
                .INFORMATION_ONLY
            ),
            perception=(
                information_perception
            ),
            selected_candidate_index=int(
                information_index
            ),
            movement_weight=float(
                FROZEN_TASK_MOVEMENT_WEIGHT
            ),
            selection_score=float(
                information_scores[
                    information_index
                ]
            ),
        ),
        StrategyPerceptionBundle(
            strategy_id=(
                StrategyId
                .FULL_TASK_AWARE
            ),
            perception=(
                full_perception
            ),
            selected_candidate_index=int(
                full_index
            ),
            movement_weight=float(
                FROZEN_TASK_MOVEMENT_WEIGHT
            ),
            selection_score=float(
                full_selected
                .score
            ),
        ),
        StrategyPerceptionBundle(
            strategy_id=(
                StrategyId.ORACLE
            ),
            perception=(
                oracle_perception
            ),
            selected_candidate_index=int(
                oracle_index
            ),
            movement_weight=float(
                FROZEN_TASK_MOVEMENT_WEIGHT
            ),
            selection_score=float(
                oracle_selected
                .oracle_score
            ),
        ),
    )


def _record_from_navigation(
    *,
    scenario: RobustnessScenario,
    repetition: int,
    bundle: StrategyPerceptionBundle,
    navigation,
    seeds: dict[
        str,
        int,
    ],
) -> Phase1FinalRecord:
    """Convert one navigation result into the frozen raw-result schema."""

    planner_result = (
        navigation
        .planner_result
    )

    planning_success = bool(
        navigation.planning_success
    )

    collision = bool(
        navigation
        .collision_against_truth
    )

    violation = bool(
        navigation
        .safety_violation_against_truth
    )

    safe_navigation_success = bool(
        planning_success
        and not collision
        and not violation
    )

    return Phase1FinalRecord(
        scenario_id=int(
            scenario.scenario_id
        ),
        scenario_name=(
            scenario.name
        ),
        repetition=int(
            repetition
        ),
        strategy=(
            bundle
            .strategy_id
            .value
        ),
        initial_perception_seed=int(
            seeds[
                "initial_perception"
            ]
        ),
        final_perception_seed=int(
            seeds[
                "final_perception"
            ]
        ),
        random_viewpoint_seed=int(
            seeds[
                "random_viewpoint"
            ]
        ),
        planner_seed=int(
            seeds[
                "planner"
            ]
        ),
        selected_candidate_index=(
            bundle
            .selected_candidate_index
        ),
        movement_weight=(
            bundle
            .movement_weight
        ),
        camera_movement=float(
            bundle
            .perception
            .camera_movement
        ),
        mean_localisation_error=float(
            bundle
            .perception
            .mean_localisation_error
        ),
        mean_predicted_sigma=float(
            bundle
            .perception
            .mean_predicted_sigma
        ),
        task_relevance=(
            bundle
            .perception
            .task_relevance
        ),
        task_alignment=(
            bundle
            .perception
            .task_alignment
        ),
        planning_success=(
            planning_success
        ),
        collision_against_truth=(
            collision
        ),
        safety_violation_against_truth=(
            violation
        ),
        safe_navigation_success=(
            safe_navigation_success
        ),
        minimum_true_safety_clearance=float(
            navigation
            .minimum_true_safety_clearance
        ),
        mean_planning_safety_margin=float(
            navigation
            .mean_planning_safety_margin
        ),
        maximum_planning_safety_radius=float(
            navigation
            .maximum_planning_safety_radius
        ),
        planner_iterations=int(
            planner_result
            .iterations
        ),
        planning_time_seconds=float(
            planner_result
            .planning_time
        ),
        path_cost=float(
            planner_result
            .path_cost
        ),
        maximum_rcm_error=float(
            planner_result
            .maximum_rcm_error
        ),
    )


def run_eight_strategy_trial(
    *,
    scenario: RobustnessScenario,
    repetition: int,
) -> Phase1FinalTrial:
    """Run the complete perception-to-navigation pipeline for one trial."""

    protocol = (
        build_frozen_phase1_protocol()
    )

    seeds = (
        held_out_trial_seeds(
            protocol=protocol,
            scenario_id=(
                scenario.scenario_id
            ),
            repetition=(
                repetition
            ),
        )
    )

    inputs = (
        build_scenario_inputs(
            scenario
        )
    )

    bundles = (
        build_eight_strategy_perceptions(
            scenario=scenario,
            repetition=repetition,
        )
    )

    records: list[
        Phase1FinalRecord
    ] = []

    for bundle in bundles:
        navigation = (
            run_navigation_from_perception(
                trial=int(
                    repetition
                ),
                perception=(
                    bundle.perception
                ),
                instrument=(
                    inputs.instrument
                ),
                start_q=(
                    inputs.start_q
                ),
                goal_q=(
                    inputs.goal_q
                ),
                true_structures=(
                    inputs.true_structures
                ),
                sigma_multiplier=(
                    SIGMA_MULTIPLIER
                ),
                instrument_radius=(
                    INSTRUMENT_RADIUS
                ),
                proximal_length=(
                    PROXIMAL_LENGTH
                ),
                planner_seed=(
                    seeds[
                        "planner"
                    ]
                ),
            )
        )

        records.append(
            _record_from_navigation(
                scenario=scenario,
                repetition=repetition,
                bundle=bundle,
                navigation=navigation,
                seeds=seeds,
            )
        )

    return Phase1FinalTrial(
        scenario_id=int(
            scenario.scenario_id
        ),
        repetition=int(
            repetition
        ),
        records=tuple(
            records
        ),
    )


def _is_held_out_scenario(
    scenario: RobustnessScenario,
) -> bool:
    """Return whether a scenario belongs to the frozen held-out ID range."""

    return bool(
        HELD_OUT_ID_MIN
        <= scenario.scenario_id
        < HELD_OUT_ID_MAX_EXCLUSIVE
    )


def assert_execution_permission(
    *,
    scenarios: tuple[
        RobustnessScenario,
        ...,
    ],
    allow_held_out: bool,
) -> None:
    """Prevent accidental held-out execution."""

    held_out = tuple(
        scenario
        for scenario
        in scenarios
        if _is_held_out_scenario(
            scenario
        )
    )

    if (
        len(
            held_out
        )
        > 0
        and not allow_held_out
    ):
        raise PermissionError(
            "Held-out scenarios detected. "
            "Explicit held-out execution permission is required."
        )


def run_scenarios(
    *,
    scenarios: tuple[
        RobustnessScenario,
        ...,
    ],
    repetitions: int,
    allow_held_out: bool = False,
) -> tuple[
    Phase1FinalRecord,
    ...,
]:
    """Run all eight strategies across supplied scenarios."""

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

    assert_execution_permission(
        scenarios=scenarios,
        allow_held_out=(
            allow_held_out
        ),
    )

    records: list[
        Phase1FinalRecord
    ] = []

    for scenario in scenarios:
        for repetition in range(
            repetitions
        ):
            print(
                "Running "
                f"scenario={scenario.scenario_id} "
                f"({scenario.name}), "
                f"repetition={repetition}"
            )

            result = (
                run_eight_strategy_trial(
                    scenario=scenario,
                    repetition=repetition,
                )
            )

            records.extend(
                result.records
            )

    return tuple(
        records
    )


def _scenario_from_dict(
    payload: dict,
) -> RobustnessScenario:
    """Reconstruct one frozen scenario."""

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


def verify_frozen_protocol_file(
    path: str
    | Path,
) -> None:
    """Verify the saved frozen protocol before held-out execution."""

    path = Path(
        path
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(
            handle
        )

    stored_digest = payload.pop(
        "protocol_sha256",
        None,
    )

    if stored_digest is None:
        raise ValueError(
            "Frozen protocol file contains no SHA-256 digest."
        )

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        allow_nan=False,
    ).encode(
        "utf-8"
    )

    recomputed = hashlib.sha256(
        canonical
    ).hexdigest()

    if recomputed != stored_digest:
        raise ValueError(
            "Frozen protocol integrity check failed."
        )

    if (
        stored_digest
        != FROZEN_PROTOCOL_SHA256
    ):
        raise ValueError(
            "Protocol file does not match the frozen "
            "pre-held-out protocol."
        )

    if (
        payload[
            "scenario_manifest_sha256"
        ]
        != FROZEN_SCENARIO_MANIFEST_SHA256
    ):
        raise ValueError(
            "Protocol references the wrong scenario manifest."
        )


def load_frozen_held_out_scenarios(
    *,
    manifest_path: str
    | Path,
) -> tuple[
    RobustnessScenario,
    ...,
]:
    """Load the held-out split after checking manifest integrity."""

    manifest_path = Path(
        manifest_path
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

    if (
        stored_digest
        != FROZEN_SCENARIO_MANIFEST_SHA256
    ):
        raise ValueError(
            "Scenario manifest does not match the frozen digest."
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

    recomputed = (
        manifest_digest(
            splits=splits,
            config=config,
        )
    )

    if recomputed != stored_digest:
        raise ValueError(
            "Scenario-manifest integrity check failed."
        )

    if len(
        held_out
    ) != 30:
        raise ValueError(
            "Frozen held-out split must contain exactly 30 scenarios."
        )

    expected_ids = set(
        range(
            2000,
            2030,
        )
    )

    observed_ids = {
        scenario.scenario_id
        for scenario
        in held_out
    }

    if observed_ids != expected_ids:
        raise ValueError(
            "Held-out scenario identifiers do not match "
            "the frozen protocol."
        )

    return held_out


def save_records(
    records: tuple[
        Phase1FinalRecord,
        ...,
    ],
    *,
    output_path: str
    | Path,
    overwrite: bool = False,
) -> Path:
    """Save raw strategy-level records."""

    if len(
        records
    ) == 0:
        raise ValueError(
            "records must not be empty."
        )

    output_path = Path(
        output_path
    )

    if (
        output_path.exists()
        and not overwrite
    ):
        raise FileExistsError(
            f"Refusing to overwrite existing evidence: "
            f"{output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        asdict(
            records[
                0
            ]
        ).keys()
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                fieldnames
            ),
        )

        writer.writeheader()

        for record in records:
            writer.writerow(
                asdict(
                    record
                )
            )

    return output_path


def print_trial_summary(
    records: tuple[
        Phase1FinalRecord,
        ...,
    ],
) -> None:
    """Print compact smoke-test results."""

    print()

    print(
        "Phase 1 Eight-Strategy Trial"
    )

    print(
        "============================"
    )

    print()

    print(
        "Strategy                     "
        "Move(mm)  Error(mm) Sigma(mm) Plan  Safe"
    )

    print(
        "-" * 72
    )

    for record in records:
        print(
            f"{record.strategy:28s} "
            f"{record.camera_movement * 1000.0:8.3f} "
            f"{record.mean_localisation_error * 1000.0:9.3f} "
            f"{record.mean_predicted_sigma * 1000.0:9.3f} "
            f"{str(record.planning_success):5s} "
            f"{str(record.safe_navigation_success):5s}"
        )


def main() -> None:
    """CLI entry point with explicit held-out protection."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen eight-strategy Phase 1 experiment."
        )
    )

    mode = (
        parser
        .add_mutually_exclusive_group(
            required=True
        )
    )

    mode.add_argument(
        "--smoke-development",
        action="store_true",
        help=(
            "Run exactly one development scenario/repetition. "
            "Does not execute held-out data."
        ),
    )

    mode.add_argument(
        "--execute-held-out",
        action="store_true",
        help=(
            "Execute the complete frozen held-out experiment. "
            "Do not use until the runner and statistics code "
            "have been verified."
        ),
    )

    parser.add_argument(
        "--protocol",
        type=str,
        default=(
            "results/"
            "phase1_protocol/"
            "frozen_phase1_protocol.json"
        ),
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

    args = parser.parse_args()

    if args.smoke_development:
        scenario = (
            default_scenarios()[
                0
            ]
        )

        records = (
            run_scenarios(
                scenarios=(
                    scenario,
                ),
                repetitions=1,
                allow_held_out=False,
            )
        )

        print_trial_summary(
            records
        )

        output = save_records(
            records,
            output_path=(
                "results/"
                "phase1_smoke/"
                "eight_strategy_smoke.csv"
            ),
            overwrite=True,
        )

        print()

        print(
            f"Smoke-test evidence saved to: {output}"
        )

        return

    # --------------------------------------------------------------
    # Explicit held-out mode.
    # --------------------------------------------------------------

    verify_frozen_protocol_file(
        args.protocol
    )

    held_out = (
        load_frozen_held_out_scenarios(
            manifest_path=(
                args.manifest
            )
        )
    )

    protocol = (
        build_frozen_phase1_protocol()
    )

    records = (
        run_scenarios(
            scenarios=(
                held_out
            ),
            repetitions=(
                protocol
                .repetitions_per_scenario
            ),
            allow_held_out=True,
        )
    )

    expected_records = (
        30
        * 10
        * 8
    )

    if len(
        records
    ) != expected_records:
        raise RuntimeError(
            "Final held-out record count is incorrect. "
            f"Expected {expected_records}, "
            f"received {len(records)}."
        )

    output = save_records(
        records,
        output_path=(
            "results/"
            "phase1_held_out/"
            "held_out_raw_records.csv"
        ),
        overwrite=False,
    )

    print()

    print(
        "FINAL HELD-OUT EXECUTION COMPLETE"
    )

    print(
        f"Strategy evaluations: {len(records)}"
    )

    print(
        f"Raw evidence saved to: {output}"
    )


if __name__ == "__main__":
    main()