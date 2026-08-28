"""Frozen protocol for the final Phase 1 held-out experiment.

This module records the experimental design BEFORE any held-out scenario is
executed.

The protocol freezes:

- scenario manifest;
- strategy definitions;
- information-access rules;
- movement weights;
- random seeds;
- candidate-generation policy;
- primary comparison;
- primary and secondary outcomes;
- statistical analysis;
- interpretation rules;
- prohibition on post-held-out tuning.

Final scientific question
-------------------------
Given the same uncertain initial anatomical estimate and approximately equal
camera-motion expenditure, does adding knowledge of the intended surgical
trajectory improve downstream simulated navigation?

Primary comparison
------------------
Full Task-Aware
versus
Movement-Budget-Matched Generic

The held-out split contains 30 previously untouched procedurally generated
scenarios. Each scenario is repeated 10 times under matched stochastic
conditions.

Ground truth is permitted inside:

- the simulator observation generator;
- the privileged Oracle scoring baseline;
- final evaluation.

Ground truth is not exposed to deployable strategy decision logic.
"""

from __future__ import annotations

import argparse
import hashlib
import json

from dataclasses import (
    asdict,
    dataclass,
    is_dataclass,
)
from enum import Enum
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "phase1-final-v1.0"

FROZEN_SCENARIO_MANIFEST_SHA256 = (
    "490922fbc9f91743262257ff594a7440"
    "651016085207b3e8867b9e1bdce8ddd9"
)

FROZEN_GENERIC_MOVEMENT_WEIGHT = 0.072

FROZEN_TASK_MOVEMENT_WEIGHT = 0.200

HELD_OUT_SCENARIO_COUNT = 30

HELD_OUT_REPETITIONS = 10

ALPHA = 0.05

BOOTSTRAP_RESAMPLES = 10_000

PERMUTATION_RESAMPLES = 100_000


class StrategyId(str, Enum):
    """Frozen strategy identifiers."""

    FIXED = "fixed"

    RANDOM_ACTIVE = "random_active"

    GENERIC_ACTIVE = "generic_active"

    BUDGET_MATCHED_GENERIC = (
        "budget_matched_generic"
    )

    ALIGNMENT_ONLY = (
        "alignment_only"
    )

    INFORMATION_ONLY = (
        "information_only"
    )

    FULL_TASK_AWARE = (
        "full_task_aware"
    )

    ORACLE = "oracle"


@dataclass(frozen=True)
class StrategySpec:
    """Frozen definition of one experimental strategy."""

    strategy_id: StrategyId

    label: str

    role: str

    movement_weight: float | None

    receives_task_trajectory: bool

    uses_uniform_scene_information: bool

    uses_task_weighted_information: bool

    uses_task_alignment: bool

    uses_ground_truth_for_selection: bool

    uses_common_candidate_set: bool

    final_observation: bool

    description: str


@dataclass(frozen=True)
class InformationIsolationSpec:
    """Frozen information-access architecture."""

    initial_camera_geometry: str

    initial_pose_uses_ground_truth: bool

    initial_observation_is_shared: bool

    active_candidate_centre: str

    candidate_generation_uses_ground_truth: bool

    generic_receives_task_trajectory: bool

    task_aware_receives_task_trajectory: bool

    deployable_strategies_receive_ground_truth: bool

    oracle_receives_ground_truth: bool

    ground_truth_allowed_in_observation_generator: bool

    ground_truth_allowed_in_evaluator: bool

    fixed_receives_matched_final_observation: bool


@dataclass(frozen=True)
class SeedPlan:
    """Frozen deterministic seed allocation."""

    initial_perception_seed_base: int

    final_perception_seed_base: int

    random_viewpoint_seed_base: int

    planner_seed_base: int

    bootstrap_seed: int

    permutation_seed: int

    scenario_multiplier: int

    formula: str


@dataclass(frozen=True)
class OutcomeSpec:
    """Frozen experimental outcome definition."""

    name: str

    role: str

    direction: str

    aggregation: str

    inferential_method: str

    description: str


@dataclass(frozen=True)
class ComparisonSpec:
    """Frozen strategy comparison."""

    name: str

    role: str

    strategy_a: StrategyId

    strategy_b: StrategyId

    interpretation: str


@dataclass(frozen=True)
class StatisticalPlan:
    """Frozen statistical analysis."""

    inferential_unit: str

    alpha: float

    bootstrap_resamples: int

    permutation_resamples: int

    primary_test: str

    primary_confidence_interval: str

    primary_effect_measure: str

    superiority_rule: str

    secondary_multiplicity_rule: str

    paired_success_rule: str

    held_out_budget_rule: str


@dataclass(frozen=True)
class FrozenPhase1Protocol:
    """Complete immutable Phase 1 held-out protocol."""

    protocol_version: str

    frozen_date: str

    research_question: str

    scenario_manifest_sha256: str

    held_out_scenario_count: int

    repetitions_per_scenario: int

    total_paired_trials: int

    development_task_weight: float

    development_budget_matched_generic_weight: float

    corrected_development_budget_gap_fraction: float

    corrected_validation_budget_gap_fraction: float

    corrected_validation_gate_passed: bool

    information_isolation: InformationIsolationSpec

    seeds: SeedPlan

    strategies: tuple[
        StrategySpec,
        ...,
    ]

    comparisons: tuple[
        ComparisonSpec,
        ...,
    ]

    outcomes: tuple[
        OutcomeSpec,
        ...,
    ]

    statistics: StatisticalPlan

    tuning_rule: str

    interpretation_rule: str


def build_frozen_phase1_protocol(
) -> FrozenPhase1Protocol:
    """Construct the pre-held-out Phase 1 protocol."""

    information_isolation = (
        InformationIsolationSpec(
            initial_camera_geometry=(
                "scenario-independent nominal workspace prior"
            ),
            initial_pose_uses_ground_truth=False,
            initial_observation_is_shared=True,
            active_candidate_centre=(
                "shared noisy estimated target centre"
            ),
            candidate_generation_uses_ground_truth=False,
            generic_receives_task_trajectory=False,
            task_aware_receives_task_trajectory=True,
            deployable_strategies_receive_ground_truth=False,
            oracle_receives_ground_truth=True,
            ground_truth_allowed_in_observation_generator=True,
            ground_truth_allowed_in_evaluator=True,
            fixed_receives_matched_final_observation=True,
        )
    )

    seeds = (
        SeedPlan(
            initial_perception_seed_base=(
                60263000
            ),
            final_perception_seed_base=(
                70263000
            ),
            random_viewpoint_seed_base=(
                80263000
            ),
            planner_seed_base=(
                90263000
            ),
            bootstrap_seed=(
                202608281
            ),
            permutation_seed=(
                202608282
            ),
            scenario_multiplier=100,
            formula=(
                "seed_base + scenario_id * 100 + repetition"
            ),
        )
    )

    strategies = (
        StrategySpec(
            strategy_id=(
                StrategyId.FIXED
            ),
            label=(
                "Fixed View"
            ),
            role=(
                "passive baseline"
            ),
            movement_weight=None,
            receives_task_trajectory=False,
            uses_uniform_scene_information=False,
            uses_task_weighted_information=False,
            uses_task_alignment=False,
            uses_ground_truth_for_selection=False,
            uses_common_candidate_set=False,
            final_observation=True,
            description=(
                "Camera remains at the common initial pose. "
                "A matched final observation is generated at "
                "that same pose using the common final seed."
            ),
        ),
        StrategySpec(
            strategy_id=(
                StrategyId.RANDOM_ACTIVE
            ),
            label=(
                "Random Active"
            ),
            role=(
                "task-agnostic active baseline"
            ),
            movement_weight=None,
            receives_task_trajectory=False,
            uses_uniform_scene_information=False,
            uses_task_weighted_information=False,
            uses_task_alignment=False,
            uses_ground_truth_for_selection=False,
            uses_common_candidate_set=True,
            final_observation=True,
            description=(
                "Uniform random selection from the exact same "
                "estimate-centred candidate set."
            ),
        ),
        StrategySpec(
            strategy_id=(
                StrategyId.GENERIC_ACTIVE
            ),
            label=(
                "Generic Active"
            ),
            role=(
                "same-weight task-agnostic baseline"
            ),
            movement_weight=(
                FROZEN_TASK_MOVEMENT_WEIGHT
            ),
            receives_task_trajectory=False,
            uses_uniform_scene_information=True,
            uses_task_weighted_information=False,
            uses_task_alignment=False,
            uses_ground_truth_for_selection=False,
            uses_common_candidate_set=True,
            final_observation=True,
            description=(
                "Uniform scene-wide predicted-information objective "
                "with the same movement coefficient as Full "
                "Task-Aware."
            ),
        ),
        StrategySpec(
            strategy_id=(
                StrategyId
                .BUDGET_MATCHED_GENERIC
            ),
            label=(
                "Movement-Budget-Matched Generic"
            ),
            role=(
                "primary comparator"
            ),
            movement_weight=(
                FROZEN_GENERIC_MOVEMENT_WEIGHT
            ),
            receives_task_trajectory=False,
            uses_uniform_scene_information=True,
            uses_task_weighted_information=False,
            uses_task_alignment=False,
            uses_ground_truth_for_selection=False,
            uses_common_candidate_set=True,
            final_observation=True,
            description=(
                "Scene-wide Generic Active Perception with the "
                "development-frozen movement coefficient selected "
                "to approximately match Full Task-Aware camera "
                "movement."
            ),
        ),
        StrategySpec(
            strategy_id=(
                StrategyId.ALIGNMENT_ONLY
            ),
            label=(
                "Alignment-Only"
            ),
            role=(
                "mechanism ablation"
            ),
            movement_weight=(
                FROZEN_TASK_MOVEMENT_WEIGHT
            ),
            receives_task_trajectory=True,
            uses_uniform_scene_information=True,
            uses_task_weighted_information=False,
            uses_task_alignment=True,
            uses_ground_truth_for_selection=False,
            uses_common_candidate_set=True,
            final_observation=True,
            description=(
                "Uniform scene information plus task-derived "
                "camera alignment. This isolates the alignment "
                "contribution."
            ),
        ),
        StrategySpec(
            strategy_id=(
                StrategyId.INFORMATION_ONLY
            ),
            label=(
                "Task-Weighted Information-Only"
            ),
            role=(
                "mechanism ablation"
            ),
            movement_weight=(
                FROZEN_TASK_MOVEMENT_WEIGHT
            ),
            receives_task_trajectory=True,
            uses_uniform_scene_information=False,
            uses_task_weighted_information=True,
            uses_task_alignment=False,
            uses_ground_truth_for_selection=False,
            uses_common_candidate_set=True,
            final_observation=True,
            description=(
                "Task-weighted predicted scene information "
                "without the task-alignment reward."
            ),
        ),
        StrategySpec(
            strategy_id=(
                StrategyId.FULL_TASK_AWARE
            ),
            label=(
                "Full Task-Aware"
            ),
            role=(
                "proposed method"
            ),
            movement_weight=(
                FROZEN_TASK_MOVEMENT_WEIGHT
            ),
            receives_task_trajectory=True,
            uses_uniform_scene_information=False,
            uses_task_weighted_information=True,
            uses_task_alignment=True,
            uses_ground_truth_for_selection=False,
            uses_common_candidate_set=True,
            final_observation=True,
            description=(
                "Task-weighted predicted information plus "
                "task-derived viewpoint alignment."
            ),
        ),
        StrategySpec(
            strategy_id=(
                StrategyId.ORACLE
            ),
            label=(
                "Privileged Oracle"
            ),
            role=(
                "analysis-only upper reference"
            ),
            movement_weight=(
                FROZEN_TASK_MOVEMENT_WEIGHT
            ),
            receives_task_trajectory=True,
            uses_uniform_scene_information=False,
            uses_task_weighted_information=True,
            uses_task_alignment=True,
            uses_ground_truth_for_selection=True,
            uses_common_candidate_set=True,
            final_observation=True,
            description=(
                "Analysis-only privileged reference. Uses true "
                "anatomical and occluder geometry for scoring but "
                "must select from the same estimate-centred "
                "candidate set as every active strategy."
            ),
        ),
    )

    comparisons = (
        ComparisonSpec(
            name=(
                "primary_task_vs_budget_matched_generic"
            ),
            role=(
                "primary"
            ),
            strategy_a=(
                StrategyId.FULL_TASK_AWARE
            ),
            strategy_b=(
                StrategyId
                .BUDGET_MATCHED_GENERIC
            ),
            interpretation=(
                "Tests whether task information improves simulated "
                "navigation when camera-motion expenditure is "
                "approximately matched."
            ),
        ),
        ComparisonSpec(
            name=(
                "task_vs_same_weight_generic"
            ),
            role=(
                "secondary"
            ),
            strategy_a=(
                StrategyId.FULL_TASK_AWARE
            ),
            strategy_b=(
                StrategyId.GENERIC_ACTIVE
            ),
            interpretation=(
                "Tests task-aware scoring against Generic using "
                "the identical movement coefficient."
            ),
        ),
        ComparisonSpec(
            name=(
                "task_vs_fixed"
            ),
            role=(
                "secondary"
            ),
            strategy_a=(
                StrategyId.FULL_TASK_AWARE
            ),
            strategy_b=(
                StrategyId.FIXED
            ),
            interpretation=(
                "Measures benefit relative to no active camera "
                "repositioning."
            ),
        ),
        ComparisonSpec(
            name=(
                "task_vs_random"
            ),
            role=(
                "secondary"
            ),
            strategy_a=(
                StrategyId.FULL_TASK_AWARE
            ),
            strategy_b=(
                StrategyId.RANDOM_ACTIVE
            ),
            interpretation=(
                "Measures benefit beyond arbitrary active "
                "camera repositioning."
            ),
        ),
        ComparisonSpec(
            name=(
                "mechanism_alignment"
            ),
            role=(
                "ablation"
            ),
            strategy_a=(
                StrategyId.FULL_TASK_AWARE
            ),
            strategy_b=(
                StrategyId.ALIGNMENT_ONLY
            ),
            interpretation=(
                "Assesses whether task-weighted information adds "
                "benefit beyond alignment."
            ),
        ),
        ComparisonSpec(
            name=(
                "mechanism_information"
            ),
            role=(
                "ablation"
            ),
            strategy_a=(
                StrategyId.FULL_TASK_AWARE
            ),
            strategy_b=(
                StrategyId.INFORMATION_ONLY
            ),
            interpretation=(
                "Assesses whether alignment adds benefit beyond "
                "task-weighted predicted information."
            ),
        ),
        ComparisonSpec(
            name=(
                "task_vs_oracle"
            ),
            role=(
                "reference"
            ),
            strategy_a=(
                StrategyId.FULL_TASK_AWARE
            ),
            strategy_b=(
                StrategyId.ORACLE
            ),
            interpretation=(
                "Estimates the remaining gap to a privileged "
                "true-geometry reference. Oracle is not deployable."
            ),
        ),
    )

    outcomes = (
        OutcomeSpec(
            name=(
                "safe_navigation_success_rate"
            ),
            role=(
                "primary"
            ),
            direction=(
                "higher_is_better"
            ),
            aggregation=(
                "success proportion within each held-out scenario "
                "across repetitions"
            ),
            inferential_method=(
                "paired scenario-level sign-flip permutation test "
                "plus scenario-cluster bootstrap confidence interval"
            ),
            description=(
                "Primary endpoint for downstream simulated "
                "navigation."
            ),
        ),
        OutcomeSpec(
            name=(
                "planning_success_rate"
            ),
            role=(
                "key_secondary"
            ),
            direction=(
                "higher_is_better"
            ),
            aggregation=(
                "planning-success proportion within each scenario"
            ),
            inferential_method=(
                "paired scenario-level sign-flip permutation test "
                "with secondary-family multiplicity control"
            ),
            description=(
                "Whether uncertainty-aware motion planning finds "
                "a valid plan."
            ),
        ),
        OutcomeSpec(
            name=(
                "mean_localisation_error_mm"
            ),
            role=(
                "secondary"
            ),
            direction=(
                "lower_is_better"
            ),
            aggregation=(
                "mean within scenario, then paired across scenarios"
            ),
            inferential_method=(
                "scenario-cluster bootstrap paired difference"
            ),
            description=(
                "Realised anatomical localisation error."
            ),
        ),
        OutcomeSpec(
            name=(
                "mean_predicted_sigma_mm"
            ),
            role=(
                "secondary"
            ),
            direction=(
                "lower_is_better"
            ),
            aggregation=(
                "mean within scenario, then paired across scenarios"
            ),
            inferential_method=(
                "scenario-cluster bootstrap paired difference"
            ),
            description=(
                "Predicted positional uncertainty."
            ),
        ),
        OutcomeSpec(
            name=(
                "camera_movement_mm"
            ),
            role=(
                "fairness_diagnostic"
            ),
            direction=(
                "lower_is_better"
            ),
            aggregation=(
                "mean within scenario, then paired across scenarios"
            ),
            inferential_method=(
                "descriptive paired difference and confidence interval"
            ),
            description=(
                "Held-out movement-budget diagnostic. No weight "
                "may be retuned based on this result."
            ),
        ),
        OutcomeSpec(
            name=(
                "task_alignment"
            ),
            role=(
                "mechanism_diagnostic"
            ),
            direction=(
                "higher_is_better"
            ),
            aggregation=(
                "mean within scenario"
            ),
            inferential_method=(
                "descriptive"
            ),
            description=(
                "Post-selection task-alignment diagnostic."
            ),
        ),
        OutcomeSpec(
            name=(
                "paired_success_path_cost"
            ),
            role=(
                "secondary"
            ),
            direction=(
                "lower_is_better"
            ),
            aggregation=(
                "evaluated only where both members of the paired "
                "comparison produced successful plans"
            ),
            inferential_method=(
                "scenario-cluster bootstrap paired difference"
            ),
            description=(
                "Conditional path-efficiency comparison without "
                "giving failed strategies an artificial path cost."
            ),
        ),
        OutcomeSpec(
            name=(
                "collision_rate"
            ),
            role=(
                "safety_diagnostic"
            ),
            direction=(
                "lower_is_better"
            ),
            aggregation=(
                "rate within scenario"
            ),
            inferential_method=(
                "descriptive"
            ),
            description=(
                "Simulation collision outcome. No clinical safety "
                "claim is permitted."
            ),
        ),
        OutcomeSpec(
            name=(
                "safety_violation_rate"
            ),
            role=(
                "safety_diagnostic"
            ),
            direction=(
                "lower_is_better"
            ),
            aggregation=(
                "rate within scenario"
            ),
            inferential_method=(
                "descriptive"
            ),
            description=(
                "Simulation-defined safety-margin violation rate."
            ),
        ),
    )

    statistics = (
        StatisticalPlan(
            inferential_unit=(
                "held-out scenario; repetitions are nested within "
                "scenario and are not treated as independent "
                "experimental units"
            ),
            alpha=ALPHA,
            bootstrap_resamples=(
                BOOTSTRAP_RESAMPLES
            ),
            permutation_resamples=(
                PERMUTATION_RESAMPLES
            ),
            primary_test=(
                "two-sided paired scenario-level sign-flip "
                "permutation test on Full Task-Aware minus "
                "Movement-Budget-Matched Generic safe-navigation "
                "success rate"
            ),
            primary_confidence_interval=(
                "95% scenario-cluster bootstrap interval for the "
                "paired mean scenario-level success-rate difference"
            ),
            primary_effect_measure=(
                "absolute safe-navigation success-rate difference: "
                "Full Task-Aware minus Movement-Budget-Matched Generic"
            ),
            superiority_rule=(
                "Claim statistical superiority only if the primary "
                "two-sided p-value is below 0.05 AND the 95% "
                "confidence interval for the Task-Aware minus "
                "matched-Generic effect lies entirely above zero."
            ),
            secondary_multiplicity_rule=(
                "Any inferential p-values across secondary outcomes "
                "must use Holm correction within the secondary "
                "family. Descriptive diagnostics are not converted "
                "into post-hoc significance claims."
            ),
            paired_success_rule=(
                "Path cost is compared only for scenario/repetition "
                "pairs where both compared strategies produced a "
                "successful plan."
            ),
            held_out_budget_rule=(
                "Report held-out camera-movement mismatch exactly as "
                "observed. Do not retune either movement weight after "
                "held-out execution. If mismatch exceeds the previous "
                "10% validation tolerance, retain all results but "
                "flag movement matching as a limitation."
            ),
        )
    )

    protocol = FrozenPhase1Protocol(
        protocol_version=(
            PROTOCOL_VERSION
        ),
        frozen_date=(
            "2026-08-28"
        ),
        research_question=(
            "Given the same uncertain initial anatomical estimate "
            "and approximately equal camera-motion expenditure, "
            "does incorporating the intended surgical trajectory "
            "improve downstream simulated navigation?"
        ),
        scenario_manifest_sha256=(
            FROZEN_SCENARIO_MANIFEST_SHA256
        ),
        held_out_scenario_count=(
            HELD_OUT_SCENARIO_COUNT
        ),
        repetitions_per_scenario=(
            HELD_OUT_REPETITIONS
        ),
        total_paired_trials=(
            HELD_OUT_SCENARIO_COUNT
            * HELD_OUT_REPETITIONS
        ),
        development_task_weight=(
            FROZEN_TASK_MOVEMENT_WEIGHT
        ),
        development_budget_matched_generic_weight=(
            FROZEN_GENERIC_MOVEMENT_WEIGHT
        ),
        corrected_development_budget_gap_fraction=(
            0.0030
        ),
        corrected_validation_budget_gap_fraction=(
            0.0887
        ),
        corrected_validation_gate_passed=True,
        information_isolation=(
            information_isolation
        ),
        seeds=seeds,
        strategies=strategies,
        comparisons=comparisons,
        outcomes=outcomes,
        statistics=statistics,
        tuning_rule=(
            "No strategy definition, weight, candidate-generation "
            "rule, random-seed rule, endpoint, statistical test or "
            "interpretation threshold may be changed in response to "
            "held-out results."
        ),
        interpretation_rule=(
            "If the primary superiority criterion is not met, report "
            "that the held-out experiment did not establish superiority. "
            "Do not substitute a favourable secondary or subgroup "
            "result as the primary conclusion."
        ),
    )

    validate_frozen_protocol(
        protocol
    )

    return protocol


def _strategy_map(
    protocol: FrozenPhase1Protocol,
) -> dict[
    StrategyId,
    StrategySpec,
]:
    """Return strategies indexed by identifier."""

    return {
        strategy.strategy_id:
        strategy
        for strategy
        in protocol.strategies
    }


def validate_frozen_protocol(
    protocol: FrozenPhase1Protocol,
) -> None:
    """Assert the key examiner-proof protocol invariants."""

    if (
        protocol
        .scenario_manifest_sha256
        != FROZEN_SCENARIO_MANIFEST_SHA256
    ):
        raise ValueError(
            "Scenario manifest hash is not the frozen Phase 1 hash."
        )

    if (
        protocol.held_out_scenario_count
        != 30
    ):
        raise ValueError(
            "Held-out scenario count must remain 30."
        )

    if (
        protocol.repetitions_per_scenario
        != 10
    ):
        raise ValueError(
            "Held-out repetitions must remain 10."
        )

    if (
        protocol.total_paired_trials
        != 300
    ):
        raise ValueError(
            "Expected 300 scenario/repetition trials."
        )

    strategies = (
        _strategy_map(
            protocol
        )
    )

    expected_ids = set(
        StrategyId
    )

    if set(
        strategies
    ) != expected_ids:
        raise ValueError(
            "Frozen strategy family is incomplete."
        )

    if (
        strategies[
            StrategyId
            .BUDGET_MATCHED_GENERIC
        ]
        .movement_weight
        != FROZEN_GENERIC_MOVEMENT_WEIGHT
    ):
        raise ValueError(
            "Budget-matched Generic weight changed."
        )

    if (
        strategies[
            StrategyId
            .FULL_TASK_AWARE
        ]
        .movement_weight
        != FROZEN_TASK_MOVEMENT_WEIGHT
    ):
        raise ValueError(
            "Full Task-Aware weight changed."
        )

    if (
        strategies[
            StrategyId
            .GENERIC_ACTIVE
        ]
        .movement_weight
        != FROZEN_TASK_MOVEMENT_WEIGHT
    ):
        raise ValueError(
            "Same-weight Generic must use the Task-Aware "
            "movement coefficient."
        )

    if (
        strategies[
            StrategyId
            .BUDGET_MATCHED_GENERIC
        ]
        .receives_task_trajectory
    ):
        raise ValueError(
            "Generic comparator must remain task agnostic."
        )

    if (
        protocol
        .information_isolation
        .candidate_generation_uses_ground_truth
    ):
        raise ValueError(
            "Candidate generation must not use simulator truth."
        )

    if (
        protocol
        .information_isolation
        .initial_pose_uses_ground_truth
    ):
        raise ValueError(
            "Initial camera pose must not use simulator truth."
        )

    if (
        protocol
        .information_isolation
        .deployable_strategies_receive_ground_truth
    ):
        raise ValueError(
            "Deployable strategies must not receive ground truth."
        )

    truth_strategies = tuple(
        strategy.strategy_id
        for strategy
        in protocol.strategies
        if strategy
        .uses_ground_truth_for_selection
    )

    if truth_strategies != (
        StrategyId.ORACLE,
    ):
        raise ValueError(
            "Only the Oracle may use ground truth for selection."
        )

    primary_comparisons = tuple(
        comparison
        for comparison
        in protocol.comparisons
        if comparison.role
        == "primary"
    )

    if len(
        primary_comparisons
    ) != 1:
        raise ValueError(
            "Exactly one primary comparison must be frozen."
        )

    primary = (
        primary_comparisons[
            0
        ]
    )

    if (
        primary.strategy_a
        != StrategyId
        .FULL_TASK_AWARE
        or primary.strategy_b
        != StrategyId
        .BUDGET_MATCHED_GENERIC
    ):
        raise ValueError(
            "Primary comparison changed."
        )

    primary_outcomes = tuple(
        outcome
        for outcome
        in protocol.outcomes
        if outcome.role
        == "primary"
    )

    if len(
        primary_outcomes
    ) != 1:
        raise ValueError(
            "Exactly one primary endpoint must be frozen."
        )

    if (
        primary_outcomes[
            0
        ]
        .name
        != "safe_navigation_success_rate"
    ):
        raise ValueError(
            "Primary endpoint changed."
        )

    if (
        protocol.statistics.alpha
        != 0.05
    ):
        raise ValueError(
            "Primary alpha must remain 0.05."
        )

    if (
        protocol.statistics
        .bootstrap_resamples
        != 10_000
    ):
        raise ValueError(
            "Bootstrap resample count changed."
        )

    if (
        protocol.statistics
        .permutation_resamples
        != 100_000
    ):
        raise ValueError(
            "Permutation resample count changed."
        )


def trial_seed(
    *,
    seed_base: int,
    scenario_id: int,
    repetition: int,
    scenario_multiplier: int = 100,
) -> int:
    """Return one deterministic trial seed."""

    if scenario_id < 0:
        raise ValueError(
            "scenario_id must be non-negative."
        )

    if repetition < 0:
        raise ValueError(
            "repetition must be non-negative."
        )

    if repetition >= scenario_multiplier:
        raise ValueError(
            "repetition must be smaller than "
            "scenario_multiplier."
        )

    return int(
        seed_base
        + scenario_id
        * scenario_multiplier
        + repetition
    )


def held_out_trial_seeds(
    *,
    protocol: FrozenPhase1Protocol,
    scenario_id: int,
    repetition: int,
) -> dict[
    str,
    int,
]:
    """Return all frozen seeds for one held-out trial."""

    seed_plan = (
        protocol.seeds
    )

    return {
        "initial_perception": (
            trial_seed(
                seed_base=(
                    seed_plan
                    .initial_perception_seed_base
                ),
                scenario_id=(
                    scenario_id
                ),
                repetition=(
                    repetition
                ),
                scenario_multiplier=(
                    seed_plan
                    .scenario_multiplier
                ),
            )
        ),
        "final_perception": (
            trial_seed(
                seed_base=(
                    seed_plan
                    .final_perception_seed_base
                ),
                scenario_id=(
                    scenario_id
                ),
                repetition=(
                    repetition
                ),
                scenario_multiplier=(
                    seed_plan
                    .scenario_multiplier
                ),
            )
        ),
        "random_viewpoint": (
            trial_seed(
                seed_base=(
                    seed_plan
                    .random_viewpoint_seed_base
                ),
                scenario_id=(
                    scenario_id
                ),
                repetition=(
                    repetition
                ),
                scenario_multiplier=(
                    seed_plan
                    .scenario_multiplier
                ),
            )
        ),
        "planner": (
            trial_seed(
                seed_base=(
                    seed_plan
                    .planner_seed_base
                ),
                scenario_id=(
                    scenario_id
                ),
                repetition=(
                    repetition
                ),
                scenario_multiplier=(
                    seed_plan
                    .scenario_multiplier
                ),
            )
        ),
    }


def _to_jsonable(
    value: Any,
) -> Any:
    """Convert nested protocol values into canonical JSON data."""

    if isinstance(
        value,
        Enum,
    ):
        return value.value

    if is_dataclass(
        value
    ):
        return {
            key: _to_jsonable(
                item
            )
            for key, item
            in asdict(
                value
            ).items()
        }

    if isinstance(
        value,
        dict,
    ):
        return {
            str(
                key
            ):
            _to_jsonable(
                item
            )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (
            tuple,
            list,
        ),
    ):
        return [
            _to_jsonable(
                item
            )
            for item
            in value
        ]

    return value


def protocol_payload(
    protocol: FrozenPhase1Protocol,
) -> dict:
    """Return canonical protocol payload."""

    payload = (
        _to_jsonable(
            protocol
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(
            "Protocol payload must be a dictionary."
        )

    return payload


def protocol_digest(
    protocol: FrozenPhase1Protocol,
) -> str:
    """Return SHA-256 digest of the complete frozen protocol."""

    payload = (
        protocol_payload(
            protocol
        )
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

    return hashlib.sha256(
        canonical
    ).hexdigest()


def save_protocol(
    protocol: FrozenPhase1Protocol,
    *,
    output_path: str
    | Path = (
        "results/"
        "phase1_protocol/"
        "frozen_phase1_protocol.json"
    ),
) -> tuple[
    Path,
    str,
]:
    """Save the frozen protocol plus a cryptographic digest."""

    validate_frozen_protocol(
        protocol
    )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    digest = (
        protocol_digest(
            protocol
        )
    )

    payload = (
        protocol_payload(
            protocol
        )
    )

    payload[
        "protocol_sha256"
    ] = digest

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

    return (
        output_path,
        digest,
    )


def print_protocol(
    protocol: FrozenPhase1Protocol,
    *,
    digest: str,
) -> None:
    """Print the most important frozen decisions."""

    print()

    print(
        "Frozen Phase 1 Held-Out Protocol"
    )

    print(
        "================================"
    )

    print()

    print(
        f"Protocol version:           "
        f"{protocol.protocol_version}"
    )

    print(
        f"Held-out scenarios:         "
        f"{protocol.held_out_scenario_count}"
    )

    print(
        f"Repetitions per scenario:   "
        f"{protocol.repetitions_per_scenario}"
    )

    print(
        f"Total paired trials:        "
        f"{protocol.total_paired_trials}"
    )

    print()

    print(
        "Frozen movement weights"
    )

    print(
        f"  Budget-Matched Generic:   "
        f"{protocol.development_budget_matched_generic_weight:.3f}"
    )

    print(
        f"  Full Task-Aware:          "
        f"{protocol.development_task_weight:.3f}"
    )

    print()

    print(
        "Primary comparison:"
    )

    print(
        "  Full Task-Aware"
    )

    print(
        "        versus"
    )

    print(
        "  Movement-Budget-Matched Generic"
    )

    print()

    print(
        "Primary endpoint:"
    )

    print(
        "  safe_navigation_success_rate"
    )

    print()

    print(
        "Inferential unit:"
    )

    print(
        "  held-out scenario"
    )

    print()

    print(
        "Scenario manifest SHA-256:"
    )

    print(
        f"  {protocol.scenario_manifest_sha256}"
    )

    print()

    print(
        "Frozen protocol SHA-256:"
    )

    print(
        f"  {digest}"
    )

    print()

    print(
        "POST-HELD-OUT TUNING PERMITTED: NO"
    )


def main() -> None:
    """Freeze the complete Phase 1 held-out protocol."""

    parser = argparse.ArgumentParser(
        description=(
            "Build and save the examiner-proof Phase 1 "
            "pre-held-out protocol."
        )
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "results/"
            "phase1_protocol/"
            "frozen_phase1_protocol.json"
        ),
    )

    args = parser.parse_args()

    protocol = (
        build_frozen_phase1_protocol()
    )

    output, digest = (
        save_protocol(
            protocol,
            output_path=(
                args.output
            ),
        )
    )

    print_protocol(
        protocol,
        digest=digest,
    )

    print()

    print(
        f"Protocol saved to: {output}"
    )


if __name__ == "__main__":
    main()