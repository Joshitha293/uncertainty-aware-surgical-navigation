"""E7 uncertainty-term stress and sensitivity benchmark.

E6 showed that alignment drives the observed task-aware advantage under the
standard benchmark, while uncertainty-only behaves like the generic baseline.

E7 asks whether this is because the explicit uncertainty term is redundant
under the current observation model, or whether sufficiently different
uncertainty regimes cause it to alter viewpoint selection.

The benchmark has two stages:

1. A cheap deterministic viewpoint-selection sweep across uncertainty-model
   stress profiles and uncertainty weights.

2. A focused end-to-end confirmation comparing:
       Generic baseline
       High-weight uncertainty only
       Alignment only
       Full task-aware

The end-to-end stage uses matched seeds and hidden-ground-truth navigation
evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from src.perception.observation import (
    ViewpointObservationModel,
)
from src.perception.task_aware_active_perception import (
    TaskAwareActivePerception,
)
from src.perception.task_aware_scoring import (
    TaskAwareScoringConfig,
    TaskAwareViewpointScorer,
)
from src.perception.viewpoint_scoring import (
    GenericViewpointScorer,
)
from src.simulation.three_strategy_navigation import (
    run_navigation_from_perception,
)
from src.simulation.three_strategy_perception import (
    run_task_aware_active_perception,
)
from src.simulation.three_strategy_robustness_benchmark import (
    RobustnessScenario,
    ScenarioInputs,
    build_scenario_inputs,
    default_scenarios,
)


@dataclass(frozen=True)
class UncertaintyStressProfile:
    """Controlled modification of viewpoint-dependent uncertainty."""

    name: str

    distance_multiplier: float = 1.0
    angle_multiplier: float = 1.0

    occluded_sigma_multiplier: float = 1.0
    invisible_sigma_multiplier: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.distance_multiplier,
            self.angle_multiplier,
            self.occluded_sigma_multiplier,
            self.invisible_sigma_multiplier,
        )

        if any(
            value <= 0.0
            for value in values
        ):
            raise ValueError(
                "All stress-profile multipliers must be positive."
            )


@dataclass(frozen=True)
class StressVariant:
    """One end-to-end E7 scoring configuration."""

    name: str

    alignment_weight: float
    uncertainty_weight: float

    def __post_init__(self) -> None:
        if self.alignment_weight < 0.0:
            raise ValueError(
                "alignment_weight must be non-negative."
            )

        if self.uncertainty_weight < 0.0:
            raise ValueError(
                "uncertainty_weight must be non-negative."
            )


@dataclass(frozen=True)
class UncertaintyStressConfig:
    """Configuration for E7."""

    repetitions: int = 2

    perception_seed_base: int = 20260920
    planner_seed_base: int = 9000

    task_weight: float = 2.0

    sigma_multiplier: float = 2.0

    instrument_radius: float = 0.006
    proximal_length: float = 0.10

    uncertainty_weights: tuple[
        float,
        ...,
    ] = (
        0.0,
        0.25,
        1.0,
        4.0,
        16.0,
    )

    confidence_level: float = 0.95

    bootstrap_samples: int = 1000
    bootstrap_seed: int = 20260921

    def __post_init__(self) -> None:
        if self.repetitions <= 0:
            raise ValueError(
                "repetitions must be positive."
            )

        if self.task_weight < 0.0:
            raise ValueError(
                "task_weight must be non-negative."
            )

        if self.sigma_multiplier < 0.0:
            raise ValueError(
                "sigma_multiplier must be non-negative."
            )

        if self.instrument_radius <= 0.0:
            raise ValueError(
                "instrument_radius must be positive."
            )

        if self.proximal_length <= 0.0:
            raise ValueError(
                "proximal_length must be positive."
            )

        if len(
            self.uncertainty_weights
        ) == 0:
            raise ValueError(
                "uncertainty_weights must not be empty."
            )

        if any(
            weight < 0.0
            for weight
            in self.uncertainty_weights
        ):
            raise ValueError(
                "uncertainty weights must be non-negative."
            )

        if not (
            0.0
            < self.confidence_level
            < 1.0
        ):
            raise ValueError(
                "confidence_level must lie between 0 and 1."
            )

        if self.bootstrap_samples <= 0:
            raise ValueError(
                "bootstrap_samples must be positive."
            )


@dataclass(frozen=True)
class SelectionSweepRecord:
    """One deterministic viewpoint-selection sensitivity result."""

    scenario_id: int
    scenario_name: str

    stress_profile: str

    uncertainty_weight: float

    generic_candidate_index: int
    selected_candidate_index: int

    differs_from_generic: bool

    selected_predicted_sigma: float

    generic_predicted_sigma: float


@dataclass(frozen=True)
class StressTrialRecord:
    """One end-to-end E7 result."""

    scenario_id: int
    scenario_name: str

    stress_profile: str

    repetition: int

    variant: str

    perception_seed: int
    planner_seed: int

    selected_candidate_index: int

    mean_localisation_error: float
    mean_predicted_sigma: float

    camera_movement: float

    planning_success: bool

    collision_against_truth: bool
    safety_violation_against_truth: bool

    safe_navigation_success: bool


@dataclass(frozen=True)
class MetricEstimate:
    """Mean and scenario-clustered confidence interval."""

    n: int

    mean: float

    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class SelectionSensitivitySummary:
    """Selection response at one explicit uncertainty weight."""

    uncertainty_weight: float

    condition_count: int

    selection_change_rate: float

    mean_sigma_change: float


@dataclass(frozen=True)
class StressVariantSummary:
    """Aggregate end-to-end result for one E7 variant."""

    variant: str

    record_count: int

    localisation_error: MetricEstimate

    predicted_sigma: MetricEstimate

    camera_movement: MetricEstimate

    planning_success_rate: MetricEstimate

    safe_navigation_success_rate: MetricEstimate

    collision_rate: MetricEstimate

    safety_violation_rate: MetricEstimate


@dataclass(frozen=True)
class UncertaintyStressResult:
    """Complete E7 result."""

    config: UncertaintyStressConfig

    stress_profiles: tuple[
        UncertaintyStressProfile,
        ...,
    ]

    scenarios: tuple[
        RobustnessScenario,
        ...,
    ]

    selection_records: tuple[
        SelectionSweepRecord,
        ...,
    ]

    trial_records: tuple[
        StressTrialRecord,
        ...,
    ]

    selection_summaries: tuple[
        SelectionSensitivitySummary,
        ...,
    ]

    variant_summaries: tuple[
        StressVariantSummary,
        ...,
    ]


def default_stress_profiles() -> tuple[
    UncertaintyStressProfile,
    ...,
]:
    """Return controlled uncertainty regimes."""

    return (
        UncertaintyStressProfile(
            name="baseline",
        ),
        UncertaintyStressProfile(
            name="distance_dominant",
            distance_multiplier=4.0,
            angle_multiplier=0.25,
        ),
        UncertaintyStressProfile(
            name="angle_dominant",
            distance_multiplier=0.25,
            angle_multiplier=4.0,
        ),
        UncertaintyStressProfile(
            name="strong_combined",
            distance_multiplier=4.0,
            angle_multiplier=4.0,
            occluded_sigma_multiplier=2.0,
            invisible_sigma_multiplier=2.0,
        ),
    )


def confirmation_variants() -> tuple[
    StressVariant,
    ...,
]:
    """Variants used in the focused end-to-end confirmation."""

    return (
        StressVariant(
            name="generic_baseline",
            alignment_weight=0.0,
            uncertainty_weight=0.0,
        ),
        StressVariant(
            name="high_uncertainty_only",
            alignment_weight=0.0,
            uncertainty_weight=16.0,
        ),
        StressVariant(
            name="alignment_only",
            alignment_weight=1.0,
            uncertainty_weight=0.0,
        ),
        StressVariant(
            name="full_task_aware",
            alignment_weight=1.0,
            uncertainty_weight=1.0,
        ),
    )


def apply_stress_profile(
    *,
    inputs: ScenarioInputs,
    profile: UncertaintyStressProfile,
) -> ScenarioInputs:
    """Apply an uncertainty stress profile to one E5 scenario."""

    base_model = (
        inputs.observation_model
    )

    base_config = (
        base_model.config
    )

    modified_config = replace(
        base_config,
        distance_weight=(
            float(
                base_config.distance_weight
            )
            * profile.distance_multiplier
        ),
        angle_weight=(
            float(
                base_config.angle_weight
            )
            * profile.angle_multiplier
        ),
        occluded_sigma=(
            float(
                base_config.occluded_sigma
            )
            * profile.occluded_sigma_multiplier
        ),
        invisible_sigma=(
            float(
                base_config.invisible_sigma
            )
            * profile.invisible_sigma_multiplier
        ),
    )

    modified_model = (
        ViewpointObservationModel(
            camera=base_model.camera,
            config=modified_config,
        )
    )

    return replace(
        inputs,
        observation_model=modified_model,
    )


def build_controller(
    *,
    inputs: ScenarioInputs,
    config: UncertaintyStressConfig,
    alignment_weight: float,
    uncertainty_weight: float,
) -> TaskAwareActivePerception:
    """Build a task-aware controller with explicit component weights."""

    generic_scorer = (
        GenericViewpointScorer(
            observation_model=(
                inputs.observation_model
            )
        )
    )

    scorer = (
        TaskAwareViewpointScorer(
            generic_scorer=generic_scorer,
            task=inputs.task,
            task_config=(
                TaskAwareScoringConfig(
                    task_weight=(
                        config.task_weight
                    ),
                    alignment_weight=(
                        alignment_weight
                    ),
                    uncertainty_weight=(
                        uncertainty_weight
                    ),
                )
            ),
        )
    )

    return TaskAwareActivePerception(
        scorer=scorer
    )


def _candidate_index(
    inputs: ScenarioInputs,
    position: np.ndarray,
) -> int:
    """Return the candidate index corresponding to a selected pose."""

    position = np.asarray(
        position,
        dtype=float,
    )

    for index, candidate in enumerate(
        inputs.candidates
    ):
        if np.allclose(
            candidate.pose.position,
            position,
            atol=1e-10,
            rtol=0.0,
        ):
            return int(index)

    raise RuntimeError(
        "Selected viewpoint was not found in candidate set."
    )


def _predicted_sigma_for_candidate(
    *,
    inputs: ScenarioInputs,
    candidate_index: int,
) -> float:
    """Return target predicted sigma from one candidate."""

    candidate = (
        inputs.candidates[
            candidate_index
        ]
    )

    quality = (
        inputs
        .observation_model
        .observation_quality(
            camera_pose=(
                candidate.pose
            ),
            structure=inputs.target,
            occluders=inputs.occluders,
        )
    )

    return float(
        quality.localisation_sigma
    )


def _selected_index(
    *,
    inputs: ScenarioInputs,
    config: UncertaintyStressConfig,
    alignment_weight: float,
    uncertainty_weight: float,
) -> int:
    """Select a viewpoint under one scorer configuration."""

    controller = build_controller(
        inputs=inputs,
        config=config,
        alignment_weight=alignment_weight,
        uncertainty_weight=uncertainty_weight,
    )

    selection = (
        controller.select_viewpoint(
            current_pose=(
                inputs.initial_pose
            ),
            candidates=inputs.candidates,
            target=inputs.target,
            occluders=inputs.occluders,
        )
    )

    return _candidate_index(
        inputs,
        selection
        .selected_viewpoint
        .pose
        .position,
    )


def run_selection_sweep(
    config: UncertaintyStressConfig,
    *,
    scenarios: tuple[
        RobustnessScenario,
        ...,
    ] | None = None,
    profiles: tuple[
        UncertaintyStressProfile,
        ...,
    ] | None = None,
) -> tuple[
    SelectionSweepRecord,
    ...,
]:
    """Sweep uncertainty weight without running motion planning."""

    if scenarios is None:
        scenarios = (
            default_scenarios()
        )

    if profiles is None:
        profiles = (
            default_stress_profiles()
        )

    records: list[
        SelectionSweepRecord
    ] = []

    for scenario in scenarios:
        base_inputs = (
            build_scenario_inputs(
                scenario
            )
        )

        for profile in profiles:
            inputs = (
                apply_stress_profile(
                    inputs=base_inputs,
                    profile=profile,
                )
            )

            generic_index = (
                _selected_index(
                    inputs=inputs,
                    config=config,
                    alignment_weight=0.0,
                    uncertainty_weight=0.0,
                )
            )

            generic_sigma = (
                _predicted_sigma_for_candidate(
                    inputs=inputs,
                    candidate_index=(
                        generic_index
                    ),
                )
            )

            for uncertainty_weight in (
                config.uncertainty_weights
            ):
                selected_index = (
                    _selected_index(
                        inputs=inputs,
                        config=config,
                        alignment_weight=0.0,
                        uncertainty_weight=(
                            uncertainty_weight
                        ),
                    )
                )

                selected_sigma = (
                    _predicted_sigma_for_candidate(
                        inputs=inputs,
                        candidate_index=(
                            selected_index
                        ),
                    )
                )

                records.append(
                    SelectionSweepRecord(
                        scenario_id=(
                            scenario.scenario_id
                        ),
                        scenario_name=(
                            scenario.name
                        ),
                        stress_profile=(
                            profile.name
                        ),
                        uncertainty_weight=float(
                            uncertainty_weight
                        ),
                        generic_candidate_index=(
                            generic_index
                        ),
                        selected_candidate_index=(
                            selected_index
                        ),
                        differs_from_generic=(
                            selected_index
                            != generic_index
                        ),
                        selected_predicted_sigma=(
                            selected_sigma
                        ),
                        generic_predicted_sigma=(
                            generic_sigma
                        ),
                    )
                )

    return tuple(
        records
    )


def _run_variant_trial(
    *,
    inputs: ScenarioInputs,
    profile: UncertaintyStressProfile,
    variant: StressVariant,
    config: UncertaintyStressConfig,
    scenario: RobustnessScenario,
    repetition: int,
) -> StressTrialRecord:
    """Run one E7 end-to-end variant."""

    seed_offset = (
        scenario.scenario_id
        * 1000
        + repetition
    )

    perception_seed = (
        config.perception_seed_base
        + seed_offset
    )

    planner_seed = (
        config.planner_seed_base
        + seed_offset
    )

    controller = (
        build_controller(
            inputs=inputs,
            config=config,
            alignment_weight=(
                variant.alignment_weight
            ),
            uncertainty_weight=(
                variant.uncertainty_weight
            ),
        )
    )

    perception = (
        run_task_aware_active_perception(
            controller=controller,
            observation_model=(
                inputs.observation_model
            ),
            initial_pose=(
                inputs.initial_pose
            ),
            candidates=inputs.candidates,
            target=inputs.target,
            task=inputs.task,
            true_structures=(
                inputs.true_structures
            ),
            seed=perception_seed,
            occluders=inputs.occluders,
        )
    )

    selected_index = (
        _candidate_index(
            inputs,
            perception
            .selected_pose
            .position,
        )
    )

    navigation = (
        run_navigation_from_perception(
            trial=repetition,
            perception=perception,
            instrument=inputs.instrument,
            start_q=inputs.start_q,
            goal_q=inputs.goal_q,
            true_structures=(
                inputs.true_structures
            ),
            sigma_multiplier=(
                config.sigma_multiplier
            ),
            instrument_radius=(
                config.instrument_radius
            ),
            proximal_length=(
                config.proximal_length
            ),
            planner_seed=planner_seed,
        )
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

    safe_success = (
        planning_success
        and not collision
        and not violation
    )

    return StressTrialRecord(
        scenario_id=(
            scenario.scenario_id
        ),
        scenario_name=(
            scenario.name
        ),
        stress_profile=(
            profile.name
        ),
        repetition=int(
            repetition
        ),
        variant=variant.name,
        perception_seed=int(
            perception_seed
        ),
        planner_seed=int(
            planner_seed
        ),
        selected_candidate_index=(
            selected_index
        ),
        mean_localisation_error=float(
            navigation
            .perception
            .mean_localisation_error
        ),
        mean_predicted_sigma=float(
            navigation
            .perception
            .mean_predicted_sigma
        ),
        camera_movement=float(
            navigation
            .perception
            .camera_movement
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
            safe_success
        ),
    )


def run_end_to_end_confirmation(
    config: UncertaintyStressConfig,
    *,
    scenarios: tuple[
        RobustnessScenario,
        ...,
    ] | None = None,
    profiles: tuple[
        UncertaintyStressProfile,
        ...,
    ] | None = None,
    show_progress: bool = False,
) -> tuple[
    StressTrialRecord,
    ...,
]:
    """Run focused end-to-end stress confirmation."""

    if scenarios is None:
        scenarios = (
            default_scenarios()
        )

    if profiles is None:
        profiles = (
            default_stress_profiles()
        )

    variants = (
        confirmation_variants()
    )

    records: list[
        StressTrialRecord
    ] = []

    total = (
        len(scenarios)
        * len(profiles)
        * config.repetitions
    )

    completed = 0

    for scenario in scenarios:
        base_inputs = (
            build_scenario_inputs(
                scenario
            )
        )

        for profile in profiles:
            inputs = (
                apply_stress_profile(
                    inputs=base_inputs,
                    profile=profile,
                )
            )

            for repetition in range(
                config.repetitions
            ):
                for variant in variants:
                    records.append(
                        _run_variant_trial(
                            inputs=inputs,
                            profile=profile,
                            variant=variant,
                            config=config,
                            scenario=scenario,
                            repetition=(
                                repetition
                            ),
                        )
                    )

                completed += 1

                if (
                    show_progress
                    and (
                        completed % 10 == 0
                        or completed == total
                    )
                ):
                    print(
                        f"Completed {completed}/{total} "
                        "matched uncertainty-stress units."
                    )

    return tuple(
        records
    )


def analyse_selection_sweep(
    records: tuple[
        SelectionSweepRecord,
        ...,
    ],
) -> tuple[
    SelectionSensitivitySummary,
    ...,
]:
    """Summarise whether uncertainty weighting changes viewpoint choice."""

    weights = sorted(
        {
            record.uncertainty_weight
            for record in records
        }
    )

    summaries = []

    for weight in weights:
        selected = [
            record
            for record in records
            if record.uncertainty_weight
            == weight
        ]

        change_rate = float(
            np.mean(
                [
                    record
                    .differs_from_generic
                    for record in selected
                ]
            )
        )

        sigma_change = float(
            np.mean(
                [
                    record
                    .selected_predicted_sigma
                    - record
                    .generic_predicted_sigma
                    for record in selected
                ]
            )
        )

        summaries.append(
            SelectionSensitivitySummary(
                uncertainty_weight=float(
                    weight
                ),
                condition_count=len(
                    selected
                ),
                selection_change_rate=(
                    change_rate
                ),
                mean_sigma_change=(
                    sigma_change
                ),
            )
        )

    return tuple(
        summaries
    )


def _clustered_estimate(
    values: np.ndarray,
    scenario_ids: np.ndarray,
    *,
    config: UncertaintyStressConfig,
    seed_offset: int,
) -> MetricEstimate:
    """Return scenario-clustered bootstrap inference."""

    values = np.asarray(
        values,
        dtype=float,
    )

    scenario_ids = np.asarray(
        scenario_ids,
        dtype=int,
    )

    finite = np.isfinite(
        values
    )

    values = values[
        finite
    ]

    scenario_ids = (
        scenario_ids[
            finite
        ]
    )

    n = int(
        values.size
    )

    if n == 0:
        return MetricEstimate(
            n=0,
            mean=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
        )

    mean = float(
        np.mean(values)
    )

    unique_scenarios = (
        np.unique(
            scenario_ids
        )
    )

    if unique_scenarios.size == 1:
        return MetricEstimate(
            n=n,
            mean=mean,
            ci_low=mean,
            ci_high=mean,
        )

    rng = np.random.default_rng(
        config.bootstrap_seed
        + seed_offset
    )

    bootstrap_means = np.empty(
        config.bootstrap_samples,
        dtype=float,
    )

    for index in range(
        config.bootstrap_samples
    ):
        sampled_scenarios = (
            rng.choice(
                unique_scenarios,
                size=(
                    unique_scenarios.size
                ),
                replace=True,
            )
        )

        sampled_values = (
            np.concatenate(
                [
                    values[
                        scenario_ids
                        == scenario_id
                    ]
                    for scenario_id
                    in sampled_scenarios
                ]
            )
        )

        bootstrap_means[
            index
        ] = float(
            np.mean(
                sampled_values
            )
        )

    alpha = (
        1.0
        - config.confidence_level
    )

    return MetricEstimate(
        n=n,
        mean=mean,
        ci_low=float(
            np.quantile(
                bootstrap_means,
                alpha / 2.0,
            )
        ),
        ci_high=float(
            np.quantile(
                bootstrap_means,
                1.0 - alpha / 2.0,
            )
        ),
    )


def analyse_trial_records(
    *,
    records: tuple[
        StressTrialRecord,
        ...,
    ],
    config: UncertaintyStressConfig,
) -> tuple[
    StressVariantSummary,
    ...,
]:
    """Aggregate focused E7 end-to-end results."""

    summaries = []

    variants = (
        confirmation_variants()
    )

    for variant_index, variant in enumerate(
        variants
    ):
        selected = [
            record
            for record in records
            if record.variant
            == variant.name
        ]

        if len(selected) == 0:
            raise ValueError(
                f"No records found for {variant.name}."
            )

        scenario_ids = np.asarray(
            [
                record.scenario_id
                for record in selected
            ],
            dtype=int,
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
                    for record in selected
                ],
                dtype=float,
            )

        base_offset = (
            variant_index
            * 100
        )

        summaries.append(
            StressVariantSummary(
                variant=variant.name,
                record_count=len(
                    selected
                ),
                localisation_error=(
                    _clustered_estimate(
                        values(
                            "mean_localisation_error"
                        ),
                        scenario_ids,
                        config=config,
                        seed_offset=(
                            base_offset
                            + 1
                        ),
                    )
                ),
                predicted_sigma=(
                    _clustered_estimate(
                        values(
                            "mean_predicted_sigma"
                        ),
                        scenario_ids,
                        config=config,
                        seed_offset=(
                            base_offset
                            + 2
                        ),
                    )
                ),
                camera_movement=(
                    _clustered_estimate(
                        values(
                            "camera_movement"
                        ),
                        scenario_ids,
                        config=config,
                        seed_offset=(
                            base_offset
                            + 3
                        ),
                    )
                ),
                planning_success_rate=(
                    _clustered_estimate(
                        values(
                            "planning_success"
                        ),
                        scenario_ids,
                        config=config,
                        seed_offset=(
                            base_offset
                            + 4
                        ),
                    )
                ),
                safe_navigation_success_rate=(
                    _clustered_estimate(
                        values(
                            "safe_navigation_success"
                        ),
                        scenario_ids,
                        config=config,
                        seed_offset=(
                            base_offset
                            + 5
                        ),
                    )
                ),
                collision_rate=(
                    _clustered_estimate(
                        values(
                            "collision_against_truth"
                        ),
                        scenario_ids,
                        config=config,
                        seed_offset=(
                            base_offset
                            + 6
                        ),
                    )
                ),
                safety_violation_rate=(
                    _clustered_estimate(
                        values(
                            "safety_violation_against_truth"
                        ),
                        scenario_ids,
                        config=config,
                        seed_offset=(
                            base_offset
                            + 7
                        ),
                    )
                ),
            )
        )

    return tuple(
        summaries
    )


def run_uncertainty_stress_benchmark(
    config: UncertaintyStressConfig
    | None = None,
    *,
    scenarios: tuple[
        RobustnessScenario,
        ...,
    ] | None = None,
    profiles: tuple[
        UncertaintyStressProfile,
        ...,
    ] | None = None,
    show_progress: bool = False,
) -> UncertaintyStressResult:
    """Run both E7 stages."""

    if config is None:
        config = (
            UncertaintyStressConfig()
        )

    if scenarios is None:
        scenarios = (
            default_scenarios()
        )

    if profiles is None:
        profiles = (
            default_stress_profiles()
        )

    selection_records = (
        run_selection_sweep(
            config,
            scenarios=scenarios,
            profiles=profiles,
        )
    )

    trial_records = (
        run_end_to_end_confirmation(
            config,
            scenarios=scenarios,
            profiles=profiles,
            show_progress=show_progress,
        )
    )

    return UncertaintyStressResult(
        config=config,
        stress_profiles=profiles,
        scenarios=scenarios,
        selection_records=(
            selection_records
        ),
        trial_records=(
            trial_records
        ),
        selection_summaries=(
            analyse_selection_sweep(
                selection_records
            )
        ),
        variant_summaries=(
            analyse_trial_records(
                records=trial_records,
                config=config,
            )
        ),
    )


def save_outputs(
    result: UncertaintyStressResult,
    *,
    output_directory: str
    | Path = "results/e7_uncertainty_stress",
) -> tuple[
    Path,
    Path,
    Path,
]:
    """Save E7 raw and summary outputs."""

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    sweep_path = (
        output_directory
        / "selection_sweep.csv"
    )

    trials_path = (
        output_directory
        / "stress_trials.csv"
    )

    summary_path = (
        output_directory
        / "uncertainty_stress_summary.json"
    )

    with sweep_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                asdict(
                    result.selection_records[
                        0
                    ]
                ).keys()
            ),
        )

        writer.writeheader()

        for record in (
            result.selection_records
        ):
            writer.writerow(
                asdict(
                    record
                )
            )

    with trials_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                asdict(
                    result.trial_records[
                        0
                    ]
                ).keys()
            ),
        )

        writer.writeheader()

        for record in (
            result.trial_records
        ):
            writer.writerow(
                asdict(
                    record
                )
            )

    payload = {
        "config": asdict(
            result.config
        ),
        "stress_profiles": [
            asdict(
                profile
            )
            for profile
            in result.stress_profiles
        ],
        "selection_summaries": [
            asdict(
                summary
            )
            for summary
            in result.selection_summaries
        ],
        "variant_summaries": [
            asdict(
                summary
            )
            for summary
            in result.variant_summaries
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
            allow_nan=True,
        )

    return (
        sweep_path,
        trials_path,
        summary_path,
    )


def _format_rate(
    metric: MetricEstimate,
) -> str:
    return (
        f"{metric.mean * 100.0:.1f}% "
        f"[{metric.ci_low * 100.0:.1f}, "
        f"{metric.ci_high * 100.0:.1f}]"
    )


def _format_mm(
    metric: MetricEstimate,
) -> str:
    return (
        f"{metric.mean * 1000.0:.3f} "
        f"[{metric.ci_low * 1000.0:.3f}, "
        f"{metric.ci_high * 1000.0:.3f}]"
    )


def print_summary(
    result: UncertaintyStressResult,
) -> None:
    """Print the key E7 results."""

    print()
    print(
        "E7 Explicit-Uncertainty Stress Benchmark"
    )
    print(
        "========================================"
    )

    print()
    print(
        "Uncertainty-only selection sensitivity"
    )
    print(
        "--------------------------------------"
    )

    for summary in (
        result.selection_summaries
    ):
        print(
            "Weight "
            f"{summary.uncertainty_weight:6.2f}: "
            "selection changed in "
            f"{summary.selection_change_rate * 100.0:5.1f}% "
            "of conditions; mean sigma change "
            f"{summary.mean_sigma_change * 1000.0:+.3f} mm"
        )

    print()
    print(
        "Focused end-to-end stress confirmation"
    )
    print(
        "--------------------------------------"
    )

    names = {
        "generic_baseline": (
            "Generic Baseline"
        ),
        "high_uncertainty_only": (
            "High Uncertainty Only"
        ),
        "alignment_only": (
            "Alignment Only"
        ),
        "full_task_aware": (
            "Full Task-Aware"
        ),
    }

    for summary in (
        result.variant_summaries
    ):
        print()
        print(
            names[
                summary.variant
            ]
        )

        print(
            "  Localisation error (mm): "
            + _format_mm(
                summary
                .localisation_error
            )
        )

        print(
            "  Predicted sigma (mm):    "
            + _format_mm(
                summary
                .predicted_sigma
            )
        )

        print(
            "  Camera movement (mm):    "
            + _format_mm(
                summary
                .camera_movement
            )
        )

        print(
            "  Planning success:        "
            + _format_rate(
                summary
                .planning_success_rate
            )
        )

        print(
            "  Safe navigation success: "
            + _format_rate(
                summary
                .safe_navigation_success_rate
            )
        )


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repetitions",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "results/e7_uncertainty_stress"
        ),
    )

    args = parser.parse_args()

    config = (
        UncertaintyStressConfig(
            repetitions=(
                args.repetitions
            ),
            bootstrap_samples=(
                args.bootstrap_samples
            ),
        )
    )

    result = (
        run_uncertainty_stress_benchmark(
            config,
            show_progress=True,
        )
    )

    print_summary(
        result
    )

    paths = save_outputs(
        result,
        output_directory=(
            args.output
        ),
    )

    print()
    print(
        f"Selection sweep saved to: {paths[0]}"
    )

    print(
        f"Stress trials saved to: {paths[1]}"
    )

    print(
        f"Summary saved to: {paths[2]}"
    )


if __name__ == "__main__":
    main()