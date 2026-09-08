"""Final Phase 6 closed-loop uncertainty-aware execution benchmark.

Three strategies are compared:

1. open_loop_nominal
   Plans once from the initial estimate and ignores subsequent perception
   changes.

2. static_robust
   Applies the previously defined static robustification once before
   execution. It does not adapt to subsequent information.

3. closed_loop_adaptive
   Re-evaluates the remaining trajectory as perception changes and may
   continue, replan, reacquire perception, or issue a fail-safe stop.

Final dynamic scenarios
-----------------------

stable
    Static hidden anatomy with routine noisy perception after controlled
    initialisation.

recoverable_structure_update
    A secondary synthetic spherical structure exists in hidden truth but is
    not available to the runtime system until execution index 11. Its geometry
    was selected using a development-only search before final evaluation.

    Frozen development candidate:

        update index:
            11

        secondary centre:
            [0.0420764, 0.00799174, 0.02476202] m

        physical radius:
            5 mm

        safety margin:
            4 mm

        positional sigma:
            2 mm isotropic

    Development evidence established:

        - pre-update trajectory section safe;
        - current configuration at discovery safe;
        - one future nominal safety-margin violation;
        - zero nominal physical collisions;
        - chance-constrained replanning succeeded in 3/3 development seeds.

    The candidate is frozen before final benchmark seeds and must not be
    retuned using final evaluation results.

transient_uncertainty_spike
    Reported localisation uncertainty rises temporarily and then returns to
    nominal levels.

severe_perception_loss
    Reported localisation uncertainty becomes sufficiently large to trigger
    the fail-safe perception-loss response.

Hidden truth is used only by the simulator to generate observations and score
executed configurations. Hidden truth is not passed directly into runtime
planning or supervisory decisions.

The benchmark is simulation-only engineering evidence. It does not establish
clinical safety, patient risk, medical-device validation, or real-robot
performance.
"""

from __future__ import annotations

import csv
import json

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

from src.geometry.workspace import (
    SphericalStructure,
)
from src.perception.uncertainty import (
    EstimatedStructure,
    PositionUncertainty,
)
from src.robotics.chance_constrained_planner import (
    ChanceConstrainedRRTConfig,
    plan_rrt_chance_constrained,
)
from src.robotics.closed_loop_risk_manager import (
    ClosedLoopRiskConfig,
    ClosedLoopRiskManager,
    RiskManagementAction,
)
from src.robotics.planner import (
    PlanningResult,
    joint_distance,
)
from src.robotics.robust_chance_policy import (
    RobustChancePolicy,
    robust_chance_constraint_config,
    robustify_estimated_structure,
)
from src.robotics.safety import (
    evaluate_instrument_safety,
)
from src.simulation.phase6_chance_constraint_benchmark import (
    CHANCE_THRESHOLD,
    INSTRUMENT_RADIUS,
    PROXIMAL_LENGTH,
    SHAFT_SAMPLE_SPACING,
    chance_config,
    goal_configuration,
    make_instrument,
    select_disagreement_clearance,
    start_configuration,
)


DEFAULT_REPETITIONS = 30
DEFAULT_SEED = 69101

EXECUTION_SAMPLES = 36
MAX_EXECUTION_STEPS = 60

PERCEPTION_NOISE_FRACTION = 0.50
REACQUIRED_STD_SCALE = 0.55

RESULT_DIRECTORY = Path(
    "results/phase6_closed_loop"
)


# ---------------------------------------------------------------------------
# Frozen recoverable structure-update scenario
# ---------------------------------------------------------------------------

STRUCTURE_UPDATE_ONSET_INDEX = 11

FROZEN_SECONDARY_CENTRE = np.asarray(
    [
        0.0420764,
        0.00799174,
        0.02476202,
    ],
    dtype=float,
)

SECONDARY_PHYSICAL_RADIUS = 0.005
SECONDARY_SAFETY_MARGIN = 0.004
SECONDARY_POSITION_SIGMA = 0.002


# ---------------------------------------------------------------------------
# Frozen policies
# ---------------------------------------------------------------------------

STATIC_ROBUST_POLICY = RobustChancePolicy(
    covariance_std_scale=1.10,
    bias_bound=0.001,
    contamination_probability=0.005,
    target_violation_probability=(
        CHANCE_THRESHOLD
    ),
)


CLOSED_LOOP_CONFIG = ClosedLoopRiskConfig(
    replan_probability=0.05,
    reacquire_probability=0.20,
    stop_probability=0.45,
    reacquire_principal_sigma=0.015,
    stop_principal_sigma=0.030,
    minimum_replan_interval_steps=2,
)


@dataclass(frozen=True)
class EpisodeEnvironment:
    """Pre-generated hidden truth and planner-facing observations."""

    scenario: str

    truth_centres: np.ndarray

    reported_estimates: tuple[
        EstimatedStructure,
        ...,
    ]


@dataclass(frozen=True)
class ExecutionOutcome:
    """Outcome of one strategy in one simulated episode."""

    initial_plan_success: bool

    completed: bool

    safe_completion: bool

    executed_steps: int

    unsafe_steps: int

    collision_steps: int

    any_safety_violation: bool

    any_collision: bool

    replans: int

    reacquisitions: int

    stops: int

    initial_blocked: bool

    final_progress: float

    total_planning_iterations: int


def scenario_names() -> tuple[
    str,
    ...,
]:
    """Return frozen final Phase 6 scenario order."""

    return (
        "stable",
        "recoverable_structure_update",
        "transient_uncertainty_spike",
        "severe_perception_loss",
    )


@lru_cache(maxsize=1)
def _base_condition():
    """Return cached tangential-major Phase 6 base condition.

    The controlled geometry is deterministic. Caching removes repeated
    clearance-search computation without caching perception, execution,
    planning, or hidden-truth outcomes.
    """

    (
        _clearance,
        conditions,
    ) = (
        select_disagreement_clearance()
    )

    tangential, _radial = conditions

    return tangential


def _progress(
    step_index: int,
) -> float:
    """Return clipped nominal execution progress."""

    return float(
        np.clip(
            step_index
            / float(
                EXECUTION_SAMPLES
                - 1
            ),
            0.0,
            1.0,
        )
    )


def _truth_centre(
    scenario: str,
    step_index: int,
) -> np.ndarray:
    """Return hidden base-anatomy centre.

    The final recoverable scenario no longer moves the original Phase 6
    structure. Recoverability is instead introduced through a late-discovered
    secondary structure.
    """

    if scenario not in scenario_names():
        raise ValueError(
            f"Unknown scenario: {scenario}"
        )

    if step_index < 0:
        raise ValueError(
            "step_index must be non-negative."
        )

    return (
        _base_condition()
        .estimate
        .estimated_centre
        .copy()
    )


def _reported_std_scale(
    scenario: str,
    step_index: int,
) -> float:
    """Return base-structure reported uncertainty scale."""

    if step_index < 0:
        raise ValueError(
            "step_index must be non-negative."
        )

    if scenario == "stable":
        return 1.0

    if (
        scenario
        == "recoverable_structure_update"
    ):
        return 1.0

    if (
        scenario
        == "transient_uncertainty_spike"
    ):
        if (
            11
            <= step_index
            <= 13
        ):
            return 1.80

        return 1.0

    if (
        scenario
        == "severe_perception_loss"
    ):
        if step_index >= 12:
            return 3.20

        return 1.0

    raise ValueError(
        f"Unknown scenario: {scenario}"
    )


def _estimate_from_truth(
    *,
    truth_centre: np.ndarray,
    std_scale: float,
    rng: np.random.Generator,
    exact_centre: bool = False,
) -> EstimatedStructure:
    """Generate planner-facing base-structure perception."""

    condition = _base_condition()

    truth_centre = np.asarray(
        truth_centre,
        dtype=float,
    )

    if truth_centre.shape != (
        3,
    ):
        raise ValueError(
            "truth_centre must have shape (3,)."
        )

    if not np.all(
        np.isfinite(
            truth_centre
        )
    ):
        raise ValueError(
            "truth_centre must contain finite values."
        )

    if (
        not np.isfinite(
            std_scale
        )
        or std_scale <= 0.0
    ):
        raise ValueError(
            "std_scale must be finite and positive."
        )

    covariance = (
        condition
        .estimate
        .uncertainty
        .covariance
        * std_scale**2
    )

    if exact_centre:
        error = np.zeros(
            3,
            dtype=float,
        )

    else:
        error = (
            rng.multivariate_normal(
                mean=np.zeros(
                    3,
                    dtype=float,
                ),
                cov=(
                    covariance
                    * PERCEPTION_NOISE_FRACTION**2
                ),
            )
        )

    return EstimatedStructure(
        estimated_centre=(
            truth_centre
            + error
        ),
        physical_radius=(
            condition
            .estimate
            .physical_radius
        ),
        base_safety_margin=(
            condition
            .estimate
            .base_safety_margin
        ),
        uncertainty=PositionUncertainty(
            covariance=covariance
        ),
    )


def _secondary_structure_estimate() -> EstimatedStructure:
    """Return frozen planner-facing secondary-structure estimate."""

    covariance = (
        np.eye(
            3,
            dtype=float,
        )
        * SECONDARY_POSITION_SIGMA**2
    )

    return EstimatedStructure(
        estimated_centre=(
            FROZEN_SECONDARY_CENTRE
            .copy()
        ),
        physical_radius=(
            SECONDARY_PHYSICAL_RADIUS
        ),
        base_safety_margin=(
            SECONDARY_SAFETY_MARGIN
        ),
        uncertainty=PositionUncertainty(
            covariance=covariance
        ),
    )


def _secondary_truth_structure() -> SphericalStructure:
    """Return frozen hidden secondary structure."""

    return SphericalStructure(
        centre=(
            FROZEN_SECONDARY_CENTRE
            .copy()
        ),
        physical_radius=(
            SECONDARY_PHYSICAL_RADIUS
        ),
        safety_margin=(
            SECONDARY_SAFETY_MARGIN
        ),
    )


def make_environment(
    scenario: str,
    seed: int,
) -> EpisodeEnvironment:
    """Pre-generate matched base perception for one episode.

    All scenarios use a controlled initial base-structure mean with non-zero
    covariance.

    Stable/transient/severe scenarios contain noisy base observations after
    initialisation.

    The recoverable structure-update scenario deliberately keeps the original
    base structure mean controlled throughout the episode so the newly
    discovered secondary structure is the isolated experimental intervention.
    """

    if scenario not in scenario_names():
        raise ValueError(
            f"Unknown scenario: {scenario}"
        )

    rng = np.random.default_rng(
        seed
    )

    truth_centres: list[
        np.ndarray
    ] = []

    estimates: list[
        EstimatedStructure
    ] = []

    for step_index in range(
        MAX_EXECUTION_STEPS
    ):
        centre = _truth_centre(
            scenario,
            step_index,
        )

        scale = _reported_std_scale(
            scenario,
            step_index,
        )

        exact_centre = bool(
            step_index == 0
            or scenario
            == "recoverable_structure_update"
        )

        estimate = (
            _estimate_from_truth(
                truth_centre=centre,
                std_scale=scale,
                rng=rng,
                exact_centre=(
                    exact_centre
                ),
            )
        )

        truth_centres.append(
            np.asarray(
                centre,
                dtype=float,
            ).copy()
        )

        estimates.append(
            estimate
        )

    return EpisodeEnvironment(
        scenario=scenario,
        truth_centres=np.vstack(
            truth_centres
        ),
        reported_estimates=tuple(
            estimates
        ),
    )


def _estimated_structures_for_step(
    environment: EpisodeEnvironment,
    step_index: int,
) -> tuple[
    EstimatedStructure,
    ...,
]:
    """Return structures available to runtime perception at one step."""

    if step_index < 0:
        raise ValueError(
            "step_index must be non-negative."
        )

    if step_index >= len(
        environment.reported_estimates
    ):
        raise ValueError(
            "step_index exceeds environment horizon."
        )

    structures: list[
        EstimatedStructure
    ] = [
        environment
        .reported_estimates[
            step_index
        ]
    ]

    if (
        environment.scenario
        == "recoverable_structure_update"
        and step_index
        >= STRUCTURE_UPDATE_ONSET_INDEX
    ):
        structures.append(
            _secondary_structure_estimate()
        )

    return tuple(
        structures
    )


def _truth_structure(
    centre: np.ndarray,
) -> SphericalStructure:
    """Return hidden base structure."""

    condition = _base_condition()

    return SphericalStructure(
        centre=np.asarray(
            centre,
            dtype=float,
        ),
        physical_radius=(
            condition
            .estimate
            .physical_radius
        ),
        safety_margin=(
            condition
            .estimate
            .base_safety_margin
        ),
    )


def _truth_structures_for_step(
    environment: EpisodeEnvironment,
    step_index: int,
) -> tuple[
    SphericalStructure,
    ...,
]:
    """Return hidden structures used only for simulator evaluation.

    The secondary structure exists in hidden truth throughout the recoverable
    structure-update episode. It is merely unavailable to runtime perception
    until the frozen update index.
    """

    if step_index < 0:
        raise ValueError(
            "step_index must be non-negative."
        )

    if step_index >= (
        environment
        .truth_centres
        .shape[
            0
        ]
    ):
        raise ValueError(
            "step_index exceeds environment horizon."
        )

    structures: list[
        SphericalStructure
    ] = [
        _truth_structure(
            environment
            .truth_centres[
                step_index
            ]
        )
    ]

    if (
        environment.scenario
        == "recoverable_structure_update"
    ):
        structures.append(
            _secondary_truth_structure()
        )

    return tuple(
        structures
    )


def _planner_config(
    seed: int,
) -> ChanceConstrainedRRTConfig:
    """Return shared RRT planning parameters."""

    return ChanceConstrainedRRTConfig(
        max_iterations=2000,
        step_size=0.08,
        goal_bias=0.20,
        edge_resolution=16,
        proximal_length=(
            PROXIMAL_LENGTH
        ),
        shaft_sample_spacing=(
            SHAFT_SAMPLE_SPACING
        ),
        seed=int(
            seed
        ),
    )


def _plan_nominal_with_structures(
    *,
    start_q: np.ndarray,
    estimated_structures: Sequence[
        EstimatedStructure
    ],
    seed: int,
) -> PlanningResult:
    """Plan using one or more estimated uncertain structures."""

    if len(
        estimated_structures
    ) < 1:
        raise ValueError(
            "At least one estimated structure is required."
        )

    return (
        plan_rrt_chance_constrained(
            instrument=make_instrument(),
            start_q=np.asarray(
                start_q,
                dtype=float,
            ),
            goal_q=goal_configuration(),
            estimated_structures=tuple(
                estimated_structures
            ),
            instrument_radius=(
                INSTRUMENT_RADIUS
            ),
            chance_config=(
                chance_config()
            ),
            planner_config=(
                _planner_config(
                    seed
                )
            ),
        )
    )


def _plan_nominal(
    *,
    start_q: np.ndarray,
    estimate: EstimatedStructure,
    seed: int,
) -> PlanningResult:
    """Compatibility wrapper for one-structure nominal planning."""

    return (
        _plan_nominal_with_structures(
            start_q=start_q,
            estimated_structures=(
                estimate,
            ),
            seed=seed,
        )
    )


def _plan_static_robust(
    *,
    start_q: np.ndarray,
    estimate: EstimatedStructure,
    seed: int,
) -> PlanningResult:
    """Plan once using the frozen static robust policy."""

    robust_estimate = (
        robustify_estimated_structure(
            estimate,
            STATIC_ROBUST_POLICY,
        )
    )

    robust_config = (
        robust_chance_constraint_config(
            STATIC_ROBUST_POLICY,
            sample_spacing=(
                SHAFT_SAMPLE_SPACING
            ),
        )
    )

    return (
        plan_rrt_chance_constrained(
            instrument=make_instrument(),
            start_q=np.asarray(
                start_q,
                dtype=float,
            ),
            goal_q=goal_configuration(),
            estimated_structures=(
                robust_estimate,
            ),
            instrument_radius=(
                INSTRUMENT_RADIUS
            ),
            chance_config=(
                robust_config
            ),
            planner_config=(
                _planner_config(
                    seed
                )
            ),
        )
    )


def resample_joint_path(
    path: np.ndarray,
    sample_count: int = EXECUTION_SAMPLES,
) -> np.ndarray:
    """Resample joint path by cumulative scaled joint-space distance."""

    joint_path = np.asarray(
        path,
        dtype=float,
    )

    if (
        joint_path.ndim != 2
        or joint_path.shape[1] != 4
        or joint_path.shape[0] < 1
    ):
        raise ValueError(
            "path must have shape (N, 4) with N >= 1."
        )

    if not np.all(
        np.isfinite(
            joint_path
        )
    ):
        raise ValueError(
            "path must contain finite values."
        )

    if sample_count < 2:
        raise ValueError(
            "sample_count must be at least 2."
        )

    if joint_path.shape[0] == 1:
        return np.repeat(
            joint_path,
            sample_count,
            axis=0,
        )

    segment_costs = np.asarray(
        [
            joint_distance(
                joint_path[
                    index
                ],
                joint_path[
                    index + 1
                ],
            )
            for index in range(
                joint_path.shape[
                    0
                ]
                - 1
            )
        ],
        dtype=float,
    )

    total_cost = float(
        np.sum(
            segment_costs
        )
    )

    if total_cost <= 1e-12:
        return np.repeat(
            joint_path[
                :1
            ],
            sample_count,
            axis=0,
        )

    cumulative = np.concatenate(
        [
            np.array(
                [
                    0.0
                ],
                dtype=float,
            ),
            np.cumsum(
                segment_costs
            ),
        ]
    )

    targets = np.linspace(
        0.0,
        total_cost,
        num=sample_count,
        dtype=float,
    )

    output: list[
        np.ndarray
    ] = []

    for target in targets:
        if target >= total_cost:
            output.append(
                joint_path[
                    -1
                ].copy()
            )

            continue

        segment_index = int(
            np.searchsorted(
                cumulative,
                target,
                side="right",
            )
            - 1
        )

        segment_index = int(
            np.clip(
                segment_index,
                0,
                joint_path.shape[
                    0
                ]
                - 2,
            )
        )

        segment_cost = (
            segment_costs[
                segment_index
            ]
        )

        if segment_cost <= 1e-12:
            fraction = 0.0

        else:
            fraction = float(
                (
                    target
                    - cumulative[
                        segment_index
                    ]
                )
                / segment_cost
            )

        configuration = (
            joint_path[
                segment_index
            ]
            + fraction
            * (
                joint_path[
                    segment_index
                    + 1
                ]
                - joint_path[
                    segment_index
                ]
            )
        )

        output.append(
            configuration
        )

    return np.vstack(
        output
    )


def _evaluate_configuration_against_truth(
    q: np.ndarray,
    truth_centre: np.ndarray,
) -> tuple[
    bool,
    bool,
]:
    """Compatibility helper evaluating only the base hidden structure."""

    instrument = make_instrument()

    shaft_start, shaft_end = (
        instrument.shaft_segment(
            q,
            proximal_length=(
                PROXIMAL_LENGTH
            ),
        )
    )

    evaluation = (
        evaluate_instrument_safety(
            shaft_start=shaft_start,
            shaft_end=shaft_end,
            structures=(
                _truth_structure(
                    truth_centre
                ),
            ),
            instrument_radius=(
                INSTRUMENT_RADIUS
            ),
        )
    )

    return (
        bool(
            evaluation
            .safety_margin_violation
        ),
        bool(
            evaluation
            .collision
        ),
    )


def _evaluate_configuration_against_environment(
    *,
    q: np.ndarray,
    environment: EpisodeEnvironment,
    step_index: int,
) -> tuple[
    bool,
    bool,
]:
    """Evaluate executed configuration against complete hidden truth."""

    instrument = make_instrument()

    shaft_start, shaft_end = (
        instrument.shaft_segment(
            q,
            proximal_length=(
                PROXIMAL_LENGTH
            ),
        )
    )

    evaluation = (
        evaluate_instrument_safety(
            shaft_start=shaft_start,
            shaft_end=shaft_end,
            structures=(
                _truth_structures_for_step(
                    environment,
                    step_index,
                )
            ),
            instrument_radius=(
                INSTRUMENT_RADIUS
            ),
        )
    )

    return (
        bool(
            evaluation
            .safety_margin_violation
        ),
        bool(
            evaluation
            .collision
        ),
    )


def _reacquired_estimate(
    *,
    truth_centre: np.ndarray,
    rng: np.random.Generator,
) -> EstimatedStructure:
    """Simulate improved base-structure localisation."""

    return (
        _estimate_from_truth(
            truth_centre=truth_centre,
            std_scale=(
                REACQUIRED_STD_SCALE
            ),
            rng=rng,
            exact_centre=False,
        )
    )


def _reacquired_structures(
    *,
    environment: EpisodeEnvironment,
    step_index: int,
    rng: np.random.Generator,
) -> tuple[
    EstimatedStructure,
    ...,
]:
    """Return structures after simulated perception reacquisition."""

    base_estimate = (
        _reacquired_estimate(
            truth_centre=(
                environment
                .truth_centres[
                    step_index
                ]
            ),
            rng=rng,
        )
    )

    structures: list[
        EstimatedStructure
    ] = [
        base_estimate
    ]

    if (
        environment.scenario
        == "recoverable_structure_update"
        and step_index
        >= STRUCTURE_UPDATE_ONSET_INDEX
    ):
        structures.append(
            _secondary_structure_estimate()
        )

    return tuple(
        structures
    )


def _final_progress(
    current_q: np.ndarray,
) -> float:
    """Return bounded progress toward the fixed goal."""

    initial_distance = (
        joint_distance(
            start_configuration(),
            goal_configuration(),
        )
    )

    remaining = (
        joint_distance(
            current_q,
            goal_configuration(),
        )
    )

    if initial_distance <= 1e-12:
        return 1.0

    return float(
        np.clip(
            1.0
            - remaining
            / initial_distance,
            0.0,
            1.0,
        )
    )


def _empty_failed_outcome(
    *,
    total_iterations: int,
) -> ExecutionOutcome:
    """Return outcome for a strategy blocked before execution."""

    return ExecutionOutcome(
        initial_plan_success=False,
        completed=False,
        safe_completion=False,
        executed_steps=0,
        unsafe_steps=0,
        collision_steps=0,
        any_safety_violation=False,
        any_collision=False,
        replans=0,
        reacquisitions=0,
        stops=0,
        initial_blocked=True,
        final_progress=0.0,
        total_planning_iterations=int(
            total_iterations
        ),
    )


def _execute_fixed_plan(
    *,
    result: PlanningResult,
    environment: EpisodeEnvironment,
) -> ExecutionOutcome:
    """Execute one fixed path without closed-loop adaptation."""

    if not result.success:
        return (
            _empty_failed_outcome(
                total_iterations=(
                    result.iterations
                )
            )
        )

    path = resample_joint_path(
        result.path,
        EXECUTION_SAMPLES,
    )

    unsafe_steps = 0
    collision_steps = 0
    executed_steps = 0

    current_q = (
        path[
            0
        ].copy()
    )

    for cursor in range(
        1,
        path.shape[
            0
        ],
    ):
        step_index = min(
            cursor,
            MAX_EXECUTION_STEPS
            - 1,
        )

        next_q = (
            path[
                cursor
            ]
        )

        (
            unsafe,
            collision,
        ) = (
            _evaluate_configuration_against_environment(
                q=next_q,
                environment=environment,
                step_index=step_index,
            )
        )

        unsafe_steps += int(
            unsafe
        )

        collision_steps += int(
            collision
        )

        executed_steps += 1

        current_q = (
            next_q.copy()
        )

    completed = bool(
        np.allclose(
            current_q,
            goal_configuration(),
            atol=1e-8,
        )
    )

    return ExecutionOutcome(
        initial_plan_success=True,
        completed=completed,
        safe_completion=bool(
            completed
            and unsafe_steps == 0
        ),
        executed_steps=int(
            executed_steps
        ),
        unsafe_steps=int(
            unsafe_steps
        ),
        collision_steps=int(
            collision_steps
        ),
        any_safety_violation=bool(
            unsafe_steps > 0
        ),
        any_collision=bool(
            collision_steps > 0
        ),
        replans=0,
        reacquisitions=0,
        stops=0,
        initial_blocked=False,
        final_progress=(
            1.0
            if completed
            else _final_progress(
                current_q
            )
        ),
        total_planning_iterations=int(
            result.iterations
        ),
    )


def execute_open_loop_nominal(
    environment: EpisodeEnvironment,
    seed: int,
) -> ExecutionOutcome:
    """Plan once nominally and ignore subsequent perception changes."""

    initial_structures = (
        _estimated_structures_for_step(
            environment,
            0,
        )
    )

    result = (
        _plan_nominal_with_structures(
            start_q=(
                start_configuration()
            ),
            estimated_structures=(
                initial_structures
            ),
            seed=seed,
        )
    )

    return (
        _execute_fixed_plan(
            result=result,
            environment=environment,
        )
    )


def execute_static_robust(
    environment: EpisodeEnvironment,
    seed: int,
) -> ExecutionOutcome:
    """Plan once with static robustification and never update."""

    initial_estimate = (
        environment
        .reported_estimates[
            0
        ]
    )

    result = (
        _plan_static_robust(
            start_q=(
                start_configuration()
            ),
            estimate=(
                initial_estimate
            ),
            seed=seed,
        )
    )

    return (
        _execute_fixed_plan(
            result=result,
            environment=environment,
        )
    )


def execute_closed_loop_adaptive(
    environment: EpisodeEnvironment,
    seed: int,
) -> ExecutionOutcome:
    """Execute with closed-loop risk supervision."""

    initial_structures = (
        _estimated_structures_for_step(
            environment,
            0,
        )
    )

    initial_result = (
        _plan_nominal_with_structures(
            start_q=(
                start_configuration()
            ),
            estimated_structures=(
                initial_structures
            ),
            seed=seed,
        )
    )

    if not initial_result.success:
        return (
            _empty_failed_outcome(
                total_iterations=(
                    initial_result
                    .iterations
                )
            )
        )

    planned_path = (
        resample_joint_path(
            initial_result.path,
            EXECUTION_SAMPLES,
        )
    )

    manager = (
        ClosedLoopRiskManager(
            CLOSED_LOOP_CONFIG
        )
    )

    reacquisition_rng = (
        np.random.default_rng(
            seed
            + 500000
        )
    )

    current_q = (
        planned_path[
            0
        ].copy()
    )

    cursor = 1

    executed_steps = 0
    unsafe_steps = 0
    collision_steps = 0

    replans = 0
    reacquisitions = 0
    stops = 0

    total_iterations = int(
        initial_result.iterations
    )

    completed = False

    for step_index in range(
        1,
        MAX_EXECUTION_STEPS,
    ):
        if cursor >= (
            planned_path.shape[
                0
            ]
        ):
            completed = bool(
                np.allclose(
                    current_q,
                    goal_configuration(),
                    atol=1e-8,
                )
            )

            break

        current_structures = (
            _estimated_structures_for_step(
                environment,
                step_index,
            )
        )

        remaining_path = np.vstack(
            [
                current_q,
                planned_path[
                    cursor:
                ],
            ]
        )

        decision = (
            manager.evaluate(
                instrument=(
                    make_instrument()
                ),
                remaining_path=(
                    remaining_path
                ),
                estimated_structures=(
                    current_structures
                ),
                instrument_radius=(
                    INSTRUMENT_RADIUS
                ),
                chance_config=(
                    chance_config()
                ),
                step_index=(
                    step_index
                ),
                proximal_length=(
                    PROXIMAL_LENGTH
                ),
                shaft_sample_spacing=(
                    SHAFT_SAMPLE_SPACING
                ),
                edge_resolution=16,
            )
        )

        active_structures = (
            current_structures
        )

        if (
            decision.action
            == RiskManagementAction.STOP
        ):
            stops += 1

            break

        if (
            decision.action
            == RiskManagementAction.REACQUIRE
        ):
            reacquisitions += 1

            active_structures = (
                _reacquired_structures(
                    environment=environment,
                    step_index=step_index,
                    rng=reacquisition_rng,
                )
            )

            replanned = (
                _plan_nominal_with_structures(
                    start_q=current_q,
                    estimated_structures=(
                        active_structures
                    ),
                    seed=(
                        seed
                        + 10000
                        + step_index
                    ),
                )
            )

            total_iterations += int(
                replanned.iterations
            )

            if not replanned.success:
                stops += 1

                break

            replans += 1

            manager.register_replan(
                step_index
            )

            remaining_samples = max(
                8,
                EXECUTION_SAMPLES
                - min(
                    step_index,
                    EXECUTION_SAMPLES
                    - 8,
                ),
            )

            planned_path = (
                resample_joint_path(
                    replanned.path,
                    remaining_samples,
                )
            )

            current_q = (
                planned_path[
                    0
                ].copy()
            )

            cursor = 1

        elif (
            decision.action
            == RiskManagementAction.REPLAN
        ):
            replanned = (
                _plan_nominal_with_structures(
                    start_q=current_q,
                    estimated_structures=(
                        active_structures
                    ),
                    seed=(
                        seed
                        + 20000
                        + step_index
                    ),
                )
            )

            total_iterations += int(
                replanned.iterations
            )

            if not replanned.success:
                reacquisitions += 1

                active_structures = (
                    _reacquired_structures(
                        environment=environment,
                        step_index=step_index,
                        rng=reacquisition_rng,
                    )
                )

                replanned = (
                    _plan_nominal_with_structures(
                        start_q=current_q,
                        estimated_structures=(
                            active_structures
                        ),
                        seed=(
                            seed
                            + 30000
                            + step_index
                        ),
                    )
                )

                total_iterations += int(
                    replanned.iterations
                )

            if not replanned.success:
                stops += 1

                break

            replans += 1

            manager.register_replan(
                step_index
            )

            remaining_samples = max(
                8,
                EXECUTION_SAMPLES
                - min(
                    step_index,
                    EXECUTION_SAMPLES
                    - 8,
                ),
            )

            planned_path = (
                resample_joint_path(
                    replanned.path,
                    remaining_samples,
                )
            )

            current_q = (
                planned_path[
                    0
                ].copy()
            )

            cursor = 1

        if cursor >= (
            planned_path.shape[
                0
            ]
        ):
            completed = bool(
                np.allclose(
                    current_q,
                    goal_configuration(),
                    atol=1e-8,
                )
            )

            break

        next_q = (
            planned_path[
                cursor
            ]
        )

        (
            unsafe,
            collision,
        ) = (
            _evaluate_configuration_against_environment(
                q=next_q,
                environment=environment,
                step_index=step_index,
            )
        )

        unsafe_steps += int(
            unsafe
        )

        collision_steps += int(
            collision
        )

        executed_steps += 1

        current_q = (
            next_q.copy()
        )

        cursor += 1

        if np.allclose(
            current_q,
            goal_configuration(),
            atol=1e-8,
        ):
            completed = True

            break

    return ExecutionOutcome(
        initial_plan_success=True,
        completed=bool(
            completed
        ),
        safe_completion=bool(
            completed
            and unsafe_steps == 0
        ),
        executed_steps=int(
            executed_steps
        ),
        unsafe_steps=int(
            unsafe_steps
        ),
        collision_steps=int(
            collision_steps
        ),
        any_safety_violation=bool(
            unsafe_steps > 0
        ),
        any_collision=bool(
            collision_steps > 0
        ),
        replans=int(
            replans
        ),
        reacquisitions=int(
            reacquisitions
        ),
        stops=int(
            stops
        ),
        initial_blocked=False,
        final_progress=(
            1.0
            if completed
            else _final_progress(
                current_q
            )
        ),
        total_planning_iterations=int(
            total_iterations
        ),
    )


def _wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[
    float,
    float,
]:
    """Return Wilson confidence interval for a binomial proportion."""

    if total <= 0:
        return (
            0.0,
            0.0,
        )

    n = float(
        total
    )

    p = (
        float(
            successes
        )
        / n
    )

    z_squared = (
        z**2
    )

    denominator = (
        1.0
        + z_squared
        / n
    )

    centre = (
        p
        + z_squared
        / (
            2.0
            * n
        )
    ) / denominator

    half_width = (
        z
        / denominator
        * np.sqrt(
            (
                p
                * (
                    1.0
                    - p
                )
                / n
            )
            + z_squared
            / (
                4.0
                * n**2
            )
        )
    )

    return (
        float(
            max(
                0.0,
                centre
                - half_width,
            )
        ),
        float(
            min(
                1.0,
                centre
                + half_width,
            )
        ),
    )


def _summarise(
    outcomes: list[
        ExecutionOutcome
    ],
) -> dict:
    """Aggregate one strategy/scenario group."""

    total = len(
        outcomes
    )

    if total < 1:
        raise ValueError(
            "outcomes must contain at least one result."
        )

    completion_count = int(
        sum(
            outcome.completed
            for outcome in outcomes
        )
    )

    safe_completion_count = int(
        sum(
            outcome.safe_completion
            for outcome in outcomes
        )
    )

    violation_episode_count = int(
        sum(
            outcome.any_safety_violation
            for outcome in outcomes
        )
    )

    collision_episode_count = int(
        sum(
            outcome.any_collision
            for outcome in outcomes
        )
    )

    initial_block_count = int(
        sum(
            outcome.initial_blocked
            for outcome in outcomes
        )
    )

    total_executed_steps = int(
        sum(
            outcome.executed_steps
            for outcome in outcomes
        )
    )

    total_unsafe_steps = int(
        sum(
            outcome.unsafe_steps
            for outcome in outcomes
        )
    )

    completion_ci = (
        _wilson_interval(
            completion_count,
            total,
        )
    )

    safe_completion_ci = (
        _wilson_interval(
            safe_completion_count,
            total,
        )
    )

    if total_executed_steps == 0:
        unsafe_step_rate = 0.0

    else:
        unsafe_step_rate = float(
            total_unsafe_steps
            / total_executed_steps
        )

    return {
        "completion_rate": float(
            completion_count
            / total
        ),
        "completion_95_ci": [
            float(
                completion_ci[
                    0
                ]
            ),
            float(
                completion_ci[
                    1
                ]
            ),
        ],
        "safe_completion_rate": float(
            safe_completion_count
            / total
        ),
        "safe_completion_95_ci": [
            float(
                safe_completion_ci[
                    0
                ]
            ),
            float(
                safe_completion_ci[
                    1
                ]
            ),
        ],
        "safety_violation_episode_rate": float(
            violation_episode_count
            / total
        ),
        "collision_episode_rate": float(
            collision_episode_count
            / total
        ),
        "unsafe_executed_step_rate": float(
            unsafe_step_rate
        ),
        "initial_block_rate": float(
            initial_block_count
            / total
        ),
        "mean_replans": float(
            np.mean(
                [
                    outcome.replans
                    for outcome in outcomes
                ]
            )
        ),
        "mean_reacquisitions": float(
            np.mean(
                [
                    outcome.reacquisitions
                    for outcome in outcomes
                ]
            )
        ),
        "mean_stops": float(
            np.mean(
                [
                    outcome.stops
                    for outcome in outcomes
                ]
            )
        ),
        "mean_final_progress": float(
            np.mean(
                [
                    outcome.final_progress
                    for outcome in outcomes
                ]
            )
        ),
        "mean_executed_steps": float(
            np.mean(
                [
                    outcome.executed_steps
                    for outcome in outcomes
                ]
            )
        ),
        "mean_total_planning_iterations": float(
            np.mean(
                [
                    outcome
                    .total_planning_iterations
                    for outcome in outcomes
                ]
            )
        ),
    }


def run_benchmark(
    repetitions: int = DEFAULT_REPETITIONS,
    seed: int = DEFAULT_SEED,
    output_directory: Path = RESULT_DIRECTORY,
) -> dict:
    """Run final Phase 6 dynamic closed-loop comparison."""

    if repetitions < 1:
        raise ValueError(
            "repetitions must be positive."
        )

    _base_condition()

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows: list[
        dict
    ] = []

    summary = {
        "benchmark": (
            "phase6_closed_loop_execution"
        ),
        "repetitions_per_scenario": int(
            repetitions
        ),
        "base_seed": int(
            seed
        ),
        "execution_samples": int(
            EXECUTION_SAMPLES
        ),
        "maximum_execution_steps": int(
            MAX_EXECUTION_STEPS
        ),
        "initialisation": (
            "controlled nominal mean with non-zero covariance; "
            "routine observations are noisy except in the isolated "
            "recoverable structure-update scenario"
        ),
        "recoverable_structure_update": {
            "update_index": int(
                STRUCTURE_UPDATE_ONSET_INDEX
            ),
            "centre_m": (
                FROZEN_SECONDARY_CENTRE
                .tolist()
            ),
            "physical_radius_mm": float(
                SECONDARY_PHYSICAL_RADIUS
                * 1000.0
            ),
            "safety_margin_mm": float(
                SECONDARY_SAFETY_MARGIN
                * 1000.0
            ),
            "position_sigma_mm": float(
                SECONDARY_POSITION_SIGMA
                * 1000.0
            ),
            "development_future_unsafe_samples": 1,
            "development_future_collision_samples": 0,
            "development_replan_successes": 3,
            "development_replan_trials": 3,
            "frozen_before_final_evaluation": True,
        },
        "strategies": [
            "open_loop_nominal",
            "static_robust",
            "closed_loop_adaptive",
        ],
        "scenarios": {},
        "interpretation_boundary": (
            "Dynamic simulation evidence only. "
            "No clinical efficacy, patient safety, medical-device "
            "validation, or real-robot validation claim is supported."
        ),
    }

    for (
        scenario_index,
        scenario,
    ) in enumerate(
        scenario_names()
    ):
        grouped: dict[
            str,
            list[
                ExecutionOutcome
            ],
        ] = {
            "open_loop_nominal": [],
            "static_robust": [],
            "closed_loop_adaptive": [],
        }

        for repetition in range(
            repetitions
        ):
            episode_seed = (
                seed
                + scenario_index
                * 100000
                + repetition
                * 1000
            )

            environment = (
                make_environment(
                    scenario,
                    episode_seed,
                )
            )

            open_loop = (
                execute_open_loop_nominal(
                    environment,
                    episode_seed
                    + 11,
                )
            )

            static = (
                execute_static_robust(
                    environment,
                    episode_seed
                    + 22,
                )
            )

            adaptive = (
                execute_closed_loop_adaptive(
                    environment,
                    episode_seed
                    + 33,
                )
            )

            outcomes = {
                "open_loop_nominal": (
                    open_loop
                ),
                "static_robust": (
                    static
                ),
                "closed_loop_adaptive": (
                    adaptive
                ),
            }

            for (
                strategy,
                outcome,
            ) in outcomes.items():
                grouped[
                    strategy
                ].append(
                    outcome
                )

                all_rows.append(
                    {
                        "scenario": (
                            scenario
                        ),
                        "repetition": int(
                            repetition
                        ),
                        "episode_seed": int(
                            episode_seed
                        ),
                        "strategy": (
                            strategy
                        ),
                        "initial_plan_success": bool(
                            outcome
                            .initial_plan_success
                        ),
                        "completed": bool(
                            outcome.completed
                        ),
                        "safe_completion": bool(
                            outcome
                            .safe_completion
                        ),
                        "executed_steps": int(
                            outcome
                            .executed_steps
                        ),
                        "unsafe_steps": int(
                            outcome
                            .unsafe_steps
                        ),
                        "collision_steps": int(
                            outcome
                            .collision_steps
                        ),
                        "any_safety_violation": bool(
                            outcome
                            .any_safety_violation
                        ),
                        "any_collision": bool(
                            outcome
                            .any_collision
                        ),
                        "replans": int(
                            outcome.replans
                        ),
                        "reacquisitions": int(
                            outcome
                            .reacquisitions
                        ),
                        "stops": int(
                            outcome.stops
                        ),
                        "initial_blocked": bool(
                            outcome
                            .initial_blocked
                        ),
                        "final_progress": float(
                            outcome
                            .final_progress
                        ),
                        "total_planning_iterations": int(
                            outcome
                            .total_planning_iterations
                        ),
                    }
                )

        summary[
            "scenarios"
        ][
            scenario
        ] = {
            strategy: (
                _summarise(
                    strategy_outcomes
                )
            )
            for (
                strategy,
                strategy_outcomes,
            ) in grouped.items()
        }

    csv_path = (
        output_directory
        / "phase6_closed_loop_trials.csv"
    )

    json_path = (
        output_directory
        / "phase6_closed_loop_summary.json"
    )

    fieldnames = [
        "scenario",
        "repetition",
        "episode_seed",
        "strategy",
        "initial_plan_success",
        "completed",
        "safe_completion",
        "executed_steps",
        "unsafe_steps",
        "collision_steps",
        "any_safety_violation",
        "any_collision",
        "replans",
        "reacquisitions",
        "stops",
        "initial_blocked",
        "final_progress",
        "total_planning_iterations",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            all_rows
        )

    summary[
        "trial_csv"
    ] = str(
        csv_path
    )

    summary[
        "summary_json"
    ] = str(
        json_path
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            summary,
            json_file,
            indent=2,
        )

    return summary


def print_summary(
    summary: dict,
) -> None:
    """Print compact final Phase 6 evidence."""

    print(
        "\n"
        "=== Phase 6 Closed-Loop Execution Benchmark ==="
    )

    print(
        "Final repetitions per scenario: "
        f"{summary['repetitions_per_scenario']}"
    )

    print(
        "Initialisation: "
        f"{summary['initialisation']}"
    )

    structure = (
        summary[
            "recoverable_structure_update"
        ]
    )

    print(
        "Recoverable structure update: "
        f"index={structure['update_index']} | "
        f"radius={structure['physical_radius_mm']:.1f} mm | "
        f"margin={structure['safety_margin_mm']:.1f} mm | "
        f"sigma={structure['position_sigma_mm']:.1f} mm"
    )

    for (
        scenario,
        strategies,
    ) in (
        summary[
            "scenarios"
        ].items()
    ):
        print(
            "\n"
            f"--- {scenario} ---"
        )

        for (
            strategy,
            metrics,
        ) in strategies.items():
            print(
                f"{strategy}: "
                f"complete="
                f"{100.0 * metrics['completion_rate']:.1f}% | "
                f"safe_complete="
                f"{100.0 * metrics['safe_completion_rate']:.1f}% | "
                f"unsafe_episode="
                f"{100.0 * metrics['safety_violation_episode_rate']:.1f}% | "
                f"unsafe_steps="
                f"{100.0 * metrics['unsafe_executed_step_rate']:.2f}% | "
                f"collision_ep="
                f"{100.0 * metrics['collision_episode_rate']:.1f}% | "
                f"block="
                f"{100.0 * metrics['initial_block_rate']:.1f}% | "
                f"replans="
                f"{metrics['mean_replans']:.2f} | "
                f"reacq="
                f"{metrics['mean_reacquisitions']:.2f} | "
                f"stops="
                f"{metrics['mean_stops']:.2f} | "
                f"progress="
                f"{100.0 * metrics['mean_final_progress']:.1f}%"
            )

    print(
        "\nArtifacts:"
    )

    print(
        summary[
            "trial_csv"
        ]
    )

    print(
        summary[
            "summary_json"
        ]
    )

    print(
        "\nInterpretation: dynamic simulation evidence only."
    )


def main() -> None:
    """Command-line entry point."""

    summary = run_benchmark()

    print_summary(
        summary
    )


if __name__ == "__main__":
    main()