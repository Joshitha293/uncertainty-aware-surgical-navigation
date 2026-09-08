"""Controlled Phase 6 benchmark for anisotropic chance-constrained planning.

The benchmark compares three planner-facing uncertainty treatments:

1. Deterministic estimated geometry
   - estimated anatomical centre;
   - physical radius;
   - base safety margin;
   - no localisation-uncertainty compensation.

2. Principal-sigma scalar inflation
   - the established project baseline;
   - base margin + k * largest positional sigma.

3. Anisotropic chance-constrained planning
   - preserves the complete 3 x 3 covariance;
   - projects covariance along local clearance directions;
   - accepts/rejects robot configurations using a probability threshold.

Controlled mechanism test
-------------------------

Two uncertainty conditions deliberately use:

- the same estimated anatomical centre;
- the same physical radius;
- the same base safety margin;
- the same covariance eigenvalues;
- the same principal sigma.

Only covariance orientation changes.

Tangential condition:
    largest uncertainty axis is approximately aligned with the local
    instrument-shaft direction.

Radial condition:
    largest uncertainty axis is aligned with the nominal clearance direction.

The scalar principal-sigma planner therefore receives effectively identical
protected geometry in both conditions.

The anisotropic planner does not.

Hidden ground truth
-------------------

Ground-truth anatomical centres are sampled only after the planners have
produced their paths. Hidden truth is then used for independent trajectory
safety evaluation and is never supplied to a runtime planning function.

This is a simulation-only controlled mechanism experiment. The estimated
probabilities are not clinical collision-risk estimates.
"""

from __future__ import annotations

import csv
import json

from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from time import perf_counter

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
    edge_is_chance_constrained_safe,
    evaluate_joint_path_chance_constraint,
    plan_rrt_chance_constrained,
)
from src.robotics.instrument import (
    SurgicalInstrument,
)
from src.robotics.planner import (
    PlanningResult,
    edge_is_safe,
    path_cost,
    plan_rrt,
)
from src.robotics.risk_aware_planning import (
    ChanceConstraintConfig,
)
from src.robotics.safety import (
    evaluate_instrument_safety,
)


INSTRUMENT_RADIUS = 0.006
PROXIMAL_LENGTH = 0.10

PHYSICAL_RADIUS = 0.015
BASE_SAFETY_MARGIN = 0.008

SIGMA_MAJOR = 0.010
SIGMA_MIDDLE = 0.002
SIGMA_MINOR = 0.001

CHANCE_THRESHOLD = 0.05

SCALAR_SIGMA_MULTIPLIER = float(
    NormalDist().inv_cdf(
        1.0 - CHANCE_THRESHOLD
    )
)

DIRECT_EDGE_RESOLUTION = 31
PLANNER_EDGE_RESOLUTION = 16

SHAFT_SAMPLE_SPACING = 0.004

DEFAULT_REPETITIONS = 300
DEFAULT_SEED = 66001

RESULT_DIRECTORY = Path(
    "results/phase6_chance_constraint"
)


@dataclass(frozen=True)
class OrientationCondition:
    """One covariance-orientation condition."""

    name: str

    estimate: EstimatedStructure

    nominal_clearance_direction: np.ndarray

    nominal_shaft_direction: np.ndarray

    selected_clearance: float


@dataclass(frozen=True)
class HiddenTruthEvaluation:
    """Independent hidden-truth evaluation of one planned path."""

    minimum_surface_clearance: float

    minimum_safety_clearance: float

    collision: bool

    safety_margin_violation: bool


def make_instrument() -> SurgicalInstrument:
    """Create the established RCM-constrained instrument."""

    return SurgicalInstrument(
        rcm_position=np.zeros(
            3,
            dtype=float,
        )
    )


def start_configuration() -> np.ndarray:
    """Return controlled benchmark start configuration."""

    return np.array(
        [
            0.0,
            0.0,
            0.10,
            0.0,
        ],
        dtype=float,
    )


def goal_configuration() -> np.ndarray:
    """Return controlled benchmark goal configuration."""

    return np.array(
        [
            np.deg2rad(
                35.0
            ),
            np.deg2rad(
                25.0
            ),
            0.25,
            0.0,
        ],
        dtype=float,
    )


def _normalise(
    vector: np.ndarray,
) -> np.ndarray:
    """Return unit vector."""

    vector = np.asarray(
        vector,
        dtype=float,
    )

    magnitude = float(
        np.linalg.norm(
            vector
        )
    )

    if magnitude <= 1e-12:
        raise ValueError(
            "Cannot normalise a zero vector."
        )

    return (
        vector
        / magnitude
    )


def _benchmark_local_frame(
    instrument: SurgicalInstrument,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Create a local orthonormal frame around the nominal direct path.

    Returns
    -------
    tuple
        reference_point,
        clearance_normal,
        binormal,
        shaft_direction
    """

    midpoint_q = 0.5 * (
        start_configuration()
        + goal_configuration()
    )

    shaft_start, shaft_end = (
        instrument.shaft_segment(
            midpoint_q,
            proximal_length=(
                PROXIMAL_LENGTH
            ),
        )
    )

    shaft_direction = _normalise(
        shaft_end
        - shaft_start
    )

    # Select a point on the distal part of the represented shaft.
    reference_point = (
        shaft_start
        + 0.72
        * (
            shaft_end
            - shaft_start
        )
    )

    world_z = np.array(
        [
            0.0,
            0.0,
            1.0,
        ],
        dtype=float,
    )

    clearance_normal = np.cross(
        shaft_direction,
        world_z,
    )

    if (
        np.linalg.norm(
            clearance_normal
        )
        <= 1e-8
    ):
        clearance_normal = np.cross(
            shaft_direction,
            np.array(
                [
                    0.0,
                    1.0,
                    0.0,
                ],
                dtype=float,
            ),
        )

    clearance_normal = _normalise(
        clearance_normal
    )

    binormal = _normalise(
        np.cross(
            shaft_direction,
            clearance_normal,
        )
    )

    return (
        reference_point,
        clearance_normal,
        binormal,
        shaft_direction,
    )


def _covariance_from_axes(
    axis_1: np.ndarray,
    sigma_1: float,
    axis_2: np.ndarray,
    sigma_2: float,
    axis_3: np.ndarray,
    sigma_3: float,
) -> np.ndarray:
    """Construct covariance from orthonormal principal axes."""

    rotation = np.column_stack(
        [
            _normalise(
                axis_1
            ),
            _normalise(
                axis_2
            ),
            _normalise(
                axis_3
            ),
        ]
    )

    covariance = (
        rotation
        @ np.diag(
            [
                sigma_1**2,
                sigma_2**2,
                sigma_3**2,
            ]
        )
        @ rotation.T
    )

    return 0.5 * (
        covariance
        + covariance.T
    )


def build_orientation_conditions(
    selected_clearance: float,
) -> tuple[
    OrientationCondition,
    OrientationCondition,
]:
    """Build matched tangential and radial covariance conditions."""

    if selected_clearance <= 0.0:
        raise ValueError(
            "selected_clearance must be positive."
        )

    instrument = make_instrument()

    (
        reference_point,
        clearance_normal,
        binormal,
        shaft_direction,
    ) = _benchmark_local_frame(
        instrument
    )

    estimated_centre = (
        reference_point
        + selected_clearance
        * clearance_normal
    )

    # Largest uncertainty approximately tangent to the local shaft.
    tangential_covariance = (
        _covariance_from_axes(
            shaft_direction,
            SIGMA_MAJOR,
            binormal,
            SIGMA_MIDDLE,
            clearance_normal,
            SIGMA_MINOR,
        )
    )

    # Same eigenvalues, but largest uncertainty now points across clearance.
    radial_covariance = (
        _covariance_from_axes(
            clearance_normal,
            SIGMA_MAJOR,
            binormal,
            SIGMA_MIDDLE,
            shaft_direction,
            SIGMA_MINOR,
        )
    )

    tangential = OrientationCondition(
        name="tangential_major_axis",
        estimate=EstimatedStructure(
            estimated_centre=(
                estimated_centre.copy()
            ),
            physical_radius=(
                PHYSICAL_RADIUS
            ),
            base_safety_margin=(
                BASE_SAFETY_MARGIN
            ),
            uncertainty=PositionUncertainty(
                covariance=(
                    tangential_covariance
                )
            ),
        ),
        nominal_clearance_direction=(
            clearance_normal.copy()
        ),
        nominal_shaft_direction=(
            shaft_direction.copy()
        ),
        selected_clearance=float(
            selected_clearance
        ),
    )

    radial = OrientationCondition(
        name="radial_major_axis",
        estimate=EstimatedStructure(
            estimated_centre=(
                estimated_centre.copy()
            ),
            physical_radius=(
                PHYSICAL_RADIUS
            ),
            base_safety_margin=(
                BASE_SAFETY_MARGIN
            ),
            uncertainty=PositionUncertainty(
                covariance=(
                    radial_covariance
                )
            ),
        ),
        nominal_clearance_direction=(
            clearance_normal.copy()
        ),
        nominal_shaft_direction=(
            shaft_direction.copy()
        ),
        selected_clearance=float(
            selected_clearance
        ),
    )

    return (
        tangential,
        radial,
    )


def _deterministic_structure(
    estimate: EstimatedStructure,
) -> SphericalStructure:
    """Create non-inflated estimated geometry."""

    return SphericalStructure(
        centre=(
            estimate.estimated_centre.copy()
        ),
        physical_radius=(
            estimate.physical_radius
        ),
        safety_margin=(
            estimate.base_safety_margin
        ),
    )


def _scalar_structure(
    estimate: EstimatedStructure,
) -> SphericalStructure:
    """Create established principal-sigma inflated geometry."""

    return (
        estimate.uncertainty_aware_structure(
            sigma_multiplier=(
                SCALAR_SIGMA_MULTIPLIER
            )
        )
    )


def chance_config() -> ChanceConstraintConfig:
    """Return primary Phase 6 chance-constraint settings."""

    return ChanceConstraintConfig(
        max_point_violation_probability=(
            CHANCE_THRESHOLD
        ),
        max_path_union_bound_probability=None,
        sample_spacing=(
            SHAFT_SAMPLE_SPACING
        ),
    )


def direct_edge_decisions(
    condition: OrientationCondition,
) -> dict[str, bool]:
    """Evaluate direct-path acceptance for all three strategies."""

    instrument = make_instrument()

    start_q = start_configuration()
    goal_q = goal_configuration()

    deterministic_safe = edge_is_safe(
        instrument=instrument,
        q_start=start_q,
        q_goal=goal_q,
        structures=(
            _deterministic_structure(
                condition.estimate
            ),
        ),
        instrument_radius=(
            INSTRUMENT_RADIUS
        ),
        proximal_length=(
            PROXIMAL_LENGTH
        ),
        resolution=(
            DIRECT_EDGE_RESOLUTION
        ),
    )

    scalar_safe = edge_is_safe(
        instrument=instrument,
        q_start=start_q,
        q_goal=goal_q,
        structures=(
            _scalar_structure(
                condition.estimate
            ),
        ),
        instrument_radius=(
            INSTRUMENT_RADIUS
        ),
        proximal_length=(
            PROXIMAL_LENGTH
        ),
        resolution=(
            DIRECT_EDGE_RESOLUTION
        ),
    )

    chance_safe = (
        edge_is_chance_constrained_safe(
            instrument=instrument,
            q_start=start_q,
            q_goal=goal_q,
            estimated_structures=(
                condition.estimate,
            ),
            instrument_radius=(
                INSTRUMENT_RADIUS
            ),
            chance_config=(
                chance_config()
            ),
            proximal_length=(
                PROXIMAL_LENGTH
            ),
            shaft_sample_spacing=(
                SHAFT_SAMPLE_SPACING
            ),
            resolution=(
                DIRECT_EDGE_RESOLUTION
            ),
        )
    )

    return {
        "deterministic": bool(
            deterministic_safe
        ),
        "scalar_principal_sigma": bool(
            scalar_safe
        ),
        "anisotropic_chance": bool(
            chance_safe
        ),
    }


def select_disagreement_clearance(
) -> tuple[
    float,
    tuple[
        OrientationCondition,
        OrientationCondition,
    ],
]:
    """Find a controlled geometry exposing covariance-orientation behaviour.

    The search is performed entirely using planner-facing estimated geometry.

    Hidden ground truth does not participate in this design selection.

    Desired signature:

        deterministic:
            accepts both orientations

        scalar principal sigma:
            rejects both orientations

        anisotropic chance:
            accepts tangential-major uncertainty
            rejects radial-major uncertainty

    This is explicitly a mechanism-demonstration scenario rather than a
    population-level performance benchmark.
    """

    candidate_clearances = np.arange(
        0.050,
        0.0711,
        0.0005,
        dtype=float,
    )

    for candidate in (
        candidate_clearances
    ):
        conditions = (
            build_orientation_conditions(
                float(
                    candidate
                )
            )
        )

        tangential, radial = (
            conditions
        )

        tangent_decisions = (
            direct_edge_decisions(
                tangential
            )
        )

        radial_decisions = (
            direct_edge_decisions(
                radial
            )
        )

        desired_signature = (
            tangent_decisions[
                "deterministic"
            ]
            and radial_decisions[
                "deterministic"
            ]
            and not tangent_decisions[
                "scalar_principal_sigma"
            ]
            and not radial_decisions[
                "scalar_principal_sigma"
            ]
            and tangent_decisions[
                "anisotropic_chance"
            ]
            and not radial_decisions[
                "anisotropic_chance"
            ]
        )

        if desired_signature:
            return (
                float(
                    candidate
                ),
                conditions,
            )

    raise RuntimeError(
        "Could not identify the controlled Phase 6 "
        "strategy-disagreement geometry."
    )


def _run_deterministic_planner(
    condition: OrientationCondition,
    seed: int,
) -> tuple[
    PlanningResult,
    float,
]:
    """Run deterministic estimated-geometry RRT."""

    instrument = make_instrument()

    start_time = perf_counter()

    result = plan_rrt(
        instrument=instrument,
        start_q=(
            start_configuration()
        ),
        goal_q=(
            goal_configuration()
        ),
        structures=(
            _deterministic_structure(
                condition.estimate
            ),
        ),
        instrument_radius=(
            INSTRUMENT_RADIUS
        ),
        proximal_length=(
            PROXIMAL_LENGTH
        ),
        max_iterations=2500,
        step_size=0.08,
        goal_bias=0.20,
        edge_resolution=(
            PLANNER_EDGE_RESOLUTION
        ),
        seed=seed,
    )

    elapsed = (
        perf_counter()
        - start_time
    )

    return (
        result,
        float(
            elapsed
        ),
    )


def _run_scalar_planner(
    condition: OrientationCondition,
    seed: int,
) -> tuple[
    PlanningResult,
    float,
]:
    """Run principal-sigma uncertainty-inflated RRT."""

    instrument = make_instrument()

    start_time = perf_counter()

    result = plan_rrt(
        instrument=instrument,
        start_q=(
            start_configuration()
        ),
        goal_q=(
            goal_configuration()
        ),
        structures=(
            _scalar_structure(
                condition.estimate
            ),
        ),
        instrument_radius=(
            INSTRUMENT_RADIUS
        ),
        proximal_length=(
            PROXIMAL_LENGTH
        ),
        max_iterations=2500,
        step_size=0.08,
        goal_bias=0.20,
        edge_resolution=(
            PLANNER_EDGE_RESOLUTION
        ),
        seed=seed,
    )

    elapsed = (
        perf_counter()
        - start_time
    )

    return (
        result,
        float(
            elapsed
        ),
    )


def _run_chance_planner(
    condition: OrientationCondition,
    seed: int,
) -> tuple[
    PlanningResult,
    float,
]:
    """Run anisotropic chance-constrained RRT."""

    instrument = make_instrument()

    start_time = perf_counter()

    result = (
        plan_rrt_chance_constrained(
            instrument=instrument,
            start_q=(
                start_configuration()
            ),
            goal_q=(
                goal_configuration()
            ),
            estimated_structures=(
                condition.estimate,
            ),
            instrument_radius=(
                INSTRUMENT_RADIUS
            ),
            chance_config=(
                chance_config()
            ),
            planner_config=(
                ChanceConstrainedRRTConfig(
                    max_iterations=2500,
                    step_size=0.08,
                    goal_bias=0.20,
                    edge_resolution=(
                        PLANNER_EDGE_RESOLUTION
                    ),
                    proximal_length=(
                        PROXIMAL_LENGTH
                    ),
                    shaft_sample_spacing=(
                        SHAFT_SAMPLE_SPACING
                    ),
                    seed=seed,
                )
            ),
        )
    )

    elapsed = (
        perf_counter()
        - start_time
    )

    return (
        result,
        float(
            elapsed
        ),
    )


def _estimated_path_risk(
    condition: OrientationCondition,
    result: PlanningResult,
) -> float | None:
    """Evaluate one result using the same estimated anisotropic model."""

    if not result.success:
        return None

    evaluation = (
        evaluate_joint_path_chance_constraint(
            instrument=(
                make_instrument()
            ),
            path=result.path,
            estimated_structures=(
                condition.estimate,
            ),
            instrument_radius=(
                INSTRUMENT_RADIUS
            ),
            chance_config=(
                chance_config()
            ),
            proximal_length=(
                PROXIMAL_LENGTH
            ),
            shaft_sample_spacing=(
                SHAFT_SAMPLE_SPACING
            ),
            edge_resolution=(
                PLANNER_EDGE_RESOLUTION
            ),
        )
    )

    return float(
        evaluation.maximum_point_violation_probability
    )


def _plan_all_strategies(
    condition: OrientationCondition,
    seed: int,
) -> dict[str, dict]:
    """Plan all strategies before hidden ground truth is generated."""

    direct = (
        direct_edge_decisions(
            condition
        )
    )

    deterministic_result, deterministic_time = (
        _run_deterministic_planner(
            condition,
            seed,
        )
    )

    scalar_result, scalar_time = (
        _run_scalar_planner(
            condition,
            seed,
        )
    )

    chance_result, chance_time = (
        _run_chance_planner(
            condition,
            seed,
        )
    )

    results = {
        "deterministic": (
            deterministic_result,
            deterministic_time,
        ),
        "scalar_principal_sigma": (
            scalar_result,
            scalar_time,
        ),
        "anisotropic_chance": (
            chance_result,
            chance_time,
        ),
    }

    output: dict[str, dict] = {}

    for strategy, (
        planning_result,
        elapsed,
    ) in results.items():
        output[strategy] = {
            "result": planning_result,
            "planning_time_seconds": float(
                elapsed
            ),
            "direct_edge_accepted": bool(
                direct[
                    strategy
                ]
            ),
            "path_cost": (
                float(
                    path_cost(
                        planning_result.path
                    )
                )
                if planning_result.success
                else None
            ),
            "estimated_max_point_risk": (
                _estimated_path_risk(
                    condition,
                    planning_result,
                )
            ),
        }

    return output


def _evaluate_path_against_truth(
    result: PlanningResult,
    truth_structure: SphericalStructure,
    *,
    edge_resolution: int = 40,
) -> HiddenTruthEvaluation | None:
    """Evaluate a completed plan against independently hidden truth."""

    if not result.success:
        return None

    if edge_resolution < 2:
        raise ValueError(
            "edge_resolution must be at least 2."
        )

    instrument = make_instrument()

    minimum_surface_clearance = float(
        "inf"
    )

    minimum_safety_clearance = float(
        "inf"
    )

    collision = False
    safety_violation = False

    path = np.asarray(
        result.path,
        dtype=float,
    )

    if (
        path.ndim != 2
        or path.shape[1] != 4
        or path.shape[0] < 1
    ):
        raise ValueError(
            "Successful path must have shape (N, 4)."
        )

    if path.shape[0] == 1:
        configurations = [
            path[0]
        ]

    else:
        configurations = []

        for edge_index in range(
            path.shape[0] - 1
        ):
            edge_samples = np.linspace(
                path[
                    edge_index
                ],
                path[
                    edge_index + 1
                ],
                num=edge_resolution,
                dtype=float,
            )

            if edge_index > 0:
                edge_samples = (
                    edge_samples[
                        1:
                    ]
                )

            configurations.extend(
                edge_samples
            )

    for configuration in (
        configurations
    ):
        shaft_start, shaft_end = (
            instrument.shaft_segment(
                configuration,
                proximal_length=(
                    PROXIMAL_LENGTH
                ),
            )
        )

        safety = (
            evaluate_instrument_safety(
                shaft_start=(
                    shaft_start
                ),
                shaft_end=(
                    shaft_end
                ),
                structures=(
                    truth_structure,
                ),
                instrument_radius=(
                    INSTRUMENT_RADIUS
                ),
            )
        )

        minimum_surface_clearance = min(
            minimum_surface_clearance,
            safety.minimum_surface_clearance,
        )

        minimum_safety_clearance = min(
            minimum_safety_clearance,
            safety.minimum_safety_clearance,
        )

        collision = bool(
            collision
            or safety.collision
        )

        safety_violation = bool(
            safety_violation
            or safety.safety_margin_violation
        )

    return HiddenTruthEvaluation(
        minimum_surface_clearance=float(
            minimum_surface_clearance
        ),
        minimum_safety_clearance=float(
            minimum_safety_clearance
        ),
        collision=bool(
            collision
        ),
        safety_margin_violation=bool(
            safety_violation
        ),
    )


def _sample_hidden_truth(
    condition: OrientationCondition,
    rng: np.random.Generator,
) -> SphericalStructure:
    """Sample hidden truth consistently with estimate = truth + error."""

    localisation_error = (
        rng.multivariate_normal(
            mean=np.zeros(
                3,
                dtype=float,
            ),
            cov=(
                condition
                .estimate
                .uncertainty
                .covariance
            ),
        )
    )

    true_centre = (
        condition.estimate.estimated_centre
        - localisation_error
    )

    return SphericalStructure(
        centre=true_centre,
        physical_radius=(
            condition.estimate.physical_radius
        ),
        safety_margin=(
            condition.estimate.base_safety_margin
        ),
    )


def _wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[
    float | None,
    float | None,
]:
    """Return Wilson interval for a binomial proportion."""

    if total <= 0:
        return (
            None,
            None,
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
            + (
                z_squared
                / (
                    4.0
                    * n**2
                )
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


def _summarise_strategy_records(
    records: list[dict],
    planning: dict,
) -> dict:
    """Aggregate hidden-truth outcomes for one strategy."""

    total = len(
        records
    )

    safe_count = sum(
        int(
            row[
                "safe_navigation"
            ]
        )
        for row in records
    )

    planning_success = bool(
        planning[
            "result"
        ].success
    )

    if planning_success:
        successful_records = [
            row
            for row in records
            if row[
                "planning_success"
            ]
        ]

        safety_successes = sum(
            int(
                not row[
                    "safety_margin_violation"
                ]
            )
            for row in successful_records
        )

        collision_count = sum(
            int(
                row[
                    "collision"
                ]
            )
            for row in successful_records
        )

        safety_violation_count = sum(
            int(
                row[
                    "safety_margin_violation"
                ]
            )
            for row in successful_records
        )

        mean_clearance_mm = float(
            np.mean(
                [
                    row[
                        "minimum_safety_clearance_mm"
                    ]
                    for row in successful_records
                ]
            )
        )

        conditional_safety_rate = (
            float(
                safety_successes
            )
            / float(
                len(
                    successful_records
                )
            )
        )

        collision_rate = (
            float(
                collision_count
            )
            / float(
                len(
                    successful_records
                )
            )
        )

        violation_rate = (
            float(
                safety_violation_count
            )
            / float(
                len(
                    successful_records
                )
            )
        )

    else:
        conditional_safety_rate = None
        collision_rate = None
        violation_rate = None
        mean_clearance_mm = None

    safe_rate = (
        float(
            safe_count
        )
        / float(
            total
        )
    )

    ci_low, ci_high = (
        _wilson_interval(
            safe_count,
            total,
        )
    )

    return {
        "direct_edge_accepted": bool(
            planning[
                "direct_edge_accepted"
            ]
        ),
        "planning_success": bool(
            planning_success
        ),
        "planning_iterations": int(
            planning[
                "result"
            ].iterations
        ),
        "planning_time_seconds": float(
            planning[
                "planning_time_seconds"
            ]
        ),
        "path_cost": (
            planning[
                "path_cost"
            ]
        ),
        "estimated_max_point_risk": (
            planning[
                "estimated_max_point_risk"
            ]
        ),
        "safe_navigation_rate": float(
            safe_rate
        ),
        "safe_navigation_95_ci": [
            ci_low,
            ci_high,
        ],
        "conditional_truth_safety_rate_given_plan": (
            conditional_safety_rate
        ),
        "collision_rate_given_plan": (
            collision_rate
        ),
        "safety_violation_rate_given_plan": (
            violation_rate
        ),
        "mean_minimum_truth_safety_clearance_mm": (
            mean_clearance_mm
        ),
    }


def _paired_difference(
    records_a: list[dict],
    records_b: list[dict],
) -> dict:
    """Summarise paired safe-navigation outcomes."""

    if len(
        records_a
    ) != len(
        records_b
    ):
        raise ValueError(
            "Paired records must have equal length."
        )

    safe_a = np.asarray(
        [
            bool(
                row[
                    "safe_navigation"
                ]
            )
            for row in records_a
        ],
        dtype=bool,
    )

    safe_b = np.asarray(
        [
            bool(
                row[
                    "safe_navigation"
                ]
            )
            for row in records_b
        ],
        dtype=bool,
    )

    difference = float(
        np.mean(
            safe_a.astype(
                float
            )
            - safe_b.astype(
                float
            )
        )
    )

    return {
        "safe_navigation_rate_difference": (
            difference
        ),
        "difference_percentage_points": (
            100.0
            * difference
        ),
        "first_only_safe_count": int(
            np.sum(
                safe_a
                & ~safe_b
            )
        ),
        "second_only_safe_count": int(
            np.sum(
                ~safe_a
                & safe_b
            )
        ),
    }


def run_benchmark(
    repetitions: int = DEFAULT_REPETITIONS,
    seed: int = DEFAULT_SEED,
    output_directory: Path = RESULT_DIRECTORY,
) -> dict:
    """Run controlled Phase 6 anisotropy benchmark."""

    if repetitions < 1:
        raise ValueError(
            "repetitions must be positive."
        )

    (
        selected_clearance,
        conditions,
    ) = (
        select_disagreement_clearance()
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    trial_records: list[
        dict
    ] = []

    summary: dict = {
        "benchmark": (
            "phase6_chance_constraint"
        ),
        "repetitions_per_condition": int(
            repetitions
        ),
        "base_seed": int(
            seed
        ),
        "chance_threshold": float(
            CHANCE_THRESHOLD
        ),
        "scalar_sigma_multiplier": float(
            SCALAR_SIGMA_MULTIPLIER
        ),
        "selected_clearance_mm": float(
            1000.0
            * selected_clearance
        ),
        "covariance_principal_sigmas_mm": [
            float(
                1000.0
                * SIGMA_MINOR
            ),
            float(
                1000.0
                * SIGMA_MIDDLE
            ),
            float(
                1000.0
                * SIGMA_MAJOR
            ),
        ],
        "conditions": {},
        "interpretation_boundary": (
            "Controlled simulation mechanism evidence only; "
            "not a clinical collision probability or "
            "medical-device safety validation."
        ),
    }

    for condition_index, condition in enumerate(
        conditions
    ):
        planning_seed = (
            seed
            + 1000
            * condition_index
        )

        planning = (
            _plan_all_strategies(
                condition=condition,
                seed=planning_seed,
            )
        )

        rng = np.random.default_rng(
            seed
            + 100000
            * (
                condition_index
                + 1
            )
        )

        records_by_strategy: dict[
            str,
            list[dict],
        ] = {
            "deterministic": [],
            "scalar_principal_sigma": [],
            "anisotropic_chance": [],
        }

        for repetition in range(
            repetitions
        ):
            hidden_truth = (
                _sample_hidden_truth(
                    condition,
                    rng,
                )
            )

            true_centre = (
                hidden_truth.centre
            )

            for strategy, strategy_plan in (
                planning.items()
            ):
                result = (
                    strategy_plan[
                        "result"
                    ]
                )

                truth_evaluation = (
                    _evaluate_path_against_truth(
                        result,
                        hidden_truth,
                    )
                )

                if truth_evaluation is None:
                    collision = False
                    safety_violation = False

                    minimum_surface_mm = None
                    minimum_safety_mm = None

                    safe_navigation = False

                else:
                    collision = bool(
                        truth_evaluation.collision
                    )

                    safety_violation = bool(
                        truth_evaluation
                        .safety_margin_violation
                    )

                    minimum_surface_mm = float(
                        1000.0
                        * truth_evaluation
                        .minimum_surface_clearance
                    )

                    minimum_safety_mm = float(
                        1000.0
                        * truth_evaluation
                        .minimum_safety_clearance
                    )

                    safe_navigation = bool(
                        result.success
                        and not safety_violation
                    )

                row = {
                    "condition": (
                        condition.name
                    ),
                    "repetition": int(
                        repetition
                    ),
                    "strategy": strategy,
                    "planning_success": bool(
                        result.success
                    ),
                    "direct_edge_accepted": bool(
                        strategy_plan[
                            "direct_edge_accepted"
                        ]
                    ),
                    "planning_iterations": int(
                        result.iterations
                    ),
                    "planning_time_seconds": float(
                        strategy_plan[
                            "planning_time_seconds"
                        ]
                    ),
                    "path_cost": (
                        strategy_plan[
                            "path_cost"
                        ]
                    ),
                    "estimated_max_point_risk": (
                        strategy_plan[
                            "estimated_max_point_risk"
                        ]
                    ),
                    "true_centre_x": float(
                        true_centre[
                            0
                        ]
                    ),
                    "true_centre_y": float(
                        true_centre[
                            1
                        ]
                    ),
                    "true_centre_z": float(
                        true_centre[
                            2
                        ]
                    ),
                    "minimum_surface_clearance_mm": (
                        minimum_surface_mm
                    ),
                    "minimum_safety_clearance_mm": (
                        minimum_safety_mm
                    ),
                    "collision": bool(
                        collision
                    ),
                    "safety_margin_violation": bool(
                        safety_violation
                    ),
                    "safe_navigation": bool(
                        safe_navigation
                    ),
                }

                trial_records.append(
                    row
                )

                records_by_strategy[
                    strategy
                ].append(
                    row
                )

        eigenvalues = np.linalg.eigvalsh(
            condition
            .estimate
            .uncertainty
            .covariance
        )

        condition_summary = {
            "estimated_centre_m": (
                condition
                .estimate
                .estimated_centre
                .tolist()
            ),
            "covariance_eigenvalues_m2": (
                eigenvalues.tolist()
            ),
            "principal_sigma_mm": float(
                1000.0
                * condition
                .estimate
                .uncertainty
                .principal_sigma
            ),
            "strategies": {},
        }

        for strategy in (
            records_by_strategy
        ):
            condition_summary[
                "strategies"
            ][
                strategy
            ] = (
                _summarise_strategy_records(
                    records_by_strategy[
                        strategy
                    ],
                    planning[
                        strategy
                    ],
                )
            )

        condition_summary[
            "paired_comparisons"
        ] = {
            "chance_minus_deterministic": (
                _paired_difference(
                    records_by_strategy[
                        "anisotropic_chance"
                    ],
                    records_by_strategy[
                        "deterministic"
                    ],
                )
            ),
            "chance_minus_scalar": (
                _paired_difference(
                    records_by_strategy[
                        "anisotropic_chance"
                    ],
                    records_by_strategy[
                        "scalar_principal_sigma"
                    ],
                )
            ),
        }

        summary[
            "conditions"
        ][
            condition.name
        ] = condition_summary

    csv_path = (
        output_directory
        / "phase6_chance_constraint_trials.csv"
    )

    json_path = (
        output_directory
        / "phase6_chance_constraint_summary.json"
    )

    fieldnames = [
        "condition",
        "repetition",
        "strategy",
        "planning_success",
        "direct_edge_accepted",
        "planning_iterations",
        "planning_time_seconds",
        "path_cost",
        "estimated_max_point_risk",
        "true_centre_x",
        "true_centre_y",
        "true_centre_z",
        "minimum_surface_clearance_mm",
        "minimum_safety_clearance_mm",
        "collision",
        "safety_margin_violation",
        "safe_navigation",
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

        for row in trial_records:
            clean_row = {
                key: (
                    ""
                    if value is None
                    else value
                )
                for key, value in (
                    row.items()
                )
            }

            writer.writerow(
                clean_row
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

    return summary


def _format_optional(
    value: float | None,
    digits: int = 3,
) -> str:
    if value is None:
        return "N/A"

    return (
        f"{value:.{digits}f}"
    )


def print_summary(
    summary: dict,
) -> None:
    """Print compact Phase 6 benchmark evidence."""

    print(
        "\n"
        "=== Phase 6 Chance-Constrained Planning Benchmark ==="
    )

    print(
        "Controlled disagreement clearance: "
        f"{summary['selected_clearance_mm']:.3f} mm"
    )

    print(
        "Principal sigmas: "
        f"{summary['covariance_principal_sigmas_mm']} mm"
    )

    print(
        "Chance threshold: "
        f"{summary['chance_threshold']:.3f}"
    )

    print(
        "Scalar multiplier: "
        f"{summary['scalar_sigma_multiplier']:.6f}"
    )

    for condition_name, condition in (
        summary[
            "conditions"
        ].items()
    ):
        print(
            "\n"
            f"--- {condition_name} ---"
        )

        print(
            "Principal sigma: "
            f"{condition['principal_sigma_mm']:.3f} mm"
        )

        for strategy, metrics in (
            condition[
                "strategies"
            ].items()
        ):
            safe_ci = (
                metrics[
                    "safe_navigation_95_ci"
                ]
            )

            print(
                f"{strategy}: "
                f"direct={metrics['direct_edge_accepted']} | "
                f"plan={metrics['planning_success']} | "
                f"iters={metrics['planning_iterations']} | "
                f"cost={_format_optional(metrics['path_cost'])} | "
                f"risk={_format_optional(metrics['estimated_max_point_risk'], 4)} | "
                f"safe={100.0 * metrics['safe_navigation_rate']:.1f}% "
                f"[{100.0 * safe_ci[0]:.1f}, "
                f"{100.0 * safe_ci[1]:.1f}]"
            )

        chance_vs_det = (
            condition[
                "paired_comparisons"
            ][
                "chance_minus_deterministic"
            ]
        )

        chance_vs_scalar = (
            condition[
                "paired_comparisons"
            ][
                "chance_minus_scalar"
            ]
        )

        print(
            "Chance - deterministic safe-navigation: "
            f"{chance_vs_det['difference_percentage_points']:+.2f} pp"
        )

        print(
            "Chance - scalar safe-navigation: "
            f"{chance_vs_scalar['difference_percentage_points']:+.2f} pp"
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
        "\nInterpretation: controlled simulation mechanism evidence only."
    )


def main() -> None:
    """Run benchmark from command line."""

    summary = run_benchmark()

    print_summary(
        summary
    )


if __name__ == "__main__":
    main()