"""Chance-constrained robot motion planning for Phase 6.

This module connects the Phase 6 probabilistic clearance model to the
existing surgical-instrument planning stack.

Earlier uncertainty-aware planning converts each anatomical covariance into
a scalar margin based on the largest principal standard deviation:

    base_margin + k * principal_sigma

That construction is useful and conservative but discards covariance
orientation.

This module instead evaluates the estimated probability that sampled points
on the surgical-instrument shaft violate the protected radius of an uncertain
anatomical structure.

The runtime chain is:

    EstimatedStructure
        ->
    UncertainSphere
        ->
    instrument shaft geometry
        ->
    directional covariance projection
        ->
    local clearance-violation probability
        ->
    configuration / edge chance constraint
        ->
    RRT exploration

Ground-truth anatomical geometry is deliberately absent from every runtime
planning interface. Hidden truth remains reserved for experimental evaluation.

Scientific boundary
-------------------

The probability model is a first-order local Gaussian approximation. It is
not a clinical collision probability or patient-risk estimate.

The optional union-bound diagnostic is conservative and discretisation
dependent. Pointwise chance constraints are therefore the primary acceptance
criterion used by the planner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.perception.uncertainty import EstimatedStructure
from src.robotics.instrument import SurgicalInstrument
from src.robotics.planner import (
    PlanningResult,
    joint_distance,
    sample_random_configuration,
    steer,
)
from src.robotics.risk_aware_planning import (
    ChanceConstraintConfig,
    PathRiskEstimate,
    UncertainSphere,
    evaluate_path_clearance_risk,
)


@dataclass(frozen=True)
class ConfigurationRiskEvaluation:
    """Risk diagnostics for one robot configuration."""

    accepted: bool

    maximum_point_violation_probability: float

    union_bound_violation_probability: float

    shaft_risk: PathRiskEstimate


@dataclass(frozen=True)
class EdgeRiskEvaluation:
    """Risk diagnostics for a joint-space edge."""

    accepted: bool

    maximum_point_violation_probability: float

    union_bound_violation_probability: float

    configurations_evaluated: int

    worst_configuration_index: int | None


@dataclass(frozen=True)
class ChanceConstrainedRRTConfig:
    """Configuration for Phase 6 chance-constrained RRT."""

    max_iterations: int = 5000

    step_size: float = 0.08

    goal_bias: float = 0.15

    edge_resolution: int = 20

    proximal_length: float = 0.10

    shaft_sample_spacing: float = 0.003

    seed: int = 7

    def __post_init__(
        self,
    ) -> None:
        if self.max_iterations < 1:
            raise ValueError(
                "max_iterations must be positive."
            )

        if (
            not np.isfinite(
                self.step_size
            )
            or self.step_size <= 0.0
        ):
            raise ValueError(
                "step_size must be finite and positive."
            )

        if (
            not np.isfinite(
                self.goal_bias
            )
            or self.goal_bias < 0.0
            or self.goal_bias > 1.0
        ):
            raise ValueError(
                "goal_bias must lie in [0, 1]."
            )

        if self.edge_resolution < 2:
            raise ValueError(
                "edge_resolution must be at least 2."
            )

        if (
            not np.isfinite(
                self.proximal_length
            )
            or self.proximal_length <= 0.0
        ):
            raise ValueError(
                "proximal_length must be finite and positive."
            )

        if (
            not np.isfinite(
                self.shaft_sample_spacing
            )
            or self.shaft_sample_spacing <= 0.0
        ):
            raise ValueError(
                "shaft_sample_spacing must be finite and positive."
            )


def estimated_structure_to_uncertain_sphere(
    estimate: EstimatedStructure,
    *,
    label: str = "",
) -> UncertainSphere:
    """Convert one perception estimate into Phase 6 uncertain geometry.

    Unlike the previous principal-sigma inflation approach, the complete
    3 x 3 covariance is retained.
    """

    if not isinstance(
        estimate,
        EstimatedStructure,
    ):
        raise TypeError(
            "estimate must be an EstimatedStructure."
        )

    return UncertainSphere(
        center_mean=np.array(
            estimate.estimated_centre,
            dtype=float,
            copy=True,
        ),
        center_covariance=np.array(
            estimate.uncertainty.covariance,
            dtype=float,
            copy=True,
        ),
        physical_radius=float(
            estimate.physical_radius
        ),
        safety_margin=float(
            estimate.base_safety_margin
        ),
        label=str(
            label
        ),
    )


def estimated_structures_to_uncertain_spheres(
    estimates: Sequence[
        EstimatedStructure
    ],
) -> tuple[
    UncertainSphere,
    ...,
]:
    """Convert perception structures while preserving full covariance."""

    converted: list[
        UncertainSphere
    ] = []

    for index, estimate in enumerate(
        estimates
    ):
        converted.append(
            estimated_structure_to_uncertain_sphere(
                estimate,
                label=f"structure_{index}",
            )
        )

    return tuple(
        converted
    )


def _chance_config_for_shaft(
    chance_config: ChanceConstraintConfig,
    shaft_sample_spacing: float,
) -> ChanceConstraintConfig:
    """Create a chance configuration using explicit shaft discretisation."""

    spacing = float(
        shaft_sample_spacing
    )

    if (
        not np.isfinite(spacing)
        or spacing <= 0.0
    ):
        raise ValueError(
            "shaft_sample_spacing must be finite and positive."
        )

    return ChanceConstraintConfig(
        max_point_violation_probability=(
            chance_config.max_point_violation_probability
        ),
        max_path_union_bound_probability=(
            chance_config.max_path_union_bound_probability
        ),
        sample_spacing=spacing,
        sigma_epsilon=(
            chance_config.sigma_epsilon
        ),
        geometry_epsilon=(
            chance_config.geometry_epsilon
        ),
    )


def evaluate_configuration_chance_constraint(
    instrument: SurgicalInstrument,
    q: np.ndarray,
    estimated_structures: Sequence[
        EstimatedStructure
    ],
    instrument_radius: float,
    chance_config: ChanceConstraintConfig,
    *,
    proximal_length: float = 0.10,
    shaft_sample_spacing: float = 0.003,
) -> ConfigurationRiskEvaluation:
    """Evaluate chance-constrained safety for one robot configuration.

    The physical instrument shaft is sampled between the proximal shaft point
    and tool tip.

    Every sampled shaft point is evaluated against every uncertain anatomical
    structure.
    """

    configuration = np.asarray(
        q,
        dtype=float,
    )

    instrument.validate_configuration(
        configuration
    )

    radius = float(
        instrument_radius
    )

    if (
        not np.isfinite(radius)
        or radius < 0.0
    ):
        raise ValueError(
            "instrument_radius must be finite and non-negative."
        )

    if (
        not np.isfinite(
            proximal_length
        )
        or proximal_length <= 0.0
    ):
        raise ValueError(
            "proximal_length must be finite and positive."
        )

    uncertain_structures = (
        estimated_structures_to_uncertain_spheres(
            estimated_structures
        )
    )

    proximal_point, tip_point = (
        instrument.shaft_segment(
            configuration,
            proximal_length=(
                proximal_length
            ),
        )
    )

    shaft_path = np.vstack(
        [
            proximal_point,
            tip_point,
        ]
    )

    shaft_config = (
        _chance_config_for_shaft(
            chance_config,
            shaft_sample_spacing,
        )
    )

    risk = evaluate_path_clearance_risk(
        path_points=shaft_path,
        structures=(
            uncertain_structures
        ),
        instrument_radius=radius,
        config=shaft_config,
    )

    return ConfigurationRiskEvaluation(
        accepted=bool(
            risk.accepted
        ),
        maximum_point_violation_probability=float(
            risk.maximum_point_violation_probability
        ),
        union_bound_violation_probability=float(
            risk.union_bound_violation_probability
        ),
        shaft_risk=risk,
    )


def configuration_is_chance_constrained_safe(
    instrument: SurgicalInstrument,
    q: np.ndarray,
    estimated_structures: Sequence[
        EstimatedStructure
    ],
    instrument_radius: float,
    chance_config: ChanceConstraintConfig,
    *,
    proximal_length: float = 0.10,
    shaft_sample_spacing: float = 0.003,
) -> bool:
    """Return whether one configuration satisfies the chance constraint."""

    evaluation = (
        evaluate_configuration_chance_constraint(
            instrument=instrument,
            q=q,
            estimated_structures=(
                estimated_structures
            ),
            instrument_radius=(
                instrument_radius
            ),
            chance_config=(
                chance_config
            ),
            proximal_length=(
                proximal_length
            ),
            shaft_sample_spacing=(
                shaft_sample_spacing
            ),
        )
    )

    return bool(
        evaluation.accepted
    )


def evaluate_edge_chance_constraint(
    instrument: SurgicalInstrument,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    estimated_structures: Sequence[
        EstimatedStructure
    ],
    instrument_radius: float,
    chance_config: ChanceConstraintConfig,
    *,
    proximal_length: float = 0.10,
    shaft_sample_spacing: float = 0.003,
    resolution: int = 20,
) -> EdgeRiskEvaluation:
    """Evaluate a straight joint-space edge under the chance constraint.

    Joint configurations are interpolated exactly as in the existing planner
    safety checker.

    Pointwise probability is the primary acceptance criterion.

    If a path-level union-bound threshold is configured, probabilities from
    all sampled shaft/configuration events are additionally accumulated. This
    value is deliberately conservative and depends on discretisation.
    """

    start = np.asarray(
        q_start,
        dtype=float,
    )

    goal = np.asarray(
        q_goal,
        dtype=float,
    )

    if (
        start.shape != (4,)
        or goal.shape != (4,)
    ):
        raise ValueError(
            "q_start and q_goal must each have shape (4,)."
        )

    if resolution < 2:
        raise ValueError(
            "resolution must be at least 2."
        )

    samples = np.linspace(
        start,
        goal,
        num=resolution,
        dtype=float,
    )

    maximum_probability = 0.0

    probability_sum = 0.0

    worst_configuration_index: (
        int | None
    ) = None

    configurations_evaluated = 0

    for sample_index, configuration in enumerate(
        samples
    ):
        try:
            instrument.validate_configuration(
                configuration
            )
        except ValueError:
            return EdgeRiskEvaluation(
                accepted=False,
                maximum_point_violation_probability=1.0,
                union_bound_violation_probability=1.0,
                configurations_evaluated=(
                    configurations_evaluated
                ),
                worst_configuration_index=(
                    sample_index
                ),
            )

        evaluation = (
            evaluate_configuration_chance_constraint(
                instrument=instrument,
                q=configuration,
                estimated_structures=(
                    estimated_structures
                ),
                instrument_radius=(
                    instrument_radius
                ),
                chance_config=(
                    chance_config
                ),
                proximal_length=(
                    proximal_length
                ),
                shaft_sample_spacing=(
                    shaft_sample_spacing
                ),
            )
        )

        configurations_evaluated += 1

        local_probability = float(
            evaluation.maximum_point_violation_probability
        )

        if (
            worst_configuration_index
            is None
            or local_probability
            > maximum_probability
        ):
            maximum_probability = (
                local_probability
            )

            worst_configuration_index = (
                sample_index
            )

        probability_sum += float(
            sum(
                event.estimate.violation_probability
                for event in (
                    evaluation.shaft_risk.events
                )
            )
        )

        if (
            local_probability
            > chance_config.max_point_violation_probability
        ):
            return EdgeRiskEvaluation(
                accepted=False,
                maximum_point_violation_probability=(
                    maximum_probability
                ),
                union_bound_violation_probability=float(
                    min(
                        1.0,
                        probability_sum,
                    )
                ),
                configurations_evaluated=(
                    configurations_evaluated
                ),
                worst_configuration_index=(
                    worst_configuration_index
                ),
            )

    union_bound = float(
        min(
            1.0,
            probability_sum,
        )
    )

    accepted = (
        maximum_probability
        <= chance_config.max_point_violation_probability
    )

    if (
        chance_config.max_path_union_bound_probability
        is not None
    ):
        accepted = (
            accepted
            and union_bound
            <= chance_config.max_path_union_bound_probability
        )

    return EdgeRiskEvaluation(
        accepted=bool(
            accepted
        ),
        maximum_point_violation_probability=(
            maximum_probability
        ),
        union_bound_violation_probability=(
            union_bound
        ),
        configurations_evaluated=(
            configurations_evaluated
        ),
        worst_configuration_index=(
            worst_configuration_index
        ),
    )


def edge_is_chance_constrained_safe(
    instrument: SurgicalInstrument,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    estimated_structures: Sequence[
        EstimatedStructure
    ],
    instrument_radius: float,
    chance_config: ChanceConstraintConfig,
    *,
    proximal_length: float = 0.10,
    shaft_sample_spacing: float = 0.003,
    resolution: int = 20,
) -> bool:
    """Return whether a joint-space edge satisfies the chance constraint."""

    evaluation = (
        evaluate_edge_chance_constraint(
            instrument=instrument,
            q_start=q_start,
            q_goal=q_goal,
            estimated_structures=(
                estimated_structures
            ),
            instrument_radius=(
                instrument_radius
            ),
            chance_config=(
                chance_config
            ),
            proximal_length=(
                proximal_length
            ),
            shaft_sample_spacing=(
                shaft_sample_spacing
            ),
            resolution=resolution,
        )
    )

    return bool(
        evaluation.accepted
    )


def evaluate_joint_path_chance_constraint(
    instrument: SurgicalInstrument,
    path: np.ndarray,
    estimated_structures: Sequence[
        EstimatedStructure
    ],
    instrument_radius: float,
    chance_config: ChanceConstraintConfig,
    *,
    proximal_length: float = 0.10,
    shaft_sample_spacing: float = 0.003,
    edge_resolution: int = 20,
) -> EdgeRiskEvaluation:
    """Evaluate an entire joint-space path.

    Consecutive waypoint pairs are checked using the same edge evaluator used
    during planning.
    """

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

    if joint_path.shape[0] == 1:
        configuration = (
            evaluate_configuration_chance_constraint(
                instrument=instrument,
                q=joint_path[0],
                estimated_structures=(
                    estimated_structures
                ),
                instrument_radius=(
                    instrument_radius
                ),
                chance_config=(
                    chance_config
                ),
                proximal_length=(
                    proximal_length
                ),
                shaft_sample_spacing=(
                    shaft_sample_spacing
                ),
            )
        )

        return EdgeRiskEvaluation(
            accepted=bool(
                configuration.accepted
            ),
            maximum_point_violation_probability=float(
                configuration.maximum_point_violation_probability
            ),
            union_bound_violation_probability=float(
                configuration.union_bound_violation_probability
            ),
            configurations_evaluated=1,
            worst_configuration_index=0,
        )

    maximum_probability = 0.0

    probability_sum = 0.0

    total_configurations = 0

    worst_index: (
        int | None
    ) = None

    configuration_offset = 0

    all_accepted = True

    for edge_index in range(
        joint_path.shape[0] - 1
    ):
        evaluation = (
            evaluate_edge_chance_constraint(
                instrument=instrument,
                q_start=joint_path[
                    edge_index
                ],
                q_goal=joint_path[
                    edge_index + 1
                ],
                estimated_structures=(
                    estimated_structures
                ),
                instrument_radius=(
                    instrument_radius
                ),
                chance_config=(
                    chance_config
                ),
                proximal_length=(
                    proximal_length
                ),
                shaft_sample_spacing=(
                    shaft_sample_spacing
                ),
                resolution=(
                    edge_resolution
                ),
            )
        )

        if (
            evaluation.maximum_point_violation_probability
            > maximum_probability
        ):
            maximum_probability = float(
                evaluation.maximum_point_violation_probability
            )

            if (
                evaluation.worst_configuration_index
                is not None
            ):
                worst_index = (
                    configuration_offset
                    + evaluation.worst_configuration_index
                )

        probability_sum += float(
            evaluation.union_bound_violation_probability
        )

        total_configurations += int(
            evaluation.configurations_evaluated
        )

        configuration_offset += int(
            evaluation.configurations_evaluated
        )

        if not evaluation.accepted:
            all_accepted = False

    union_bound = float(
        min(
            1.0,
            probability_sum,
        )
    )

    if (
        chance_config.max_path_union_bound_probability
        is not None
    ):
        all_accepted = (
            all_accepted
            and union_bound
            <= chance_config.max_path_union_bound_probability
        )

    return EdgeRiskEvaluation(
        accepted=bool(
            all_accepted
        ),
        maximum_point_violation_probability=(
            maximum_probability
        ),
        union_bound_violation_probability=(
            union_bound
        ),
        configurations_evaluated=(
            total_configurations
        ),
        worst_configuration_index=(
            worst_index
        ),
    )


def _reconstruct_path(
    nodes: list[
        np.ndarray
    ],
    parents: list[int],
    final_index: int,
) -> np.ndarray:
    """Reconstruct an RRT path from the parent tree."""

    indices: list[
        int
    ] = []

    current = int(
        final_index
    )

    while current != -1:
        indices.append(
            current
        )

        current = parents[
            current
        ]

    indices.reverse()

    return np.vstack(
        [
            nodes[index]
            for index in indices
        ]
    )


def plan_rrt_chance_constrained(
    instrument: SurgicalInstrument,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    estimated_structures: Sequence[
        EstimatedStructure
    ],
    instrument_radius: float,
    chance_config: ChanceConstraintConfig,
    *,
    planner_config: (
        ChanceConstrainedRRTConfig
        | None
    ) = None,
) -> PlanningResult:
    """Plan with RRT using probabilistic edge acceptance.

    This deliberately mirrors the existing RRT architecture while replacing
    deterministic scalar-inflated collision checking with anisotropic
    chance-constrained shaft evaluation.
    """

    if planner_config is None:
        planner_config = (
            ChanceConstrainedRRTConfig()
        )

    start = np.asarray(
        start_q,
        dtype=float,
    )

    goal = np.asarray(
        goal_q,
        dtype=float,
    )

    instrument.validate_configuration(
        start
    )

    instrument.validate_configuration(
        goal
    )

    estimated_tuple = tuple(
        estimated_structures
    )

    if not (
        configuration_is_chance_constrained_safe(
            instrument=instrument,
            q=start,
            estimated_structures=(
                estimated_tuple
            ),
            instrument_radius=(
                instrument_radius
            ),
            chance_config=(
                chance_config
            ),
            proximal_length=(
                planner_config.proximal_length
            ),
            shaft_sample_spacing=(
                planner_config.shaft_sample_spacing
            ),
        )
    ):
        return PlanningResult(
            path=np.empty(
                (
                    0,
                    4,
                ),
                dtype=float,
            ),
            success=False,
            iterations=0,
        )

    if not (
        configuration_is_chance_constrained_safe(
            instrument=instrument,
            q=goal,
            estimated_structures=(
                estimated_tuple
            ),
            instrument_radius=(
                instrument_radius
            ),
            chance_config=(
                chance_config
            ),
            proximal_length=(
                planner_config.proximal_length
            ),
            shaft_sample_spacing=(
                planner_config.shaft_sample_spacing
            ),
        )
    ):
        return PlanningResult(
            path=np.empty(
                (
                    0,
                    4,
                ),
                dtype=float,
            ),
            success=False,
            iterations=0,
        )

    if edge_is_chance_constrained_safe(
        instrument=instrument,
        q_start=start,
        q_goal=goal,
        estimated_structures=(
            estimated_tuple
        ),
        instrument_radius=(
            instrument_radius
        ),
        chance_config=(
            chance_config
        ),
        proximal_length=(
            planner_config.proximal_length
        ),
        shaft_sample_spacing=(
            planner_config.shaft_sample_spacing
        ),
        resolution=(
            planner_config.edge_resolution
        ),
    ):
        return PlanningResult(
            path=np.vstack(
                [
                    start,
                    goal,
                ]
            ),
            success=True,
            iterations=0,
        )

    rng = np.random.default_rng(
        planner_config.seed
    )

    nodes: list[
        np.ndarray
    ] = [
        np.array(
            start,
            dtype=float,
            copy=True,
        )
    ]

    parents: list[
        int
    ] = [
        -1
    ]

    for iteration in range(
        1,
        planner_config.max_iterations
        + 1,
    ):
        if (
            rng.random()
            < planner_config.goal_bias
        ):
            sample = np.array(
                goal,
                dtype=float,
                copy=True,
            )

        else:
            sample = (
                sample_random_configuration(
                    instrument=instrument,
                    rng=rng,
                )
            )

        distances = np.asarray(
            [
                joint_distance(
                    node,
                    sample,
                )
                for node in nodes
            ],
            dtype=float,
        )

        nearest_index = int(
            np.argmin(
                distances
            )
        )

        nearest = nodes[
            nearest_index
        ]

        candidate = steer(
            nearest,
            sample,
            planner_config.step_size,
        )

        try:
            instrument.validate_configuration(
                candidate
            )
        except ValueError:
            continue

        if not (
            edge_is_chance_constrained_safe(
                instrument=instrument,
                q_start=nearest,
                q_goal=candidate,
                estimated_structures=(
                    estimated_tuple
                ),
                instrument_radius=(
                    instrument_radius
                ),
                chance_config=(
                    chance_config
                ),
                proximal_length=(
                    planner_config.proximal_length
                ),
                shaft_sample_spacing=(
                    planner_config.shaft_sample_spacing
                ),
                resolution=(
                    planner_config.edge_resolution
                ),
            )
        ):
            continue

        nodes.append(
            np.array(
                candidate,
                dtype=float,
                copy=True,
            )
        )

        parents.append(
            nearest_index
        )

        candidate_index = (
            len(nodes)
            - 1
        )

        if joint_distance(
            candidate,
            goal,
        ) > planner_config.step_size:
            continue

        if not (
            edge_is_chance_constrained_safe(
                instrument=instrument,
                q_start=candidate,
                q_goal=goal,
                estimated_structures=(
                    estimated_tuple
                ),
                instrument_radius=(
                    instrument_radius
                ),
                chance_config=(
                    chance_config
                ),
                proximal_length=(
                    planner_config.proximal_length
                ),
                shaft_sample_spacing=(
                    planner_config.shaft_sample_spacing
                ),
                resolution=(
                    planner_config.edge_resolution
                ),
            )
        ):
            continue

        nodes.append(
            np.array(
                goal,
                dtype=float,
                copy=True,
            )
        )

        parents.append(
            candidate_index
        )

        goal_index = (
            len(nodes)
            - 1
        )

        path = _reconstruct_path(
            nodes,
            parents,
            goal_index,
        )

        return PlanningResult(
            path=path,
            success=True,
            iterations=iteration,
        )

    return PlanningResult(
        path=np.empty(
            (
                0,
                4,
            ),
            dtype=float,
        ),
        success=False,
        iterations=(
            planner_config.max_iterations
        ),
    )