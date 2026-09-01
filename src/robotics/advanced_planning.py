"""Advanced motion-planning algorithms for Phase 2.

The established Phase 1 planner provides:

- joint-space safety checking;
- collision-checked edge validation;
- scaled joint-space distance;
- random configuration sampling;
- steering;
- basic RRT;
- path shortcutting.

This module builds on those verified primitives and adds:

1. RRT*
   A sampling-based asymptotically improving planner with local parent
   optimisation and rewiring.

2. Discretised configuration-space A*
   A deterministic graph-search baseline over the same four robot DOFs.

A* is deliberately used only as a discretised configuration-space baseline.
The continuous surgical manipulator remains primarily a sampling-based
planning problem.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools

import numpy as np

from src.geometry.workspace import SphericalStructure
from src.robotics.instrument import SurgicalInstrument
from src.robotics.planner import (
    PlanningResult,
    configuration_is_safe,
    edge_is_safe,
    joint_distance,
    sample_random_configuration,
    steer,
)


@dataclass(frozen=True)
class AStarGridConfig:
    """Resolution of the discretised 4-DOF configuration space."""

    yaw_samples: int = 9
    pitch_samples: int = 7
    insertion_samples: int = 7
    roll_samples: int = 3

    max_expansions: int = 50000

    def __post_init__(self) -> None:
        for name, value in (
            (
                "yaw_samples",
                self.yaw_samples,
            ),
            (
                "pitch_samples",
                self.pitch_samples,
            ),
            (
                "insertion_samples",
                self.insertion_samples,
            ),
            (
                "roll_samples",
                self.roll_samples,
            ),
        ):
            if value < 2:
                raise ValueError(
                    f"{name} must be at least 2."
                )

        if self.max_expansions < 1:
            raise ValueError(
                "max_expansions must be positive."
            )


def _validate_planning_inputs(
    instrument: SurgicalInstrument,
    start_q: np.ndarray,
    goal_q: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Validate and copy planner endpoints."""

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

    return (
        start,
        goal,
    )


def _failed_result(
    iterations: int,
) -> PlanningResult:
    """Return a standard failed planning result."""

    return PlanningResult(
        path=np.empty(
            (
                0,
                4,
            ),
            dtype=float,
        ),
        success=False,
        iterations=int(
            iterations
        ),
    )


def _is_ancestor(
    candidate_ancestor: int,
    node_index: int,
    parents: list[int],
) -> bool:
    """Return whether one tree node is an ancestor of another."""

    current = int(
        node_index
    )

    while current != -1:
        if (
            current
            == candidate_ancestor
        ):
            return True

        current = parents[
            current
        ]

    return False


def _propagate_cost_delta(
    root_index: int,
    delta: float,
    children: list[
        set[int]
    ],
    costs: list[float],
) -> None:
    """Propagate a rewiring cost change to descendants."""

    pending = list(
        children[
            root_index
        ]
    )

    while pending:
        current = pending.pop()

        costs[
            current
        ] += delta

        pending.extend(
            children[
                current
            ]
        )


def _reconstruct_tree_path(
    nodes: list[np.ndarray],
    parents: list[int],
    final_node_index: int,
    goal_q: np.ndarray,
) -> np.ndarray:
    """Reconstruct a tree path and append the exact goal."""

    indices: list[int] = []

    current = int(
        final_node_index
    )

    while current != -1:
        indices.append(
            current
        )

        current = parents[
            current
        ]

    indices.reverse()

    configurations = [
        nodes[
            index
        ]
        for index in indices
    ]

    if not np.allclose(
        configurations[
            -1
        ],
        goal_q,
        atol=1e-12,
    ):
        configurations.append(
            np.array(
                goal_q,
                dtype=float,
                copy=True,
            )
        )

    return np.vstack(
        configurations
    )


def plan_rrt_star(
    instrument: SurgicalInstrument,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    structures: tuple[
        SphericalStructure,
        ...,
    ],
    instrument_radius: float,
    proximal_length: float = 0.10,
    max_iterations: int = 2000,
    step_size: float = 0.08,
    goal_bias: float = 0.15,
    edge_resolution: int = 20,
    rewire_radius: float = 0.60,
    seed: int = 7,
) -> PlanningResult:
    """Plan using deterministic-seeded RRT*.

    RRT* uses the existing project's collision checker, edge validator,
    joint-distance metric, configuration sampler and steering primitive.

    Unlike the existing basic RRT implementation, the algorithm continues
    exploring after finding potentially goal-connectable configurations.
    Local parent optimisation and rewiring reduce tree cost before the
    lowest-cost safe goal connection is selected.
    """

    start, goal = (
        _validate_planning_inputs(
            instrument,
            start_q,
            goal_q,
        )
    )

    if instrument_radius <= 0.0:
        raise ValueError(
            "instrument_radius must be positive."
        )

    if proximal_length <= 0.0:
        raise ValueError(
            "proximal_length must be positive."
        )

    if max_iterations < 1:
        raise ValueError(
            "max_iterations must be positive."
        )

    if step_size <= 0.0:
        raise ValueError(
            "step_size must be positive."
        )

    if not (
        0.0
        <= goal_bias
        <= 1.0
    ):
        raise ValueError(
            "goal_bias must lie between 0 and 1."
        )

    if edge_resolution < 2:
        raise ValueError(
            "edge_resolution must be at least 2."
        )

    if rewire_radius <= 0.0:
        raise ValueError(
            "rewire_radius must be positive."
        )

    if not configuration_is_safe(
        instrument=instrument,
        q=start,
        structures=structures,
        instrument_radius=(
            instrument_radius
        ),
        proximal_length=(
            proximal_length
        ),
    ):
        return _failed_result(
            0
        )

    if not configuration_is_safe(
        instrument=instrument,
        q=goal,
        structures=structures,
        instrument_radius=(
            instrument_radius
        ),
        proximal_length=(
            proximal_length
        ),
    ):
        return _failed_result(
            0
        )

    # A direct safe edge is already the shortest possible path under
    # the metric because of the triangle inequality.
    if edge_is_safe(
        instrument=instrument,
        q_start=start,
        q_goal=goal,
        structures=structures,
        instrument_radius=(
            instrument_radius
        ),
        proximal_length=(
            proximal_length
        ),
        resolution=(
            edge_resolution
        ),
    ):
        return PlanningResult(
            path=np.vstack(
                (
                    start,
                    goal,
                )
            ),
            success=True,
            iterations=0,
        )

    rng = np.random.default_rng(
        seed
    )

    nodes: list[np.ndarray] = [
        np.array(
            start,
            dtype=float,
            copy=True,
        )
    ]

    parents: list[int] = [
        -1
    ]

    costs: list[float] = [
        0.0
    ]

    children: list[
        set[int]
    ] = [
        set()
    ]

    for _iteration in range(
        1,
        max_iterations + 1,
    ):
        if (
            rng.random()
            < goal_bias
        ):
            sample = goal

        else:
            sample = (
                sample_random_configuration(
                    instrument,
                    rng,
                )
            )

        sample_distances = np.asarray(
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
                sample_distances
            )
        )

        nearest = nodes[
            nearest_index
        ]

        new_q = steer(
            nearest,
            sample,
            step_size=step_size,
        )

        try:
            instrument.validate_configuration(
                new_q
            )
        except ValueError:
            continue

        if (
            joint_distance(
                nearest,
                new_q,
            )
            <= 1e-12
        ):
            continue

        if not edge_is_safe(
            instrument=instrument,
            q_start=nearest,
            q_goal=new_q,
            structures=structures,
            instrument_radius=(
                instrument_radius
            ),
            proximal_length=(
                proximal_length
            ),
            resolution=(
                edge_resolution
            ),
        ):
            continue

        distances_to_new = np.asarray(
            [
                joint_distance(
                    node,
                    new_q,
                )
                for node in nodes
            ],
            dtype=float,
        )

        neighbour_indices = [
            index
            for index, distance
            in enumerate(
                distances_to_new
            )
            if distance
            <= rewire_radius
        ]

        if (
            nearest_index
            not in neighbour_indices
        ):
            neighbour_indices.append(
                nearest_index
            )

        best_parent = (
            nearest_index
        )

        best_cost = (
            costs[
                nearest_index
            ]
            + joint_distance(
                nearest,
                new_q,
            )
        )

        # Parent optimisation.
        for neighbour_index in (
            neighbour_indices
        ):
            candidate_parent = (
                nodes[
                    neighbour_index
                ]
            )

            candidate_cost = (
                costs[
                    neighbour_index
                ]
                + joint_distance(
                    candidate_parent,
                    new_q,
                )
            )

            if (
                candidate_cost
                >= best_cost
                - 1e-12
            ):
                continue

            if not edge_is_safe(
                instrument=instrument,
                q_start=(
                    candidate_parent
                ),
                q_goal=new_q,
                structures=structures,
                instrument_radius=(
                    instrument_radius
                ),
                proximal_length=(
                    proximal_length
                ),
                resolution=(
                    edge_resolution
                ),
            ):
                continue

            best_parent = (
                neighbour_index
            )

            best_cost = float(
                candidate_cost
            )

        new_index = len(
            nodes
        )

        nodes.append(
            np.array(
                new_q,
                dtype=float,
                copy=True,
            )
        )

        parents.append(
            best_parent
        )

        costs.append(
            float(
                best_cost
            )
        )

        children.append(
            set()
        )

        children[
            best_parent
        ].add(
            new_index
        )

        # Rewire nearby nodes through the new node when this gives a
        # lower-cost safe route.
        for neighbour_index in (
            neighbour_indices
        ):
            if neighbour_index in (
                0,
                best_parent,
            ):
                continue

            # Rewiring an ancestor beneath its descendant would create
            # a cycle.
            if _is_ancestor(
                neighbour_index,
                new_index,
                parents,
            ):
                continue

            proposed_cost = (
                costs[
                    new_index
                ]
                + joint_distance(
                    new_q,
                    nodes[
                        neighbour_index
                    ],
                )
            )

            if (
                proposed_cost
                >= costs[
                    neighbour_index
                ]
                - 1e-12
            ):
                continue

            if not edge_is_safe(
                instrument=instrument,
                q_start=new_q,
                q_goal=nodes[
                    neighbour_index
                ],
                structures=structures,
                instrument_radius=(
                    instrument_radius
                ),
                proximal_length=(
                    proximal_length
                ),
                resolution=(
                    edge_resolution
                ),
            ):
                continue

            previous_parent = (
                parents[
                    neighbour_index
                ]
            )

            if (
                previous_parent
                != -1
            ):
                children[
                    previous_parent
                ].discard(
                    neighbour_index
                )

            old_cost = costs[
                neighbour_index
            ]

            parents[
                neighbour_index
            ] = new_index

            children[
                new_index
            ].add(
                neighbour_index
            )

            costs[
                neighbour_index
            ] = float(
                proposed_cost
            )

            cost_delta = (
                proposed_cost
                - old_cost
            )

            _propagate_cost_delta(
                root_index=(
                    neighbour_index
                ),
                delta=float(
                    cost_delta
                ),
                children=children,
                costs=costs,
            )

    # RRT* chooses the cheapest node from which the exact goal can be
    # safely connected.
    goal_connections: list[
        tuple[
            float,
            int,
        ]
    ] = []

    for node_index, node in (
        enumerate(
            nodes
        )
    ):
        if not edge_is_safe(
            instrument=instrument,
            q_start=node,
            q_goal=goal,
            structures=structures,
            instrument_radius=(
                instrument_radius
            ),
            proximal_length=(
                proximal_length
            ),
            resolution=(
                edge_resolution
            ),
        ):
            continue

        total_cost = (
            costs[
                node_index
            ]
            + joint_distance(
                node,
                goal,
            )
        )

        goal_connections.append(
            (
                float(
                    total_cost
                ),
                int(
                    node_index
                ),
            )
        )

    if not goal_connections:
        return _failed_result(
            max_iterations
        )

    _best_cost, best_node_index = min(
        goal_connections,
        key=lambda item: item[
            0
        ],
    )

    path = _reconstruct_tree_path(
        nodes=nodes,
        parents=parents,
        final_node_index=(
            best_node_index
        ),
        goal_q=goal,
    )

    return PlanningResult(
        path=path,
        success=True,
        iterations=max_iterations,
    )


def _grid_axis(
    lower: float,
    upper: float,
    sample_count: int,
    start_value: float,
    goal_value: float,
) -> np.ndarray:
    """Create one configuration-grid axis including exact endpoints."""

    regular = np.linspace(
        lower,
        upper,
        sample_count,
        dtype=float,
    )

    axis = np.concatenate(
        (
            regular,
            np.asarray(
                [
                    start_value,
                    goal_value,
                ],
                dtype=float,
            ),
        )
    )

    axis = np.unique(
        axis
    )

    axis.sort()

    return axis


def _configuration_from_index(
    axes: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    index: tuple[
        int,
        int,
        int,
        int,
    ],
) -> np.ndarray:
    """Convert a four-dimensional grid index to a configuration."""

    return np.asarray(
        [
            axes[
                dimension
            ][
                index[
                    dimension
                ]
            ]
            for dimension
            in range(
                4
            )
        ],
        dtype=float,
    )


def _nearest_axis_index(
    axis: np.ndarray,
    value: float,
) -> int:
    """Return index of a value explicitly inserted into an axis."""

    return int(
        np.argmin(
            np.abs(
                axis
                - value
            )
        )
    )


def _grid_neighbours(
    index: tuple[
        int,
        int,
        int,
        int,
    ],
    axes: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
) -> tuple[
    tuple[
        int,
        int,
        int,
        int,
    ],
    ...,
]:
    """Return axis-adjacent neighbours in the 4-D grid."""

    neighbours: list[
        tuple[
            int,
            int,
            int,
            int,
        ]
    ] = []

    for dimension in range(
        4
    ):
        for direction in (
            -1,
            1,
        ):
            candidate = list(
                index
            )

            candidate[
                dimension
            ] += direction

            if candidate[
                dimension
            ] < 0:
                continue

            if candidate[
                dimension
            ] >= len(
                axes[
                    dimension
                ]
            ):
                continue

            neighbours.append(
                tuple(
                    candidate
                )
            )

    return tuple(
        neighbours
    )


def plan_astar_configuration_grid(
    instrument: SurgicalInstrument,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    structures: tuple[
        SphericalStructure,
        ...,
    ],
    instrument_radius: float,
    proximal_length: float = 0.10,
    edge_resolution: int = 15,
    grid_config: AStarGridConfig | None = None,
) -> PlanningResult:
    """Plan with A* on a discretised 4-DOF configuration space.

    The exact start and goal joint values are inserted into each grid axis,
    so the returned path starts and ends at the requested configurations.

    Neighbours differ by one adjacent grid coordinate. Edge cost uses the
    same scaled joint-distance metric as the continuous RRT planner.

    The Euclidean scaled joint distance to the goal is used as the A*
    heuristic. It does not overestimate the cost of a path composed from
    these edge costs.
    """

    start, goal = (
        _validate_planning_inputs(
            instrument,
            start_q,
            goal_q,
        )
    )

    if grid_config is None:
        grid_config = (
            AStarGridConfig()
        )

    if instrument_radius <= 0.0:
        raise ValueError(
            "instrument_radius must be positive."
        )

    if proximal_length <= 0.0:
        raise ValueError(
            "proximal_length must be positive."
        )

    if edge_resolution < 2:
        raise ValueError(
            "edge_resolution must be at least 2."
        )

    if not configuration_is_safe(
        instrument=instrument,
        q=start,
        structures=structures,
        instrument_radius=(
            instrument_radius
        ),
        proximal_length=(
            proximal_length
        ),
    ):
        return _failed_result(
            0
        )

    if not configuration_is_safe(
        instrument=instrument,
        q=goal,
        structures=structures,
        instrument_radius=(
            instrument_radius
        ),
        proximal_length=(
            proximal_length
        ),
    ):
        return _failed_result(
            0
        )

    if edge_is_safe(
        instrument=instrument,
        q_start=start,
        q_goal=goal,
        structures=structures,
        instrument_radius=(
            instrument_radius
        ),
        proximal_length=(
            proximal_length
        ),
        resolution=(
            edge_resolution
        ),
    ):
        return PlanningResult(
            path=np.vstack(
                (
                    start,
                    goal,
                )
            ),
            success=True,
            iterations=0,
        )

    limits = (
        instrument.joint_limits
    )

    axes = (
        _grid_axis(
            limits.yaw_min,
            limits.yaw_max,
            grid_config.yaw_samples,
            float(
                start[
                    0
                ]
            ),
            float(
                goal[
                    0
                ]
            ),
        ),
        _grid_axis(
            limits.pitch_min,
            limits.pitch_max,
            grid_config.pitch_samples,
            float(
                start[
                    1
                ]
            ),
            float(
                goal[
                    1
                ]
            ),
        ),
        _grid_axis(
            limits.insertion_min,
            limits.insertion_max,
            grid_config.insertion_samples,
            float(
                start[
                    2
                ]
            ),
            float(
                goal[
                    2
                ]
            ),
        ),
        _grid_axis(
            limits.roll_min,
            limits.roll_max,
            grid_config.roll_samples,
            float(
                start[
                    3
                ]
            ),
            float(
                goal[
                    3
                ]
            ),
        ),
    )

    start_index = tuple(
        _nearest_axis_index(
            axes[
                dimension
            ],
            float(
                start[
                    dimension
                ]
            ),
        )
        for dimension
        in range(
            4
        )
    )

    goal_index = tuple(
        _nearest_axis_index(
            axes[
                dimension
            ],
            float(
                goal[
                    dimension
                ]
            ),
        )
        for dimension
        in range(
            4
        )
    )

    frontier: list[
        tuple[
            float,
            int,
            tuple[
                int,
                int,
                int,
                int,
            ],
        ]
    ] = []

    tie_breaker = (
        itertools.count()
    )

    heapq.heappush(
        frontier,
        (
            joint_distance(
                start,
                goal,
            ),
            next(
                tie_breaker
            ),
            start_index,
        ),
    )

    cost_to_come: dict[
        tuple[
            int,
            int,
            int,
            int,
        ],
        float,
    ] = {
        start_index: 0.0
    }

    parent: dict[
        tuple[
            int,
            int,
            int,
            int,
        ],
        tuple[
            int,
            int,
            int,
            int,
        ],
    ] = {}

    closed: set[
        tuple[
            int,
            int,
            int,
            int,
        ]
    ] = set()

    expansions = 0

    while (
        frontier
        and expansions
        < grid_config.max_expansions
    ):
        (
            _priority,
            _tie,
            current_index,
        ) = heapq.heappop(
            frontier
        )

        if (
            current_index
            in closed
        ):
            continue

        closed.add(
            current_index
        )

        expansions += 1

        if (
            current_index
            == goal_index
        ):
            index_path = [
                current_index
            ]

            while (
                index_path[
                    -1
                ]
                != start_index
            ):
                index_path.append(
                    parent[
                        index_path[
                            -1
                        ]
                    ]
                )

            index_path.reverse()

            path = np.vstack(
                [
                    _configuration_from_index(
                        axes,
                        index,
                    )
                    for index
                    in index_path
                ]
            )

            # Preserve exact user-provided endpoints.
            path[
                0
            ] = start

            path[
                -1
            ] = goal

            return PlanningResult(
                path=path,
                success=True,
                iterations=(
                    expansions
                ),
            )

        current_q = (
            _configuration_from_index(
                axes,
                current_index,
            )
        )

        for neighbour_index in (
            _grid_neighbours(
                current_index,
                axes,
            )
        ):
            if (
                neighbour_index
                in closed
            ):
                continue

            neighbour_q = (
                _configuration_from_index(
                    axes,
                    neighbour_index,
                )
            )

            try:
                instrument.validate_configuration(
                    neighbour_q
                )
            except ValueError:
                continue

            if not configuration_is_safe(
                instrument=instrument,
                q=neighbour_q,
                structures=structures,
                instrument_radius=(
                    instrument_radius
                ),
                proximal_length=(
                    proximal_length
                ),
            ):
                continue

            if not edge_is_safe(
                instrument=instrument,
                q_start=current_q,
                q_goal=neighbour_q,
                structures=structures,
                instrument_radius=(
                    instrument_radius
                ),
                proximal_length=(
                    proximal_length
                ),
                resolution=(
                    edge_resolution
                ),
            ):
                continue

            tentative_cost = (
                cost_to_come[
                    current_index
                ]
                + joint_distance(
                    current_q,
                    neighbour_q,
                )
            )

            previous_cost = (
                cost_to_come.get(
                    neighbour_index,
                    float(
                        "inf"
                    ),
                )
            )

            if (
                tentative_cost
                >= previous_cost
                - 1e-12
            ):
                continue

            cost_to_come[
                neighbour_index
            ] = float(
                tentative_cost
            )

            parent[
                neighbour_index
            ] = current_index

            heuristic = (
                joint_distance(
                    neighbour_q,
                    goal,
                )
            )

            priority = (
                tentative_cost
                + heuristic
            )

            heapq.heappush(
                frontier,
                (
                    float(
                        priority
                    ),
                    next(
                        tie_breaker
                    ),
                    neighbour_index,
                ),
            )

    return _failed_result(
        expansions
    )