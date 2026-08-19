"""Collision-aware joint-space planning for the surgical instrument."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry.workspace import SphericalStructure
from src.robotics.instrument import SurgicalInstrument
from src.robotics.safety import evaluate_instrument_safety


@dataclass(frozen=True)
class PlanningResult:
    """Result returned by a planner."""

    path: np.ndarray
    success: bool
    iterations: int


def configuration_is_safe(
    instrument: SurgicalInstrument,
    q: np.ndarray,
    structures: tuple[SphericalStructure, ...],
    instrument_radius: float,
    proximal_length: float = 0.10,
) -> bool:
    """Check whether a joint configuration is collision-free and margin-safe."""

    proximal_point, tip_point = instrument.shaft_segment(
        q,
        proximal_length=proximal_length,
    )

    evaluation = evaluate_instrument_safety(
        shaft_start=proximal_point,
        shaft_end=tip_point,
        structures=structures,
        instrument_radius=instrument_radius,
    )

    return bool(
        not evaluation.collision
        and not evaluation.safety_margin_violation
    )


def edge_is_safe(
    instrument: SurgicalInstrument,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    structures: tuple[SphericalStructure, ...],
    instrument_radius: float,
    proximal_length: float = 0.10,
    resolution: int = 20,
) -> bool:
    """Check whether a straight joint-space edge is safe."""

    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)

    if q_start.shape != (4,) or q_goal.shape != (4,):
        raise ValueError(
            "q_start and q_goal must each have shape (4,)."
        )

    if resolution < 2:
        raise ValueError(
            "resolution must be at least 2."
        )

    samples = np.linspace(
        q_start,
        q_goal,
        num=resolution,
        dtype=float,
    )

    for q in samples:
        try:
            instrument.validate_configuration(q)
        except ValueError:
            return False

        if not configuration_is_safe(
            instrument=instrument,
            q=q,
            structures=structures,
            instrument_radius=instrument_radius,
            proximal_length=proximal_length,
        ):
            return False

    return True


def sample_random_configuration(
    instrument: SurgicalInstrument,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a random valid joint configuration."""

    limits = instrument.joint_limits

    return np.array(
        [
            rng.uniform(
                limits.yaw_min,
                limits.yaw_max,
            ),
            rng.uniform(
                limits.pitch_min,
                limits.pitch_max,
            ),
            rng.uniform(
                limits.insertion_min,
                limits.insertion_max,
            ),
            rng.uniform(
                limits.roll_min,
                limits.roll_max,
            ),
        ],
        dtype=float,
    )


def joint_distance(
    q_a: np.ndarray,
    q_b: np.ndarray,
) -> float:
    """Return Euclidean distance in scaled joint space."""

    q_a = np.asarray(q_a, dtype=float)
    q_b = np.asarray(q_b, dtype=float)

    if q_a.shape != (4,) or q_b.shape != (4,):
        raise ValueError(
            "q_a and q_b must each have shape (4,)."
        )

    delta = q_b - q_a

    scaled_delta = np.array(
        [
            delta[0],
            delta[1],
            delta[2] / 0.10,
            delta[3] * 0.25,
        ],
        dtype=float,
    )

    return float(
        np.linalg.norm(scaled_delta)
    )


def path_cost(
    path: np.ndarray,
) -> float:
    """Return total scaled joint-space length of a path."""

    path = np.asarray(path, dtype=float)

    if (
        path.ndim != 2
        or path.shape[1] != 4
    ):
        raise ValueError(
            "path must have shape (N, 4)."
        )

    if len(path) < 2:
        return 0.0

    total_cost = 0.0

    for index in range(
        len(path) - 1
    ):
        total_cost += joint_distance(
            path[index],
            path[index + 1],
        )

    return float(total_cost)


def steer(
    q_from: np.ndarray,
    q_to: np.ndarray,
    step_size: float,
) -> np.ndarray:
    """Move from q_from toward q_to by at most step_size."""

    q_from = np.asarray(q_from, dtype=float)
    q_to = np.asarray(q_to, dtype=float)

    if q_from.shape != (4,) or q_to.shape != (4,):
        raise ValueError(
            "q_from and q_to must each have shape (4,)."
        )

    if step_size <= 0.0:
        raise ValueError(
            "step_size must be positive."
        )

    delta = q_to - q_from
    distance = np.linalg.norm(delta)

    if distance <= step_size:
        return q_to.copy()

    direction = delta / distance

    return q_from + step_size * direction


def shortcut_path(
    instrument: SurgicalInstrument,
    path: np.ndarray,
    structures: tuple[SphericalStructure, ...],
    instrument_radius: float,
    proximal_length: float = 0.10,
    edge_resolution: int = 30,
    attempts: int = 300,
    seed: int = 11,
) -> np.ndarray:
    """Shorten an RRT path using collision-checked random shortcuts.

    Random pairs of non-adjacent waypoints are selected. If the direct
    connection between them is safe, the intermediate waypoints are removed.

    The start and goal configurations are always preserved.
    """

    path = np.asarray(
        path,
        dtype=float,
    )

    if (
        path.ndim != 2
        or path.shape[1] != 4
    ):
        raise ValueError(
            "path must have shape (N, 4)."
        )

    if len(path) < 2:
        raise ValueError(
            "path must contain at least two configurations."
        )

    if attempts < 0:
        raise ValueError(
            "attempts must be non-negative."
        )

    if edge_resolution < 2:
        raise ValueError(
            "edge_resolution must be at least 2."
        )

    shortened_path = path.copy()

    rng = np.random.default_rng(
        seed
    )

    for _ in range(attempts):
        if len(shortened_path) <= 2:
            break

        indices = rng.choice(
            len(shortened_path),
            size=2,
            replace=False,
        )

        first_index = int(
            min(indices)
        )

        second_index = int(
            max(indices)
        )

        if second_index - first_index <= 1:
            continue

        if edge_is_safe(
            instrument=instrument,
            q_start=shortened_path[first_index],
            q_goal=shortened_path[second_index],
            structures=structures,
            instrument_radius=instrument_radius,
            proximal_length=proximal_length,
            resolution=edge_resolution,
        ):
            shortened_path = np.vstack(
                [
                    shortened_path[
                        : first_index + 1
                    ],
                    shortened_path[
                        second_index:
                    ],
                ]
            )

    return shortened_path


def plan_rrt(
    instrument: SurgicalInstrument,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    structures: tuple[SphericalStructure, ...],
    instrument_radius: float,
    proximal_length: float = 0.10,
    max_iterations: int = 5000,
    step_size: float = 0.08,
    goal_bias: float = 0.15,
    edge_resolution: int = 20,
    seed: int = 7,
) -> PlanningResult:
    """Plan a collision-free path using a basic RRT algorithm."""

    start_q = np.asarray(
        start_q,
        dtype=float,
    )

    goal_q = np.asarray(
        goal_q,
        dtype=float,
    )

    instrument.validate_configuration(
        start_q
    )

    instrument.validate_configuration(
        goal_q
    )

    if not configuration_is_safe(
        instrument,
        start_q,
        structures,
        instrument_radius,
        proximal_length,
    ):
        return PlanningResult(
            path=np.empty((0, 4)),
            success=False,
            iterations=0,
        )

    if not configuration_is_safe(
        instrument,
        goal_q,
        structures,
        instrument_radius,
        proximal_length,
    ):
        return PlanningResult(
            path=np.empty((0, 4)),
            success=False,
            iterations=0,
        )

    if edge_is_safe(
        instrument,
        start_q,
        goal_q,
        structures,
        instrument_radius,
        proximal_length,
        edge_resolution,
    ):
        return PlanningResult(
            path=np.vstack(
                [
                    start_q,
                    goal_q,
                ]
            ),
            success=True,
            iterations=0,
        )

    rng = np.random.default_rng(
        seed
    )

    nodes: list[np.ndarray] = [
        start_q.copy()
    ]

    parents: list[int] = [
        -1
    ]

    for iteration in range(
        1,
        max_iterations + 1,
    ):
        if rng.random() < goal_bias:
            q_sample = goal_q
        else:
            q_sample = (
                sample_random_configuration(
                    instrument,
                    rng,
                )
            )

        distances = np.array(
            [
                joint_distance(
                    node,
                    q_sample,
                )
                for node in nodes
            ]
        )

        nearest_index = int(
            np.argmin(distances)
        )

        q_nearest = nodes[
            nearest_index
        ]

        q_new = steer(
            q_nearest,
            q_sample,
            step_size=step_size,
        )

        try:
            instrument.validate_configuration(
                q_new
            )
        except ValueError:
            continue

        if not edge_is_safe(
            instrument,
            q_nearest,
            q_new,
            structures,
            instrument_radius,
            proximal_length,
            edge_resolution,
        ):
            continue

        nodes.append(
            q_new
        )

        parents.append(
            nearest_index
        )

        new_index = (
            len(nodes) - 1
        )

        if edge_is_safe(
            instrument,
            q_new,
            goal_q,
            structures,
            instrument_radius,
            proximal_length,
            edge_resolution,
        ):
            nodes.append(
                goal_q.copy()
            )

            parents.append(
                new_index
            )

            path_indices: list[int] = []

            current_index = (
                len(nodes) - 1
            )

            while current_index != -1:
                path_indices.append(
                    current_index
                )

                current_index = parents[
                    current_index
                ]

            path_indices.reverse()

            path = np.vstack(
                [
                    nodes[index]
                    for index in path_indices
                ]
            )

            return PlanningResult(
                path=path,
                success=True,
                iterations=iteration,
            )

    return PlanningResult(
        path=np.empty((0, 4)),
        success=False,
        iterations=max_iterations,
    )