"""Task-relevance modelling for safety-critical active perception."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TaskRelevanceConfig:
    """Configuration for trajectory-dependent task relevance."""

    relevance_sigma: float = 0.03

    minimum_relevance: float = 0.0

    maximum_relevance: float = 1.0

    def __post_init__(self) -> None:
        if self.relevance_sigma <= 0.0:
            raise ValueError(
                "relevance_sigma must be positive."
            )

        if not 0.0 <= self.minimum_relevance <= 1.0:
            raise ValueError(
                "minimum_relevance must be in [0, 1]."
            )

        if not 0.0 <= self.maximum_relevance <= 1.0:
            raise ValueError(
                "maximum_relevance must be in [0, 1]."
            )

        if (
            self.minimum_relevance
            > self.maximum_relevance
        ):
            raise ValueError(
                "minimum_relevance must not exceed "
                "maximum_relevance."
            )


@dataclass(frozen=True)
class SurgicalTask:
    """Represent the safety-critical surgical task."""

    trajectory: np.ndarray
    safety_critical_points: np.ndarray

    def __post_init__(self) -> None:
        trajectory = np.asarray(
            self.trajectory,
            dtype=float,
        )

        critical_points = np.asarray(
            self.safety_critical_points,
            dtype=float,
        )

        if trajectory.ndim != 2:
            raise ValueError(
                "trajectory must have shape (N, 3)."
            )

        if trajectory.shape[1] != 3:
            raise ValueError(
                "trajectory must contain 3D points."
            )

        if len(trajectory) < 2:
            raise ValueError(
                "trajectory must contain at least two points."
            )

        if critical_points.ndim != 2:
            raise ValueError(
                "safety_critical_points must have "
                "shape (N, 3)."
            )

        if critical_points.shape[1] != 3:
            raise ValueError(
                "safety_critical_points must contain "
                "3D points."
            )

        if len(critical_points) == 0:
            raise ValueError(
                "At least one safety-critical point "
                "is required."
            )


def point_to_polyline_distance(
    point: np.ndarray,
    polyline: np.ndarray,
) -> float:
    """Return the minimum Euclidean distance to a 3D polyline."""

    point = np.asarray(
        point,
        dtype=float,
    )

    polyline = np.asarray(
        polyline,
        dtype=float,
    )

    if point.shape != (3,):
        raise ValueError(
            "point must have shape (3,)."
        )

    if polyline.ndim != 2 or polyline.shape[1] != 3:
        raise ValueError(
            "polyline must have shape (N, 3)."
        )

    if len(polyline) < 2:
        raise ValueError(
            "polyline must contain at least two points."
        )

    starts = polyline[:-1]

    ends = polyline[1:]

    segments = ends - starts

    segment_lengths_squared = np.sum(
        segments * segments,
        axis=1,
    )

    nonzero = (
        segment_lengths_squared > 0.0
    )

    distances = []

    for start, segment, length_sq, valid in zip(
        starts,
        segments,
        segment_lengths_squared,
        nonzero,
    ):
        if not valid:
            distances.append(
                np.linalg.norm(
                    point - start
                )
            )
            continue

        projection = np.dot(
            point - start,
            segment,
        ) / length_sq

        projection = np.clip(
            projection,
            0.0,
            1.0,
        )

        closest = (
            start
            + projection * segment
        )

        distances.append(
            np.linalg.norm(
                point - closest
            )
        )

    return float(
        min(distances)
    )


def task_relevance(
    point: np.ndarray,
    task: SurgicalTask,
    config: TaskRelevanceConfig | None = None,
) -> float:
    """Calculate trajectory-dependent relevance for one point."""

    if config is None:
        config = TaskRelevanceConfig()

    distance = point_to_polyline_distance(
        point=point,
        polyline=task.trajectory,
    )

    relevance = np.exp(
        -0.5
        * (
            distance
            / config.relevance_sigma
        )
        ** 2
    )

    return float(
        np.clip(
            relevance,
            config.minimum_relevance,
            config.maximum_relevance,
        )
    )


def task_relevance_weights(
    task: SurgicalTask,
    config: TaskRelevanceConfig | None = None,
) -> np.ndarray:
    """Calculate relevance weights for all safety-critical points."""

    if config is None:
        config = TaskRelevanceConfig()

    return np.asarray(
        [
            task_relevance(
                point=point,
                task=task,
                config=config,
            )
            for point in task.safety_critical_points
        ],
        dtype=float,
    )


def mean_task_relevance(
    task: SurgicalTask,
    config: TaskRelevanceConfig | None = None,
) -> float:
    """Return mean relevance across safety-critical points."""

    weights = task_relevance_weights(
        task=task,
        config=config,
    )

    return float(
        np.mean(weights)
    )