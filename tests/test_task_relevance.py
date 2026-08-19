"""Tests for trajectory-dependent task relevance."""

import numpy as np
import pytest

from src.perception.task_relevance import (
    SurgicalTask,
    TaskRelevanceConfig,
    mean_task_relevance,
    point_to_polyline_distance,
    task_relevance,
    task_relevance_weights,
)


def make_task() -> SurgicalTask:
    trajectory = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.10, 0.0, 0.0],
            [0.20, 0.0, 0.0],
        ],
        dtype=float,
    )

    critical_points = np.array(
        [
            [0.05, 0.0, 0.0],
            [0.10, 0.0, 0.0],
            [0.15, 0.02, 0.0],
        ],
        dtype=float,
    )

    return SurgicalTask(
        trajectory=trajectory,
        safety_critical_points=critical_points,
    )


def test_task_accepts_valid_trajectory() -> None:
    task = make_task()

    assert task.trajectory.shape == (
        3,
        3,
    )

    assert task.safety_critical_points.shape == (
        3,
        3,
    )


def test_short_trajectory_is_rejected() -> None:
    with pytest.raises(ValueError):
        SurgicalTask(
            trajectory=np.array(
                [[0.0, 0.0, 0.0]]
            ),
            safety_critical_points=np.array(
                [[0.1, 0.0, 0.0]]
            ),
        )


def test_invalid_trajectory_shape_is_rejected() -> None:
    with pytest.raises(ValueError):
        SurgicalTask(
            trajectory=np.array(
                [
                    [0.0, 0.0],
                    [0.1, 0.0],
                ]
            ),
            safety_critical_points=np.array(
                [[0.1, 0.0, 0.0]]
            ),
        )


def test_empty_critical_points_are_rejected() -> None:
    with pytest.raises(ValueError):
        SurgicalTask(
            trajectory=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.0],
                ]
            ),
            safety_critical_points=np.empty(
                (0, 3)
            ),
        )


def test_point_on_trajectory_has_zero_distance() -> None:
    task = make_task()

    distance = point_to_polyline_distance(
        point=np.array(
            [0.10, 0.0, 0.0]
        ),
        polyline=task.trajectory,
    )

    assert distance == pytest.approx(
        0.0
    )


def test_point_off_trajectory_has_positive_distance() -> None:
    task = make_task()

    distance = point_to_polyline_distance(
        point=np.array(
            [0.10, 0.03, 0.0]
        ),
        polyline=task.trajectory,
    )

    assert distance == pytest.approx(
        0.03
    )


def test_relevance_is_maximum_on_trajectory() -> None:
    task = make_task()

    relevance = task_relevance(
        point=np.array(
            [0.10, 0.0, 0.0]
        ),
        task=task,
    )

    assert relevance == pytest.approx(
        1.0
    )


def test_relevance_decreases_with_distance() -> None:
    task = make_task()

    near = task_relevance(
        point=np.array(
            [0.10, 0.01, 0.0]
        ),
        task=task,
    )

    far = task_relevance(
        point=np.array(
            [0.10, 0.10, 0.0]
        ),
        task=task,
    )

    assert near > far


def test_relevance_is_bounded() -> None:
    task = make_task()

    points = [
        np.array(
            [0.10, 0.0, 0.0]
        ),
        np.array(
            [0.10, 0.20, 0.0]
        ),
        np.array(
            [1.0, 1.0, 1.0]
        ),
    ]

    for point in points:
        relevance = task_relevance(
            point=point,
            task=task,
        )

        assert 0.0 <= relevance <= 1.0


def test_weights_match_number_of_critical_points() -> None:
    task = make_task()

    weights = task_relevance_weights(
        task
    )

    assert weights.shape == (
        len(
            task.safety_critical_points
        ),
    )


def test_weights_are_finite() -> None:
    task = make_task()

    weights = task_relevance_weights(
        task
    )

    assert np.all(
        np.isfinite(weights)
    )


def test_mean_relevance_is_bounded() -> None:
    task = make_task()

    mean = mean_task_relevance(
        task
    )

    assert 0.0 <= mean <= 1.0


def test_relevance_sigma_controls_spatial_decay() -> None:
    task = make_task()

    narrow = TaskRelevanceConfig(
        relevance_sigma=0.01
    )

    broad = TaskRelevanceConfig(
        relevance_sigma=0.10
    )

    point = np.array(
        [0.10, 0.03, 0.0]
    )

    narrow_value = task_relevance(
        point=point,
        task=task,
        config=narrow,
    )

    broad_value = task_relevance(
        point=point,
        task=task,
        config=broad,
    )

    assert broad_value > narrow_value


def test_invalid_relevance_sigma_is_rejected() -> None:
    with pytest.raises(ValueError):
        TaskRelevanceConfig(
            relevance_sigma=0.0
        )

    with pytest.raises(ValueError):
        TaskRelevanceConfig(
            relevance_sigma=-0.01
        )