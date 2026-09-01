"""Tests for Phase 2 RRT* and discretised configuration-space A*."""

import numpy as np
import pytest

from src.geometry.workspace import (
    SphericalStructure,
)
from src.robotics.advanced_planning import (
    AStarGridConfig,
    plan_astar_configuration_grid,
    plan_rrt_star,
)
from src.robotics.instrument import (
    SurgicalInstrument,
)
from src.robotics.planner import (
    edge_is_safe,
    path_cost,
)


def make_instrument() -> SurgicalInstrument:
    """Return deterministic RCM test instrument."""

    return SurgicalInstrument(
        rcm_position=np.zeros(
            3,
            dtype=float,
        )
    )


def make_structures() -> tuple[
    SphericalStructure,
    ...,
]:
    """Return the established obstacle configuration."""

    return (
        SphericalStructure(
            centre=np.asarray(
                [
                    0.14,
                    0.04,
                    0.00,
                ],
                dtype=float,
            ),
            physical_radius=0.025,
            safety_margin=0.015,
        ),
        SphericalStructure(
            centre=np.asarray(
                [
                    0.18,
                    -0.06,
                    0.02,
                ],
                dtype=float,
            ),
            physical_radius=0.025,
            safety_margin=0.015,
        ),
    )


def obstacle_start() -> np.ndarray:
    return np.asarray(
        [
            np.deg2rad(
                -25.0
            ),
            np.deg2rad(
                -15.0
            ),
            0.16,
            0.0,
        ],
        dtype=float,
    )


def obstacle_goal() -> np.ndarray:
    return np.asarray(
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


def test_rrt_star_returns_direct_path_when_safe():
    instrument = (
        make_instrument()
    )

    start = np.asarray(
        [
            -0.20,
            -0.10,
            0.15,
            0.0,
        ]
    )

    goal = np.asarray(
        [
            -0.10,
            -0.05,
            0.18,
            0.0,
        ]
    )

    result = plan_rrt_star(
        instrument=instrument,
        start_q=start,
        goal_q=goal,
        structures=(),
        instrument_radius=0.006,
        seed=123,
    )

    assert result.success
    assert result.iterations == 0
    assert result.path.shape == (
        2,
        4,
    )

    np.testing.assert_allclose(
        result.path[
            0
        ],
        start,
    )

    np.testing.assert_allclose(
        result.path[
            -1
        ],
        goal,
    )


def test_rrt_star_finds_obstacle_avoiding_path():
    instrument = (
        make_instrument()
    )

    result = plan_rrt_star(
        instrument=instrument,
        start_q=obstacle_start(),
        goal_q=obstacle_goal(),
        structures=make_structures(),
        instrument_radius=0.006,
        max_iterations=500,
        rewire_radius=0.60,
        seed=7,
    )

    assert result.success
    assert result.path.shape[
        0
    ] >= 3

    assert np.isfinite(
        path_cost(
            result.path
        )
    )


def test_rrt_star_path_edges_are_safe():
    instrument = (
        make_instrument()
    )

    structures = (
        make_structures()
    )

    result = plan_rrt_star(
        instrument=instrument,
        start_q=obstacle_start(),
        goal_q=obstacle_goal(),
        structures=structures,
        instrument_radius=0.006,
        max_iterations=500,
        rewire_radius=0.60,
        edge_resolution=20,
        seed=7,
    )

    assert result.success

    for index in range(
        len(
            result.path
        )
        - 1
    ):
        assert edge_is_safe(
            instrument=instrument,
            q_start=result.path[
                index
            ],
            q_goal=result.path[
                index
                + 1
            ],
            structures=structures,
            instrument_radius=0.006,
            resolution=20,
        )


def test_rrt_star_is_reproducible_for_fixed_seed():
    instrument = (
        make_instrument()
    )

    kwargs = dict(
        instrument=instrument,
        start_q=obstacle_start(),
        goal_q=obstacle_goal(),
        structures=make_structures(),
        instrument_radius=0.006,
        max_iterations=400,
        rewire_radius=0.60,
        seed=17,
    )

    first = plan_rrt_star(
        **kwargs
    )

    second = plan_rrt_star(
        **kwargs
    )

    assert (
        first.success
        == second.success
    )

    assert (
        first.iterations
        == second.iterations
    )

    np.testing.assert_allclose(
        first.path,
        second.path,
        atol=1e-12,
    )


def test_rrt_star_rejects_invalid_rewire_radius():
    instrument = (
        make_instrument()
    )

    with pytest.raises(
        ValueError,
        match="rewire_radius",
    ):
        plan_rrt_star(
            instrument=instrument,
            start_q=obstacle_start(),
            goal_q=obstacle_goal(),
            structures=(),
            instrument_radius=0.006,
            rewire_radius=0.0,
        )


def test_astar_returns_direct_path_when_safe():
    instrument = (
        make_instrument()
    )

    start = np.asarray(
        [
            -0.20,
            -0.10,
            0.15,
            0.0,
        ]
    )

    goal = np.asarray(
        [
            -0.10,
            -0.05,
            0.18,
            0.0,
        ]
    )

    result = (
        plan_astar_configuration_grid(
            instrument=instrument,
            start_q=start,
            goal_q=goal,
            structures=(),
            instrument_radius=0.006,
        )
    )

    assert result.success
    assert result.iterations == 0

    np.testing.assert_allclose(
        result.path[
            0
        ],
        start,
    )

    np.testing.assert_allclose(
        result.path[
            -1
        ],
        goal,
    )


def test_astar_finds_path_through_discretised_configuration_space():
    instrument = (
        make_instrument()
    )

    result = (
        plan_astar_configuration_grid(
            instrument=instrument,
            start_q=obstacle_start(),
            goal_q=obstacle_goal(),
            structures=make_structures(),
            instrument_radius=0.006,
            edge_resolution=12,
            grid_config=(
                AStarGridConfig(
                    yaw_samples=7,
                    pitch_samples=7,
                    insertion_samples=5,
                    roll_samples=3,
                    max_expansions=10000,
                )
            ),
        )
    )

    assert result.success

    assert (
        result.iterations
        > 0
    )

    assert (
        result.path.shape[
            0
        ]
        >= 3
    )


def test_astar_path_edges_remain_safe():
    instrument = (
        make_instrument()
    )

    structures = (
        make_structures()
    )

    result = (
        plan_astar_configuration_grid(
            instrument=instrument,
            start_q=obstacle_start(),
            goal_q=obstacle_goal(),
            structures=structures,
            instrument_radius=0.006,
            edge_resolution=12,
            grid_config=(
                AStarGridConfig(
                    yaw_samples=7,
                    pitch_samples=7,
                    insertion_samples=5,
                    roll_samples=3,
                    max_expansions=10000,
                )
            ),
        )
    )

    assert result.success

    for index in range(
        len(
            result.path
        )
        - 1
    ):
        assert edge_is_safe(
            instrument=instrument,
            q_start=result.path[
                index
            ],
            q_goal=result.path[
                index
                + 1
            ],
            structures=structures,
            instrument_radius=0.006,
            resolution=12,
        )


def test_astar_is_deterministic():
    instrument = (
        make_instrument()
    )

    config = AStarGridConfig(
        yaw_samples=7,
        pitch_samples=7,
        insertion_samples=5,
        roll_samples=3,
        max_expansions=10000,
    )

    first = (
        plan_astar_configuration_grid(
            instrument=instrument,
            start_q=obstacle_start(),
            goal_q=obstacle_goal(),
            structures=make_structures(),
            instrument_radius=0.006,
            edge_resolution=12,
            grid_config=config,
        )
    )

    second = (
        plan_astar_configuration_grid(
            instrument=instrument,
            start_q=obstacle_start(),
            goal_q=obstacle_goal(),
            structures=make_structures(),
            instrument_radius=0.006,
            edge_resolution=12,
            grid_config=config,
        )
    )

    assert first.success
    assert second.success

    assert (
        first.iterations
        == second.iterations
    )

    np.testing.assert_allclose(
        first.path,
        second.path,
        atol=1e-12,
    )


def test_astar_grid_configuration_validation():
    with pytest.raises(
        ValueError,
        match="yaw_samples",
    ):
        AStarGridConfig(
            yaw_samples=1,
        )