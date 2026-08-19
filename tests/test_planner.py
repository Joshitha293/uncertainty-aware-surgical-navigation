import numpy as np

from src.geometry.workspace import SphericalStructure
from src.robotics.instrument import SurgicalInstrument
from src.robotics.planner import (
    configuration_is_safe,
    edge_is_safe,
    joint_distance,
    path_cost,
    plan_rrt,
    sample_random_configuration,
    shortcut_path,
    steer,
)


def make_test_instrument() -> SurgicalInstrument:
    return SurgicalInstrument(
        rcm_position=np.zeros(3),
    )


def make_structures() -> tuple[SphericalStructure, ...]:
    return (
        SphericalStructure(
            centre=np.array([0.14, 0.04, 0.00]),
            physical_radius=0.025,
            safety_margin=0.015,
        ),
        SphericalStructure(
            centre=np.array([0.18, -0.06, 0.02]),
            physical_radius=0.025,
            safety_margin=0.015,
        ),
    )


def test_safe_configuration_is_accepted():
    instrument = make_test_instrument()
    structures = make_structures()

    q = np.array(
        [
            np.deg2rad(-25.0),
            np.deg2rad(-15.0),
            0.16,
            0.0,
        ]
    )

    assert configuration_is_safe(
        instrument=instrument,
        q=q,
        structures=structures,
        instrument_radius=0.006,
    )


def test_random_configuration_respects_joint_limits():
    instrument = make_test_instrument()

    rng = np.random.default_rng(123)

    for _ in range(100):
        q = sample_random_configuration(
            instrument,
            rng,
        )

        instrument.validate_configuration(q)


def test_joint_distance_zero_for_identical_states():
    q = np.array(
        [
            0.1,
            -0.2,
            0.15,
            0.5,
        ]
    )

    assert joint_distance(q, q) == 0.0


def test_joint_distance_is_symmetric():
    q_a = np.array(
        [
            0.1,
            -0.2,
            0.15,
            0.5,
        ]
    )

    q_b = np.array(
        [
            -0.1,
            0.3,
            0.25,
            -0.4,
        ]
    )

    distance_ab = joint_distance(
        q_a,
        q_b,
    )

    distance_ba = joint_distance(
        q_b,
        q_a,
    )

    np.testing.assert_allclose(
        distance_ab,
        distance_ba,
        atol=1e-12,
    )


def test_path_cost_zero_for_single_waypoint():
    path = np.array(
        [
            [0.0, 0.0, 0.15, 0.0]
        ]
    )

    assert path_cost(path) == 0.0


def test_path_cost_positive_for_nontrivial_path():
    path = np.array(
        [
            [0.0, 0.0, 0.15, 0.0],
            [0.1, 0.0, 0.15, 0.0],
            [0.2, 0.0, 0.15, 0.0],
        ]
    )

    assert path_cost(path) > 0.0


def test_steer_returns_goal_when_within_step():
    q_from = np.zeros(4)

    q_to = np.array(
        [
            0.01,
            0.01,
            0.01,
            0.01,
        ]
    )

    result = steer(
        q_from,
        q_to,
        step_size=0.10,
    )

    np.testing.assert_allclose(
        result,
        q_to,
        atol=1e-12,
    )


def test_steer_limits_step_size():
    q_from = np.zeros(4)

    q_to = np.array(
        [
            1.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    result = steer(
        q_from,
        q_to,
        step_size=0.10,
    )

    distance = np.linalg.norm(
        result - q_from
    )

    np.testing.assert_allclose(
        distance,
        0.10,
        atol=1e-12,
    )


def test_safe_edge_is_accepted():
    instrument = make_test_instrument()

    q_start = np.array(
        [
            np.deg2rad(-20.0),
            np.deg2rad(-10.0),
            0.15,
            0.0,
        ]
    )

    q_goal = np.array(
        [
            np.deg2rad(-10.0),
            np.deg2rad(-5.0),
            0.18,
            0.0,
        ]
    )

    assert edge_is_safe(
        instrument=instrument,
        q_start=q_start,
        q_goal=q_goal,
        structures=(),
        instrument_radius=0.006,
        resolution=20,
    )


def test_direct_path_is_returned_when_safe():
    instrument = make_test_instrument()

    q_start = np.array(
        [
            np.deg2rad(-20.0),
            np.deg2rad(-10.0),
            0.15,
            0.0,
        ]
    )

    q_goal = np.array(
        [
            np.deg2rad(-10.0),
            np.deg2rad(-5.0),
            0.18,
            0.0,
        ]
    )

    result = plan_rrt(
        instrument=instrument,
        start_q=q_start,
        goal_q=q_goal,
        structures=(),
        instrument_radius=0.006,
        seed=123,
    )

    assert result.success
    assert result.path.shape == (2, 4)
    assert result.iterations == 0


def test_rrt_result_is_reproducible_for_fixed_seed():
    instrument = make_test_instrument()

    q_start = np.array(
        [
            np.deg2rad(-25.0),
            np.deg2rad(-15.0),
            0.16,
            0.0,
        ]
    )

    q_goal = np.array(
        [
            np.deg2rad(35.0),
            np.deg2rad(25.0),
            0.25,
            0.0,
        ]
    )

    result_a = plan_rrt(
        instrument=instrument,
        start_q=q_start,
        goal_q=q_goal,
        structures=make_structures(),
        instrument_radius=0.006,
        max_iterations=5000,
        seed=7,
    )

    result_b = plan_rrt(
        instrument=instrument,
        start_q=q_start,
        goal_q=q_goal,
        structures=make_structures(),
        instrument_radius=0.006,
        max_iterations=5000,
        seed=7,
    )

    assert result_a.success == result_b.success
    assert result_a.iterations == result_b.iterations

    np.testing.assert_allclose(
        result_a.path,
        result_b.path,
        atol=1e-12,
    )


def test_shortcut_preserves_start_and_goal():
    instrument = make_test_instrument()

    q_start = np.array(
        [
            np.deg2rad(-25.0),
            np.deg2rad(-15.0),
            0.16,
            0.0,
        ]
    )

    q_goal = np.array(
        [
            np.deg2rad(35.0),
            np.deg2rad(25.0),
            0.25,
            0.0,
        ]
    )

    result = plan_rrt(
        instrument=instrument,
        start_q=q_start,
        goal_q=q_goal,
        structures=make_structures(),
        instrument_radius=0.006,
        max_iterations=5000,
        seed=7,
    )

    assert result.success

    shortened = shortcut_path(
        instrument=instrument,
        path=result.path,
        structures=make_structures(),
        instrument_radius=0.006,
        attempts=300,
        seed=11,
    )

    np.testing.assert_allclose(
        shortened[0],
        q_start,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        shortened[-1],
        q_goal,
        atol=1e-12,
    )


def test_shortcut_does_not_increase_waypoint_count():
    instrument = make_test_instrument()

    q_start = np.array(
        [
            np.deg2rad(-25.0),
            np.deg2rad(-15.0),
            0.16,
            0.0,
        ]
    )

    q_goal = np.array(
        [
            np.deg2rad(35.0),
            np.deg2rad(25.0),
            0.25,
            0.0,
        ]
    )

    result = plan_rrt(
        instrument=instrument,
        start_q=q_start,
        goal_q=q_goal,
        structures=make_structures(),
        instrument_radius=0.006,
        max_iterations=5000,
        seed=7,
    )

    assert result.success

    shortened = shortcut_path(
        instrument=instrument,
        path=result.path,
        structures=make_structures(),
        instrument_radius=0.006,
        attempts=300,
        seed=11,
    )

    assert len(shortened) <= len(result.path)


def test_shortcut_does_not_increase_path_cost():
    instrument = make_test_instrument()

    q_start = np.array(
        [
            np.deg2rad(-25.0),
            np.deg2rad(-15.0),
            0.16,
            0.0,
        ]
    )

    q_goal = np.array(
        [
            np.deg2rad(35.0),
            np.deg2rad(25.0),
            0.25,
            0.0,
        ]
    )

    result = plan_rrt(
        instrument=instrument,
        start_q=q_start,
        goal_q=q_goal,
        structures=make_structures(),
        instrument_radius=0.006,
        max_iterations=5000,
        seed=7,
    )

    assert result.success

    shortened = shortcut_path(
        instrument=instrument,
        path=result.path,
        structures=make_structures(),
        instrument_radius=0.006,
        attempts=300,
        seed=11,
    )

    assert (
        path_cost(shortened)
        <= path_cost(result.path) + 1e-12
    )


def test_every_shortcut_edge_remains_safe():
    instrument = make_test_instrument()

    q_start = np.array(
        [
            np.deg2rad(-25.0),
            np.deg2rad(-15.0),
            0.16,
            0.0,
        ]
    )

    q_goal = np.array(
        [
            np.deg2rad(35.0),
            np.deg2rad(25.0),
            0.25,
            0.0,
        ]
    )

    structures = make_structures()

    result = plan_rrt(
        instrument=instrument,
        start_q=q_start,
        goal_q=q_goal,
        structures=structures,
        instrument_radius=0.006,
        max_iterations=5000,
        seed=7,
    )

    assert result.success

    shortened = shortcut_path(
        instrument=instrument,
        path=result.path,
        structures=structures,
        instrument_radius=0.006,
        attempts=300,
        edge_resolution=30,
        seed=11,
    )

    for index in range(
        len(shortened) - 1
    ):
        assert edge_is_safe(
            instrument=instrument,
            q_start=shortened[index],
            q_goal=shortened[index + 1],
            structures=structures,
            instrument_radius=0.006,
            resolution=30,
        )