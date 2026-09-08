"""Integration tests for Phase 6 chance-constrained robot planning."""

from __future__ import annotations

from dataclasses import dataclass
import inspect

import numpy as np
import pytest

from src.perception.uncertainty import (
    EstimatedStructure,
    PositionUncertainty,
)
from src.robotics.chance_constrained_planner import (
    ChanceConstrainedRRTConfig,
    configuration_is_chance_constrained_safe,
    edge_is_chance_constrained_safe,
    estimated_structure_to_uncertain_sphere,
    estimated_structures_to_uncertain_spheres,
    evaluate_configuration_chance_constraint,
    evaluate_edge_chance_constraint,
    evaluate_joint_path_chance_constraint,
    plan_rrt_chance_constrained,
)
from src.robotics.risk_aware_planning import (
    ChanceConstraintConfig,
)


@dataclass(frozen=True)
class _JointLimits:
    yaw_min: float = -1.0
    yaw_max: float = 1.0

    pitch_min: float = -1.0
    pitch_max: float = 1.0

    insertion_min: float = 0.05
    insertion_max: float = 0.20

    roll_min: float = -1.0
    roll_max: float = 1.0


class _SimpleInstrument:
    """Minimal deterministic instrument implementing planner interfaces.

    The four joint variables are accepted so that the existing planner
    primitives can be reused.

    For these unit tests:

        q[0] -> shaft x position
        q[1] -> shaft y position
        q[2] -> tip z coordinate
        q[3] -> unused roll

    The proximal shaft point is displaced along -z.
    """

    def __init__(
        self,
    ) -> None:
        self.joint_limits = (
            _JointLimits()
        )

    def validate_configuration(
        self,
        q: np.ndarray,
    ) -> None:
        configuration = np.asarray(
            q,
            dtype=float,
        )

        if configuration.shape != (4,):
            raise ValueError(
                "configuration must have shape (4,)."
            )

        limits = self.joint_limits

        if not (
            limits.yaw_min
            <= configuration[0]
            <= limits.yaw_max
        ):
            raise ValueError(
                "yaw outside limits."
            )

        if not (
            limits.pitch_min
            <= configuration[1]
            <= limits.pitch_max
        ):
            raise ValueError(
                "pitch outside limits."
            )

        if not (
            limits.insertion_min
            <= configuration[2]
            <= limits.insertion_max
        ):
            raise ValueError(
                "insertion outside limits."
            )

        if not (
            limits.roll_min
            <= configuration[3]
            <= limits.roll_max
        ):
            raise ValueError(
                "roll outside limits."
            )

    def shaft_segment(
        self,
        q: np.ndarray,
        proximal_length: float = 0.10,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
        self.validate_configuration(
            q
        )

        configuration = np.asarray(
            q,
            dtype=float,
        )

        tip = np.array(
            [
                configuration[0],
                configuration[1],
                configuration[2],
            ],
            dtype=float,
        )

        proximal = tip + np.array(
            [
                0.0,
                0.0,
                -float(
                    proximal_length
                ),
            ],
            dtype=float,
        )

        return (
            proximal,
            tip,
        )


def _estimate(
    centre: np.ndarray,
    covariance: np.ndarray,
    *,
    radius: float = 0.005,
    margin: float = 0.0,
) -> EstimatedStructure:
    return EstimatedStructure(
        estimated_centre=np.asarray(
            centre,
            dtype=float,
        ),
        physical_radius=radius,
        base_safety_margin=margin,
        uncertainty=PositionUncertainty(
            covariance=np.asarray(
                covariance,
                dtype=float,
            )
        ),
    )


def _configuration(
    x: float,
    y: float = 0.0,
    z: float = 0.15,
) -> np.ndarray:
    return np.array(
        [
            x,
            y,
            z,
            0.0,
        ],
        dtype=float,
    )


def test_adapter_preserves_estimated_centre() -> None:
    estimate = _estimate(
        np.array(
            [
                0.01,
                0.02,
                0.03,
            ]
        ),
        np.eye(
            3
        ) * 1e-6,
    )

    sphere = (
        estimated_structure_to_uncertain_sphere(
            estimate
        )
    )

    assert np.allclose(
        sphere.center_mean,
        estimate.estimated_centre,
    )


def test_adapter_preserves_full_covariance() -> None:
    covariance = np.array(
        [
            [
                9e-6,
                1e-6,
                0.0,
            ],
            [
                1e-6,
                4e-6,
                0.0,
            ],
            [
                0.0,
                0.0,
                1e-6,
            ],
        ]
    )

    estimate = _estimate(
        np.zeros(
            3
        ),
        covariance,
    )

    sphere = (
        estimated_structure_to_uncertain_sphere(
            estimate
        )
    )

    assert np.allclose(
        sphere.center_covariance,
        covariance,
    )


def test_multiple_structure_adapter_labels_structures() -> None:
    estimates = (
        _estimate(
            np.zeros(
                3
            ),
            np.eye(
                3
            ) * 1e-6,
        ),
        _estimate(
            np.ones(
                3
            ),
            np.eye(
                3
            ) * 2e-6,
        ),
    )

    spheres = (
        estimated_structures_to_uncertain_spheres(
            estimates
        )
    )

    assert len(
        spheres
    ) == 2

    assert (
        spheres[0].label
        == "structure_0"
    )

    assert (
        spheres[1].label
        == "structure_1"
    )


def test_safe_configuration_is_accepted() -> None:
    instrument = (
        _SimpleInstrument()
    )

    structure = _estimate(
        np.array(
            [
                0.0,
                0.0,
                0.15,
            ]
        ),
        np.eye(
            3
        )
        * (
            0.002
            ** 2
        ),
    )

    result = (
        evaluate_configuration_chance_constraint(
            instrument=instrument,
            q=_configuration(
                0.030
            ),
            estimated_structures=(
                structure,
            ),
            instrument_radius=0.001,
            chance_config=ChanceConstraintConfig(
                max_point_violation_probability=0.05
            ),
            shaft_sample_spacing=0.005,
        )
    )

    assert result.accepted is True


def test_high_risk_configuration_is_rejected() -> None:
    instrument = (
        _SimpleInstrument()
    )

    structure = _estimate(
        np.array(
            [
                0.0,
                0.0,
                0.15,
            ]
        ),
        np.eye(
            3
        )
        * (
            0.005
            ** 2
        ),
    )

    result = (
        evaluate_configuration_chance_constraint(
            instrument=instrument,
            q=_configuration(
                0.010
            ),
            estimated_structures=(
                structure,
            ),
            instrument_radius=0.001,
            chance_config=ChanceConstraintConfig(
                max_point_violation_probability=0.05
            ),
            shaft_sample_spacing=0.005,
        )
    )

    assert result.accepted is False

    assert (
        result.maximum_point_violation_probability
        > 0.05
    )


def test_boolean_configuration_interface_matches_diagnostic() -> None:
    instrument = (
        _SimpleInstrument()
    )

    structure = _estimate(
        np.zeros(
            3
        ),
        np.eye(
            3
        )
        * (
            0.002
            ** 2
        ),
    )

    q = _configuration(
        0.040
    )

    chance_config = (
        ChanceConstraintConfig(
            max_point_violation_probability=0.05
        )
    )

    diagnostic = (
        evaluate_configuration_chance_constraint(
            instrument,
            q,
            (
                structure,
            ),
            0.001,
            chance_config,
        )
    )

    boolean = (
        configuration_is_chance_constrained_safe(
            instrument,
            q,
            (
                structure,
            ),
            0.001,
            chance_config,
        )
    )

    assert (
        boolean
        == diagnostic.accepted
    )


def test_anisotropic_radial_uncertainty_changes_robot_decision() -> None:
    instrument = (
        _SimpleInstrument()
    )

    q = _configuration(
        0.020
    )

    low_radial = _estimate(
        np.array(
            [
                0.0,
                0.0,
                0.15,
            ]
        ),
        np.diag(
            [
                0.001 ** 2,
                0.010 ** 2,
                0.001 ** 2,
            ]
        ),
    )

    high_radial = _estimate(
        np.array(
            [
                0.0,
                0.0,
                0.15,
            ]
        ),
        np.diag(
            [
                0.010 ** 2,
                0.001 ** 2,
                0.001 ** 2,
            ]
        ),
    )

    chance_config = (
        ChanceConstraintConfig(
            max_point_violation_probability=0.05
        )
    )

    low_result = (
        evaluate_configuration_chance_constraint(
            instrument,
            q,
            (
                low_radial,
            ),
            0.001,
            chance_config,
        )
    )

    high_result = (
        evaluate_configuration_chance_constraint(
            instrument,
            q,
            (
                high_radial,
            ),
            0.001,
            chance_config,
        )
    )

    assert (
        high_result.maximum_point_violation_probability
        > low_result.maximum_point_violation_probability
    )

    assert (
        low_result.accepted
        is True
    )

    assert (
        high_result.accepted
        is False
    )


def test_edge_rejects_path_that_moves_through_high_risk_region() -> None:
    instrument = (
        _SimpleInstrument()
    )

    structure = _estimate(
        np.array(
            [
                0.0,
                0.0,
                0.15,
            ]
        ),
        np.eye(
            3
        )
        * (
            0.002
            ** 2
        ),
    )

    result = (
        evaluate_edge_chance_constraint(
            instrument=instrument,
            q_start=_configuration(
                -0.030
            ),
            q_goal=_configuration(
                0.030
            ),
            estimated_structures=(
                structure,
            ),
            instrument_radius=0.001,
            chance_config=ChanceConstraintConfig(
                max_point_violation_probability=0.05
            ),
            resolution=31,
            shaft_sample_spacing=0.005,
        )
    )

    assert result.accepted is False


def test_edge_accepts_path_far_from_structure() -> None:
    instrument = (
        _SimpleInstrument()
    )

    structure = _estimate(
        np.array(
            [
                0.0,
                0.0,
                0.15,
            ]
        ),
        np.eye(
            3
        )
        * (
            0.002
            ** 2
        ),
    )

    result = (
        evaluate_edge_chance_constraint(
            instrument=instrument,
            q_start=_configuration(
                -0.050,
                y=0.050,
            ),
            q_goal=_configuration(
                0.050,
                y=0.050,
            ),
            estimated_structures=(
                structure,
            ),
            instrument_radius=0.001,
            chance_config=ChanceConstraintConfig(
                max_point_violation_probability=0.05
            ),
            resolution=21,
            shaft_sample_spacing=0.005,
        )
    )

    assert result.accepted is True


def test_edge_boolean_interface_matches_diagnostic() -> None:
    instrument = (
        _SimpleInstrument()
    )

    structure = _estimate(
        np.zeros(
            3
        ),
        np.eye(
            3
        )
        * (
            0.001
            ** 2
        ),
    )

    chance_config = (
        ChanceConstraintConfig(
            max_point_violation_probability=0.05
        )
    )

    start = _configuration(
        -0.05,
        y=0.05,
    )

    goal = _configuration(
        0.05,
        y=0.05,
    )

    diagnostic = (
        evaluate_edge_chance_constraint(
            instrument,
            start,
            goal,
            (
                structure,
            ),
            0.001,
            chance_config,
        )
    )

    boolean = (
        edge_is_chance_constrained_safe(
            instrument,
            start,
            goal,
            (
                structure,
            ),
            0.001,
            chance_config,
        )
    )

    assert (
        boolean
        == diagnostic.accepted
    )


def test_stricter_probability_threshold_changes_edge_acceptance() -> None:
    instrument = (
        _SimpleInstrument()
    )

    structure = _estimate(
        np.array(
            [
                0.0,
                0.0,
                0.15,
            ]
        ),
        np.diag(
            [
                0.010 ** 2,
                0.001 ** 2,
                0.001 ** 2,
            ]
        ),
    )

    path_start = _configuration(
        0.020,
        y=0.0,
    )

    path_goal = _configuration(
        0.020,
        y=0.020,
    )

    loose = (
        evaluate_edge_chance_constraint(
            instrument,
            path_start,
            path_goal,
            (
                structure,
            ),
            0.001,
            ChanceConstraintConfig(
                max_point_violation_probability=0.20
            ),
            resolution=10,
        )
    )

    strict = (
        evaluate_edge_chance_constraint(
            instrument,
            path_start,
            path_goal,
            (
                structure,
            ),
            0.001,
            ChanceConstraintConfig(
                max_point_violation_probability=0.05
            ),
            resolution=10,
        )
    )

    assert loose.accepted is True

    assert strict.accepted is False


def test_joint_path_evaluator_accepts_safe_multiedge_path() -> None:
    instrument = (
        _SimpleInstrument()
    )

    structure = _estimate(
        np.zeros(
            3
        ),
        np.eye(
            3
        )
        * (
            0.001
            ** 2
        ),
    )

    path = np.vstack(
        [
            _configuration(
                -0.05,
                y=0.05,
            ),
            _configuration(
                0.0,
                y=0.05,
            ),
            _configuration(
                0.05,
                y=0.05,
            ),
        ]
    )

    result = (
        evaluate_joint_path_chance_constraint(
            instrument,
            path,
            (
                structure,
            ),
            0.001,
            ChanceConstraintConfig(
                max_point_violation_probability=0.05
            ),
        )
    )

    assert result.accepted is True


def test_joint_path_evaluator_rejects_unsafe_multiedge_path() -> None:
    instrument = (
        _SimpleInstrument()
    )

    structure = _estimate(
        np.array(
            [
                0.0,
                0.0,
                0.15,
            ]
        ),
        np.eye(
            3
        )
        * (
            0.002
            ** 2
        ),
    )

    path = np.vstack(
        [
            _configuration(
                -0.03
            ),
            _configuration(
                0.0
            ),
            _configuration(
                0.03
            ),
        ]
    )

    result = (
        evaluate_joint_path_chance_constraint(
            instrument,
            path,
            (
                structure,
            ),
            0.001,
            ChanceConstraintConfig(
                max_point_violation_probability=0.05
            ),
            edge_resolution=15,
        )
    )

    assert result.accepted is False


def test_rrt_returns_direct_path_when_direct_edge_is_safe() -> None:
    instrument = (
        _SimpleInstrument()
    )

    structure = _estimate(
        np.zeros(
            3
        ),
        np.eye(
            3
        )
        * (
            0.001
            ** 2
        ),
    )

    start = _configuration(
        -0.05,
        y=0.05,
    )

    goal = _configuration(
        0.05,
        y=0.05,
    )

    result = (
        plan_rrt_chance_constrained(
            instrument=instrument,
            start_q=start,
            goal_q=goal,
            estimated_structures=(
                structure,
            ),
            instrument_radius=0.001,
            chance_config=ChanceConstraintConfig(
                max_point_violation_probability=0.05
            ),
            planner_config=ChanceConstrainedRRTConfig(
                max_iterations=100,
                edge_resolution=10,
                shaft_sample_spacing=0.005,
                seed=3,
            ),
        )
    )

    assert result.success is True

    assert result.iterations == 0

    assert result.path.shape == (
        2,
        4,
    )

    assert np.allclose(
        result.path[0],
        start,
    )

    assert np.allclose(
        result.path[-1],
        goal,
    )


def test_rrt_rejects_high_risk_start_configuration() -> None:
    instrument = (
        _SimpleInstrument()
    )

    structure = _estimate(
        np.array(
            [
                0.0,
                0.0,
                0.15,
            ]
        ),
        np.eye(
            3
        )
        * (
            0.004
            ** 2
        ),
    )

    result = (
        plan_rrt_chance_constrained(
            instrument=instrument,
            start_q=_configuration(
                0.006
            ),
            goal_q=_configuration(
                0.05
            ),
            estimated_structures=(
                structure,
            ),
            instrument_radius=0.001,
            chance_config=ChanceConstraintConfig(
                max_point_violation_probability=0.05
            ),
        )
    )

    assert result.success is False

    assert result.iterations == 0

    assert result.path.shape == (
        0,
        4,
    )


def test_rrt_is_deterministic_for_fixed_seed() -> None:
    instrument = (
        _SimpleInstrument()
    )

    structure = _estimate(
        np.array(
            [
                0.0,
                0.0,
                0.15,
            ]
        ),
        np.eye(
            3
        )
        * (
            0.001
            ** 2
        ),
    )

    planner_config = (
        ChanceConstrainedRRTConfig(
            max_iterations=100,
            seed=123,
        )
    )

    kwargs = dict(
        instrument=instrument,
        start_q=_configuration(
            -0.05,
            y=0.05,
        ),
        goal_q=_configuration(
            0.05,
            y=0.05,
        ),
        estimated_structures=(
            structure,
        ),
        instrument_radius=0.001,
        chance_config=(
            ChanceConstraintConfig(
                max_point_violation_probability=0.05
            )
        ),
        planner_config=(
            planner_config
        ),
    )

    first = (
        plan_rrt_chance_constrained(
            **kwargs
        )
    )

    second = (
        plan_rrt_chance_constrained(
            **kwargs
        )
    )

    assert (
        first.success
        == second.success
    )

    assert (
        first.iterations
        == second.iterations
    )

    assert np.allclose(
        first.path,
        second.path,
    )


def test_empty_structure_set_is_safe() -> None:
    instrument = (
        _SimpleInstrument()
    )

    result = (
        evaluate_configuration_chance_constraint(
            instrument=instrument,
            q=_configuration(
                0.0
            ),
            estimated_structures=(),
            instrument_radius=0.001,
            chance_config=ChanceConstraintConfig(
                max_point_violation_probability=0.05
            ),
        )
    )

    assert result.accepted is True

    assert (
        result.maximum_point_violation_probability
        == 0.0
    )


def test_runtime_interfaces_do_not_accept_ground_truth() -> None:
    runtime_functions = (
        evaluate_configuration_chance_constraint,
        configuration_is_chance_constrained_safe,
        evaluate_edge_chance_constraint,
        edge_is_chance_constrained_safe,
        evaluate_joint_path_chance_constraint,
        plan_rrt_chance_constrained,
    )

    forbidden_names = {
        "ground_truth",
        "true_structure",
        "true_structures",
        "true_position",
        "true_centre",
    }

    for function in runtime_functions:
        parameter_names = set(
            inspect.signature(
                function
            ).parameters
        )

        assert (
            parameter_names
            & forbidden_names
            == set()
        )


def test_planner_config_validates_resolution() -> None:
    with pytest.raises(
        ValueError
    ):
        ChanceConstrainedRRTConfig(
            edge_resolution=1
        )


def test_planner_config_validates_goal_bias() -> None:
    with pytest.raises(
        ValueError
    ):
        ChanceConstrainedRRTConfig(
            goal_bias=1.1
        )