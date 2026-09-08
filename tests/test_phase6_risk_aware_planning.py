"""Tests for Phase 6 probabilistic clearance and chance constraints."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.robotics.risk_aware_planning import (
    ChanceConstraintConfig,
    UncertainSphere,
    estimate_clearance_risk,
    evaluate_path_clearance_risk,
    gaussian_clearance_violation_probability,
    sample_polyline,
)


def _sphere(
    covariance: np.ndarray,
    *,
    center: np.ndarray | None = None,
    physical_radius: float = 0.005,
    safety_margin: float = 0.0,
) -> UncertainSphere:
    if center is None:
        center = np.zeros(
            3,
            dtype=float,
        )

    return UncertainSphere(
        center_mean=center,
        center_covariance=covariance,
        physical_radius=physical_radius,
        safety_margin=safety_margin,
        label="critical",
    )


def test_config_rejects_invalid_probability() -> None:
    with pytest.raises(
        ValueError
    ):
        ChanceConstraintConfig(
            max_point_violation_probability=1.1
        )

    with pytest.raises(
        ValueError
    ):
        ChanceConstraintConfig(
            max_point_violation_probability=-0.01
        )


def test_uncertain_sphere_rejects_bad_covariance_shape() -> None:
    with pytest.raises(
        ValueError
    ):
        _sphere(
            np.eye(
                2
            )
        )


def test_uncertain_sphere_rejects_asymmetric_covariance() -> None:
    covariance = np.eye(
        3
    )

    covariance[
        0,
        1,
    ] = 0.01

    with pytest.raises(
        ValueError
    ):
        _sphere(
            covariance
        )


def test_uncertain_sphere_rejects_non_psd_covariance() -> None:
    covariance = np.diag(
        [
            1.0,
            1.0,
            -0.01,
        ]
    )

    with pytest.raises(
        ValueError
    ):
        _sphere(
            covariance
        )


def test_zero_covariance_safe_point_has_zero_probability() -> None:
    structure = _sphere(
        np.zeros(
            (
                3,
                3,
            )
        )
    )

    estimate = estimate_clearance_risk(
        np.array(
            [
                0.020,
                0.0,
                0.0,
            ]
        ),
        structure,
    )

    assert (
        estimate.violation_probability
        == 0.0
    )

    assert (
        estimate.deterministic_limit_used
        is True
    )


def test_zero_covariance_collision_has_probability_one() -> None:
    structure = _sphere(
        np.zeros(
            (
                3,
                3,
            )
        )
    )

    estimate = estimate_clearance_risk(
        np.array(
            [
                0.004,
                0.0,
                0.0,
            ]
        ),
        structure,
    )

    assert (
        estimate.violation_probability
        == 1.0
    )

    assert (
        estimate.mean_clearance
        < 0.0
    )


def test_exact_structure_centre_is_handled_conservatively() -> None:
    structure = _sphere(
        np.eye(
            3
        )
        * (
            0.002
            ** 2
        )
    )

    estimate = estimate_clearance_risk(
        np.zeros(
            3
        ),
        structure,
    )

    assert (
        estimate.linearisation_valid
        is False
    )

    assert (
        estimate.violation_probability
        == 1.0
    )


def test_smaller_clearance_produces_higher_violation_probability() -> None:
    structure = _sphere(
        np.eye(
            3
        )
        * (
            0.005
            ** 2
        )
    )

    far = estimate_clearance_risk(
        np.array(
            [
                0.025,
                0.0,
                0.0,
            ]
        ),
        structure,
    )

    near = estimate_clearance_risk(
        np.array(
            [
                0.010,
                0.0,
                0.0,
            ]
        ),
        structure,
    )

    assert (
        near.mean_clearance
        < far.mean_clearance
    )

    assert (
        near.violation_probability
        > far.violation_probability
    )


def test_greater_radial_uncertainty_increases_risk() -> None:
    point = np.array(
        [
            0.020,
            0.0,
            0.0,
        ]
    )

    low_radial = _sphere(
        np.diag(
            [
                0.001 ** 2,
                0.010 ** 2,
                0.010 ** 2,
            ]
        )
    )

    high_radial = _sphere(
        np.diag(
            [
                0.010 ** 2,
                0.001 ** 2,
                0.001 ** 2,
            ]
        )
    )

    low = estimate_clearance_risk(
        point,
        low_radial,
    )

    high = estimate_clearance_risk(
        point,
        high_radial,
    )

    assert (
        high.directional_sigma
        > low.directional_sigma
    )

    assert (
        high.violation_probability
        > low.violation_probability
    )


def test_covariance_orientation_relative_to_obstacle_matters() -> None:
    structure = _sphere(
        np.diag(
            [
                0.010 ** 2,
                0.001 ** 2,
                0.001 ** 2,
            ]
        )
    )

    x_direction = estimate_clearance_risk(
        np.array(
            [
                0.020,
                0.0,
                0.0,
            ]
        ),
        structure,
    )

    y_direction = estimate_clearance_risk(
        np.array(
            [
                0.0,
                0.020,
                0.0,
            ]
        ),
        structure,
    )

    assert x_direction.mean_clearance == pytest.approx(
        y_direction.mean_clearance
    )

    assert (
        x_direction.directional_sigma
        > y_direction.directional_sigma
    )

    assert (
        x_direction.violation_probability
        > y_direction.violation_probability
    )


def test_instrument_radius_increases_risk() -> None:
    structure = _sphere(
        np.eye(
            3
        )
        * (
            0.003
            ** 2
        )
    )

    point = np.array(
        [
            0.015,
            0.0,
            0.0,
        ]
    )

    no_instrument_radius = estimate_clearance_risk(
        point,
        structure,
        instrument_radius=0.0,
    )

    finite_instrument_radius = estimate_clearance_risk(
        point,
        structure,
        instrument_radius=0.004,
    )

    assert (
        finite_instrument_radius.mean_clearance
        < no_instrument_radius.mean_clearance
    )

    assert (
        finite_instrument_radius.violation_probability
        > no_instrument_radius.violation_probability
    )


def test_safety_margin_increases_risk() -> None:
    covariance = np.eye(
        3
    ) * (
        0.003
        ** 2
    )

    no_margin = _sphere(
        covariance,
        safety_margin=0.0,
    )

    larger_margin = _sphere(
        covariance,
        safety_margin=0.004,
    )

    point = np.array(
        [
            0.015,
            0.0,
            0.0,
        ]
    )

    result_no_margin = estimate_clearance_risk(
        point,
        no_margin,
    )

    result_larger_margin = estimate_clearance_risk(
        point,
        larger_margin,
    )

    assert (
        result_larger_margin.mean_clearance
        < result_no_margin.mean_clearance
    )

    assert (
        result_larger_margin.violation_probability
        > result_no_margin.violation_probability
    )


def test_sample_polyline_respects_maximum_spacing() -> None:
    path = np.array(
        [
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                0.010,
                0.0,
                0.0,
            ],
        ]
    )

    sampled = sample_polyline(
        path,
        max_spacing=0.003,
    )

    segment_lengths = np.linalg.norm(
        np.diff(
            sampled,
            axis=0,
        ),
        axis=1,
    )

    assert np.allclose(
        sampled[0],
        path[0],
    )

    assert np.allclose(
        sampled[-1],
        path[-1],
    )

    assert (
        np.max(
            segment_lengths
        )
        <= 0.003 + 1e-12
    )


def test_path_risk_identifies_closest_high_risk_event() -> None:
    structure = _sphere(
        np.eye(
            3
        )
        * (
            0.002
            ** 2
        )
    )

    path = np.array(
        [
            [
                0.030,
                0.0,
                0.0,
            ],
            [
                0.008,
                0.0,
                0.0,
            ],
            [
                0.030,
                0.0,
                0.0,
            ],
        ]
    )

    result = evaluate_path_clearance_risk(
        path,
        [
            structure
        ],
        config=ChanceConstraintConfig(
            max_point_violation_probability=1.0
        ),
    )

    assert (
        result.worst_event
        is not None
    )

    assert (
        result.worst_event.point_index
        == 1
    )


def test_union_bound_is_not_smaller_than_maximum_point_risk() -> None:
    structure = _sphere(
        np.eye(
            3
        )
        * (
            0.003
            ** 2
        )
    )

    path = np.array(
        [
            [
                0.015,
                0.0,
                0.0,
            ],
            [
                0.016,
                0.0,
                0.0,
            ],
            [
                0.017,
                0.0,
                0.0,
            ],
        ]
    )

    result = evaluate_path_clearance_risk(
        path,
        [
            structure
        ],
        config=ChanceConstraintConfig(
            max_point_violation_probability=1.0
        ),
    )

    assert (
        result.union_bound_violation_probability
        >= result.maximum_point_violation_probability
    )


def test_stricter_point_risk_threshold_rejects_same_path() -> None:
    structure = _sphere(
        np.diag(
            [
                0.010 ** 2,
                0.001 ** 2,
                0.001 ** 2,
            ]
        )
    )

    path = np.array(
        [
            [
                0.020,
                0.0,
                0.0,
            ]
        ]
    )

    loose = evaluate_path_clearance_risk(
        path,
        [
            structure
        ],
        config=ChanceConstraintConfig(
            max_point_violation_probability=0.20
        ),
    )

    strict = evaluate_path_clearance_risk(
        path,
        [
            structure
        ],
        config=ChanceConstraintConfig(
            max_point_violation_probability=0.05
        ),
    )

    assert loose.accepted is True

    assert strict.accepted is False


def test_optional_path_union_bound_can_reject_locally_acceptable_path() -> None:
    structure = _sphere(
        np.eye(
            3
        )
        * (
            0.005
            ** 2
        )
    )

    # Mean clearance is 0.010 m = 2 sigma, giving a pointwise probability
    # of approximately 0.0228. Three repeated samples therefore satisfy the
    # 0.05 pointwise limit but violate a 0.05 union-bound limit.
    path = np.array(
        [
            [
                0.015,
                0.0,
                0.0,
            ],
            [
                0.015,
                0.0,
                0.0,
            ],
            [
                0.015,
                0.0,
                0.0,
            ],
        ]
    )

    result = evaluate_path_clearance_risk(
        path,
        [
            structure
        ],
        config=ChanceConstraintConfig(
            max_point_violation_probability=0.05,
            max_path_union_bound_probability=0.05,
        ),
    )

    assert (
        result.maximum_point_violation_probability
        < 0.05
    )

    assert (
        result.union_bound_violation_probability
        > 0.05
    )

    assert result.accepted is False


def test_gaussian_probability_is_half_at_zero_mean_clearance() -> None:
    probability = (
        gaussian_clearance_violation_probability(
            mean_clearance=0.0,
            directional_sigma=0.005,
        )
    )

    assert probability == pytest.approx(
        0.5
    )


def test_runtime_risk_interfaces_do_not_accept_ground_truth() -> None:
    point_signature = inspect.signature(
        estimate_clearance_risk
    )

    path_signature = inspect.signature(
        evaluate_path_clearance_risk
    )

    for signature in (
        point_signature,
        path_signature,
    ):
        parameter_names = set(
            signature.parameters
        )

        assert (
            "ground_truth"
            not in parameter_names
        )

        assert (
            "true_position"
            not in parameter_names
        )

        assert (
            "true_structure"
            not in parameter_names
        )