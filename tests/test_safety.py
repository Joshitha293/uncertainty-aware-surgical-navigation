import numpy as np
import pytest

from src.geometry.workspace import SphericalStructure
from src.robotics.safety import (
    evaluate_instrument_safety,
    point_to_segment_distance,
    structure_clearance_to_instrument,
)


def test_point_to_segment_distance_known_case():
    point = np.array([0.0, 1.0, 0.0])
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([1.0, 0.0, 0.0])

    distance = point_to_segment_distance(
        point,
        start,
        end,
    )

    assert distance == pytest.approx(1.0)


def test_point_projection_beyond_segment_uses_endpoint():
    point = np.array([2.0, 1.0, 0.0])
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([1.0, 0.0, 0.0])

    distance = point_to_segment_distance(
        point,
        start,
        end,
    )

    expected = np.sqrt(2.0)

    assert distance == pytest.approx(expected)


def test_zero_length_segment_is_rejected():
    point = np.zeros(3)
    start = np.zeros(3)
    end = np.zeros(3)

    with pytest.raises(ValueError):
        point_to_segment_distance(
            point,
            start,
            end,
        )


def test_structure_clearance_safe_case():
    structure = SphericalStructure(
        centre=np.array([0.0, 0.10, 0.0]),
        physical_radius=0.02,
        safety_margin=0.02,
    )

    shaft_start = np.array([-0.10, 0.0, 0.0])
    shaft_end = np.array([0.20, 0.0, 0.0])

    surface_clearance, safety_clearance = (
        structure_clearance_to_instrument(
            structure=structure,
            shaft_start=shaft_start,
            shaft_end=shaft_end,
            instrument_radius=0.005,
        )
    )

    assert surface_clearance == pytest.approx(0.075)
    assert safety_clearance == pytest.approx(0.055)


def test_structure_clearance_collision_case():
    structure = SphericalStructure(
        centre=np.array([0.0, 0.015, 0.0]),
        physical_radius=0.02,
        safety_margin=0.02,
    )

    shaft_start = np.array([-0.10, 0.0, 0.0])
    shaft_end = np.array([0.20, 0.0, 0.0])

    surface_clearance, safety_clearance = (
        structure_clearance_to_instrument(
            structure=structure,
            shaft_start=shaft_start,
            shaft_end=shaft_end,
            instrument_radius=0.005,
        )
    )

    assert surface_clearance < 0.0
    assert safety_clearance < 0.0


def test_safety_margin_violation_without_collision():
    structure = SphericalStructure(
        centre=np.array([0.0, 0.035, 0.0]),
        physical_radius=0.02,
        safety_margin=0.02,
    )

    evaluation = evaluate_instrument_safety(
        shaft_start=np.array([-0.10, 0.0, 0.0]),
        shaft_end=np.array([0.20, 0.0, 0.0]),
        structures=(structure,),
        instrument_radius=0.005,
    )

    assert not evaluation.collision
    assert evaluation.safety_margin_violation


def test_collision_detected():
    structure = SphericalStructure(
        centre=np.array([0.0, 0.015, 0.0]),
        physical_radius=0.02,
        safety_margin=0.02,
    )

    evaluation = evaluate_instrument_safety(
        shaft_start=np.array([-0.10, 0.0, 0.0]),
        shaft_end=np.array([0.20, 0.0, 0.0]),
        structures=(structure,),
        instrument_radius=0.005,
    )

    assert evaluation.collision
    assert evaluation.safety_margin_violation


def test_safe_instrument_configuration():
    structure = SphericalStructure(
        centre=np.array([0.0, 0.10, 0.0]),
        physical_radius=0.02,
        safety_margin=0.02,
    )

    evaluation = evaluate_instrument_safety(
        shaft_start=np.array([-0.10, 0.0, 0.0]),
        shaft_end=np.array([0.20, 0.0, 0.0]),
        structures=(structure,),
        instrument_radius=0.005,
    )

    assert not evaluation.collision
    assert not evaluation.safety_margin_violation
    assert evaluation.minimum_surface_clearance > 0.0
    assert evaluation.minimum_safety_clearance > 0.0


def test_nearest_structure_controls_minimum_clearance():
    structure_a = SphericalStructure(
        centre=np.array([0.0, 0.08, 0.0]),
        physical_radius=0.02,
        safety_margin=0.01,
    )

    structure_b = SphericalStructure(
        centre=np.array([0.0, 0.15, 0.0]),
        physical_radius=0.02,
        safety_margin=0.01,
    )

    evaluation = evaluate_instrument_safety(
        shaft_start=np.array([-0.10, 0.0, 0.0]),
        shaft_end=np.array([0.20, 0.0, 0.0]),
        structures=(
            structure_a,
            structure_b,
        ),
        instrument_radius=0.005,
    )

    expected_surface_clearance = (
        0.08 - 0.02 - 0.005
    )

    assert (
        evaluation.minimum_surface_clearance
        == pytest.approx(expected_surface_clearance)
    )


def test_empty_structure_set_is_safe():
    evaluation = evaluate_instrument_safety(
        shaft_start=np.array([-0.10, 0.0, 0.0]),
        shaft_end=np.array([0.20, 0.0, 0.0]),
        structures=(),
        instrument_radius=0.005,
    )

    assert np.isinf(
        evaluation.minimum_surface_clearance
    )

    assert np.isinf(
        evaluation.minimum_safety_clearance
    )

    assert not evaluation.collision
    assert not evaluation.safety_margin_violation


def test_non_positive_instrument_radius_is_rejected():
    structure = SphericalStructure(
        centre=np.zeros(3),
        physical_radius=0.02,
        safety_margin=0.01,
    )

    with pytest.raises(ValueError):
        evaluate_instrument_safety(
            shaft_start=np.array([-0.10, 0.0, 0.0]),
            shaft_end=np.array([0.20, 0.0, 0.0]),
            structures=(structure,),
            instrument_radius=0.0,
        )


def test_invalid_point_shape_is_rejected():
    with pytest.raises(ValueError):
        point_to_segment_distance(
            point=np.array([0.0, 0.0]),
            segment_start=np.zeros(3),
            segment_end=np.ones(3),
        )