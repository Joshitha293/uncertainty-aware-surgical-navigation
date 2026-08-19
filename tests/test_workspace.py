import numpy as np
import pytest

from src.geometry.workspace import (
    SphericalStructure,
    SurgicalWorkspace,
)


def test_safety_radius_is_physical_plus_margin():
    structure = SphericalStructure(
        centre=np.zeros(3),
        physical_radius=0.05,
        safety_margin=0.02,
    )

    assert structure.safety_radius == pytest.approx(0.07)


def test_surface_clearance_outside_structure():
    structure = SphericalStructure(
        centre=np.zeros(3),
        physical_radius=0.05,
        safety_margin=0.02,
    )

    point = np.array([0.10, 0.0, 0.0])

    clearance = structure.signed_surface_clearance(point)

    assert clearance == pytest.approx(0.05)


def test_surface_clearance_on_boundary():
    structure = SphericalStructure(
        centre=np.zeros(3),
        physical_radius=0.05,
        safety_margin=0.02,
    )

    point = np.array([0.05, 0.0, 0.0])

    clearance = structure.signed_surface_clearance(point)

    assert clearance == pytest.approx(0.0)


def test_surface_clearance_inside_structure_is_negative():
    structure = SphericalStructure(
        centre=np.zeros(3),
        physical_radius=0.05,
        safety_margin=0.02,
    )

    point = np.array([0.02, 0.0, 0.0])

    clearance = structure.signed_surface_clearance(point)

    assert clearance < 0.0


def test_point_inside_structure_is_collision():
    structure = SphericalStructure(
        centre=np.zeros(3),
        physical_radius=0.05,
        safety_margin=0.02,
    )

    point = np.array([0.01, 0.0, 0.0])

    assert structure.contains_point(point)


def test_point_inside_safety_region_but_outside_structure():
    structure = SphericalStructure(
        centre=np.zeros(3),
        physical_radius=0.05,
        safety_margin=0.02,
    )

    point = np.array([0.06, 0.0, 0.0])

    assert not structure.contains_point(point)
    assert structure.violates_safety_margin(point)


def test_point_outside_safety_region_is_safe():
    structure = SphericalStructure(
        centre=np.zeros(3),
        physical_radius=0.05,
        safety_margin=0.02,
    )

    point = np.array([0.10, 0.0, 0.0])

    assert not structure.violates_safety_margin(point)


def test_workspace_detects_collision():
    structure = SphericalStructure(
        centre=np.array([0.10, 0.0, 0.0]),
        physical_radius=0.03,
        safety_margin=0.02,
    )

    workspace = SurgicalWorkspace(
        target_position=np.array([0.20, 0.0, 0.0]),
        structures=(structure,),
    )

    assert workspace.point_in_collision(
        np.array([0.10, 0.0, 0.0])
    )


def test_workspace_detects_safety_margin_violation():
    structure = SphericalStructure(
        centre=np.array([0.10, 0.0, 0.0]),
        physical_radius=0.03,
        safety_margin=0.02,
    )

    workspace = SurgicalWorkspace(
        target_position=np.array([0.20, 0.0, 0.0]),
        structures=(structure,),
    )

    point = np.array([0.14, 0.0, 0.0])

    assert not workspace.point_in_collision(point)
    assert workspace.point_violates_safety_margin(point)


def test_nearest_surface_clearance_uses_closest_structure():
    structure_a = SphericalStructure(
        centre=np.array([0.0, 0.0, 0.0]),
        physical_radius=0.02,
        safety_margin=0.01,
    )

    structure_b = SphericalStructure(
        centre=np.array([0.20, 0.0, 0.0]),
        physical_radius=0.02,
        safety_margin=0.01,
    )

    workspace = SurgicalWorkspace(
        target_position=np.array([0.30, 0.0, 0.0]),
        structures=(structure_a, structure_b),
    )

    point = np.array([0.16, 0.0, 0.0])

    clearance = workspace.nearest_surface_clearance(point)

    assert clearance == pytest.approx(0.02)


def test_empty_workspace_clearance_is_infinite():
    workspace = SurgicalWorkspace(
        target_position=np.array([0.20, 0.0, 0.0]),
        structures=(),
    )

    assert np.isinf(
        workspace.nearest_surface_clearance(
            np.zeros(3)
        )
    )

    assert np.isinf(
        workspace.nearest_safety_clearance(
            np.zeros(3)
        )
    )


def test_invalid_structure_centre_shape_is_rejected():
    with pytest.raises(ValueError):
        SphericalStructure(
            centre=np.array([0.0, 0.0]),
            physical_radius=0.05,
            safety_margin=0.02,
        )


def test_non_positive_physical_radius_is_rejected():
    with pytest.raises(ValueError):
        SphericalStructure(
            centre=np.zeros(3),
            physical_radius=0.0,
            safety_margin=0.02,
        )


def test_negative_safety_margin_is_rejected():
    with pytest.raises(ValueError):
        SphericalStructure(
            centre=np.zeros(3),
            physical_radius=0.05,
            safety_margin=-0.01,
        )


def test_invalid_target_shape_is_rejected():
    with pytest.raises(ValueError):
        SurgicalWorkspace(
            target_position=np.array([0.0, 0.0]),
            structures=(),
        )