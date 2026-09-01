"""Tests for Phase 5 rigid point registration."""

import numpy as np
import pytest

from src.geometry.registration import (
    registration_error_at_targets,
    register_corresponding_points,
    rms_point_error,
    rotation_angle_between,
    transform_error,
    transform_points,
)
from src.geometry.transforms import (
    is_homogeneous_transform,
    make_transform,
)


def rotation_z(
    angle_radians: float,
) -> np.ndarray:
    """Construct a rotation about the z axis."""

    cosine = np.cos(
        angle_radians
    )

    sine = np.sin(
        angle_radians
    )

    return np.asarray(
        [
            [
                cosine,
                -sine,
                0.0,
            ],
            [
                sine,
                cosine,
                0.0,
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
        ],
        dtype=float,
    )


def rotation_xyz(
    ax: float,
    ay: float,
    az: float,
) -> np.ndarray:
    """Construct a proper 3-D rotation."""

    cx = np.cos(
        ax
    )
    sx = np.sin(
        ax
    )

    cy = np.cos(
        ay
    )
    sy = np.sin(
        ay
    )

    cz = np.cos(
        az
    )
    sz = np.sin(
        az
    )

    rx = np.asarray(
        [
            [
                1.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                cx,
                -sx,
            ],
            [
                0.0,
                sx,
                cx,
            ],
        ]
    )

    ry = np.asarray(
        [
            [
                cy,
                0.0,
                sy,
            ],
            [
                0.0,
                1.0,
                0.0,
            ],
            [
                -sy,
                0.0,
                cy,
            ],
        ]
    )

    rz = rotation_z(
        az
    )

    return (
        rz
        @ ry
        @ rx
    )


def source_landmarks() -> np.ndarray:
    """Return non-degenerate synthetic landmarks."""

    return np.asarray(
        [
            [
                0.000,
                0.000,
                0.000,
            ],
            [
                0.040,
                0.000,
                0.005,
            ],
            [
                0.000,
                0.050,
                -0.004,
            ],
            [
                0.030,
                0.025,
                0.020,
            ],
            [
                -0.020,
                0.018,
                0.015,
            ],
            [
                0.012,
                -0.025,
                0.030,
            ],
        ],
        dtype=float,
    )


def reference_transform() -> np.ndarray:
    """Return known source-to-target transform."""

    return make_transform(
        rotation=rotation_xyz(
            np.deg2rad(
                8.0
            ),
            np.deg2rad(
                -12.0
            ),
            np.deg2rad(
                18.0
            ),
        ),
        translation=np.asarray(
            [
                0.080,
                -0.030,
                0.120,
            ],
            dtype=float,
        ),
    )


def test_transform_points_matches_known_rigid_transform():
    source = source_landmarks()

    transform = reference_transform()

    transformed = transform_points(
        transform,
        source,
    )

    expected = (
        source
        @ transform[
            :3,
            :3,
        ].T
        + transform[
            :3,
            3,
        ]
    )

    np.testing.assert_allclose(
        transformed,
        expected,
        atol=1e-12,
    )


def test_exact_correspondence_recovers_known_transform():
    source = source_landmarks()

    reference = reference_transform()

    target = transform_points(
        reference,
        source,
    )

    result = register_corresponding_points(
        source,
        target,
    )

    np.testing.assert_allclose(
        result.transform,
        reference,
        atol=1e-10,
    )


def test_registration_transform_is_valid_homogeneous_transform():
    source = source_landmarks()

    target = transform_points(
        reference_transform(),
        source,
    )

    result = register_corresponding_points(
        source,
        target,
    )

    assert is_homogeneous_transform(
        result.transform
    )


def test_exact_registration_has_near_zero_fre():
    source = source_landmarks()

    target = transform_points(
        reference_transform(),
        source,
    )

    result = register_corresponding_points(
        source,
        target,
    )

    assert result.fre_rms < 1e-10

    assert result.fre_mean < 1e-10

    assert result.fre_max < 1e-10


def test_exact_registration_transformed_points_match_targets():
    source = source_landmarks()

    target = transform_points(
        reference_transform(),
        source,
    )

    result = register_corresponding_points(
        source,
        target,
    )

    np.testing.assert_allclose(
        result.transformed_source_points,
        target,
        atol=1e-10,
    )


def test_noisy_registration_remains_close_to_reference_transform():
    source = source_landmarks()

    reference = reference_transform()

    exact_target = transform_points(
        reference,
        source,
    )

    rng = np.random.default_rng(
        5101
    )

    noisy_target = (
        exact_target
        + rng.normal(
            loc=0.0,
            scale=0.0005,
            size=exact_target.shape,
        )
    )

    result = register_corresponding_points(
        source,
        noisy_target,
    )

    error = transform_error(
        result.transform,
        reference,
    )

    assert (
        error.translation_error
        < 0.002
    )

    assert (
        error.rotation_angle_error_degrees
        < 2.0
    )


def test_noisy_registration_reports_nonzero_fre():
    source = source_landmarks()

    target = transform_points(
        reference_transform(),
        source,
    )

    rng = np.random.default_rng(
        42
    )

    target = (
        target
        + rng.normal(
            0.0,
            0.001,
            size=target.shape,
        )
    )

    result = register_corresponding_points(
        source,
        target,
    )

    assert result.fre_rms > 0.0

    assert result.fre_mean > 0.0

    assert result.fre_max >= result.fre_mean


def test_rms_point_error_is_zero_for_identical_points():
    points = source_landmarks()

    assert rms_point_error(
        points,
        points,
    ) == pytest.approx(
        0.0
    )


def test_rms_point_error_matches_known_translation():
    first = source_landmarks()

    displacement = np.asarray(
        [
            0.003,
            -0.004,
            0.0,
        ]
    )

    second = (
        first
        + displacement
    )

    error = rms_point_error(
        first,
        second,
    )

    assert error == pytest.approx(
        0.005
    )


def test_transform_error_is_zero_for_identical_transforms():
    transform = reference_transform()

    error = transform_error(
        transform,
        transform,
    )

    assert (
        error.translation_error
        == pytest.approx(
            0.0
        )
    )

    assert (
        error.rotation_angle_error_degrees
        == pytest.approx(
            0.0,
            abs=1e-6,
        )
    )


def test_rotation_angle_between_matches_known_rotation():
    identity = np.eye(
        3
    )

    rotated = rotation_z(
        np.deg2rad(
            25.0
        )
    )

    angle = rotation_angle_between(
        identity,
        rotated,
    )

    assert np.degrees(
        angle
    ) == pytest.approx(
        25.0
    )


def test_target_registration_error_is_zero_for_exact_transform():
    reference = reference_transform()

    evaluation_targets = np.asarray(
        [
            [
                0.02,
                0.01,
                0.05,
            ],
            [
                -0.03,
                0.02,
                0.08,
            ],
            [
                0.01,
                -0.04,
                0.10,
            ],
        ],
        dtype=float,
    )

    errors = registration_error_at_targets(
        reference,
        reference,
        evaluation_targets,
    )

    np.testing.assert_allclose(
        errors,
        np.zeros(
            3
        ),
        atol=1e-12,
    )


def test_registration_rejects_mismatched_correspondence_count():
    source = source_landmarks()

    target = source[
        :-1
    ]

    with pytest.raises(
        ValueError,
        match="identical shapes",
    ):
        register_corresponding_points(
            source,
            target,
        )


def test_registration_rejects_too_few_landmarks():
    source = np.asarray(
        [
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                1.0,
                0.0,
                0.0,
            ],
        ]
    )

    with pytest.raises(
        ValueError,
        match="at least 3",
    ):
        register_corresponding_points(
            source,
            source,
        )


def test_registration_rejects_collinear_geometry():
    source = np.asarray(
        [
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                1.0,
                0.0,
                0.0,
            ],
            [
                2.0,
                0.0,
                0.0,
            ],
            [
                3.0,
                0.0,
                0.0,
            ],
        ],
        dtype=float,
    )

    target = source.copy()

    with pytest.raises(
        ValueError,
        match="degenerate",
    ):
        register_corresponding_points(
            source,
            target,
        )


def test_registration_never_returns_improper_reflection():
    source = source_landmarks()

    # Reflecting correspondences cannot be represented exactly by a
    # proper rigid-body transformation.
    reflected_target = (
        source.copy()
    )

    reflected_target[
        :,
        0
    ] *= -1.0

    result = register_corresponding_points(
        source,
        reflected_target,
    )

    determinant = float(
        np.linalg.det(
            result.rotation
        )
    )

    assert determinant == pytest.approx(
        1.0,
        abs=1e-10,
    )

    # Because the target relation is a reflection rather than a proper
    # rotation, residual error should remain instead of returning det=-1.
    assert result.fre_rms > 1e-6