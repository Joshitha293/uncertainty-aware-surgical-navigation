"""Tests for Phase 5 robust registration and ICP."""

import numpy as np
import pytest

from src.geometry.registration import (
    transform_error,
    transform_points,
)
from src.geometry.robust_registration import (
    iterative_closest_point,
    nearest_neighbour_correspondences,
    ransac_rigid_registration,
)
from src.geometry.transforms import (
    is_homogeneous_transform,
    make_transform,
)


def rotation_xyz(
    ax: float,
    ay: float,
    az: float,
) -> np.ndarray:
    """Construct XYZ rigid rotation."""

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
            [1.0, 0.0, 0.0],
            [0.0, cx, -sx],
            [0.0, sx, cx],
        ],
        dtype=float,
    )

    ry = np.asarray(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ],
        dtype=float,
    )

    rz = np.asarray(
        [
            [cz, -sz, 0.0],
            [sz, cz, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    return (
        rz
        @ ry
        @ rx
    )


def reference_transform() -> np.ndarray:
    return make_transform(
        rotation=rotation_xyz(
            np.deg2rad(
                4.0
            ),
            np.deg2rad(
                -5.0
            ),
            np.deg2rad(
                6.0
            ),
        ),
        translation=np.asarray(
            [
                0.004,
                -0.003,
                0.005,
            ],
            dtype=float,
        ),
    )


def make_point_cloud(
    *,
    count: int = 40,
    seed: int = 5211,
) -> np.ndarray:
    """Generate deterministic asymmetric 3-D point cloud."""

    rng = np.random.default_rng(
        seed
    )

    points = rng.uniform(
        low=np.asarray(
            [
                -0.050,
                -0.040,
                -0.030,
            ]
        ),
        high=np.asarray(
            [
                0.060,
                0.050,
                0.045,
            ]
        ),
        size=(
            count,
            3,
        ),
    )

    # Introduce asymmetric geometry to avoid ambiguous alignment.
    points[
        :,
        2
    ] += (
        0.25
        * points[
            :,
            0
        ]
        + 0.10
        * points[
            :,
            1
        ]
    )

    return points


def test_nearest_neighbour_finds_exact_permuted_points():
    source = make_point_cloud(
        count=10
    )

    permutation = np.asarray(
        [
            5,
            2,
            8,
            0,
            9,
            1,
            7,
            3,
            6,
            4,
        ]
    )

    target = source[
        permutation
    ]

    indices, distances = (
        nearest_neighbour_correspondences(
            source,
            target,
        )
    )

    recovered = target[
        indices
    ]

    np.testing.assert_allclose(
        recovered,
        source,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        distances,
        np.zeros(
            10
        ),
        atol=1e-12,
    )


def test_nearest_neighbour_returns_expected_shapes():
    source = make_point_cloud(
        count=12
    )

    target = make_point_cloud(
        count=18,
        seed=99,
    )

    indices, distances = (
        nearest_neighbour_correspondences(
            source,
            target,
        )
    )

    assert indices.shape == (
        12,
    )

    assert distances.shape == (
        12,
    )

    assert np.all(
        distances
        >= 0.0
    )


def test_ransac_exact_correspondences_recovers_transform():
    source = make_point_cloud(
        count=12
    )

    reference = (
        reference_transform()
    )

    target = transform_points(
        reference,
        source,
    )

    result = (
        ransac_rigid_registration(
            source,
            target,
            iterations=100,
            inlier_threshold=1e-5,
            seed=1,
        )
    )

    error = transform_error(
        result
        .registration
        .transform,
        reference,
    )

    assert (
        error.translation_error
        < 1e-8
    )

    assert (
        error.rotation_angle_error_degrees
        < 1e-5
    )


def test_ransac_rejects_gross_correspondence_outliers():
    source = make_point_cloud(
        count=14
    )

    reference = (
        reference_transform()
    )

    rng = np.random.default_rng(
        5221
    )

    target = (
        transform_points(
            reference,
            source,
        )
        + rng.normal(
            0.0,
            0.0003,
            size=source.shape,
        )
    )

    target[
        2
    ] += np.asarray(
        [
            0.035,
            -0.025,
            0.030,
        ]
    )

    target[
        9
    ] += np.asarray(
        [
            -0.030,
            0.040,
            -0.025,
        ]
    )

    result = (
        ransac_rigid_registration(
            source,
            target,
            iterations=300,
            inlier_threshold=0.002,
            seed=41,
        )
    )

    assert (
        result.inlier_count
        >= 12
    )

    assert not result.inlier_mask[
        2
    ]

    assert not result.inlier_mask[
        9
    ]

    error = transform_error(
        result
        .registration
        .transform,
        reference,
    )

    assert (
        error.translation_error
        < 0.002
    )

    assert (
        error.rotation_angle_error_degrees
        < 1.5
    )


def test_ransac_result_is_reproducible():
    source = make_point_cloud(
        count=12
    )

    target = transform_points(
        reference_transform(),
        source,
    )

    first = (
        ransac_rigid_registration(
            source,
            target,
            iterations=50,
            seed=123,
        )
    )

    second = (
        ransac_rigid_registration(
            source,
            target,
            iterations=50,
            seed=123,
        )
    )

    np.testing.assert_allclose(
        first.registration.transform,
        second.registration.transform,
        atol=1e-12,
    )


def test_ransac_rejects_invalid_threshold():
    source = make_point_cloud(
        count=10
    )

    with pytest.raises(
        ValueError,
        match="inlier_threshold",
    ):
        ransac_rigid_registration(
            source,
            source,
            inlier_threshold=0.0,
        )


def test_icp_identity_alignment_has_near_zero_error():
    source = make_point_cloud(
        count=20
    )

    target = source[
        ::-1
    ].copy()

    result = (
        iterative_closest_point(
            source,
            target,
            max_iterations=20,
            trim_fraction=1.0,
        )
    )

    assert (
        result.rms_error
        < 1e-10
    )

    np.testing.assert_allclose(
        result.transform,
        np.eye(
            4
        ),
        atol=1e-8,
    )


def test_icp_recovers_small_unknown_alignment():
    source = make_point_cloud(
        count=50
    )

    reference = (
        reference_transform()
    )

    target = transform_points(
        reference,
        source,
    )

    rng = np.random.default_rng(
        55
    )

    target = (
        target
        + rng.normal(
            0.0,
            0.00015,
            size=target.shape,
        )
    )

    target = target[
        rng.permutation(
            target.shape[
                0
            ]
        )
    ]

    result = (
        iterative_closest_point(
            source,
            target,
            max_iterations=80,
            tolerance=1e-10,
            trim_fraction=1.0,
        )
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
        < 1.5
    )

    assert (
        result.rms_error
        < 0.002
    )


def test_icp_accepts_useful_initial_transform():
    source = make_point_cloud(
        count=45
    )

    reference = make_transform(
        rotation=rotation_xyz(
            np.deg2rad(
                8.0
            ),
            np.deg2rad(
                -7.0
            ),
            np.deg2rad(
                10.0
            ),
        ),
        translation=np.asarray(
            [
                0.010,
                -0.007,
                0.009,
            ]
        ),
    )

    target = transform_points(
        reference,
        source,
    )

    initial = make_transform(
        rotation=rotation_xyz(
            np.deg2rad(
                6.5
            ),
            np.deg2rad(
                -6.0
            ),
            np.deg2rad(
                8.0
            ),
        ),
        translation=np.asarray(
            [
                0.008,
                -0.006,
                0.007,
            ]
        ),
    )

    result = (
        iterative_closest_point(
            source,
            target,
            initial_transform=initial,
            max_iterations=60,
            tolerance=1e-10,
            trim_fraction=1.0,
        )
    )

    error = transform_error(
        result.transform,
        reference,
    )

    assert (
        error.translation_error
        < 1e-4
    )

    assert (
        error.rotation_angle_error_degrees
        < 0.1
    )


def test_icp_result_transform_is_valid():
    source = make_point_cloud(
        count=25
    )

    target = transform_points(
        reference_transform(),
        source,
    )

    result = (
        iterative_closest_point(
            source,
            target,
        )
    )

    assert is_homogeneous_transform(
        result.transform
    )


def test_icp_history_contains_finite_nonnegative_errors():
    source = make_point_cloud(
        count=30
    )

    target = transform_points(
        reference_transform(),
        source,
    )

    result = (
        iterative_closest_point(
            source,
            target,
        )
    )

    assert (
        result.rms_history.size
        >= 1
    )

    assert np.all(
        np.isfinite(
            result.rms_history
        )
    )

    assert np.all(
        result.rms_history
        >= 0.0
    )


def test_icp_trimmed_alignment_handles_extra_source_outliers():
    clean_source = make_point_cloud(
        count=40
    )

    reference = (
        reference_transform()
    )

    target = transform_points(
        reference,
        clean_source,
    )

    source_outliers = np.asarray(
        [
            [
                0.20,
                0.20,
                0.20,
            ],
            [
                -0.20,
                0.18,
                -0.15,
            ],
        ]
    )

    source = np.vstack(
        (
            clean_source,
            source_outliers,
        )
    )

    result = (
        iterative_closest_point(
            source,
            target,
            max_iterations=80,
            tolerance=1e-9,
            trim_fraction=0.90,
        )
    )

    error = transform_error(
        result.transform,
        reference,
    )

    assert (
        error.translation_error
        < 0.003
    )

    assert (
        error.rotation_angle_error_degrees
        < 2.0
    )

    assert (
        result.used_correspondence_count
        < source.shape[
            0
        ]
    )


def test_icp_rejects_invalid_trim_fraction():
    source = make_point_cloud(
        count=10
    )

    with pytest.raises(
        ValueError,
        match="trim_fraction",
    ):
        iterative_closest_point(
            source,
            source,
            trim_fraction=0.0,
        )


def test_icp_rejects_invalid_initial_transform():
    source = make_point_cloud(
        count=10
    )

    invalid = np.eye(
        4
    )

    invalid[
        0,
        0
    ] = 2.0

    with pytest.raises(
        ValueError,
        match="initial_transform",
    ):
        iterative_closest_point(
            source,
            source,
            initial_transform=invalid,
        )


def test_icp_rejects_too_small_correspondence_gate():
    source = make_point_cloud(
        count=10
    )

    target = (
        source
        + 1.0
    )

    with pytest.raises(
        RuntimeError,
        match="Fewer than three",
    ):
        iterative_closest_point(
            source,
            target,
            maximum_correspondence_distance=(
                1e-6
            ),
        )