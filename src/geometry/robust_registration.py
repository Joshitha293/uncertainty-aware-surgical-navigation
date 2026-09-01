"""Robust rigid registration and iterative closest point alignment.

Phase 5 extends corresponding-landmark registration with two capabilities:

1. RANSAC rigid registration
   Handles known landmark correspondences containing gross outliers.

2. Trimmed iterative closest point (ICP)
   Estimates alignment when source-to-target point correspondences are not
   known in advance.

Both methods reuse the project's existing rigid transformation and Kabsch
registration infrastructure.

These algorithms are evaluated using simulated engineering data and do not
constitute clinical patient registration validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry.registration import (
    RigidRegistrationResult,
    register_corresponding_points,
    transform_points,
)
from src.geometry.transforms import (
    compose_transforms,
    is_homogeneous_transform,
    make_transform,
)


@dataclass(frozen=True)
class RansacRegistrationResult:
    """Robust corresponding-point registration result."""

    registration: RigidRegistrationResult

    inlier_mask: np.ndarray

    inlier_count: int

    inlier_fraction: float

    iterations: int


@dataclass(frozen=True)
class ICPResult:
    """Result of iterative closest point registration."""

    transform: np.ndarray

    transformed_source_points: np.ndarray

    correspondence_indices: np.ndarray

    correspondence_distances: np.ndarray

    rms_error: float

    iterations: int

    converged: bool

    used_correspondence_count: int

    rms_history: np.ndarray


def _validate_points(
    points: np.ndarray,
    *,
    name: str,
    minimum_points: int = 3,
) -> np.ndarray:
    """Validate an N x 3 point cloud."""

    points = np.asarray(
        points,
        dtype=float,
    )

    if (
        points.ndim != 2
        or points.shape[1] != 3
    ):
        raise ValueError(
            f"{name} must have shape (N, 3)."
        )

    if points.shape[0] < minimum_points:
        raise ValueError(
            f"{name} must contain at least {minimum_points} points."
        )

    if not np.all(
        np.isfinite(
            points
        )
    ):
        raise ValueError(
            f"{name} must contain finite values."
        )

    return points


def nearest_neighbour_correspondences(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Find nearest target point for each source point.

    Returns
    -------
    indices:
        Target-point index assigned to every source point.

    distances:
        Euclidean source-to-nearest-target distances.
    """

    source = _validate_points(
        source_points,
        name="source_points",
    )

    target = _validate_points(
        target_points,
        name="target_points",
    )

    differences = (
        source[
            :,
            None,
            :
        ]
        - target[
            None,
            :,
            :
        ]
    )

    squared_distances = np.sum(
        differences**2,
        axis=2,
    )

    indices = np.argmin(
        squared_distances,
        axis=1,
    )

    minimum_squared = squared_distances[
        np.arange(
            source.shape[
                0
            ]
        ),
        indices,
    ]

    distances = np.sqrt(
        minimum_squared
    )

    return (
        np.asarray(
            indices,
            dtype=int,
        ),
        np.asarray(
            distances,
            dtype=float,
        ),
    )


def ransac_rigid_registration(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    iterations: int = 250,
    inlier_threshold: float = 0.003,
    seed: int = 5201,
) -> RansacRegistrationResult:
    """Robustly register corresponding landmarks using RANSAC.

    Source and target points retain known correspondence by row, but some
    correspondences may be gross outliers.
    """

    source = _validate_points(
        source_points,
        name="source_points",
    )

    target = _validate_points(
        target_points,
        name="target_points",
    )

    if source.shape != target.shape:
        raise ValueError(
            "source_points and target_points must have identical shapes."
        )

    if iterations < 1:
        raise ValueError(
            "iterations must be positive."
        )

    if (
        not np.isfinite(
            inlier_threshold
        )
        or inlier_threshold <= 0.0
    ):
        raise ValueError(
            "inlier_threshold must be finite and positive."
        )

    rng = np.random.default_rng(
        seed
    )

    point_count = source.shape[
        0
    ]

    best_mask = None

    best_count = -1

    best_mean_error = float(
        "inf"
    )

    for _ in range(
        iterations
    ):
        sample_indices = rng.choice(
            point_count,
            size=3,
            replace=False,
        )

        try:
            candidate = (
                register_corresponding_points(
                    source[
                        sample_indices
                    ],
                    target[
                        sample_indices
                    ],
                )
            )

        except ValueError:
            continue

        transformed = transform_points(
            candidate.transform,
            source,
        )

        residuals = np.linalg.norm(
            transformed
            - target,
            axis=1,
        )

        inlier_mask = (
            residuals
            <= inlier_threshold
        )

        inlier_count = int(
            np.count_nonzero(
                inlier_mask
            )
        )

        if inlier_count < 3:
            continue

        mean_error = float(
            np.mean(
                residuals[
                    inlier_mask
                ]
            )
        )

        better = (
            inlier_count
            > best_count
        )

        tie_but_lower_error = (
            inlier_count
            == best_count
            and mean_error
            < best_mean_error
        )

        if (
            better
            or tie_but_lower_error
        ):
            best_mask = (
                inlier_mask.copy()
            )

            best_count = (
                inlier_count
            )

            best_mean_error = (
                mean_error
            )

    if (
        best_mask is None
        or best_count < 3
    ):
        raise RuntimeError(
            "RANSAC could not identify a valid rigid-registration consensus."
        )

    refined = (
        register_corresponding_points(
            source[
                best_mask
            ],
            target[
                best_mask
            ],
        )
    )

    transformed_all = transform_points(
        refined.transform,
        source,
    )

    final_residuals = np.linalg.norm(
        transformed_all
        - target,
        axis=1,
    )

    final_mask = (
        final_residuals
        <= inlier_threshold
    )

    final_count = int(
        np.count_nonzero(
            final_mask
        )
    )

    if final_count >= 3:
        refined = (
            register_corresponding_points(
                source[
                    final_mask
                ],
                target[
                    final_mask
                ],
            )
        )

        best_mask = (
            final_mask
        )

        best_count = (
            final_count
        )

    return RansacRegistrationResult(
        registration=refined,
        inlier_mask=np.asarray(
            best_mask,
            dtype=bool,
        ),
        inlier_count=int(
            best_count
        ),
        inlier_fraction=float(
            best_count
            / point_count
        ),
        iterations=int(
            iterations
        ),
    )


def _select_trimmed_correspondences(
    distances: np.ndarray,
    *,
    trim_fraction: float,
    maximum_distance: float | None,
) -> np.ndarray:
    """Select the best subset of nearest-neighbour correspondences."""

    distances = np.asarray(
        distances,
        dtype=float,
    )

    if (
        distances.ndim != 1
        or distances.size < 3
    ):
        raise ValueError(
            "distances must contain at least three values."
        )

    if not (
        0.0
        < trim_fraction
        <= 1.0
    ):
        raise ValueError(
            "trim_fraction must lie in (0, 1]."
        )

    eligible = np.arange(
        distances.size,
        dtype=int,
    )

    if maximum_distance is not None:
        if (
            not np.isfinite(
                maximum_distance
            )
            or maximum_distance <= 0.0
        ):
            raise ValueError(
                "maximum_distance must be finite and positive."
            )

        eligible = eligible[
            distances[
                eligible
            ]
            <= maximum_distance
        ]

    if eligible.size < 3:
        raise RuntimeError(
            "Fewer than three usable ICP correspondences remain."
        )

    ordered = eligible[
        np.argsort(
            distances[
                eligible
            ]
        )
    ]

    keep_count = int(
        np.ceil(
            trim_fraction
            * ordered.size
        )
    )

    keep_count = max(
        3,
        keep_count,
    )

    keep_count = min(
        keep_count,
        ordered.size,
    )

    return ordered[
        :keep_count
    ]


def iterative_closest_point(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    initial_transform: np.ndarray | None = None,
    max_iterations: int = 60,
    tolerance: float = 1e-8,
    trim_fraction: float = 0.90,
    maximum_correspondence_distance: float | None = None,
) -> ICPResult:
    """Register point clouds with unknown correspondence using trimmed ICP.

    The algorithm repeatedly:

    1. transforms the source cloud;
    2. assigns nearest target neighbours;
    3. removes the worst correspondence residuals;
    4. estimates an incremental rigid transformation;
    5. composes the increment with the accumulated transformation.

    ICP is a local optimiser. A sufficiently reasonable initial alignment is
    required for difficult geometries.
    """

    source = _validate_points(
        source_points,
        name="source_points",
    )

    target = _validate_points(
        target_points,
        name="target_points",
    )

    if max_iterations < 1:
        raise ValueError(
            "max_iterations must be positive."
        )

    if (
        not np.isfinite(
            tolerance
        )
        or tolerance < 0.0
    ):
        raise ValueError(
            "tolerance must be finite and non-negative."
        )

    if not (
        0.0
        < trim_fraction
        <= 1.0
    ):
        raise ValueError(
            "trim_fraction must lie in (0, 1]."
        )

    if initial_transform is None:
        current_transform = make_transform(
            rotation=np.eye(
                3,
                dtype=float,
            ),
            translation=np.zeros(
                3,
                dtype=float,
            ),
        )

    else:
        initial = np.asarray(
            initial_transform,
            dtype=float,
        )

        if not is_homogeneous_transform(
            initial
        ):
            raise ValueError(
                "initial_transform must be a valid rigid transformation."
            )

        current_transform = (
            initial.copy()
        )

    previous_rms = None

    history: list[
        float
    ] = []

    converged = False

    used_count = 0

    completed_iterations = 0

    for iteration in range(
        1,
        max_iterations + 1,
    ):
        transformed = transform_points(
            current_transform,
            source,
        )

        (
            correspondence_indices,
            correspondence_distances,
        ) = nearest_neighbour_correspondences(
            transformed,
            target,
        )

        selected = (
            _select_trimmed_correspondences(
                correspondence_distances,
                trim_fraction=(
                    trim_fraction
                ),
                maximum_distance=(
                    maximum_correspondence_distance
                ),
            )
        )

        used_count = int(
            selected.size
        )

        matched_target = target[
            correspondence_indices[
                selected
            ]
        ]

        increment = (
            register_corresponding_points(
                transformed[
                    selected
                ],
                matched_target,
            )
        )

        current_transform = (
            compose_transforms(
                increment.transform,
                current_transform,
            )
        )

        transformed_updated = (
            transform_points(
                current_transform,
                source,
            )
        )

        (
            updated_indices,
            updated_distances,
        ) = nearest_neighbour_correspondences(
            transformed_updated,
            target,
        )

        updated_selected = (
            _select_trimmed_correspondences(
                updated_distances,
                trim_fraction=(
                    trim_fraction
                ),
                maximum_distance=(
                    maximum_correspondence_distance
                ),
            )
        )

        rms = float(
            np.sqrt(
                np.mean(
                    updated_distances[
                        updated_selected
                    ]
                    ** 2
                )
            )
        )

        history.append(
            rms
        )

        completed_iterations = (
            iteration
        )

        if previous_rms is not None:
            improvement = abs(
                previous_rms
                - rms
            )

            if improvement <= tolerance:
                converged = True
                break

        previous_rms = (
            rms
        )

    final_transformed = transform_points(
        current_transform,
        source,
    )

    (
        final_indices,
        final_distances,
    ) = nearest_neighbour_correspondences(
        final_transformed,
        target,
    )

    final_selected = (
        _select_trimmed_correspondences(
            final_distances,
            trim_fraction=(
                trim_fraction
            ),
            maximum_distance=(
                maximum_correspondence_distance
            ),
        )
    )

    final_rms = float(
        np.sqrt(
            np.mean(
                final_distances[
                    final_selected
                ]
                ** 2
            )
        )
    )

    return ICPResult(
        transform=np.asarray(
            current_transform,
            dtype=float,
        ),
        transformed_source_points=np.asarray(
            final_transformed,
            dtype=float,
        ),
        correspondence_indices=np.asarray(
            final_indices,
            dtype=int,
        ),
        correspondence_distances=np.asarray(
            final_distances,
            dtype=float,
        ),
        rms_error=final_rms,
        iterations=int(
            completed_iterations
        ),
        converged=bool(
            converged
        ),
        used_correspondence_count=int(
            final_selected.size
        ),
        rms_history=np.asarray(
            history,
            dtype=float,
        ),
    )