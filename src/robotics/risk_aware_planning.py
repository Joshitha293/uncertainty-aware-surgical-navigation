"""Risk-aware planning primitives for Phase 6.

This module introduces a probabilistic clearance model for uncertainty-aware
motion planning.

Earlier project phases primarily represented positional uncertainty through
scalar inflation of protected spherical geometry. That is useful and
conservative, but it discards covariance orientation.

Phase 6 adds a first-order chance-constrained clearance model.

For a deterministic query point p and an uncertain spherical-structure centre

    X ~ N(mu, Sigma),

the signed mean clearance is

    c_bar = ||p - mu|| - r_protected.

The clearance is linearised with respect to centre uncertainty. The covariance
is projected onto the local obstacle-clearance direction n:

    sigma_c^2 = n^T Sigma n.

The probability of violating the protected radius is then approximated as

    P(c <= 0) ~= Phi(-c_bar / sigma_c).

This is a local Gaussian approximation. It is not an exact probability for an
anisotropic Gaussian integrated over a sphere.

Important interpretation boundaries
-----------------------------------

* The probability is an engineering model, not a patient-risk probability.
* The model assumes Gaussian positional uncertainty locally.
* The first-order approximation becomes unreliable close to the uncertain
  sphere centre, where the distance gradient is undefined. That case is
  handled conservatively.
* Ground-truth geometry is not accepted by the runtime risk functions.
* A path-level union bound is exposed as a conservative diagnostic. Because
  neighbouring path samples are correlated, it must not be interpreted as an
  exact path collision probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from math import erfc
from math import sqrt
from typing import Sequence

import numpy as np


_VECTOR_DIMENSION = 3
_DEFAULT_NUMERICAL_TOLERANCE = 1e-12


def _as_vector3(
    value: np.ndarray | Sequence[float],
    *,
    name: str,
) -> np.ndarray:
    """Validate and copy a finite 3-D vector."""

    vector = np.asarray(
        value,
        dtype=float,
    )

    if vector.shape != (_VECTOR_DIMENSION,):
        raise ValueError(
            f"{name} must have shape (3,)."
        )

    if not np.all(
        np.isfinite(vector)
    ):
        raise ValueError(
            f"{name} must contain only finite values."
        )

    return np.array(
        vector,
        dtype=float,
        copy=True,
    )


def _as_covariance3(
    value: np.ndarray,
    *,
    name: str,
    psd_tolerance: float = _DEFAULT_NUMERICAL_TOLERANCE,
) -> np.ndarray:
    """Validate and copy a finite symmetric positive-semidefinite covariance."""

    covariance = np.asarray(
        value,
        dtype=float,
    )

    if covariance.shape != (
        _VECTOR_DIMENSION,
        _VECTOR_DIMENSION,
    ):
        raise ValueError(
            f"{name} must have shape (3, 3)."
        )

    if not np.all(
        np.isfinite(covariance)
    ):
        raise ValueError(
            f"{name} must contain only finite values."
        )

    if not np.allclose(
        covariance,
        covariance.T,
        atol=psd_tolerance,
        rtol=1e-10,
    ):
        raise ValueError(
            f"{name} must be symmetric."
        )

    symmetric = 0.5 * (
        covariance
        + covariance.T
    )

    eigenvalues = np.linalg.eigvalsh(
        symmetric
    )

    if float(
        np.min(eigenvalues)
    ) < -float(
        psd_tolerance
    ):
        raise ValueError(
            f"{name} must be positive semidefinite."
        )

    return np.array(
        symmetric,
        dtype=float,
        copy=True,
    )


def _validate_nonnegative_finite(
    value: float,
    *,
    name: str,
) -> float:
    """Validate a finite scalar that must be non-negative."""

    result = float(
        value
    )

    if (
        not np.isfinite(result)
        or result < 0.0
    ):
        raise ValueError(
            f"{name} must be finite and non-negative."
        )

    return result


def _validate_probability(
    value: float,
    *,
    name: str,
) -> float:
    """Validate a probability threshold in the closed interval [0, 1]."""

    result = float(
        value
    )

    if (
        not np.isfinite(result)
        or result < 0.0
        or result > 1.0
    ):
        raise ValueError(
            f"{name} must lie in [0, 1]."
        )

    return result


@dataclass(frozen=True)
class ChanceConstraintConfig:
    """Configuration for probabilistic clearance evaluation.

    ``max_point_violation_probability`` is the primary pointwise chance
    constraint.

    ``max_path_union_bound_probability`` is optional. When supplied, the
    conservative union bound over all sampled point/structure events must also
    satisfy this limit.

    The union bound is intentionally optional because its numerical value
    depends on path discretisation and because neighbouring samples are not
    statistically independent.
    """

    max_point_violation_probability: float = 0.05

    max_path_union_bound_probability: float | None = None

    sample_spacing: float | None = None

    sigma_epsilon: float = 1e-12

    geometry_epsilon: float = 1e-12

    def __post_init__(
        self,
    ) -> None:
        object.__setattr__(
            self,
            "max_point_violation_probability",
            _validate_probability(
                self.max_point_violation_probability,
                name="max_point_violation_probability",
            ),
        )

        if (
            self.max_path_union_bound_probability
            is not None
        ):
            object.__setattr__(
                self,
                "max_path_union_bound_probability",
                _validate_probability(
                    self.max_path_union_bound_probability,
                    name="max_path_union_bound_probability",
                ),
            )

        if self.sample_spacing is not None:
            spacing = float(
                self.sample_spacing
            )

            if (
                not np.isfinite(spacing)
                or spacing <= 0.0
            ):
                raise ValueError(
                    "sample_spacing must be finite and positive."
                )

            object.__setattr__(
                self,
                "sample_spacing",
                spacing,
            )

        sigma_epsilon = _validate_nonnegative_finite(
            self.sigma_epsilon,
            name="sigma_epsilon",
        )

        geometry_epsilon = _validate_nonnegative_finite(
            self.geometry_epsilon,
            name="geometry_epsilon",
        )

        object.__setattr__(
            self,
            "sigma_epsilon",
            sigma_epsilon,
        )

        object.__setattr__(
            self,
            "geometry_epsilon",
            geometry_epsilon,
        )


@dataclass(frozen=True)
class UncertainSphere:
    """Estimated spherical structure used by the Phase 6 risk model.

    Parameters
    ----------
    center_mean:
        Estimated 3-D centre of the structure.

    center_covariance:
        Estimated 3 x 3 covariance of the structure centre.

    physical_radius:
        Physical simulated radius of the structure.

    safety_margin:
        Additional deterministic protected margin.

    label:
        Optional identifier used only for diagnostics.
    """

    center_mean: np.ndarray

    center_covariance: np.ndarray

    physical_radius: float

    safety_margin: float = 0.0

    label: str = ""

    def __post_init__(
        self,
    ) -> None:
        object.__setattr__(
            self,
            "center_mean",
            _as_vector3(
                self.center_mean,
                name="center_mean",
            ),
        )

        object.__setattr__(
            self,
            "center_covariance",
            _as_covariance3(
                self.center_covariance,
                name="center_covariance",
            ),
        )

        object.__setattr__(
            self,
            "physical_radius",
            _validate_nonnegative_finite(
                self.physical_radius,
                name="physical_radius",
            ),
        )

        object.__setattr__(
            self,
            "safety_margin",
            _validate_nonnegative_finite(
                self.safety_margin,
                name="safety_margin",
            ),
        )

        object.__setattr__(
            self,
            "label",
            str(
                self.label
            ),
        )

    def protected_radius(
        self,
        instrument_radius: float = 0.0,
    ) -> float:
        """Return deterministic protected centre-to-centre radius."""

        radius = _validate_nonnegative_finite(
            instrument_radius,
            name="instrument_radius",
        )

        return float(
            self.physical_radius
            + self.safety_margin
            + radius
        )


@dataclass(frozen=True)
class ClearanceRiskEstimate:
    """First-order risk estimate for one point against one structure."""

    query_point: np.ndarray

    structure_center_mean: np.ndarray

    protected_radius: float

    centre_distance: float

    mean_clearance: float

    radial_direction: np.ndarray

    directional_variance: float

    directional_sigma: float

    violation_probability: float

    linearisation_valid: bool

    deterministic_limit_used: bool


@dataclass(frozen=True)
class PathRiskEvent:
    """Risk estimate associated with one path-point/structure pair."""

    point_index: int

    structure_index: int

    structure_label: str

    estimate: ClearanceRiskEstimate


@dataclass(frozen=True)
class PathRiskEstimate:
    """Aggregated chance-constraint diagnostics for a sampled path."""

    sampled_points: np.ndarray

    events: tuple[
        PathRiskEvent,
        ...,
    ]

    maximum_point_violation_probability: float

    union_bound_violation_probability: float

    worst_event_index: int | None

    accepted: bool

    @property
    def worst_event(
        self,
    ) -> PathRiskEvent | None:
        """Return the event with the greatest estimated violation risk."""

        if self.worst_event_index is None:
            return None

        return self.events[
            self.worst_event_index
        ]


def gaussian_clearance_violation_probability(
    mean_clearance: float,
    directional_sigma: float,
    *,
    sigma_epsilon: float = 1e-12,
) -> float:
    """Return ``P(clearance <= 0)`` under a 1-D Gaussian approximation.

    When uncertainty is numerically zero, the deterministic geometric limit is
    used:

    * positive clearance -> probability 0;
    * zero or negative clearance -> probability 1.
    """

    clearance = float(
        mean_clearance
    )

    sigma = _validate_nonnegative_finite(
        directional_sigma,
        name="directional_sigma",
    )

    epsilon = _validate_nonnegative_finite(
        sigma_epsilon,
        name="sigma_epsilon",
    )

    if not np.isfinite(
        clearance
    ):
        raise ValueError(
            "mean_clearance must be finite."
        )

    if sigma <= epsilon:
        return (
            1.0
            if clearance <= 0.0
            else 0.0
        )

    probability = 0.5 * erfc(
        clearance
        / (
            sigma
            * sqrt(2.0)
        )
    )

    return float(
        np.clip(
            probability,
            0.0,
            1.0,
        )
    )


def estimate_clearance_risk(
    query_point: np.ndarray | Sequence[float],
    structure: UncertainSphere,
    *,
    instrument_radius: float = 0.0,
    config: ChanceConstraintConfig | None = None,
) -> ClearanceRiskEstimate:
    """Estimate protected-radius violation risk at one 3-D point.

    The structure-centre covariance is projected onto the local radial
    clearance direction.

    This preserves covariance orientation, unlike a scalar maximum-eigenvalue
    inflation rule.
    """

    if config is None:
        config = ChanceConstraintConfig()

    point = _as_vector3(
        query_point,
        name="query_point",
    )

    instrument_radius_value = (
        _validate_nonnegative_finite(
            instrument_radius,
            name="instrument_radius",
        )
    )

    protected_radius = (
        structure.protected_radius(
            instrument_radius_value
        )
    )

    displacement = (
        point
        - structure.center_mean
    )

    centre_distance = float(
        np.linalg.norm(
            displacement
        )
    )

    mean_clearance = float(
        centre_distance
        - protected_radius
    )

    if (
        centre_distance
        <= config.geometry_epsilon
    ):
        # The gradient of Euclidean distance is undefined at the sphere
        # centre. Returning a conservative violation probability prevents the
        # invalid linearisation from creating a false sense of safety.
        maximum_variance = float(
            max(
                0.0,
                np.max(
                    np.linalg.eigvalsh(
                        structure.center_covariance
                    )
                ),
            )
        )

        sigma = float(
            sqrt(
                maximum_variance
            )
        )

        return ClearanceRiskEstimate(
            query_point=point,
            structure_center_mean=np.array(
                structure.center_mean,
                dtype=float,
                copy=True,
            ),
            protected_radius=float(
                protected_radius
            ),
            centre_distance=centre_distance,
            mean_clearance=mean_clearance,
            radial_direction=np.zeros(
                3,
                dtype=float,
            ),
            directional_variance=maximum_variance,
            directional_sigma=sigma,
            violation_probability=1.0,
            linearisation_valid=False,
            deterministic_limit_used=(
                sigma
                <= config.sigma_epsilon
            ),
        )

    radial_direction = (
        displacement
        / centre_distance
    )

    directional_variance = float(
        radial_direction
        @ structure.center_covariance
        @ radial_direction
    )

    directional_variance = max(
        0.0,
        directional_variance,
    )

    directional_sigma = float(
        sqrt(
            directional_variance
        )
    )

    probability = (
        gaussian_clearance_violation_probability(
            mean_clearance,
            directional_sigma,
            sigma_epsilon=(
                config.sigma_epsilon
            ),
        )
    )

    return ClearanceRiskEstimate(
        query_point=point,
        structure_center_mean=np.array(
            structure.center_mean,
            dtype=float,
            copy=True,
        ),
        protected_radius=float(
            protected_radius
        ),
        centre_distance=centre_distance,
        mean_clearance=mean_clearance,
        radial_direction=np.array(
            radial_direction,
            dtype=float,
            copy=True,
        ),
        directional_variance=(
            directional_variance
        ),
        directional_sigma=(
            directional_sigma
        ),
        violation_probability=(
            probability
        ),
        linearisation_valid=True,
        deterministic_limit_used=(
            directional_sigma
            <= config.sigma_epsilon
        ),
    )


def sample_polyline(
    path_points: np.ndarray,
    max_spacing: float,
) -> np.ndarray:
    """Densely sample a 3-D polyline.

    Consecutive returned samples are separated by no more than
    ``max_spacing`` up to floating-point tolerance.
    """

    path = np.asarray(
        path_points,
        dtype=float,
    )

    if (
        path.ndim != 2
        or path.shape[1] != 3
        or path.shape[0] < 1
    ):
        raise ValueError(
            "path_points must have shape (N, 3) with N >= 1."
        )

    if not np.all(
        np.isfinite(path)
    ):
        raise ValueError(
            "path_points must contain only finite values."
        )

    spacing = float(
        max_spacing
    )

    if (
        not np.isfinite(spacing)
        or spacing <= 0.0
    ):
        raise ValueError(
            "max_spacing must be finite and positive."
        )

    if path.shape[0] == 1:
        return np.array(
            path,
            dtype=float,
            copy=True,
        )

    samples: list[
        np.ndarray
    ] = [
        np.array(
            path[0],
            dtype=float,
            copy=True,
        )
    ]

    for segment_index in range(
        path.shape[0] - 1
    ):
        start = path[
            segment_index
        ]

        end = path[
            segment_index + 1
        ]

        segment = (
            end
            - start
        )

        length = float(
            np.linalg.norm(
                segment
            )
        )

        if length <= (
            _DEFAULT_NUMERICAL_TOLERANCE
        ):
            if not np.allclose(
                samples[-1],
                end,
                atol=(
                    _DEFAULT_NUMERICAL_TOLERANCE
                ),
                rtol=0.0,
            ):
                samples.append(
                    np.array(
                        end,
                        dtype=float,
                        copy=True,
                    )
                )

            continue

        interval_count = max(
            1,
            int(
                ceil(
                    length
                    / spacing
                )
            ),
        )

        for step_index in range(
            1,
            interval_count + 1,
        ):
            fraction = (
                step_index
                / interval_count
            )

            sample = (
                start
                + fraction
                * segment
            )

            samples.append(
                np.array(
                    sample,
                    dtype=float,
                    copy=True,
                )
            )

    return np.vstack(
        samples
    )


def evaluate_path_clearance_risk(
    path_points: np.ndarray,
    structures: Sequence[
        UncertainSphere
    ],
    *,
    instrument_radius: float = 0.0,
    config: ChanceConstraintConfig | None = None,
) -> PathRiskEstimate:
    """Evaluate chance constraints along a 3-D path.

    Every sampled path point is evaluated against every supplied uncertain
    sphere.

    Two path-level diagnostics are returned:

    ``maximum_point_violation_probability``
        Maximum local chance-constraint violation probability.

    ``union_bound_violation_probability``
        ``min(1, sum(p_i))`` across all sampled point/structure events.

    The union bound does not assume independence and is therefore conservative,
    but it depends on path discretisation. It should not be interpreted as an
    exact trajectory collision probability.
    """

    if config is None:
        config = ChanceConstraintConfig()

    raw_path = np.asarray(
        path_points,
        dtype=float,
    )

    if (
        raw_path.ndim != 2
        or raw_path.shape[1] != 3
        or raw_path.shape[0] < 1
    ):
        raise ValueError(
            "path_points must have shape (N, 3) with N >= 1."
        )

    if not np.all(
        np.isfinite(raw_path)
    ):
        raise ValueError(
            "path_points must contain only finite values."
        )

    radius = _validate_nonnegative_finite(
        instrument_radius,
        name="instrument_radius",
    )

    structure_tuple = tuple(
        structures
    )

    for structure in structure_tuple:
        if not isinstance(
            structure,
            UncertainSphere,
        ):
            raise TypeError(
                "structures must contain only UncertainSphere instances."
            )

    if config.sample_spacing is None:
        sampled_points = np.array(
            raw_path,
            dtype=float,
            copy=True,
        )

    else:
        sampled_points = sample_polyline(
            raw_path,
            config.sample_spacing,
        )

    events: list[
        PathRiskEvent
    ] = []

    probabilities: list[
        float
    ] = []

    for point_index, point in enumerate(
        sampled_points
    ):
        for (
            structure_index,
            structure,
        ) in enumerate(
            structure_tuple
        ):
            estimate = (
                estimate_clearance_risk(
                    point,
                    structure,
                    instrument_radius=(
                        radius
                    ),
                    config=config,
                )
            )

            events.append(
                PathRiskEvent(
                    point_index=(
                        point_index
                    ),
                    structure_index=(
                        structure_index
                    ),
                    structure_label=(
                        structure.label
                    ),
                    estimate=(
                        estimate
                    ),
                )
            )

            probabilities.append(
                estimate.violation_probability
            )

    if probabilities:
        probability_array = np.asarray(
            probabilities,
            dtype=float,
        )

        worst_event_index = int(
            np.argmax(
                probability_array
            )
        )

        maximum_probability = float(
            probability_array[
                worst_event_index
            ]
        )

        union_bound_probability = float(
            min(
                1.0,
                np.sum(
                    probability_array
                ),
            )
        )

    else:
        worst_event_index = None

        maximum_probability = 0.0

        union_bound_probability = 0.0

    accepted = (
        maximum_probability
        <= config.max_point_violation_probability
    )

    if (
        config.max_path_union_bound_probability
        is not None
    ):
        accepted = (
            accepted
            and union_bound_probability
            <= config.max_path_union_bound_probability
        )

    return PathRiskEstimate(
        sampled_points=np.array(
            sampled_points,
            dtype=float,
            copy=True,
        ),
        events=tuple(
            events
        ),
        maximum_point_violation_probability=(
            maximum_probability
        ),
        union_bound_violation_probability=(
            union_bound_probability
        ),
        worst_event_index=(
            worst_event_index
        ),
        accepted=bool(
            accepted
        ),
    )