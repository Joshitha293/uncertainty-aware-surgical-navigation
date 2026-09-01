"""Temporal state estimation for uncertainty-aware surgical navigation.

This module implements a six-state constant-velocity Kalman filter:

    x = [px, py, pz, vx, vy, vz]

The filter consumes 3-D position measurements and their associated
3 x 3 measurement covariance matrices.

A continuous white-acceleration model is used to construct process noise.

The resulting position covariance can be converted directly into the
project's existing PositionUncertainty representation.

This is simulation-only engineering validation and is not intended as a
clinical anatomical-motion model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.perception.uncertainty import (
    EstimatedStructure,
    PositionUncertainty,
)


@dataclass(frozen=True)
class KalmanFilterConfig:
    """Configuration for constant-velocity state estimation."""

    acceleration_sigma: float = 0.020

    initial_velocity_sigma: float = 0.050

    innovation_gate_threshold: float | None = None

    def __post_init__(self) -> None:
        if (
            not np.isfinite(
                self.acceleration_sigma
            )
            or self.acceleration_sigma < 0.0
        ):
            raise ValueError(
                "acceleration_sigma must be finite and non-negative."
            )

        if (
            not np.isfinite(
                self.initial_velocity_sigma
            )
            or self.initial_velocity_sigma < 0.0
        ):
            raise ValueError(
                "initial_velocity_sigma must be finite and non-negative."
            )

        if (
            self.innovation_gate_threshold
            is not None
        ):
            if (
                not np.isfinite(
                    self.innovation_gate_threshold
                )
                or self.innovation_gate_threshold
                <= 0.0
            ):
                raise ValueError(
                    "innovation_gate_threshold must be finite and positive."
                )


@dataclass(frozen=True)
class TrackingState:
    """One Gaussian position-and-velocity state estimate."""

    mean: np.ndarray

    covariance: np.ndarray

    timestamp: float

    accepted_measurements: int

    def __post_init__(self) -> None:
        mean = np.asarray(
            self.mean,
            dtype=float,
        )

        covariance = np.asarray(
            self.covariance,
            dtype=float,
        )

        if mean.shape != (
            6,
        ):
            raise ValueError(
                "mean must have shape (6,)."
            )

        if covariance.shape != (
            6,
            6,
        ):
            raise ValueError(
                "covariance must have shape (6, 6)."
            )

        if (
            not np.all(
                np.isfinite(
                    mean
                )
            )
            or not np.all(
                np.isfinite(
                    covariance
                )
            )
        ):
            raise ValueError(
                "state must contain finite values."
            )

        if not np.allclose(
            covariance,
            covariance.T,
            atol=1e-10,
        ):
            raise ValueError(
                "covariance must be symmetric."
            )

        if np.any(
            np.linalg.eigvalsh(
                covariance
            )
            < -1e-10
        ):
            raise ValueError(
                "covariance must be positive semi-definite."
            )

        if (
            not np.isfinite(
                self.timestamp
            )
        ):
            raise ValueError(
                "timestamp must be finite."
            )

        if self.accepted_measurements < 0:
            raise ValueError(
                "accepted_measurements must be non-negative."
            )

        object.__setattr__(
            self,
            "mean",
            mean,
        )

        object.__setattr__(
            self,
            "covariance",
            covariance,
        )

    @property
    def position(
        self,
    ) -> np.ndarray:
        """Return estimated 3-D position."""

        return self.mean[
            :3
        ].copy()

    @property
    def velocity(
        self,
    ) -> np.ndarray:
        """Return estimated 3-D velocity."""

        return self.mean[
            3:
        ].copy()

    @property
    def position_covariance(
        self,
    ) -> np.ndarray:
        """Return 3 x 3 marginal positional covariance."""

        return self.covariance[
            :3,
            :3
        ].copy()

    @property
    def position_uncertainty(
        self,
    ) -> PositionUncertainty:
        """Convert marginal covariance to existing uncertainty type."""

        return PositionUncertainty(
            covariance=(
                self.position_covariance
            )
        )


@dataclass(frozen=True)
class KalmanUpdateResult:
    """Result and diagnostics from one measurement update."""

    state: TrackingState

    innovation: np.ndarray

    innovation_covariance: np.ndarray

    mahalanobis_squared: float

    accepted: bool


def _validate_measurement(
    measurement: np.ndarray,
) -> np.ndarray:
    """Validate one 3-D measurement."""

    measurement = np.asarray(
        measurement,
        dtype=float,
    )

    if measurement.shape != (
        3,
    ):
        raise ValueError(
            "measurement must have shape (3,)."
        )

    if not np.all(
        np.isfinite(
            measurement
        )
    ):
        raise ValueError(
            "measurement must contain finite values."
        )

    return measurement


def _validate_measurement_covariance(
    covariance: np.ndarray,
) -> np.ndarray:
    """Validate one 3 x 3 measurement covariance."""

    covariance = np.asarray(
        covariance,
        dtype=float,
    )

    if covariance.shape != (
        3,
        3,
    ):
        raise ValueError(
            "measurement_covariance must have shape (3, 3)."
        )

    if not np.all(
        np.isfinite(
            covariance
        )
    ):
        raise ValueError(
            "measurement_covariance must contain finite values."
        )

    if not np.allclose(
        covariance,
        covariance.T,
        atol=1e-10,
    ):
        raise ValueError(
            "measurement_covariance must be symmetric."
        )

    if np.any(
        np.linalg.eigvalsh(
            covariance
        )
        < -1e-10
    ):
        raise ValueError(
            "measurement_covariance must be positive semi-definite."
        )

    return covariance


def transition_matrix(
    dt: float,
) -> np.ndarray:
    """Return constant-velocity state-transition matrix."""

    if (
        not np.isfinite(
            dt
        )
        or dt < 0.0
    ):
        raise ValueError(
            "dt must be finite and non-negative."
        )

    matrix = np.eye(
        6,
        dtype=float,
    )

    matrix[
        :3,
        3:
    ] = (
        dt
        * np.eye(
            3,
            dtype=float,
        )
    )

    return matrix


def process_noise_covariance(
    dt: float,
    acceleration_sigma: float,
) -> np.ndarray:
    """Return white-acceleration process-noise covariance."""

    if (
        not np.isfinite(
            dt
        )
        or dt < 0.0
    ):
        raise ValueError(
            "dt must be finite and non-negative."
        )

    if (
        not np.isfinite(
            acceleration_sigma
        )
        or acceleration_sigma < 0.0
    ):
        raise ValueError(
            "acceleration_sigma must be finite and non-negative."
        )

    variance = (
        acceleration_sigma
        ** 2
    )

    q = np.zeros(
        (
            6,
            6,
        ),
        dtype=float,
    )

    identity = np.eye(
        3,
        dtype=float,
    )

    q[
        :3,
        :3
    ] = (
        0.25
        * dt**4
        * variance
        * identity
    )

    q[
        :3,
        3:
    ] = (
        0.5
        * dt**3
        * variance
        * identity
    )

    q[
        3:,
        :3
    ] = (
        0.5
        * dt**3
        * variance
        * identity
    )

    q[
        3:,
        3:
    ] = (
        dt**2
        * variance
        * identity
    )

    return q


class ConstantVelocityKalmanFilter:
    """Six-state constant-velocity Kalman filter."""

    def __init__(
        self,
        config: KalmanFilterConfig
        | None = None,
    ) -> None:
        if config is None:
            config = (
                KalmanFilterConfig()
            )

        self.config = config

    def initialise(
        self,
        *,
        measurement: np.ndarray,
        measurement_covariance: np.ndarray,
        timestamp: float,
    ) -> TrackingState:
        """Initialise state directly from first position measurement."""

        measurement = (
            _validate_measurement(
                measurement
            )
        )

        measurement_covariance = (
            _validate_measurement_covariance(
                measurement_covariance
            )
        )

        if not np.isfinite(
            timestamp
        ):
            raise ValueError(
                "timestamp must be finite."
            )

        mean = np.zeros(
            6,
            dtype=float,
        )

        mean[
            :3
        ] = measurement

        covariance = np.zeros(
            (
                6,
                6,
            ),
            dtype=float,
        )

        covariance[
            :3,
            :3
        ] = (
            measurement_covariance
        )

        covariance[
            3:,
            3:
        ] = (
            self.config
            .initial_velocity_sigma
            ** 2
        ) * np.eye(
            3,
            dtype=float,
        )

        return TrackingState(
            mean=mean,
            covariance=covariance,
            timestamp=float(
                timestamp
            ),
            accepted_measurements=1,
        )

    def predict(
        self,
        state: TrackingState,
        *,
        timestamp: float,
    ) -> TrackingState:
        """Predict state to a later timestamp."""

        if not np.isfinite(
            timestamp
        ):
            raise ValueError(
                "timestamp must be finite."
            )

        dt = float(
            timestamp
            - state.timestamp
        )

        if dt < 0.0:
            raise ValueError(
                "prediction timestamp cannot precede state timestamp."
            )

        transition = (
            transition_matrix(
                dt
            )
        )

        process_noise = (
            process_noise_covariance(
                dt,
                self.config
                .acceleration_sigma,
            )
        )

        mean = (
            transition
            @ state.mean
        )

        covariance = (
            transition
            @ state.covariance
            @ transition.T
            + process_noise
        )

        covariance = (
            0.5
            * (
                covariance
                + covariance.T
            )
        )

        return TrackingState(
            mean=mean,
            covariance=covariance,
            timestamp=float(
                timestamp
            ),
            accepted_measurements=(
                state.accepted_measurements
            ),
        )

    def update(
        self,
        predicted_state: TrackingState,
        *,
        measurement: np.ndarray,
        measurement_covariance: np.ndarray,
    ) -> KalmanUpdateResult:
        """Correct a predicted state using one uncertain 3-D measurement."""

        measurement = (
            _validate_measurement(
                measurement
            )
        )

        measurement_covariance = (
            _validate_measurement_covariance(
                measurement_covariance
            )
        )

        measurement_matrix = np.zeros(
            (
                3,
                6,
            ),
            dtype=float,
        )

        measurement_matrix[
            :,
            :3
        ] = np.eye(
            3,
            dtype=float,
        )

        innovation = (
            measurement
            - measurement_matrix
            @ predicted_state.mean
        )

        innovation_covariance = (
            measurement_matrix
            @ predicted_state.covariance
            @ measurement_matrix.T
            + measurement_covariance
        )

        inverse_innovation_covariance = (
            np.linalg.pinv(
                innovation_covariance,
                hermitian=True,
            )
        )

        mahalanobis_squared = float(
            innovation.T
            @ inverse_innovation_covariance
            @ innovation
        )

        gate = (
            self.config
            .innovation_gate_threshold
        )

        if (
            gate is not None
            and mahalanobis_squared
            > gate
        ):
            return KalmanUpdateResult(
                state=predicted_state,
                innovation=np.asarray(
                    innovation,
                    dtype=float,
                ),
                innovation_covariance=np.asarray(
                    innovation_covariance,
                    dtype=float,
                ),
                mahalanobis_squared=(
                    mahalanobis_squared
                ),
                accepted=False,
            )

        kalman_gain = (
            predicted_state.covariance
            @ measurement_matrix.T
            @ inverse_innovation_covariance
        )

        corrected_mean = (
            predicted_state.mean
            + kalman_gain
            @ innovation
        )

        identity = np.eye(
            6,
            dtype=float,
        )

        correction = (
            identity
            - kalman_gain
            @ measurement_matrix
        )

        # Joseph-form covariance update for numerical robustness.
        corrected_covariance = (
            correction
            @ predicted_state.covariance
            @ correction.T
            + kalman_gain
            @ measurement_covariance
            @ kalman_gain.T
        )

        corrected_covariance = (
            0.5
            * (
                corrected_covariance
                + corrected_covariance.T
            )
        )

        corrected_state = TrackingState(
            mean=corrected_mean,
            covariance=(
                corrected_covariance
            ),
            timestamp=(
                predicted_state.timestamp
            ),
            accepted_measurements=(
                predicted_state
                .accepted_measurements
                + 1
            ),
        )

        return KalmanUpdateResult(
            state=corrected_state,
            innovation=np.asarray(
                innovation,
                dtype=float,
            ),
            innovation_covariance=np.asarray(
                innovation_covariance,
                dtype=float,
            ),
            mahalanobis_squared=(
                mahalanobis_squared
            ),
            accepted=True,
        )

    def step(
        self,
        state: TrackingState,
        *,
        measurement: np.ndarray,
        measurement_covariance: np.ndarray,
        timestamp: float,
    ) -> KalmanUpdateResult:
        """Predict then update using one timestamped observation."""

        predicted = self.predict(
            state,
            timestamp=timestamp,
        )

        return self.update(
            predicted,
            measurement=measurement,
            measurement_covariance=(
                measurement_covariance
            ),
        )


def tracked_estimated_structure(
    state: TrackingState,
    *,
    physical_radius: float,
    base_safety_margin: float,
) -> EstimatedStructure:
    """Convert a temporal state estimate into navigation representation."""

    return EstimatedStructure(
        estimated_centre=(
            state.position
        ),
        physical_radius=float(
            physical_radius
        ),
        base_safety_margin=float(
            base_safety_margin
        ),
        uncertainty=(
            state.position_uncertainty
        ),
    )