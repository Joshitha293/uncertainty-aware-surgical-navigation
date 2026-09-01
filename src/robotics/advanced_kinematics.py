"""Advanced kinematics for the RCM-constrained surgical instrument.

This module extends the deliberately simple SurgicalInstrument model
without changing its established Phase 1 API.

The existing instrument defines the configuration

    q = [yaw, pitch, insertion, roll]

and constrains the shaft to pass through a fixed remote centre of motion
(RCM).

This module adds:

- full tool-pose forward kinematics;
- analytical position Jacobian;
- analytical 6 x 4 geometric Jacobian;
- numerical Jacobian verification;
- velocity kinematics;
- normalised joint-limit margin;
- dimensionally normalised position-Jacobian metrics;
- manipulability;
- Jacobian condition number;
- verified analytical position IK.

The simplified RCM mechanism is parameterised directly in yaw, pitch,
insertion and roll. A Denavit-Hartenberg representation is therefore not
introduced artificially: the direct geometric formulation is both simpler
and more transparent for this model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.robotics.instrument import SurgicalInstrument


@dataclass(frozen=True)
class PositionIKResult:
    """Verification result for analytical position inverse kinematics."""

    q: np.ndarray

    target_position: np.ndarray

    reconstructed_position: np.ndarray

    position_error: float

    rcm_error: float

    success: bool


@dataclass(frozen=True)
class KinematicState:
    """Quantitative kinematic state of the surgical instrument."""

    q: np.ndarray

    transform: np.ndarray

    position_jacobian: np.ndarray

    geometric_jacobian: np.ndarray

    normalised_position_jacobian: np.ndarray

    singular_values: np.ndarray

    manipulability: float

    condition_number: float

    normalised_joint_limit_margin: float

    rcm_error: float


def _as_configuration(
    instrument: SurgicalInstrument,
    q: np.ndarray,
) -> np.ndarray:
    """Convert and validate one joint configuration."""

    configuration = np.asarray(
        q,
        dtype=float,
    )

    instrument.validate_configuration(
        configuration
    )

    return configuration


def joint_limit_arrays(
    instrument: SurgicalInstrument,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lower and upper limits in joint-vector order."""

    limits = instrument.joint_limits

    lower = np.asarray(
        [
            limits.yaw_min,
            limits.pitch_min,
            limits.insertion_min,
            limits.roll_min,
        ],
        dtype=float,
    )

    upper = np.asarray(
        [
            limits.yaw_max,
            limits.pitch_max,
            limits.insertion_max,
            limits.roll_max,
        ],
        dtype=float,
    )

    if not np.all(
        upper > lower
    ):
        raise ValueError(
            "Every joint must have a positive joint range."
        )

    return (
        lower,
        upper,
    )


def tool_rotation_matrix(
    instrument: SurgicalInstrument,
    q: np.ndarray,
) -> np.ndarray:
    """Return the world-to-tool orientation basis.

    The tool x-axis is aligned with the instrument shaft.

    At:

        yaw = 0
        pitch = 0
        roll = 0

    the tool frame is aligned with the world frame.

    The orientation is equivalent to:

        Rz(yaw) @ Ry(-pitch) @ Rx(roll)

    under the shaft-direction convention used by SurgicalInstrument.
    """

    configuration = _as_configuration(
        instrument,
        q,
    )

    yaw, pitch, _insertion, roll = (
        configuration
    )

    shaft_axis = (
        instrument.shaft_direction(
            yaw,
            pitch,
        )
    )

    azimuth_axis = np.asarray(
        [
            -np.sin(
                yaw
            ),
            np.cos(
                yaw
            ),
            0.0,
        ],
        dtype=float,
    )

    normal_axis = np.asarray(
        [
            -np.sin(
                pitch
            )
            * np.cos(
                yaw
            ),
            -np.sin(
                pitch
            )
            * np.sin(
                yaw
            ),
            np.cos(
                pitch
            ),
        ],
        dtype=float,
    )

    cosine_roll = float(
        np.cos(
            roll
        )
    )

    sine_roll = float(
        np.sin(
            roll
        )
    )

    rolled_y = (
        cosine_roll
        * azimuth_axis
        + sine_roll
        * normal_axis
    )

    rolled_z = (
        -sine_roll
        * azimuth_axis
        + cosine_roll
        * normal_axis
    )

    rotation = np.column_stack(
        (
            shaft_axis,
            rolled_y,
            rolled_z,
        )
    )

    return rotation


def forward_transform(
    instrument: SurgicalInstrument,
    q: np.ndarray,
) -> np.ndarray:
    """Return the 4 x 4 homogeneous tool-tip transform."""

    configuration = _as_configuration(
        instrument,
        q,
    )

    transform = np.eye(
        4,
        dtype=float,
    )

    transform[
        :3,
        :3,
    ] = tool_rotation_matrix(
        instrument,
        configuration,
    )

    transform[
        :3,
        3,
    ] = instrument.forward_position(
        configuration
    )

    return transform


def position_jacobian(
    instrument: SurgicalInstrument,
    q: np.ndarray,
) -> np.ndarray:
    """Return the analytical 3 x 4 tool-position Jacobian.

    The Jacobian maps joint velocity to Cartesian tip velocity:

        p_dot = J_p(q) q_dot

    Joint order:

        yaw
        pitch
        insertion
        roll
    """

    configuration = _as_configuration(
        instrument,
        q,
    )

    yaw, pitch, insertion, _roll = (
        configuration
    )

    cosine_yaw = float(
        np.cos(
            yaw
        )
    )

    sine_yaw = float(
        np.sin(
            yaw
        )
    )

    cosine_pitch = float(
        np.cos(
            pitch
        )
    )

    sine_pitch = float(
        np.sin(
            pitch
        )
    )

    derivative_yaw = np.asarray(
        [
            -insertion
            * cosine_pitch
            * sine_yaw,
            insertion
            * cosine_pitch
            * cosine_yaw,
            0.0,
        ],
        dtype=float,
    )

    derivative_pitch = np.asarray(
        [
            -insertion
            * sine_pitch
            * cosine_yaw,
            -insertion
            * sine_pitch
            * sine_yaw,
            insertion
            * cosine_pitch,
        ],
        dtype=float,
    )

    derivative_insertion = (
        instrument.shaft_direction(
            yaw,
            pitch,
        )
    )

    derivative_roll = np.zeros(
        3,
        dtype=float,
    )

    return np.column_stack(
        (
            derivative_yaw,
            derivative_pitch,
            derivative_insertion,
            derivative_roll,
        )
    )


def geometric_jacobian(
    instrument: SurgicalInstrument,
    q: np.ndarray,
) -> np.ndarray:
    """Return the analytical 6 x 4 geometric Jacobian.

    The upper three rows describe Cartesian tool-tip velocity.

    The lower three rows describe angular velocity expressed in the
    world frame:

        [v]
        [w] = J(q) q_dot

    Roll produces no tip translation but rotates the tool about its
    own shaft.
    """

    configuration = _as_configuration(
        instrument,
        q,
    )

    yaw, pitch, _insertion, _roll = (
        configuration
    )

    linear = position_jacobian(
        instrument,
        configuration,
    )

    yaw_axis = np.asarray(
        [
            0.0,
            0.0,
            1.0,
        ],
        dtype=float,
    )

    pitch_axis = np.asarray(
        [
            np.sin(
                yaw
            ),
            -np.cos(
                yaw
            ),
            0.0,
        ],
        dtype=float,
    )

    insertion_axis = np.zeros(
        3,
        dtype=float,
    )

    roll_axis = (
        instrument.shaft_direction(
            yaw,
            pitch,
        )
    )

    angular = np.column_stack(
        (
            yaw_axis,
            pitch_axis,
            insertion_axis,
            roll_axis,
        )
    )

    return np.vstack(
        (
            linear,
            angular,
        )
    )


def _difference_pair(
    q: np.ndarray,
    *,
    joint_index: int,
    step: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
]:
    """Choose central or one-sided finite-difference samples."""

    q_plus = np.array(
        q,
        dtype=float,
        copy=True,
    )

    q_minus = np.array(
        q,
        dtype=float,
        copy=True,
    )

    can_step_positive = bool(
        q[
            joint_index
        ]
        + step
        <= upper[
            joint_index
        ]
    )

    can_step_negative = bool(
        q[
            joint_index
        ]
        - step
        >= lower[
            joint_index
        ]
    )

    if (
        can_step_positive
        and can_step_negative
    ):
        q_plus[
            joint_index
        ] += step

        q_minus[
            joint_index
        ] -= step

        denominator = (
            2.0
            * step
        )

    elif can_step_positive:
        q_plus[
            joint_index
        ] += step

        denominator = step

    elif can_step_negative:
        q_minus[
            joint_index
        ] -= step

        denominator = step

    else:
        raise ValueError(
            "Finite-difference step cannot be taken within "
            f"joint limits for joint {joint_index}."
        )

    return (
        q_plus,
        q_minus,
        denominator,
    )


def numerical_geometric_jacobian(
    instrument: SurgicalInstrument,
    q: np.ndarray,
    *,
    step: float = 1e-7,
) -> np.ndarray:
    """Estimate the geometric Jacobian using finite differences.

    This function is primarily a verification reference for the analytical
    Jacobian rather than the preferred runtime implementation.
    """

    configuration = _as_configuration(
        instrument,
        q,
    )

    if (
        not np.isfinite(
            step
        )
        or step <= 0.0
    ):
        raise ValueError(
            "step must be finite and positive."
        )

    lower, upper = (
        joint_limit_arrays(
            instrument
        )
    )

    reference_rotation = (
        tool_rotation_matrix(
            instrument,
            configuration,
        )
    )

    reference_position = (
        instrument.forward_position(
            configuration
        )
    )

    jacobian = np.zeros(
        (
            6,
            4,
        ),
        dtype=float,
    )

    for joint_index in range(
        4
    ):
        (
            q_plus,
            q_minus,
            denominator,
        ) = _difference_pair(
            configuration,
            joint_index=joint_index,
            step=step,
            lower=lower,
            upper=upper,
        )

        plus_is_reference = bool(
            np.array_equal(
                q_plus,
                configuration,
            )
        )

        minus_is_reference = bool(
            np.array_equal(
                q_minus,
                configuration,
            )
        )

        if plus_is_reference:
            position_plus = (
                reference_position
            )

            rotation_plus = (
                reference_rotation
            )

        else:
            position_plus = (
                instrument.forward_position(
                    q_plus
                )
            )

            rotation_plus = (
                tool_rotation_matrix(
                    instrument,
                    q_plus,
                )
            )

        if minus_is_reference:
            position_minus = (
                reference_position
            )

            rotation_minus = (
                reference_rotation
            )

        else:
            position_minus = (
                instrument.forward_position(
                    q_minus
                )
            )

            rotation_minus = (
                tool_rotation_matrix(
                    instrument,
                    q_minus,
                )
            )

        position_derivative = (
            position_plus
            - position_minus
        ) / denominator

        rotation_derivative = (
            rotation_plus
            - rotation_minus
        ) / denominator

        angular_matrix = (
            rotation_derivative
            @ reference_rotation.T
        )

        angular_matrix = (
            0.5
            * (
                angular_matrix
                - angular_matrix.T
            )
        )

        angular_velocity = np.asarray(
            [
                angular_matrix[
                    2,
                    1,
                ],
                angular_matrix[
                    0,
                    2,
                ],
                angular_matrix[
                    1,
                    0,
                ],
            ],
            dtype=float,
        )

        jacobian[
            :3,
            joint_index,
        ] = position_derivative

        jacobian[
            3:,
            joint_index,
        ] = angular_velocity

    return jacobian


def numerical_position_jacobian(
    instrument: SurgicalInstrument,
    q: np.ndarray,
    *,
    step: float = 1e-7,
) -> np.ndarray:
    """Estimate the 3 x 4 position Jacobian numerically."""

    return numerical_geometric_jacobian(
        instrument,
        q,
        step=step,
    )[
        :3,
        :
    ]


def jacobian_max_abs_error(
    instrument: SurgicalInstrument,
    q: np.ndarray,
    *,
    step: float = 1e-7,
) -> float:
    """Return maximum element-wise analytical/numerical Jacobian error."""

    analytical = geometric_jacobian(
        instrument,
        q,
    )

    numerical = (
        numerical_geometric_jacobian(
            instrument,
            q,
            step=step,
        )
    )

    return float(
        np.max(
            np.abs(
                analytical
                - numerical
            )
        )
    )


def tool_twist(
    instrument: SurgicalInstrument,
    q: np.ndarray,
    q_dot: np.ndarray,
) -> np.ndarray:
    """Map joint rates to a 6-D tool twist.

    Returns:

        [vx, vy, vz, wx, wy, wz]
    """

    configuration = _as_configuration(
        instrument,
        q,
    )

    velocity = np.asarray(
        q_dot,
        dtype=float,
    )

    if velocity.shape != (4,):
        raise ValueError(
            "q_dot must have shape (4,)."
        )

    if not np.all(
        np.isfinite(
            velocity
        )
    ):
        raise ValueError(
            "q_dot must contain finite values."
        )

    return (
        geometric_jacobian(
            instrument,
            configuration,
        )
        @ velocity
    )


def normalised_joint_limit_margin(
    instrument: SurgicalInstrument,
    q: np.ndarray,
) -> float:
    """Return minimum normalised distance to any joint limit.

    For each joint:

        lower limit -> 0
        upper limit -> 0
        midpoint    -> 0.5

    The returned value is the minimum over all four joints.

    A small value therefore indicates proximity to at least one joint limit.
    """

    configuration = _as_configuration(
        instrument,
        q,
    )

    lower, upper = (
        joint_limit_arrays(
            instrument
        )
    )

    ranges = (
        upper
        - lower
    )

    distance_from_lower = (
        configuration
        - lower
    ) / ranges

    distance_from_upper = (
        upper
        - configuration
    ) / ranges

    margin = np.minimum(
        distance_from_lower,
        distance_from_upper,
    )

    return float(
        np.min(
            margin
        )
    )


def normalised_position_jacobian(
    instrument: SurgicalInstrument,
    q: np.ndarray,
) -> np.ndarray:
    """Return a joint-range-normalised position Jacobian.

    The raw Jacobian mixes angular and translational joint units.

    For kinematic-quality comparisons we instead use normalised joint
    coordinates where each joint coordinate spans its configured range.

    If:

        q = q_lower + diag(range) s

    then:

        dp/ds = J_p diag(range)

    and every column of this Jacobian has Cartesian length units.

    This provides a more defensible basis for relative manipulability and
    conditioning metrics than applying determinant-based metrics directly
    to the mixed-unit raw Jacobian.
    """

    configuration = _as_configuration(
        instrument,
        q,
    )

    lower, upper = (
        joint_limit_arrays(
            instrument
        )
    )

    ranges = (
        upper
        - lower
    )

    return (
        position_jacobian(
            instrument,
            configuration,
        )
        @ np.diag(
            ranges
        )
    )


def position_singular_values(
    instrument: SurgicalInstrument,
    q: np.ndarray,
) -> np.ndarray:
    """Return singular values of the normalised position Jacobian."""

    jacobian = (
        normalised_position_jacobian(
            instrument,
            q,
        )
    )

    return np.linalg.svd(
        jacobian,
        compute_uv=False,
    )


def manipulability_index(
    instrument: SurgicalInstrument,
    q: np.ndarray,
) -> float:
    """Return a normalised position-manipulability index.

    The metric is the product of the three singular values of the
    joint-range-normalised position Jacobian.

    It is intended for relative comparison inside this simulated model,
    not as a clinical or hardware capability metric.
    """

    singular_values = (
        position_singular_values(
            instrument,
            q,
        )
    )

    return float(
        np.prod(
            singular_values
        )
    )


def position_jacobian_condition_number(
    instrument: SurgicalInstrument,
    q: np.ndarray,
    *,
    singular_tolerance: float = 1e-12,
) -> float:
    """Return condition number of the normalised position Jacobian."""

    if (
        not np.isfinite(
            singular_tolerance
        )
        or singular_tolerance < 0.0
    ):
        raise ValueError(
            "singular_tolerance must be finite and non-negative."
        )

    singular_values = (
        position_singular_values(
            instrument,
            q,
        )
    )

    minimum = float(
        np.min(
            singular_values
        )
    )

    maximum = float(
        np.max(
            singular_values
        )
    )

    if minimum <= singular_tolerance:
        return float(
            "inf"
        )

    return float(
        maximum
        / minimum
    )


def verify_position_ik(
    instrument: SurgicalInstrument,
    target_position: np.ndarray,
    *,
    roll: float = 0.0,
    position_tolerance: float = 1e-9,
    rcm_tolerance: float = 1e-8,
) -> PositionIKResult:
    """Solve and independently verify the existing analytical position IK."""

    if (
        not np.isfinite(
            position_tolerance
        )
        or position_tolerance < 0.0
    ):
        raise ValueError(
            "position_tolerance must be finite and non-negative."
        )

    if (
        not np.isfinite(
            rcm_tolerance
        )
        or rcm_tolerance < 0.0
    ):
        raise ValueError(
            "rcm_tolerance must be finite and non-negative."
        )

    target = np.asarray(
        target_position,
        dtype=float,
    )

    q = instrument.inverse_position(
        target,
        roll=roll,
    )

    reconstructed = (
        instrument.forward_position(
            q
        )
    )

    position_error = float(
        np.linalg.norm(
            reconstructed
            - target
        )
    )

    rcm_error = float(
        instrument.rcm_error(
            q
        )
    )

    success = bool(
        position_error
        <= position_tolerance
        and rcm_error
        <= rcm_tolerance
    )

    return PositionIKResult(
        q=np.array(
            q,
            dtype=float,
            copy=True,
        ),
        target_position=np.array(
            target,
            dtype=float,
            copy=True,
        ),
        reconstructed_position=np.array(
            reconstructed,
            dtype=float,
            copy=True,
        ),
        position_error=(
            position_error
        ),
        rcm_error=(
            rcm_error
        ),
        success=(
            success
        ),
    )


def evaluate_kinematics(
    instrument: SurgicalInstrument,
    q: np.ndarray,
) -> KinematicState:
    """Return a complete deterministic kinematic assessment."""

    configuration = _as_configuration(
        instrument,
        q,
    )

    normalised_jacobian = (
        normalised_position_jacobian(
            instrument,
            configuration,
        )
    )

    singular_values = np.linalg.svd(
        normalised_jacobian,
        compute_uv=False,
    )

    minimum = float(
        np.min(
            singular_values
        )
    )

    maximum = float(
        np.max(
            singular_values
        )
    )

    if minimum <= 1e-12:
        condition_number = float(
            "inf"
        )

    else:
        condition_number = float(
            maximum
            / minimum
        )

    return KinematicState(
        q=np.array(
            configuration,
            dtype=float,
            copy=True,
        ),
        transform=forward_transform(
            instrument,
            configuration,
        ),
        position_jacobian=(
            position_jacobian(
                instrument,
                configuration,
            )
        ),
        geometric_jacobian=(
            geometric_jacobian(
                instrument,
                configuration,
            )
        ),
        normalised_position_jacobian=(
            normalised_jacobian
        ),
        singular_values=np.array(
            singular_values,
            dtype=float,
            copy=True,
        ),
        manipulability=float(
            np.prod(
                singular_values
            )
        ),
        condition_number=(
            condition_number
        ),
        normalised_joint_limit_margin=(
            normalised_joint_limit_margin(
                instrument,
                configuration,
            )
        ),
        rcm_error=float(
            instrument.rcm_error(
                configuration
            )
        ),
    )