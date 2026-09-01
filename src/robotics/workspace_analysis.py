"""Workspace characterisation for the Phase 2 RCM surgical instrument."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.robotics.advanced_kinematics import (
    evaluate_kinematics,
)
from src.robotics.instrument import SurgicalInstrument


@dataclass(frozen=True)
class WorkspaceCharacterisation:
    """Summary of a sampled kinematic workspace."""

    sample_count: int

    tip_minimum: np.ndarray
    tip_maximum: np.ndarray

    minimum_manipulability: float
    maximum_manipulability: float
    mean_manipulability: float

    minimum_condition_number: float
    maximum_condition_number: float
    mean_condition_number: float

    minimum_singular_value: float
    maximum_singular_value: float

    near_singular_count: int
    near_singular_fraction: float


def characterise_kinematic_workspace(
    instrument: SurgicalInstrument,
    *,
    yaw_samples: int = 9,
    pitch_samples: int = 7,
    insertion_samples: int = 7,
    singular_value_threshold: float = 1e-6,
) -> WorkspaceCharacterisation:
    """Sample and quantitatively characterise the admissible workspace.

    Roll does not change tip position or the singular values of the
    positional Jacobian in this simplified instrument, so roll is fixed to
    zero during workspace sampling.
    """

    for name, value in (
        ("yaw_samples", yaw_samples),
        ("pitch_samples", pitch_samples),
        ("insertion_samples", insertion_samples),
    ):
        if value < 2:
            raise ValueError(
                f"{name} must be at least 2."
            )

    if (
        not np.isfinite(
            singular_value_threshold
        )
        or singular_value_threshold < 0.0
    ):
        raise ValueError(
            "singular_value_threshold must be finite and non-negative."
        )

    limits = instrument.joint_limits

    yaw_values = np.linspace(
        limits.yaw_min,
        limits.yaw_max,
        yaw_samples,
        dtype=float,
    )

    pitch_values = np.linspace(
        limits.pitch_min,
        limits.pitch_max,
        pitch_samples,
        dtype=float,
    )

    insertion_values = np.linspace(
        limits.insertion_min,
        limits.insertion_max,
        insertion_samples,
        dtype=float,
    )

    tip_positions: list[np.ndarray] = []
    manipulabilities: list[float] = []
    condition_numbers: list[float] = []
    minimum_singular_values: list[float] = []
    maximum_singular_values: list[float] = []

    near_singular_count = 0

    for yaw in yaw_values:
        for pitch in pitch_values:
            for insertion in insertion_values:
                q = np.asarray(
                    [
                        yaw,
                        pitch,
                        insertion,
                        0.0,
                    ],
                    dtype=float,
                )

                state = evaluate_kinematics(
                    instrument,
                    q,
                )

                tip_positions.append(
                    instrument.forward_position(
                        q
                    )
                )

                manipulabilities.append(
                    state.manipulability
                )

                condition_numbers.append(
                    state.condition_number
                )

                minimum_singular = float(
                    np.min(
                        state.singular_values
                    )
                )

                maximum_singular = float(
                    np.max(
                        state.singular_values
                    )
                )

                minimum_singular_values.append(
                    minimum_singular
                )

                maximum_singular_values.append(
                    maximum_singular
                )

                if (
                    minimum_singular
                    <= singular_value_threshold
                ):
                    near_singular_count += 1

    positions = np.asarray(
        tip_positions,
        dtype=float,
    )

    manipulability_array = np.asarray(
        manipulabilities,
        dtype=float,
    )

    condition_array = np.asarray(
        condition_numbers,
        dtype=float,
    )

    if not np.all(
        np.isfinite(
            condition_array
        )
    ):
        finite_conditions = condition_array[
            np.isfinite(
                condition_array
            )
        ]

        mean_condition = (
            float(
                np.mean(
                    finite_conditions
                )
            )
            if finite_conditions.size
            else float(
                "inf"
            )
        )

    else:
        mean_condition = float(
            np.mean(
                condition_array
            )
        )

    sample_count = int(
        positions.shape[
            0
        ]
    )

    return WorkspaceCharacterisation(
        sample_count=sample_count,
        tip_minimum=np.min(
            positions,
            axis=0,
        ),
        tip_maximum=np.max(
            positions,
            axis=0,
        ),
        minimum_manipulability=float(
            np.min(
                manipulability_array
            )
        ),
        maximum_manipulability=float(
            np.max(
                manipulability_array
            )
        ),
        mean_manipulability=float(
            np.mean(
                manipulability_array
            )
        ),
        minimum_condition_number=float(
            np.min(
                condition_array
            )
        ),
        maximum_condition_number=float(
            np.max(
                condition_array
            )
        ),
        mean_condition_number=(
            mean_condition
        ),
        minimum_singular_value=float(
            np.min(
                minimum_singular_values
            )
        ),
        maximum_singular_value=float(
            np.max(
                maximum_singular_values
            )
        ),
        near_singular_count=int(
            near_singular_count
        ),
        near_singular_fraction=float(
            near_singular_count
            / sample_count
        ),
    )