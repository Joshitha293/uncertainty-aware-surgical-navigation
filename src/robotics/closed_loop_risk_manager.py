"""Closed-loop uncertainty-aware risk supervision.

The supervisor distinguishes between:

1. immediate/current-state danger; and
2. risk located only later on the remaining trajectory.

Immediate danger can justify STOP. Future-only trajectory risk should normally
trigger replanning while the robot remains in an acceptable current state.

The probability thresholds are not clinical safety limits. This module is
simulation/research software and does not provide a clinical safety guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

from src.perception.uncertainty import (
    EstimatedStructure,
)
from src.robotics.chance_constrained_planner import (
    evaluate_joint_path_chance_constraint,
)
from src.robotics.instrument import (
    SurgicalInstrument,
)


class RiskManagementAction(str, Enum):
    """Available closed-loop supervisory actions."""

    CONTINUE = "continue"
    REPLAN = "replan"
    REACQUIRE = "reacquire"
    STOP = "stop"


@dataclass(frozen=True)
class ClosedLoopRiskConfig:
    """Thresholds controlling closed-loop supervisory behaviour."""

    replan_probability: float = 0.05
    reacquire_probability: float = 0.20
    stop_probability: float = 0.45

    reacquire_principal_sigma: float = 0.015
    stop_principal_sigma: float = 0.030

    minimum_replan_interval_steps: int = 2

    def __post_init__(self) -> None:
        """Validate supervisory thresholds."""

        probabilities = (
            self.replan_probability,
            self.reacquire_probability,
            self.stop_probability,
        )

        if not all(
            np.isfinite(
                value
            )
            for value in probabilities
        ):
            raise ValueError(
                "Probability thresholds must be finite."
            )

        if not (
            0.0
            < self.replan_probability
            < self.reacquire_probability
            < self.stop_probability
            <= 1.0
        ):
            raise ValueError(
                "Probability thresholds must satisfy "
                "0 < replan < reacquire < stop <= 1."
            )

        if (
            not np.isfinite(
                self.reacquire_principal_sigma
            )
            or self.reacquire_principal_sigma
            <= 0.0
        ):
            raise ValueError(
                "reacquire_principal_sigma must be finite and positive."
            )

        if (
            not np.isfinite(
                self.stop_principal_sigma
            )
            or self.stop_principal_sigma
            <= 0.0
        ):
            raise ValueError(
                "stop_principal_sigma must be finite and positive."
            )

        if (
            self.reacquire_principal_sigma
            >= self.stop_principal_sigma
        ):
            raise ValueError(
                "reacquire_principal_sigma must be smaller than "
                "stop_principal_sigma."
            )

        if (
            self.minimum_replan_interval_steps
            < 0
        ):
            raise ValueError(
                "minimum_replan_interval_steps must be non-negative."
            )


@dataclass(frozen=True)
class ClosedLoopRiskDecision:
    """Result of one closed-loop supervisory assessment."""

    action: RiskManagementAction

    maximum_point_violation_probability: float

    maximum_principal_sigma: float

    path_accepted: bool

    reason: str


def _maximum_principal_sigma(
    estimated_structures: Sequence[
        EstimatedStructure
    ],
) -> float:
    """Return largest principal positional standard deviation."""

    if len(
        estimated_structures
    ) == 0:
        return 0.0

    maximum = 0.0

    for structure in (
        estimated_structures
    ):
        covariance = np.asarray(
            structure
            .uncertainty
            .covariance,
            dtype=float,
        )

        if covariance.shape != (
            3,
            3,
        ):
            raise ValueError(
                "Each uncertainty covariance must have shape (3, 3)."
            )

        if not np.all(
            np.isfinite(
                covariance
            )
        ):
            raise ValueError(
                "Uncertainty covariance must contain finite values."
            )

        covariance = (
            0.5
            * (
                covariance
                + covariance.T
            )
        )

        eigenvalues = (
            np.linalg.eigvalsh(
                covariance
            )
        )

        largest_variance = float(
            np.max(
                eigenvalues
            )
        )

        largest_variance = max(
            0.0,
            largest_variance,
        )

        sigma = float(
            np.sqrt(
                largest_variance
            )
        )

        maximum = max(
            maximum,
            sigma,
        )

    return float(
        maximum
    )


def _maximum_probability(
    evaluation,
) -> float:
    """Extract maximum pointwise violation probability."""

    if hasattr(
        evaluation,
        "maximum_point_violation_probability",
    ):
        return float(
            evaluation
            .maximum_point_violation_probability
        )

    risk = getattr(
        evaluation,
        "risk",
        None,
    )

    if (
        risk is not None
        and hasattr(
            risk,
            "maximum_point_violation_probability",
        )
    ):
        return float(
            risk
            .maximum_point_violation_probability
        )

    path_risk = getattr(
        evaluation,
        "path_risk",
        None,
    )

    if (
        path_risk is not None
        and hasattr(
            path_risk,
            "maximum_point_violation_probability",
        )
    ):
        return float(
            path_risk
            .maximum_point_violation_probability
        )

    raise AttributeError(
        "Chance-constraint evaluation does not expose "
        "maximum_point_violation_probability."
    )


def _evaluation_accepted(
    evaluation,
) -> bool:
    """Extract chance-constraint acceptance."""

    if hasattr(
        evaluation,
        "accepted",
    ):
        return bool(
            evaluation.accepted
        )

    if hasattr(
        evaluation,
        "safe",
    ):
        return bool(
            evaluation.safe
        )

    raise AttributeError(
        "Chance-constraint evaluation does not expose "
        "an accepted/safe flag."
    )


def _evaluate_joint_path(
    *,
    instrument: SurgicalInstrument,
    path: np.ndarray,
    estimated_structures: Sequence[
        EstimatedStructure
    ],
    instrument_radius: float,
    chance_config,
    proximal_length: float,
    shaft_sample_spacing: float,
    edge_resolution: int,
):
    """Evaluate one joint-space path using the Phase 6 chance model."""

    return (
        evaluate_joint_path_chance_constraint(
            instrument=instrument,
            path=path,
            estimated_structures=(
                estimated_structures
            ),
            instrument_radius=(
                instrument_radius
            ),
            chance_config=(
                chance_config
            ),
            proximal_length=(
                proximal_length
            ),
            shaft_sample_spacing=(
                shaft_sample_spacing
            ),
            edge_resolution=(
                edge_resolution
            ),
        )
    )


class ClosedLoopRiskManager:
    """Stateful uncertainty-aware trajectory-risk supervisor."""

    def __init__(
        self,
        config: ClosedLoopRiskConfig | None = None,
    ) -> None:
        """Create the supervisor."""

        if config is None:
            config = (
                ClosedLoopRiskConfig()
            )

        self.config = config

        self._last_replan_step: (
            int | None
        ) = None

    @property
    def last_replan_step(
        self,
    ) -> int | None:
        """Return last registered successful replan step."""

        return (
            self._last_replan_step
        )

    def reset(
        self,
    ) -> None:
        """Reset state between independent episodes."""

        self._last_replan_step = None

    def register_replan(
        self,
        step_index: int,
    ) -> None:
        """Record a successful replanning event."""

        if step_index < 0:
            raise ValueError(
                "step_index must be non-negative."
            )

        self._last_replan_step = int(
            step_index
        )

    def _replan_is_allowed(
        self,
        step_index: int,
    ) -> bool:
        """Return whether replan hysteresis permits another replan."""

        if (
            self._last_replan_step
            is None
        ):
            return True

        elapsed = (
            int(
                step_index
            )
            - self._last_replan_step
        )

        return bool(
            elapsed
            >= self.config
            .minimum_replan_interval_steps
        )

    def evaluate(
        self,
        *,
        instrument: SurgicalInstrument,
        remaining_path: np.ndarray,
        estimated_structures: Sequence[
            EstimatedStructure
        ],
        instrument_radius: float,
        chance_config,
        step_index: int,
        proximal_length: float = 0.1,
        shaft_sample_spacing: float = 0.003,
        edge_resolution: int = 20,
    ) -> ClosedLoopRiskDecision:
        """Select a closed-loop supervisory action.

        Decision hierarchy
        ------------------
        1. Extreme perception uncertainty:
           STOP.

        2. Immediate/current configuration exceeds fail-safe risk:
           STOP.

        3. Perception uncertainty or current-state risk warrants a better
           observation:
           REACQUIRE.

        4. Current configuration remains acceptable but the future path
           violates the chance constraint:
           REPLAN.

        5. Current configuration and remaining trajectory remain acceptable:
           CONTINUE.

        Importantly, a high probability located only later on the remaining
        trajectory does not itself force STOP. If the current configuration
        remains acceptable, replanning is the appropriate response.
        """

        if step_index < 0:
            raise ValueError(
                "step_index must be non-negative."
            )

        path = np.asarray(
            remaining_path,
            dtype=float,
        )

        if (
            path.ndim != 2
            or path.shape[1] != 4
            or path.shape[0] < 2
        ):
            raise ValueError(
                "remaining_path must have shape (N, 4) with N >= 2."
            )

        if not np.all(
            np.isfinite(
                path
            )
        ):
            raise ValueError(
                "remaining_path must contain finite values."
            )

        if (
            not np.isfinite(
                instrument_radius
            )
            or instrument_radius < 0.0
        ):
            raise ValueError(
                "instrument_radius must be finite and non-negative."
            )

        if (
            not np.isfinite(
                proximal_length
            )
            or proximal_length <= 0.0
        ):
            raise ValueError(
                "proximal_length must be finite and positive."
            )

        if (
            not np.isfinite(
                shaft_sample_spacing
            )
            or shaft_sample_spacing <= 0.0
        ):
            raise ValueError(
                "shaft_sample_spacing must be finite and positive."
            )

        if edge_resolution < 2:
            raise ValueError(
                "edge_resolution must be at least 2."
            )

        maximum_sigma = (
            _maximum_principal_sigma(
                estimated_structures
            )
        )

        # -------------------------------------------------------------
        # Complete remaining trajectory evaluation
        # -------------------------------------------------------------

        path_evaluation = (
            _evaluate_joint_path(
                instrument=instrument,
                path=path,
                estimated_structures=(
                    estimated_structures
                ),
                instrument_radius=(
                    instrument_radius
                ),
                chance_config=(
                    chance_config
                ),
                proximal_length=(
                    proximal_length
                ),
                shaft_sample_spacing=(
                    shaft_sample_spacing
                ),
                edge_resolution=(
                    edge_resolution
                ),
            )
        )

        path_probability = (
            _maximum_probability(
                path_evaluation
            )
        )

        path_accepted = (
            _evaluation_accepted(
                path_evaluation
            )
        )

        # -------------------------------------------------------------
        # Current-state evaluation
        # -------------------------------------------------------------
        #
        # Evaluate a zero-motion path containing the current configuration
        # twice. This reuses exactly the same Phase 6 risk machinery rather
        # than introducing a second risk implementation.

        current_q = (
            path[
                0
            ].copy()
        )

        current_path = np.vstack(
            [
                current_q,
                current_q,
            ]
        )

        current_evaluation = (
            _evaluate_joint_path(
                instrument=instrument,
                path=current_path,
                estimated_structures=(
                    estimated_structures
                ),
                instrument_radius=(
                    instrument_radius
                ),
                chance_config=(
                    chance_config
                ),
                proximal_length=(
                    proximal_length
                ),
                shaft_sample_spacing=(
                    shaft_sample_spacing
                ),
                edge_resolution=max(
                    2,
                    edge_resolution,
                ),
            )
        )

        current_probability = (
            _maximum_probability(
                current_evaluation
            )
        )

        # -------------------------------------------------------------
        # 1. Extreme perception uncertainty -> STOP
        # -------------------------------------------------------------

        if (
            maximum_sigma
            >= self.config
            .stop_principal_sigma
        ):
            return ClosedLoopRiskDecision(
                action=(
                    RiskManagementAction.STOP
                ),
                maximum_point_violation_probability=(
                    path_probability
                ),
                maximum_principal_sigma=(
                    maximum_sigma
                ),
                path_accepted=(
                    path_accepted
                ),
                reason=(
                    "Perception uncertainty exceeded the "
                    "fail-safe principal-sigma limit."
                ),
            )

        # -------------------------------------------------------------
        # 2. Immediate/current-state risk -> STOP
        # -------------------------------------------------------------

        if (
            current_probability
            >= self.config
            .stop_probability
        ):
            return ClosedLoopRiskDecision(
                action=(
                    RiskManagementAction.STOP
                ),
                maximum_point_violation_probability=(
                    path_probability
                ),
                maximum_principal_sigma=(
                    maximum_sigma
                ),
                path_accepted=(
                    path_accepted
                ),
                reason=(
                    "Estimated risk at the current configuration "
                    "exceeded the fail-safe probability limit."
                ),
            )

        # -------------------------------------------------------------
        # 3. High uncertainty -> REACQUIRE
        # -------------------------------------------------------------

        if (
            maximum_sigma
            >= self.config
            .reacquire_principal_sigma
        ):
            return ClosedLoopRiskDecision(
                action=(
                    RiskManagementAction.REACQUIRE
                ),
                maximum_point_violation_probability=(
                    path_probability
                ),
                maximum_principal_sigma=(
                    maximum_sigma
                ),
                path_accepted=(
                    path_accepted
                ),
                reason=(
                    "Perception uncertainty exceeded the "
                    "reacquisition principal-sigma threshold."
                ),
            )

        # -------------------------------------------------------------
        # 4. Current-state concern -> REACQUIRE
        # -------------------------------------------------------------

        if (
            current_probability
            >= self.config
            .reacquire_probability
        ):
            return ClosedLoopRiskDecision(
                action=(
                    RiskManagementAction.REACQUIRE
                ),
                maximum_point_violation_probability=(
                    path_probability
                ),
                maximum_principal_sigma=(
                    maximum_sigma
                ),
                path_accepted=(
                    path_accepted
                ),
                reason=(
                    "Estimated current-configuration risk warrants "
                    "perception reacquisition before further motion."
                ),
            )

        # -------------------------------------------------------------
        # 5. Future trajectory risk -> REPLAN
        # -------------------------------------------------------------

        future_path_requires_action = bool(
            (
                not path_accepted
            )
            or (
                path_probability
                >= self.config
                .replan_probability
            )
        )

        if future_path_requires_action:
            if (
                self._replan_is_allowed(
                    step_index
                )
            ):
                return ClosedLoopRiskDecision(
                    action=(
                        RiskManagementAction.REPLAN
                    ),
                    maximum_point_violation_probability=(
                        path_probability
                    ),
                    maximum_principal_sigma=(
                        maximum_sigma
                    ),
                    path_accepted=(
                        path_accepted
                    ),
                    reason=(
                        "Current configuration remains acceptable, "
                        "but estimated future trajectory risk requires "
                        "replanning."
                    ),
                )

            # Hysteresis prevents immediate repeated replanning. Instead,
            # request a new perception observation before deciding again.
            return ClosedLoopRiskDecision(
                action=(
                    RiskManagementAction.REACQUIRE
                ),
                maximum_point_violation_probability=(
                    path_probability
                ),
                maximum_principal_sigma=(
                    maximum_sigma
                ),
                path_accepted=(
                    path_accepted
                ),
                reason=(
                    "Future trajectory requires intervention, but "
                    "replan hysteresis is active; reacquire perception."
                ),
            )

        # -------------------------------------------------------------
        # 6. Everything acceptable -> CONTINUE
        # -------------------------------------------------------------

        return ClosedLoopRiskDecision(
            action=(
                RiskManagementAction.CONTINUE
            ),
            maximum_point_violation_probability=(
                path_probability
            ),
            maximum_principal_sigma=(
                maximum_sigma
            ),
            path_accepted=True,
            reason=(
                "Current configuration and remaining trajectory "
                "satisfy the supervisory risk criteria."
            ),
        )