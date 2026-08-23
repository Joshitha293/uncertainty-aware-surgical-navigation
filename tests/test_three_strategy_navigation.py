"""Tests for perception-to-navigation integration."""

from __future__ import annotations

import numpy as np
import pytest

from src.geometry.workspace import SphericalStructure
from src.perception.camera import CameraPose
from src.perception.perception import PerceptionResult
from src.perception.uncertainty import (
    EstimatedStructure,
    PositionUncertainty,
)
from src.robotics.instrument import SurgicalInstrument
from src.simulation.three_strategy_navigation import (
    build_uncertainty_aware_planning_perception,
    run_navigation_from_perception,
)
from src.simulation.three_strategy_perception import (
    PerceptionStrategy,
    StrategyPerceptionResult,
)


def make_start_configuration() -> np.ndarray:
    """Return a valid benchmark start configuration."""

    return np.array(
        [
            np.deg2rad(-25.0),
            np.deg2rad(-15.0),
            0.16,
            0.0,
        ],
        dtype=float,
    )


def make_goal_configuration() -> np.ndarray:
    """Return a valid benchmark goal configuration."""

    return np.array(
        [
            np.deg2rad(35.0),
            np.deg2rad(25.0),
            0.25,
            0.0,
        ],
        dtype=float,
    )


def make_instrument() -> SurgicalInstrument:
    """Create the benchmark surgical instrument."""

    return SurgicalInstrument(
        rcm_position=np.zeros(
            3,
            dtype=float,
        )
    )


def make_strategy_perception(
    *,
    strategy: PerceptionStrategy = (
        PerceptionStrategy.FIXED
    ),
    estimated_centre: np.ndarray | None = None,
    sigma: float = 0.005,
    physical_radius: float = 0.020,
    base_safety_margin: float = 0.015,
) -> StrategyPerceptionResult:
    """Create a controlled planner-compatible perception result."""

    if estimated_centre is None:
        estimated_centre = np.array(
            [0.70, 0.70, 0.70],
            dtype=float,
        )

    uncertainty = (
        PositionUncertainty.isotropic(
            sigma=sigma
        )
    )

    estimated_structure = EstimatedStructure(
        estimated_centre=np.asarray(
            estimated_centre,
            dtype=float,
        ),
        physical_radius=physical_radius,
        base_safety_margin=(
            base_safety_margin
        ),
        uncertainty=uncertainty,
    )

    perception_result = PerceptionResult(
        estimated_structures=(
            estimated_structure,
        ),
        localisation_errors=np.array(
            [0.0],
            dtype=float,
        ),
    )

    return StrategyPerceptionResult(
        strategy=strategy,
        selected_pose=CameraPose(
            position=np.zeros(
                3,
                dtype=float,
            ),
            rotation=np.eye(
                3,
                dtype=float,
            ),
        ),
        perception_result=perception_result,
        mean_localisation_error=0.0,
        mean_predicted_sigma=sigma,
        camera_movement=0.0,
        visible_fraction=1.0,
        occluded_fraction=0.0,
        candidate_count=0,
        task_relevance=None,
        task_alignment=None,
    )


def test_uncertainty_is_propagated_into_planning_margin():
    """Perception sigma must explicitly alter planner safety geometry."""

    perception = make_strategy_perception(
        sigma=0.005,
        base_safety_margin=0.015,
    )

    planning = (
        build_uncertainty_aware_planning_perception(
            perception=perception,
            sigma_multiplier=2.0,
        )
    )

    assert planning.uncertainty_aware

    assert planning.sigma_multiplier == 2.0

    assert len(
        planning.structures
    ) == 1

    expected_margin = (
        0.015
        + 2.0 * 0.005
    )

    assert (
        planning.structures[0]
        .safety_margin
        == pytest.approx(
            expected_margin
        )
    )


def test_negative_sigma_multiplier_is_rejected():
    """Planning conservatism cannot use a negative uncertainty multiplier."""

    perception = make_strategy_perception()

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        build_uncertainty_aware_planning_perception(
            perception=perception,
            sigma_multiplier=-1.0,
        )


def test_navigation_preserves_strategy_identity():
    """The planner result must retain the originating perception strategy."""

    perception = make_strategy_perception(
        strategy=(
            PerceptionStrategy.GENERIC_ACTIVE
        )
    )

    true_structure = SphericalStructure(
        centre=np.array(
            [0.70, 0.70, 0.70],
            dtype=float,
        ),
        physical_radius=0.020,
        safety_margin=0.015,
    )

    result = run_navigation_from_perception(
        trial=3,
        perception=perception,
        instrument=make_instrument(),
        start_q=(
            make_start_configuration()
        ),
        goal_q=(
            make_goal_configuration()
        ),
        true_structures=(
            true_structure,
        ),
        sigma_multiplier=2.0,
        instrument_radius=0.006,
        proximal_length=0.10,
        planner_seed=1234,
    )

    assert (
        result.strategy
        == "generic_active"
    )

    assert (
        result.planner_result.method
        == "generic_active"
    )

    assert result.planning_success


def test_hidden_truth_can_reveal_unsafe_planner_result():
    """Hidden truth must independently expose perception-induced danger.

    The planner is deliberately given an anatomical estimate far from the
    instrument path.  Ground truth is instead placed directly on the true
    trajectory.  Planning should therefore succeed according to perception
    while independent evaluation detects the unsafe physical result.
    """

    instrument = make_instrument()

    start_q = (
        make_start_configuration()
    )

    goal_q = (
        make_goal_configuration()
    )

    midpoint_q = (
        0.5
        * (
            start_q
            + goal_q
        )
    )

    _, midpoint_tip = (
        instrument.shaft_segment(
            midpoint_q,
            proximal_length=0.10,
        )
    )

    true_structure = SphericalStructure(
        centre=midpoint_tip,
        physical_radius=0.035,
        safety_margin=0.015,
    )

    perception = make_strategy_perception(
        strategy=(
            PerceptionStrategy.FIXED
        ),
        estimated_centre=np.array(
            [0.70, 0.70, 0.70],
            dtype=float,
        ),
        sigma=0.002,
        physical_radius=0.035,
        base_safety_margin=0.015,
    )

    result = run_navigation_from_perception(
        trial=0,
        perception=perception,
        instrument=instrument,
        start_q=start_q,
        goal_q=goal_q,
        true_structures=(
            true_structure,
        ),
        sigma_multiplier=2.0,
        instrument_radius=0.006,
        proximal_length=0.10,
        planner_seed=5678,
    )

    assert result.planning_success

    assert (
        result.collision_against_truth
        or
        result.safety_violation_against_truth
    )

    assert (
        result.minimum_true_safety_clearance
        < 0.0
    )