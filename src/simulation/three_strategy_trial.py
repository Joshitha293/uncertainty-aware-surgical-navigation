"""End-to-end comparison of three surgical perception strategies.

This experiment connects:

    Fixed View
    Generic Active Perception
    Task-Aware Active Perception

to the same uncertainty-aware navigation pipeline.

All strategies use:

- the same hidden ground-truth anatomy;
- the same start and goal configurations;
- the same surgical instrument;
- matched perception random seeds;
- the same uncertainty inflation;
- the same RRT planner seed.

Ground truth is used only for simulated observation generation and final
safety evaluation. The planner operates on noisy perceived anatomy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry.workspace import SphericalStructure
from src.perception.camera import (
    CameraPose,
)
from src.perception.observation import (
    ViewpointObservationModel,
)
from src.perception.task_relevance import (
    SurgicalTask,
)
from src.perception.viewpoints import (
    CandidateViewpoint,
    generate_candidate_viewpoints,
)
from src.robotics.instrument import (
    SurgicalInstrument,
)
from src.simulation.statistical_benchmark import (
    make_goal_configuration,
    make_instrument,
    make_start_configuration,
    make_true_structures,
)
from src.simulation.task_aware_benchmark import (
    TaskAwareBenchmarkConfig,
    make_generic_controller,
    make_initial_pose,
    make_observation_model,
    make_task_aware_controller,
)
from src.simulation.three_strategy_navigation import (
    StrategyNavigationResult,
    run_navigation_from_perception,
)
from src.simulation.three_strategy_perception import (
    PerceptionStrategy,
    StrategyPerceptionResult,
    run_fixed_perception,
    run_generic_active_perception,
    run_task_aware_active_perception,
)


@dataclass(frozen=True)
class ThreeStrategyTrialConfig:
    """Configuration for one matched three-strategy trial."""

    trial: int = 0

    perception_seed: int = 20260823
    planner_seed: int = 2000

    sigma_multiplier: float = 2.0

    instrument_radius: float = 0.006
    proximal_length: float = 0.10

    task_weight: float = 2.0
    alignment_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.trial < 0:
            raise ValueError(
                "trial must be non-negative."
            )

        if not np.isfinite(
            self.sigma_multiplier
        ):
            raise ValueError(
                "sigma_multiplier must be finite."
            )

        if self.sigma_multiplier < 0.0:
            raise ValueError(
                "sigma_multiplier must be non-negative."
            )

        if self.instrument_radius <= 0.0:
            raise ValueError(
                "instrument_radius must be positive."
            )

        if self.proximal_length <= 0.0:
            raise ValueError(
                "proximal_length must be positive."
            )

        if self.task_weight < 0.0:
            raise ValueError(
                "task_weight must be non-negative."
            )

        if self.alignment_weight < 0.0:
            raise ValueError(
                "alignment_weight must be non-negative."
            )


@dataclass(frozen=True)
class ThreeStrategyTrialInputs:
    """Shared physical and perception inputs for the matched trial."""

    instrument: SurgicalInstrument

    start_q: np.ndarray
    goal_q: np.ndarray

    true_structures: tuple[
        SphericalStructure,
        ...,
    ]

    target: SphericalStructure

    observation_model: ViewpointObservationModel
    initial_pose: CameraPose

    candidates: tuple[
        CandidateViewpoint,
        ...,
    ]

    task: SurgicalTask


@dataclass(frozen=True)
class ThreeStrategyTrialResult:
    """Complete matched result for the three perception strategies."""

    config: ThreeStrategyTrialConfig

    fixed: StrategyNavigationResult

    generic_active: StrategyNavigationResult

    task_aware_active: StrategyNavigationResult

    @property
    def results(
        self,
    ) -> tuple[
        StrategyNavigationResult,
        ...,
    ]:
        """Return results in experimental comparison order."""

        return (
            self.fixed,
            self.generic_active,
            self.task_aware_active,
        )


def make_trial_task(
    *,
    instrument: SurgicalInstrument,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    true_structures: tuple[
        SphericalStructure,
        ...,
    ],
    samples: int = 9,
) -> SurgicalTask:
    """Create task information from the actual navigation problem.

    The task trajectory is derived from Cartesian instrument-tip positions
    between the validated start and goal configurations.

    Ground-truth structure centres define safety-critical regions for the
    simulated task-aware scoring stage.
    """

    if samples < 2:
        raise ValueError(
            "samples must be at least 2."
        )

    if len(true_structures) == 0:
        raise ValueError(
            "true_structures must not be empty."
        )

    start_q = np.asarray(
        start_q,
        dtype=float,
    )

    goal_q = np.asarray(
        goal_q,
        dtype=float,
    )

    interpolation = np.linspace(
        0.0,
        1.0,
        samples,
    )

    configurations = tuple(
        (
            (1.0 - value) * start_q
            + value * goal_q
        )
        for value in interpolation
    )

    trajectory = np.asarray(
        [
            instrument.forward_position(
                configuration
            )
            for configuration in configurations
        ],
        dtype=float,
    )

    critical_points = np.asarray(
        [
            structure.centre
            for structure in true_structures
        ],
        dtype=float,
    )

    return SurgicalTask(
        trajectory=trajectory,
        safety_critical_points=critical_points,
    )


def build_trial_inputs() -> ThreeStrategyTrialInputs:
    """Build the common experimental state used by all strategies."""

    instrument = make_instrument()

    start_q = make_start_configuration()

    goal_q = make_goal_configuration()

    true_structures = make_true_structures()

    if len(true_structures) == 0:
        raise RuntimeError(
            "Validated anatomy unexpectedly contains no structures."
        )

    # Primary safety-critical anatomical target.
    target = true_structures[0]

    observation_model = (
        make_observation_model()
    )

    initial_pose = make_initial_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre
    )

    task = make_trial_task(
        instrument=instrument,
        start_q=start_q,
        goal_q=goal_q,
        true_structures=true_structures,
    )

    return ThreeStrategyTrialInputs(
        instrument=instrument,
        start_q=start_q,
        goal_q=goal_q,
        true_structures=true_structures,
        target=target,
        observation_model=observation_model,
        initial_pose=initial_pose,
        candidates=candidates,
        task=task,
    )


def run_three_strategy_perception(
    config: ThreeStrategyTrialConfig,
    inputs: ThreeStrategyTrialInputs | None = None,
) -> tuple[
    StrategyPerceptionResult,
    StrategyPerceptionResult,
    StrategyPerceptionResult,
]:
    """Run all three perception strategies under matched conditions."""

    if inputs is None:
        inputs = build_trial_inputs()

    generic_controller = (
        make_generic_controller(
            inputs.observation_model
        )
    )

    task_config = TaskAwareBenchmarkConfig(
        trial_count=1,
        random_seed=config.perception_seed,
        task_weight=config.task_weight,
        alignment_weight=(
            config.alignment_weight
        ),
    )

    task_aware_controller = (
        make_task_aware_controller(
            model=inputs.observation_model,
            task=inputs.task,
            config=task_config,
        )
    )

    fixed = run_fixed_perception(
        observation_model=(
            inputs.observation_model
        ),
        initial_pose=inputs.initial_pose,
        true_structures=(
            inputs.true_structures
        ),
        seed=config.perception_seed,
        occluders=(),
    )

    generic = (
        run_generic_active_perception(
            controller=generic_controller,
            observation_model=(
                inputs.observation_model
            ),
            initial_pose=inputs.initial_pose,
            candidates=inputs.candidates,
            target=inputs.target,
            true_structures=(
                inputs.true_structures
            ),
            seed=config.perception_seed,
            occluders=(),
        )
    )

    task_aware = (
        run_task_aware_active_perception(
            controller=task_aware_controller,
            observation_model=(
                inputs.observation_model
            ),
            initial_pose=inputs.initial_pose,
            candidates=inputs.candidates,
            target=inputs.target,
            task=inputs.task,
            true_structures=(
                inputs.true_structures
            ),
            seed=config.perception_seed,
            occluders=(),
        )
    )

    return (
        fixed,
        generic,
        task_aware,
    )


def run_three_strategy_trial(
    config: ThreeStrategyTrialConfig
    | None = None,
) -> ThreeStrategyTrialResult:
    """Run the complete matched perception-to-navigation experiment."""

    if config is None:
        config = ThreeStrategyTrialConfig()

    inputs = build_trial_inputs()

    (
        fixed_perception,
        generic_perception,
        task_aware_perception,
    ) = run_three_strategy_perception(
        config=config,
        inputs=inputs,
    )

    common_navigation_arguments = {
        "trial": config.trial,
        "instrument": inputs.instrument,
        "start_q": inputs.start_q,
        "goal_q": inputs.goal_q,
        "true_structures": (
            inputs.true_structures
        ),
        "sigma_multiplier": (
            config.sigma_multiplier
        ),
        "instrument_radius": (
            config.instrument_radius
        ),
        "proximal_length": (
            config.proximal_length
        ),
        "planner_seed": (
            config.planner_seed
        ),
    }

    fixed_navigation = (
        run_navigation_from_perception(
            perception=fixed_perception,
            **common_navigation_arguments,
        )
    )

    generic_navigation = (
        run_navigation_from_perception(
            perception=generic_perception,
            **common_navigation_arguments,
        )
    )

    task_aware_navigation = (
        run_navigation_from_perception(
            perception=task_aware_perception,
            **common_navigation_arguments,
        )
    )

    return ThreeStrategyTrialResult(
        config=config,
        fixed=fixed_navigation,
        generic_active=generic_navigation,
        task_aware_active=(
            task_aware_navigation
        ),
    )


def _format_clearance(
    value: float,
) -> str:
    """Format ground-truth safety clearance in millimetres."""

    if not np.isfinite(value):
        return str(value)

    return f"{value * 1000.0:.3f}"


def print_trial_summary(
    result: ThreeStrategyTrialResult,
) -> None:
    """Print the final end-to-end three-strategy comparison."""

    print()
    print(
        "End-to-End Three-Strategy Surgical Navigation Trial"
    )
    print(
        "=================================================="
    )

    print()
    print(
        f"Trial: {result.config.trial}"
    )
    print(
        "Perception seed: "
        f"{result.config.perception_seed}"
    )
    print(
        "Planner seed: "
        f"{result.config.planner_seed}"
    )
    print(
        "Uncertainty multiplier: "
        f"{result.config.sigma_multiplier:.2f}"
    )

    print()
    print(
        "Strategy | "
        "Error (mm) | "
        "Sigma (mm) | "
        "Camera move (mm) | "
        "Plan | "
        "True clearance (mm) | "
        "Collision | "
        "Safety violation"
    )

    print(
        "-" * 130
    )

    names = {
        PerceptionStrategy.FIXED.value: (
            "Fixed View"
        ),
        PerceptionStrategy.GENERIC_ACTIVE.value: (
            "Generic Active"
        ),
        PerceptionStrategy.TASK_AWARE_ACTIVE.value: (
            "Task-Aware Active"
        ),
    }

    for navigation in result.results:
        perception = navigation.perception

        print(
            f"{names[navigation.strategy]:17s} | "
            f"{perception.mean_localisation_error * 1000:10.3f} | "
            f"{perception.mean_predicted_sigma * 1000:10.3f} | "
            f"{perception.camera_movement * 1000:16.3f} | "
            f"{str(navigation.planning_success):5s} | "
            f"{_format_clearance(navigation.minimum_true_safety_clearance):19s} | "
            f"{str(navigation.collision_against_truth):9s} | "
            f"{navigation.safety_violation_against_truth}"
        )

    print()
    print(
        "Planner-facing anatomy was generated from noisy "
        "perception with uncertainty-inflated safety margins."
    )

    print(
        "Ground-truth anatomy was used independently for "
        "final collision and safety evaluation."
    )


def main() -> None:
    """Execute the final matched three-strategy experiment."""

    result = run_three_strategy_trial()

    print_trial_summary(
        result
    )


if __name__ == "__main__":
    main()