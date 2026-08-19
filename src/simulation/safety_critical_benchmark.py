"""Safety-critical planning benchmark under perception uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from src.geometry.workspace import SphericalStructure
from src.robotics.instrument import SurgicalInstrument
from src.robotics.planner import plan_rrt, path_cost
from src.robotics.safety import evaluate_instrument_safety


@dataclass(frozen=True)
class StrategyConfig:
    """Perception uncertainty used by one experimental strategy."""

    name: str
    predicted_sigma: float

    def __post_init__(self) -> None:
        if self.predicted_sigma <= 0.0:
            raise ValueError(
                "predicted_sigma must be positive."
            )


@dataclass(frozen=True)
class PlanningBenchmarkConfig:
    """Configuration for the safety-critical benchmark."""

    trial_count: int = 100

    random_seed: int = 20260819

    instrument_radius: float = 0.005

    proximal_length: float = 0.10

    max_iterations: int = 2500

    step_size: float = 0.08

    goal_bias: float = 0.20

    edge_resolution: int = 15

    path_evaluation_resolution: int = 30


@dataclass(frozen=True)
class PathSafetyResult:
    """Ground-truth safety evaluation of one planned path."""

    evaluated: bool

    minimum_surface_clearance: float

    minimum_safety_clearance: float

    collision: bool

    safety_margin_violation: bool

    unsafe: bool


@dataclass(frozen=True)
class StrategySummary:
    """Summary statistics for one perception strategy."""

    strategy: str

    predicted_sigma: float

    trials: int

    planning_success_rate_percent: float

    ground_truth_safe_rate_percent: float

    collision_rate_percent: float

    safety_margin_violation_rate_percent: float

    unsafe_path_rate_percent: float

    mean_minimum_surface_clearance: float

    mean_minimum_safety_clearance: float

    mean_path_cost: float

    mean_planning_iterations: float

    mean_planning_time_ms: float


@dataclass(frozen=True)
class SafetyBenchmarkResult:
    """Complete three-strategy safety comparison."""

    fixed_view: StrategySummary

    generic_active_perception: StrategySummary

    task_aware_active_perception: StrategySummary


def make_instrument() -> SurgicalInstrument:
    """Create the simplified RCM surgical instrument."""

    return SurgicalInstrument(
        rcm_position=np.array(
            [0.0, 0.0, 0.0],
            dtype=float,
        )
    )


def make_true_structure() -> SphericalStructure:
    """Create the ground-truth safety-critical anatomy."""

    return SphericalStructure(
        centre=np.array(
            [0.2175, 0.0, 0.0329],
            dtype=float,
        ),
        physical_radius=0.018,
        safety_margin=0.012,
    )


def make_start_configuration() -> np.ndarray:
    """Create a safe starting configuration."""

    return np.array(
        [
            np.deg2rad(-25.0),
            np.deg2rad(8.6),
            0.22,
            0.0,
        ],
        dtype=float,
    )


def make_goal_configuration() -> np.ndarray:
    """Create a safe goal configuration."""

    return np.array(
        [
            np.deg2rad(25.0),
            np.deg2rad(8.6),
            0.22,
            0.0,
        ],
        dtype=float,
    )


def estimate_structure(
    true_structure: SphericalStructure,
    predicted_sigma: float,
    rng: np.random.Generator,
) -> SphericalStructure:
    """Generate a noisy estimated structure from perception uncertainty."""

    localisation_error = rng.normal(
        loc=0.0,
        scale=predicted_sigma,
        size=3,
    )

    estimated_centre = (
        true_structure.centre
        + localisation_error
    )

    return SphericalStructure(
        centre=estimated_centre,
        physical_radius=(
            true_structure.physical_radius
        ),
        safety_margin=(
            true_structure.safety_margin
        ),
    )


def evaluate_path_against_truth(
    instrument: SurgicalInstrument,
    path: np.ndarray,
    true_structures: tuple[
        SphericalStructure,
        ...,
    ],
    instrument_radius: float,
    proximal_length: float,
    resolution: int,
) -> PathSafetyResult:
    """Evaluate a planned path against ground-truth anatomy."""

    path = np.asarray(
        path,
        dtype=float,
    )

    if (
        path.ndim != 2
        or path.shape[1] != 4
    ):
        raise ValueError(
            "path must have shape (N, 4)."
        )

    if len(path) < 2:
        raise ValueError(
            "path must contain at least two configurations."
        )

    if resolution < 2:
        raise ValueError(
            "resolution must be at least 2."
        )

    minimum_surface_clearance = (
        float("inf")
    )

    minimum_safety_clearance = (
        float("inf")
    )

    collision = False

    safety_margin_violation = False

    for index in range(
        len(path) - 1
    ):
        q_start = path[index]
        q_end = path[index + 1]

        samples = np.linspace(
            q_start,
            q_end,
            num=resolution,
            dtype=float,
        )

        for q in samples:
            proximal_point, tip_point = (
                instrument.shaft_segment(
                    q,
                    proximal_length=(
                        proximal_length
                    ),
                )
            )

            evaluation = (
                evaluate_instrument_safety(
                    shaft_start=proximal_point,
                    shaft_end=tip_point,
                    structures=true_structures,
                    instrument_radius=(
                        instrument_radius
                    ),
                )
            )

            minimum_surface_clearance = min(
                minimum_surface_clearance,
                evaluation.minimum_surface_clearance,
            )

            minimum_safety_clearance = min(
                minimum_safety_clearance,
                evaluation.minimum_safety_clearance,
            )

            collision = (
                collision
                or evaluation.collision
            )

            safety_margin_violation = (
                safety_margin_violation
                or evaluation.safety_margin_violation
            )

    unsafe = (
        collision
        or safety_margin_violation
    )

    return PathSafetyResult(
        evaluated=True,
        minimum_surface_clearance=float(
            minimum_surface_clearance
        ),
        minimum_safety_clearance=float(
            minimum_safety_clearance
        ),
        collision=bool(
            collision
        ),
        safety_margin_violation=bool(
            safety_margin_violation
        ),
        unsafe=bool(
            unsafe
        ),
    )


def run_strategy(
    strategy: StrategyConfig,
    config: PlanningBenchmarkConfig,
) -> StrategySummary:
    """Run the matched planning experiment for one strategy."""

    instrument = make_instrument()

    true_structure = make_true_structure()

    true_structures = (
        true_structure,
    )

    start_q = (
        make_start_configuration()
    )

    goal_q = (
        make_goal_configuration()
    )

    planning_successes = 0

    safe_paths = 0

    collisions = 0

    safety_violations = 0

    unsafe_paths = 0

    surface_clearances: list[float] = []

    safety_clearances: list[float] = []

    path_costs: list[float] = []

    planning_iterations: list[float] = []

    planning_times: list[float] = []

    for trial in range(
        config.trial_count
    ):
        trial_seed = (
            config.random_seed
            + trial
        )

        rng = np.random.default_rng(
            trial_seed
        )

        estimated_structure = (
            estimate_structure(
                true_structure=true_structure,
                predicted_sigma=(
                    strategy.predicted_sigma
                ),
                rng=rng,
            )
        )

        planning_seed = (
            10000
            + trial
        )

        start_time = time.perf_counter()

        planning_result = plan_rrt(
            instrument=instrument,
            start_q=start_q,
            goal_q=goal_q,
            structures=(
                estimated_structure,
            ),
            instrument_radius=(
                config.instrument_radius
            ),
            proximal_length=(
                config.proximal_length
            ),
            max_iterations=(
                config.max_iterations
            ),
            step_size=(
                config.step_size
            ),
            goal_bias=(
                config.goal_bias
            ),
            edge_resolution=(
                config.edge_resolution
            ),
            seed=planning_seed,
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        planning_times.append(
            elapsed
            * 1000.0
        )

        planning_iterations.append(
            float(
                planning_result.iterations
            )
        )

        if not planning_result.success:
            continue

        planning_successes += 1

        path_costs.append(
            path_cost(
                planning_result.path
            )
        )

        safety = (
            evaluate_path_against_truth(
                instrument=instrument,
                path=planning_result.path,
                true_structures=true_structures,
                instrument_radius=(
                    config.instrument_radius
                ),
                proximal_length=(
                    config.proximal_length
                ),
                resolution=(
                    config.path_evaluation_resolution
                ),
            )
        )

        surface_clearances.append(
            safety.minimum_surface_clearance
        )

        safety_clearances.append(
            safety.minimum_safety_clearance
        )

        if safety.collision:
            collisions += 1

        if safety.safety_margin_violation:
            safety_violations += 1

        if safety.unsafe:
            unsafe_paths += 1
        else:
            safe_paths += 1

    planning_success_rate = (
        100.0
        * planning_successes
        / config.trial_count
    )

    if planning_successes == 0:
        safe_rate = 0.0
        collision_rate = 0.0
        violation_rate = 0.0
        unsafe_rate = 0.0
        mean_surface_clearance = (
            float("nan")
        )
        mean_safety_clearance = (
            float("nan")
        )
        mean_path_cost = float("nan")
        mean_iterations = float(
            np.mean(
                planning_iterations
            )
        )
    else:
        safe_rate = (
            100.0
            * safe_paths
            / planning_successes
        )

        collision_rate = (
            100.0
            * collisions
            / planning_successes
        )

        violation_rate = (
            100.0
            * safety_violations
            / planning_successes
        )

        unsafe_rate = (
            100.0
            * unsafe_paths
            / planning_successes
        )

        mean_surface_clearance = float(
            np.mean(
                surface_clearances
            )
        )

        mean_safety_clearance = float(
            np.mean(
                safety_clearances
            )
        )

        mean_path_cost = float(
            np.mean(
                path_costs
            )
        )

        mean_iterations = float(
            np.mean(
                planning_iterations
            )
        )

    return StrategySummary(
        strategy=strategy.name,
        predicted_sigma=(
            strategy.predicted_sigma
        ),
        trials=config.trial_count,
        planning_success_rate_percent=float(
            planning_success_rate
        ),
        ground_truth_safe_rate_percent=float(
            safe_rate
        ),
        collision_rate_percent=float(
            collision_rate
        ),
        safety_margin_violation_rate_percent=float(
            violation_rate
        ),
        unsafe_path_rate_percent=float(
            unsafe_rate
        ),
        mean_minimum_surface_clearance=(
            mean_surface_clearance
        ),
        mean_minimum_safety_clearance=(
            mean_safety_clearance
        ),
        mean_path_cost=(
            mean_path_cost
        ),
        mean_planning_iterations=(
            mean_iterations
        ),
        mean_planning_time_ms=float(
            np.mean(
                planning_times
            )
        ),
    )


def run_benchmark(
    config: PlanningBenchmarkConfig | None = None,
) -> SafetyBenchmarkResult:
    """Run the three matched perception-to-planning strategies."""

    if config is None:
        config = PlanningBenchmarkConfig()

    strategies = (
        StrategyConfig(
            name="Fixed view",
            predicted_sigma=0.030000,
        ),
        StrategyConfig(
            name="Generic active perception",
            predicted_sigma=0.002667,
        ),
        StrategyConfig(
            name="Task-aware active perception",
            predicted_sigma=0.002000,
        ),
    )

    summaries = tuple(
        run_strategy(
            strategy=strategy,
            config=config,
        )
        for strategy in strategies
    )

    return SafetyBenchmarkResult(
        fixed_view=summaries[0],
        generic_active_perception=summaries[1],
        task_aware_active_perception=summaries[2],
    )


def print_summary(
    result: SafetyBenchmarkResult,
) -> None:
    """Print the safety benchmark results."""

    print()
    print(
        "Perception-to-Planning Safety Benchmark"
    )
    print(
        "========================================"
    )

    summaries = (
        result.fixed_view,
        result.generic_active_perception,
        result.task_aware_active_perception,
    )

    for summary in summaries:
        print()
        print(
            summary.strategy
        )
        print(
            "-" * len(summary.strategy)
        )

        print(
            f"Trials: {summary.trials}"
        )

        print(
            "Predicted sigma: "
            f"{summary.predicted_sigma:.6f} m"
        )

        print(
            "Planning success rate: "
            f"{summary.planning_success_rate_percent:.2f}%"
        )

        print(
            "Ground-truth safe rate: "
            f"{summary.ground_truth_safe_rate_percent:.2f}%"
        )

        print(
            "Collision rate: "
            f"{summary.collision_rate_percent:.2f}%"
        )

        print(
            "Safety-margin violation rate: "
            f"{summary.safety_margin_violation_rate_percent:.2f}%"
        )

        print(
            "Unsafe path rate: "
            f"{summary.unsafe_path_rate_percent:.2f}%"
        )

        print(
            "Mean minimum surface clearance: "
            f"{summary.mean_minimum_surface_clearance:.6f} m"
        )

        print(
            "Mean minimum safety clearance: "
            f"{summary.mean_minimum_safety_clearance:.6f} m"
        )

        print(
            "Mean path cost: "
            f"{summary.mean_path_cost:.6f}"
        )

        print(
            "Mean planning iterations: "
            f"{summary.mean_planning_iterations:.2f}"
        )

        print(
            "Mean planning time: "
            f"{summary.mean_planning_time_ms:.3f} ms"
        )


def main() -> None:
    """Run the benchmark."""

    result = run_benchmark()

    print_summary(
        result
    )


if __name__ == "__main__":
    main()