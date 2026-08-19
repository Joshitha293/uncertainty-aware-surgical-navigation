"""Benchmark fixed-view and generic active perception."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry.workspace import SphericalStructure
from src.perception.active_perception import (
    GenericActivePerception,
)
from src.perception.camera import (
    CameraIntrinsics,
    CameraPose,
    SurgicalCamera,
    look_at_rotation,
)
from src.perception.closed_loop import (
    ClosedLoopActivePerception,
)
from src.perception.observation import (
    ObservationModelConfig,
    ViewpointObservationModel,
)
from src.perception.viewpoint_scoring import (
    GenericViewpointScorer,
    ViewpointScoringConfig,
)
from src.perception.viewpoints import (
    generate_candidate_viewpoints,
    viewpoint_displacement,
)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for the matched perception benchmark."""

    trial_count: int = 100

    random_seed: int = 20260819

    movement_weight: float = 0.05

    uncertainty_weight: float = 1.0

    occlusion_penalty: float = 2.0

    invisibility_penalty: float = 4.0


@dataclass(frozen=True)
class BenchmarkSummary:
    """Summary statistics for one perception strategy."""

    strategy: str

    trials: int

    mean_localisation_error: float

    median_localisation_error: float

    mean_predicted_sigma: float

    mean_movement: float

    occlusion_rate: float

    visibility_rate: float


@dataclass(frozen=True)
class BenchmarkComparison:
    """Comparison between fixed-view and active perception."""

    fixed: BenchmarkSummary

    active: BenchmarkSummary

    localisation_error_reduction_percent: float

    sigma_reduction_percent: float

    occlusion_reduction_percentage_points: float


def make_camera() -> SurgicalCamera:
    """Create the simulated surgical camera."""

    return SurgicalCamera(
        CameraIntrinsics(
            horizontal_fov=np.deg2rad(70.0),
            vertical_fov=np.deg2rad(55.0),
            near_distance=0.02,
            far_distance=0.60,
        )
    )


def make_observation_model() -> ViewpointObservationModel:
    """Create the viewpoint-dependent observation model."""

    return ViewpointObservationModel(
        camera=make_camera(),
        config=ObservationModelConfig(
            base_sigma=0.002,
            reference_distance=0.15,
            distance_weight=1.0,
            angle_weight=1.0,
            invisible_sigma=0.050,
            occluded_sigma=0.030,
        ),
    )


def make_target() -> SphericalStructure:
    """Create the safety-critical anatomical structure."""

    return SphericalStructure(
        centre=np.array(
            [0.24, 0.00, 0.00],
            dtype=float,
        ),
        physical_radius=0.025,
        safety_margin=0.015,
    )


def make_occluder() -> SphericalStructure:
    """Create an anatomical structure blocking the initial view."""

    return SphericalStructure(
        centre=np.array(
            [0.12, 0.00, 0.00],
            dtype=float,
        ),
        physical_radius=0.035,
        safety_margin=0.010,
    )


def make_initial_pose(
    target: SphericalStructure,
) -> CameraPose:
    """Create the fixed starting camera pose."""

    position = np.array(
        [0.00, 0.00, 0.00],
        dtype=float,
    )

    return CameraPose(
        position=position,
        rotation=look_at_rotation(
            position,
            target.centre,
        ),
    )


def make_active_controller(
    model: ViewpointObservationModel,
    config: BenchmarkConfig,
) -> GenericActivePerception:
    """Create the generic active-perception controller."""

    scorer = GenericViewpointScorer(
        observation_model=model,
        config=ViewpointScoringConfig(
            uncertainty_weight=(
                config.uncertainty_weight
            ),
            movement_weight=(
                config.movement_weight
            ),
            occlusion_penalty=(
                config.occlusion_penalty
            ),
            invisibility_penalty=(
                config.invisibility_penalty
            ),
        ),
    )

    return GenericActivePerception(
        observation_model=model,
        scorer=scorer,
    )


def run_fixed_view_benchmark(
    model: ViewpointObservationModel,
    target: SphericalStructure,
    initial_pose: CameraPose,
    occluders: tuple[
        SphericalStructure,
        ...,
    ],
    config: BenchmarkConfig,
) -> BenchmarkSummary:
    """Evaluate repeated observations from one fixed viewpoint."""

    errors: list[float] = []

    sigmas: list[float] = []

    movements: list[float] = []

    occluded_count = 0

    visible_count = 0

    initial_quality = (
        model.observation_quality(
            camera_pose=initial_pose,
            structure=target,
            occluders=occluders,
        )
    )

    for trial in range(
        config.trial_count
    ):
        seed = (
            config.random_seed
            + trial
        )

        observation = (
            model.observe_structure(
                camera_pose=initial_pose,
                structure=target,
                rng=np.random.default_rng(
                    seed
                ),
                occluders=occluders,
            )
        )

        errors.append(
            float(
                observation.localisation_error
            )
        )

        sigmas.append(
            float(
                initial_quality.localisation_sigma
            )
        )

        movements.append(
            0.0
        )

        if initial_quality.occluded:
            occluded_count += 1

        if initial_quality.visible:
            visible_count += 1

    return BenchmarkSummary(
        strategy="Fixed view",
        trials=config.trial_count,
        mean_localisation_error=float(
            np.mean(errors)
        ),
        median_localisation_error=float(
            np.median(errors)
        ),
        mean_predicted_sigma=float(
            np.mean(sigmas)
        ),
        mean_movement=float(
            np.mean(movements)
        ),
        occlusion_rate=(
            100.0
            * occluded_count
            / config.trial_count
        ),
        visibility_rate=(
            100.0
            * visible_count
            / config.trial_count
        ),
    )


def run_active_view_benchmark(
    model: ViewpointObservationModel,
    target: SphericalStructure,
    initial_pose: CameraPose,
    occluders: tuple[
        SphericalStructure,
        ...,
    ],
    config: BenchmarkConfig,
) -> BenchmarkSummary:
    """Evaluate generic active perception over matched trials."""

    active_controller = (
        make_active_controller(
            model=model,
            config=config,
        )
    )

    closed_loop = ClosedLoopActivePerception(
        observation_model=model,
        active_perception=active_controller,
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre,
    )

    errors: list[float] = []

    sigmas: list[float] = []

    movements: list[float] = []

    occluded_count = 0

    visible_count = 0

    for trial in range(
        config.trial_count
    ):
        seed = (
            config.random_seed
            + trial
        )

        result = (
            closed_loop.execute_cycle(
                current_pose=initial_pose,
                candidates=candidates,
                target=target,
                rng=np.random.default_rng(
                    seed
                ),
                occluders=occluders,
            )
        )

        errors.append(
            float(
                result.localisation_error
            )
        )

        sigmas.append(
            float(
                result.post_observation_quality
                .localisation_sigma
            )
        )

        movement = viewpoint_displacement(
            current_pose=initial_pose,
            candidate_pose=result.selected_pose,
        )

        movements.append(
            float(movement)
        )

        if (
            result.post_observation_quality
            .occluded
        ):
            occluded_count += 1

        if (
            result.post_observation_quality
            .visible
        ):
            visible_count += 1

    return BenchmarkSummary(
        strategy="Generic active perception",
        trials=config.trial_count,
        mean_localisation_error=float(
            np.mean(errors)
        ),
        median_localisation_error=float(
            np.median(errors)
        ),
        mean_predicted_sigma=float(
            np.mean(sigmas)
        ),
        mean_movement=float(
            np.mean(movements)
        ),
        occlusion_rate=(
            100.0
            * occluded_count
            / config.trial_count
        ),
        visibility_rate=(
            100.0
            * visible_count
            / config.trial_count
        ),
    )


def compare_strategies(
    fixed: BenchmarkSummary,
    active: BenchmarkSummary,
) -> BenchmarkComparison:
    """Calculate comparative performance metrics."""

    if fixed.mean_localisation_error <= 0.0:
        raise ValueError(
            "Fixed-view localisation error must be positive."
        )

    if fixed.mean_predicted_sigma <= 0.0:
        raise ValueError(
            "Fixed-view predicted sigma must be positive."
        )

    error_reduction = (
        100.0
        * (
            fixed.mean_localisation_error
            - active.mean_localisation_error
        )
        / fixed.mean_localisation_error
    )

    sigma_reduction = (
        100.0
        * (
            fixed.mean_predicted_sigma
            - active.mean_predicted_sigma
        )
        / fixed.mean_predicted_sigma
    )

    occlusion_reduction = (
        fixed.occlusion_rate
        - active.occlusion_rate
    )

    return BenchmarkComparison(
        fixed=fixed,
        active=active,
        localisation_error_reduction_percent=float(
            error_reduction
        ),
        sigma_reduction_percent=float(
            sigma_reduction
        ),
        occlusion_reduction_percentage_points=float(
            occlusion_reduction
        ),
    )


def print_summary(
    comparison: BenchmarkComparison,
) -> None:
    """Print the benchmark results."""

    fixed = comparison.fixed
    active = comparison.active

    print()
    print(
        "Fixed View vs Generic Active Perception"
    )
    print(
        "======================================="
    )

    print()
    print("Fixed view")
    print("----------")

    print(
        f"Trials: "
        f"{fixed.trials}"
    )

    print(
        "Mean localisation error: "
        f"{fixed.mean_localisation_error:.6f} m"
    )

    print(
        "Median localisation error: "
        f"{fixed.median_localisation_error:.6f} m"
    )

    print(
        "Mean predicted sigma: "
        f"{fixed.mean_predicted_sigma:.6f} m"
    )

    print(
        "Mean camera movement: "
        f"{fixed.mean_movement:.6f} m"
    )

    print(
        "Occlusion rate: "
        f"{fixed.occlusion_rate:.1f}%"
    )

    print(
        "Visibility rate: "
        f"{fixed.visibility_rate:.1f}%"
    )

    print()
    print("Generic active perception")
    print("-------------------------")

    print(
        f"Trials: "
        f"{active.trials}"
    )

    print(
        "Mean localisation error: "
        f"{active.mean_localisation_error:.6f} m"
    )

    print(
        "Median localisation error: "
        f"{active.median_localisation_error:.6f} m"
    )

    print(
        "Mean predicted sigma: "
        f"{active.mean_predicted_sigma:.6f} m"
    )

    print(
        "Mean camera movement: "
        f"{active.mean_movement:.6f} m"
    )

    print(
        "Occlusion rate: "
        f"{active.occlusion_rate:.1f}%"
    )

    print(
        "Visibility rate: "
        f"{active.visibility_rate:.1f}%"
    )

    print()
    print("Comparative improvement")
    print("-----------------------")

    print(
        "Mean localisation-error reduction: "
        f"{comparison.localisation_error_reduction_percent:.2f}%"
    )

    print(
        "Predicted uncertainty reduction: "
        f"{comparison.sigma_reduction_percent:.2f}%"
    )

    print(
        "Occlusion-rate reduction: "
        f"{comparison.occlusion_reduction_percentage_points:.2f} "
        "percentage points"
    )


def main() -> None:
    """Run the matched benchmark."""

    config = BenchmarkConfig()

    model = make_observation_model()

    target = make_target()

    initial_pose = make_initial_pose(
        target
    )

    occluders = (
        make_occluder(),
    )

    fixed = run_fixed_view_benchmark(
        model=model,
        target=target,
        initial_pose=initial_pose,
        occluders=occluders,
        config=config,
    )

    active = run_active_view_benchmark(
        model=model,
        target=target,
        initial_pose=initial_pose,
        occluders=occluders,
        config=config,
    )

    comparison = compare_strategies(
        fixed=fixed,
        active=active,
    )

    print_summary(
        comparison
    )


if __name__ == "__main__":
    main()