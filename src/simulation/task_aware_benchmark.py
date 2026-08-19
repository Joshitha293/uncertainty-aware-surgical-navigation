"""Matched benchmark for generic and task-aware active perception."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry.workspace import SphericalStructure
from src.perception.active_perception import GenericActivePerception
from src.perception.camera import (
    CameraIntrinsics,
    CameraPose,
    SurgicalCamera,
    look_at_rotation,
)
from src.perception.observation import (
    ObservationModelConfig,
    ViewpointObservationModel,
)
from src.perception.task_aware_active_perception import (
    TaskAwareActivePerception,
)
from src.perception.task_aware_scoring import (
    TaskAwareScoringConfig,
    TaskAwareViewpointScorer,
)
from src.perception.task_relevance import (
    SurgicalTask,
)
from src.perception.viewpoint_scoring import (
    GenericViewpointScorer,
)
from src.perception.viewpoints import (
    generate_candidate_viewpoints,
    viewpoint_displacement,
)


@dataclass(frozen=True)
class TaskAwareBenchmarkConfig:
    """Configuration for the matched benchmark."""

    trial_count: int = 100

    random_seed: int = 20260819

    task_weight: float = 2.0

    alignment_weight: float = 1.0


@dataclass(frozen=True)
class StrategySummary:
    """Summary statistics for one strategy."""

    strategy: str

    trials: int

    mean_localisation_error: float

    median_localisation_error: float

    mean_predicted_sigma: float

    mean_camera_movement: float

    mean_task_alignment: float

    mean_task_relevance: float


@dataclass(frozen=True)
class TaskAwareComparison:
    """Matched comparison between generic and task-aware perception."""

    generic: StrategySummary

    task_aware: StrategySummary

    localisation_error_reduction_percent: float

    predicted_sigma_reduction_percent: float

    task_alignment_change_percent: float

    selection_difference_rate_percent: float


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
            [0.24, 0.0, 0.0],
            dtype=float,
        ),
        physical_radius=0.025,
        safety_margin=0.015,
    )


def make_occluder() -> SphericalStructure:
    """Create an occluding anatomical structure."""

    return SphericalStructure(
        centre=np.array(
            [0.12, 0.0, 0.0],
            dtype=float,
        ),
        physical_radius=0.035,
        safety_margin=0.010,
    )


def make_initial_pose(
    target: SphericalStructure,
) -> CameraPose:
    """Create the initial camera pose."""

    position = np.array(
        [0.0, 0.0, 0.0],
        dtype=float,
    )

    return CameraPose(
        position=position,
        rotation=look_at_rotation(
            position,
            target.centre,
        ),
    )


def make_task(
    target: SphericalStructure,
) -> SurgicalTask:
    """Create a safety-critical surgical trajectory."""

    trajectory = np.array(
        [
            [0.08, -0.04, 0.0],
            [0.13, -0.01, 0.0],
            [0.18, 0.02, 0.0],
            [0.22, 0.04, 0.0],
        ],
        dtype=float,
    )

    critical_points = np.array(
        [
            target.centre
            + np.array(
                [0.0, -0.015, 0.0]
            ),
            target.centre
            + np.array(
                [0.0, 0.0, 0.0]
            ),
            target.centre
            + np.array(
                [0.0, 0.015, 0.0]
            ),
        ],
        dtype=float,
    )

    return SurgicalTask(
        trajectory=trajectory,
        safety_critical_points=critical_points,
    )


def make_generic_controller(
    model: ViewpointObservationModel,
) -> GenericActivePerception:
    """Create the generic active-perception controller."""

    scorer = GenericViewpointScorer(
        observation_model=model
    )

    return GenericActivePerception(
        observation_model=model,
        scorer=scorer,
    )


def make_task_aware_controller(
    model: ViewpointObservationModel,
    task: SurgicalTask,
    config: TaskAwareBenchmarkConfig,
) -> TaskAwareActivePerception:
    """Create the task-aware active-perception controller."""

    generic_scorer = GenericViewpointScorer(
        observation_model=model
    )

    task_scorer = TaskAwareViewpointScorer(
        generic_scorer=generic_scorer,
        task=task,
        task_config=TaskAwareScoringConfig(
            task_weight=config.task_weight,
            alignment_weight=config.alignment_weight,
        ),
    )

    return TaskAwareActivePerception(
        scorer=task_scorer
    )


def run_benchmark(
    config: TaskAwareBenchmarkConfig,
) -> TaskAwareComparison:
    """Run a matched generic versus task-aware benchmark."""

    model = make_observation_model()

    target = make_target()

    initial_pose = make_initial_pose(
        target
    )

    occluders = (
        make_occluder(),
    )

    task = make_task(
        target
    )

    generic_controller = (
        make_generic_controller(
            model
        )
    )

    task_aware_controller = (
        make_task_aware_controller(
            model=model,
            task=task,
            config=config,
        )
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre
    )

    generic_errors: list[float] = []

    generic_sigmas: list[float] = []

    generic_movements: list[float] = []

    generic_alignments: list[float] = []

    task_aware_errors: list[float] = []

    task_aware_sigmas: list[float] = []

    task_aware_movements: list[float] = []

    task_aware_alignments: list[float] = []

    task_relevances: list[float] = []

    selection_differences = 0

    for trial in range(
        config.trial_count
    ):
        seed = (
            config.random_seed
            + trial
        )

        generic_result = (
            generic_controller.select_viewpoint(
                current_pose=initial_pose,
                candidates=candidates,
                target=target,
                occluders=occluders,
            )
        )

        task_aware_result = (
            task_aware_controller.select_viewpoint(
                current_pose=initial_pose,
                candidates=candidates,
                target=target,
                occluders=occluders,
            )
        )

        generic_position = np.asarray(
            generic_result.selected_viewpoint
            .pose.position,
            dtype=float,
        )

        task_aware_position = np.asarray(
            task_aware_result.selected_position,
            dtype=float,
        )

        if not np.allclose(
            generic_position,
            task_aware_position,
            atol=1e-10,
            rtol=0.0,
        ):
            selection_differences += 1

        generic_quality = (
            model.observation_quality(
                camera_pose=(
                    generic_result
                    .selected_viewpoint
                    .pose
                ),
                structure=target,
                occluders=occluders,
            )
        )

        task_aware_quality = (
            model.observation_quality(
                camera_pose=(
                    task_aware_result
                    .selected_viewpoint
                    .pose
                ),
                structure=target,
                occluders=occluders,
            )
        )

        generic_observation = (
            model.observe_structure(
                camera_pose=(
                    generic_result
                    .selected_viewpoint
                    .pose
                ),
                structure=target,
                rng=np.random.default_rng(
                    seed
                ),
                occluders=occluders,
            )
        )

        task_aware_observation = (
            model.observe_structure(
                camera_pose=(
                    task_aware_result
                    .selected_viewpoint
                    .pose
                ),
                structure=target,
                rng=np.random.default_rng(
                    seed
                ),
                occluders=occluders,
            )
        )

        generic_errors.append(
            float(
                generic_observation
                .localisation_error
            )
        )

        generic_sigmas.append(
            float(
                generic_quality
                .localisation_sigma
            )
        )

        generic_movements.append(
            float(
                viewpoint_displacement(
                    current_pose=initial_pose,
                    candidate_pose=(
                        generic_result
                        .selected_viewpoint
                        .pose
                    ),
                )
            )
        )

        generic_alignments.append(
            float(
                task_aware_controller
                .scorer
                ._task_alignment(
                    generic_result
                    .selected_viewpoint
                    .pose
                )
            )
        )

        task_aware_errors.append(
            float(
                task_aware_observation
                .localisation_error
            )
        )

        task_aware_sigmas.append(
            float(
                task_aware_quality
                .localisation_sigma
            )
        )

        task_aware_movements.append(
            float(
                viewpoint_displacement(
                    current_pose=initial_pose,
                    candidate_pose=(
                        task_aware_result
                        .selected_viewpoint
                        .pose
                    ),
                )
            )
        )

        task_aware_alignments.append(
            float(
                task_aware_result
                .task_alignment
            )
        )

        task_relevances.append(
            float(
                task_aware_result
                .task_relevance
            )
        )

    generic_summary = StrategySummary(
        strategy="Generic active perception",
        trials=config.trial_count,
        mean_localisation_error=float(
            np.mean(
                generic_errors
            )
        ),
        median_localisation_error=float(
            np.median(
                generic_errors
            )
        ),
        mean_predicted_sigma=float(
            np.mean(
                generic_sigmas
            )
        ),
        mean_camera_movement=float(
            np.mean(
                generic_movements
            )
        ),
        mean_task_alignment=float(
            np.mean(
                generic_alignments
            )
        ),
        mean_task_relevance=float(
            np.mean(
                task_relevances
            )
        ),
    )

    task_aware_summary = StrategySummary(
        strategy="Task-aware active perception",
        trials=config.trial_count,
        mean_localisation_error=float(
            np.mean(
                task_aware_errors
            )
        ),
        median_localisation_error=float(
            np.median(
                task_aware_errors
            )
        ),
        mean_predicted_sigma=float(
            np.mean(
                task_aware_sigmas
            )
        ),
        mean_camera_movement=float(
            np.mean(
                task_aware_movements
            )
        ),
        mean_task_alignment=float(
            np.mean(
                task_aware_alignments
            )
        ),
        mean_task_relevance=float(
            np.mean(
                task_relevances
            )
        ),
    )

    if (
        generic_summary.mean_localisation_error
        <= 0.0
    ):
        raise ValueError(
            "Generic localisation error must be positive."
        )

    if (
        generic_summary.mean_predicted_sigma
        <= 0.0
    ):
        raise ValueError(
            "Generic predicted sigma must be positive."
        )

    localisation_reduction = (
        100.0
        * (
            generic_summary
            .mean_localisation_error
            - task_aware_summary
            .mean_localisation_error
        )
        / generic_summary
        .mean_localisation_error
    )

    sigma_reduction = (
        100.0
        * (
            generic_summary
            .mean_predicted_sigma
            - task_aware_summary
            .mean_predicted_sigma
        )
        / generic_summary
        .mean_predicted_sigma
    )

    generic_alignment = (
        generic_summary.mean_task_alignment
    )

    if generic_alignment <= 0.0:
        alignment_change = 0.0
    else:
        alignment_change = (
            100.0
            * (
                task_aware_summary
                .mean_task_alignment
                - generic_alignment
            )
            / generic_alignment
        )

    selection_difference_rate = (
        100.0
        * selection_differences
        / config.trial_count
    )

    return TaskAwareComparison(
        generic=generic_summary,
        task_aware=task_aware_summary,
        localisation_error_reduction_percent=float(
            localisation_reduction
        ),
        predicted_sigma_reduction_percent=float(
            sigma_reduction
        ),
        task_alignment_change_percent=float(
            alignment_change
        ),
        selection_difference_rate_percent=float(
            selection_difference_rate
        ),
    )


def print_summary(
    comparison: TaskAwareComparison,
) -> None:
    """Print benchmark results."""

    generic = comparison.generic
    task_aware = comparison.task_aware

    print()
    print(
        "Generic vs Task-Aware Active Perception"
    )
    print(
        "======================================="
    )

    print()
    print(
        "Generic active perception"
    )
    print(
        "-------------------------"
    )

    print(
        f"Trials: {generic.trials}"
    )

    print(
        "Mean localisation error: "
        f"{generic.mean_localisation_error:.6f} m"
    )

    print(
        "Median localisation error: "
        f"{generic.median_localisation_error:.6f} m"
    )

    print(
        "Mean predicted sigma: "
        f"{generic.mean_predicted_sigma:.6f} m"
    )

    print(
        "Mean camera movement: "
        f"{generic.mean_camera_movement:.6f} m"
    )

    print(
        "Mean task alignment: "
        f"{generic.mean_task_alignment:.6f}"
    )

    print()
    print(
        "Task-aware active perception"
    )
    print(
        "----------------------------"
    )

    print(
        f"Trials: {task_aware.trials}"
    )

    print(
        "Mean localisation error: "
        f"{task_aware.mean_localisation_error:.6f} m"
    )

    print(
        "Median localisation error: "
        f"{task_aware.median_localisation_error:.6f} m"
    )

    print(
        "Mean predicted sigma: "
        f"{task_aware.mean_predicted_sigma:.6f} m"
    )

    print(
        "Mean camera movement: "
        f"{task_aware.mean_camera_movement:.6f} m"
    )

    print(
        "Mean task alignment: "
        f"{task_aware.mean_task_alignment:.6f}"
    )

    print(
        "Mean task relevance: "
        f"{task_aware.mean_task_relevance:.6f}"
    )

    print()
    print(
        "Comparative results"
    )
    print(
        "-------------------"
    )

    print(
        "Localisation-error reduction: "
        f"{comparison.localisation_error_reduction_percent:.2f}%"
    )

    print(
        "Predicted-sigma reduction: "
        f"{comparison.predicted_sigma_reduction_percent:.2f}%"
    )

    print(
        "Task-alignment change: "
        f"{comparison.task_alignment_change_percent:.2f}%"
    )

    print(
        "Selection difference rate: "
        f"{comparison.selection_difference_rate_percent:.2f}%"
    )


def main() -> None:
    """Run the matched experiment."""

    comparison = run_benchmark(
        TaskAwareBenchmarkConfig()
    )

    print_summary(
        comparison
    )


if __name__ == "__main__":
    main()