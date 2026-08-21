"""Ablation study for task-aware active perception."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry.workspace import SphericalStructure
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
    CandidateViewpoint,
    generate_candidate_viewpoints,
)


@dataclass(frozen=True)
class AblationVariant:
    """Configuration for one ablation variant."""

    name: str

    use_alignment: bool

    use_uncertainty: bool


@dataclass(frozen=True)
class AblationResult:
    """Result for one ablation variant."""

    variant: str

    trial_count: int

    mean_selected_uncertainty: float

    mean_task_alignment: float

    mean_generic_score: float

    mean_final_score: float

    selection_difference_rate_percent: float


@dataclass(frozen=True)
class AblationStudy:
    """Complete ablation study."""

    results: tuple[
        AblationResult,
        ...,
    ]


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
    """Create the safety-critical structure."""

    return SphericalStructure(
        centre=np.array(
            [0.24, 0.0, 0.0],
            dtype=float,
        ),
        physical_radius=0.025,
        safety_margin=0.015,
    )


def make_task(
    target: SphericalStructure,
) -> SurgicalTask:
    """Create the task used for all ablation variants."""

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
            target.centre,
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


def make_initial_pose(
    target: SphericalStructure,
) -> CameraPose:
    """Create the common starting camera pose."""

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


def make_variants() -> tuple[
    AblationVariant,
    ...,
]:
    """Return the four ablation conditions."""

    return (
        AblationVariant(
            name="Generic baseline",
            use_alignment=False,
            use_uncertainty=False,
        ),
        AblationVariant(
            name="Alignment-only",
            use_alignment=True,
            use_uncertainty=False,
        ),
        AblationVariant(
            name="Uncertainty-only",
            use_alignment=False,
            use_uncertainty=True,
        ),
        AblationVariant(
            name="Full task-aware",
            use_alignment=True,
            use_uncertainty=True,
        ),
    )


def build_scorer(
    model: ViewpointObservationModel,
    task: SurgicalTask,
    variant: AblationVariant,
) -> TaskAwareViewpointScorer:
    """Build an isolated scorer for one ablation variant."""

    generic_scorer = GenericViewpointScorer(
        observation_model=model
    )

    return TaskAwareViewpointScorer(
        generic_scorer=generic_scorer,
        task=task,
        task_config=TaskAwareScoringConfig(
            task_weight=2.0,
            alignment_weight=(
                1.0
                if variant.use_alignment
                else 0.0
            ),
            uncertainty_weight=(
                1.0
                if variant.use_uncertainty
                else 0.0
            ),
        ),
    )


def run_variant(
    variant: AblationVariant,
    trial_count: int = 100,
) -> AblationResult:
    """Run one ablation variant."""

    if trial_count <= 0:
        raise ValueError(
            "trial_count must be positive."
        )

    model = make_observation_model()

    target = make_target()

    task = make_task(
        target
    )

    initial_pose = make_initial_pose(
        target
    )

    scorer = build_scorer(
        model=model,
        task=task,
        variant=variant,
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre
    )

    baseline_scorer = GenericViewpointScorer(
        observation_model=model
    )

    baseline_scores = (
        baseline_scorer.score_candidates(
            current_pose=initial_pose,
            candidates=candidates,
            target=target,
        )
    )

    generic_best = max(
        baseline_scores,
        key=lambda item: item.score,
    )

    selected_uncertainties: list[
        float
    ] = []

    alignments: list[
        float
    ] = []

    generic_scores: list[
        float
    ] = []

    final_scores: list[
        float
    ] = []

    selection_differences = 0

    for _ in range(
        trial_count
    ):
        scores = scorer.score_candidates(
            current_pose=initial_pose,
            candidates=candidates,
            target=target,
        )

        selected = max(
            scores,
            key=lambda item: item.score,
        )

        selected_uncertainties.append(
            selected.task_uncertainty
        )

        alignments.append(
            selected.task_alignment
        )

        generic_scores.append(
            selected.generic_score.score
        )

        final_scores.append(
            selected.score
        )

        if not np.allclose(
            selected.candidate.pose.position,
            generic_best.candidate.pose.position,
            atol=1e-10,
            rtol=0.0,
        ):
            selection_differences += 1

    return AblationResult(
        variant=variant.name,
        trial_count=trial_count,
        mean_selected_uncertainty=float(
            np.mean(
                selected_uncertainties
            )
        ),
        mean_task_alignment=float(
            np.mean(
                alignments
            )
        ),
        mean_generic_score=float(
            np.mean(
                generic_scores
            )
        ),
        mean_final_score=float(
            np.mean(
                final_scores
            )
        ),
        selection_difference_rate_percent=(
            100.0
            * selection_differences
            / trial_count
        ),
    )


def run_ablation(
    trial_count: int = 100,
) -> AblationStudy:
    """Run the complete four-condition ablation."""

    results = tuple(
        run_variant(
            variant=variant,
            trial_count=trial_count,
        )
        for variant in make_variants()
    )

    return AblationStudy(
        results=results
    )


def print_summary(
    study: AblationStudy,
) -> None:
    """Print the ablation results."""

    print()
    print(
        "Task-Aware Active Perception Ablation"
    )
    print(
        "====================================="
    )

    for result in study.results:
        print()
        print(
            result.variant
        )
        print(
            "-" * len(result.variant)
        )

        print(
            f"Trials: {result.trial_count}"
        )

        print(
            "Mean selected task uncertainty: "
            f"{result.mean_selected_uncertainty:.6f} m"
        )

        print(
            "Mean task alignment: "
            f"{result.mean_task_alignment:.6f}"
        )

        print(
            "Mean generic score: "
            f"{result.mean_generic_score:.6f}"
        )

        print(
            "Mean final score: "
            f"{result.mean_final_score:.6f}"
        )

        print(
            "Selection difference from generic: "
            f"{result.selection_difference_rate_percent:.2f}%"
        )


def main() -> None:
    """Run the ablation study."""

    study = run_ablation(
        trial_count=100
    )

    print_summary(
        study
    )


if __name__ == "__main__":
    main()