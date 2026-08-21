"""Sensitivity analysis of viewpoint-dependent uncertainty heterogeneity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.perception.camera import (
    CameraIntrinsics,
    SurgicalCamera,
)
from src.perception.observation import (
    ObservationModelConfig,
    ViewpointObservationModel,
)
from src.perception.viewpoint_scoring import (
    GenericViewpointScorer,
)
from src.perception.viewpoints import (
    CandidateViewpoint,
    generate_candidate_viewpoints,
)
from src.simulation.normalised_ablation import (
    alignment_for_candidates,
    make_initial_pose,
    make_target,
    make_task,
    normalise_uncertainty,
)


@dataclass(frozen=True)
class HeterogeneityCondition:
    """Observation-model condition for one sensitivity experiment."""

    name: str

    distance_weight: float

    angle_weight: float


@dataclass(frozen=True)
class HeterogeneitySensitivityResult:
    """Result for one heterogeneity condition."""

    condition: str

    distance_weight: float

    angle_weight: float

    generic_candidate_index: int

    task_aware_candidate_index: int

    selection_changed: bool

    generic_selected_uncertainty_m: float

    task_aware_selected_uncertainty_m: float

    uncertainty_range_m: float

    uncertainty_coefficient_of_variation: float

    task_aware_alignment: float


@dataclass(frozen=True)
class HeterogeneitySensitivityStudy:
    """Complete uncertainty-heterogeneity sensitivity study."""

    results: tuple[
        HeterogeneitySensitivityResult,
        ...
    ]


def make_camera() -> SurgicalCamera:
    """Create the common simulated camera."""

    return SurgicalCamera(
        CameraIntrinsics(
            horizontal_fov=np.deg2rad(70.0),
            vertical_fov=np.deg2rad(55.0),
            near_distance=0.02,
            far_distance=0.60,
        )
    )


def make_conditions() -> tuple[
    HeterogeneityCondition,
    ...
]:
    """Return controlled observation-model conditions."""

    return (
        HeterogeneityCondition(
            name="Low heterogeneity",
            distance_weight=0.25,
            angle_weight=0.25,
        ),
        HeterogeneityCondition(
            name="Moderate heterogeneity",
            distance_weight=0.50,
            angle_weight=0.50,
        ),
        HeterogeneityCondition(
            name="Distance-dominant",
            distance_weight=2.00,
            angle_weight=0.50,
        ),
        HeterogeneityCondition(
            name="Angle-dominant",
            distance_weight=0.50,
            angle_weight=2.00,
        ),
        HeterogeneityCondition(
            name="Strong combined",
            distance_weight=2.00,
            angle_weight=2.00,
        ),
    )


def make_observation_model(
    condition: HeterogeneityCondition,
) -> ViewpointObservationModel:
    """Create an observation model for one condition."""

    if (
        condition.distance_weight < 0.0
        or condition.angle_weight < 0.0
    ):
        raise ValueError(
            "Observation-model weights must be non-negative."
        )

    return ViewpointObservationModel(
        camera=make_camera(),
        config=ObservationModelConfig(
            base_sigma=0.002,
            reference_distance=0.15,
            distance_weight=condition.distance_weight,
            angle_weight=condition.angle_weight,
            invisible_sigma=0.050,
            occluded_sigma=0.030,
        ),
    )


def evaluate_condition(
    condition: HeterogeneityCondition,
) -> HeterogeneitySensitivityResult:
    """Evaluate generic and task-aware selection."""

    model = make_observation_model(
        condition
    )

    target = make_target()

    task = make_task(
        target
    )

    initial_pose = make_initial_pose(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre
    )

    generic_scorer = GenericViewpointScorer(
        observation_model=model
    )

    generic_scores = (
        generic_scorer.score_candidates(
            current_pose=initial_pose,
            candidates=candidates,
            target=target,
        )
    )

    generic_values = np.asarray(
        [
            score.score
            for score in generic_scores
        ],
        dtype=float,
    )

    uncertainties = np.asarray(
        [
            model.observation_quality(
                camera_pose=candidate.pose,
                structure=target,
                occluders=(),
            ).localisation_sigma
            for candidate in candidates
        ],
        dtype=float,
    )

    if not np.all(
        np.isfinite(uncertainties)
    ):
        raise ValueError(
            "Candidate uncertainties must be finite."
        )

    uncertainty_information = (
        normalise_uncertainty(
            uncertainties
        )
    )

    alignments = alignment_for_candidates(
        task=task,
        candidates=candidates,
    )

    generic_index = int(
        np.argmax(
            generic_values
        )
    )

    task_aware_values = (
        generic_values
        + alignments
        + uncertainty_information
    )

    task_aware_index = int(
        np.argmax(
            task_aware_values
        )
    )

    mean_uncertainty = float(
        np.mean(
            uncertainties
        )
    )

    uncertainty_std = float(
        np.std(
            uncertainties
        )
    )

    coefficient_of_variation = (
        uncertainty_std
        / mean_uncertainty
        if mean_uncertainty > 1e-12
        else 0.0
    )

    return HeterogeneitySensitivityResult(
        condition=condition.name,
        distance_weight=(
            condition.distance_weight
        ),
        angle_weight=(
            condition.angle_weight
        ),
        generic_candidate_index=(
            generic_index
        ),
        task_aware_candidate_index=(
            task_aware_index
        ),
        selection_changed=(
            generic_index
            != task_aware_index
        ),
        generic_selected_uncertainty_m=float(
            uncertainties[
                generic_index
            ]
        ),
        task_aware_selected_uncertainty_m=float(
            uncertainties[
                task_aware_index
            ]
        ),
        uncertainty_range_m=float(
            np.max(uncertainties)
            - np.min(uncertainties)
        ),
        uncertainty_coefficient_of_variation=(
            float(
                coefficient_of_variation
            )
        ),
        task_aware_alignment=float(
            alignments[
                task_aware_index
            ]
        ),
    )


def run_sensitivity() -> HeterogeneitySensitivityStudy:
    """Run all heterogeneity conditions."""

    return HeterogeneitySensitivityStudy(
        results=tuple(
            evaluate_condition(
                condition
            )
            for condition in make_conditions()
        )
    )


def print_summary(
    study: HeterogeneitySensitivityStudy,
) -> None:
    """Print the sensitivity results."""

    print()
    print(
        "Uncertainty-Heterogeneity Sensitivity Analysis"
    )
    print(
        "=============================================="
    )

    print()

    print(
        "Condition | "
        "Distance | "
        "Angle | "
        "Generic candidate | "
        "Task-aware candidate | "
        "Changed | "
        "Generic σ (mm) | "
        "Task-aware σ (mm) | "
        "σ range (mm) | "
        "CV | "
        "Alignment"
    )

    print(
        "-" * 150
    )

    for result in study.results:
        print(
            f"{result.condition:20s} | "
            f"{result.distance_weight:8.2f} | "
            f"{result.angle_weight:5.2f} | "
            f"{result.generic_candidate_index:17d} | "
            f"{result.task_aware_candidate_index:20d} | "
            f"{str(result.selection_changed):7s} | "
            f"{result.generic_selected_uncertainty_m * 1000:13.3f} | "
            f"{result.task_aware_selected_uncertainty_m * 1000:15.3f} | "
            f"{result.uncertainty_range_m * 1000:11.3f} | "
            f"{result.uncertainty_coefficient_of_variation:5.3f} | "
            f"{result.task_aware_alignment:.6f}"
        )


def main() -> None:
    """Run and print the heterogeneity sensitivity analysis."""

    study = run_sensitivity()

    print_summary(
        study
    )


if __name__ == "__main__":
    main()