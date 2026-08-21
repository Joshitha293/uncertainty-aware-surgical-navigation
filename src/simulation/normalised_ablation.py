"""Normalised ablation study for task-aware active perception."""

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
from src.perception.task_relevance import SurgicalTask
from src.perception.viewpoint_scoring import (
    GenericViewpointScorer,
)
from src.perception.viewpoints import (
    CandidateViewpoint,
    generate_candidate_viewpoints,
)


@dataclass(frozen=True)
class NormalisedAblationVariant:
    """Configuration for one normalised ablation condition."""

    name: str

    use_alignment: bool

    use_uncertainty: bool


@dataclass(frozen=True)
class NormalisedCandidateScore:
    """Normalised score for one candidate viewpoint."""

    candidate: CandidateViewpoint

    generic_score: float

    alignment: float

    uncertainty: float

    final_score: float


@dataclass(frozen=True)
class NormalisedAblationResult:
    """Result for one normalised ablation condition."""

    variant: str

    trial_count: int

    mean_selected_uncertainty: float

    mean_selected_alignment: float

    mean_generic_score: float

    mean_final_score: float

    selection_difference_rate_percent: float


@dataclass(frozen=True)
class NormalisedAblationStudy:
    """Complete normalised ablation study."""

    results: tuple[
        NormalisedAblationResult,
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
    """Create the common surgical task."""

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
    NormalisedAblationVariant,
    ...,
]:
    """Return the four normalised ablation conditions."""

    return (
        NormalisedAblationVariant(
            name="Generic baseline",
            use_alignment=False,
            use_uncertainty=False,
        ),
        NormalisedAblationVariant(
            name="Alignment-only",
            use_alignment=True,
            use_uncertainty=False,
        ),
        NormalisedAblationVariant(
            name="Uncertainty-only",
            use_alignment=False,
            use_uncertainty=True,
        ),
        NormalisedAblationVariant(
            name="Full task-aware",
            use_alignment=True,
            use_uncertainty=True,
        ),
    )


def candidate_uncertainties(
    model: ViewpointObservationModel,
    candidates: tuple[
        CandidateViewpoint,
        ...,
    ],
    target: SphericalStructure,
) -> np.ndarray:
    """Return viewpoint-dependent target uncertainty."""

    values = []

    for candidate in candidates:
        quality = model.observation_quality(
            camera_pose=candidate.pose,
            structure=target,
            occluders=(),
        )

        values.append(
            float(
                quality.localisation_sigma
            )
        )

    return np.asarray(
        values,
        dtype=float,
    )


def normalise_uncertainty(
    uncertainties: np.ndarray,
) -> np.ndarray:
    """Convert uncertainty into a [0, 1] information score.

    Lower uncertainty receives a higher score.
    """

    values = np.asarray(
        uncertainties,
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError(
            "uncertainties must be one-dimensional."
        )

    if len(values) == 0:
        raise ValueError(
            "uncertainties must not be empty."
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "uncertainties must be finite."
        )

    minimum = float(
        np.min(values)
    )

    maximum = float(
        np.max(values)
    )

    span = maximum - minimum

    if span <= 1e-12:
        return np.ones_like(
            values,
            dtype=float,
        )

    return (
        maximum - values
    ) / span


def alignment_for_candidates(
    task: SurgicalTask,
    candidates: tuple[
        CandidateViewpoint,
        ...,
    ],
) -> np.ndarray:
    """Calculate task alignment for every candidate."""

    values = []

    for candidate in candidates:
        camera_position = np.asarray(
            candidate.pose.position,
            dtype=float,
        )

        forward = np.asarray(
            candidate.pose.forward,
            dtype=float,
        )

        norm = np.linalg.norm(
            forward
        )

        if norm <= 0.0:
            raise ValueError(
                "Candidate forward vector must be non-zero."
            )

        forward = (
            forward
            / norm
        )

        point_alignments = []

        for point in task.safety_critical_points:
            direction = (
                np.asarray(
                    point,
                    dtype=float,
                )
                - camera_position
            )

            direction_norm = np.linalg.norm(
                direction
            )

            if direction_norm <= 0.0:
                value = 1.0
            else:
                direction = (
                    direction
                    / direction_norm
                )

                value = float(
                    np.dot(
                        forward,
                        direction,
                    )
                )

            point_alignments.append(
                np.clip(
                    value,
                    -1.0,
                    1.0,
                )
            )

        mean_alignment = float(
            np.mean(
                point_alignments
            )
        )

        values.append(
            0.5
            * (
                mean_alignment
                + 1.0
            )
        )

    return np.asarray(
        values,
        dtype=float,
    )


def score_candidates(
    model: ViewpointObservationModel,
    task: SurgicalTask,
    candidates: tuple[
        CandidateViewpoint,
        ...,
    ],
    variant: NormalisedAblationVariant,
    task_weight: float = 1.0,
) -> tuple[
    NormalisedCandidateScore,
    ...,
]:
    """Score candidates using normalised task components."""

    if len(candidates) == 0:
        raise ValueError(
            "candidates must not be empty."
        )

    if task_weight < 0.0:
        raise ValueError(
            "task_weight must be non-negative."
        )

    generic_scorer = GenericViewpointScorer(
        observation_model=model
    )

    generic_scores = (
        generic_scorer.score_candidates(
            current_pose=make_initial_pose(
                SphericalStructure(
                    centre=np.array(
                        [0.24, 0.0, 0.0]
                    ),
                    physical_radius=0.025,
                    safety_margin=0.015,
                )
            ),
            candidates=candidates,
            target=SphericalStructure(
                centre=np.array(
                    [0.24, 0.0, 0.0]
                ),
                physical_radius=0.025,
                safety_margin=0.015,
            ),
        )
    )

    target = SphericalStructure(
        centre=np.array(
            [0.24, 0.0, 0.0]
        ),
        physical_radius=0.025,
        safety_margin=0.015,
    )

    generic_values = np.asarray(
        [
            score.score
            for score in generic_scores
        ],
        dtype=float,
    )

    uncertainties = (
        candidate_uncertainties(
            model=model,
            candidates=candidates,
            target=target,
        )
    )

    uncertainty_scores = (
        normalise_uncertainty(
            uncertainties
        )
    )

    alignments = (
        alignment_for_candidates(
            task=task,
            candidates=candidates,
        )
    )

    results = []

    for index, candidate in enumerate(
        candidates
    ):
        alignment_term = (
            alignments[index]
            if variant.use_alignment
            else 0.0
        )

        uncertainty_term = (
            uncertainty_scores[index]
            if variant.use_uncertainty
            else 0.0
        )

        final_score = (
            generic_values[index]
            + task_weight
            * (
                alignment_term
                + uncertainty_term
            )
        )

        results.append(
            NormalisedCandidateScore(
                candidate=candidate,
                generic_score=float(
                    generic_values[index]
                ),
                alignment=float(
                    alignments[index]
                ),
                uncertainty=float(
                    uncertainties[index]
                ),
                final_score=float(
                    final_score
                ),
            )
        )

    return tuple(
        results
    )


def run_variant(
    variant: NormalisedAblationVariant,
    trial_count: int = 100,
) -> NormalisedAblationResult:
    """Run one normalised ablation condition."""

    if trial_count <= 0:
        raise ValueError(
            "trial_count must be positive."
        )

    model = make_observation_model()

    target = make_target()

    task = make_task(
        target
    )

    candidates = generate_candidate_viewpoints(
        target_position=target.centre
    )

    scored = score_candidates(
        model=model,
        task=task,
        candidates=candidates,
        variant=variant,
        task_weight=1.0,
    )

    baseline = max(
        scored,
        key=lambda item:
        item.generic_score,
    )

    selected = max(
        scored,
        key=lambda item:
        item.final_score,
    )

    selection_difference = not np.allclose(
        selected.candidate.pose.position,
        baseline.candidate.pose.position,
        atol=1e-10,
        rtol=0.0,
    )

    return NormalisedAblationResult(
        variant=variant.name,
        trial_count=trial_count,
        mean_selected_uncertainty=float(
            selected.uncertainty
        ),
        mean_selected_alignment=float(
            selected.alignment
        ),
        mean_generic_score=float(
            selected.generic_score
        ),
        mean_final_score=float(
            selected.final_score
        ),
        selection_difference_rate_percent=(
            100.0
            if selection_difference
            else 0.0
        ),
    )


def run_study(
    trial_count: int = 100,
) -> NormalisedAblationStudy:
    """Run all normalised ablation variants."""

    return NormalisedAblationStudy(
        results=tuple(
            run_variant(
                variant=variant,
                trial_count=trial_count,
            )
            for variant in make_variants()
        )
    )


def print_summary(
    study: NormalisedAblationStudy,
) -> None:
    """Print the normalised ablation results."""

    print()
    print(
        "Normalised Task-Aware Ablation"
    )
    print(
        "=============================="
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
            "Selected uncertainty: "
            f"{result.mean_selected_uncertainty:.6f} m"
        )

        print(
            "Selected alignment: "
            f"{result.mean_selected_alignment:.6f}"
        )

        print(
            "Generic score: "
            f"{result.mean_generic_score:.6f}"
        )

        print(
            "Final score: "
            f"{result.mean_final_score:.6f}"
        )

        print(
            "Selection difference: "
            f"{result.selection_difference_rate_percent:.2f}%"
        )


def main() -> None:
    """Run the normalised ablation."""

    study = run_study(
        trial_count=100
    )

    print_summary(
        study
    )


if __name__ == "__main__":
    main()