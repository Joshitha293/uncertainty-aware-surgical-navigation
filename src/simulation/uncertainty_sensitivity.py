"""Perception-uncertainty sensitivity analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.simulation.normalised_ablation import (
    alignment_for_candidates,
    make_initial_pose,
    make_observation_model,
    make_target,
    make_task,
    normalise_uncertainty,
)
from src.perception.viewpoint_scoring import (
    GenericViewpointScorer,
)
from src.perception.viewpoints import (
    generate_candidate_viewpoints,
)


@dataclass(frozen=True)
class UncertaintySensitivityResult:
    """Result for one perception-uncertainty level."""

    sigma_m: float

    generic_selected_uncertainty_m: float

    task_aware_selected_uncertainty_m: float

    generic_candidate_index: int

    task_aware_candidate_index: int

    selection_changed: bool

    task_aware_alignment: float


@dataclass(frozen=True)
class UncertaintySensitivityStudy:
    """Complete uncertainty sensitivity experiment."""

    results: tuple[
        UncertaintySensitivityResult,
        ...
    ]


def default_uncertainty_levels() -> tuple[float, ...]:
    """Return uncertainty levels used in the sensitivity analysis."""

    return (
        0.001,
        0.002,
        0.005,
        0.010,
        0.020,
        0.030,
    )


def scale_candidate_uncertainty(
    base_uncertainties: np.ndarray,
    sigma_m: float,
) -> np.ndarray:
    """Scale candidate uncertainty values to a requested sigma level.

    The relative viewpoint-dependent structure is preserved while the
    overall perception uncertainty is controlled explicitly.
    """

    if sigma_m <= 0.0:
        raise ValueError(
            "sigma_m must be positive."
        )

    values = np.asarray(
        base_uncertainties,
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError(
            "base_uncertainties must be one-dimensional."
        )

    if len(values) == 0:
        raise ValueError(
            "base_uncertainties must not be empty."
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "base_uncertainties must be finite."
        )

    minimum = float(
        np.min(values)
    )

    maximum = float(
        np.max(values)
    )

    if maximum - minimum <= 1e-12:
        return np.full_like(
            values,
            sigma_m,
            dtype=float,
        )

    normalised = (
        values - minimum
    ) / (
        maximum - minimum
    )

    return (
        sigma_m
        * (
            0.6
            + 0.4 * normalised
        )
    )


def run_uncertainty_level(
    sigma_m: float,
) -> UncertaintySensitivityResult:
    """Evaluate generic and task-aware selection at one sigma level."""

    if sigma_m <= 0.0:
        raise ValueError(
            "sigma_m must be positive."
        )

    model = make_observation_model()

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

    base_uncertainties = np.asarray(
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

    uncertainties = (
        scale_candidate_uncertainty(
            base_uncertainties=(
                base_uncertainties
            ),
            sigma_m=sigma_m,
        )
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

    generic_values = np.asarray(
        [
            score.score
            for score in generic_scores
        ],
        dtype=float,
    )

    generic_index = int(
        np.argmax(
            generic_values
        )
    )

    task_aware_values = (
        generic_values
        + (
            alignments
            + uncertainty_information
        )
    )

    task_aware_index = int(
        np.argmax(
            task_aware_values
        )
    )

    return UncertaintySensitivityResult(
        sigma_m=float(
            sigma_m
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
        task_aware_alignment=float(
            alignments[
                task_aware_index
            ]
        ),
    )


def run_sensitivity(
    levels: tuple[
        float,
        ...
    ] | None = None,
) -> UncertaintySensitivityStudy:
    """Run the complete uncertainty sensitivity experiment."""

    if levels is None:
        levels = default_uncertainty_levels()

    if len(levels) == 0:
        raise ValueError(
            "levels must not be empty."
        )

    return UncertaintySensitivityStudy(
        results=tuple(
            run_uncertainty_level(
                sigma_m=level
            )
            for level in levels
        )
    )


def print_summary(
    study: UncertaintySensitivityStudy,
) -> None:
    """Print uncertainty sensitivity results."""

    print()
    print(
        "Perception-Uncertainty Sensitivity Analysis"
    )
    print(
        "==========================================="
    )

    print()

    print(
        "Sigma (mm) | "
        "Generic selected (mm) | "
        "Task-aware selected (mm) | "
        "Generic candidate | "
        "Task-aware candidate | "
        "Changed | "
        "Alignment"
    )

    print(
        "-" * 110
    )

    for result in study.results:
        print(
            f"{result.sigma_m * 1000:9.1f} | "
            f"{result.generic_selected_uncertainty_m * 1000:21.3f} | "
            f"{result.task_aware_selected_uncertainty_m * 1000:24.3f} | "
            f"{result.generic_candidate_index:17d} | "
            f"{result.task_aware_candidate_index:20d} | "
            f"{str(result.selection_changed):7s} | "
            f"{result.task_aware_alignment:.6f}"
        )


def main() -> None:
    """Run and print the sensitivity analysis."""

    study = run_sensitivity()

    print_summary(
        study
    )


if __name__ == "__main__":
    main()