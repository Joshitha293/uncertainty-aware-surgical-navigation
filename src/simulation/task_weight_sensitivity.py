"""Task-weight sensitivity analysis for task-aware active perception."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.simulation.normalised_ablation import (
    alignment_for_candidates,
    candidate_uncertainties,
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
class WeightSensitivityResult:
    """Result for one task-weight value."""

    task_weight: float

    selected_uncertainty: float

    selected_alignment: float

    selected_generic_score: float

    selected_final_score: float

    selection_difference_from_generic: bool

    selected_candidate_index: int


@dataclass(frozen=True)
class WeightSensitivityStudy:
    """Complete task-weight sensitivity experiment."""

    results: tuple[
        WeightSensitivityResult,
        ...,
    ]


def default_task_weights() -> tuple[float, ...]:
    """Return the task weights used in the sensitivity experiment."""

    return (
        0.0,
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,
    )


def run_weight(
    task_weight: float,
) -> WeightSensitivityResult:
    """Evaluate one task-weight value."""

    if task_weight < 0.0:
        raise ValueError(
            "task_weight must be non-negative."
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

    generic_values = np.asarray(
        [
            score.score
            for score in generic_scores
        ],
        dtype=float,
    )

    uncertainties = candidate_uncertainties(
        model=model,
        candidates=candidates,
        target=target,
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

    final_scores = (
        generic_values
        + task_weight
        * (
            alignments
            + uncertainty_information
        )
    )

    selected_index = int(
        np.argmax(
            final_scores
        )
    )

    generic_index = int(
        np.argmax(
            generic_values
        )
    )

    return WeightSensitivityResult(
        task_weight=float(
            task_weight
        ),
        selected_uncertainty=float(
            uncertainties[
                selected_index
            ]
        ),
        selected_alignment=float(
            alignments[
                selected_index
            ]
        ),
        selected_generic_score=float(
            generic_values[
                selected_index
            ]
        ),
        selected_final_score=float(
            final_scores[
                selected_index
            ]
        ),
        selection_difference_from_generic=(
            selected_index
            != generic_index
        ),
        selected_candidate_index=(
            selected_index
        ),
    )


def run_sensitivity(
    weights: tuple[
        float,
        ...
    ] | None = None,
) -> WeightSensitivityStudy:
    """Run the complete task-weight sensitivity experiment."""

    if weights is None:
        weights = default_task_weights()

    if len(weights) == 0:
        raise ValueError(
            "weights must not be empty."
        )

    return WeightSensitivityStudy(
        results=tuple(
            run_weight(
                task_weight=weight
            )
            for weight in weights
        )
    )


def print_summary(
    study: WeightSensitivityStudy,
) -> None:
    """Print the sensitivity results."""

    print()
    print(
        "Task-Weight Sensitivity Analysis"
    )
    print(
        "================================"
    )

    print()

    print(
        "Weight | "
        "Uncertainty | "
        "Alignment | "
        "Generic | "
        "Final | "
        "Changed | "
        "Candidate"
    )

    print(
        "-" * 86
    )

    for result in study.results:
        print(
            f"{result.task_weight:6.2f} | "
            f"{result.selected_uncertainty:.6f} | "
            f"{result.selected_alignment:.6f} | "
            f"{result.selected_generic_score:.6f} | "
            f"{result.selected_final_score:.6f} | "
            f"{str(result.selection_difference_from_generic):7s} | "
            f"{result.selected_candidate_index}"
        )


def main() -> None:
    """Run and print the sensitivity experiment."""

    study = run_sensitivity()

    print_summary(
        study
    )


if __name__ == "__main__":
    main()