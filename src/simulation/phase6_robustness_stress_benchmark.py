"""Phase 6 robustness stress benchmark.

This benchmark deliberately violates assumptions made by the nominal
anisotropic Gaussian chance-constrained planner.

The planner receives one fixed reported anatomical estimate.

Hidden localisation errors are then generated under:

1. calibrated Gaussian uncertainty;
2. underestimated covariance;
3. overestimated covariance;
4. heavy-tailed contamination;
5. covariance-orientation mismatch;
6. temporally correlated drift.

Two planner policies are compared:

- nominal anisotropic chance constraint;
- explicitly robustified chance constraint.

The purpose is not to prove universal superiority of the robust policy.
Instead, the experiment quantifies the trade-off between:

    unsafe trajectory release
        versus
    conservative trajectory blocking.

Ground truth is used only after the planning decision.

This is simulation-only engineering evidence and is not a clinical risk
estimate or medical-device safety validation.
"""

from __future__ import annotations

import csv
import json

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.geometry.workspace import (
    SphericalStructure,
)
from src.robotics.chance_constrained_planner import (
    edge_is_chance_constrained_safe,
)
from src.robotics.robust_chance_policy import (
    RobustChancePolicy,
    robust_chance_constraint_config,
    robustify_estimated_structure,
)
from src.robotics.safety import (
    evaluate_instrument_safety,
)
from src.simulation.phase6_chance_constraint_benchmark import (
    CHANCE_THRESHOLD,
    DIRECT_EDGE_RESOLUTION,
    INSTRUMENT_RADIUS,
    PROXIMAL_LENGTH,
    SHAFT_SAMPLE_SPACING,
    build_orientation_conditions,
    chance_config,
    goal_configuration,
    make_instrument,
    select_disagreement_clearance,
    start_configuration,
)


DEFAULT_REPETITIONS = 500
DEFAULT_SEED = 67501

UNDER_ESTIMATED_STD_SCALE = 1.50
OVER_ESTIMATED_STD_SCALE = 0.67

HEAVY_TAIL_PROBABILITY = 0.05
HEAVY_TAIL_STD_SCALE = 3.0

TEMPORAL_CORRELATION = 0.92
MAX_DRIFT = 0.006

RESULT_DIRECTORY = Path(
    "results/phase6_robustness_stress"
)


ROBUST_POLICY = RobustChancePolicy(
    covariance_std_scale=1.10,
    bias_bound=0.001,
    contamination_probability=0.005,
    target_violation_probability=(
        CHANCE_THRESHOLD
    ),
)


@dataclass(frozen=True)
class PlannerDecision:
    """One planner-facing accept/reject decision."""

    name: str

    accepted: bool

    reported_threshold: float


@dataclass(frozen=True)
class TruthEvaluation:
    """Hidden-truth safety of the candidate direct trajectory."""

    minimum_surface_clearance: float

    minimum_safety_clearance: float

    collision: bool

    safety_margin_violation: bool


def _candidate_condition():
    """Return the tangential condition from the controlled 50% benchmark."""

    (
        _selected_clearance,
        conditions,
    ) = (
        select_disagreement_clearance()
    )

    tangential, _radial = (
        conditions
    )

    return tangential


def nominal_decision() -> PlannerDecision:
    """Evaluate candidate trajectory using the nominal chance model."""

    condition = (
        _candidate_condition()
    )

    accepted = (
        edge_is_chance_constrained_safe(
            instrument=make_instrument(),
            q_start=start_configuration(),
            q_goal=goal_configuration(),
            estimated_structures=(
                condition.estimate,
            ),
            instrument_radius=(
                INSTRUMENT_RADIUS
            ),
            chance_config=(
                chance_config()
            ),
            proximal_length=(
                PROXIMAL_LENGTH
            ),
            shaft_sample_spacing=(
                SHAFT_SAMPLE_SPACING
            ),
            resolution=(
                DIRECT_EDGE_RESOLUTION
            ),
        )
    )

    return PlannerDecision(
        name="nominal_chance",
        accepted=bool(
            accepted
        ),
        reported_threshold=float(
            CHANCE_THRESHOLD
        ),
    )


def robust_decision() -> PlannerDecision:
    """Evaluate candidate trajectory using explicit robustness allowances."""

    condition = (
        _candidate_condition()
    )

    robust_estimate = (
        robustify_estimated_structure(
            condition.estimate,
            ROBUST_POLICY,
        )
    )

    robust_config = (
        robust_chance_constraint_config(
            ROBUST_POLICY,
            sample_spacing=(
                SHAFT_SAMPLE_SPACING
            ),
        )
    )

    accepted = (
        edge_is_chance_constrained_safe(
            instrument=make_instrument(),
            q_start=start_configuration(),
            q_goal=goal_configuration(),
            estimated_structures=(
                robust_estimate,
            ),
            instrument_radius=(
                INSTRUMENT_RADIUS
            ),
            chance_config=(
                robust_config
            ),
            proximal_length=(
                PROXIMAL_LENGTH
            ),
            shaft_sample_spacing=(
                SHAFT_SAMPLE_SPACING
            ),
            resolution=(
                DIRECT_EDGE_RESOLUTION
            ),
        )
    )

    return PlannerDecision(
        name="robust_chance",
        accepted=bool(
            accepted
        ),
        reported_threshold=float(
            robust_config
            .max_point_violation_probability
        ),
    )


def planner_decisions() -> tuple[
    PlannerDecision,
    PlannerDecision,
]:
    """Return nominal and robust decisions."""

    return (
        nominal_decision(),
        robust_decision(),
    )


def _evaluate_candidate_against_truth(
    truth_structure: SphericalStructure,
    *,
    resolution: int = 60,
) -> TruthEvaluation:
    """Evaluate the fixed direct joint-space trajectory against hidden truth."""

    if resolution < 2:
        raise ValueError(
            "resolution must be at least 2."
        )

    instrument = (
        make_instrument()
    )

    configurations = np.linspace(
        start_configuration(),
        goal_configuration(),
        num=resolution,
        dtype=float,
    )

    minimum_surface = float(
        "inf"
    )

    minimum_safety = float(
        "inf"
    )

    collision = False

    safety_violation = False

    for configuration in (
        configurations
    ):
        shaft_start, shaft_end = (
            instrument.shaft_segment(
                configuration,
                proximal_length=(
                    PROXIMAL_LENGTH
                ),
            )
        )

        evaluation = (
            evaluate_instrument_safety(
                shaft_start=shaft_start,
                shaft_end=shaft_end,
                structures=(
                    truth_structure,
                ),
                instrument_radius=(
                    INSTRUMENT_RADIUS
                ),
            )
        )

        minimum_surface = min(
            minimum_surface,
            evaluation.minimum_surface_clearance,
        )

        minimum_safety = min(
            minimum_safety,
            evaluation.minimum_safety_clearance,
        )

        collision = bool(
            collision
            or evaluation.collision
        )

        safety_violation = bool(
            safety_violation
            or evaluation.safety_margin_violation
        )

    return TruthEvaluation(
        minimum_surface_clearance=float(
            minimum_surface
        ),
        minimum_safety_clearance=float(
            minimum_safety
        ),
        collision=bool(
            collision
        ),
        safety_margin_violation=bool(
            safety_violation
        ),
    )


def _truth_from_error(
    error: np.ndarray,
) -> SphericalStructure:
    """Construct hidden truth from localisation error.

    Convention:

        estimate = truth + error

    therefore:

        truth = estimate - error
    """

    condition = (
        _candidate_condition()
    )

    error = np.asarray(
        error,
        dtype=float,
    )

    if error.shape != (3,):
        raise ValueError(
            "error must have shape (3,)."
        )

    true_centre = (
        condition
        .estimate
        .estimated_centre
        - error
    )

    return SphericalStructure(
        centre=true_centre,
        physical_radius=(
            condition
            .estimate
            .physical_radius
        ),
        safety_margin=(
            condition
            .estimate
            .base_safety_margin
        ),
    )


def _reported_covariance() -> np.ndarray:
    """Return covariance reported to the planner."""

    return np.array(
        _candidate_condition()
        .estimate
        .uncertainty
        .covariance,
        dtype=float,
        copy=True,
    )


def _orientation_mismatch_covariance() -> np.ndarray:
    """Return matched eigenvalues with major uncertainty rotated radially."""

    (
        selected_clearance,
        _,
    ) = (
        select_disagreement_clearance()
    )

    _tangential, radial = (
        build_orientation_conditions(
            selected_clearance
        )
    )

    return np.array(
        radial
        .estimate
        .uncertainty
        .covariance,
        dtype=float,
        copy=True,
    )


def _sample_independent_gaussian(
    covariance: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample independent zero-mean Gaussian errors."""

    return rng.multivariate_normal(
        mean=np.zeros(
            3,
            dtype=float,
        ),
        cov=covariance,
        size=repetitions,
    )


def _sample_heavy_tailed_errors(
    covariance: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a Gaussian scale-mixture contamination model."""

    errors = np.empty(
        (
            repetitions,
            3,
        ),
        dtype=float,
    )

    for index in range(
        repetitions
    ):
        contaminated = bool(
            rng.random()
            < HEAVY_TAIL_PROBABILITY
        )

        scale = (
            HEAVY_TAIL_STD_SCALE
            if contaminated
            else 1.0
        )

        errors[
            index
        ] = rng.multivariate_normal(
            mean=np.zeros(
                3,
                dtype=float,
            ),
            cov=(
                covariance
                * scale**2
            ),
        )

    return errors


def _sample_temporally_correlated_drift(
    covariance: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate AR(1) localisation error plus progressive systematic drift.

    The AR(1) component has approximately the reported stationary covariance.

    Drift moves the hidden structure progressively toward the nominal
    instrument-clearance direction.
    """

    condition = (
        _candidate_condition()
    )

    rho = float(
        TEMPORAL_CORRELATION
    )

    innovation_scale = float(
        np.sqrt(
            1.0
            - rho**2
        )
    )

    errors = np.empty(
        (
            repetitions,
            3,
        ),
        dtype=float,
    )

    previous = rng.multivariate_normal(
        mean=np.zeros(
            3,
            dtype=float,
        ),
        cov=covariance,
    )

    for index in range(
        repetitions
    ):
        if index > 0:
            innovation = (
                rng.multivariate_normal(
                    mean=np.zeros(
                        3,
                        dtype=float,
                    ),
                    cov=covariance,
                )
            )

            previous = (
                rho
                * previous
                + innovation_scale
                * innovation
            )

        fraction = (
            0.0
            if repetitions == 1
            else float(
                index
            )
            / float(
                repetitions
                - 1
            )
        )

        drift = (
            fraction
            * MAX_DRIFT
            * condition
            .nominal_clearance_direction
        )

        errors[
            index
        ] = (
            previous
            + drift
        )

    return errors


def generate_stress_errors(
    stress_name: str,
    repetitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate hidden errors for one stress condition."""

    if repetitions < 1:
        raise ValueError(
            "repetitions must be positive."
        )

    covariance = (
        _reported_covariance()
    )

    if (
        stress_name
        == "calibrated_gaussian"
    ):
        return (
            _sample_independent_gaussian(
                covariance,
                repetitions,
                rng,
            )
        )

    if (
        stress_name
        == "covariance_underestimated"
    ):
        return (
            _sample_independent_gaussian(
                covariance
                * UNDER_ESTIMATED_STD_SCALE**2,
                repetitions,
                rng,
            )
        )

    if (
        stress_name
        == "covariance_overestimated"
    ):
        return (
            _sample_independent_gaussian(
                covariance
                * OVER_ESTIMATED_STD_SCALE**2,
                repetitions,
                rng,
            )
        )

    if (
        stress_name
        == "heavy_tailed_contamination"
    ):
        return (
            _sample_heavy_tailed_errors(
                covariance,
                repetitions,
                rng,
            )
        )

    if (
        stress_name
        == "orientation_mismatch"
    ):
        return (
            _sample_independent_gaussian(
                _orientation_mismatch_covariance(),
                repetitions,
                rng,
            )
        )

    if (
        stress_name
        == "temporally_correlated_drift"
    ):
        return (
            _sample_temporally_correlated_drift(
                covariance,
                repetitions,
                rng,
            )
        )

    raise ValueError(
        f"Unknown stress condition: {stress_name}"
    )


def stress_condition_names() -> tuple[
    str,
    ...,
]:
    """Return ordered stress conditions."""

    return (
        "calibrated_gaussian",
        "covariance_underestimated",
        "covariance_overestimated",
        "heavy_tailed_contamination",
        "orientation_mismatch",
        "temporally_correlated_drift",
    )


def _longest_true_run(
    values: list[bool],
) -> int:
    """Return longest consecutive True run."""

    longest = 0
    current = 0

    for value in values:
        if value:
            current += 1
            longest = max(
                longest,
                current,
            )
        else:
            current = 0

    return int(
        longest
    )


def _wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[
    float,
    float,
]:
    """Return Wilson 95% interval."""

    if total <= 0:
        return (
            0.0,
            0.0,
        )

    n = float(
        total
    )

    p = (
        float(
            successes
        )
        / n
    )

    z2 = (
        z**2
    )

    denominator = (
        1.0
        + z2
        / n
    )

    centre = (
        p
        + z2
        / (
            2.0
            * n
        )
    ) / denominator

    half = (
        z
        / denominator
        * np.sqrt(
            (
                p
                * (
                    1.0
                    - p
                )
                / n
            )
            + z2
            / (
                4.0
                * n**2
            )
        )
    )

    return (
        float(
            max(
                0.0,
                centre
                - half,
            )
        ),
        float(
            min(
                1.0,
                centre
                + half,
            )
        ),
    )


def _summarise_policy(
    rows: list[dict],
    decision: PlannerDecision,
) -> dict:
    """Summarise one planner policy under one hidden-truth stress."""

    total = len(
        rows
    )

    truth_violations = [
        bool(
            row[
                "truth_safety_violation"
            ]
        )
        for row in rows
    ]

    truth_collisions = [
        bool(
            row[
                "truth_collision"
            ]
        )
        for row in rows
    ]

    if decision.accepted:
        unsafe_release_count = int(
            sum(
                truth_violations
            )
        )

        safe_release_count = int(
            total
            - unsafe_release_count
        )

        conservative_block_count = 0

    else:
        unsafe_release_count = 0

        safe_release_count = 0

        conservative_block_count = int(
            sum(
                not violation
                for violation in (
                    truth_violations
                )
            )
        )

    unsafe_ci = (
        _wilson_interval(
            unsafe_release_count,
            total,
        )
    )

    truth_violation_count = int(
        sum(
            truth_violations
        )
    )

    truth_collision_count = int(
        sum(
            truth_collisions
        )
    )

    return {
        "trajectory_released": bool(
            decision.accepted
        ),
        "reported_probability_threshold": float(
            decision.reported_threshold
        ),
        "truth_candidate_safety_violation_rate": float(
            truth_violation_count
            / total
        ),
        "truth_candidate_collision_rate": float(
            truth_collision_count
            / total
        ),
        "unsafe_release_rate": float(
            unsafe_release_count
            / total
        ),
        "unsafe_release_95_ci": [
            float(
                unsafe_ci[
                    0
                ]
            ),
            float(
                unsafe_ci[
                    1
                ]
            ),
        ],
        "safe_release_rate": float(
            safe_release_count
            / total
        ),
        "conservative_safe_block_rate": float(
            conservative_block_count
            / total
        ),
        "longest_truth_violation_run": (
            _longest_true_run(
                truth_violations
            )
        ),
        "mean_minimum_truth_safety_clearance_mm": float(
            np.mean(
                [
                    row[
                        "minimum_safety_clearance_mm"
                    ]
                    for row in rows
                ]
            )
        ),
    }


def run_benchmark(
    repetitions: int = DEFAULT_REPETITIONS,
    seed: int = DEFAULT_SEED,
    output_directory: Path = RESULT_DIRECTORY,
) -> dict:
    """Run the full robustness stress experiment."""

    if repetitions < 1:
        raise ValueError(
            "repetitions must be positive."
        )

    nominal, robust = (
        planner_decisions()
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows: list[
        dict
    ] = []

    summary = {
        "benchmark": (
            "phase6_robustness_stress"
        ),
        "repetitions_per_condition": int(
            repetitions
        ),
        "base_seed": int(
            seed
        ),
        "reported_chance_threshold": float(
            CHANCE_THRESHOLD
        ),
        "robust_policy": {
            "covariance_std_scale": float(
                ROBUST_POLICY
                .covariance_std_scale
            ),
            "bias_bound_mm": float(
                1000.0
                * ROBUST_POLICY
                .bias_bound
            ),
            "contamination_probability": float(
                ROBUST_POLICY
                .contamination_probability
            ),
            "effective_gaussian_threshold": float(
                ROBUST_POLICY
                .effective_gaussian_threshold
            ),
        },
        "planner_decisions": {
            nominal.name: bool(
                nominal.accepted
            ),
            robust.name: bool(
                robust.accepted
            ),
        },
        "stress_conditions": {},
        "interpretation_boundary": (
            "Simulation stress testing only. "
            "Unsafe-release rate is conditional on this controlled candidate "
            "trajectory and does not constitute clinical risk."
        ),
    }

    for condition_index, stress_name in enumerate(
        stress_condition_names()
    ):
        rng = np.random.default_rng(
            seed
            + 10000
            * condition_index
        )

        errors = (
            generate_stress_errors(
                stress_name,
                repetitions,
                rng,
            )
        )

        truth_rows: list[
            dict
        ] = []

        for repetition in range(
            repetitions
        ):
            truth = (
                _truth_from_error(
                    errors[
                        repetition
                    ]
                )
            )

            evaluation = (
                _evaluate_candidate_against_truth(
                    truth
                )
            )

            base_row = {
                "stress_condition": (
                    stress_name
                ),
                "repetition": int(
                    repetition
                ),
                "error_x_mm": float(
                    1000.0
                    * errors[
                        repetition,
                        0
                    ]
                ),
                "error_y_mm": float(
                    1000.0
                    * errors[
                        repetition,
                        1
                    ]
                ),
                "error_z_mm": float(
                    1000.0
                    * errors[
                        repetition,
                        2
                    ]
                ),
                "minimum_surface_clearance_mm": float(
                    1000.0
                    * evaluation
                    .minimum_surface_clearance
                ),
                "minimum_safety_clearance_mm": float(
                    1000.0
                    * evaluation
                    .minimum_safety_clearance
                ),
                "truth_collision": bool(
                    evaluation.collision
                ),
                "truth_safety_violation": bool(
                    evaluation
                    .safety_margin_violation
                ),
            }

            truth_rows.append(
                base_row
            )

            for decision in (
                nominal,
                robust,
            ):
                row = dict(
                    base_row
                )

                row[
                    "planner_policy"
                ] = decision.name

                row[
                    "trajectory_released"
                ] = bool(
                    decision.accepted
                )

                row[
                    "unsafe_release"
                ] = bool(
                    decision.accepted
                    and evaluation
                    .safety_margin_violation
                )

                row[
                    "safe_release"
                ] = bool(
                    decision.accepted
                    and not evaluation
                    .safety_margin_violation
                )

                row[
                    "conservative_safe_block"
                ] = bool(
                    not decision.accepted
                    and not evaluation
                    .safety_margin_violation
                )

                all_rows.append(
                    row
                )

        condition_summary = {
            "candidate_truth_safety_violation_rate": float(
                np.mean(
                    [
                        row[
                            "truth_safety_violation"
                        ]
                        for row in truth_rows
                    ]
                )
            ),
            "candidate_truth_collision_rate": float(
                np.mean(
                    [
                        row[
                            "truth_collision"
                        ]
                        for row in truth_rows
                    ]
                )
            ),
            "longest_truth_violation_run": int(
                _longest_true_run(
                    [
                        bool(
                            row[
                                "truth_safety_violation"
                            ]
                        )
                        for row in truth_rows
                    ]
                )
            ),
            "policies": {},
        }

        for decision in (
            nominal,
            robust,
        ):
            condition_summary[
                "policies"
            ][
                decision.name
            ] = (
                _summarise_policy(
                    truth_rows,
                    decision,
                )
            )

        summary[
            "stress_conditions"
        ][
            stress_name
        ] = condition_summary

    csv_path = (
        output_directory
        / "phase6_robustness_stress_trials.csv"
    )

    json_path = (
        output_directory
        / "phase6_robustness_stress_summary.json"
    )

    fieldnames = [
        "stress_condition",
        "repetition",
        "planner_policy",
        "trajectory_released",
        "unsafe_release",
        "safe_release",
        "conservative_safe_block",
        "error_x_mm",
        "error_y_mm",
        "error_z_mm",
        "minimum_surface_clearance_mm",
        "minimum_safety_clearance_mm",
        "truth_collision",
        "truth_safety_violation",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in all_rows:
            writer.writerow(
                row
            )

    summary[
        "trial_csv"
    ] = str(
        csv_path
    )

    summary[
        "summary_json"
    ] = str(
        json_path
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            summary,
            json_file,
            indent=2,
        )

    return summary


def print_summary(
    summary: dict,
) -> None:
    """Print compact robustness evidence."""

    print(
        "\n"
        "=== Phase 6 Robustness Stress Benchmark ==="
    )

    print(
        "Nominal candidate released: "
        f"{summary['planner_decisions']['nominal_chance']}"
    )

    print(
        "Robust candidate released: "
        f"{summary['planner_decisions']['robust_chance']}"
    )

    print(
        "Robust covariance std scale: "
        f"{summary['robust_policy']['covariance_std_scale']:.3f}"
    )

    print(
        "Robust bias allowance: "
        f"{summary['robust_policy']['bias_bound_mm']:.3f} mm"
    )

    print(
        "Robust contamination allowance: "
        f"{summary['robust_policy']['contamination_probability']:.3f}"
    )

    print(
        "Effective Gaussian threshold: "
        f"{summary['robust_policy']['effective_gaussian_threshold']:.5f}"
    )

    for stress_name, condition in (
        summary[
            "stress_conditions"
        ].items()
    ):
        print(
            "\n"
            f"--- {stress_name} ---"
        )

        print(
            "Candidate hidden-truth safety violation: "
            f"{100.0 * condition['candidate_truth_safety_violation_rate']:.1f}%"
        )

        print(
            "Candidate hidden-truth collision: "
            f"{100.0 * condition['candidate_truth_collision_rate']:.1f}%"
        )

        print(
            "Longest unsafe run: "
            f"{condition['longest_truth_violation_run']}"
        )

        for policy_name, policy in (
            condition[
                "policies"
            ].items()
        ):
            print(
                f"{policy_name}: "
                f"released={policy['trajectory_released']} | "
                f"unsafe_release="
                f"{100.0 * policy['unsafe_release_rate']:.1f}% | "
                f"safe_release="
                f"{100.0 * policy['safe_release_rate']:.1f}% | "
                f"safe_block="
                f"{100.0 * policy['conservative_safe_block_rate']:.1f}%"
            )

    print(
        "\nArtifacts:"
    )

    print(
        summary[
            "trial_csv"
        ]
    )

    print(
        summary[
            "summary_json"
        ]
    )

    print(
        "\nInterpretation: robustness stress evidence only."
    )


def main() -> None:
    """Command-line entry point."""

    summary = (
        run_benchmark()
    )

    print_summary(
        summary
    )


if __name__ == "__main__":
    main()