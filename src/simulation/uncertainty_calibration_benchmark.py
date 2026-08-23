"""Supplementary formal calibration benchmark for localisation uncertainty.

The observation model reports an isotropic positional standard deviation
(localisation_sigma). This benchmark evaluates whether realised simulated
localisation errors are statistically consistent with that reported sigma.

For an isotropic 3-D Gaussian localisation error e ~ N(0, sigma^2 I),

    ||e||^2 / sigma^2 ~ chi-square(df=3)

Therefore a calibrated model should approximately satisfy:

- mean normalised squared error = 3;
- empirical chi-square coverage matches nominal coverage;
- mean Euclidean localisation error / sigma ~= 1.59577.

This is a simulation-model calibration experiment. It does not constitute
calibration of a physical camera, clinical imaging system, or learned model.
"""

from __future__ import annotations

import argparse
import csv
import json

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.simulation.three_strategy_robustness_benchmark import (
    RobustnessScenario,
    build_scenario_inputs,
    default_scenarios,
)


CHI_SQUARE_3_THRESHOLDS = {
    0.50: 2.3659738843753377,
    0.90: 6.251388631170325,
    0.95: 7.814727903251179,
    0.99: 11.344866730144373,
}

EXPECTED_MEAN_NEES = 3.0

EXPECTED_MEAN_RADIAL_ERROR_OVER_SIGMA = float(
    2.0 * np.sqrt(
        2.0 / np.pi
    )
)


@dataclass(frozen=True)
class CalibrationConfig:
    """Configuration for the uncertainty calibration experiment."""

    samples_per_viewpoint: int = 100

    candidate_viewpoints_per_scenario: int = 5

    seed_base: int = 20260930

    bootstrap_samples: int = 2000

    bootstrap_seed: int = 20261001

    confidence_level: float = 0.95

    calibration_bin_count: int = 6

    def __post_init__(self) -> None:
        if self.samples_per_viewpoint <= 0:
            raise ValueError(
                "samples_per_viewpoint must be positive."
            )

        if (
            self.candidate_viewpoints_per_scenario
            <= 0
        ):
            raise ValueError(
                "candidate_viewpoints_per_scenario "
                "must be positive."
            )

        if self.bootstrap_samples <= 0:
            raise ValueError(
                "bootstrap_samples must be positive."
            )

        if not (
            0.0
            < self.confidence_level
            < 1.0
        ):
            raise ValueError(
                "confidence_level must lie between 0 and 1."
            )

        if self.calibration_bin_count <= 0:
            raise ValueError(
                "calibration_bin_count must be positive."
            )


@dataclass(frozen=True)
class CalibrationRecord:
    """One simulated uncertainty-calibration observation."""

    scenario_id: int

    scenario_name: str

    viewpoint_label: str

    viewpoint_index: int

    sample: int

    seed: int

    predicted_sigma: float

    localisation_error: float

    normalized_squared_error: float

    radial_error_over_sigma: float

    covered_50: bool

    covered_90: bool

    covered_95: bool

    covered_99: bool


@dataclass(frozen=True)
class IntervalEstimate:
    """Point estimate with confidence interval."""

    n: int

    mean: float

    ci_low: float

    ci_high: float


@dataclass(frozen=True)
class CalibrationBin:
    """Calibration result over one predicted-sigma range."""

    bin_index: int

    n: int

    sigma_min: float

    sigma_max: float

    mean_sigma: float

    mean_localisation_error: float

    mean_nees: float

    nees_ratio: float

    empirical_95_coverage: float


@dataclass(frozen=True)
class CalibrationSummary:
    """Final aggregate calibration diagnostics."""

    record_count: int

    scenario_count: int

    viewpoint_condition_count: int

    predicted_sigma_min: float

    predicted_sigma_max: float

    mean_nees: IntervalEstimate

    nees_ratio: float

    mean_radial_error_over_sigma: IntervalEstimate

    expected_mean_radial_error_over_sigma: float

    coverage_50: IntervalEstimate

    coverage_90: IntervalEstimate

    coverage_95: IntervalEstimate

    coverage_99: IntervalEstimate

    calibration_slope: float

    calibration_label: str


@dataclass(frozen=True)
class CalibrationResult:
    """Complete formal uncertainty calibration result."""

    config: CalibrationConfig

    scenarios: tuple[
        RobustnessScenario,
        ...,
    ]

    records: tuple[
        CalibrationRecord,
        ...,
    ]

    bins: tuple[
        CalibrationBin,
        ...,
    ]

    summary: CalibrationSummary


def _wilson_interval(
    *,
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[
    float,
    float,
]:
    """Return a two-sided Wilson confidence interval."""

    if total <= 0:
        raise ValueError(
            "total must be positive."
        )

    if successes < 0:
        raise ValueError(
            "successes must be non-negative."
        )

    if successes > total:
        raise ValueError(
            "successes cannot exceed total."
        )

    proportion = (
        successes
        / total
    )

    denominator = (
        1.0
        + (
            z
            * z
            / total
        )
    )

    centre = (
        proportion
        + (
            z
            * z
            / (
                2.0
                * total
            )
        )
    ) / denominator

    half_width = (
        z
        * np.sqrt(
            (
                proportion
                * (
                    1.0
                    - proportion
                )
                / total
            )
            + (
                z
                * z
                / (
                    4.0
                    * total
                    * total
                )
            )
        )
        / denominator
    )

    return (
        float(
            max(
                0.0,
                centre
                - half_width,
            )
        ),
        float(
            min(
                1.0,
                centre
                + half_width,
            )
        ),
    )


def _bootstrap_mean(
    values: np.ndarray,
    *,
    config: CalibrationConfig,
    seed_offset: int,
) -> IntervalEstimate:
    """Return a deterministic bootstrap confidence interval."""

    values = np.asarray(
        values,
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError(
            "values must be one-dimensional."
        )

    if values.size == 0:
        raise ValueError(
            "values must not be empty."
        )

    if not np.all(
        np.isfinite(
            values
        )
    ):
        raise ValueError(
            "values must be finite."
        )

    mean = float(
        np.mean(
            values
        )
    )

    rng = np.random.default_rng(
        config.bootstrap_seed
        + seed_offset
    )

    n = int(
        values.size
    )

    bootstrap_means = np.empty(
        config.bootstrap_samples,
        dtype=float,
    )

    for index in range(
        config.bootstrap_samples
    ):
        sample = rng.choice(
            values,
            size=n,
            replace=True,
        )

        bootstrap_means[
            index
        ] = float(
            np.mean(
                sample
            )
        )

    alpha = (
        1.0
        - config.confidence_level
    )

    return IntervalEstimate(
        n=n,
        mean=mean,
        ci_low=float(
            np.quantile(
                bootstrap_means,
                alpha / 2.0,
            )
        ),
        ci_high=float(
            np.quantile(
                bootstrap_means,
                1.0
                - alpha / 2.0,
            )
        ),
    )


def _coverage_estimate(
    values: np.ndarray,
) -> IntervalEstimate:
    """Return empirical coverage and Wilson interval."""

    values = np.asarray(
        values,
        dtype=bool,
    )

    if values.ndim != 1:
        raise ValueError(
            "coverage values must be one-dimensional."
        )

    if values.size == 0:
        raise ValueError(
            "coverage values must not be empty."
        )

    successes = int(
        np.count_nonzero(
            values
        )
    )

    total = int(
        values.size
    )

    low, high = _wilson_interval(
        successes=successes,
        total=total,
    )

    return IntervalEstimate(
        n=total,
        mean=float(
            successes
            / total
        ),
        ci_low=low,
        ci_high=high,
    )


def _viewpoint_conditions(
    *,
    inputs,
    count: int,
) -> tuple[
    tuple[
        str,
        int,
        object,
    ],
    ...,
]:
    """Select initial plus evenly distributed candidate camera poses."""

    candidate_count = len(
        inputs.candidates
    )

    if candidate_count == 0:
        raise ValueError(
            "Scenario must contain candidate viewpoints."
        )

    number = min(
        int(
            count
        ),
        candidate_count,
    )

    indices = np.linspace(
        0,
        candidate_count - 1,
        num=number,
        dtype=int,
    )

    indices = np.unique(
        indices
    )

    conditions: list[
        tuple[
            str,
            int,
            object,
        ]
    ] = [
        (
            "initial",
            -1,
            inputs.initial_pose,
        )
    ]

    for index in indices:
        integer_index = int(
            index
        )

        conditions.append(
            (
                f"candidate_{integer_index}",
                integer_index,
                inputs
                .candidates[
                    integer_index
                ]
                .pose,
            )
        )

    return tuple(
        conditions
    )


def run_calibration_trials(
    config: CalibrationConfig,
    *,
    scenarios: tuple[
        RobustnessScenario,
        ...,
    ] | None = None,
    show_progress: bool = False,
) -> tuple[
    CalibrationRecord,
    ...,
]:
    """Generate formal uncertainty-calibration samples."""

    if scenarios is None:
        scenarios = (
            default_scenarios()
        )

    if len(
        scenarios
    ) == 0:
        raise ValueError(
            "At least one scenario is required."
        )

    records: list[
        CalibrationRecord
    ] = []

    total_conditions = 0

    for scenario in scenarios:
        inputs = (
            build_scenario_inputs(
                scenario
            )
        )

        total_conditions += len(
            _viewpoint_conditions(
                inputs=inputs,
                count=(
                    config
                    .candidate_viewpoints_per_scenario
                ),
            )
        )

    completed_conditions = 0

    for scenario in scenarios:
        inputs = (
            build_scenario_inputs(
                scenario
            )
        )

        conditions = (
            _viewpoint_conditions(
                inputs=inputs,
                count=(
                    config
                    .candidate_viewpoints_per_scenario
                ),
            )
        )

        for (
            viewpoint_label,
            viewpoint_index,
            pose,
        ) in conditions:
            quality = (
                inputs
                .observation_model
                .observation_quality(
                    camera_pose=pose,
                    structure=inputs.target,
                    occluders=(
                        inputs.occluders
                    ),
                )
            )

            sigma = float(
                quality.localisation_sigma
            )

            if not np.isfinite(
                sigma
            ):
                raise ValueError(
                    "Predicted sigma must be finite."
                )

            if sigma <= 0.0:
                raise ValueError(
                    "Predicted sigma must be positive."
                )

            for sample in range(
                config.samples_per_viewpoint
            ):
                seed = (
                    config.seed_base
                    + (
                        scenario.scenario_id
                        * 1_000_000
                    )
                    + (
                        (
                            viewpoint_index
                            + 2
                        )
                        * 10_000
                    )
                    + sample
                )

                observations = (
                    inputs
                    .observation_model
                    .observe_structures(
                        camera_pose=pose,
                        structures=(
                            inputs.target,
                        ),
                        rng=(
                            np.random.default_rng(
                                seed
                            )
                        ),
                        occluders=(
                            inputs.occluders
                        ),
                    )
                )

                if len(
                    observations
                ) != 1:
                    raise RuntimeError(
                        "Expected exactly one target observation."
                    )

                observation = (
                    observations[0]
                )

                localisation_error = float(
                    observation
                    .localisation_error
                )

                if not np.isfinite(
                    localisation_error
                ):
                    raise ValueError(
                        "Localisation error must be finite."
                    )

                if localisation_error < 0.0:
                    raise ValueError(
                        "Localisation error must be non-negative."
                    )

                normalized_squared_error = float(
                    (
                        localisation_error
                        * localisation_error
                    )
                    / (
                        sigma
                        * sigma
                    )
                )

                radial_ratio = float(
                    localisation_error
                    / sigma
                )

                records.append(
                    CalibrationRecord(
                        scenario_id=(
                            scenario
                            .scenario_id
                        ),
                        scenario_name=(
                            scenario.name
                        ),
                        viewpoint_label=(
                            viewpoint_label
                        ),
                        viewpoint_index=(
                            viewpoint_index
                        ),
                        sample=sample,
                        seed=seed,
                        predicted_sigma=(
                            sigma
                        ),
                        localisation_error=(
                            localisation_error
                        ),
                        normalized_squared_error=(
                            normalized_squared_error
                        ),
                        radial_error_over_sigma=(
                            radial_ratio
                        ),
                        covered_50=(
                            normalized_squared_error
                            <= (
                                CHI_SQUARE_3_THRESHOLDS[
                                    0.50
                                ]
                            )
                        ),
                        covered_90=(
                            normalized_squared_error
                            <= (
                                CHI_SQUARE_3_THRESHOLDS[
                                    0.90
                                ]
                            )
                        ),
                        covered_95=(
                            normalized_squared_error
                            <= (
                                CHI_SQUARE_3_THRESHOLDS[
                                    0.95
                                ]
                            )
                        ),
                        covered_99=(
                            normalized_squared_error
                            <= (
                                CHI_SQUARE_3_THRESHOLDS[
                                    0.99
                                ]
                            )
                        ),
                    )
                )

            completed_conditions += 1

            if show_progress:
                print(
                    "Completed "
                    f"{completed_conditions}/"
                    f"{total_conditions} "
                    "calibration conditions."
                )

    return tuple(
        records
    )


def _make_calibration_bins(
    *,
    records: tuple[
        CalibrationRecord,
        ...,
    ],
    bin_count: int,
) -> tuple[
    CalibrationBin,
    ...,
]:
    """Build approximately equal-count predicted-sigma calibration bins."""

    if len(
        records
    ) == 0:
        raise ValueError(
            "records must not be empty."
        )

    ordered = sorted(
        records,
        key=lambda record: (
            record.predicted_sigma
        ),
    )

    groups = np.array_split(
        np.asarray(
            ordered,
            dtype=object,
        ),
        min(
            bin_count,
            len(
                ordered
            ),
        ),
    )

    bins: list[
        CalibrationBin
    ] = []

    for index, group in enumerate(
        groups
    ):
        selected = list(
            group
        )

        if len(
            selected
        ) == 0:
            continue

        sigmas = np.asarray(
            [
                record.predicted_sigma
                for record
                in selected
            ],
            dtype=float,
        )

        errors = np.asarray(
            [
                record.localisation_error
                for record
                in selected
            ],
            dtype=float,
        )

        nees = np.asarray(
            [
                record
                .normalized_squared_error
                for record
                in selected
            ],
            dtype=float,
        )

        coverage = np.asarray(
            [
                record.covered_95
                for record
                in selected
            ],
            dtype=float,
        )

        mean_nees = float(
            np.mean(
                nees
            )
        )

        bins.append(
            CalibrationBin(
                bin_index=index,
                n=len(
                    selected
                ),
                sigma_min=float(
                    np.min(
                        sigmas
                    )
                ),
                sigma_max=float(
                    np.max(
                        sigmas
                    )
                ),
                mean_sigma=float(
                    np.mean(
                        sigmas
                    )
                ),
                mean_localisation_error=float(
                    np.mean(
                        errors
                    )
                ),
                mean_nees=(
                    mean_nees
                ),
                nees_ratio=float(
                    mean_nees
                    / EXPECTED_MEAN_NEES
                ),
                empirical_95_coverage=float(
                    np.mean(
                        coverage
                    )
                ),
            )
        )

    return tuple(
        bins
    )


def analyse_calibration(
    *,
    records: tuple[
        CalibrationRecord,
        ...,
    ],
    config: CalibrationConfig,
) -> tuple[
    tuple[
        CalibrationBin,
        ...,
    ],
    CalibrationSummary,
]:
    """Calculate aggregate uncertainty-calibration diagnostics."""

    if len(
        records
    ) == 0:
        raise ValueError(
            "records must not be empty."
        )

    sigmas = np.asarray(
        [
            record.predicted_sigma
            for record
            in records
        ],
        dtype=float,
    )

    errors = np.asarray(
        [
            record.localisation_error
            for record
            in records
        ],
        dtype=float,
    )

    nees = np.asarray(
        [
            record
            .normalized_squared_error
            for record
            in records
        ],
        dtype=float,
    )

    radial_ratios = np.asarray(
        [
            record
            .radial_error_over_sigma
            for record
            in records
        ],
        dtype=float,
    )

    mean_nees = _bootstrap_mean(
        nees,
        config=config,
        seed_offset=1,
    )

    mean_radial_ratio = (
        _bootstrap_mean(
            radial_ratios,
            config=config,
            seed_offset=2,
        )
    )

    coverage_50 = _coverage_estimate(
        np.asarray(
            [
                record.covered_50
                for record
                in records
            ],
            dtype=bool,
        )
    )

    coverage_90 = _coverage_estimate(
        np.asarray(
            [
                record.covered_90
                for record
                in records
            ],
            dtype=bool,
        )
    )

    coverage_95 = _coverage_estimate(
        np.asarray(
            [
                record.covered_95
                for record
                in records
            ],
            dtype=bool,
        )
    )

    coverage_99 = _coverage_estimate(
        np.asarray(
            [
                record.covered_99
                for record
                in records
            ],
            dtype=bool,
        )
    )

    expected_squared_error = (
        EXPECTED_MEAN_NEES
        * sigmas
        * sigmas
    )

    realised_squared_error = (
        errors
        * errors
    )

    denominator = float(
        np.dot(
            expected_squared_error,
            expected_squared_error,
        )
    )

    if denominator <= 1e-30:
        calibration_slope = float(
            "nan"
        )

    else:
        calibration_slope = float(
            np.dot(
                expected_squared_error,
                realised_squared_error,
            )
            / denominator
        )

    nees_ratio = float(
        mean_nees.mean
        / EXPECTED_MEAN_NEES
    )

    coverage_error = abs(
        coverage_95.mean
        - 0.95
    )

    if (
        abs(
            nees_ratio
            - 1.0
        )
        <= 0.10
        and coverage_error
        <= 0.02
    ):
        calibration_label = (
            "well_calibrated_under_simulation"
        )

    elif (
        abs(
            nees_ratio
            - 1.0
        )
        <= 0.20
        and coverage_error
        <= 0.04
    ):
        calibration_label = (
            "approximately_calibrated_under_simulation"
        )

    else:
        calibration_label = (
            "calibration_mismatch_detected"
        )

    condition_keys = {
        (
            record.scenario_id,
            record.viewpoint_label,
        )
        for record
        in records
    }

    scenario_ids = {
        record.scenario_id
        for record
        in records
    }

    bins = _make_calibration_bins(
        records=records,
        bin_count=(
            config.calibration_bin_count
        ),
    )

    summary = CalibrationSummary(
        record_count=len(
            records
        ),
        scenario_count=len(
            scenario_ids
        ),
        viewpoint_condition_count=len(
            condition_keys
        ),
        predicted_sigma_min=float(
            np.min(
                sigmas
            )
        ),
        predicted_sigma_max=float(
            np.max(
                sigmas
            )
        ),
        mean_nees=(
            mean_nees
        ),
        nees_ratio=(
            nees_ratio
        ),
        mean_radial_error_over_sigma=(
            mean_radial_ratio
        ),
        expected_mean_radial_error_over_sigma=float(
            EXPECTED_MEAN_RADIAL_ERROR_OVER_SIGMA
        ),
        coverage_50=(
            coverage_50
        ),
        coverage_90=(
            coverage_90
        ),
        coverage_95=(
            coverage_95
        ),
        coverage_99=(
            coverage_99
        ),
        calibration_slope=(
            calibration_slope
        ),
        calibration_label=(
            calibration_label
        ),
    )

    return (
        bins,
        summary,
    )


def run_uncertainty_calibration(
    config: CalibrationConfig
    | None = None,
    *,
    scenarios: tuple[
        RobustnessScenario,
        ...,
    ] | None = None,
    show_progress: bool = False,
) -> CalibrationResult:
    """Run and analyse the complete calibration benchmark."""

    if config is None:
        config = (
            CalibrationConfig()
        )

    if scenarios is None:
        scenarios = (
            default_scenarios()
        )

    records = run_calibration_trials(
        config,
        scenarios=scenarios,
        show_progress=show_progress,
    )

    bins, summary = analyse_calibration(
        records=records,
        config=config,
    )

    return CalibrationResult(
        config=config,
        scenarios=scenarios,
        records=records,
        bins=bins,
        summary=summary,
    )


def save_calibration_outputs(
    result: CalibrationResult,
    *,
    output_directory: str
    | Path = (
        "results/"
        "supplementary_uncertainty_calibration"
    ),
) -> tuple[
    Path,
    Path,
]:
    """Save raw calibration samples and statistical summary."""

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        output_directory
        / "uncertainty_calibration_samples.csv"
    )

    summary_path = (
        output_directory
        / "uncertainty_calibration_summary.json"
    )

    if len(
        result.records
    ) == 0:
        raise ValueError(
            "Cannot save an empty calibration result."
        )

    with raw_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                asdict(
                    result.records[
                        0
                    ]
                ).keys()
            ),
        )

        writer.writeheader()

        for record in (
            result.records
        ):
            writer.writerow(
                asdict(
                    record
                )
            )

    payload = {
        "config": asdict(
            result.config
        ),
        "chi_square_df": 3,
        "chi_square_thresholds": {
            str(
                key
            ): value
            for key, value
            in CHI_SQUARE_3_THRESHOLDS.items()
        },
        "expected_mean_nees": (
            EXPECTED_MEAN_NEES
        ),
        "expected_mean_radial_error_over_sigma": float(
            EXPECTED_MEAN_RADIAL_ERROR_OVER_SIGMA
        ),
        "summary": asdict(
            result.summary
        ),
        "calibration_bins": [
            asdict(
                calibration_bin
            )
            for calibration_bin
            in result.bins
        ],
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            allow_nan=True,
        )

    return (
        raw_path,
        summary_path,
    )


def print_calibration_summary(
    result: CalibrationResult,
) -> None:
    """Print the principal uncertainty-calibration results."""

    summary = (
        result.summary
    )

    print()

    print(
        "Formal Localisation-Uncertainty Calibration"
    )

    print(
        "==========================================="
    )

    print(
        f"Scenarios: "
        f"{summary.scenario_count}"
    )

    print(
        "Viewpoint conditions: "
        f"{summary.viewpoint_condition_count}"
    )

    print(
        f"Samples: "
        f"{summary.record_count}"
    )

    print(
        "Predicted sigma range (mm): "
        f"{summary.predicted_sigma_min * 1000.0:.3f}"
        " - "
        f"{summary.predicted_sigma_max * 1000.0:.3f}"
    )

    print()

    print(
        "Mean NEES-like statistic: "
        f"{summary.mean_nees.mean:.4f} "
        f"["
        f"{summary.mean_nees.ci_low:.4f}, "
        f"{summary.mean_nees.ci_high:.4f}"
        f"]"
    )

    print(
        "Expected mean for calibrated "
        "3-D Gaussian: "
        f"{EXPECTED_MEAN_NEES:.4f}"
    )

    print(
        "NEES ratio observed/expected: "
        f"{summary.nees_ratio:.4f}"
    )

    print()

    print(
        "Mean radial error / sigma: "
        f"{summary.mean_radial_error_over_sigma.mean:.4f} "
        f"["
        f"{summary.mean_radial_error_over_sigma.ci_low:.4f}, "
        f"{summary.mean_radial_error_over_sigma.ci_high:.4f}"
        f"]"
    )

    print(
        "Expected radial error / sigma: "
        f"{summary.expected_mean_radial_error_over_sigma:.4f}"
    )

    print()

    print(
        "50% nominal coverage: "
        f"{summary.coverage_50.mean * 100.0:.2f}% "
        f"["
        f"{summary.coverage_50.ci_low * 100.0:.2f}, "
        f"{summary.coverage_50.ci_high * 100.0:.2f}"
        f"]"
    )

    print(
        "90% nominal coverage: "
        f"{summary.coverage_90.mean * 100.0:.2f}% "
        f"["
        f"{summary.coverage_90.ci_low * 100.0:.2f}, "
        f"{summary.coverage_90.ci_high * 100.0:.2f}"
        f"]"
    )

    print(
        "95% nominal coverage: "
        f"{summary.coverage_95.mean * 100.0:.2f}% "
        f"["
        f"{summary.coverage_95.ci_low * 100.0:.2f}, "
        f"{summary.coverage_95.ci_high * 100.0:.2f}"
        f"]"
    )

    print(
        "99% nominal coverage: "
        f"{summary.coverage_99.mean * 100.0:.2f}% "
        f"["
        f"{summary.coverage_99.ci_low * 100.0:.2f}, "
        f"{summary.coverage_99.ci_high * 100.0:.2f}"
        f"]"
    )

    print()

    print(
        "Squared-error calibration slope: "
        f"{summary.calibration_slope:.4f}"
    )

    print(
        "Calibration interpretation: "
        f"{summary.calibration_label}"
    )


def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(
        description=(
            "Run formal simulated localisation-"
            "uncertainty calibration."
        )
    )

    parser.add_argument(
        "--samples-per-viewpoint",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--candidate-viewpoints",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "results/"
            "supplementary_uncertainty_calibration"
        ),
    )

    args = (
        parser.parse_args()
    )

    config = CalibrationConfig(
        samples_per_viewpoint=(
            args.samples_per_viewpoint
        ),
        candidate_viewpoints_per_scenario=(
            args.candidate_viewpoints
        ),
        bootstrap_samples=(
            args.bootstrap_samples
        ),
    )

    result = run_uncertainty_calibration(
        config,
        show_progress=True,
    )

    print_calibration_summary(
        result
    )

    raw_path, summary_path = (
        save_calibration_outputs(
            result,
            output_directory=(
                args.output
            ),
        )
    )

    print()

    print(
        "Raw calibration samples saved to: "
        f"{raw_path}"
    )

    print(
        "Calibration summary saved to: "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()