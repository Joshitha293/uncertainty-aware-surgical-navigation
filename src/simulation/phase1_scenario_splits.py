"""Frozen scenario splits for the Phase 1 fairness experiment.

The ten historical robustness scenarios have already been inspected and used
during algorithm development. They therefore remain DEVELOPMENT scenarios.

Two new deterministic procedural sets are generated:

- validation scenarios;
- untouched held-out test scenarios.

The validation and held-out sets are sampled independently from the same
pre-specified scenario distribution. Hyperparameters must not be tuned using
held-out performance.

The saved manifest and SHA-256 digest provide reproducible evidence that the
scenario definitions were frozen before final testing.
"""

from __future__ import annotations

import argparse
import hashlib
import json

from dataclasses import (
    asdict,
    dataclass,
)
from pathlib import Path

import numpy as np

from src.simulation.three_strategy_robustness_benchmark import (
    RobustnessScenario,
    default_scenarios,
)


@dataclass(frozen=True)
class Phase1SplitConfig:
    """Configuration for deterministic Phase 1 scenario generation."""

    validation_count: int = 20
    held_out_count: int = 30

    validation_seed: int = 20261117
    held_out_seed: int = 20261229

    validation_id_base: int = 1000
    held_out_id_base: int = 2000

    translation_x_min: float = -0.004
    translation_x_max: float = 0.004

    translation_y_min: float = -0.010
    translation_y_max: float = 0.010

    translation_z_min: float = -0.006
    translation_z_max: float = 0.008

    radius_scale_min: float = 0.88
    radius_scale_max: float = 1.12

    safety_margin_scale_min: float = 0.90
    safety_margin_scale_max: float = 1.25

    occluder_probability: float = 0.60

    occluder_radius_min: float = 0.004
    occluder_radius_max: float = 0.014

    initial_view_index_min: int = 0
    initial_view_index_max_exclusive: int = 24

    def __post_init__(self) -> None:
        if self.validation_count <= 0:
            raise ValueError(
                "validation_count must be positive."
            )

        if self.held_out_count <= 0:
            raise ValueError(
                "held_out_count must be positive."
            )

        if not (
            0.0
            <= self.occluder_probability
            <= 1.0
        ):
            raise ValueError(
                "occluder_probability must be in [0, 1]."
            )

        ordered_pairs = (
            (
                self.translation_x_min,
                self.translation_x_max,
            ),
            (
                self.translation_y_min,
                self.translation_y_max,
            ),
            (
                self.translation_z_min,
                self.translation_z_max,
            ),
            (
                self.radius_scale_min,
                self.radius_scale_max,
            ),
            (
                self.safety_margin_scale_min,
                self.safety_margin_scale_max,
            ),
            (
                self.occluder_radius_min,
                self.occluder_radius_max,
            ),
        )

        for lower, upper in ordered_pairs:
            if not (
                np.isfinite(lower)
                and np.isfinite(upper)
            ):
                raise ValueError(
                    "Scenario bounds must be finite."
                )

            if lower >= upper:
                raise ValueError(
                    "Each lower bound must be "
                    "strictly smaller than its upper bound."
                )

        if self.radius_scale_min <= 0.0:
            raise ValueError(
                "radius scales must be positive."
            )

        if self.safety_margin_scale_min <= 0.0:
            raise ValueError(
                "safety-margin scales must be positive."
            )

        if self.occluder_radius_min < 0.0:
            raise ValueError(
                "occluder radii must be non-negative."
            )

        if (
            self.initial_view_index_min < 0
            or self.initial_view_index_min
            >= self.initial_view_index_max_exclusive
        ):
            raise ValueError(
                "Initial-view index range is invalid."
            )


@dataclass(frozen=True)
class Phase1ScenarioSplits:
    """Development, validation and held-out scenario sets."""

    development: tuple[
        RobustnessScenario,
        ...,
    ]

    validation: tuple[
        RobustnessScenario,
        ...,
    ]

    held_out: tuple[
        RobustnessScenario,
        ...,
    ]


def scenario_signature(
    scenario: RobustnessScenario,
) -> tuple:
    """Return a name/id-independent physical scenario signature."""

    return (
        tuple(
            round(
                float(value),
                6,
            )
            for value in scenario.translation
        ),
        round(
            float(
                scenario.radius_scale
            ),
            6,
        ),
        round(
            float(
                scenario.safety_margin_scale
            ),
            6,
        ),
        int(
            scenario.initial_view_index
        ),
        round(
            float(
                scenario.occluder_radius
            ),
            6,
        ),
    )


def _rounded_uniform(
    rng: np.random.Generator,
    lower: float,
    upper: float,
) -> float:
    """Sample and quantise one parameter for stable manifests."""

    return float(
        np.round(
            rng.uniform(
                lower,
                upper,
            ),
            6,
        )
    )


def _sample_one(
    *,
    rng: np.random.Generator,
    scenario_id: int,
    name: str,
    config: Phase1SplitConfig,
) -> RobustnessScenario:
    """Sample one deterministic procedural scene variation."""

    translation = (
        _rounded_uniform(
            rng,
            config.translation_x_min,
            config.translation_x_max,
        ),
        _rounded_uniform(
            rng,
            config.translation_y_min,
            config.translation_y_max,
        ),
        _rounded_uniform(
            rng,
            config.translation_z_min,
            config.translation_z_max,
        ),
    )

    radius_scale = (
        _rounded_uniform(
            rng,
            config.radius_scale_min,
            config.radius_scale_max,
        )
    )

    safety_margin_scale = (
        _rounded_uniform(
            rng,
            config.safety_margin_scale_min,
            config.safety_margin_scale_max,
        )
    )

    initial_view_index = int(
        rng.integers(
            config.initial_view_index_min,
            config.initial_view_index_max_exclusive,
        )
    )

    if (
        float(
            rng.random()
        )
        < config.occluder_probability
    ):
        occluder_radius = (
            _rounded_uniform(
                rng,
                config.occluder_radius_min,
                config.occluder_radius_max,
            )
        )

    else:
        occluder_radius = 0.0

    return RobustnessScenario(
        scenario_id=int(
            scenario_id
        ),
        name=name,
        translation=translation,
        radius_scale=(
            radius_scale
        ),
        safety_margin_scale=(
            safety_margin_scale
        ),
        initial_view_index=(
            initial_view_index
        ),
        occluder_radius=(
            occluder_radius
        ),
    )


def _generate_unique_scenarios(
    *,
    count: int,
    seed: int,
    id_base: int,
    name_prefix: str,
    config: Phase1SplitConfig,
    excluded_signatures: set[
        tuple
    ],
) -> tuple[
    RobustnessScenario,
    ...,
]:
    """Generate unique scenarios excluding all earlier split signatures."""

    rng = np.random.default_rng(
        int(
            seed
        )
    )

    generated: list[
        RobustnessScenario
    ] = []

    used = set(
        excluded_signatures
    )

    attempts = 0
    maximum_attempts = (
        max(
            1000,
            count * 100,
        )
    )

    while len(
        generated
    ) < count:
        if attempts >= maximum_attempts:
            raise RuntimeError(
                "Unable to generate enough unique scenarios."
            )

        index = len(
            generated
        )

        candidate = (
            _sample_one(
                rng=rng,
                scenario_id=(
                    id_base
                    + index
                ),
                name=(
                    f"{name_prefix}_{index:03d}"
                ),
                config=config,
            )
        )

        signature = (
            scenario_signature(
                candidate
            )
        )

        attempts += 1

        if signature in used:
            continue

        used.add(
            signature
        )

        generated.append(
            candidate
        )

    return tuple(
        generated
    )


def build_phase1_splits(
    config: Phase1SplitConfig
    | None = None,
) -> Phase1ScenarioSplits:
    """Build deterministic development/validation/held-out splits."""

    if config is None:
        config = (
            Phase1SplitConfig()
        )

    development = tuple(
        default_scenarios()
    )

    development_signatures = {
        scenario_signature(
            scenario
        )
        for scenario
        in development
    }

    validation = (
        _generate_unique_scenarios(
            count=(
                config.validation_count
            ),
            seed=(
                config.validation_seed
            ),
            id_base=(
                config.validation_id_base
            ),
            name_prefix=(
                "validation"
            ),
            config=config,
            excluded_signatures=(
                development_signatures
            ),
        )
    )

    validation_signatures = {
        scenario_signature(
            scenario
        )
        for scenario
        in validation
    }

    held_out = (
        _generate_unique_scenarios(
            count=(
                config.held_out_count
            ),
            seed=(
                config.held_out_seed
            ),
            id_base=(
                config.held_out_id_base
            ),
            name_prefix=(
                "held_out"
            ),
            config=config,
            excluded_signatures=(
                development_signatures
                | validation_signatures
            ),
        )
    )

    return Phase1ScenarioSplits(
        development=development,
        validation=validation,
        held_out=held_out,
    )


def _manifest_payload(
    *,
    splits: Phase1ScenarioSplits,
    config: Phase1SplitConfig,
) -> dict:
    """Return canonical manifest data."""

    return {
        "experiment": (
            "phase1_examiner_proof_fairness"
        ),
        "split_policy": {
            "development": (
                "Historical scenarios already used "
                "during method development."
            ),
            "validation": (
                "Procedural scenarios used only after "
                "development hyperparameters were selected."
            ),
            "held_out": (
                "Procedural scenarios reserved for final "
                "Phase 1 testing. No parameter tuning is "
                "permitted after inspecting held-out outcomes."
            ),
        },
        "config": asdict(
            config
        ),
        "development": [
            asdict(
                scenario
            )
            for scenario
            in splits.development
        ],
        "validation": [
            asdict(
                scenario
            )
            for scenario
            in splits.validation
        ],
        "held_out": [
            asdict(
                scenario
            )
            for scenario
            in splits.held_out
        ],
    }


def manifest_digest(
    *,
    splits: Phase1ScenarioSplits,
    config: Phase1SplitConfig,
) -> str:
    """Return SHA-256 digest of canonical scenario manifest."""

    payload = (
        _manifest_payload(
            splits=splits,
            config=config,
        )
    )

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        allow_nan=False,
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        canonical
    ).hexdigest()


def save_manifest(
    *,
    splits: Phase1ScenarioSplits,
    config: Phase1SplitConfig,
    output_path: str
    | Path = (
        "results/"
        "phase1_scenario_splits/"
        "scenario_manifest.json"
    ),
) -> tuple[
    Path,
    str,
]:
    """Save immutable scenario definitions plus their digest."""

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = (
        _manifest_payload(
            splits=splits,
            config=config,
        )
    )

    digest = (
        manifest_digest(
            splits=splits,
            config=config,
        )
    )

    payload[
        "sha256"
    ] = digest

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            allow_nan=False,
        )

    return (
        output_path,
        digest,
    )


def _describe_split(
    scenarios: tuple[
        RobustnessScenario,
        ...,
    ],
) -> dict:
    """Calculate compact diagnostics for one split."""

    translations = np.asarray(
        [
            scenario.translation
            for scenario
            in scenarios
        ],
        dtype=float,
    )

    radius = np.asarray(
        [
            scenario.radius_scale
            for scenario
            in scenarios
        ],
        dtype=float,
    )

    safety = np.asarray(
        [
            scenario.safety_margin_scale
            for scenario
            in scenarios
        ],
        dtype=float,
    )

    occlusion = np.asarray(
        [
            scenario.occluder_radius
            for scenario
            in scenarios
        ],
        dtype=float,
    )

    return {
        "count": len(
            scenarios
        ),
        "mean_translation_norm_mm": float(
            np.mean(
                np.linalg.norm(
                    translations,
                    axis=1,
                )
            )
            * 1000.0
        ),
        "radius_scale_range": (
            float(
                np.min(
                    radius
                )
            ),
            float(
                np.max(
                    radius
                )
            ),
        ),
        "safety_margin_scale_range": (
            float(
                np.min(
                    safety
                )
            ),
            float(
                np.max(
                    safety
                )
            ),
        ),
        "occluded_scene_fraction": float(
            np.mean(
                occlusion
                > 0.0
            )
        ),
    }


def print_summary(
    *,
    splits: Phase1ScenarioSplits,
    digest: str,
) -> None:
    """Print frozen split summary."""

    print()

    print(
        "Phase 1 Frozen Scenario Splits"
    )

    print(
        "=============================="
    )

    for name, scenarios in (
        (
            "Development",
            splits.development,
        ),
        (
            "Validation",
            splits.validation,
        ),
        (
            "Held-out",
            splits.held_out,
        ),
    ):
        summary = (
            _describe_split(
                scenarios
            )
        )

        print()

        print(
            name
        )

        print(
            f"  Scenarios:                  "
            f"{summary['count']}"
        )

        print(
            f"  Mean translation magnitude: "
            f"{summary['mean_translation_norm_mm']:.3f} mm"
        )

        print(
            "  Radius-scale range:          "
            f"{summary['radius_scale_range'][0]:.3f} "
            f"to "
            f"{summary['radius_scale_range'][1]:.3f}"
        )

        print(
            "  Safety-margin range:         "
            f"{summary['safety_margin_scale_range'][0]:.3f} "
            f"to "
            f"{summary['safety_margin_scale_range'][1]:.3f}"
        )

        print(
            "  Scenes with occluder:        "
            f"{summary['occluded_scene_fraction'] * 100.0:.1f}%"
        )

    print()

    print(
        "Scenario manifest SHA-256:"
    )

    print(
        f"  {digest}"
    )


def main() -> None:
    """Generate and freeze the Phase 1 split manifest."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic Phase 1 "
            "development, validation and held-out scenarios."
        )
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "results/"
            "phase1_scenario_splits/"
            "scenario_manifest.json"
        ),
    )

    args = parser.parse_args()

    config = (
        Phase1SplitConfig()
    )

    splits = (
        build_phase1_splits(
            config
        )
    )

    output_path, digest = (
        save_manifest(
            splits=splits,
            config=config,
            output_path=(
                args.output
            ),
        )
    )

    print_summary(
        splits=splits,
        digest=digest,
    )

    print()

    print(
        f"Manifest saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()