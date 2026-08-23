"""Tests for E8 final evidence consolidation."""

import json

import pytest

from src.simulation.final_evidence_package import (
    FinalEvidenceConfig,
    build_final_claims,
    build_final_metrics,
    final_limitations,
    generate_final_package,
    validate_evidence,
)


def metric(
    mean,
    low=None,
    high=None,
):
    """Create a compact test metric."""

    if low is None:
        low = mean

    if high is None:
        high = mean

    return {
        "n": 100,
        "mean": mean,
        "ci_low": low,
        "ci_high": high,
    }


def synthetic_evidence():
    """Return minimal evidence satisfying all final claims."""

    common_fixed = {
        "localisation_error": metric(
            0.020
        ),
        "predicted_sigma": metric(
            0.015
        ),
        "camera_movement": metric(
            0.0
        ),
        "planning_success_rate": metric(
            0.45
        ),
        "safe_navigation_success_rate": metric(
            0.43
        ),
    }

    common_generic = {
        "localisation_error": metric(
            0.017
        ),
        "predicted_sigma": metric(
            0.012
        ),
        "camera_movement": metric(
            0.015
        ),
        "planning_success_rate": metric(
            0.63
        ),
        "safe_navigation_success_rate": metric(
            0.61
        ),
    }

    common_task = {
        "localisation_error": metric(
            0.0047
        ),
        "predicted_sigma": metric(
            0.0030
        ),
        "camera_movement": metric(
            0.110
        ),
        "planning_success_rate": metric(
            1.0
        ),
        "safe_navigation_success_rate": metric(
            0.97
        ),
    }

    e4 = {
        "config": {
            "trial_count": 100
        },
        "paired_comparisons": [
            {
                "comparator": "fixed",
                "localisation_error_difference": metric(
                    -0.0003,
                    -0.0004,
                    -0.0002,
                ),
            },
            {
                "comparator": "generic_active",
                "planning_success_rate_difference": metric(
                    0.86,
                    0.79,
                    0.92,
                ),
            },
        ],
    }

    e5 = {
        "scenarios": [
            {
                "scenario_id": index
            }
            for index in range(
                10
            )
        ],
        "summaries": [
            {
                "strategy": "fixed",
                **common_fixed,
                "worst_scenario_safe_navigation_rate": 0.0,
            },
            {
                "strategy": "generic_active",
                **common_generic,
                "worst_scenario_safe_navigation_rate": 0.0,
            },
            {
                "strategy": "task_aware_active",
                **common_task,
                "worst_scenario_safe_navigation_rate": 0.90,
            },
        ],
        "paired_comparisons": [
            {
                "comparator": "fixed",
                "localisation_error_difference": metric(
                    -0.021,
                    -0.034,
                    -0.009,
                ),
                "planning_success_difference": metric(
                    0.55,
                    0.27,
                    0.83,
                ),
                "safe_navigation_success_difference": metric(
                    0.54,
                    0.27,
                    0.81,
                ),
            },
            {
                "comparator": "generic_active",
                "localisation_error_difference": metric(
                    -0.012,
                    -0.022,
                    -0.003,
                ),
                "planning_success_difference": metric(
                    0.37,
                    0.09,
                    0.65,
                ),
                "safe_navigation_success_difference": metric(
                    0.36,
                    0.09,
                    0.63,
                ),
            },
        ],
    }

    e6_generic = {
        "variant": "generic_baseline",
        **common_generic,
    }

    e6_uncertainty = {
        "variant": "uncertainty_only",
        **common_generic,
    }

    e6_alignment = {
        "variant": "alignment_only",
        **common_task,
        "safe_navigation_success_rate": metric(
            1.0
        ),
        "planning_success_rate": metric(
            1.0
        ),
    }

    e6_full = {
        "variant": "full_task_aware",
        **common_task,
        "safe_navigation_success_rate": metric(
            1.0
        ),
        "planning_success_rate": metric(
            1.0
        ),
    }

    e6 = {
        "summaries": [
            e6_generic,
            e6_alignment,
            e6_uncertainty,
            e6_full,
        ],
        "full_comparisons": [
            {
                "comparator": "generic_baseline",
                "safe_navigation_success_difference": metric(
                    0.37,
                    0.09,
                    0.65,
                ),
            },
            {
                "comparator": "alignment_only",
                "localisation_error_difference": metric(
                    0.00002,
                    -0.00010,
                    0.00016,
                ),
            },
            {
                "comparator": "uncertainty_only",
                "safe_navigation_success_difference": metric(
                    0.37,
                    0.09,
                    0.65,
                ),
            },
        ],
    }

    e7_generic = {
        "variant": "generic_baseline",
        **common_generic,
    }

    e7_high_uncertainty = {
        "variant": "high_uncertainty_only",
        **common_generic,
    }

    e7_alignment = {
        "variant": "alignment_only",
        **common_task,
    }

    e7_full = {
        "variant": "full_task_aware",
        **common_task,
    }

    e7 = {
        "selection_summaries": [
            {
                "uncertainty_weight": weight,
                "condition_count": 40,
                "selection_change_rate": 0.0,
                "mean_sigma_change": 0.0,
            }
            for weight in (
                0.0,
                1.0,
                16.0,
            )
        ],
        "variant_summaries": [
            e7_generic,
            e7_high_uncertainty,
            e7_alignment,
            e7_full,
        ],
    }

    return (
        e4,
        e5,
        e6,
        e7,
    )


def test_final_config_uses_expected_evidence_paths():
    config = FinalEvidenceConfig()

    assert (
        "e4_100_trials"
        in str(
            config.e4_path
        )
    )

    assert (
        "e5_robustness"
        in str(
            config.e5_path
        )
    )

    assert (
        "e6_mechanism"
        in str(
            config.e6_path
        )
    )

    assert (
        "e7_uncertainty_stress"
        in str(
            config.e7_path
        )
    )


def test_supported_evidence_passes_validation():
    e4, e5, e6, e7 = (
        synthetic_evidence()
    )

    checks = validate_evidence(
        e4=e4,
        e5=e5,
        e6=e6,
        e7=e7,
    )

    assert len(
        checks
    ) == 4


def test_invalid_robustness_claim_is_rejected():
    e4, e5, e6, e7 = (
        synthetic_evidence()
    )

    fixed_comparison = (
        e5[
            "paired_comparisons"
        ][0]
    )

    fixed_comparison[
        "safe_navigation_success_difference"
    ] = metric(
        0.10,
        -0.10,
        0.30,
    )

    with pytest.raises(
        ValueError,
        match="safe-navigation",
    ):
        validate_evidence(
            e4=e4,
            e5=e5,
            e6=e6,
            e7=e7,
        )


def test_final_metrics_include_e5_e6_and_e7():
    _, e5, e6, e7 = (
        synthetic_evidence()
    )

    rows = build_final_metrics(
        e5=e5,
        e6=e6,
        e7=e7,
    )

    assert len(
        rows
    ) == 11

    assert {
        row["stage"]
        for row in rows
    } == {
        "E5 multi-scenario",
        "E6 mechanism",
        "E7 uncertainty stress",
    }


def test_final_claims_capture_alignment_mechanism():
    _, e5, e6, e7 = (
        synthetic_evidence()
    )

    claims = build_final_claims(
        e5=e5,
        e6=e6,
        e7=e7,
    )

    assert len(
        claims
    ) == 4

    combined = " ".join(
        claim.statement
        for claim in claims
    ).lower()

    assert (
        "task alignment"
        in combined
    )

    assert (
        "uncertainty"
        in combined
    )


def test_final_limitations_prevent_clinical_overclaim():
    limitations = (
        final_limitations()
    )

    combined = " ".join(
        limitations
    ).lower()

    assert (
        "simulation"
        in combined
    )

    assert (
        "clinical"
        in combined
    )

    assert (
        "collision"
        in combined
    )


def test_final_package_files_are_generated(
    tmp_path,
):
    e4, e5, e6, e7 = (
        synthetic_evidence()
    )

    evidence_directory = (
        tmp_path
        / "evidence"
    )

    evidence_directory.mkdir()

    paths = []

    for index, payload in enumerate(
        (
            e4,
            e5,
            e6,
            e7,
        ),
        start=4,
    ):
        path = (
            evidence_directory
            / f"e{index}.json"
        )

        path.write_text(
            json.dumps(
                payload
            ),
            encoding="utf-8",
        )

        paths.append(
            path
        )

    output = (
        tmp_path
        / "final"
    )

    config = FinalEvidenceConfig(
        e4_path=paths[0],
        e5_path=paths[1],
        e6_path=paths[2],
        e7_path=paths[3],
        output_directory=output,
    )

    generated = (
        generate_final_package(
            config
        )
    )

    assert (
        generated.summary_json.exists()
    )

    assert (
        generated.metrics_csv.exists()
    )

    assert (
        generated.report_markdown.exists()
    )

    assert (
        generated
        .reproducibility_manifest
        .exists()
    )

    report = (
        generated
        .report_markdown
        .read_text(
            encoding="utf-8"
        )
    )

    assert (
        "Supported claims"
        in report
    )

    assert (
        "Claims not supported"
        in report
    )

    manifest = json.loads(
        generated
        .reproducibility_manifest
        .read_text(
            encoding="utf-8"
        )
    )

    assert set(
        manifest[
            "evidence_files"
        ]
    ) == {
        "E4",
        "E5",
        "E6",
        "E7",
    }

    for item in (
        manifest[
            "evidence_files"
        ].values()
    ):
        assert len(
            item["sha256"]
        ) == 64