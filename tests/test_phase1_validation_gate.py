"""Tests for the corrected frozen Phase 1 validation gate."""

import json

import numpy as np
import pytest

from src.simulation.phase1_scenario_splits import (
    Phase1SplitConfig,
    build_phase1_splits,
    manifest_digest,
    save_manifest,
)
from src.simulation.phase1_validation_gate import (
    FROZEN_GENERIC_MOVEMENT_WEIGHT,
    FROZEN_MANIFEST_SHA256,
    FROZEN_TASK_MOVEMENT_WEIGHT,
    Phase1ValidationRecord,
    VALIDATION_BUDGET_TOLERANCE,
    analyse_validation_records,
    load_frozen_validation_scenarios,
    run_validation_records,
)


def test_frozen_manifest_digest_matches_protocol():
    """Current split generator must reproduce the frozen manifest."""

    config = Phase1SplitConfig()

    splits = build_phase1_splits(
        config
    )

    digest = manifest_digest(
        splits=splits,
        config=config,
    )

    assert (
        digest
        == FROZEN_MANIFEST_SHA256
    )


def test_frozen_movement_weights_are_expected():
    """Corrected development-selected movement weights must remain fixed."""

    assert (
        FROZEN_GENERIC_MOVEMENT_WEIGHT
        == pytest.approx(
            0.072
        )
    )

    assert (
        FROZEN_TASK_MOVEMENT_WEIGHT
        == pytest.approx(
            0.200
        )
    )

    assert (
        VALIDATION_BUDGET_TOLERANCE
        == pytest.approx(
            0.10
        )
    )


def test_manifest_loader_returns_validation_split_only(
    tmp_path,
):
    """Integrity-checked loader must expose validation scenarios only."""

    config = Phase1SplitConfig()

    splits = build_phase1_splits(
        config
    )

    path, _ = save_manifest(
        splits=splits,
        config=config,
        output_path=(
            tmp_path
            / "manifest.json"
        ),
    )

    validation = (
        load_frozen_validation_scenarios(
            path
        )
    )

    assert len(
        validation
    ) == 20

    assert all(
        1000
        <= scenario.scenario_id
        < 2000
        for scenario
        in validation
    )


def test_manifest_tampering_is_rejected(
    tmp_path,
):
    """Changing a frozen scenario without updating its digest must fail."""

    config = Phase1SplitConfig()

    splits = build_phase1_splits(
        config
    )

    path, _ = save_manifest(
        splits=splits,
        config=config,
        output_path=(
            tmp_path
            / "manifest.json"
        ),
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(
            handle
        )

    payload[
        "validation"
    ][
        0
    ][
        "radius_scale"
    ] += 0.001

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
        )

    with pytest.raises(
        ValueError,
        match="integrity",
    ):
        load_frozen_validation_scenarios(
            path
        )


def test_small_validation_run_uses_frozen_weights():
    """Validation runner must preserve corrected development parameters."""

    splits = build_phase1_splits()

    scenarios = tuple(
        splits.validation[
            :2
        ]
    )

    records = (
        run_validation_records(
            scenarios=scenarios,
            repetitions=1,
        )
    )

    assert len(
        records
    ) == 2

    for record in records:
        assert (
            record.generic_movement_weight
            == pytest.approx(
                0.072
            )
        )

        assert (
            record.task_movement_weight
            == pytest.approx(
                0.200
            )
        )


def test_validation_metrics_are_finite():
    """Corrected validation output must remain numerically interpretable."""

    splits = build_phase1_splits()

    records = (
        run_validation_records(
            scenarios=tuple(
                splits.validation[
                    :2
                ]
            ),
            repetitions=1,
        )
    )

    for record in records:
        assert (
            record.generic_camera_movement
            >= 0.0
        )

        assert (
            record.task_camera_movement
            >= 0.0
        )

        assert (
            record.generic_localisation_error
            >= 0.0
        )

        assert (
            record.task_localisation_error
            >= 0.0
        )

        assert (
            record.generic_predicted_sigma
            > 0.0
        )

        assert (
            record.task_predicted_sigma
            > 0.0
        )

        assert (
            0.0
            <= record.task_alignment
            <= 1.0
        )

        assert np.isfinite(
            record.generic_camera_movement
        )

        assert np.isfinite(
            record.task_camera_movement
        )

        assert np.isfinite(
            record.generic_localisation_error
        )

        assert np.isfinite(
            record.task_localisation_error
        )

        assert np.isfinite(
            record.generic_predicted_sigma
        )

        assert np.isfinite(
            record.task_predicted_sigma
        )


def _synthetic_record(
    *,
    generic_movement: float,
    task_movement: float,
) -> Phase1ValidationRecord:
    """Build one compact synthetic validation record."""

    return Phase1ValidationRecord(
        scenario_id=1000,
        scenario_name=(
            "synthetic_validation"
        ),
        repetition=0,
        initial_seed=1,
        final_seed=2,
        generic_movement_weight=(
            0.072
        ),
        task_movement_weight=(
            0.200
        ),
        generic_candidate_index=0,
        task_candidate_index=1,
        different_candidate=True,
        generic_camera_movement=(
            generic_movement
        ),
        task_camera_movement=(
            task_movement
        ),
        generic_localisation_error=(
            0.010
        ),
        task_localisation_error=(
            0.009
        ),
        generic_predicted_sigma=(
            0.008
        ),
        task_predicted_sigma=(
            0.007
        ),
        task_alignment=0.95,
    )


def test_budget_gate_passes_within_tolerance():
    """Validation gate should pass an 8% movement mismatch."""

    records = (
        _synthetic_record(
            generic_movement=0.092,
            task_movement=0.100,
        ),
    )

    summary = (
        analyse_validation_records(
            records=records,
            budget_tolerance=0.10,
        )
    )

    assert (
        summary.relative_budget_gap
        == pytest.approx(
            0.08
        )
    )

    assert (
        summary.budget_gate_passed
        is True
    )


def test_budget_gate_fails_outside_tolerance():
    """Validation gate must report movement mismatch above tolerance."""

    records = (
        _synthetic_record(
            generic_movement=0.075,
            task_movement=0.100,
        ),
    )

    summary = (
        analyse_validation_records(
            records=records,
            budget_tolerance=0.10,
        )
    )

    assert (
        summary.relative_budget_gap
        == pytest.approx(
            0.25
        )
    )

    assert (
        summary.budget_gate_passed
        is False
    )