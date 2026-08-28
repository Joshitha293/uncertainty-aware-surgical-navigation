"""Tests for ground-truth-isolated Phase 1 candidate generation."""

import inspect

import numpy as np

from src.perception.viewpoints import (
    generate_candidate_viewpoints,
)
from src.simulation.phase1_fair_candidate_context import (
    NOMINAL_TARGET_PRIOR,
    build_fair_candidate_context,
    estimated_target_position,
    generate_estimate_centred_candidates,
    make_nominal_initial_pose,
    nominal_initial_candidates,
)
from src.simulation.three_strategy_perception import (
    run_fixed_perception,
)
from src.simulation.three_strategy_robustness_benchmark import (
    RobustnessScenario,
    build_scenario_inputs,
    default_scenarios,
)


def make_inputs():
    """Return one established simulator scenario."""

    return build_scenario_inputs(
        default_scenarios()[0]
    )


def test_nominal_prior_is_finite_three_vector():
    """The initial geometric prior must be explicit and fixed."""

    assert (
        NOMINAL_TARGET_PRIOR.shape
        == (3,)
    )

    assert np.all(
        np.isfinite(
            NOMINAL_TARGET_PRIOR
        )
    )


def test_nominal_initial_pose_does_not_require_truth():
    """Initial camera pose API must not receive anatomical ground truth."""

    signature = inspect.signature(
        make_nominal_initial_pose
    )

    forbidden = (
        "true_structures",
        "target",
        "true_target",
        "occluders",
    )

    for parameter in forbidden:
        assert (
            parameter
            not in signature.parameters
        )


def test_nominal_initial_candidates_are_reproducible():
    """Fixed prior must produce an identical initial candidate set."""

    first = (
        nominal_initial_candidates()
    )

    second = (
        nominal_initial_candidates()
    )

    assert len(
        first
    ) == len(
        second
    )

    for candidate_a, candidate_b in zip(
        first,
        second,
    ):
        assert np.allclose(
            candidate_a.pose.position,
            candidate_b.pose.position,
        )

        assert np.allclose(
            candidate_a.pose.rotation,
            candidate_b.pose.rotation,
        )


def test_initial_pose_is_independent_of_scenario_truth():
    """Changing hidden anatomy must not change the nominal initial pose."""

    first = (
        make_nominal_initial_pose(
            initial_view_index=3
        )
    )

    second = (
        make_nominal_initial_pose(
            initial_view_index=3
        )
    )

    assert np.allclose(
        first.position,
        second.position,
    )

    assert np.allclose(
        first.rotation,
        second.rotation,
    )


def test_estimated_target_comes_from_planner_facing_perception():
    """Candidate centre must be extracted from the noisy estimate."""

    inputs = (
        make_inputs()
    )

    initial_pose = (
        make_nominal_initial_pose(
            initial_view_index=(
                inputs
                .scenario
                .initial_view_index
            )
        )
    )

    initial = (
        run_fixed_perception(
            observation_model=(
                inputs.observation_model
            ),
            initial_pose=(
                initial_pose
            ),
            true_structures=(
                inputs.true_structures
            ),
            seed=12345,
            occluders=(
                inputs.occluders
            ),
        )
    )

    target = (
        estimated_target_position(
            initial.perception_result
        )
    )

    assert np.allclose(
        target,
        initial
        .perception_result
        .estimated_structures[
            0
        ]
        .estimated_centre,
    )


def test_active_candidates_equal_direct_estimate_centred_generation():
    """Active search geometry must be centred on the noisy estimate."""

    inputs = (
        make_inputs()
    )

    initial_pose = (
        make_nominal_initial_pose(
            initial_view_index=(
                inputs
                .scenario
                .initial_view_index
            )
        )
    )

    initial = (
        run_fixed_perception(
            observation_model=(
                inputs.observation_model
            ),
            initial_pose=(
                initial_pose
            ),
            true_structures=(
                inputs.true_structures
            ),
            seed=12345,
            occluders=(
                inputs.occluders
            ),
        )
    )

    estimated_target = (
        estimated_target_position(
            initial.perception_result
        )
    )

    actual = (
        generate_estimate_centred_candidates(
            perception_result=(
                initial.perception_result
            )
        )
    )

    expected = (
        generate_candidate_viewpoints(
            target_position=(
                estimated_target
            )
        )
    )

    assert len(
        actual
    ) == len(
        expected
    )

    for actual_candidate, expected_candidate in zip(
        actual,
        expected,
    ):
        assert np.allclose(
            actual_candidate.pose.position,
            expected_candidate.pose.position,
        )

        assert np.allclose(
            actual_candidate.pose.rotation,
            expected_candidate.pose.rotation,
        )


def test_context_uses_same_estimated_target_for_candidate_generation():
    """Complete context must expose internally consistent search geometry."""

    inputs = (
        make_inputs()
    )

    context = (
        build_fair_candidate_context(
            observation_model=(
                inputs.observation_model
            ),
            true_structures=(
                inputs.true_structures
            ),
            occluders=(
                inputs.occluders
            ),
            initial_view_index=(
                inputs
                .scenario
                .initial_view_index
            ),
            initial_seed=12345,
        )
    )

    expected = (
        generate_candidate_viewpoints(
            target_position=(
                context
                .estimated_target_position
            )
        )
    )

    assert len(
        context.candidates
    ) == len(
        expected
    )

    for actual_candidate, expected_candidate in zip(
        context.candidates,
        expected,
    ):
        assert np.allclose(
            actual_candidate.pose.position,
            expected_candidate.pose.position,
        )


def test_context_is_reproducible_under_fixed_seed():
    """Identical simulation state and seed must reproduce the entire context."""

    inputs = (
        make_inputs()
    )

    kwargs = dict(
        observation_model=(
            inputs.observation_model
        ),
        true_structures=(
            inputs.true_structures
        ),
        occluders=(
            inputs.occluders
        ),
        initial_view_index=(
            inputs
            .scenario
            .initial_view_index
        ),
        initial_seed=12345,
    )

    first = (
        build_fair_candidate_context(
            **kwargs
        )
    )

    second = (
        build_fair_candidate_context(
            **kwargs
        )
    )

    assert np.allclose(
        first.estimated_target_position,
        second.estimated_target_position,
    )

    assert len(
        first.candidates
    ) == len(
        second.candidates
    )

    for first_candidate, second_candidate in zip(
        first.candidates,
        second.candidates,
    ):
        assert np.allclose(
            first_candidate.pose.position,
            second_candidate.pose.position,
        )


def test_hidden_scenario_translation_is_not_used_by_initial_pose():
    """Different hidden targets with the same view index share initial geometry."""

    baseline = (
        RobustnessScenario(
            scenario_id=9000,
            name="candidate_context_baseline",
            translation=(
                0.0,
                0.0,
                0.0,
            ),
            initial_view_index=5,
        )
    )

    translated = (
        RobustnessScenario(
            scenario_id=9001,
            name="candidate_context_translated",
            translation=(
                0.004,
                -0.008,
                0.006,
            ),
            initial_view_index=5,
        )
    )

    baseline_inputs = (
        build_scenario_inputs(
            baseline
        )
    )

    translated_inputs = (
        build_scenario_inputs(
            translated
        )
    )

    baseline_context = (
        build_fair_candidate_context(
            observation_model=(
                baseline_inputs
                .observation_model
            ),
            true_structures=(
                baseline_inputs
                .true_structures
            ),
            occluders=(
                baseline_inputs
                .occluders
            ),
            initial_view_index=5,
            initial_seed=123,
        )
    )

    translated_context = (
        build_fair_candidate_context(
            observation_model=(
                translated_inputs
                .observation_model
            ),
            true_structures=(
                translated_inputs
                .true_structures
            ),
            occluders=(
                translated_inputs
                .occluders
            ),
            initial_view_index=5,
            initial_seed=123,
        )
    )

    assert np.allclose(
        baseline_context
        .initial_pose
        .position,
        translated_context
        .initial_pose
        .position,
    )

    assert np.allclose(
        baseline_context
        .initial_pose
        .rotation,
        translated_context
        .initial_pose
        .rotation,
    )