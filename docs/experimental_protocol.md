# Experimental Protocol

## 1. Purpose

This document defines the experimental protocol for evaluating fixed-view perception, generic active perception, and task-aware active perception within the uncertainty-aware surgical-navigation framework.

The purpose is to ensure that final comparisons are controlled, reproducible, statistically defensible, and directly aligned with the project research question.

The protocol is specified before the final end-to-end experiment is executed to reduce post-hoc changes to experimental conditions or evaluation criteria.

---

## 2. Research Question

**Can task-aware uncertainty-driven active perception improve the safety and efficiency of motion planning in simulated minimally invasive surgical environments compared with fixed-view and task-agnostic active perception?**

The primary comparison is:

```text
Fixed-view perception
        vs
Generic active perception
        vs
Task-aware active perception
```

All three strategies must operate under matched simulated conditions and be evaluated using the same ground-truth safety and planning metrics.

---

## 3. Experimental Hypotheses

### H1 — Task-aware perception and localisation

Task-aware active perception will reduce task-relevant localisation uncertainty relative to fixed-view perception and generic active perception.

### H2 — Navigation safety

Task-aware active perception will reduce ground-truth safety-margin violations relative to fixed-view perception and generic active perception.

### H3 — True clearance

Task-aware active perception will increase true minimum anatomical clearance along successfully planned trajectories.

### H4 — Planning robustness

Improved task-relevant perception will increase or maintain planning success under moderate perception uncertainty.

### H5 — Efficiency trade-off

Task-aware active perception may incur additional sensing or computation cost, including camera movement and viewpoint-selection time.

The project therefore does not assume that task-aware perception improves every metric simultaneously.

---

## 4. Experimental Strategies

### Strategy A — Fixed View

The camera remains at a predefined baseline pose.

No active viewpoint selection occurs.

Pipeline:

```text
Fixed camera
    ↓
Observation
    ↓
Estimated anatomy
    ↓
Estimated uncertainty
    ↓
Uncertainty-aware planner
    ↓
Ground-truth evaluation
```

This strategy provides the passive-perception baseline.

---

### Strategy B — Generic Active Perception

Candidate viewpoints are scored using task-agnostic perception utility.

Pipeline:

```text
Candidate viewpoints
    ↓
Generic viewpoint scoring
    ↓
Selected viewpoint
    ↓
Observation
    ↓
Estimated anatomy
    ↓
Estimated uncertainty
    ↓
Uncertainty-aware planner
    ↓
Ground-truth evaluation
```

The generic strategy must not receive the planned trajectory or task-relevance information.

---

### Strategy C — Task-Aware Active Perception

Candidate viewpoints are evaluated using both perception quality and task information.

Pipeline:

```text
Initial task / trajectory
        +
Candidate viewpoints
        ↓
Task relevance
        ↓
Task-aware viewpoint scoring
        ↓
Selected viewpoint
        ↓
Observation
        ↓
Estimated anatomy
        ↓
Estimated uncertainty
        ↓
Uncertainty-aware planner
        ↓
Ground-truth evaluation
```

This is the proposed strategy.

---

## 5. Experimental Unit

One experimental trial consists of one simulated surgical-navigation scenario evaluated independently under all three perception strategies.

A trial includes:

* ground-truth anatomy;
* start configuration;
* goal configuration;
* safety-critical structures;
* initial camera pose;
* candidate viewpoints;
* localisation perturbation;
* random seed;
* planner seed or equivalent controlled stochastic state.

The same scenario must be reused across all three strategies.

---

## 6. Matched-Trial Design

The experiment uses a paired/matched design.

For scenario (i):

```text
Scenario i
 ├── Fixed View
 ├── Generic Active Perception
 └── Task-Aware Active Perception
```

The following must remain identical between strategies:

* ground-truth geometry;
* start state;
* goal state;
* critical anatomy;
* uncertainty condition;
* observation-noise seed where applicable;
* planner stochastic seed where scientifically appropriate;
* candidate viewpoint set.

Only the perception strategy should differ.

This minimises between-strategy confounding.

---

## 7. Independent Variable

The principal independent variable is:

**Perception strategy**

with three levels:

1. Fixed view
2. Generic active perception
3. Task-aware active perception

Secondary experimental factors may include:

* localisation uncertainty magnitude;
* occlusion severity;
* anatomical configuration;
* task trajectory;
* uncertainty composition.

These factors are used for robustness analysis rather than replacing the principal strategy comparison.

---

## 8. Dependent Variables

### 8.1 Primary safety outcomes

The primary outcomes are:

* ground-truth safety-margin violation;
* ground-truth collision;
* minimum true anatomical clearance.

These are prioritised because the central project question concerns safety-critical navigation.

### 8.2 Planning outcomes

Planning outcomes include:

* planning success;
* planning time;
* planner iterations;
* path cost;
* trajectory length where applicable;
* failure mode.

### 8.3 Perception outcomes

Perception outcomes include:

* localisation error;
* predicted localisation uncertainty;
* selected viewpoint;
* visibility/occlusion quality;
* task relevance;
* task alignment.

### 8.4 Active-perception cost

Active-perception overhead includes:

* camera movement distance;
* viewpoint-selection computation time;
* number of viewpoints evaluated.

This prevents perception improvement from being reported without considering its cost.

---

## 9. Primary Outcome

The primary outcome for the final navigation comparison is:

**ground-truth safety-margin violation rate**

because a trajectory can be free from physical collision while still entering an unacceptable protected region around critical anatomy.

Physical collision remains an important secondary safety outcome.

---

## 10. Ground-Truth Isolation

Ground-truth anatomy must never be supplied directly to:

* the viewpoint-selection policy;
* the perception estimate;
* the uncertainty-aware planner.

Ground truth may be used only by:

* the observation simulator where required to generate synthetic measurements;
* the independent evaluation stage.

Conceptually:

```text
GROUND TRUTH
    ↓
synthetic observation
    ↓
PERCEIVED STATE
    ↓
decision-making

GROUND TRUTH
    ↓
independent evaluation
```

This separation is mandatory.

---

## 11. Controlled Variables

Unless deliberately varied in a sensitivity experiment, the following should remain fixed across matched strategy comparisons:

* robot model;
* RCM constraint;
* planning algorithm;
* planner hyperparameters;
* collision-check resolution;
* anatomical safety margin;
* uncertainty multiplier;
* camera intrinsics;
* candidate viewpoint-generation parameters;
* task-relevance formulation;
* workspace limits;
* trajectory-cost definition.

Any change to these values must be recorded in experimental metadata.

---

## 12. Randomness and Seeds

All stochastic processes must use explicit recorded seeds.

A master trial seed should deterministically generate any required component seeds.

Example:

```text
trial_seed
    ├── perception_seed
    ├── planner_seed
    └── scenario_seed
```

The same scenario-level stochastic conditions should be used for all three strategies wherever doing so preserves scientific fairness.

Seeds must be stored with the experimental output.

---

## 13. Number of Trials

The definitive comparison should use substantially more than a single demonstration trial.

A target of at least:

```text
100 matched scenarios
×
3 strategies
=
300 strategy evaluations
```

should be used for the principal experiment where computationally practical.

Additional sensitivity analyses may increase the total number of simulations substantially.

The final sample size should be reported explicitly rather than inferred from aggregate results.

---

## 14. Uncertainty Conditions

The main benchmark should use one predefined representative uncertainty condition selected before final analysis.

Additional robustness experiments should examine multiple localisation uncertainty levels.

Example levels:

```text
1 mm
3 mm
5 mm
8 mm
10 mm
```

The exact levels should remain physically interpretable within the simplified simulation model.

Sensitivity experiments must be distinguished from the primary comparison.

---

## 15. Occlusion Conditions

Where occlusion is studied, conditions should be defined systematically.

Possible categories include:

* no occlusion;
* mild occlusion;
* moderate occlusion;
* severe occlusion.

Occlusion parameters must be generated or configured consistently across matched strategies.

---

## 16. Trial Validity

A trial is considered valid when:

* the scenario is generated successfully;
* required anatomical geometry is valid;
* the start state is valid;
* the goal state is valid;
* candidate viewpoints are generated as required;
* the perception strategy executes without software error;
* the planner returns either a defined success or defined failure outcome.

Planner failure is generally an **experimental result**, not grounds for deleting a trial.

---

## 17. Exclusion Policy

Trials must not be excluded simply because they produce undesirable results.

Valid exclusion reasons include:

* corrupted experimental output;
* software exception caused by a confirmed implementation defect;
* invalid scenario generation that violates predefined workspace constraints;
* missing required metadata.

Any excluded trial must retain:

* trial identifier;
* exclusion reason;
* affected strategy;
* relevant diagnostic information.

Where possible, matched trials should be treated consistently across strategies.

---

## 18. Failure Handling

Failures should be categorised rather than silently removed.

Possible failure categories include:

```text
PERCEPTION_FAILURE
NO_VALID_VIEWPOINT
PLANNING_FAILURE
COLLISION
SAFETY_MARGIN_VIOLATION
NUMERICAL_FAILURE
SOFTWARE_ERROR
```

This makes it possible to distinguish algorithmic failures from infrastructure failures.

---

## 19. Experimental Record

Each strategy evaluation should generate a machine-readable record containing at least:

```text
experiment_id
trial_id
strategy
scenario_seed
perception_seed
planner_seed

uncertainty_condition
occlusion_condition

start_configuration
goal_configuration

selected_viewpoint
camera_movement

localisation_error
predicted_uncertainty

planning_success
planning_time
planner_iterations
path_cost

true_minimum_clearance
safety_margin_violation
collision

software_revision
timestamp
```

Additional intermediate metrics may be stored where useful.

---

## 20. Software Revision

Final experimental results must be associated with the Git revision that generated them.

For example:

```bash
git rev-parse HEAD
```

The resulting commit hash should be included in the experiment metadata.

This prevents later code changes from becoming indistinguishable from the version used to produce the reported results.

---

## 21. Statistical Analysis

Because all three strategies are evaluated on matched scenarios, paired statistical comparisons should be preferred where appropriate.

### Continuous outcomes

Examples:

* localisation error;
* true clearance;
* path cost;
* planning time;
* camera movement.

Analysis should report:

* mean;
* median;
* standard deviation;
* interquartile range where useful;
* 95% confidence interval;
* paired strategy difference;
* effect size where appropriate.

### Binary outcomes

Examples:

* planning success;
* collision;
* safety-margin violation.

Report:

* counts;
* proportions;
* absolute difference;
* relative difference where meaningful;
* uncertainty/confidence interval.

Matched binary comparisons should account for the paired experimental design.

---

## 22. Multiple Comparisons

The principal comparisons are predefined:

```text
Task-aware vs Fixed
Task-aware vs Generic
Generic vs Fixed
```

The analysis should distinguish primary hypotheses from exploratory comparisons.

If many secondary statistical tests are introduced, appropriate multiple-comparison considerations should be documented.

---

## 23. Effect Size

Statistical significance alone is insufficient.

Where suitable, results should also report the magnitude of improvement.

Examples:

```text
absolute clearance improvement
relative reduction in violation rate
paired localisation-error reduction
planning-time increase
camera-movement increase
```

This makes engineering importance easier to interpret.

---

## 24. Confidence Intervals

Confidence intervals should accompany major estimates wherever practical.

For example:

```text
Mean paired clearance improvement:
X mm
95% CI [lower, upper]
```

The objective is to quantify uncertainty in the experimental result itself.

---

## 25. Sensitivity Analysis

Sensitivity experiments should evaluate whether conclusions remain stable when key assumptions vary.

Existing/target sensitivity dimensions include:

* task weight;
* localisation uncertainty magnitude;
* uncertainty composition;
* occlusion;
* planning conservatism.

Sensitivity analyses should not replace the predefined primary experiment.

---

## 26. Ablation Analysis

Ablation experiments should determine which task-aware components contribute to observed behaviour.

Relevant variants may include:

```text
Generic baseline
Alignment only
Uncertainty only
Task relevance only
Full task-aware strategy
```

All ablations should use matched scenarios.

---

## 27. Reproducibility

The core environment is defined using:

```text
environment.yml
```

Tests are configured using:

```text
pytest.ini
```

The validated baseline is:

```text
377 passed
0 failed
```

in a fresh Python 3.11 Conda environment.

Experimental execution must use the documented environment or an explicitly recorded equivalent.

---

## 28. Automated Verification Before Experiment Execution

Before definitive experiments are generated, the complete regression suite must pass.

Required precondition:

```text
python -m pytest -q

377 passed
0 failed
```

If the regression suite fails, definitive experimental outputs should not be generated until the failure is understood and resolved.

---

## 29. Experimental Output Organisation

Recommended structure:

```text
results/
└── final_experiment/
    ├── metadata/
    ├── raw/
    ├── processed/
    ├── figures/
    └── summary/
```

### `raw/`

Immutable per-trial machine-readable results.

### `processed/`

Derived analysis tables.

### `figures/`

Generated plots.

### `metadata/`

Experiment configuration and software revision.

### `summary/`

Aggregate statistics and human-readable summaries.

Raw experiment files should not be manually edited after generation.

---

## 30. Figure Generation

Publication-quality figures should be generated programmatically from stored raw or processed data.

Potential final figures include:

* safety-violation rate by strategy;
* minimum true clearance distribution;
* localisation-error distribution;
* planning-success rate;
* path-cost comparison;
* planning-time comparison;
* camera-movement comparison;
* uncertainty-level sensitivity curves.

Figures should not rely on manually entered numerical values.

---

## 31. Interpretation Rules

The following distinctions must be preserved when interpreting results.

### Better perception does not automatically mean safer planning

Navigation outcomes must be evaluated directly.

### Planner success does not mean safety

Successful trajectories must still be checked against hidden ground truth.

### Statistical significance does not automatically mean engineering significance

Effect magnitude must also be considered.

### Simulation success does not imply clinical effectiveness

All conclusions remain restricted to the implemented simulation model.

---

## 32. Decision Criteria

The task-aware strategy will be considered supported by the experiment if it demonstrates a consistent and practically meaningful improvement in safety-related navigation outcomes relative to the comparison strategies without unacceptable degradation of planning feasibility or computational efficiency.

A mixed result is valid.

For example:

```text
lower safety-violation rate
+
higher true clearance
+
greater camera movement
+
slightly greater computation
```

would represent an interpretable safety–efficiency trade-off rather than a failed hypothesis.

---

## 33. Negative Results

The protocol explicitly allows negative findings.

If task-aware active perception:

* does not outperform generic active perception;
* worsens planning success;
* increases safety violations;
* provides negligible effect size;

the result must be retained and analysed.

Experimental parameters should not be modified retrospectively solely to manufacture a favourable outcome.

---

## 34. Final Experimental Comparison

The final comparison will therefore follow:

```text
Generate matched surgical scenario
              ↓
        Fixed View
        Generic AP
       Task-Aware AP
              ↓
     Perception estimate
              ↓
     Explicit uncertainty
              ↓
  Uncertainty-aware planning
              ↓
Hidden ground-truth evaluation
              ↓
   Common outcome metrics
              ↓
     Paired statistics
              ↓
     Sensitivity checks
```

---

## 35. Protocol Freeze

Once the definitive experimental campaign begins, the following should be treated as frozen unless a documented implementation defect is discovered:

* primary hypotheses;
* primary outcome;
* strategy definitions;
* main uncertainty condition;
* safety-margin definition;
* primary planning configuration;
* exclusion policy;
* principal statistical comparisons.

Any subsequent change must be documented with:

* reason;
* affected experiment version;
* whether previous results were regenerated.

---

## 36. Scope Boundary

This protocol evaluates computational behaviour in a simulated surgical-navigation framework.

It does not constitute:

* clinical experimentation;
* clinical validation;
* regulatory testing;
* verification of a medical device;
* evidence for autonomous patient treatment.
