# Requirements Verification Traceability Matrix

## 1. Purpose

This document provides bidirectional traceability between the research-system requirements, software implementation, automated verification, and experimental evidence for the uncertainty-aware surgical navigation framework.

The matrix complements:

* `docs/requirements.md`
* `docs/verification_plan.md`
* `docs/architecture.md`

Its purpose is to distinguish clearly between:

* requirements that are currently verified;
* requirements that have partial evidence;
* requirements that remain dependent on the final end-to-end experiment.

Verification refers to computational/software behaviour only. It does not constitute clinical validation or medical-device certification.

---

## 2. Status Definitions

| Status                     | Meaning                                                                                                                                 |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **Verified**               | Direct implementation and automated or quantitative verification evidence currently exist.                                              |
| **Partially Verified**     | Substantial implementation/evidence exists, but one part of the requirement still requires additional integration or validation.        |
| **Pending**                | The requirement depends primarily on work that has not yet been completed.                                                              |
| **Documentation Verified** | Evidence is principally architectural, reproducibility, configuration, or interpretation documentation rather than algorithmic testing. |

---

# 3. Functional Requirements Traceability

## REQ-01 — Reproducible Surgical Workspace

**Requirement:** The system shall generate a reproducible simulated minimally invasive surgical workspace containing defined ground-truth geometry.

**Implementation evidence**

* `src/geometry/workspace.py`
* simulation scene/configuration modules

**Verification evidence**

* `tests/test_workspace.py`
* deterministic seeded simulation behaviour

**Status:** **Verified**

---

## REQ-02 — Target and Critical Structures

**Requirement:** The environment shall contain a target region, safety-critical structures, and defined planning start/goal conditions.

**Implementation evidence**

* workspace geometry
* simulation scene definitions
* safety-critical structure representation

**Verification evidence**

* `tests/test_workspace.py`
* `tests/test_safety.py`
* `tests/test_safety_critical_benchmark.py`

**Status:** **Verified**

---

## REQ-03 — Explicit Coordinate Frames

**Requirement:** Explicit world, camera, and instrument-related coordinate frames shall be maintained using validated rigid transformations.

**Implementation evidence**

* `src/geometry/transforms.py`
* camera and instrument geometry

**Verification evidence**

* `tests/test_transforms.py`
* analytical translation/rotation cases
* inverse and round-trip transformation checks

**Status:** **Verified**

---

## REQ-04 — Camera Observation

**Requirement:** The system shall generate simulated observations from defined camera viewpoints.

**Implementation evidence**

* `src/perception/camera.py`
* `src/perception/observation.py`
* `src/perception/viewpoints.py`

**Verification evidence**

* `tests/test_camera.py`
* `tests/test_observation.py`
* `tests/test_viewpoints.py`

**Status:** **Verified**

---

## REQ-05 — Controlled Perception Degradation

**Requirement:** Localisation uncertainty and occlusion shall be controllable and experimental perturbation conditions shall be reproducible.

**Implementation evidence**

* `src/perception/uncertainty.py`
* `src/perception/occlusion.py`
* uncertainty-sensitivity experiment modules
* uncertainty-heterogeneity experiments

**Verification evidence**

* `tests/test_uncertainty.py`
* `tests/test_occlusion.py`
* `tests/test_uncertainty_sensitivity.py`
* `tests/test_uncertainty_heterogeneity_sensitivity.py`

**Status:** **Verified**

---

## REQ-06 — Perception Output

**Requirement:** Perception shall provide estimated spatial information required by downstream planning.

**Implementation evidence**

* `src/perception/perception.py`
* `src/perception/observation.py`
* perception/planning interface

**Verification evidence**

* `tests/test_perception.py`
* `tests/test_observation.py`
* `tests/test_perception_planning.py`

**Status:** **Verified**

---

## REQ-07 — Uncertainty Representation

**Requirement:** Perceived spatial estimates shall contain an explicit uncertainty representation suitable for decision-making.

**Implementation evidence**

* `src/perception/uncertainty.py`
* observation-quality/uncertainty outputs
* uncertainty-aware planning interface

**Verification evidence**

* `tests/test_uncertainty.py`
* `tests/test_perception_planning.py`
* uncertainty-sensitivity tests

**Status:** **Verified**

---

## REQ-08 — Fixed-View Baseline

**Requirement:** A fixed-view strategy shall allow planning/perception evaluation without active camera adjustment.

**Implementation evidence**

* fixed-view benchmark pathway
* active-perception benchmark simulation modules

**Verification evidence**

* `tests/test_active_perception_benchmark.py`

**Status:** **Verified**

---

## REQ-09 — Generic Active-Perception Baseline

**Requirement:** A task-agnostic active-perception strategy shall select viewpoints according to global perception utility.

**Implementation evidence**

* `src/perception/viewpoint_scoring.py`
* `src/perception/active_perception.py`
* `src/perception/closed_loop.py`

**Verification evidence**

* `tests/test_viewpoint_scoring.py`
* `tests/test_active_perception.py`
* `tests/test_active_perception_benchmark.py`
* `tests/test_closed_loop.py`

**Status:** **Verified**

---

## REQ-10 — Task-Aware Active Perception

**Requirement:** Active perception shall incorporate trajectory and safety-critical task relevance into viewpoint selection.

**Implementation evidence**

* `src/perception/task_relevance.py`
* `src/perception/task_aware_scoring.py`
* `src/perception/task_aware_active_perception.py`

**Verification evidence**

* `tests/test_task_relevance.py`
* `tests/test_task_aware_scoring.py`
* `tests/test_task_aware_active_perception.py`
* `tests/test_task_aware_ablation.py`
* `tests/test_task_weight_sensitivity.py`

**Status:** **Verified at component/selection level**

The remaining end-to-end work is tracked separately under REQ-19.

---

## REQ-11 — Candidate Viewpoint Evaluation

**Requirement:** Candidate viewpoints shall be evaluated using a quantitative selection objective.

**Implementation evidence**

* candidate-viewpoint generation
* generic viewpoint scorer
* task-aware viewpoint scorer

**Verification evidence**

* `tests/test_viewpoints.py`
* `tests/test_viewpoint_scoring.py`
* `tests/test_task_aware_scoring.py`

**Status:** **Verified**

---

## REQ-12 — Motion Planning

**Requirement:** The system shall generate geometrically feasible candidate trajectories using perceived environmental information.

**Implementation evidence**

* `src/robotics/planner.py`
* trajectory-processing modules
* perception/planning interface

**Verification evidence**

* `tests/test_planner.py`
* `tests/test_trajectory.py`
* `tests/test_perception_planning.py`

**Status:** **Verified**

---

## REQ-13 — Uncertainty-Aware Planning

**Requirement:** Perception uncertainty shall be capable of modifying planning safety/risk behaviour.

**Implementation evidence**

* uncertainty-dependent planning margins
* uncertainty-aware benchmark framework
* perception/planning coupling

**Verification evidence**

* `tests/test_perception_planning.py`
* `tests/test_safety_critical_benchmark.py`
* `tests/test_statistical_benchmark.py`
* uncertainty benchmark experiments

**Status:** **Verified**

---

## REQ-14 — Ground-Truth Safety Evaluation

**Requirement:** Planned trajectories shall be evaluated independently against simulator ground truth.

**Implementation evidence**

* ground-truth geometry retained independently from perceived geometry
* safety evaluation subsystem

**Verification evidence**

* `tests/test_safety.py`
* `tests/test_safety_critical_benchmark.py`
* `tests/test_statistical_benchmark.py`

**Status:** **Verified**

---

## REQ-15 — Safety-Margin Violation Detection

**Requirement:** The system shall identify entry into a protected region surrounding critical anatomy.

**Implementation evidence**

* robotics safety subsystem
* clearance/safety-margin calculations

**Verification evidence**

* `tests/test_safety.py`
* `tests/test_safety_critical_benchmark.py`

**Status:** **Verified**

---

## REQ-16 — Collision Detection

**Requirement:** The system shall detect geometric intersection between the instrument trajectory and critical simulated structures.

**Implementation evidence**

* collision checking
* configuration safety
* edge safety
* trajectory evaluation

**Verification evidence**

* `tests/test_safety.py`
* `tests/test_planner.py`
* `tests/test_safety_critical_benchmark.py`

**Status:** **Verified**

---

## REQ-17 — Quantitative Metrics

**Requirement:** Safety, perception, planning, and efficiency metrics shall be calculated for experimental trials.

**Implementation evidence**

Experimental modules expose metrics including:

* localisation error;
* uncertainty;
* camera movement;
* planning success;
* planner iterations;
* planning time;
* path cost;
* true clearance;
* safety-margin violation;
* collision outcome.

**Verification evidence**

* `tests/test_active_perception_benchmark.py`
* `tests/test_safety_critical_benchmark.py`
* `tests/test_statistical_results.py`
* `tests/test_statistical_validation.py`

**Status:** **Verified**

---

## REQ-18 — Experimental Logging

**Requirement:** Trials shall retain sufficient machine-readable information to identify conditions, strategy, perturbation, seed, intermediate outputs, and outcomes.

**Implementation evidence**

* benchmark-result data structures
* experiment outputs
* stored result files
* seeded experimental execution

**Current gap**

A single formally defined experiment-record schema linking all final three-strategy trials to software/version metadata has not yet been demonstrated as part of the final experiment pipeline.

**Status:** **Partially Verified**

---

## REQ-19 — Matched Experimental Comparison

**Requirement:** Equivalent scenarios shall be reused across fixed-view, generic active-perception, and task-aware active-perception strategies.

**Existing evidence**

Matched comparisons already exist in individual benchmark and task-aware experiments.

Relevant verification includes:

* `tests/test_active_perception_benchmark.py`
* task-aware benchmark/ablation experiments
* controlled random seeds
* statistical benchmarking utilities

**Remaining gap**

The final unified:

```text
Fixed view
    vs
Generic active perception
    vs
Task-aware active perception
```

comparison has not yet been completed through the same full:

```text
perception
→ uncertainty
→ motion planning
→ ground-truth safety
```

pipeline.

**Status:** **Partially Verified — principal outstanding research requirement**

---

## REQ-20 — Automated Experiment Execution

**Requirement:** Repeated predefined experiments shall execute without manual intervention between trials.

**Implementation evidence**

Automated experiment drivers exist for:

* Monte Carlo evaluation;
* active-perception benchmarking;
* uncertainty sweeps;
* task-aware benchmarking;
* ablation;
* sensitivity analysis;
* statistical validation.

**Verification evidence**

* benchmark tests
* sensitivity tests
* statistical-validation tests

**Status:** **Verified**

---

# 4. Non-Functional Requirements Traceability

## NFR-01 — Modularity

Core concerns are separated into:

```text
geometry
robotics
perception
simulation
ROS 2 integration
```

Further decomposition separates camera modelling, observation, occlusion, uncertainty, generic active perception, task relevance, task-aware scoring, planning, and safety.

**Evidence**

* package architecture
* `docs/architecture.md`
* independently testable modules

**Status:** **Verified**

---

## NFR-02 — Reproducibility

The project uses:

* Python 3.11;
* `environment.yml`;
* controlled random seeds;
* `pytest.ini`;
* automated regression tests.

A completely fresh Conda environment reproduced:

```text
377 passed
0 failed
```

**Status:** **Verified**

---

## NFR-03 — Traceability

Requirements, verification methods, architecture, software modules, and test evidence are explicitly documented.

This document provides requirement-to-evidence traceability.

The final experiment should additionally record software revision identifiers alongside final experiment outputs.

**Status:** **Partially Verified**

---

## NFR-04 — Testability

The project contains independent automated tests spanning:

* geometry;
* robotics;
* safety;
* planning;
* perception;
* uncertainty;
* active perception;
* task awareness;
* integration;
* experiments;
* statistics.

Current regression baseline:

```text
377 passed
0 failed
```

**Status:** **Verified**

---

## NFR-05 — Numerical Robustness

Geometric and transformation operations are evaluated using numerical tolerances rather than inappropriate exact floating-point comparisons.

**Evidence**

* transformation tests;
* geometry tests;
* RCM verification;
* statistical/numerical checks.

**Status:** **Verified**

---

## NFR-06 — Quantitative Evaluation

The framework uses quantitative metrics rather than relying on visual demonstration alone.

Evidence includes:

* Monte Carlo benchmarking;
* safety outcomes;
* uncertainty sweeps;
* task-aware comparisons;
* statistical confidence intervals;
* statistical-validation utilities.

**Status:** **Verified**

---

## NFR-07 — Robustness Evaluation

The framework supports evaluation across multiple perception-degradation conditions.

**Evidence**

* `tests/test_uncertainty_sensitivity.py`
* `tests/test_uncertainty_heterogeneity_sensitivity.py`
* `tests/test_task_weight_sensitivity.py`
* uncertainty sweeps
* ablation experiments

**Status:** **Verified**

---

## NFR-08 — Extensibility

Alternative algorithms can be incorporated through modular interfaces rather than rewriting the complete framework.

Existing evidence includes separate:

* viewpoint scorers;
* perception strategies;
* planning algorithms;
* observation models;
* experiment drivers.

**Status:** **Verified by architecture and implementation structure**

---

## NFR-09 — Computational Observability

Intermediate algorithm outputs are available for experiment logging and analysis, including:

* viewpoint scores;
* selected viewpoints;
* localisation uncertainty;
* localisation error;
* planner success;
* iterations;
* path cost;
* clearance;
* safety outcomes.

**Status:** **Verified**

---

## NFR-10 — Interpretation Safety

The repository explicitly distinguishes simulation evidence from:

* clinical validation;
* patient-risk estimation;
* medical-device safety;
* clinical effectiveness.

This boundary is documented in:

* `README.md`;
* `docs/requirements.md`;
* `docs/verification_plan.md`;
* `docs/architecture.md`.

**Status:** **Documentation Verified**

---

# 5. Traceability Summary

## Functional Requirements

| Status             | Requirements                |
| ------------------ | --------------------------- |
| Verified           | REQ-01–REQ-17, REQ-20       |
| Partially Verified | REQ-18, REQ-19              |
| Pending            | None as isolated components |

The major remaining research gap is therefore not the absence of core algorithms.

It is completion of the **unified end-to-end three-strategy experiment** required to fully close REQ-19 and strengthen REQ-18.

---

## Non-Functional Requirements

| Status                 | Requirements                                                   |
| ---------------------- | -------------------------------------------------------------- |
| Verified               | NFR-01, NFR-02, NFR-04, NFR-05, NFR-06, NFR-07, NFR-08, NFR-09 |
| Partially Verified     | NFR-03                                                         |
| Documentation Verified | NFR-10                                                         |

---

# 6. Verification Baseline

At the time of this traceability update, the core Python framework has been reproduced in a fresh documented environment with:

```text
377 passed
0 failed
```

The regression suite covers component, algorithm, integration, experiment, and statistical behaviour.

ROS 2 validation is maintained separately because ROS 2-specific dependencies require a correctly configured ROS 2 Jazzy environment.

---

# 7. Remaining Closure Actions

The following evidence is still required before all major research requirements can be considered fully closed:

1. implement the unified fixed-view vs generic active-perception vs task-aware active-perception experiment;
2. propagate viewpoint-dependent uncertainty into the common motion-planning pipeline;
3. evaluate all three strategies against identical hidden ground-truth geometry;
4. record common safety, planning, perception, and efficiency metrics;
5. retain matched scenario identifiers and random seeds;
6. define a final machine-readable experiment-record schema;
7. retain the software revision associated with final experimental results;
8. perform final paired statistical comparison.

Once these actions are complete, REQ-18, REQ-19, and NFR-03 can be reassessed for full verification.

---

# 8. Interpretation Boundary

Verification evidence in this matrix demonstrates implementation and computational behaviour relative to the research-system requirements.

It does **not** demonstrate:

* clinical safety;
* clinical effectiveness;
* regulatory compliance;
* medical-device certification;
* suitability for patient use.
