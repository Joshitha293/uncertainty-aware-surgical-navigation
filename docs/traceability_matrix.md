Requirements Verification Traceability Matrix

1. Purpose

This document links the completed research-system requirements to implementation, automated verification, and experimental evidence.

Verification refers to computational behaviour of the simulation framework only and does not constitute clinical or regulatory validation.

2. Final Verification Status

All 20 functional requirements are implemented and computationally verified at project closure.

The final local regression baseline is:

449 passed
0 failed

3. Functional Requirements

Requirement

Principal Implementation / Evidence

Verification Evidence

Status

REQ-01 Reproducible workspace

workspace and scenario construction

workspace tests; robustness scenarios

Verified

REQ-02 Target and critical structures

anatomical/workspace representations; start/goal definitions

workspace, safety, navigation tests

Verified

REQ-03 Coordinate frames

src/geometry/transforms.py; camera/instrument geometry

tests/test_transforms.py; coordinate-frame checks

Verified

REQ-04 Camera observation

camera, observation, viewpoint modules

camera, observation, viewpoint tests

Verified

REQ-05 Controlled degradation

uncertainty, occlusion, heterogeneity and synthetic illumination modules

uncertainty/occlusion tests; illumination benchmark

Verified

REQ-06 Perception output

perception and observation interfaces; common three-strategy interface

perception, observation and integration tests

Verified

REQ-07 Explicit uncertainty

localisation sigma/covariance and observation uncertainty

uncertainty tests; calibration benchmark

Verified

REQ-08 Fixed View

common Fixed View perception pathway

three-strategy perception/navigation tests

Verified

REQ-09 Generic Active Perception

generic viewpoint scorer and controller

viewpoint-scoring and active-perception tests

Verified

REQ-10 Task-Aware Active Perception

task relevance, task-aware scorer/controller

task-aware, ablation and sensitivity tests

Verified

REQ-11 Candidate viewpoint evaluation

viewpoint generation and quantitative scoring

viewpoint and scorer tests

Verified

REQ-12 Motion planning

collision-aware RRT and trajectory modules

planner and trajectory tests

Verified

REQ-13 Uncertainty-aware planning

uncertainty-inflated planning geometry

perception-planning and safety-critical benchmark tests

Verified

REQ-14 Ground-truth evaluation

independent hidden truth geometry and evaluation

navigation, safety and statistical tests

Verified

REQ-15 Safety-margin violations

safety/clearance subsystem

safety and benchmark tests

Verified

REQ-16 Collision detection

configuration/edge/path collision checking

safety and planner tests

Verified

REQ-17 Quantitative metrics

benchmark result structures and analysis modules

statistical, robustness, illumination and efficiency tests

Verified

REQ-18 Experimental logging

CSV/JSON outputs; final evidence package; reproducibility manifest

output-writing tests and generated result artifacts

Verified

REQ-19 Matched three-strategy comparison

unified Fixed / Generic / Task-Aware pipeline

E3-E8 tests and final matched experiments

Verified

REQ-20 Automated experiment execution

benchmark drivers and CLI experiment modules

benchmark and regression tests

Verified

4. Key Experimental Evidence

End-to-End Three-Strategy Validation

The project completed a common:

Fixed View
vs
Generic Active Perception
vs
Task-Aware Active Perception

pipeline with perception uncertainty propagated into planning and hidden ground-truth safety evaluation.

A 100-trial matched fixed-scene benchmark demonstrated strong planning-feasibility differences between strategies.

Status: REQ-13, REQ-14, REQ-17, REQ-19 verified.

Multi-Scenario Robustness

The robustness benchmark evaluated:

10 scenarios
x 10 repetitions
x 3 strategies
= 300 strategy evaluations

Aggregate safe-navigation success:

Fixed:       43%
Generic:     61%
Task-Aware:  97%

Task-Aware planning success was 100% across the tested scenarios.

Status: REQ-05, REQ-17, REQ-19 and NFR-07 verified.

Mechanism Ablation

Generic, alignment-only, uncertainty-only, and full task-aware variants were compared.

The experiments showed that task alignment was the dominant viewpoint-selection mechanism. The explicit uncertainty term did not provide independent selection benefit under the tested observation model.

Status: REQ-10 and REQ-11 verified with mechanism-level evidence.

Uncertainty Stress Testing

Uncertainty-weight values from 0 through 16 and multiple uncertainty profiles were tested.

The selected viewpoint did not change as the explicit uncertainty weighting increased, while uncertainty remained important downstream through uncertainty-inflated planning geometry.

Status: REQ-07, REQ-10, REQ-13 and NFR-06 verified.

Formal Uncertainty Calibration

The calibration experiment evaluated:

10 scenarios
60 viewpoint conditions
6,000 simulated observations

Key results:

Mean normalised squared error:
Observed 3.024
Expected 3.000

95% empirical coverage:
Observed 94.98%
Nominal  95.00%

The model was classified as:

well_calibrated_under_simulation

Artifacts:

results/supplementary_uncertainty_calibration/uncertainty_calibration_samples.csv

results/supplementary_uncertainty_calibration/uncertainty_calibration_summary.json

Status: REQ-07 verified quantitatively.

Synthetic Visual-Quality Degradation

The synthetic observation-quality model used:

sigma_degraded = sigma_nominal / sqrt(quality)

Under the severe condition (quality = 0.25):

Strategy

Planning Success

Safe Navigation

Fixed

35%

35%

Generic

55%

55%

Task-Aware

95%

90%

Artifacts:

results/supplementary_illumination/illumination_trials.csv

results/supplementary_illumination/illumination_summary.json

This is explicitly a synthetic stress test, not a physical lighting model.

Status: REQ-05 and NFR-07 verified.

Planning Efficiency

The final efficiency benchmark used:

10 scenarios
x 5 matched repetitions
= 50 matched units
= 150 strategy evaluations

Metric

Fixed

Generic

Task-Aware

Planning success

46%

66%

100%

Safe navigation

38%

60%

92%

Planning time, all attempts

0.547 s

0.882 s

1.239 s

Successful-plan path cost

2.517

2.558

2.499

On the 33 matched trials where both Generic and Task-Aware planning succeeded:

Task-Aware - Generic path cost:
-0.107
95% CI [-0.257, -0.015]

Artifacts:

results/supplementary_efficiency/three_strategy_efficiency_trials.csv

results/supplementary_efficiency/three_strategy_efficiency_summary.json

Status: REQ-17 and NFR-09 verified.

5. Non-Functional Requirements

Requirement

Evidence

Status

NFR-01 Modularity

separated geometry, robotics, perception, simulation, and ROS 2 components

Verified

NFR-02 Reproducibility

Python 3.11 environment definition, deterministic seeds, reproducible experiment drivers

Verified

NFR-03 Traceability

requirements, tests, result artifacts, final-evidence manifest/hashes, version-controlled repository

Verified

NFR-04 Testability

final regression suite: 449 passing tests

Verified

NFR-05 Numerical robustness

tolerance-based geometric and transformation verification

Verified

NFR-06 Quantitative evaluation

matched trials, confidence intervals, calibration, ablation and robustness analysis

Verified

NFR-07 Robustness evaluation

10-scene robustness, uncertainty stress and synthetic visual degradation

Verified

NFR-08 Extensibility

separate scorers, controllers, observation models, planners and experiment drivers

Verified

NFR-09 Computational observability

scores, uncertainty, planning time, iterations, cost, clearance and safety outputs retained

Verified

NFR-10 Interpretation safety

README, requirements and experiment descriptions explicitly bound claims to simulation

Documentation Verified

6. Final Traceability Summary

Functional Requirements

Verified:  REQ-01 through REQ-20
Pending:   None

Non-Functional Requirements

Verified:               NFR-01 through NFR-09
Documentation Verified: NFR-10
Pending:                None

The previously outstanding unified three-strategy integration, experimental logging, matched statistical evaluation, uncertainty calibration, degradation testing, and planning-efficiency analysis are now complete.

7. Interpretation Boundary

The verification evidence in this matrix demonstrates implementation and computational behaviour relative to the project's simulation requirements.

It does not demonstrate:

clinical safety;

clinical effectiveness;

patient-specific performance;

regulatory compliance;

medical-device certification;

suitability for clinical use.