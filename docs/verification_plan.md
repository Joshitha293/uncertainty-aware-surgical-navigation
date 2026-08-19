# Verification and Validation Plan

## 1. Purpose

This document defines how the research software requirements will be verified
and how the experimental framework will be validated before comparative
conclusions are drawn.

Verification asks:

> Was the computational system implemented correctly according to its
> specified requirements?

Validation asks:

> Does the implemented simulation and experimental framework behave
> appropriately for the research question being investigated?

Verification of software correctness is kept distinct from clinical validation.
This project does not perform clinical validation.

---

## 2. Verification Strategy

Verification will use a combination of:

- unit testing;
- analytical test cases;
- integration testing;
- numerical consistency checks;
- deterministic reproducibility tests;
- automated experiment checks;
- ground-truth simulation comparisons;
- inspection of logged experimental outputs.

Critical mathematical components will be tested independently before they are
used within the integrated experimental pipeline.

---

## 3. Requirements Verification Matrix

| Requirement | Verification Method | Planned Evidence |
|---|---|---|
| REQ-01 | Recreate identical workspace from stored configuration | Configuration and scene-generation test |
| REQ-02 | Inspect generated scene and ground-truth object definitions | Scene test and stored geometry |
| REQ-03 | Analytical and numerical coordinate-transform tests | Automated geometry test suite |
| REQ-04 | Generate observations from predefined viewpoints | Camera integration test |
| REQ-05 | Apply predefined perturbation levels and verify recorded parameters | Perturbation tests and trial logs |
| REQ-06 | Compare perception outputs with simulator ground truth | Localisation-error tests |
| REQ-07 | Verify uncertainty representation against controlled perturbations | Uncertainty calibration/evaluation outputs |
| REQ-08 | Execute fixed-view strategy without viewpoint modification | Strategy integration test |
| REQ-09 | Verify generic strategy responds to global uncertainty | Viewpoint-selection trace |
| REQ-10 | Verify task-aware strategy changes weighting according to trajectory relevance | Task-relevance tests and selection trace |
| REQ-11 | Evaluate known candidate viewpoints and independently reproduce objective scores | Viewpoint-objective test |
| REQ-12 | Test path generation in analytically simple and randomised environments | Planner test suite |
| REQ-13 | Verify changes in uncertainty modify the defined planning risk representation | Risk-model tests |
| REQ-14 | Independently evaluate generated paths against ground-truth geometry | Ground-truth evaluation test |
| REQ-15 | Test known safe and violating trajectories | Safety-margin tests |
| REQ-16 | Test known intersecting and non-intersecting trajectories | Collision-detection tests |
| REQ-17 | Independently calculate selected metrics for controlled cases | Metric unit tests |
| REQ-18 | Inspect machine-readable trial records for required metadata and outputs | Logging integration test |
| REQ-19 | Verify matched scenario identifiers and seeds across strategies | Experiment-pairing check |
| REQ-20 | Execute a predefined batch without per-trial manual intervention | Automated experiment test |

---

## 4. Non-Functional Verification

### Reproducibility

Selected stochastic experiments will be repeated using identical configuration
parameters and random seeds.

Outputs expected to be deterministic will be compared using exact equality
where appropriate. Floating-point quantities will be compared using explicitly
defined numerical tolerances.

### Numerical Robustness

Rigid transformations, geometric calculations, distance computations and
planning metrics will be tested using analytically known cases and numerical
tolerance checks.

### Modularity

Core subsystems will expose defined interfaces and will be independently
testable where practical.

### Traceability

Experimental records will contain identifiers linking results to:

- strategy;
- scenario;
- perturbation condition;
- random seed;
- relevant configuration;
- software version where practical.

---

## 5. Simulation Validation

Before the main comparative experiments, the simulation framework will undergo
sanity and boundary-condition checks.

These will include:

1. zero-noise conditions;
2. zero-occlusion conditions;
3. known unobstructed paths;
4. deliberately colliding paths;
5. known safety-margin violations;
6. increasing localisation perturbation;
7. repeated identical seeded scenarios.

Expected qualitative behaviour will be defined before evaluating each check.

For example, zero localisation noise should not produce artificial localisation
error beyond numerical precision, while increasing imposed localisation noise
should increase the distribution of localisation error under the corresponding
noise model.

---

## 6. Algorithm Verification

Where possible, algorithmic outputs will be checked against analytically
tractable cases before testing more complex randomised environments.

Examples include:

- identity coordinate transformations;
- known rigid translations and rotations;
- transformation inversion and round-trip consistency;
- analytically known point-to-structure distances;
- simple collision and non-collision trajectories;
- deterministic planner scenarios;
- controlled uncertainty inputs.

---

## 7. Integration Verification

Subsystems will be integrated progressively rather than only tested after the
complete pipeline has been assembled.

Planned integration sequence:

1. geometry and coordinate transformations;
2. simulated scene representation;
3. instrument and planning geometry;
4. camera observation;
5. perception;
6. uncertainty estimation;
7. motion planning;
8. active viewpoint selection;
9. ground-truth safety evaluation;
10. automated experiment execution and logging.

Each stage must produce plausible and testable outputs before the next major
subsystem is introduced.

---

## 8. Experimental Integrity Checks

Before statistical analysis, automated checks will confirm where applicable:

- expected number of trials;
- absence of unintended duplicate trials;
- valid scenario identifiers;
- matched conditions across strategies;
- valid random seeds;
- expected perturbation levels;
- absence of missing critical outcome fields;
- finite numerical outputs;
- valid metric ranges.

Failed or invalid trials will not be silently removed. Any exclusion rule used
during analysis will be documented and reported.

---

## 9. Evidence Retention

Verification evidence will be retained through:

- automated tests;
- experiment configuration files;
- machine-readable logs;
- generated figures;
- summary tables;
- analysis outputs;
- documented software versions.

This provides a traceable path from system requirement to implementation,
verification and experimental evidence.

---

## 10. Interpretation Boundary

Successful verification demonstrates that the research software behaves
according to its computational specification.

It does not demonstrate clinical safety, surgical effectiveness, medical-device
performance, or suitability for patient use.
