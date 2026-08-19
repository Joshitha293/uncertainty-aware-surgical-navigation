# Research System Requirements

## 1. Purpose

This document defines the functional and non-functional requirements for the
simulation framework used to investigate uncertainty-aware active perception
and safety-critical motion planning in minimally invasive surgical robotics.

Requirements are written to be individually identifiable, testable and
traceable to implementation and verification evidence.

---

## 2. Functional Requirements

### REQ-01 — Reproducible Surgical Workspace

The system shall generate a reproducible simulated minimally invasive surgical
workspace containing defined ground-truth geometry.

### REQ-02 — Target and Critical Structures

The simulated environment shall contain:

- a target region;
- one or more safety-critical structures;
- defined start and goal conditions for motion planning.

### REQ-03 — Explicit Coordinate Frames

The system shall maintain explicit coordinate frames for relevant simulation
components, including the world, camera and instrument or robot reference
frames.

Transformations between frames shall use validated rigid-body transformation
operations.

### REQ-04 — Camera Observation

The system shall generate simulated observations of the surgical workspace from
defined camera viewpoints.

### REQ-05 — Controlled Perception Degradation

The system shall support controlled manipulation of perception quality,
including localisation uncertainty and occlusion.

Perturbation parameters shall be recorded for every experimental trial.

### REQ-06 — Perception Output

The perception subsystem shall provide estimated spatial information for the
target and safety-critical structures required by downstream planning.

### REQ-07 — Uncertainty Representation

The system shall associate perceived spatial estimates with an explicit
representation of uncertainty suitable for downstream decision-making.

### REQ-08 — Fixed-View Baseline

The system shall implement a fixed-view perception strategy in which motion
planning proceeds without active viewpoint adjustment.

### REQ-09 — Generic Active-Perception Baseline

The system shall implement an active-perception strategy that selects
additional observations according to expected reduction in global perception
uncertainty without task-relevance weighting.

### REQ-10 — Task-Aware Active Perception

The system shall implement an active-perception strategy that weights
uncertainty according to its relevance to the planned trajectory and nearby
safety-critical structures.

### REQ-11 — Candidate Viewpoint Evaluation

The active-perception subsystem shall evaluate a defined set of candidate
camera viewpoints using a quantitative viewpoint-selection objective.

### REQ-12 — Motion Planning

The system shall generate geometrically feasible candidate trajectories between
defined start and target states using perceived environmental information.

### REQ-13 — Uncertainty-Aware Planning

The planning subsystem shall support uncertainty-dependent safety margins,
risk costs, or equivalent mechanisms through which perception uncertainty can
influence trajectory selection.

### REQ-14 — Ground-Truth Safety Evaluation

Planned or executed trajectories shall be independently evaluated against
simulator ground-truth geometry.

### REQ-15 — Safety-Margin Violation Detection

The evaluation subsystem shall determine whether a trajectory enters a
predefined protected region surrounding a critical structure.

### REQ-16 — Collision Detection

The system shall determine whether a trajectory geometrically intersects a
critical simulated structure.

### REQ-17 — Quantitative Metrics

The system shall calculate predefined safety, perception, planning and
efficiency metrics for each experimental trial.

### REQ-18 — Experimental Logging

Each trial shall produce a machine-readable record containing sufficient
information to identify:

- experimental condition;
- perception strategy;
- perturbation parameters;
- random seed;
- relevant intermediate outputs;
- final outcome metrics.

### REQ-19 — Matched Experimental Comparison

Equivalent simulated scenarios shall be reused across the three perception
strategies to support paired or matched statistical comparison.

### REQ-20 — Automated Experiment Execution

The system shall support repeated execution of predefined experimental
conditions without requiring manual intervention between individual trials.

---

## 3. Non-Functional Requirements

### NFR-01 — Modularity

Simulation, geometry, perception, uncertainty estimation, active perception,
planning and evaluation shall be implemented as separable software components
with defined interfaces.

### NFR-02 — Reproducibility

A recorded configuration and random seed shall be sufficient to reproduce a
corresponding stochastic simulation condition within the deterministic limits
of the software environment.

### NFR-03 — Traceability

Experimental outputs shall remain traceable to the method, configuration,
software version and random seed that generated them.

### NFR-04 — Testability

Critical mathematical and algorithmic components shall be independently
testable.

### NFR-05 — Numerical Robustness

Geometric and transformation operations shall use explicit numerical
tolerances where exact floating-point equality is inappropriate.

### NFR-06 — Quantitative Evaluation

Conclusions regarding comparative system performance shall be supported by
quantitative experimental evidence rather than visual demonstrations alone.

### NFR-07 — Robustness Evaluation

The experimental framework shall support evaluation across multiple predefined
levels of perception degradation.

### NFR-08 — Extensibility

The software architecture should permit alternative perception models,
uncertainty representations, viewpoint-selection methods or motion planners to
be introduced without redesigning the complete framework.

### NFR-09 — Computational Observability

Intermediate outputs required to diagnose system behaviour shall be accessible
for logging, visualisation or analysis.

### NFR-10 — Interpretation Safety

Simulation parameters and outcomes shall not be represented as clinically
validated thresholds, patient-risk estimates, or evidence of medical-device
safety.

---

## 4. Requirements Traceability

Each requirement will later be linked to one or more of the following:

- software module;
- unit test;
- integration test;
- simulation experiment;
- quantitative metric;
- generated result or figure.

The corresponding verification method and evidence will be maintained in the
project verification plan.
