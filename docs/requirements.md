Research System Requirements

1. Purpose

This document defines the functional and non-functional requirements for the completed simulation framework used to investigate task-aware active perception coupled with uncertainty-aware motion planning for simulated minimally invasive surgical navigation.

Requirements are individually identifiable and traceable to implementation, automated tests, experimental evidence, or research documentation.

Verification applies only to the implemented simulation framework. It does not constitute clinical validation, regulatory approval, or medical-device certification.

2. Functional Requirements

REQ-01 — Reproducible Surgical Workspace

The system shall generate reproducible simulated minimally invasive surgical workspaces containing explicit hidden ground-truth geometry.

REQ-02 — Target and Safety-Critical Structures

The environment shall contain a target, one or more safety-critical structures, and defined start and goal conditions for motion planning.

REQ-03 — Explicit Coordinate Frames

The system shall maintain explicit world, camera, and instrument-related coordinate frames using validated rigid-body transformations.

REQ-04 — Camera Observation

The system shall generate simulated observations of the surgical workspace from defined camera viewpoints.

REQ-05 — Controlled Perception Degradation

The system shall support reproducible manipulation of perception quality, including localisation uncertainty, visibility, occlusion, viewpoint geometry, and synthetic visual-quality degradation.

REQ-06 — Perception Output

The perception subsystem shall provide estimated spatial information required by downstream motion planning.

REQ-07 — Explicit Uncertainty Representation

Perceived spatial estimates shall contain an explicit localisation-uncertainty representation suitable for downstream planning and quantitative calibration analysis.

REQ-08 — Fixed-View Baseline

The system shall implement a Fixed View strategy without active camera repositioning.

REQ-09 — Generic Active-Perception Baseline

The system shall implement a task-agnostic active-perception strategy using quantitative global observation utility.

REQ-10 — Task-Aware Active Perception

The system shall implement an active-perception strategy that incorporates intended trajectory or task-relevant safety information into viewpoint selection.

REQ-11 — Candidate Viewpoint Evaluation

Candidate camera viewpoints shall be evaluated using quantitative scoring objectives.

REQ-12 — Motion Planning

The system shall generate geometrically feasible candidate trajectories between defined start and goal states using perceived environmental information.

REQ-13 — Uncertainty-Aware Planning

Localisation uncertainty shall influence planning through uncertainty-dependent protected geometry or equivalent safety mechanisms.

REQ-14 — Hidden Ground-Truth Safety Evaluation

Planned trajectories shall be independently evaluated against simulator ground-truth geometry unavailable to the planner.

REQ-15 — Safety-Margin Violation Detection

The system shall identify whether a trajectory enters a predefined protected region around safety-critical anatomy.

REQ-16 — Collision Detection

The system shall identify geometric collision between the instrument trajectory and critical simulated structures.

REQ-17 — Quantitative Metrics

Experimental trials shall expose appropriate perception, planning, safety, and efficiency metrics, including localisation error, predicted uncertainty, camera movement, planning success, safe-navigation success, collisions, safety violations, clearance, planner iterations, planning time, and path cost.

REQ-18 — Experimental Logging

Trials shall produce machine-readable records containing sufficient information to identify scenario, strategy, perturbation condition, random seed, relevant intermediate outputs, and final metrics.

REQ-19 — Matched Three-Strategy Comparison

Equivalent simulated conditions shall be reused across Fixed View, Generic Active Perception, and Task-Aware Active Perception through a common:

perception
->
localisation uncertainty
->
motion planning
->
hidden ground-truth safety evaluation

pipeline to support matched statistical comparison.

REQ-20 — Automated Experiment Execution

The framework shall support repeated execution of predefined experiments without manual intervention between individual trials.

3. Non-Functional Requirements

NFR-01 — Modularity

Geometry, robotics, perception, active perception, uncertainty, planning, evaluation, and experiment logic shall remain separable software components.

NFR-02 — Reproducibility

Recorded configuration and controlled random seeds shall support reproduction of stochastic simulation conditions within the deterministic limits of the software environment.

NFR-03 — Traceability

Requirements, implementation, tests, experiment configurations, generated artifacts, and repository revision history shall remain traceable. Final evidence artifacts shall include reproducibility metadata or hashes where applicable.

NFR-04 — Testability

Critical mathematical, algorithmic, integration, and experiment components shall be independently testable.

NFR-05 — Numerical Robustness

Geometric and transformation operations shall use appropriate numerical tolerances rather than inappropriate exact floating-point equality.

NFR-06 — Quantitative Evaluation

Research conclusions shall be supported by quantitative experimental evidence rather than visual demonstrations alone.

NFR-07 — Robustness Evaluation

The framework shall support evaluation across multiple geometry, occlusion, uncertainty, safety-margin, and visual-quality conditions.

NFR-08 — Extensibility

Alternative perception models, uncertainty representations, viewpoint-selection methods, or motion planners should be introducible without redesigning the complete framework.

NFR-09 — Computational Observability

Intermediate outputs required for debugging and analysis shall be available for logging, including selected viewpoints, scores, uncertainty estimates, planning outcomes, iterations, path cost, clearance, and safety outcomes.

NFR-10 — Interpretation Safety

Simulation outcomes shall not be represented as clinically validated thresholds, patient-risk estimates, evidence of clinical effectiveness, or evidence of medical-device safety.

4. Verification Baseline

The completed local Python regression baseline is:

449 passed
0 failed

Verification spans:

geometry and coordinate transformations;

RCM-constrained kinematics;

workspace and safety geometry;

collision-aware planning;

trajectory processing;

uncertainty modelling;

camera and observation modelling;

visibility and occlusion;

Generic Active Perception;

task relevance and Task-Aware Active Perception;

perception-to-planning integration;

three-strategy end-to-end navigation;

matched statistical benchmarking;

multi-scenario robustness;

mechanism ablation;

uncertainty stress testing;

final evidence consolidation;

formal uncertainty calibration;

synthetic visual-quality degradation;

planning-efficiency comparison.

Requirement-to-evidence mapping is maintained in docs/traceability_matrix.md.

5. Scope Boundary

The completed research system does not claim or require:

patient data;

physical surgical robot validation;

deformable tissue modelling;

force or tactile sensing;

autonomous cutting or suturing;

animal or cadaver validation;

clinical validation;

regulatory certification.

These remain outside the scope of the project.