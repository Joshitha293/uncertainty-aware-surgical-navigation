# Research System Requirements

## Functional Requirements

### REQ-01 — Simulation Environment
The system shall provide a reproducible simulated minimally invasive
surgical workspace with known ground-truth geometry.

### REQ-02 — Surgical Scene
The environment shall contain a target region and safety-critical
structures relevant to motion planning.

### REQ-03 — Camera Observation
The system shall generate observations of the simulated workspace from
defined camera viewpoints.

### REQ-04 — Perception Perturbation
The system shall support controlled visual degradation, localisation
uncertainty and occlusion.

### REQ-05 — Fixed-View Baseline
The system shall implement a fixed-view perception strategy.

### REQ-06 — Generic Active Perception
The system shall implement an uncertainty-aware perception strategy that
selects viewpoints according to global uncertainty.

### REQ-07 — Task-Aware Active Perception
The system shall implement a strategy in which uncertainty is weighted
according to relevance to the planned trajectory and safety-critical
structures.

### REQ-08 — Uncertainty Representation
The system shall represent uncertainty associated with perceived target
and obstacle geometry.

### REQ-09 — Viewpoint Selection
The system shall support candidate camera viewpoints and select additional
observations according to the active-perception strategy.

### REQ-10 — Motion Planning
The system shall generate geometrically feasible paths between defined
start and target positions while considering perceived obstacles.

### REQ-11 — Risk-Aware Planning
The planner shall incorporate uncertainty-dependent safety margins or
risk costs.

### REQ-12 — Ground-Truth Evaluation
Planned trajectories shall be evaluated against known ground-truth
critical-structure geometry.

### REQ-13 — Safety Evaluation
The system shall detect safety-margin violations and collisions.

### REQ-14 — Performance Metrics
The system shall calculate predefined safety, perception, planning and
efficiency metrics.

### REQ-15 — Experimental Reproducibility
Experimental scenarios shall be reproducible using stored configurations
and random seeds.

### REQ-16 — Matched Comparison
Equivalent experimental scenarios shall be reused across the three
perception strategies to enable fair comparison.

### REQ-17 — Data Logging
The system shall automatically record experimental parameters,
intermediate outputs and outcome metrics.

### REQ-18 — Verification
Core mathematical, perception, planning and evaluation functions shall
be independently testable.


## Non-Functional Requirements

### NFR-01 — Modularity
Perception, uncertainty estimation, planning, simulation and evaluation
shall be implemented as separable software components.

### NFR-02 — Reproducibility
Experiments shall be reproducible from recorded configuration parameters
and random seeds.

### NFR-03 — Traceability
Experimental results shall be traceable to the corresponding method,
scenario and configuration.

### NFR-04 — Quantitative Evaluation
System performance shall be assessed quantitatively rather than solely
through visual demonstrations.

### NFR-05 — Robustness Testing
The three strategies shall be evaluated across multiple levels of
perception degradation and uncertainty.

### NFR-06 — Software Verification
Critical mathematical and algorithmic components shall include automated
tests where practical.

### NFR-07 — Safety of Interpretation
Simulation parameters and results shall not be presented as clinically
validated safety thresholds.

### NFR-08 — Extensibility
The software architecture should allow later integration of additional
perception models, planners or robotic simulation components without
redesigning the complete system.
