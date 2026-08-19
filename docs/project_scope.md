# Project Scope

## Project Title

Uncertainty-Aware Active Perception for Safety-Critical Motion Planning
in Minimally Invasive Surgical Robotics

## Engineering Aim

To develop a reproducible simulation framework for investigating how
perception uncertainty propagates into safety-critical motion planning
and whether task-aware active perception can improve navigation safety.

## Core System

The prototype will contain:

- a simulated minimally invasive surgical workspace;
- a target region;
- safety-critical anatomical structures;
- a virtual endoscopic camera;
- controlled visual degradation and occlusion;
- simulated perception and localisation;
- uncertainty estimation;
- task-relevance modelling;
- candidate viewpoint selection;
- motion planning;
- ground-truth collision and safety evaluation;
- automated experimental logging and analysis.

## Experimental Strategies

### Strategy A — Fixed-View Perception

Planning is performed using observations from a fixed camera viewpoint
without active viewpoint adjustment.

### Strategy B — Generic Uncertainty-Aware Active Perception

The system selects additional viewpoints according to global perception
uncertainty without considering the planned task.

### Strategy C — Task-Aware Uncertainty-Driven Active Perception

Perception uncertainty is weighted according to its relevance to the
planned trajectory and safety-critical structures. Additional viewpoints
are selected when uncertainty is likely to affect downstream planning
safety.

## Primary Engineering Question

Can task-aware uncertainty-driven perception reduce safety-margin
violations during motion planning compared with fixed-view and
task-agnostic active perception?

## Scope Boundary

The project is a simulation-based research prototype.

It does not model a complete clinical surgical robotic platform and does
not claim clinical validation or clinical safety.

Geometric constraints and surgical workspace assumptions are simplified
to enable controlled and reproducible investigation of the relationship
between perception uncertainty and motion-planning safety.

## Intended Outputs

- reproducible simulation environment;
- modular perception and planning software;
- uncertainty-aware perception framework;
- task-aware viewpoint selection;
- quantitative experimental dataset;
- statistical comparison of the three strategies;
- figures and visualisations;
- technical documentation;
- research manuscript.
