# Uncertainty-Aware Active Perception for Safety-Critical Motion Planning in Minimally Invasive Surgical Robotics

A simulation-based research engineering framework investigating how **perception uncertainty, task-aware active perception, and safety-critical motion planning** interact in minimally invasive surgical robotics.

The project combines collision-aware motion planning, uncertainty modelling, camera/viewpoint simulation, task-aware perception, statistical benchmarking, and ROS 2 integration in a reproducible software framework.

> **Research prototype:** This repository is intended for simulation and engineering research. It is not a clinical system or medical device.

---

## Research Question

**Can task-aware uncertainty-driven active perception improve localisation accuracy for safety-critical surgical navigation compared with generic active perception, while making the perception–motion trade-off explicit?**

The final experimental framework evaluates whether selecting camera viewpoints using surgical-task information changes the perception outcome in a measurable and reproducible way.

---

## Core Contribution

The project develops a pipeline in which:

1. A collision-aware RRT planner generates a surgical trajectory.
2. The planned trajectory defines a **task-relevance field**.
3. Candidate camera viewpoints are generated around the surgical workspace.
4. Each viewpoint is evaluated for observation quality and localisation uncertainty.
5. A generic active-perception baseline selects viewpoints using generic perception utility.
6. A task-aware strategy additionally considers trajectory relevance and task alignment.
7. The two strategies are evaluated over matched trials using localisation error, predicted uncertainty, task alignment, movement cost, and selection behaviour.

The central idea is that **the most informative viewpoint globally is not necessarily the most useful viewpoint for the surgical task being performed**.

---

# System Architecture

```text
                    Surgical Workspace
                           |
                           v
                 Collision-Aware RRT
                           |
                           v
                  Planned Trajectory
                           |
             +-------------+-------------+
             |                           |
             v                           v
      Task-Relevance Model       Candidate Viewpoints
             |                           |
             |                           v
             |                  Camera / Observation Model
             |                           |
             |                           v
             +----------------> Perception Utility
                                         |
                              +----------+----------+
                              |                     |
                              v                     v
                    Generic Active           Task-Aware Active
                       Perception               Perception
                              |                     |
                              +----------+----------+
                                         |
                                         v
                               Selected Viewpoint
                                         |
                                         v
                              Localisation Estimate
                                         |
                                         v
                              Statistical Evaluation

A key design principle is the separation between:

planning geometry
perception estimates
task information
ground-truth evaluation

This prevents the evaluation from simply measuring whether the planner agrees with the same estimate that it used to make its decision.

Main Components
1. Surgical Robotics and Motion Planning

The framework includes:

RCM-constrained surgical instrument modelling;
joint-space trajectory generation;
collision-aware RRT planning;
anatomical obstacle modelling;
safety-margin evaluation;
trajectory shortcutting and optimisation;
ground-truth trajectory evaluation.

The RCM constraint represents minimally invasive access through a fixed entry point.

2. Perception and Camera Model

The perception subsystem includes:

geometric camera poses;
camera intrinsics;
visibility evaluation;
viewpoint generation;
observation-quality modelling;
occlusion handling;
localisation uncertainty;
generic viewpoint scoring;
task-aware viewpoint scoring.

Candidate viewpoints are sampled around the surgical target using spherical shells with configurable radius, azimuth, and elevation.

3. Task-Aware Active Perception

The task-aware strategy incorporates the intended surgical trajectory rather than treating all observations as equally useful.

A SurgicalTask contains:

the planned trajectory;
safety-critical points associated with that task.

Task relevance is modelled from the distance between a point and the planned trajectory.

This allows the system to distinguish between:

"How uncertain is the anatomy?"

and

"How uncertain is the anatomy where that uncertainty matters to the task?"

The task-aware controller then combines generic observation utility with task-specific information.

Experimental Evaluation

The final experimental package includes:

100-trial generic vs task-aware active-perception benchmark;
paired localisation-error comparison;
predicted-uncertainty comparison;
task-alignment measurement;
task-relevance measurement;
camera-movement measurement;
selection-difference analysis;
uncertainty sensitivity analysis;
statistical confidence intervals;
effect-size analysis;
automated figure generation.
Main 100-Trial Benchmark
Generic Active Perception
Metric	Result
Trials	100
Mean localisation error	4.211 mm
Median localisation error	4.017 mm
Mean predicted sigma	2.667 mm
Mean camera movement	85.962 mm
Mean task alignment	0.999040
Task-Aware Active Perception
Metric	Result
Trials	100
Mean localisation error	3.158 mm
Median localisation error	3.013 mm
Mean predicted sigma	2.000 mm
Mean camera movement	111.544 mm
Mean task alignment	0.998298
Mean task relevance	0.349321
Comparative Result
Metric	Change
Localisation-error reduction	25.00%
Predicted-sigma reduction	25.00%
Task-alignment change	−0.07%
Selection difference rate	100.00%

The task-aware strategy therefore selected a different viewpoint in every tested trial and produced a lower mean localisation error under the implemented simulation conditions.

The improvement came with increased camera movement, making the result a performance trade-off rather than a free improvement.

Statistical Analysis

For the 100-trial localisation-error experiment:

Statistic	Result
Generic mean	4.211 mm
Generic SD	1.758 mm
Generic 95% CI	3.877–4.548 mm
Task-aware mean	3.158 mm
Task-aware SD	1.318 mm
Task-aware 95% CI	2.908–3.411 mm
Mean paired improvement	1.053 mm
95% CI of paired difference	0.969–1.137 mm reduction
Cohen's d
z
	​

	−2.395
Predicted Uncertainty

Mean predicted localisation uncertainty changed from:

Generic:     2.667 mm
Task-aware:  2.000 mm

corresponding to a:

25.00% reduction

in the benchmark.

The predicted uncertainty values in this benchmark are model outputs rather than direct physical sensor measurements.

Uncertainty Sensitivity

The viewpoint-selection behaviour was evaluated across perception uncertainty levels of:

1 mm
2 mm
5 mm
10 mm
20 mm
30 mm

For every tested uncertainty level:

the generic strategy selected candidate 90;
the task-aware strategy selected candidate 18;
the selected task-aware uncertainty was lower;
the task-aware selection differed from the generic selection.
Sigma	Generic Selected	Task-Aware Selected
1 mm	1.0 mm	0.6 mm
2 mm	2.0 mm	1.2 mm
5 mm	5.0 mm	3.0 mm
10 mm	10.0 mm	6.0 mm
20 mm	20.0 mm	12.0 mm
30 mm	30.0 mm	18.0 mm

This provides a controlled sensitivity result showing that the selection mechanism remains behaviourally distinct across the tested uncertainty range.

Safety-Critical Motion Planning Foundation

Before active perception was added, the framework established an uncertainty-aware motion-planning foundation.

The deterministic and uncertainty-aware planners maintain separate:

perceived anatomical geometry;
ground-truth anatomical geometry.

For uncertainty-aware planning, the protected region can be inflated according to positional uncertainty:

m
plan
	​

=m
base
	​

+kσ

where:

m
base
	​

 is the nominal safety margin;
σ is localisation uncertainty;
k controls the degree of uncertainty protection.

This enables trajectories planned from imperfect anatomical estimates to be evaluated against hidden ground truth.

The earlier safety experiments demonstrated the expected engineering trade-off:

greater uncertainty protection can improve ground-truth safety while increasing planning computation and trajectory cost.

ROS 2 Integration

The project also contains a ROS 2 Jazzy workspace:

ros2_jazzy/
└── ros2_ws/
    └── src/
        └── surgical_navigation_ros/

The package contains nodes for:

perception;
planning;
planner/safety bridging;
safety gating;
viewpoint reception;
visualisation.

The ROS 2 integration is kept separate from the main research Python package so that the simulation and research code remain independently testable.

Visualisation

A standalone visual simulation is provided through:

visual_surgical_simulation.py

The visualisation connects the implemented planning and perception components to a 3D representation of the surgical workspace.

Generated research figures are stored in:

results/active_perception_figures/

Current figures include:

localisation_error_comparison.png
predicted_uncertainty_comparison.png
uncertainty_sensitivity.png
performance_tradeoff.png
Automated Verification

The main Python research test suite currently passes:

373 passed

Run:

python -m pytest -q tests --ignore=tests/test_statistical_benchmark.py

The ROS 2 tests are maintained within the separate Jazzy workspace and require the ROS 2 environment rather than the ordinary Python .venv.

Repository Structure
uncertainty-aware-surgical-navigation/
│
├── docs/
│   ├── coordinate_frames.md
│   ├── project_scope.md
│   ├── requirements.md
│   └── verification_plan.md
│
├── results/
│   ├── active_perception_figures/
│   │   ├── localisation_error_comparison.png
│   │   ├── performance_tradeoff.png
│   │   ├── predicted_uncertainty_comparison.png
│   │   └── uncertainty_sensitivity.png
│   ├── day7_statistical_trials.csv
│   └── figures/
│
├── src/
│   ├── geometry/
│   ├── perception/
│   ├── robotics/
│   └── simulation/
│
├── tests/
│
├── ros2_jazzy/
│   └── ros2_ws/
│       └── src/
│           └── surgical_navigation_ros/
│
├── visual_surgical_simulation.py
├── README.md
├── .gitignore
└── sync_github.ps1
Running the Core Tests

Activate the project environment and run:

python -m pytest -q tests --ignore=tests/test_statistical_benchmark.py

Expected current result:

373 passed
Running the Active-Perception Benchmark

From the repository root:

python -m src.simulation.task_aware_benchmark

This runs the 100-trial generic vs task-aware comparison and prints the comparative and statistical results.

Running Sensitivity Analysis
python -m src.simulation.uncertainty_sensitivity

This evaluates viewpoint-selection behaviour over the configured perception-uncertainty range.

Generating Figures
python -m src.simulation.active_perception_figures

The generated figures are written to:

results/active_perception_figures/
Running the Standalone Visual Simulation
python visual_surgical_simulation.py

This launches the standalone visual simulation of the surgical navigation pipeline.

Reproducibility

The repository retains:

experiment scripts;
statistical-analysis utilities;
test suites;
generated figures;
benchmark data;
documented configuration;
ROS 2 integration;
Git history.

The main benchmark uses deterministic experimental components where appropriate and records the comparison metrics needed to reproduce the reported analysis.

Limitations

This is a simulation-based research framework and does not represent a clinically validated surgical-navigation system.

Current limitations include:

simplified anatomical geometry;
simulated rather than learned visual perception;
simplified camera and observation models;
Gaussian localisation-uncertainty assumptions;
simplified surgical instrument dynamics;
no tissue deformation;
no force/contact modelling;
no patient data;
no physical robotic-platform validation;
no clinical validation.

The reported performance should therefore be interpreted as simulation evidence supporting the proposed mechanism, not as evidence of clinical effectiveness.

Safety and Intended Use

This repository contains a simulation-based engineering research prototype.

It is not a medical device and has not undergone clinical validation, regulatory approval, medical-device certification, or clinical safety testing.

It must not be used for:

patient monitoring;
diagnosis;
treatment;
surgical guidance;
clinical decision-making;
or any other clinical purpose.
Research Status

Implementation: Complete
Active-perception benchmark: Complete
Statistical analysis: Complete
Sensitivity analysis: Complete
Visualisation: Complete
Core automated tests: 373 passed
ROS 2 integration: Implemented
Research paper: In preparation

Future Research

The next research stage is to extend the simulation toward increasingly realistic perception and surgical environments, including:

richer camera/observation models;
non-Gaussian and anisotropic uncertainty;
realistic occlusion and visual degradation;
dynamic anatomical structures;
uncertainty-aware closed-loop planning;
larger-scale statistical evaluation;
physical robotic validation;
eventually, clinically relevant validation pathways.

These extensions are deliberately separated from the current validated benchmark so that the reported results remain reproducible and interpretable.
