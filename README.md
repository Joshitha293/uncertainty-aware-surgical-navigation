Task-Aware Active Perception Coupled with Uncertainty-Aware Motion Planning for Simulated Surgical Navigation

A simulation-based research engineering framework investigating how task-aware active perception, localisation uncertainty, and safety-critical motion planning interact in minimally invasive surgical robotics.

The project compares Fixed View, Generic Active Perception, and Task-Aware Active Perception through a common perception-to-planning pipeline with hidden ground-truth safety evaluation.

Research prototype: This repository is intended for simulation and engineering research only. It is not a clinical system or medical device.

Research Question

Can task-aware active perception improve the probability of obtaining a safe executable motion plan in simulated minimally invasive surgical environments when perception uncertainty is explicitly propagated into safety-critical motion planning?

Fixed View
        vs
Generic Active Perception
        vs
Task-Aware Active Perception

Common pipeline:

Perception
    ->
Localisation estimate + predicted uncertainty
    ->
Uncertainty-inflated planning geometry
    ->
Collision-aware RRT
    ->
Hidden ground-truth safety evaluation

What This Project Implements

RCM-constrained minimally invasive surgical instrument modelling

joint-space forward kinematics

surgical workspace and anatomical geometry

collision and safety-margin evaluation

collision-aware RRT planning

edge validation and trajectory shortcutting

path-cost evaluation

explicit Gaussian localisation uncertainty

separation of perceived and hidden ground-truth anatomy

uncertainty-aware planning margins

camera pose, visibility, viewpoints, and occlusion

Generic Active Perception

task representation and task relevance

Task-Aware Active Perception

matched stochastic benchmarking

multi-scenario robustness evaluation

mechanism ablation

uncertainty stress testing

formal uncertainty calibration

synthetic visual-quality degradation

planning-time, iteration, and path-cost evaluation

automated verification

ROS 2 Jazzy integration

Core Idea

The planner does not receive perfect simulator geometry.

Hidden ground truth
        |
        +----> evaluation only

Noisy perceived anatomy
        |
        +----> motion planning

For the principal isotropic localisation model:

Sigma = sigma^2 I

Planning geometry can then be inflated according to:

m_plan = m_base + k sigma

This connects perception uncertainty directly to planning conservatism and feasibility.

Main Results

Multi-Scenario Robustness

The principal experiment evaluated:

10 simulated scene variations
x 10 matched repetitions
x 3 strategies
= 300 strategy evaluations

Metric

Fixed

Generic

Task-Aware

Mean localisation error

25.964 mm

16.934 mm

4.682 mm

Mean predicted sigma

17.777 mm

12.267 mm

3.003 mm

Camera movement

0 mm

15.628 mm

109.985 mm

Planning success

45%

63%

100%

Safe-navigation success

43%

61%

97%

Collision rate

0%

0%

0%

Safety-violation rate

2%

2%

3%

Worst-scenario safe-navigation success

0%

0%

90%

Across the tested simulated perturbations, Task-Aware Active Perception substantially improved planning feasibility and the probability of obtaining a safe executable trajectory.

The project does not claim that Task-Aware Active Perception reduces collision probability conditional on a valid trajectory already existing.

Mechanism Ablation

Four variants were evaluated:

Generic baseline
Alignment-only
Uncertainty-only
Full task-aware

The main finding was:

Task alignment was the dominant viewpoint-selection mechanism.

The uncertainty-only variant behaved essentially identically to the Generic baseline under the tested observation model.

Further uncertainty-stress experiments increased the uncertainty weight from:

0 -> 0.25 -> 1 -> 4 -> 16

without changing the selected viewpoint.

The project is therefore best described as:

Task-aware active perception coupled with uncertainty-aware motion planning

rather than as an uncertainty-driven viewpoint-selection algorithm.

Formal Uncertainty Calibration

The localisation-uncertainty model was evaluated using:

10 scenarios
60 viewpoint conditions
6,000 observations

Diagnostic

Expected

Observed

Mean normalised squared error

3.000

3.024

Mean radial error / sigma

1.596

1.603

50% coverage

50%

49.42%

90% coverage

90%

89.95%

95% coverage

95%

94.98%

99% coverage

99%

98.88%

The uncertainty model was therefore classified as:

well_calibrated_under_simulation

This applies only to the implemented simulation model.

Synthetic Visual-Quality Degradation

A supplementary stress test represented reduced visual quality using:

sigma_degraded = sigma_nominal / sqrt(quality)

Under the most severe tested degradation:

Metric

Fixed

Generic

Task-Aware

Localisation error

60.222 mm

42.250 mm

9.812 mm

Predicted sigma

35.555 mm

24.533 mm

6.007 mm

Planning success

35%

55%

95%

Safe-navigation success

35%

55%

90%

This is a synthetic observation-quality stress test, not a physical photometric model.

Planning Efficiency

A final benchmark used:

10 scenarios
x 5 matched repetitions
= 50 matched planning units

Metric

Fixed

Generic

Task-Aware

Planning success

46%

66%

100%

Safe-navigation success

38%

60%

92%

Camera movement

0 mm

15.628 mm

109.985 mm

Planning time, all attempts

0.547 s

0.882 s

1.239 s

RRT iterations, all attempts

142.8

227.2

333.7

Planning time, successful plans

1.189 s

1.337 s

1.239 s

Path cost, successful plans

2.517

2.558

2.499

Task-aware perception requires greater camera repositioning and more computation when averaged across all attempts.

However, Fixed and Generic strategies frequently fail to produce a path. Conditional on successful planning, computational effort is comparable.

On the 33 matched trials where both Generic and Task-Aware succeeded:

Task-Aware minus Generic path cost:
-0.107

95% CI:
[-0.257, -0.015]

Scientific Conclusions

The experiments support four principal conclusions:

Task-aware perception improves end-to-end navigation feasibility.

Task alignment is the dominant viewpoint-selection mechanism.

Explicit uncertainty weighting adds no measurable independent viewpoint-ranking benefit under the tested observation model.

Localisation uncertainty remains important downstream because it modifies safety-critical planning geometry.

Verification

Final local automated regression baseline:

449 passed
0 failed

Run:

python -m pytest -q

The suite covers geometry, kinematics, safety, planning, uncertainty, camera modelling, active perception, task awareness, end-to-end integration, statistical validation, robustness, ablation, stress testing, calibration, synthetic visual degradation, and efficiency benchmarking.

Reproducible Environment

The Python research environment is defined in:

environment.yml

The project targets Python 3.11.

conda env create -f environment.yml
conda activate surgical-navigation
python -m pytest -q

Key Experiments

python -m src.simulation.three_strategy_trial
python -m src.simulation.three_strategy_statistical_benchmark
python -m src.simulation.three_strategy_robustness_benchmark
python -m src.simulation.final_evidence_package
python -m src.simulation.uncertainty_calibration_benchmark
python -m src.simulation.illumination_degradation_benchmark
python -m src.simulation.three_strategy_efficiency_benchmark

Standalone simulation:

python visual_surgical_simulation.py

Repository Structure

uncertainty-aware-surgical-navigation/
|
|-- .github/
|   `-- workflows/
|
|-- docs/
|   |-- architecture.md
|   |-- coordinate_frames.md
|   |-- experimental_protocol.md
|   |-- project_scope.md
|   |-- requirements.md
|   |-- traceability_matrix.md
|   `-- verification_plan.md
|
|-- results/
|-- ros2_jazzy/
|-- src/
|   |-- geometry/
|   |-- perception/
|   |-- robotics/
|   `-- simulation/
|
|-- tests/
|-- environment.yml
|-- pytest.ini
|-- visual_surgical_simulation.py
`-- README.md

Result Artifacts

Important final result directories include:

results/final_evidence/
results/supplementary_uncertainty_calibration/
results/supplementary_illumination/
results/supplementary_efficiency/

Machine-readable CSV and JSON outputs are retained for reproducibility and analysis.

ROS 2 Integration

A separate ROS 2 Jazzy workspace is included under:

ros2_jazzy/
`-- ros2_ws/

It provides experimental integration for perception communication, planning, safety bridging, viewpoint communication, and visualisation.

ROS 2 is intentionally separated from the core Python simulation environment.

No physical surgical robot implementation is claimed.

Limitations

Important limitations include:

simulation-only validation

simplified anatomical geometry

engineered scene variations rather than independent patient anatomies

simulated rather than learned visual perception

simplified camera and observation models

isotropic Gaussian localisation uncertainty

synthetic rather than physical illumination degradation

no deformable tissue model

no force or tactile sensing

no dynamic anatomy

no calibration-drift model

no realistic specular-reflection model

no patient data

no animal or cadaver experiments

no physical robotic-platform validation

no clinical validation

Task-Aware Active Perception also requires substantially greater camera repositioning.

Results therefore demonstrate robustness to the tested simulated perturbations, not clinical generalisation.

Safety and Intended Use

This repository contains a simulation-based engineering research prototype.

It is not a medical device and must not be used for:

diagnosis

treatment

patient monitoring

surgical guidance

clinical decision-making

autonomous clinical intervention

any other clinical purpose

Final Research Position

The project provides simulation evidence that Task-Aware Active Perception can improve end-to-end surgical-navigation feasibility when coupled with uncertainty-aware motion planning.

The strongest demonstrated active-perception mechanism is task alignment, while localisation uncertainty remains important through its downstream effect on 