# Uncertainty-Aware Active Perception for Safety-Critical Motion Planning in Minimally Invasive Surgical Robotics

A simulation-based research engineering framework investigating how **perception uncertainty, task-aware active perception, and safety-critical motion planning** interact in minimally invasive surgical robotics.

The framework combines surgical robot modelling, collision-aware motion planning, uncertainty representation, camera/viewpoint simulation, active perception, task-aware viewpoint selection, controlled benchmarking, statistical evaluation, automated verification, and ROS 2 integration.

> **Research prototype:** This repository is intended for simulation and engineering research only. It is not a clinical system or medical device.

---

## Research Question

**Can task-aware uncertainty-driven active perception improve the safety and efficiency of motion planning in simulated minimally invasive surgical environments compared with fixed-view and task-agnostic active perception?**

The project investigates this question through two linked research layers:

1. **Uncertainty-aware navigation:** how localisation uncertainty affects trajectory safety and planning efficiency.
2. **Task-aware active perception:** whether selecting viewpoints using information about the intended surgical trajectory improves task-relevant perception compared with generic viewpoint selection.

The final objective is to evaluate:

```text
Fixed-view perception
        vs
Generic active perception
        vs
Task-aware active perception
```

through a common perception → uncertainty → planning → ground-truth safety pipeline.

---

## Current Research Status

| Component                                                   | Status                 |
| ----------------------------------------------------------- | ---------------------- |
| RCM-constrained surgical instrument model                   | Complete               |
| Safety-critical workspace modelling                         | Complete               |
| Collision-aware RRT planning                                | Complete               |
| Trajectory shortcutting and optimisation                    | Complete               |
| Gaussian localisation uncertainty                           | Complete               |
| Ground-truth vs perceived anatomy separation                | Complete               |
| Uncertainty-aware safety margins                            | Complete               |
| Monte Carlo uncertainty evaluation                          | Complete               |
| Camera and viewpoint modelling                              | Complete               |
| Viewpoint-dependent observation model                       | Complete               |
| Occlusion modelling                                         | Complete               |
| Generic active perception                                   | Complete               |
| Task-relevance modelling                                    | Complete               |
| Task-aware viewpoint scoring                                | Complete               |
| Task-aware viewpoint selection                              | Complete               |
| Generic closed-loop perception evaluation                   | Implemented            |
| Ablation experiments                                        | Implemented            |
| Sensitivity analysis                                        | Implemented            |
| Statistical evaluation utilities                            | Implemented            |
| ROS 2 integration                                           | Implemented separately |
| Core automated verification                                 | **377 tests passing**  |
| Fresh-environment reproduction                              | **Verified**           |
| GitHub Actions CI                                           | Configured             |
| Unified fixed vs generic vs task-aware navigation benchmark | **In progress**        |
| Full perception → planning → safety closed loop             | **In progress**        |

This distinction is deliberate: individual planning and active-perception hypotheses have been evaluated, while the full three-strategy end-to-end navigation comparison remains the main outstanding technical objective.

---

## Core Contributions

### 1. Safety-Critical Surgical Motion Planning

The project implements a simulated minimally invasive surgical instrument constrained by a **remote centre of motion (RCM)**.

The planning framework includes:

* joint-space configuration modelling;
* RCM-constrained instrument kinematics;
* collision checking;
* anatomical safety margins;
* collision-aware rapidly exploring random tree (RRT) planning;
* edge validation;
* joint-limit enforcement;
* trajectory shortcutting;
* path-cost evaluation;
* hidden ground-truth safety evaluation.

The RCM error in the implemented geometric model remains approximately at floating-point numerical precision.

---

### 2. Explicit Perception Uncertainty

The framework deliberately separates:

```text
Ground-truth anatomy
        from
Perceived anatomy
```

The planner therefore does not receive perfect knowledge of the simulated surgical environment.

Anatomical localisation uncertainty is represented using a Gaussian positional model:

[
\Sigma = \sigma^2 I
]

for isotropic uncertainty, where:

* (\Sigma) is the positional covariance matrix;
* (\sigma) is localisation standard deviation;
* (I) is the identity matrix.

A noisy anatomical estimate is supplied to the planner while hidden ground-truth geometry is retained independently for evaluation.

This allows a trajectory that appears safe according to perception to be tested against the actual simulated anatomy.

---

### 3. Uncertainty-Aware Motion Planning

The deterministic planner uses the nominal safety margin:

[
m_{\text{plan}} = m_{\text{base}}
]

The uncertainty-aware planner can instead inflate the protected region:

[
m_{\text{plan}} = m_{\text{base}} + k\sigma
]

where:

* (m_{\text{base}}) is the nominal anatomical safety margin;
* (\sigma) is estimated localisation uncertainty;
* (k) controls planning conservatism.

This explicitly exposes the engineering trade-off between:

```text
robustness
    vs
planning feasibility
    vs
computation
    vs
trajectory efficiency
```

---

## Uncertainty-Aware Planning Benchmark

A paired 30-trial Monte Carlo experiment compared deterministic and uncertainty-aware planning under matched noisy anatomical observations.

Experimental configuration:

```text
Localisation standard deviation: 5 mm
Uncertainty multiplier:          k = 2
Trials:                          30
```

| Metric                             | Deterministic RRT | Uncertainty-Aware RRT |
| ---------------------------------- | ----------------: | --------------------: |
| Planning success                   |              100% |                  100% |
| Ground-truth collision rate        |                0% |                    0% |
| Ground-truth safety-violation rate |         **56.7%** |              **3.3%** |
| Mean true safety clearance         |     **−2.106 mm** |         **+8.555 mm** |
| Mean planning time                 |           0.818 s |               1.609 s |
| Mean iterations                    |             230.1 |                 475.5 |
| Mean path cost                     |            2.1614 |                2.6381 |

Within this simulation configuration, uncertainty-aware planning substantially reduced observed safety-margin violations.

The improvement was accompanied by greater computation and trajectory cost, demonstrating a **safety–efficiency trade-off rather than a cost-free improvement**.

These results are simulation-specific and should not be interpreted as evidence of clinical performance.

---

## Active Perception

The perception subsystem models:

* camera pose;
* camera intrinsics;
* candidate viewpoint generation;
* visibility;
* geometric observation quality;
* occlusion;
* localisation uncertainty;
* camera movement cost;
* generic viewpoint utility.

Candidate viewpoints are generated around the surgical workspace and evaluated according to expected observation quality.

Generic active perception selects a viewpoint using perception-related utility without information about the intended surgical trajectory.

---

## Task-Aware Active Perception

Generic active perception asks:

> Which viewpoint provides the best observation?

Task-aware active perception additionally asks:

> Which viewpoint provides useful information **where uncertainty matters to the intended surgical task?**

A surgical task contains:

* the planned trajectory;
* task-relevant safety-critical locations.

The planned trajectory is used to create a **task-relevance model**.

The task-aware scoring framework then combines observation information with trajectory relevance and task alignment before selecting a viewpoint.

This allows the framework to distinguish between:

```text
global perception quality
        and
task-relevant perception quality
```

---

## Task-Aware Active-Perception Benchmark

A matched 100-trial benchmark compared generic and task-aware active perception.

### Generic Active Perception

| Metric                    |    Result |
| ------------------------- | --------: |
| Trials                    |       100 |
| Mean localisation error   |  4.211 mm |
| Median localisation error |  4.017 mm |
| Mean predicted sigma      |  2.667 mm |
| Mean camera movement      | 85.962 mm |
| Mean task alignment       |  0.999040 |

### Task-Aware Active Perception

| Metric                    |     Result |
| ------------------------- | ---------: |
| Trials                    |        100 |
| Mean localisation error   |   3.158 mm |
| Median localisation error |   3.013 mm |
| Mean predicted sigma      |   2.000 mm |
| Mean camera movement      | 111.544 mm |
| Mean task alignment       |   0.998298 |
| Mean task relevance       |   0.349321 |

### Comparative Result

| Metric                            |      Change |
| --------------------------------- | ----------: |
| Mean localisation-error reduction |  **25.00%** |
| Mean predicted-sigma reduction    |  **25.00%** |
| Task-alignment change             |      −0.07% |
| Selection difference rate         | **100.00%** |

The task-aware strategy selected a different viewpoint in every tested trial and produced lower simulated localisation error under the implemented benchmark conditions.

The improvement required increased camera movement, again exposing a trade-off rather than an unconditional performance improvement.

---

## Statistical Evaluation

For the 100-trial localisation-error comparison:

| Statistic                   |                   Result |
| --------------------------- | -----------------------: |
| Generic mean                |                 4.211 mm |
| Generic SD                  |                 1.758 mm |
| Generic 95% CI              |           3.877–4.548 mm |
| Task-aware mean             |                 3.158 mm |
| Task-aware SD               |                 1.318 mm |
| Task-aware 95% CI           |           2.908–3.411 mm |
| Mean paired improvement     |                 1.053 mm |
| 95% CI of paired difference | 0.969–1.137 mm reduction |

The repository also includes statistical benchmarking and validation utilities for controlled experimental analysis.

Reported uncertainty values are outputs of the implemented simulation model rather than measurements from a physical imaging system.

---

## Sensitivity and Ablation Analysis

The project includes dedicated experiments for:

* task-aware ablation;
* normalised ablation;
* task-weight sensitivity;
* perception-uncertainty sensitivity;
* uncertainty-heterogeneity sensitivity;
* uncertainty parameter sweeps.

These experiments are used to determine whether observed behaviour depends on:

* a particular task weight;
* a single uncertainty magnitude;
* one scoring term;
* one uncertainty composition.

This is intended to distinguish genuine algorithmic behaviour from results caused by a single tuned configuration.

---

## System Architecture

```text
                         Surgical Task
                              |
                              v
                     Surgical Workspace
                              |
                              v
                  RCM-Constrained Instrument
                              |
                              v
                    Collision-Aware RRT
                              |
                              v
                      Planned Trajectory
                              |
                  +-----------+-----------+
                  |                       |
                  v                       v
          Task-Relevance Model     Candidate Viewpoints
                  |                       |
                  |                       v
                  |              Camera / Observation Model
                  |                       |
                  |              Visibility / Occlusion
                  |                       |
                  |                       v
                  +-------------> Uncertainty Estimate
                                          |
                               +----------+----------+
                               |                     |
                               v                     v
                         Generic Active        Task-Aware Active
                           Perception              Perception
                               |                     |
                               +----------+----------+
                                          |
                                          v
                                  Selected Viewpoint
                                          |
                                          v
                                Updated Observation
                                          |
                                          v
                                Localisation Estimate
                                          |
                                          v
                             Ground-Truth Evaluation
```

The principal design separation is between:

* planning geometry;
* perception estimates;
* task information;
* ground truth.

This prevents evaluation from simply testing a system against the same imperfect information it used for planning.

The remaining end-to-end work extends the selected observation back into the uncertainty-aware motion planner so that all three perception strategies can be compared using common navigation safety and efficiency outcomes.

---

## Software Verification

The core Python research framework currently has:

```text
377 passed
0 failed
```

The regression suite covers areas including:

* coordinate transformations;
* surgical instrument kinematics;
* RCM constraints;
* trajectory generation;
* workspace geometry;
* collision detection;
* safety evaluation;
* motion planning;
* path optimisation;
* uncertainty modelling;
* noisy perception;
* camera geometry;
* viewpoints;
* observation models;
* occlusion;
* generic active perception;
* task relevance;
* task-aware scoring;
* task-aware active perception;
* closed-loop behaviour;
* ablation experiments;
* sensitivity analysis;
* statistical benchmarking;
* statistical validation.

A completely fresh Conda environment created from `environment.yml` reproduced the full **377-test passing baseline**.

---

## Continuous Integration

The repository contains a GitHub Actions workflow that:

1. checks out the repository;
2. creates the documented Conda environment;
3. uses Python 3.11;
4. installs the project dependencies;
5. runs the complete core research test suite.

The workflow runs on pushes and pull requests targeting `main`.

ROS 2 testing remains separate because the ROS middleware dependencies require a ROS 2 environment rather than the standard Python research environment.

---

## Reproducible Environment

The core research environment is defined in:

```text
environment.yml
```

Current principal dependencies include:

* Python 3.11;
* NumPy;
* PyBullet;
* Matplotlib;
* pandas;
* pytest.

Create the environment with:

```bash
conda env create -f environment.yml
conda activate surgical-navigation
```

Run the complete core test suite with:

```bash
python -m pytest -q
```

Expected current regression result:

```text
377 passed
```

Exact execution time is machine-dependent.

---

## Running Key Experiments

### Task-Aware Active-Perception Benchmark

```bash
python -m src.simulation.task_aware_benchmark
```

### Uncertainty Sensitivity Analysis

```bash
python -m src.simulation.uncertainty_sensitivity
```

### Generate Active-Perception Figures

```bash
python -m src.simulation.active_perception_figures
```

Generated figures are stored under:

```text
results/active_perception_figures/
```

### Standalone Surgical Simulation

```bash
python visual_surgical_simulation.py
```

---

## ROS 2 Integration

A separate ROS 2 Jazzy workspace is included under:

```text
ros2_jazzy/
└── ros2_ws/
```

The ROS layer provides experimental integration for components including:

* perception;
* planning;
* planner/safety bridging;
* safety gating;
* viewpoint communication;
* visualisation.

ROS 2 is intentionally separated from the core research environment so that the simulation framework remains independently reproducible and testable.

ROS-specific tests require a correctly configured ROS 2 Jazzy environment.

---

## Repository Structure

```text
uncertainty-aware-surgical-navigation/
│
├── .github/
│   └── workflows/
│       └── core-tests.yml
│
├── docs/
│   ├── coordinate_frames.md
│   ├── project_scope.md
│   ├── requirements.md
│   └── verification_plan.md
│
├── results/
│   ├── active_perception_figures/
│   ├── figures/
│   └── experimental outputs
│
├── ros2_jazzy/
│   └── ros2_ws/
│
├── src/
│   ├── geometry/
│   ├── perception/
│   ├── robotics/
│   └── simulation/
│
├── tests/
│
├── environment.yml
├── pytest.ini
├── visual_surgical_simulation.py
└── README.md
```

---

## Engineering and Research Practices

The repository is structured to demonstrate:

* modular software architecture;
* version-controlled research development;
* explicit requirements;
* verification planning;
* automated regression testing;
* reproducible environments;
* continuous integration;
* controlled stochastic experiments;
* matched experimental trials;
* Monte Carlo evaluation;
* baseline comparison;
* ablation analysis;
* sensitivity analysis;
* statistical confidence intervals;
* effect-size analysis;
* explicit modelling assumptions;
* separation of estimated and ground-truth state;
* safety-oriented evaluation.

---

## Remaining Technical Objectives

The principal remaining work is intentionally focused rather than feature-driven.

### 1. Unified Three-Strategy Experiment

Run:

```text
Fixed view
    vs
Generic active perception
    vs
Task-aware active perception
```

under matched simulated conditions.

### 2. Full Perception–Planning Feedback

Propagate viewpoint-dependent perception uncertainty into the uncertainty-aware motion planner.

### 3. Common Navigation Outcomes

Evaluate all three strategies using:

* planning success;
* ground-truth collision rate;
* ground-truth safety violations;
* true minimum clearance;
* path cost;
* computation time;
* planner iterations;
* perception movement cost.

### 4. Matched Repeated Trials

Use controlled seeds and matched scenarios so that differences between strategies can be attributed to the strategy rather than different random conditions.

### 5. Final End-to-End Statistical Validation

Quantify uncertainty in the measured differences using appropriate confidence intervals, effect sizes, and paired comparisons.

No unrelated functionality will be added solely to increase project size; remaining development is directed toward answering the central research question.

---

## Limitations

This framework intentionally simplifies several aspects of real surgical robotics.

Current limitations include:

* simplified anatomical geometry;
* simulated rather than learned visual perception;
* simplified camera and observation models;
* Gaussian localisation-uncertainty assumptions;
* simplified surgical instrument dynamics;
* no deformable tissue model;
* no force or contact interaction modelling;
* no patient data;
* no physical robotic-platform validation;
* no clinical validation.

The reported results therefore provide **simulation evidence about the implemented algorithms and mechanisms**, not evidence of clinical effectiveness.

---

## Safety and Intended Use

This repository contains a simulation-based engineering research prototype.

It is **not a medical device** and has not undergone:

* clinical validation;
* regulatory approval;
* medical-device certification;
* clinical safety testing.

It must not be used for:

* patient monitoring;
* diagnosis;
* treatment;
* surgical guidance;
* clinical decision-making;
* any other clinical purpose.

