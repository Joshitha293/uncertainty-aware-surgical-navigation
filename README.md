# Uncertainty-Aware Active Perception for Safety-Critical Motion Planning in Minimally Invasive Surgical Robotics

A simulation-based research engineering project investigating how perception and localisation uncertainty affect safety-critical motion planning in minimally invasive surgical robotics.

The project develops a computational framework for studying whether uncertainty-aware planning and, ultimately, task-aware active perception can improve navigation safety under imperfect visual information.

---

## Research Question

Can task-aware uncertainty-driven perception improve the safety and efficiency of motion planning in simulated minimally invasive surgical environments compared with fixed-view and task-agnostic active perception?

The current development stage establishes the uncertainty-aware motion-planning foundation required to investigate this question.

---

## Research Motivation

Surgical navigation systems do not operate with perfect knowledge of anatomy.

Errors in localisation or perception may cause the robot's internal representation of a safety-critical structure to differ from its true position. A trajectory that appears safe using this estimated geometry may therefore pass too close to the true anatomy.

This project investigates how explicitly representing perception uncertainty can be incorporated into motion planning and, in later stages, how active perception may reduce uncertainty that is particularly relevant to the planned surgical task.

---

## Experimental Strategies

The complete project is designed to compare three perception strategies:

1. **Fixed-view perception** — planning using observations from a fixed camera viewpoint without active viewpoint adjustment.

2. **Generic uncertainty-aware active perception** — additional viewpoints are selected according to global perception uncertainty.

3. **Task-aware uncertainty-driven active perception** — uncertainty is weighted according to its relevance to the planned trajectory and nearby safety-critical structures before viewpoint selection.

Active viewpoint selection is a future stage of the project.

The current implementation establishes the surgical navigation, safety, uncertainty and planning framework required for these later experiments.

---

# Current Implementation

The project currently implements:

- RCM-constrained surgical instrument kinematics;
- simulated safety-critical anatomical structures;
- geometric collision detection;
- anatomical safety-margin evaluation;
- collision-aware RRT motion planning;
- trajectory shortcutting and optimisation;
- Gaussian anatomical localisation uncertainty;
- covariance-based uncertainty representation;
- simulated noisy anatomical perception;
- deterministic planning from imperfect anatomical estimates;
- uncertainty-dependent safety-margin inflation;
- uncertainty-aware motion planning;
- hidden ground-truth trajectory evaluation;
- paired deterministic versus uncertainty-aware experiments;
- Monte Carlo benchmarking;
- uncertainty parameter sweeps;
- automated regression testing.

---

## System Architecture

The current experimental pipeline is:

```text
             Ground-truth anatomy
                     |
                     v
        Simulated uncertain perception
                     |
                     v
      Noisy anatomical estimate + covariance
                     |
          +----------+----------+
          |                     |
          v                     v
 Deterministic RRT      Uncertainty-aware RRT
   base margin             base margin + kσ
          |                     |
          +----------+----------+
                     |
                     v
            Collision-aware RRT
                     |
                     v
              Path shortcutting
                     |
                     v
        Ground-truth safety evaluation
                     |
                     v
       Safety / clearance / efficiency
                  metrics
```

A key experimental principle is the separation between **perceived anatomy** and **ground-truth anatomy**.

The planner receives only the simulated noisy anatomical estimate.

Ground-truth geometry is retained separately and used to evaluate whether the resulting trajectory was actually safe.

---

# RCM-Constrained Surgical Instrument

The simulated surgical instrument follows a remote-centre-of-motion (RCM) constraint representative of minimally invasive surgical access through a fixed entry point.

The configuration includes rotational and insertion degrees of freedom while constraining the instrument shaft to pass through the RCM.

The framework continuously evaluates RCM error during trajectory execution.

Across the current experiments, RCM errors remain on the order of approximately:

```text
10^-17 m
```

which corresponds to numerical floating-point precision in the implemented model.

---

# Deterministic Motion Planning

The deterministic baseline uses a collision-aware rapidly-exploring random tree (RRT) planner operating in instrument joint space.

Candidate configurations and edges are checked against:

- instrument joint limits;
- anatomical geometry;
- physical collision constraints;
- required anatomical safety margins.

After a valid path is generated, shortcut optimisation removes unnecessary intermediate waypoints while preserving geometric validity.

---

## Deterministic Baseline Benchmark

Multi-seed evaluation demonstrated reproducible collision-free planning.

Path optimisation reduced the mean number of waypoints from:

```text
25.9 -> 4.0
```

corresponding to an average reduction of approximately:

```text
84.0%
```

Mean path cost was reduced from approximately:

```text
2.543 -> 2.142
```

or approximately:

```text
15.5%
```

No physical collisions or nominal safety-margin violations were observed in this deterministic benchmark when planning and evaluation used the same known geometry.

However, some optimised trajectories approached the imposed safety boundary closely, motivating investigation of robustness to localisation error.

---

# Modelling Anatomical Localisation Uncertainty

Anatomical localisation uncertainty is represented using a three-dimensional Gaussian positional model.

For isotropic uncertainty:

\[
\Sigma = \sigma^2 I
\]

where:

- \(\Sigma\) is the positional covariance matrix;
- \(\sigma\) is the localisation standard deviation;
- \(I\) is the identity matrix.

A perceived anatomical centre is sampled from this uncertainty distribution.

The simulation therefore maintains two separate anatomical representations:

1. **Ground-truth anatomy** — the actual simulated structure position.
2. **Perceived anatomy** — the noisy estimate available to the planner.

This allows trajectories planned from imperfect information to be evaluated independently against hidden ground truth.

---

# Uncertainty-Aware Safety Margins

The deterministic planner uses the nominal anatomical safety margin:

\[
m_{\text{plan}} = m_{\text{base}}
\]

The uncertainty-aware planner instead uses:

\[
m_{\text{plan}}
=
m_{\text{base}} + k\sigma
\]

where:

- \(m_{\text{base}}\) is the nominal anatomical safety margin;
- \(\sigma\) represents positional uncertainty;
- \(k\) is an uncertainty multiplier controlling planning conservatism.

Increasing uncertainty therefore increases the protected region surrounding perceived safety-critical anatomy.

---

# Matched Uncertainty Experiment

A controlled experiment was performed in which both planners received the **same noisy anatomical perception**.

The deterministic planner used the nominal safety margin.

The uncertainty-aware planner received the same anatomical estimate but incorporated uncertainty-dependent margin inflation.

Both resulting trajectories were then evaluated against the hidden ground-truth anatomy.

A representative trial produced:

| Metric | Deterministic RRT | Uncertainty-Aware RRT |
|---|---:|---:|
| Planning success | Yes | Yes |
| True physical clearance | 9.980 mm | 22.404 mm |
| True safety clearance | -5.020 mm | +7.404 mm |
| Ground-truth collision | No | No |
| Ground-truth safety violation | Yes | No |
| Planning iterations | 293 | 617 |
| Smoothed waypoints | 3 | 4 |
| Path cost | 2.726 | 2.170 |

In this trial, the deterministic trajectory appeared valid according to perceived anatomy but violated the desired safety region when evaluated against ground truth.

The uncertainty-aware trajectory maintained positive ground-truth safety clearance.

This single experiment demonstrates the mechanism of interest but is not, by itself, evidence of general performance.

---

# Monte Carlo Evaluation

A paired **30-trial Monte Carlo experiment** was subsequently performed.

Experimental configuration:

```text
Localisation standard deviation: 5 mm
Uncertainty multiplier:          k = 2
Trials:                          30
```

Both methods received matched perception realisations to support direct comparison.

## Results

| Metric | Deterministic RRT | Uncertainty-Aware RRT |
|---|---:|---:|
| Planning success | 100% | 100% |
| Ground-truth collision rate | 0% | 0% |
| Ground-truth safety-violation rate | 56.7% | 3.3% |
| Mean true safety clearance | -2.106 mm | +8.555 mm |
| Minimum true safety clearance | -12.718 mm | -2.876 mm |
| Mean planning time | 0.818 s | 1.609 s |
| Mean iterations | 230.1 | 475.5 |
| Mean path cost | 2.1614 | 2.6381 |

Within this simulated 30-trial experiment, uncertainty-aware planning reduced the observed ground-truth safety-violation rate from:

```text
56.7% -> 3.3%
```

This represents a **53.4 percentage-point reduction** in observed violations.

The improvement was accompanied by increased:

- planning time;
- planner iterations;
- trajectory cost.

The experiment therefore demonstrates a measurable **safety-versus-efficiency trade-off** rather than a cost-free improvement.

These results are specific to the implemented simulation and experimental configuration.

---

# Uncertainty Parameter Sweep

To investigate whether the behaviour persisted outside a single uncertainty configuration, experiments were performed at localisation standard deviations of:

```text
2 mm
5 mm
8 mm
```

with uncertainty multipliers:

```text
k = 1
k = 2
k = 3
```

Ten trials were evaluated per condition during this exploratory parameter sweep.

---

## 2 mm Localisation Uncertainty

The deterministic baseline produced a:

```text
30% ground-truth safety-violation rate
```

All tested uncertainty-aware multiplier conditions produced:

```text
0% observed safety violations
0% observed physical collisions
100% planning success
```

within these trials.

---

## 5 mm Localisation Uncertainty

The deterministic baseline produced:

```text
60% safety violations
```

Increasing uncertainty protection progressively reduced the observed violation rate.

At:

```text
k = 2
k = 3
```

no ground-truth safety violations were observed in the tested trials.

Higher protection was accompanied by increased planning cost and computation.

---

## 8 mm Localisation Uncertainty

The highest tested uncertainty condition exposed a stronger difference between planning strategies.

The deterministic baseline produced:

```text
100% safety violations
10% physical collisions
```

Using uncertainty-aware protection with:

```text
k = 1
```

reduced the observed safety-violation rate to:

```text
20%
```

with no observed physical collisions.

At:

```text
k = 2
```

the tested successful trajectories produced:

```text
0% observed safety violations
0% observed physical collisions
90% planning success
```

At:

```text
k = 3
```

the tested successful trajectories produced:

```text
0% observed safety violations
0% observed physical collisions
80% planning success
```

Mean planning time at \(k=3\) increased to approximately:

```text
3.05 s
```

These results demonstrate an important engineering trade-off:

> Increasing uncertainty-dependent protection can improve trajectory safety while simultaneously increasing computational cost and reducing planning feasibility.

Because the parameter sweep currently contains a limited number of trials per condition, these findings should be interpreted as exploratory simulation evidence rather than definitive statistical validation.

---

# Software Verification

Automated tests currently cover:

- coordinate transformations;
- surgical instrument kinematics;
- RCM constraints;
- trajectory generation;
- workspace geometry;
- collision detection;
- safety evaluation;
- RRT planning;
- path optimisation;
- uncertainty representation;
- noisy anatomical perception;
- planner-facing uncertainty models.

Current regression status:

```text
124 passed
0 failed
```

The complete regression suite is executed before repository synchronisation through the project's local GitHub workflow.

---

# Repository Structure

```text
src/
├── geometry/
├── perception/
│   ├── __init__.py
│   ├── perception.py
│   ├── planning.py
│   └── uncertainty.py
│
├── robotics/
│   ├── instrument.py
│   ├── planner.py
│   └── safety.py
│
└── simulation/
    ├── scene.py
    ├── benchmark.py
    ├── uncertainty_experiment.py
    ├── uncertainty_benchmark.py
    └── uncertainty_sweep.py

tests/
├── ...
├── test_planner.py
├── test_safety.py
├── test_uncertainty.py
├── test_perception.py
└── test_perception_planning.py
```

---

# Current Project Status

### Completed

- [x] RCM-constrained instrument model
- [x] safety-critical anatomical geometry
- [x] collision and clearance evaluation
- [x] deterministic RRT motion planner
- [x] path optimisation
- [x] deterministic multi-seed benchmarking
- [x] Gaussian localisation uncertainty model
- [x] noisy anatomical perception
- [x] uncertainty-aware safety margins
- [x] uncertainty-aware RRT integration
- [x] hidden ground-truth safety evaluation
- [x] matched baseline comparison
- [x] Monte Carlo uncertainty benchmark
- [x] uncertainty parameter sweep
- [x] automated regression testing

### Planned

- [ ] simulated surgical camera model
- [ ] viewpoint-dependent perception quality
- [ ] observation-dependent uncertainty
- [ ] active viewpoint selection
- [ ] task-relevance modelling
- [ ] fixed-view perception baseline
- [ ] generic uncertainty-driven active perception
- [ ] task-aware active perception
- [ ] larger experimental evaluation
- [ ] statistical analysis
- [ ] research visualisation and figures

---

# Next Research Stage

The next stage extends the framework from **uncertainty-aware motion planning** toward **active perception**.

Instead of treating localisation uncertainty as fixed, the simulated camera will provide observations whose uncertainty depends on viewpoint and visibility.

This will enable investigation of whether the system should actively move or redirect perception to obtain more informative observations before executing safety-critical motion.

The eventual comparison will investigate:

```text
Fixed-view perception
        vs
Generic uncertainty-driven active perception
        vs
Task-aware uncertainty-driven active perception
```

The central question will be whether reducing uncertainty specifically relevant to the intended trajectory provides a better safety-efficiency trade-off than reducing global perception uncertainty.

---

# Limitations

The current framework is intentionally a simplified research simulation.

Current limitations include:

- simplified spherical anatomical geometry;
- simulated rather than learned visual perception;
- Gaussian localisation uncertainty assumptions;
- simplified surgical instrument geometry;
- simplified environment dynamics;
- no tissue deformation;
- no force or contact interaction modelling;
- no physical robotic platform;
- no patient data;
- no clinical validation.

These limitations must be considered when interpreting the experimental results.

---

# Safety and Intended Use

This repository contains a simulation-based engineering research prototype.

It is **not a medical device**, has not undergone clinical validation, medical electrical safety testing, regulatory approval, or medical-device certification.

It must not be used for patient monitoring, diagnosis, treatment, surgical guidance, clinical decision-making, or any other clinical purpose.
