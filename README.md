# Task-Aware Active Perception Coupled with Uncertainty-Aware Motion Planning for Simulated Surgical Navigation

A simulation-based surgical robotics research framework investigating how active camera viewpoint selection, task information, localisation uncertainty, and safety-critical motion planning interact in minimally invasive surgical navigation.

The repository implements a complete perception-to-planning pipeline in which deployable strategies operate only on estimated anatomy, while hidden simulator ground truth is reserved for observation generation and independent safety evaluation.

> **Research prototype:** This repository is intended for simulation and engineering research only. It is not a clinical system or medical device.

---

## Research Question

The final Phase 1 research question is:

> **Given the same uncertain initial anatomical estimate and approximately equal camera-motion expenditure, does incorporating the intended surgical trajectory improve downstream simulated navigation?**

The final experiment compares eight pre-specified strategies:

1. Fixed View
2. Random Active
3. Generic Active
4. Movement-Budget-Matched Generic
5. Alignment-Only
6. Task-Weighted Information-Only
7. Full Task-Aware
8. Privileged Oracle

The frozen primary comparison is:

```text
Full Task-Aware
        versus
Movement-Budget-Matched Generic
```

The primary endpoint is:

```text
Safe-navigation success rate
```

The inferential unit is the held-out simulated scenario rather than individual stochastic repetitions.

---

## System Pipeline

```text
Hidden simulator anatomy
        |
        +----> simulated observation generation
        |
        +----> independent final evaluation only


Nominal camera prior
        |
        v
Initial noisy anatomical observation
        |
        v
Shared estimated anatomy
        |
        v
Estimate-centred candidate viewpoints
        |
        +-------------------------------+
        |                               |
        v                               v
Generic scoring                  Task-aware scoring
(no task trajectory)             (+ intended trajectory)
        |                               |
        +---------------+---------------+
                        |
                        v
              Selected camera pose
                        |
                        v
               Final observation
                        |
                        v
       Estimated anatomy + uncertainty
                        |
                        v
      Uncertainty-inflated geometry
                        |
                        v
            Collision-aware RRT
                        |
                        v
        Hidden ground-truth evaluation
```

Simulator ground truth is not exposed to deployable viewpoint-selection strategies.

The Oracle is the only explicitly privileged analysis-only strategy permitted to use ground-truth geometry during viewpoint selection.

---

## What This Project Implements

The repository currently includes:

- RCM-constrained minimally invasive surgical instrument modelling
- joint-space forward kinematics
- surgical workspace and anatomical geometry
- collision and safety-margin evaluation
- collision-aware RRT planning
- edge validation and trajectory shortcutting
- path-cost evaluation
- explicit Gaussian localisation uncertainty
- separation of estimated and hidden ground-truth anatomy
- uncertainty-aware planning margins
- camera-pose and viewpoint modelling
- visibility and occlusion modelling
- Generic Active Perception
- task representation and task relevance
- Task-Aware Active Perception
- scene-wide fair viewpoint scoring
- truth-isolated candidate generation
- movement-budget matching
- Fixed and Random baselines
- task-mechanism ablations
- privileged Oracle analysis
- development / validation / held-out scenario separation
- frozen experimental protocol
- matched stochastic benchmarking
- paired statistical analysis
- scenario-cluster bootstrap confidence intervals
- paired sign-flip permutation testing
- Holm correction for secondary inference
- multi-scenario robustness evaluation
- uncertainty stress testing
- formal uncertainty calibration
- synthetic visual-quality degradation
- planning-time, iteration, and path-cost evaluation
- automated verification
- ROS 2 Jazzy integration

---

## Core Uncertainty-Aware Planning Idea

The planner does not receive perfect simulator geometry.

```text
Hidden ground truth
        |
        +----> evaluation only

Noisy perceived anatomy
        |
        +----> motion planning
```

For the principal isotropic localisation model:

```text
Σ = σ²I
```

Planning geometry is inflated according to:

```text
m_plan = m_base + kσ
```

where:

- `m_base` is the nominal safety margin,
- `σ` is predicted localisation uncertainty,
- `k` is the uncertainty multiplier.

This directly couples perception quality to planning conservatism and feasibility.

---

# Phase 1: Examiner-Proof Experimental Redesign

Earlier versions of the project showed a very large apparent Task-Aware advantage.

Phase 1 deliberately re-examined that result under a substantially stricter methodology.

The redesign addressed:

- weaker Generic comparison
- unequal camera-motion expenditure
- ground-truth influence on candidate geometry
- scoring-scale imbalance
- lack of independent development / validation / held-out separation
- insufficient baseline diversity
- absence of a frozen pre-held-out statistical protocol

The final Phase 1 architecture was designed so that the proposed Task-Aware method was allowed to succeed, tie, or fail.

No post-held-out retuning was permitted.

---

## Information Isolation

A critical Phase 1 requirement was separation between simulator truth and information available to deployable decision algorithms.

The final information flow is:

```text
Fixed nominal workspace prior
        |
        v
Initial camera pose
        |
        v
Hidden simulator anatomy
        |
        +----> noisy observation generator
                        |
                        v
              estimated anatomy
                        |
                        v
           estimated target centre
                        |
                        v
        common candidate viewpoint set
                /               \
               /                 \
              v                   v
       Generic scorer       Task-Aware scorer
       no trajectory        planned trajectory
              \                   /
               \                 /
                v               v
             selected viewpoint
                    |
                    v
              final observation
                    |
                    v
           uncertainty-aware planner
                    |
                    v
        hidden ground-truth evaluation
```

Deployable strategies do not receive:

- true anatomical centres
- true target positions
- true safety-critical anatomy for viewpoint scoring
- truth-centred candidate viewpoints

The Oracle is the only strategy allowed to use simulator truth during selection, and it remains analysis-only.

---

## Final Phase 1 Experimental Design

Before executing the final held-out experiment, the following were frozen:

- strategy definitions
- scenario split
- candidate-generation policy
- movement weights
- random-seed rules
- primary comparison
- primary endpoint
- statistical tests
- superiority criterion
- interpretation rules

### Frozen movement weights

```text
Movement-Budget-Matched Generic = 0.072
Full Task-Aware                 = 0.200
```

### Scenario split

The final protocol separates:

```text
Development
Validation
Held-out test
```

The held-out test contains:

```text
30 untouched scenarios
× 10 matched repetitions
× 8 strategies
=
2,400 strategy evaluations
```

The 10 repetitions within each scenario are treated as nested stochastic repetitions rather than 300 independent environments.

---

## Experimental Provenance

### Pre-held-out Git commit

```text
250b40d47d115397c8254556096a38525ef92aed
```

### Scenario manifest SHA-256

```text
490922fbc9f91743262257ff594a7440651016085207b3e8867b9e1bdce8ddd9
```

### Frozen protocol SHA-256

```text
dc6537d3ff75832ccbe48c9b2690c8e966711b31806076cfbc2721488dbf6361
```

### Final raw held-out evidence SHA-256

```text
3c6f0e8dee177dbfb36019ce0242b53c4a4166391824dbdd953fa8fd8dbcddcb
```

These hashes provide traceability between:

```text
scenario definition
        ↓
experimental protocol
        ↓
pre-held-out implementation
        ↓
raw held-out evidence
```

---

# Final Held-Out Results

## Primary Comparison

The frozen primary comparison was:

```text
Full Task-Aware
        versus
Movement-Budget-Matched Generic
```

### Safe-navigation success

| Metric | Full Task-Aware | Budget-Matched Generic |
|---|---:|---:|
| Safe-navigation success | **61.33%** | **71.33%** |
| Difference, Task − Generic | **−10.00 percentage points** | |
| 95% bootstrap CI | **[−17.00, −3.00] pp** | |
| Two-sided permutation p-value | **0.012390** | |
| Task-Aware superiority established | **No** | |

The frozen held-out experiment therefore **did not establish superiority of Full Task-Aware Active Perception** over the development movement-budget-matched Generic comparator.

Instead, the observed held-out effect favoured Generic by approximately 10 percentage points.

This negative result is retained without post-hoc retuning.

---

## Held-Out Movement-Budget Limitation

The intended movement match did not fully generalise from development and validation to the held-out distribution.

```text
Full Task-Aware movement:          93.544 mm
Budget-Matched Generic movement:  122.548 mm

Absolute difference:
29.004 mm

Relative mismatch:
31.01%

Previous validation tolerance:
10%
```

Therefore:

> **The held-out difference in safe-navigation performance cannot be attributed cleanly to task awareness alone.**

The project does not claim that Generic would necessarily outperform Full Task-Aware under perfectly matched held-out camera-motion expenditure.

The failed movement-budget generalisation is retained as an explicit limitation rather than corrected using held-out data.

---

# Eight-Strategy Held-Out Summary

| Strategy | Safe Navigation | Planning Success | Movement | Localisation Error | Predicted Sigma | Collision | Safety Violation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed | 13.67% | 14.67% | 0.000 mm | 42.584 mm | 26.350 mm | 0.33% | 1.00% |
| Random Active | **76.67%** | 78.67% | 238.815 mm | 14.158 mm | 8.983 mm | 0.33% | 2.00% |
| Generic Active | 69.33% | 70.67% | 113.774 mm | 16.602 mm | 10.514 mm | 0.00% | 1.33% |
| Budget-Matched Generic | 71.33% | 73.00% | 122.548 mm | 15.858 mm | 9.988 mm | 0.00% | 1.67% |
| Alignment-Only | **74.67%** | 76.67% | 131.505 mm | 14.058 mm | 9.137 mm | 0.00% | 2.00% |
| Information-Only | 56.00% | 57.33% | 74.999 mm | 21.562 mm | 13.697 mm | 0.00% | 1.33% |
| Full Task-Aware | 61.33% | 63.67% | 93.544 mm | 19.505 mm | 12.448 mm | 0.00% | 2.33% |
| Oracle | **97.00%** | **100.00%** | 111.203 mm | 5.407 mm | 3.309 mm | 0.00% | 3.00% |

The Oracle is analysis-only and is not considered a deployable strategy.

---

# Mechanism Ablation

The final held-out experiment provides a substantially stronger mechanism analysis than the earlier controlled ablation.

## Alignment-Only

```text
Safe-navigation success: 74.67%
Planning success:        76.67%
Movement:               131.505 mm
Localisation error:      14.058 mm
Predicted sigma:          9.137 mm
Mean task alignment:      0.988
```

## Task-Weighted Information-Only

```text
Safe-navigation success: 56.00%
Planning success:        57.33%
Movement:                74.999 mm
Localisation error:      21.562 mm
Predicted sigma:         13.697 mm
Mean task alignment:      0.976
```

## Full Task-Aware

```text
Safe-navigation success: 61.33%
Planning success:        63.67%
Movement:                93.544 mm
Localisation error:      19.505 mm
Predicted sigma:         12.448 mm
Mean task alignment:      0.987
```

---

## Alignment Mechanism

The strongest held-out mechanism comparison is:

```text
Full Task-Aware
minus
Alignment-Only

Safe-navigation effect:   -13.33 pp
Planning-success effect:  -13.00 pp
Holm-adjusted p-value:      0.0014

Localisation-error effect: +5.446 mm
Predicted-sigma effect:    +3.312 mm
Camera-movement effect:   -37.962 mm
```

Full Task-Aware and Alignment-Only achieve almost identical task-alignment scores:

```text
Alignment-Only:  0.988
Full Task-Aware: 0.987
```

The degradation therefore does not appear to arise from poor task alignment.

Instead, the held-out ablation indicates that the current **task-weighted information component is the principal performance bottleneck**.

---

## Information Mechanism

The complementary comparison is:

```text
Full Task-Aware
minus
Information-Only

Safe-navigation effect:   +5.33 pp
Planning-success effect:  +6.33 pp
Holm-adjusted p-value:      0.0041

Localisation-error effect: -2.058 mm
Predicted-sigma effect:    -1.248 mm
Camera-movement effect:   +18.545 mm
```

Adding task alignment to Information-Only therefore improves:

- localisation
- predicted uncertainty
- planning feasibility
- safe-navigation success

The mechanism picture is:

```text
Task-weighted information only
        |
        v
56.00% safe navigation

        + task alignment
        |
        v
61.33% safe navigation

Uniform scene information
        + task alignment
        |
        v
74.67% safe navigation
```

This suggests that the current task-weighted information formulation concentrates viewpoint selection too strongly on selected task-relevant information while accepting poorer overall scene estimation and/or reduced exploratory camera movement.

A plausible system-level pathway is:

```text
task-weighted viewpoint preference
        |
        v
weaker overall anatomical estimate
        |
        v
larger predicted uncertainty
        |
        v
larger uncertainty-inflated planning geometry
        |
        v
reduced planning feasibility
        |
        v
lower safe-navigation success
```

This is a simulation-based mechanistic interpretation rather than a clinical conclusion.

---

# Oracle Diagnostic

The privileged Oracle achieved:

```text
Safe-navigation success: 97.00%
Planning success:       100.00%
Movement:               111.203 mm
Localisation error:       5.407 mm
Predicted sigma:          3.309 mm
```

The Oracle selects from the **same estimate-centred candidate set** used by the deployable active strategies.

Its privilege is limited to access to simulator truth during scoring.

The large performance gap between Full Task-Aware and Oracle demonstrates that:

- the candidate search space can support substantially better outcomes;
- the downstream planner can achieve high feasibility when anatomical estimation is sufficiently accurate;
- viewpoint scoring under uncertain estimated anatomy remains a major bottleneck.

The Oracle is not deployable and is not evidence of achievable real-world clinical performance.

---

# Random Active Interpretation

Random Active achieved the strongest safe-navigation rate among the non-Oracle strategies:

```text
76.67%
```

However, it also used by far the greatest camera movement:

```text
238.815 mm
```

compared with:

```text
Full Task-Aware:          93.544 mm
Budget-Matched Generic:  122.548 mm
Alignment-Only:          131.505 mm
```

Therefore the correct interpretation is not simply that random viewpoint selection is superior.

Instead:

> **Aggressive camera exploration can improve perception and downstream navigation in this simulated environment, but at a substantial camera-motion cost.**

This reinforces the need to explicitly account for movement expenditure in active-perception comparisons.

---

# Secondary Planning Comparisons

Planning-success inference used paired scenario-level sign-flip permutation tests.

The complete frozen secondary planning-success family was corrected using the Holm procedure.

| Full Task-Aware minus comparator | Planning Effect | Holm-adjusted p |
|---|---:|---:|
| Budget-Matched Generic | −9.33 pp | 0.0476 |
| Same-Weight Generic | −7.00 pp | 0.0533 |
| Fixed | +49.00 pp | 0.0001 |
| Random Active | −15.00 pp | 0.0476 |
| Alignment-Only | −13.00 pp | 0.0014 |
| Information-Only | +6.33 pp | 0.0041 |
| Oracle | −36.33 pp | 0.0001 |

Secondary results do not replace the frozen primary conclusion.

---

# Historical Controlled Experiments

Earlier stages of the project used a simpler three-strategy architecture:

```text
Fixed
vs
Generic
vs
Task-Aware
```

These experiments remain useful for understanding project development and controlled mechanism behaviour, but they are **not the final held-out evidence**.

---

## Earlier 10-Scenario Robustness Benchmark

The earlier benchmark used:

```text
10 simulated scene variations
× 10 matched repetitions
× 3 strategies
=
300 strategy evaluations
```

| Metric | Fixed | Generic | Task-Aware |
|---|---:|---:|---:|
| Mean localisation error | 25.964 mm | 16.934 mm | 4.682 mm |
| Mean predicted sigma | 17.777 mm | 12.267 mm | 3.003 mm |
| Camera movement | 0 mm | 15.628 mm | 109.985 mm |
| Planning success | 45% | 63% | 100% |
| Safe-navigation success | 43% | 61% | 97% |
| Collision rate | 0% | 0% | 0% |
| Safety-violation rate | 2% | 2% | 3% |
| Worst-scenario safe-navigation success | 0% | 0% | 90% |

These historical results originally suggested a very large Task-Aware advantage.

However, later Phase 1 work identified important fairness limitations in the earlier architecture, including:

- weaker Generic comparison
- substantially different camera-motion expenditure
- candidate geometry influenced by simulator truth
- earlier task-scoring scale imbalance
- absence of a completely untouched held-out test set

The final Phase 1 experiment corrected these issues before final held-out evaluation.

The large historical Task-Aware advantage did **not** survive the stricter final methodology.

This is retained as an important scientific finding rather than hidden.

---

# Earlier Mechanism Ablation

Earlier controlled experiments evaluated:

```text
Generic
Alignment-Only
Uncertainty-Only
Full Task-Aware
```

The earlier experiments suggested that task alignment dominated explicit uncertainty weighting.

Increasing the explicit uncertainty-selection weight through:

```text
0
0.25
1
4
16
```

did not alter the selected viewpoint under the original formulation.

This motivated the later scoring reformulation and stricter Phase 1 mechanism experiments.

Localisation uncertainty nevertheless remains important downstream because it directly modifies uncertainty-inflated planning geometry.

The project is therefore best described as:

> **Task-aware active perception coupled with uncertainty-aware motion planning**

rather than as an uncertainty-driven viewpoint-selection algorithm.

---

# Formal Uncertainty Calibration

The localisation-uncertainty model was evaluated using:

```text
10 scenarios
60 viewpoint conditions
6,000 observations
```

| Diagnostic | Expected | Observed |
|---|---:|---:|
| Mean normalised squared error | 3.000 | 3.024 |
| Mean radial error / sigma | 1.596 | 1.603 |
| 50% coverage | 50% | 49.42% |
| 90% coverage | 90% | 89.95% |
| 95% coverage | 95% | 94.98% |
| 99% coverage | 99% | 98.88% |

The implemented uncertainty model was classified as:

```text
well_calibrated_under_simulation
```

This applies only to the implemented simulation model.

It must not be interpreted as physical sensor calibration.

---

# Synthetic Visual-Quality Degradation

A supplementary stress test represented reduced visual quality using:

```text
sigma_degraded = sigma_nominal / sqrt(quality)
```

Under the most severe tested degradation:

| Metric | Fixed | Generic | Task-Aware |
|---|---:|---:|---:|
| Localisation error | 60.222 mm | 42.250 mm | 9.812 mm |
| Predicted sigma | 35.555 mm | 24.533 mm | 6.007 mm |
| Planning success | 35% | 55% | 95% |
| Safe-navigation success | 35% | 55% | 90% |

This is a synthetic observation-quality stress test, not a physical photometric model.

It predates the final Phase 1 fairness architecture and is retained as supplementary controlled evidence rather than final held-out evidence.

---

# Planning Efficiency

An earlier supplementary planning-efficiency benchmark used:

```text
10 scenarios
× 5 matched repetitions
=
50 matched planning units
```

| Metric | Fixed | Generic | Task-Aware |
|---|---:|---:|---:|
| Planning success | 46% | 66% | 100% |
| Safe-navigation success | 38% | 60% | 92% |
| Camera movement | 0 mm | 15.628 mm | 109.985 mm |
| Planning time, all attempts | 0.547 s | 0.882 s | 1.239 s |
| RRT iterations, all attempts | 142.8 | 227.2 | 333.7 |
| Planning time, successful plans | 1.189 s | 1.337 s | 1.239 s |
| Path cost, successful plans | 2.517 | 2.558 | 2.499 |

On the 33 matched trials where both Generic and Task-Aware succeeded:

```text
Task-Aware minus Generic path cost:
-0.107

95% CI:
[-0.257, -0.015]
```

These results remain supplementary historical evidence rather than the final Phase 1 comparison.

---

# Final Phase 1 Scientific Conclusions

## 1. Full Task-Aware superiority was not established

On the untouched held-out test set:

```text
Full Task-Aware:          61.33%
Budget-Matched Generic:  71.33%
```

The primary Task-Aware effect was:

```text
-10.00 percentage points

95% CI:
[-17.00, -3.00]

Two-sided paired permutation p:
0.012390
```

The frozen superiority criterion was therefore not satisfied.

---

## 2. Held-out movement-budget matching failed to generalise

The development-selected comparator moved substantially farther on held-out scenarios:

```text
Task-Aware:   93.544 mm
Generic:     122.548 mm

Relative mismatch:
31.01%
```

This prevents clean causal attribution of the primary performance difference specifically to task information.

---

## 3. Task alignment itself appears useful

Alignment-Only achieved:

```text
74.67% safe-navigation success
```

compared with:

```text
61.33% Full Task-Aware
56.00% Information-Only
```

The final ablation therefore supports task alignment more strongly than the current task-weighted information formulation.

---

## 4. Task-weighted information is the current mechanism bottleneck

Task-weighted information reduced camera movement but also increased:

- localisation error
- predicted uncertainty
- planning failure
- safe-navigation failure

relative to Alignment-Only.

The current formulation should therefore be redesigned rather than simply assigned a larger weight.

---

## 5. Uncertainty remains important downstream

Although explicit uncertainty weighting has not consistently improved viewpoint ranking, localisation uncertainty continues to affect motion planning through uncertainty-inflated geometry.

Perception uncertainty therefore remains an important system variable even when it is not independently useful as a viewpoint-selection reward.

---

## 6. Better viewpoint selection remains possible

The Oracle achieved:

```text
97% safe-navigation success
100% planning success
```

from the same candidate search space.

The remaining gap therefore motivates improved perception and viewpoint-selection methods rather than indicating a fundamental planning ceiling.

---

## 7. Aggressive exploration can improve performance at substantial cost

Random Active achieved:

```text
76.67% safe-navigation success
```

but required:

```text
238.815 mm
```

mean camera movement.

The result highlights a genuine exploration-versus-motion-cost trade-off.

---

# Scientific Integrity

The final Phase 1 experiment was deliberately designed so that the proposed method was allowed to fail.

The workflow was:

```text
development tuning
        |
        v
independent validation
        |
        v
freeze strategy definitions
        |
        v
freeze scenario manifest
        |
        v
freeze statistical protocol
        |
        v
Git commit before held-out execution
        |
        v
execute untouched held-out experiment
        |
        v
run pre-specified primary analysis
        |
        v
run pre-specified secondary/ablation analysis
        |
        v
retain result without retuning
```

The observed negative primary result was retained.

No movement weight, candidate-generation rule, primary endpoint, scenario, statistical test, superiority threshold, or random-seed rule was changed in response to held-out performance.

---

# Post-Held-Out Analysis-Code Correction

After held-out execution, the statistical validator initially rejected raw records containing:

```text
path_cost = inf
```

for failed planning attempts.

This was an implementation error in the validator rather than an experimental-design change.

The frozen protocol had already specified that path cost should be analysed only for pairs where both strategies successfully generated a plan.

The validator was therefore corrected so that:

```text
successful planning
-> path cost must be finite

failed planning
-> +inf path cost is permitted
```

The raw held-out CSV was not modified.

The 2,400-strategy held-out experiment was not rerun as a consequence of this correction.

Regression tests were added for the corrected behaviour.

---

# Verification

Final repository-wide automated regression:

```text
612 passed
0 failed
```

Run:

```bash
python -m pytest -q
```

The suite covers:

- geometry
- surgical kinematics
- safety
- collision checking
- RRT planning
- uncertainty modelling
- camera modelling
- active perception
- task-aware perception
- fair candidate generation
- information isolation
- movement-budget matching
- scenario splitting
- frozen-protocol integrity
- baseline implementations
- end-to-end integration
- robustness experiments
- mechanism ablations
- stress testing
- uncertainty calibration
- synthetic visual degradation
- efficiency benchmarking
- held-out execution guards
- statistical inference
- secondary-analysis logic
- provenance checks

---

# Reproducible Environment

The Python research environment is defined in:

```text
environment.yml
```

The project targets Python 3.11.

```bash
conda env create -f environment.yml
conda activate surgical-navigation
python -m pytest -q
```

The primary development environment used Python 3.11 with PyBullet.

---

# Key Experiments

## Historical and Supplementary Experiments

```bash
python -m src.simulation.three_strategy_trial
python -m src.simulation.three_strategy_statistical_benchmark
python -m src.simulation.three_strategy_robustness_benchmark
python -m src.simulation.final_evidence_package
python -m src.simulation.uncertainty_calibration_benchmark
python -m src.simulation.illumination_degradation_benchmark
python -m src.simulation.three_strategy_efficiency_benchmark
```

## Phase 1 Development and Validation

```bash
python -m src.simulation.fair_scene_development_sweep
python -m src.simulation.movement_budget_matching
python -m src.simulation.phase1_validation_gate
python -m src.simulation.phase1_frozen_protocol
```

## Final Held-Out Analysis

Final held-out evidence is retained under:

```text
results/phase1_held_out/
```

Analyse the existing evidence using:

```bash
python -m src.simulation.phase1_statistics
python -m src.simulation.phase1_secondary_analysis
```

The final held-out experiment should not be casually rerun when reproducing the documented result because the existing raw result artifact is retained with a SHA-256 digest.

---

## Standalone Simulation

```bash
python visual_surgical_simulation.py
```

---

# Repository Structure

```text
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
|   |-- phase1_fair_scene_development/
|   |-- phase1_movement_budget/
|   |-- phase1_protocol/
|   |-- phase1_scenario_splits/
|   |-- phase1_validation/
|   |-- phase1_held_out/
|   |-- final_evidence/
|   |-- supplementary_uncertainty_calibration/
|   |-- supplementary_illumination/
|   `-- supplementary_efficiency/
|
|-- ros2_jazzy/
|
|-- src/
|   |-- geometry/
|   |-- perception/
|   |-- robotics/
|   `-- simulation/
|
|-- tests/
|
|-- environment.yml
|-- pytest.ini
|-- visual_surgical_simulation.py
`-- README.md
```

---

# Result Artifacts

Important final Phase 1 result artifacts include:

```text
results/phase1_scenario_splits/scenario_manifest.json

results/phase1_protocol/frozen_phase1_protocol.json

results/phase1_movement_budget/movement_budget_match_corrected.json

results/phase1_validation/validation_gate_corrected.json

results/phase1_held_out/held_out_raw_records.csv

results/phase1_held_out/held_out_primary_analysis.json

results/phase1_held_out/held_out_secondary_analysis.json
```

Earlier and supplementary evidence remains available under:

```text
results/final_evidence/

results/supplementary_uncertainty_calibration/

results/supplementary_illumination/

results/supplementary_efficiency/
```

Machine-readable CSV and JSON outputs are retained for reproducibility and analysis.

---

# ROS 2 Integration

A separate ROS 2 Jazzy workspace is included under:

```text
ros2_jazzy/
`-- ros2_ws/
```

It provides experimental integration for:

- perception communication
- planning
- safety bridging
- viewpoint communication
- visualisation

ROS 2 is intentionally separated from the core Python simulation environment.

No physical surgical robot implementation is claimed.

---

# Current Limitations

Important limitations include:

- simulation-only validation
- simplified anatomical geometry
- engineered scene variations rather than independent patient anatomies
- simulated rather than learned visual perception
- simplified camera and observation models
- isotropic Gaussian localisation uncertainty
- synthetic rather than physical illumination degradation
- no deformable tissue model
- no force or tactile sensing
- no dynamic anatomy
- no calibration-drift model
- no realistic specular-reflection model
- no patient data
- no animal or cadaver experiments
- no physical robotic-platform validation
- no clinical validation
- active candidate generation remains simplified
- task relevance is based on estimated synthetic geometry
- no real endoscopic image pipeline
- no physical camera calibration
- no image-to-robot registration experiment
- no real-time physical control loop
- held-out movement-budget matching did not remain within the previous 10% tolerance
- Random Active used substantially greater movement than structured strategies
- the current task-weighted information objective reduced held-out performance relative to Alignment-Only

Results demonstrate behaviour under the implemented simulated perturbations only.

They do not establish:

- patient generalisation
- anatomical generalisation
- physical sensor performance
- physical robot performance
- surgical safety
- clinical efficacy

---

# Safety and Intended Use

This repository contains a simulation-based engineering research prototype.

It is not a medical device and must not be used for:

- diagnosis
- treatment
- patient monitoring
- surgical guidance
- clinical decision-making
- autonomous clinical intervention
- any other clinical purpose

No claim of ISO 14971, IEC 62366, regulatory, or clinical compliance is made.

---

# Final Research Position

The project does **not** conclude that the current Full Task-Aware formulation is superior to a strengthened Generic Active Perception baseline.

Instead, the final Phase 1 evidence shows that:

> **Task information can materially change active-perception decisions, but the way that task information is incorporated matters. Task alignment showed promising held-out behaviour, while the current task-weighted information formulation degraded localisation quality, predicted uncertainty, planning feasibility, and safe-navigation success.**

At the same time:

> **Localisation uncertainty remains important downstream because it directly changes uncertainty-inflated safety geometry used by the motion planner.**

The final held-out experiment therefore transforms the project from a simple demonstration that a proposed method appears to work into a stricter investigation of:

```text
when task-aware active perception helps
                |
                v
when it fails
                |
                v
which mechanism is responsible
                |
                v
how perception uncertainty propagates into planning
```

The most important Phase 1 engineering insight is that **task alignment and task-weighted information should not be treated as interchangeable forms of task awareness**.

Under the current formulation:

```text
Alignment-Only
        >
Full Task-Aware
        >
Information-Only
```

for held-out navigation feasibility.

The next stages therefore focus on addressing the limitations revealed by Phase 1 rather than tuning the existing formulation against the held-out test set.

---

# Next Research Phases

The broader project roadmap continues with:

## Phase 2 — Advanced Robot Kinematics and Reachable Viewpoint Planning

- explicit robot degrees of freedom
- forward and inverse kinematics
- Jacobians
- workspace analysis
- singularity and manipulability analysis
- joint limits
- RCM-constrained endoscope motion
- viewpoint reachability
- collision-aware viewpoint feasibility
- execution-cost-aware candidate selection

## Phase 3 — Genuine Image-Based Surgical Perception

- synthetic/endoscopic-like image generation
- anatomy and instrument visibility
- occlusion
- lighting degradation
- image filtering
- thresholding
- edge detection
- morphology
- contours
- visual localisation
- camera projection
- pixel-to-ray geometry

## Phase 4 — Machine Learning / Deep Learning Perception

- labelled datasets
- PyTorch training pipeline
- segmentation
- held-out image testing
- Dice / IoU / precision / recall
- localisation accuracy
- inference latency
- predictive uncertainty

## Phase 5 — 3-D Perception and Registration

- 3-D target localisation
- camera coordinate frames
- rigid registration
- fiducial registration
- image-to-robot transforms
- FRE
- TRE
- registration uncertainty propagation

## Phase 6 — Dynamic Anatomy and State Estimation

- time-varying anatomy
- respiratory-like motion
- temporal observations
- Kalman filtering
- covariance propagation
- prediction and update
- tracking robustness
- occlusion recovery

## Phase 7 — Model Mismatch and Generalisation

- procedural scene randomisation
- biased uncertainty
- under-confidence
- over-confidence
- heavy-tailed errors
- outliers
- temporal correlation
- drift
- visual failure modes
- large post-freeze robustness evaluation

## Phase 8 — Medical-Device-Style Safety Engineering

- hazard analysis
- FMEA
- safety supervisor
- system states
- uncertainty monitoring
- clearance monitoring
- joint-limit monitoring
- stale-data detection
- timeout handling
- fault injection
- ISO 14971 awareness
- IEC 62366 awareness

## Phase 9 — Preliminary Benchtop Validation

- camera
- physical phantom
- known geometry
- fiducials
- calibration
- localisation-error measurement
- repeatability
- registration
- viewpoint accuracy
- preliminary simulation-to-real comparison

## Phase 10 — ROS 2 and Gazebo Deployment

- surgical/endoscopic robot model
- URDF / Xacro
- TF2
- RViz
- Gazebo
- ros2_control
- camera nodes
- perception nodes
- state-estimation nodes
- viewpoint planner
- motion planner
- safety supervisor
- trajectory execution
- rosbag2
- QoS
- lifecycle management

## Phase 11 — Control and Real-Time Behaviour

- trajectory tracking
- position control
- velocity control
- PID
- tracking error
- settling behaviour
- loop-frequency measurement
- perception latency
- planning latency
- control latency
- latency stress testing

## Phase 12 — Final End-to-End Image-Guided Experiment

```text
image
  ↓
segmentation / detection
  ↓
3-D localisation
  ↓
registration
  ↓
state estimate + covariance
  ↓
task-aware viewpoint selection
  ↓
reachability / singularity / collision checks
  ↓
uncertainty-aware planning
  ↓
safety supervisor
  ↓
trajectory execution
  ↓
Gazebo / preliminary benchtop system
  ↓
targeting and navigation evaluation
```

## Phase 13 — Final Verification and Portfolio Packaging

- full regression
- ROS integration tests
- C++ tests where introduced
- fresh environment build
- CI
- requirements
- traceability
- architecture documentation
- frozen experiment configurations
- publication-quality plots
- demonstration media
- reproducibility instructions
- explicit simulation-versus-physical evidence separation
- final tagged GitHub release

---

# Research Scope

The project should currently be described as:

> **Task-aware active perception coupled with uncertainty-aware motion planning for simulated surgical navigation.**

It should not currently be described as:

- clinically validated surgical robotics
- autonomous surgery
- clinically safe surgical navigation
- real patient navigation
- physical robot validation
- real endoscopic perception
- medical-device validation

Those capabilities require later project phases and separate evidence.

---

# Status

Phase 1 has completed its:

- final held-out simulation experiment
- frozen primary statistical analysis
- secondary and mechanism analysis
- full repository regression test

The remaining Phase 1 repository work is final documentation, traceability updating, evidence commit, and repository freeze before progressing to Phase 2.