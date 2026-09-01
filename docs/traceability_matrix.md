# Requirements Verification Traceability Matrix

## 1. Purpose

This document links the research-system requirements to their principal implementation, automated verification, and experimental evidence.

The matrix covers the current simulation-based surgical-navigation framework and the completed Phase 1 experimental programme.

Verification in this document refers to computational behaviour of the implemented simulation framework only.

It does **not** constitute:

- clinical validation;
- medical-device validation;
- regulatory compliance;
- physical robot validation;
- patient-specific validation;
- evidence of suitability for clinical use.

---

# 2. Current Verification Status

All 20 currently defined functional requirements are implemented and computationally verified for the present Phase 1 system scope.

The final repository-wide local regression baseline following the Phase 1 held-out experiment and statistical-analysis updates is:

```text
612 passed
0 failed
```

The final Phase 1 experiment additionally includes:

```text
30 untouched held-out scenarios
× 10 matched repetitions
× 8 pre-specified strategies
=
2,400 strategy evaluations
```

The final Phase 1 primary result was retained without post-held-out parameter tuning.

---

# 3. Experimental Provenance

The final Phase 1 experiment was frozen before held-out execution.

## Pre-Held-Out Git Commit

```text
250b40d47d115397c8254556096a38525ef92aed
```

## Frozen Scenario Manifest SHA-256

```text
490922fbc9f91743262257ff594a7440651016085207b3e8867b9e1bdce8ddd9
```

Artifact:

```text
results/phase1_scenario_splits/scenario_manifest.json
```

## Frozen Phase 1 Protocol SHA-256

```text
dc6537d3ff75832ccbe48c9b2690c8e966711b31806076cfbc2721488dbf6361
```

Artifact:

```text
results/phase1_protocol/frozen_phase1_protocol.json
```

## Raw Held-Out Evidence SHA-256

```text
3c6f0e8dee177dbfb36019ce0242b53c4a4166391824dbdd953fa8fd8dbcddcb
```

Artifact:

```text
results/phase1_held_out/held_out_raw_records.csv
```

The provenance chain is therefore:

```text
scenario manifest
        |
        v
frozen experimental protocol
        |
        v
pre-held-out source-code commit
        |
        v
untouched held-out execution
        |
        v
raw held-out evidence
        |
        v
pre-specified statistical analysis
```

---

# 4. Functional Requirements

| Requirement | Principal Implementation / Evidence | Verification Evidence | Status |
|---|---|---|---|
| **REQ-01 Reproducible workspace** | deterministic workspace and scenario construction; frozen scenario manifest | workspace tests; scenario-split tests; manifest digest verification | **Verified** |
| **REQ-02 Target and critical structures** | anatomical/workspace representations; start/goal definitions; estimated and hidden true anatomy | workspace, safety, navigation and perception tests | **Verified** |
| **REQ-03 Coordinate frames** | `src/geometry/transforms.py`; camera and instrument geometry | `tests/test_transforms.py`; coordinate-frame checks | **Verified** |
| **REQ-04 Camera observation** | camera, observation and viewpoint modules; final matched observations | camera, observation, viewpoint and fair-scene perception tests | **Verified** |
| **REQ-05 Controlled degradation** | uncertainty, occlusion, heterogeneity and synthetic visual-quality degradation | uncertainty/occlusion tests; illumination benchmark | **Verified** |
| **REQ-06 Perception output** | common perception result interfaces; shared planner-facing estimated anatomy | perception, fair-scene perception and integration tests | **Verified** |
| **REQ-07 Explicit uncertainty** | localisation sigma/covariance; observation uncertainty; uncertainty propagation | uncertainty tests; calibration benchmark; planning-margin tests | **Verified** |
| **REQ-08 Fixed View** | common fixed-view pathway with matched final observation | three-strategy tests; final eight-strategy runner tests | **Verified** |
| **REQ-09 Generic Active Perception** | scene-wide Generic scorer; same-weight and movement-budget-matched variants | viewpoint-scoring tests; fair-scene strategy tests; held-out results | **Verified** |
| **REQ-10 Task-Aware Active Perception** | task relevance; task-aware scorer; alignment and task-weighted information mechanisms | task-aware tests; ablation tests; final held-out mechanism analysis | **Verified** |
| **REQ-11 Candidate viewpoint evaluation** | estimate-centred viewpoint generation; quantitative candidate scoring | viewpoint tests; scorer tests; fair-candidate-context tests | **Verified** |
| **REQ-12 Motion planning** | collision-aware RRT; edge checking; trajectory processing | planner and trajectory tests | **Verified** |
| **REQ-13 Uncertainty-aware planning** | uncertainty-inflated planning geometry and safety margins | perception-planning tests; navigation tests; held-out planning results | **Verified** |
| **REQ-14 Ground-truth evaluation** | simulator truth isolated from deployable strategy selection and reserved for observation generation/evaluation | fair-candidate-context tests; strategy API tests; frozen protocol tests | **Verified** |
| **REQ-15 Safety-margin violations** | safety and clearance subsystem | safety, navigation and benchmark tests | **Verified** |
| **REQ-16 Collision detection** | configuration, edge and path collision checking | safety and planner tests | **Verified** |
| **REQ-17 Quantitative metrics** | benchmark result structures; planning, perception, safety and statistical metrics | primary and secondary statistical tests; generated CSV/JSON evidence | **Verified** |
| **REQ-18 Experimental logging** | CSV/JSON outputs; frozen manifest; protocol; raw held-out evidence and analysis artifacts | output-writing tests; cryptographic hashes; result directories | **Verified** |
| **REQ-19 Fair multi-strategy comparison** | Fixed, Random, Generic, Budget-Matched Generic, Alignment-Only, Information-Only, Full Task-Aware and Oracle | frozen protocol; final runner; 2,400-evaluation held-out experiment | **Verified** |
| **REQ-20 Automated experiment execution** | benchmark drivers and CLI experiment modules with held-out execution guards | benchmark, runner, protocol and full-regression tests | **Verified** |

---

# 5. Information-Isolation Verification

One of the most important Phase 1 corrections was elimination of simulator-truth leakage into deployable viewpoint selection.

The final information flow is:

```text
scenario-independent nominal workspace prior
        |
        v
initial camera pose
        |
        v
hidden simulator anatomy
        |
        +----> noisy observation generator
                        |
                        v
               shared estimated anatomy
                        |
                        v
              estimated target centre
                        |
                        v
       common estimate-centred candidate set
                /                    \
               /                      \
              v                        v
       Generic scoring          Task-Aware scoring
       no task trajectory       + planned trajectory
               \                      /
                \                    /
                 v                  v
                  selected viewpoint
                         |
                         v
                  final observation
                         |
                         v
           uncertainty-aware motion planning
                         |
                         v
             hidden ground-truth evaluation
```

Verified properties include:

- initial camera geometry does not use scenario truth;
- active viewpoint candidates are centred on the noisy estimated target;
- Generic and Task-Aware strategies receive identical estimated anatomy;
- Generic receives no intended task trajectory;
- Task-Aware receives the intended instrument trajectory;
- deployable strategies do not receive hidden anatomical truth;
- only the explicitly privileged Oracle may use true geometry during viewpoint scoring;
- the Oracle must select from the same estimate-centred candidate set.

Principal implementation:

```text
src/simulation/phase1_fair_candidate_context.py
src/perception/fair_scene_strategies.py
src/perception/phase1_baselines.py
src/simulation/phase1_final_runner.py
```

Principal tests:

```text
tests/test_phase1_fair_candidate_context.py
tests/test_fair_scene_strategies.py
tests/test_phase1_baselines.py
tests/test_phase1_final_runner.py
tests/test_phase1_frozen_protocol.py
```

Status:

```text
REQ-06
REQ-09
REQ-10
REQ-11
REQ-14
REQ-19

Verified
```

---

# 6. Development and Movement-Budget Matching

Phase 1 introduced a strengthened scene-wide Generic comparator and explicitly accounted for camera movement.

The corrected development sweep used:

```text
10 development scenarios
× 5 repetitions
```

The selected Full Task-Aware operating point was:

```text
Task-Aware movement weight:
0.200

Mean camera movement:
67.944 mm

Mean predicted sigma:
8.565 mm

Normalised movement:
0.2965

Normalised sigma:
0.4596

Utopia distance:
0.5469
```

The development movement-budget-matched Generic operating point was:

```text
Generic movement weight:
0.072

Mean camera movement:
67.742 mm

Absolute gap:
0.203 mm

Relative gap:
0.30%
```

The pre-defined development requirement was:

```text
movement mismatch <= 5%
```

Result:

```text
0.30%

PASS
```

Artifact:

```text
results/phase1_movement_budget/movement_budget_match_corrected.json
```

Status:

```text
REQ-09
REQ-10
REQ-11
REQ-17
REQ-19
NFR-02
NFR-06

Verified
```

---

# 7. Independent Validation Gate

The development-selected parameters were then evaluated on the frozen validation split.

Validation design:

```text
20 scenarios
× 5 repetitions
=
100 paired validation trials
```

Frozen parameters:

```text
Generic:
0.072

Task-Aware:
0.200
```

Observed validation camera movement:

```text
Generic:
91.187 mm

Task-Aware:
83.761 mm

Absolute gap:
7.427 mm

Relative gap:
8.87%
```

Pre-specified validation tolerance:

```text
10%
```

Result:

```text
8.87%

PASS
```

Diagnostic outcomes not used for weight selection:

```text
Generic localisation error:
16.184 mm

Task-Aware localisation error:
15.842 mm

Generic predicted sigma:
9.521 mm

Task-Aware predicted sigma:
9.595 mm

Task alignment:
0.9862

Different viewpoint rate:
42.0%
```

Artifact:

```text
results/phase1_validation/validation_gate_corrected.json
```

The earlier failed validation attempt is also retained in the repository rather than removed.

Status:

```text
REQ-17
REQ-18
REQ-19
NFR-02
NFR-03
NFR-06

Verified
```

---

# 8. Frozen Held-Out Protocol

Before any final held-out execution, the experiment protocol was frozen.

The final strategy family contains:

1. Fixed View
2. Random Active
3. Generic Active
4. Movement-Budget-Matched Generic
5. Alignment-Only
6. Task-Weighted Information-Only
7. Full Task-Aware
8. Privileged Oracle

Primary comparison:

```text
Full Task-Aware
        versus
Movement-Budget-Matched Generic
```

Primary endpoint:

```text
safe-navigation success rate
```

Inferential unit:

```text
held-out scenario
```

Primary statistical methods:

```text
paired scenario-level sign-flip permutation test

95% scenario-cluster bootstrap confidence interval
```

Frozen superiority rule:

```text
p < 0.05

AND

95% CI for Task-Aware minus Generic effect
lies entirely above zero
```

Post-held-out tuning:

```text
NOT PERMITTED
```

Artifact:

```text
results/phase1_protocol/frozen_phase1_protocol.json
```

Status:

```text
REQ-17
REQ-18
REQ-19
REQ-20
NFR-02
NFR-03
NFR-06
NFR-10

Verified
```

---

# 9. Final Held-Out Experiment

The untouched held-out test contained:

```text
30 scenarios
× 10 stochastic repetitions
× 8 strategies
=
2,400 strategy evaluations
```

Raw artifact:

```text
results/phase1_held_out/held_out_raw_records.csv
```

Raw evidence SHA-256:

```text
3c6f0e8dee177dbfb36019ce0242b53c4a4166391824dbdd953fa8fd8dbcddcb
```

The held-out simulation was executed using the pre-frozen protocol and implementation.

No held-out scenario was used for hyperparameter tuning.

Status:

```text
REQ-17
REQ-18
REQ-19
REQ-20
NFR-02
NFR-03
NFR-06
NFR-07

Verified
```

---

# 10. Final Primary Held-Out Result

The frozen primary comparison was:

```text
Full Task-Aware
        versus
Movement-Budget-Matched Generic
```

Observed safe-navigation success:

```text
Full Task-Aware:
61.33%

Movement-Budget-Matched Generic:
71.33%
```

Primary effect:

```text
Task-Aware minus Generic:

-10.00 percentage points
```

95% scenario-cluster bootstrap confidence interval:

```text
[-17.00, -3.00] percentage points
```

Two-sided paired sign-flip permutation test:

```text
p = 0.012390
```

Frozen superiority criterion:

```text
NOT MET
```

Therefore:

> **The final held-out experiment did not establish superiority of Full Task-Aware Active Perception over the strengthened Generic comparator.**

The result was retained without post-hoc retuning.

Artifact:

```text
results/phase1_held_out/held_out_primary_analysis.json
```

Status:

```text
REQ-17
REQ-18
REQ-19
NFR-03
NFR-06
NFR-10

Verified
```

---

# 11. Held-Out Movement-Budget Limitation

Although movement was closely matched during development and remained within tolerance during validation, the match did not generalise to the held-out distribution.

Observed held-out movement:

```text
Full Task-Aware:
93.544 mm

Budget-Matched Generic:
122.548 mm
```

Absolute difference:

```text
29.004 mm
```

Relative mismatch:

```text
31.01%
```

Previous validation tolerance:

```text
10%
```

Result:

```text
FAIL
```

This is retained as an explicit experimental limitation.

The project therefore does **not** claim:

> Generic outperforms Task-Aware at equal held-out camera movement.

The scientifically defensible conclusion is:

> The frozen Task-Aware method did not establish superiority over the development movement-budget-matched Generic comparator, but the 31.01% held-out movement mismatch prevents clean causal attribution of the performance difference specifically to task awareness.

Status:

```text
REQ-17
REQ-19
NFR-06
NFR-10

Verified with explicit limitation
```

---

# 12. Eight-Strategy Held-Out Evidence

| Strategy | Safe Navigation | Planning Success | Movement | Localisation Error | Predicted Sigma | Collision | Safety Violation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed | 13.67% | 14.67% | 0.000 mm | 42.584 mm | 26.350 mm | 0.33% | 1.00% |
| Random Active | 76.67% | 78.67% | 238.815 mm | 14.158 mm | 8.983 mm | 0.33% | 2.00% |
| Generic Active | 69.33% | 70.67% | 113.774 mm | 16.602 mm | 10.514 mm | 0.00% | 1.33% |
| Budget-Matched Generic | 71.33% | 73.00% | 122.548 mm | 15.858 mm | 9.988 mm | 0.00% | 1.67% |
| Alignment-Only | 74.67% | 76.67% | 131.505 mm | 14.058 mm | 9.137 mm | 0.00% | 2.00% |
| Information-Only | 56.00% | 57.33% | 74.999 mm | 21.562 mm | 13.697 mm | 0.00% | 1.33% |
| Full Task-Aware | 61.33% | 63.67% | 93.544 mm | 19.505 mm | 12.448 mm | 0.00% | 2.33% |
| Oracle | 97.00% | 100.00% | 111.203 mm | 5.407 mm | 3.309 mm | 0.00% | 3.00% |

The Oracle is analysis-only and is not considered a deployable strategy.

Status:

```text
REQ-08
REQ-09
REQ-10
REQ-11
REQ-13
REQ-14
REQ-17
REQ-19

Verified
```

---

# 13. Final Mechanism Ablation

The final held-out mechanism analysis distinguishes:

```text
Alignment-Only
Task-Weighted Information-Only
Full Task-Aware
```

## Alignment-Only

```text
Safe-navigation success:
74.67%

Planning success:
76.67%

Camera movement:
131.505 mm

Localisation error:
14.058 mm

Predicted sigma:
9.137 mm

Mean task alignment:
0.988
```

## Information-Only

```text
Safe-navigation success:
56.00%

Planning success:
57.33%

Camera movement:
74.999 mm

Localisation error:
21.562 mm

Predicted sigma:
13.697 mm
```

## Full Task-Aware

```text
Safe-navigation success:
61.33%

Planning success:
63.67%

Camera movement:
93.544 mm

Localisation error:
19.505 mm

Predicted sigma:
12.448 mm

Mean task alignment:
0.987
```

---

# 14. Alignment Mechanism Evidence

Comparison:

```text
Full Task-Aware
minus
Alignment-Only
```

Observed effects:

```text
Safe-navigation:
-13.33 percentage points

Planning success:
-13.00 percentage points

Holm-adjusted p:
0.0014

Localisation error:
+5.446 mm

Predicted sigma:
+3.312 mm

Camera movement:
-37.962 mm
```

Task-alignment values were nearly identical:

```text
Alignment-Only:
0.988

Full Task-Aware:
0.987
```

This evidence indicates that the degradation in Full Task-Aware performance is not primarily caused by poor alignment.

Instead, the current task-weighted information component is implicated as the main mechanism-level bottleneck.

Status:

```text
REQ-10
REQ-11
REQ-17
REQ-19
NFR-06

Verified with mechanism-level evidence
```

---

# 15. Information Mechanism Evidence

Comparison:

```text
Full Task-Aware
minus
Information-Only
```

Observed effects:

```text
Safe-navigation:
+5.33 percentage points

Planning success:
+6.33 percentage points

Holm-adjusted p:
0.0041

Localisation error:
-2.058 mm

Predicted sigma:
-1.248 mm

Camera movement:
+18.545 mm
```

Adding task alignment to the Information-Only formulation improved:

- localisation accuracy;
- predicted uncertainty;
- planning feasibility;
- safe-navigation success.

The final mechanism interpretation is therefore:

```text
Task-weighted information only
        |
        v
56.00% safe navigation

        + alignment
        |
        v
61.33% safe navigation

Uniform scene information
        + alignment
        |
        v
74.67% safe navigation
```

Status:

```text
REQ-10
REQ-11
REQ-17
NFR-06

Verified
```

---

# 16. Oracle Diagnostic

The privileged Oracle achieved:

```text
Safe-navigation:
97.00%

Planning success:
100.00%

Camera movement:
111.203 mm

Localisation error:
5.407 mm

Predicted sigma:
3.309 mm
```

The Oracle uses the same estimate-centred candidate set as the other active strategies.

Its only privilege is access to simulator truth during candidate scoring.

The gap between:

```text
Full Task-Aware:
61.33%

Oracle:
97.00%
```

indicates that substantial performance remains available within the existing candidate space if viewpoint scoring can better identify informative viewpoints under uncertain anatomy.

This supports the conclusion that viewpoint selection and perception remain major system bottlenecks.

The Oracle is not deployable and is not evidence of achievable clinical performance.

Status:

```text
REQ-11
REQ-14
REQ-17
REQ-19

Verified as analysis-only reference
```

---

# 17. Random Active Diagnostic

Random Active achieved:

```text
76.67% safe-navigation success
```

but required:

```text
238.815 mm
```

mean camera movement.

This greatly exceeds:

```text
Full Task-Aware:
93.544 mm

Budget-Matched Generic:
122.548 mm

Alignment-Only:
131.505 mm
```

The result therefore provides evidence of an exploration-versus-motion-cost trade-off.

It does not establish Random Active as a generally superior strategy under matched motion expenditure.

Status:

```text
REQ-09
REQ-11
REQ-17
REQ-19
NFR-06

Verified as diagnostic baseline
```

---

# 18. Secondary Statistical Evidence

Planning-success inference used paired scenario-level sign-flip permutation tests.

The complete frozen secondary planning-success comparison family was adjusted using the Holm procedure.

| Full Task-Aware minus Comparator | Planning Effect | Holm-Adjusted p |
|---|---:|---:|
| Budget-Matched Generic | −9.33 pp | 0.0476 |
| Same-Weight Generic | −7.00 pp | 0.0533 |
| Fixed | +49.00 pp | 0.0001 |
| Random Active | −15.00 pp | 0.0476 |
| Alignment-Only | −13.00 pp | 0.0014 |
| Information-Only | +6.33 pp | 0.0041 |
| Oracle | −36.33 pp | 0.0001 |

Secondary results do not replace the frozen primary conclusion.

Artifact:

```text
results/phase1_held_out/held_out_secondary_analysis.json
```

Status:

```text
REQ-17
REQ-18
REQ-19
NFR-06

Verified
```

---

# 19. Historical Three-Strategy Evidence

Earlier project stages used a simpler:

```text
Fixed
vs
Generic
vs
Task-Aware
```

architecture.

The earlier robustness benchmark evaluated:

```text
10 scenarios
× 10 repetitions
× 3 strategies
=
300 strategy evaluations
```

Aggregate safe-navigation success was:

```text
Fixed:
43%

Generic:
61%

Task-Aware:
97%
```

Task-Aware planning success was:

```text
100%
```

These results remain valid evidence for the historical implementation.

However, they are **not the final Phase 1 held-out result**.

Later methodological review identified weaknesses including:

- weaker Generic comparison;
- large differences in camera movement;
- simulator-truth influence on candidate geometry;
- scale imbalance in the earlier task-aware scorer;
- absence of a completely untouched held-out test set.

The final Phase 1 methodology corrected these issues.

The large historical Task-Aware advantage did not survive the stricter held-out experiment.

Status:

```text
Historical evidence retained
```

---

# 20. Historical Mechanism Ablation

Earlier experiments compared:

```text
Generic
Alignment-Only
Uncertainty-Only
Full Task-Aware
```

They showed that task alignment dominated explicit uncertainty weighting under the earlier scoring formulation.

Increasing the explicit uncertainty weight through:

```text
0
0.25
1
4
16
```

did not change the selected viewpoint.

This finding motivated the later fair-scoring redesign.

Localisation uncertainty nevertheless remained important downstream through uncertainty-inflated planning geometry.

Status:

```text
REQ-07
REQ-10
REQ-11
REQ-13

Verified as historical mechanism evidence
```

---

# 21. Formal Uncertainty Calibration

The uncertainty-calibration experiment evaluated:

```text
10 scenarios
60 viewpoint conditions
6,000 simulated observations
```

Key results:

```text
Mean normalised squared error

Expected:
3.000

Observed:
3.024
```

```text
95% empirical coverage

Nominal:
95.00%

Observed:
94.98%
```

Additional coverage:

```text
50% nominal:
49.42% observed

90% nominal:
89.95% observed

99% nominal:
98.88% observed
```

The model was classified as:

```text
well_calibrated_under_simulation
```

Artifacts:

```text
results/supplementary_uncertainty_calibration/uncertainty_calibration_samples.csv

results/supplementary_uncertainty_calibration/uncertainty_calibration_summary.json
```

This classification applies only to the implemented stochastic simulation model.

It is not physical sensor calibration.

Status:

```text
REQ-07
REQ-17
NFR-06

Verified quantitatively
```

---

# 22. Synthetic Visual-Quality Degradation

The supplementary synthetic observation-quality model used:

```text
sigma_degraded = sigma_nominal / sqrt(quality)
```

Under the severe condition:

```text
quality = 0.25
```

the historical three-strategy results were:

| Strategy | Planning Success | Safe Navigation |
|---|---:|---:|
| Fixed | 35% | 35% |
| Generic | 55% | 55% |
| Task-Aware | 95% | 90% |

Artifacts:

```text
results/supplementary_illumination/illumination_trials.csv

results/supplementary_illumination/illumination_summary.json
```

This remains explicitly a synthetic stress test rather than a physical photometric or endoscopic-lighting model.

It predates the final Phase 1 fairness redesign.

Status:

```text
REQ-05
NFR-07

Verified as supplementary simulation evidence
```

---

# 23. Planning Efficiency

The supplementary efficiency benchmark evaluated:

```text
10 scenarios
× 5 repetitions
=
50 matched experimental units

=
150 strategy evaluations
```

| Metric | Fixed | Generic | Task-Aware |
|---|---:|---:|---:|
| Planning success | 46% | 66% | 100% |
| Safe navigation | 38% | 60% | 92% |
| Planning time, all attempts | 0.547 s | 0.882 s | 1.239 s |
| Successful-plan path cost | 2.517 | 2.558 | 2.499 |

On the 33 matched trials where both Generic and Task-Aware planning succeeded:

```text
Task-Aware minus Generic path cost:

-0.107

95% CI:

[-0.257, -0.015]
```

Artifacts:

```text
results/supplementary_efficiency/three_strategy_efficiency_trials.csv

results/supplementary_efficiency/three_strategy_efficiency_summary.json
```

These results remain supplementary historical evidence.

Status:

```text
REQ-17
NFR-09

Verified
```

---

# 24. Post-Held-Out Analysis-Code Correction

After the 2,400-strategy held-out simulation had completed, the statistical validator initially rejected records where failed planning attempts contained:

```text
path_cost = inf
```

The frozen experimental protocol had already specified that path-cost comparisons should include only pairs in which both strategies successfully generated a path.

Therefore:

```text
successful planning
        |
        +----> path cost must be finite

failed planning
        |
        +----> +inf path cost is permitted
```

The validator was corrected to implement this already-frozen analysis rule.

Important integrity properties:

- the raw held-out CSV was not modified;
- the held-out scenarios were not changed;
- the random seeds were not changed;
- movement weights were not changed;
- strategy definitions were not changed;
- the primary endpoint was not changed;
- the statistical test was not changed;
- the superiority criterion was not changed;
- the 2,400-strategy held-out simulation was not rerun because of the validator correction.

Regression tests were added for both:

```text
failed plan + infinite path cost
```

and:

```text
successful plan + non-finite path cost
```

Status:

```text
REQ-17
REQ-18
NFR-03
NFR-04
NFR-10

Verified
```

---

# 25. Non-Functional Requirements

| Requirement | Evidence | Status |
|---|---|---|
| **NFR-01 Modularity** | separated geometry, robotics, perception, simulation and ROS 2 components; separate scorers and experiment drivers | **Verified** |
| **NFR-02 Reproducibility** | Python 3.11 environment; deterministic seeds; frozen scenario manifest; reproducible experiment drivers | **Verified** |
| **NFR-03 Traceability** | requirements; tests; Git commit; scenario hash; protocol hash; raw-evidence hash; machine-readable results | **Verified** |
| **NFR-04 Testability** | final repository regression: **612 passing tests** | **Verified** |
| **NFR-05 Numerical robustness** | tolerance-based geometry, transformation, uncertainty and planning verification | **Verified** |
| **NFR-06 Quantitative evaluation** | paired trials; bootstrap CIs; permutation testing; Holm correction; calibration; ablation; robustness analysis | **Verified** |
| **NFR-07 Robustness evaluation** | development/validation/held-out scenario separation; uncertainty stress; synthetic visual degradation | **Verified** |
| **NFR-08 Extensibility** | separate scorers, strategies, controllers, observation models, planners and experiment drivers | **Verified** |
| **NFR-09 Computational observability** | scores, uncertainty, movement, planning time, iterations, path cost, clearance and safety outputs retained | **Verified** |
| **NFR-10 Interpretation safety** | README, protocol and requirements explicitly restrict claims to simulation and retain negative/limiting evidence | **Documentation Verified** |

---

# 26. Final Phase 1 Verification Summary

## Functional Requirements

```text
Verified:
REQ-01 through REQ-20

Pending:
None within current Phase 1 scope
```

## Non-Functional Requirements

```text
Verified:
NFR-01 through NFR-09

Documentation Verified:
NFR-10

Pending:
None within current Phase 1 scope
```

## Final Regression

```text
612 passed
0 failed
```

## Final Held-Out Experiment

```text
30 scenarios
× 10 repetitions
× 8 strategies
=
2,400 evaluations
```

## Primary Result

```text
Full Task-Aware:
61.33%

Movement-Budget-Matched Generic:
71.33%

Task-Aware effect:
-10.00 percentage points

95% CI:
[-17.00, -3.00]

p:
0.012390

Task-Aware superiority:
NOT ESTABLISHED
```

## Principal Experimental Limitation

```text
Held-out movement-budget mismatch:
31.01%

Previous validation tolerance:
10%
```

## Principal Mechanism Result

```text
Alignment-Only:
74.67% safe navigation

Full Task-Aware:
61.33%

Information-Only:
56.00%
```

The held-out evidence therefore implicates the current task-weighted information formulation rather than task alignment itself as the main Task-Aware mechanism bottleneck.

---

# 27. Scientific Interpretation Boundary

The verification evidence in this matrix demonstrates implementation correctness and computational behaviour relative to the current simulation requirements.

It does not demonstrate:

- clinical safety;
- clinical effectiveness;
- patient-specific performance;
- anatomical generalisation to real patients;
- physical sensor accuracy;
- physical camera calibration;
- physical robot performance;
- regulatory compliance;
- medical-device certification;
- suitability for clinical use.

The tested scenarios are engineered simulated variations rather than patients or independent patient anatomies.

The final Phase 1 result should therefore be interpreted as:

> **A controlled simulation investigation showing that the current Full Task-Aware formulation did not outperform a strengthened Generic comparator on untouched held-out scenarios, while mechanism ablation identified task-weighted information as a likely performance bottleneck and retained uncertainty as an important downstream planning variable.**

It should not be interpreted as evidence that task-aware active perception is clinically ineffective or that Generic Active Perception is clinically superior.

---

# 28. Phase 1 Research Position

The final Phase 1 evidence changes the research position from:

```text
Task-Aware perception clearly improves navigation.
```

to the substantially more rigorous conclusion:

> **Task information can materially change active-perception decisions, but its representation matters. Under the current simulated formulation, task alignment showed stronger held-out behaviour than task-weighted information. The combined Full Task-Aware strategy did not demonstrate superiority over the strengthened Generic comparator, and held-out movement-budget mismatch limits direct causal attribution.**

At the same time:

> **Localisation uncertainty remains important because it directly changes uncertainty-inflated safety geometry used by the motion planner.**

The final evidence therefore motivates redesign and deeper modelling in later phases rather than post-hoc tuning against the held-out Phase 1 dataset.

---

# 29. Future Requirements

Requirements associated with later project phases are intentionally outside the completed Phase 1 verification scope.

Future work includes:

- advanced robot kinematics;
- reachable and executable viewpoint planning;
- Jacobian and manipulability analysis;
- genuine image-based surgical perception;
- machine-learning perception;
- 3-D localisation;
- image-to-robot registration;
- dynamic anatomy;
- temporal state estimation;
- model-mismatch experiments;
- medical-device-style safety supervision;
- preliminary benchtop validation;
- deeper ROS 2 / Gazebo deployment;
- closed-loop trajectory control;
- end-to-end image-guided intervention experiments.

These future capabilities must receive separate requirements, verification evidence and traceability as they are implemented.

They are not claimed as completed by this Phase 1 matrix.