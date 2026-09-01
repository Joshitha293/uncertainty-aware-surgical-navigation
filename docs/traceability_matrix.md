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

# 29. Phase 2-5 Requirements Extension

The original Phase 1 requirements, experimental provenance, held-out evidence, statistical results, mechanism analysis, and interpretation boundaries documented above remain frozen.

Phases 2-5 extend the engineering framework with additional functionality and verification requirements defined in:

`docs/requirements.md`

These extensions do not modify or reinterpret the Phase 1 held-out result.

---

# 30. Phase 2 Traceability - Robot Kinematics and Advanced Motion Planning

| Requirement | Implementation Evidence | Verification / Experimental Evidence | Status |
| --- | --- | --- | --- |
| REQ-21 Advanced robot kinematics | `src/robotics/advanced_kinematics.py` | `tests/test_phase2_advanced_kinematics.py` | **Verified** |
| REQ-22 RCM-constrained motion | `src/robotics/advanced_kinematics.py` | `tests/test_phase2_advanced_kinematics.py`; trajectory benchmark evidence | **Verified** |
| REQ-23 Robot workspace and reachability | `src/robotics/workspace_analysis.py`; `src/perception/robot_reachability.py` | `tests/test_phase2_robot_reachability.py`; Phase 2 benchmark | **Verified** |
| REQ-24 Advanced motion planning | `src/robotics/advanced_planning.py` | `tests/test_phase2_advanced_planning.py`; Phase 2 benchmark | **Verified** |
| REQ-25 Collision-aware planning | `src/robotics/advanced_planning.py`; existing collision/safety modules | Phase 2 planning tests and benchmark | **Verified** |
| REQ-26 Trajectory processing and execution metrics | `src/robotics/trajectory_execution.py` | `tests/test_phase2_trajectory_execution.py` | **Verified** |
| REQ-27 Robot-aware viewpoint feasibility | `src/perception/robot_aware_selection.py`; `src/perception/robot_reachability.py` | `tests/test_phase2_robot_aware_selection.py`; `tests/test_phase2_robot_reachability.py` | **Verified** |

## Phase 2 Experimental Evidence

Primary artifact:

`results/phase2/phase2_planning_benchmark.json`

Representative benchmark evidence:

Workspace samples:
441

Reachable demonstration viewpoints:
2 / 3

Planner results:

RRT:

success = 3 / 3
raw path length = 2.546479
smoothed path length = 2.067215
planning time = 0.969383 s
trajectory duration = 2.166667 s
maximum RCM deviation = 1.51e-17

RRT*:

success = 3 / 3
raw path length = 1.768143
smoothed path length = 1.764382
planning time = 5.199598 s
trajectory duration = 1.913333 s
maximum RCM deviation = 1.18e-17

A*:

success = 1 / 1
raw path length = 3.168928
smoothed path length = 2.334149
planning time = 2.411415 s
trajectory duration = 2.700000 s
maximum RCM deviation = 1.34e-17

The implemented planners expose different path-quality, runtime, and execution trade-offs.

The evidence does not justify a universal planner-dominance claim.

**Phase 2 status: Verified in simulation**

---

# 31. Phase 3 Traceability - Image Processing and Stereo 3-D Perception

| Requirement | Implementation Evidence | Verification / Experimental Evidence | Status |
| --- | --- | --- | --- |
| REQ-28 Explicit camera geometry | `src/perception/image_geometry.py`; `src/perception/camera.py` | `tests/test_phase3_image_geometry.py` | **Verified** |
| REQ-29 Image processing | `src/perception/image_processing.py`; `src/perception/image_driven_perception.py` | `tests/test_phase3_image_processing.py` | **Verified** |
| REQ-30 Stereo geometry | `src/perception/stereo_geometry.py` | `tests/test_phase3_stereo_geometry.py` | **Verified** |
| REQ-31 3-D triangulation | `src/perception/stereo_geometry.py` | stereo geometry tests and benchmark | **Verified** |
| REQ-32 Stereo uncertainty propagation | `src/perception/stereo_uncertainty.py` | `tests/test_phase3_image_uncertainty.py`; Phase 3 benchmark | **Verified** |
| REQ-33 Quantitative stereo validation | `src/simulation/phase3_stereo_uncertainty_benchmark.py` | `results/phase3/phase3_stereo_uncertainty_benchmark.json` | **Verified** |

## Phase 3 Camera Convention

The implemented camera convention uses:

CameraPose rotation:
camera frame -> world frame

+x:
image right

+y:
camera up

+z:
optical forward

Image vertical coordinates therefore use the implemented downward-image-axis projection convention.

## Phase 3 Experimental Evidence

Primary artifact:

`results/phase3/phase3_stereo_uncertainty_benchmark.json`

Image-driven localisation demonstration:

localisation error:
0.707 mm

predicted principal sigma:
8.854 mm

runtime ground-truth error entries:
0

Monte Carlo stereo conditions:

baseline = 10 mm, pixel sigma = 0.25 px:
empirical RMS = 8.778 mm
predicted RMS = 8.883 mm
ratio = 0.988
95% coverage = 95.6%

baseline = 10 mm, pixel sigma = 0.50 px:
empirical RMS = 18.251 mm
predicted RMS = 17.907 mm
ratio = 1.019
95% coverage = 92.8%

baseline = 10 mm, pixel sigma = 1.00 px:
empirical RMS = 35.473 mm
predicted RMS = 38.364 mm
ratio = 0.925
95% coverage = 92.2%

baseline = 20 mm, pixel sigma = 0.25 px:
empirical RMS = 4.559 mm
predicted RMS = 4.427 mm
ratio = 1.030
95% coverage = 94.8%

baseline = 20 mm, pixel sigma = 0.50 px:
empirical RMS = 9.193 mm
predicted RMS = 8.949 mm
ratio = 1.027
95% coverage = 96.8%

baseline = 20 mm, pixel sigma = 1.00 px:
empirical RMS = 18.434 mm
predicted RMS = 18.071 mm
ratio = 1.020
95% coverage = 93.0%

baseline = 40 mm, pixel sigma = 0.25 px:
empirical RMS = 2.305 mm
predicted RMS = 2.229 mm
ratio = 1.034
95% coverage = 95.4%

baseline = 40 mm, pixel sigma = 0.50 px:
empirical RMS = 4.468 mm
predicted RMS = 4.462 mm
ratio = 1.001
95% coverage = 96.2%

baseline = 40 mm, pixel sigma = 1.00 px:
empirical RMS = 8.693 mm
predicted RMS = 8.951 mm
ratio = 0.971
95% coverage = 95.2%

Observed trends:

uncertainty increased with pixel noise:
TRUE

uncertainty decreased with stereo baseline:
TRUE

Predicted and empirical uncertainty were approximately consistent across the tested synthetic calibrated-camera experiment.

This is not evidence of physical stereo-camera calibration.

**Phase 3 status: Verified in simulation**

---

# 32. Phase 4 Traceability - Learned Perception and Uncertainty

| Requirement | Implementation Evidence | Verification / Experimental Evidence | Status |
| --- | --- | --- | --- |
| REQ-34 Leakage-resistant dataset splitting | `src/perception/ml_dataset.py` | `tests/test_phase4_ml_dataset.py` | **Verified** |
| REQ-35 Trainable segmentation model | `src/perception/ml_segmentation.py`; `src/simulation/phase4_train_segmentation.py` | `tests/test_phase4_ml_segmentation.py`; training artifact | **Verified** |
| REQ-36 Validation-only model selection | Phase 4 training pipeline | `results/phase4/phase4_segmentation_training.json` | **Verified** |
| REQ-37 Held-out segmentation evaluation | segmentation evaluation pipeline | segmentation tests and training artifact | **Verified** |
| REQ-38 Classical vision comparator | HSV segmentation evaluation | Phase 4 training/evaluation artifact | **Verified** |
| REQ-39 Predictive uncertainty | `src/perception/ml_uncertainty.py` | `tests/test_phase4_ml_uncertainty.py` | **Verified** |
| REQ-40 Calibration and distribution-shift evaluation | `src/simulation/phase4_uncertainty_robustness_benchmark.py` | `results/phase4/phase4_uncertainty_robustness.json` | **Verified with limitations** |
| REQ-41 Learned stereo integration | `src/perception/ml_stereo_perception.py` | `tests/test_phase4_ml_stereo_perception.py` | **Verified** |
| REQ-42 Learned 3-D covariance propagation | `src/perception/ml_stereo_perception.py` | learned stereo tests and final benchmark | **Verified** |
| REQ-43 Runtime ground-truth isolation | learned stereo `PerceptionResult` interface | learned stereo integration tests | **Verified** |

## Phase 4 Dataset and Model Selection

The synthetic segmentation dataset uses scenario-level splitting.

Frames generated from the same latent scenario are not intentionally divided across training, validation, and test partitions.

The selected Tiny U-Net uses:

base channels:
8

loss:
binary cross entropy + soft Dice

optimizer:
Adam

learning rate:
1e-3

weight decay:
1e-5

Model selection used validation performance only.

Selected epoch:

3

## Phase 4 Held-Out Segmentation Evidence

Artifacts:

`results/phase4/phase4_segmentation_training.json`

`results/phase4/tiny_unet_best.pt`

Held-out Tiny U-Net:

Dice:
0.8807

IoU:
0.8084

Detection:
100%

Centroid error:
1.017 px

Classical HSV comparator:

Dice:
0.8466

IoU:
0.7450

Detection:
100%

Centroid error:
0.982 px

Observed differences:

Tiny U-Net Dice improvement:
+0.0341

Tiny U-Net IoU improvement:
+0.0634

Classical centroid advantage:
approximately 0.035 px

The learned model improved segmentation overlap in this experiment but did not improve centroid localisation.

No universal learned-perception superiority claim is made.

---

## Phase 4 Predictive-Uncertainty Evidence

Artifact:

`results/phase4/phase4_uncertainty_robustness.json`

Representative conditions:

CLEAN

Dice:
0.9303

IoU:
0.8759

Detection:
100%

Centroid error:
0.885 px

Brier score:
0.16345

Foreground probability ECE:
approximately 0.397

Foreground entropy:
0.67283

SEVERE

Dice:
0.8073

IoU:
0.6888

Detection:
100%

Centroid error:
1.314 px

Brier score:
0.17606

Foreground entropy:
0.67851

COLOUR-SHIFT OOD

Dice:
0.0769

IoU:
0.0478

Detection:
100%

Centroid error:
14.875 px

Brier score:
0.19827

Foreground entropy:
0.68309

The robustness experiment identified important limitations:

- segmentation quality degraded under severe perturbation;
- colour-shift OOD caused catastrophic segmentation-overlap degradation;
- Brier score worsened under degradation;
- predictive entropy increased only weakly;
- perturbation-based probability variance decreased under severe/OOD conditions;
- perturbation variance was therefore not a reliable OOD detector;
- foreground probability calibration remained poor.

The perturbation ensemble must not be described as a Bayesian posterior.

---

## Phase 4 Final Learned-Stereo Evidence

Artifact:

`results/phase4/phase4_final_integration_benchmark.json`

Results:

CLEAN

Success:
100%

Mean localisation error:
31.794 mm

Maximum localisation error:
90.027 mm

Mean principal sigma:
15.261 mm

Nominal 95% covariance coverage:
66.7%

Mean 2-sigma planner margin:
45.522 mm

MODERATE

Success:
100%

Mean localisation error:
14.101 mm

Maximum localisation error:
40.604 mm

Mean principal sigma:
11.895 mm

Nominal 95% covariance coverage:
66.7%

Mean 2-sigma planner margin:
38.790 mm

COLOUR-SHIFT OOD

Success:
100%

Mean localisation error:
112.345 mm

Maximum localisation error:
227.164 mm

Mean principal sigma:
34.866 mm

Nominal 95% covariance coverage:
0.0%

Mean 2-sigma planner margin:
84.732 mm

The key scientific result is:

**Strong held-out 2-D segmentation performance did not guarantee reliable downstream stereo 3-D localisation.**

The unexpectedly lower mean error under moderate degradation than under the clean condition is not interpreted as evidence that degradation improves localisation.

Only six targets were evaluated per condition.

The result instead demonstrates sensitivity of triangulated depth to left/right centroid and disparity behaviour.

**Phase 4 status: Software integration verified; important localisation and uncertainty limitations identified**

---

# 33. Phase 5 Traceability - Registration, Tracking and State Estimation

| Requirement | Implementation Evidence | Verification / Experimental Evidence | Status |
| --- | --- | --- | --- |
| REQ-44 Corresponding-point rigid registration | `src/geometry/registration.py` | `tests/test_phase5_registration.py` | **Verified** |
| REQ-45 Registration error metrics | `src/geometry/registration.py` | registration tests and benchmark | **Verified** |
| REQ-46 Robust registration | `src/geometry/robust_registration.py` | `tests/test_phase5_robust_registration.py`; registration benchmark | **Verified** |
| REQ-47 Unknown-correspondence point-cloud registration | trimmed ICP in `src/geometry/robust_registration.py` | robust-registration tests and benchmark | **Verified** |
| REQ-48 Temporal state estimation | `src/perception/state_estimation.py` | `tests/test_phase5_state_estimation.py` | **Verified** |
| REQ-49 Measurement covariance fusion | `src/perception/state_estimation.py` | state-estimation tests | **Verified** |
| REQ-50 Prediction during dropout | `src/perception/state_estimation.py` | state-estimation tests; tracking benchmark | **Verified** |
| REQ-51 Innovation-based outlier rejection | `src/perception/state_estimation.py` | state-estimation tests; tracking benchmarks | **Verified** |
| REQ-52 Numerically stable covariance update | Joseph-form update in `src/perception/state_estimation.py` | covariance symmetry/PSD tests | **Verified** |
| REQ-53 State-uncertainty calibration evaluation | Phase 5 tracking/calibration benchmarks | calibration evidence | **Verified** |
| REQ-54 Validation/held-out tuning separation | `src/simulation/phase5_tracking_calibration_benchmark.py` | disjoint validation and held-out seed sets | **Verified** |
| REQ-55 Registration uncertainty propagation | `src/perception/tracked_navigation.py` | `tests/test_phase5_tracked_navigation.py` | **Verified** |
| REQ-56 Registered temporal navigation integration | `src/perception/tracked_navigation.py` | tracked-navigation tests and final benchmark | **Verified** |
| REQ-57 Integrated planner uncertainty propagation | tracked navigation + existing planner interface | tracked-navigation tests and final benchmark | **Verified** |
| REQ-58 Integrated ground-truth isolation | runtime registered/tracked interfaces | integration tests and benchmark design | **Verified** |

---

# 34. Phase 5 Registration Evidence

Primary artifact:

`results/phase5/phase5_registration_benchmark.json`

## Known Correspondences with Outliers

Across 20 synthetic trials:

Naive mean translation error:
6.860 mm

RANSAC mean translation error:
0.295 mm

Naive mean rotation error:
7.064 deg

RANSAC mean rotation error:
0.440 deg

Naive mean RMS TRE:
9.605 mm

RANSAC mean RMS TRE:
0.485 mm

RANSAC consensus classification:

Mean inlier precision:
100.0%

Mean inlier recall:
100.0%

RANSAC lower TRE than naive:
100.0% of trials

These results demonstrate robust behaviour against the deliberately injected correspondence outliers in this benchmark.

They do not establish robustness for arbitrary physical registration conditions.

---

## Unknown-Correspondence ICP Evidence

Across 15 controlled synthetic trials:

Mean translation error:
0.060 mm

Mean rotation error:
0.057 deg

Mean RMS TRE:
0.084 mm

Mean ICP residual RMS:
0.376 mm

Convergence:
100.0%

ICP was evaluated from sufficiently nearby starting alignments.

The evidence does not establish global convergence from arbitrary initialisation.

---

# 35. Phase 5 Temporal State-Estimation Evidence

Initial temporal benchmark artifact:

`results/phase5/phase5_tracking_benchmark.json`

Experimental configuration:

Trials:
20

Steps per trial:
120

Dropout probability:
10%

Outlier probability:
6%

Initial results:

Raw position RMSE:
10.238 mm

Raw non-outlier position RMSE:
4.916 mm

Tracked position RMSE:
2.530 mm

Tracked measurement-step RMSE:
2.507 mm

Tracked dropout RMSE:
2.671 mm

Velocity:

Raw finite-difference velocity RMSE:
278.665 mm/s

Tracked velocity RMSE:
8.508 mm/s

Initial uncertainty behaviour:

Nominal 95% position covariance coverage:
80.5%

Outlier rejection precision:
75.4%

Outlier rejection recall:
100.0%

The tracker reduced position and velocity error, but its initial covariance was overconfident.

The nominal 95% region contained truth only 80.5% of the time.

This failure triggered a separate validation-based process-noise calibration experiment rather than post-hoc adjustment using the held-out set.

---

## Validation-Based State-Estimator Calibration

Artifact:

`results/phase5/phase5_tracking_calibration_benchmark.json`

Validation candidates:

sigma = 0.008 m/s^2:
coverage = 67.0%
position RMSE = 2.594 mm
velocity RMSE = 8.460 mm/s

sigma = 0.012 m/s^2:
coverage = 80.1%
position RMSE = 2.447 mm
velocity RMSE = 8.369 mm/s

sigma = 0.018 m/s^2:
coverage = 90.5%
position RMSE = 2.328 mm
velocity RMSE = 8.170 mm/s

sigma = 0.025 m/s^2:
coverage = 94.7%
position RMSE = 2.252 mm
velocity RMSE = 7.990 mm/s

sigma = 0.035 m/s^2:
coverage = 97.1%
position RMSE = 2.232 mm
velocity RMSE = 7.949 mm/s

sigma = 0.050 m/s^2:
coverage = 98.0%
position RMSE = 2.290 mm
velocity RMSE = 8.284 mm/s

Validation selection rule:

Primary:
minimise absolute deviation from nominal 95% covariance coverage

Secondary:
tracked position RMSE

Selected value:

acceleration_sigma:
0.025 m/s^2

The selected parameter was frozen before evaluation on a disjoint held-out seed set.

Held-out results:

Raw position RMSE:
10.356 mm

Raw non-outlier position RMSE:
4.994 mm

Tracked position RMSE:
2.330 mm

Tracked dropout RMSE:
2.687 mm

Tracked velocity RMSE:
8.029 mm/s

Nominal 95% covariance coverage:
95.3%

Outlier rejection precision:
90.3%

Outlier rejection recall:
100.0%

The 95.3% result demonstrates approximate calibration only for the implemented synthetic motion and noise model.

It is not evidence of clinical uncertainty calibration.

---

# 36. Phase 5 Final Registration-Tracking-Navigation Integration

Artifact:

`results/phase5/phase5_final_integration_benchmark.json`

Experimental configuration:

Trials:
20

Steps per trial:
120

Frozen acceleration sigma:
0.025 m/s^2

The benchmark integrates:

synthetic uncertain 3-D observation
->
estimated RANSAC registration
->
measurement + registration covariance propagation
->
Kalman prediction/update
->
tracked position + velocity + covariance
->
EstimatedStructure
->
uncertainty-aware planner geometry

Ground-truth transforms and states are retained for benchmark evaluation only.

They are not used as runtime corrections.

## Registration

Mean translation error:
1.005 mm

Mean rotation error:
0.340 deg

## Position Estimation

Raw registered RMSE:
10.065 mm

Raw clean registered RMSE:
5.016 mm

Tracked RMSE:
2.417 mm

Tracked dropout RMSE:
2.703 mm

Tracker lower RMSE:
100.0% of trials

## Uncertainty

Raw all-measurement nominal 95% coverage:
93.6%

Raw clean-measurement nominal 95% coverage:
99.1%

Tracked nominal 95% coverage:
99.3%

The tracked 99.3% coverage is conservative relative to the nominal 95% confidence region.

It must not be described as perfect calibration.

## Outlier Gating

Precision:
97.7%

Recall:
100.0%

## Planner Uncertainty Propagation

Mean raw safety margin:
14.548 mm

Mean tracked safety margin:
9.182 mm

Mean tracked dropout safety margin:
9.455 mm

The lower mean tracked margin is consistent with temporal fusion reducing state uncertainty.

The increase in the dropout margin is consistent with covariance growth during prediction without a new observation.

These planner margins are engineering consequences of the implemented uncertainty model.

They are not clinically validated safety margins.

---

# 37. Phase 2-5 Verification Baseline

The previously frozen Phase 1 repository baseline was:

449 passed
0 failed

During development of Phases 2-5, the completed regression suite reached:

822 passed

A later full-suite invocation was manually interrupted after:

144 passed

That interrupted run is not considered release verification.

The final consolidated Phase 2-5 release regression subsequently completed successfully with 822 tests passing.

Until that release test finishes successfully:

Final Phase 2-5 release regression:
822 passed
0 failed

---

# 38. Phase 2-5 Environment Evidence

The verified development environment includes:

Python:
3.11.15

NumPy:
2.4.6

OpenCV:
4.10.0

PyTorch:
2.13.0

PyTorch CUDA available:
False

PyBullet:
imported successfully

PyTorch uses the CPU conda-forge build.

The environment specification is recorded in:

`environment.yml`

A Conda dry-run successfully resolved the declared Windows environment.

The project deliberately uses:

`opencv-python-headless==4.10.0.84`

for Phase 3-4 image processing because graphical OpenCV windows are not required.

---

# 39. Phase 2-5 Scientific Interpretation Boundary

The Phase 2-5 evidence demonstrates implemented engineering behaviour under controlled simulation.

It does not establish:

- patient-specific performance;
- anatomical generalisation to real patients;
- clinical image-segmentation performance;
- clinical OOD detection;
- physical camera calibration;
- physical stereo reconstruction accuracy;
- physical image-to-patient registration accuracy;
- deformable registration performance;
- physical robot accuracy;
- physical RCM accuracy;
- medical-device safety;
- surgical safety;
- clinical efficacy;
- regulatory compliance;
- suitability for clinical use.

Phase 4 learned perception uses synthetic generated imagery.

Phase 5 registration and tracking use synthetic geometry and synthetic uncertain observations.

The final Phase 5 temporal benchmark uses a software-compatible uncertain stereo-observation representation.

It does not rerun the Tiny U-Net image inference pipeline inside every temporal trial.

The actual trained learned-image-to-stereo pathway remains separately evaluated by the Phase 4 benchmark.

---

# 40. Current Research Position After Phase 5

The current research system can now demonstrate the following simulated computational chain:

camera / image observation
->
classical or learned perception
->
stereo 3-D localisation
->
3-D covariance
->
rigid registration
->
registered measurement covariance
->
temporal state estimation
->
tracked positional covariance
->
EstimatedStructure
->
uncertainty-aware protected geometry
->
motion planning

The evidence also identifies important unresolved problems:

Phase 1:
Full Task-Aware superiority was not established.

Phase 3:
stereo depth uncertainty remains sensitive to image measurement noise and baseline.

Phase 4:
strong 2-D segmentation did not produce reliable 3-D localisation under all conditions.

Phase 4:
colour-shift OOD produced severe learned-perception failure.

Phase 4:
perturbation variance was not a reliable OOD detector.

Phase 5:
the initial temporal covariance model was overconfident before validation-based tuning.

Phase 5:
the final integrated covariance became conservative at 99.3% coverage.

These limitations motivate subsequent phases rather than being hidden through post-hoc claims.

---

# 41. Remaining Roadmap

## Phase 6 - Robust Planning Under Uncertainty

Planned work includes:

- risk-aware planning beyond scalar uncertainty inflation;
- robustness to covariance miscalibration;
- non-Gaussian and heavy-tailed localisation errors;
- correlated errors;
- temporal drift;
- chance-constrained or risk-bounded planning concepts;
- replanning under evolving uncertainty;
- safety-efficiency-conservatism comparison.

## Phase 7 - Safety and Autonomous Task Execution

Planned work includes:

- supervisory safety logic;
- autonomous state-machine execution;
- stale-data detection;
- uncertainty thresholds;
- clearance monitoring;
- joint-limit monitoring;
- timeout and recovery behaviour;
- fault injection;
- hazard-oriented verification.

## Phase 8 - ROS 2 and Gazebo Surgical Robotics System

Planned work includes:

- ROS 2 nodes;
- TF2;
- URDF / Xacro;
- Gazebo;
- RViz;
- ros2_control;
- perception nodes;
- registration and state-estimation nodes;
- active-perception node;
- motion-planning node;
- safety-supervisor node;
- trajectory execution;
- rosbag2;
- QoS and lifecycle considerations.

## Phase 9 - Control and Real-Time Trajectory Execution

Planned work includes:

- trajectory tracking;
- feedback control;
- position and velocity control;
- PID evaluation where appropriate;
- execution-error measurement;
- settling behaviour;
- loop-rate measurement;
- perception latency;
- estimation latency;
- planning latency;
- control latency.

## Phase 10 - Final End-to-End Experiment and Verification

The final target system is:

image
->
perception
->
stereo localisation + covariance
->
registration
->
temporal state estimate
->
task-aware viewpoint selection
->
robot feasibility
->
robust uncertainty-aware planning
->
safety supervisor
->
trajectory execution
->
feedback control
->
ROS 2 / Gazebo
->
quantitative end-to-end verification

Later-phase capabilities are not claimed as complete by the current Phase 1-5 evidence.
