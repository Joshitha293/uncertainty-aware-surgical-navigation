# Final Experimental Evidence

This report consolidates the final end-to-end experimental evidence for task-aware active perception coupled to uncertainty-aware surgical motion planning.

## Evidence validation

- E4 stochastic evidence validated.
- E5 multi-scenario robustness evidence validated.
- E6 mechanism-ablation evidence validated.
- E7 uncertainty-stress evidence validated.

## Final quantitative summary

| Stage | Condition | Localisation error (mm) | Predicted σ (mm) | Camera movement (mm) | Planning success | Safe-navigation success |
|---|---|---:|---:|---:|---:|---:|
| E5 multi-scenario | fixed | 25.964 | 17.777 | 0.000 | 45.0% | 43.0% |
| E5 multi-scenario | generic_active | 16.934 | 12.267 | 15.628 | 63.0% | 61.0% |
| E5 multi-scenario | task_aware_active | 4.682 | 3.003 | 109.985 | 100.0% | 97.0% |
| E6 mechanism | generic_baseline | 20.785 | 12.267 | 15.628 | 63.0% | 63.0% |
| E6 mechanism | alignment_only | 4.755 | 3.015 | 103.591 | 100.0% | 100.0% |
| E6 mechanism | uncertainty_only | 20.785 | 12.267 | 15.628 | 63.0% | 63.0% |
| E6 mechanism | full_task_aware | 4.777 | 3.003 | 109.985 | 100.0% | 100.0% |
| E7 uncertainty stress | generic_baseline | 32.746 | 15.642 | 15.628 | 68.8% | 61.3% |
| E7 uncertainty stress | high_uncertainty_only | 32.746 | 15.642 | 15.628 | 68.8% | 61.3% |
| E7 uncertainty stress | alignment_only | 7.358 | 4.329 | 107.385 | 95.0% | 92.5% |
| E7 uncertainty stress | full_task_aware | 7.376 | 4.320 | 109.985 | 95.0% | 92.5% |

## Supported claims

### C1

Task-aware active perception improved end-to-end navigation feasibility across the tested multi-scenario simulation benchmark.

Evidence: E5 safe-navigation success was 97.0% for task-aware perception versus 43.0% for fixed view and 61.0% for generic active perception; paired scenario-clustered 95% confidence intervals for both improvements excluded zero.

### C2

Task alignment was the dominant viewpoint-selection mechanism responsible for the observed task-aware benefit.

Evidence: E6 alignment-only achieved 100.0% safe-navigation success, whereas uncertainty-only achieved 63.0% and the generic baseline achieved 63.0%.

### C3

The explicit uncertainty term did not provide independent viewpoint-selection information under the tested scoring and observation models.

Evidence: E7 produced zero uncertainty-only viewpoint changes for all tested uncertainty weights up to 16 across every scenario/profile condition.

### C4

Uncertainty remains operationally important downstream because perceived localisation uncertainty is propagated into uncertainty-inflated planning geometry.

Evidence: The final architecture separates viewpoint-selection mechanism evidence from the uncertainty-aware motion-planning stage and hidden-ground-truth safety evaluation.

## Claims not supported by the evidence

- The experiments do not support claiming that the explicit uncertainty term independently improves viewpoint selection.
- The experiments do not support claiming clinical safety, clinical effectiveness or patient-level generalisation.
- The experiments do not support claiming that task-aware perception reduces collision probability for trajectories that are already successfully planned.

## Limitations

- All reported experiments are simulation-based; no physical surgical robot or clinical system was evaluated.
- The robustness benchmark contains ten engineered surgical-scene variations rather than ten independent patient anatomies.
- The results therefore demonstrate robustness to the tested scene perturbations, not clinical generalisation.
- The experiments do not establish that task-aware perception reduces collision probability once a valid trajectory already exists.
- The explicit uncertainty term was redundant for viewpoint selection under the tested model; the principal selection benefit was attributable to task alignment.
- Camera repositioning was substantially greater for task-aware strategies and should be treated as a real operational trade-off.
- The simulated observation model does not capture the full range of real surgical effects such as tissue deformation, calibration drift, specularities and dynamic anatomy.

## Final interpretation

The experimental evidence supports a task-aware active-perception architecture in which task alignment improves viewpoint selection, the resulting uncertain anatomical estimates are propagated into uncertainty-aware planning geometry, and final trajectories are evaluated against hidden ground truth. The explicit uncertainty term in the viewpoint scorer was redundant under the tested observation model and should not be presented as an independent source of viewpoint-selection improvement.
