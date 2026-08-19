# Verification Plan

| ID | Requirement | Verification Method | Evidence |
|---|---|---|---|
| REQ-01 | Reproducible simulated surgical workspace | Automated configuration recreation test | Saved configuration + deterministic scene output |
| REQ-04 | Controlled localisation uncertainty and occlusion | Parameter sweep against predefined perturbation levels | Experiment logs |
| REQ-07 | Task-aware active perception | Unit/integration test demonstrating trajectory-relevance weighting | Test output + viewpoint selection trace |
| REQ-10 | Motion planning | Path feasibility and obstacle-intersection tests | Automated planner tests |
| REQ-12 | Ground-truth evaluation | Independent comparison of planned trajectory against simulator truth | Evaluation logs |
| REQ-15 | Reproducibility | Repeat experiment with identical seed/configuration | Identical outputs within deterministic tolerance |
| REQ-16 | Matched comparison | Confirm same scenario seed across all three strategies | Trial metadata |
| REQ-18 | Verification | Automated test suite for critical functions | Test report |
