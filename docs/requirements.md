# Research System Requirements

## 1. Purpose

This document defines the functional and non-functional requirements for the research engineering framework investigating:

**Task-aware active perception coupled with uncertainty-aware motion planning for simulated minimally invasive surgical navigation.**

The system currently includes verified simulation capabilities from Phases 1-5:

- held-out active-perception experimentation;
- robot kinematics and advanced motion planning;
- image processing and stereo 3-D perception;
- learned segmentation and predictive uncertainty;
- rigid registration;
- temporal state estimation;
- uncertainty propagation into navigation geometry.

Requirements are individually identifiable and traceable to software implementation, automated tests, experimental evidence, or research documentation.

Verification refers only to implemented computational behaviour in simulation.

It does not constitute:

- clinical validation;
- regulatory approval;
- medical-device certification;
- evidence of surgical safety;
- evidence of clinical efficacy.

---

# 2. Phase 1 Functional Requirements

## REQ-01 - Reproducible Surgical Workspace

The system shall generate reproducible simulated minimally invasive surgical workspaces containing explicit hidden ground-truth geometry.

## REQ-02 - Target and Safety-Critical Structures

The environment shall contain a target, one or more safety-critical structures, and defined start and goal conditions for motion planning.

## REQ-03 - Explicit Coordinate Frames

The system shall maintain explicit world, camera, and instrument-related coordinate frames using validated rigid-body transformations.

## REQ-04 - Camera Observation

The system shall generate simulated observations of the surgical workspace from defined camera viewpoints.

## REQ-05 - Controlled Perception Degradation

The system shall support reproducible manipulation of perception quality, including:

- localisation uncertainty;
- visibility;
- occlusion;
- viewpoint geometry;
- synthetic visual-quality degradation.

## REQ-06 - Perception Output

The perception subsystem shall provide estimated spatial information required by downstream motion planning.

## REQ-07 - Explicit Uncertainty Representation

Perceived spatial estimates shall contain an explicit localisation-uncertainty representation suitable for downstream planning and quantitative calibration analysis.

## REQ-08 - Fixed-View Baseline

The system shall implement a Fixed View strategy without active camera repositioning.

## REQ-09 - Generic Active-Perception Baseline

The system shall implement a task-agnostic active-perception strategy using quantitative global observation utility.

## REQ-10 - Task-Aware Active Perception

The system shall implement an active-perception strategy incorporating intended trajectory or task-relevant safety information into viewpoint selection.

## REQ-11 - Candidate Viewpoint Evaluation

Candidate camera viewpoints shall be evaluated using quantitative scoring objectives.

## REQ-12 - Motion Planning

The system shall generate geometrically feasible candidate trajectories between defined start and goal states using perceived environmental information.

## REQ-13 - Uncertainty-Aware Planning

Localisation uncertainty shall influence planning through uncertainty-dependent protected geometry or equivalent safety mechanisms.

## REQ-14 - Hidden Ground-Truth Safety Evaluation

Planned trajectories shall be independently evaluated against simulator ground-truth geometry unavailable to the planner.

## REQ-15 - Safety-Margin Violation Detection

The system shall identify whether a trajectory enters a predefined protected region around safety-critical anatomy.

## REQ-16 - Collision Detection

The system shall identify geometric collision between the instrument trajectory and critical simulated structures.

## REQ-17 - Quantitative Metrics

Experimental trials shall expose appropriate perception, planning, safety, and efficiency metrics.

These may include:

- localisation error;
- predicted uncertainty;
- camera movement;
- planning success;
- safe-navigation success;
- collisions;
- safety-margin violations;
- clearance;
- planner iterations;
- planning time;
- path cost.

## REQ-18 - Experimental Logging

Trials shall produce machine-readable records containing sufficient information to identify:

- scenario;
- strategy;
- perturbation condition;
- controlled random seed;
- relevant intermediate outputs;
- final quantitative metrics.

## REQ-19 - Matched Three-Strategy Comparison

Equivalent simulated conditions shall be reused across Fixed View, Generic Active Perception, and Task-Aware Active Perception through a common:

perception
->
localisation uncertainty
->
motion planning
->
hidden ground-truth safety evaluation

pipeline.

## REQ-20 - Automated Experiment Execution

The framework shall support repeated execution of predefined experiments without manual intervention between individual trials.

---

# 3. Phase 2 Requirements - Robot Kinematics and Advanced Planning

## REQ-21 - Advanced Robot Kinematics

The robotics subsystem shall provide explicit kinematic representations sufficient to evaluate robot configuration, tool pose, workspace behaviour, and motion feasibility.

Verification shall include, where implemented:

- forward kinematics;
- homogeneous transforms;
- Jacobians;
- joint constraints;
- manipulability;
- singularity-related metrics.

## REQ-22 - RCM-Constrained Motion

The simulated surgical robot shall support motion subject to a remote-centre-of-motion constraint appropriate to minimally invasive instrument insertion.

RCM deviation shall be quantitatively measurable.

## REQ-23 - Robot Workspace and Reachability

The system shall evaluate whether candidate tool or camera poses are reachable under the implemented robot model and constraints.

## REQ-24 - Advanced Motion Planning

The system shall support multiple motion-planning algorithms sufficient for quantitative engineering comparison.

The implemented Phase 2 methods include:

- RRT;
- RRT*;
- A*.

Planner comparisons shall expose relevant trade-offs rather than assuming a universally superior method.

## REQ-25 - Collision-Aware Planning

Robot configuration and path feasibility checks shall account for relevant simulated collision geometry and constraints.

## REQ-26 - Trajectory Processing and Execution Metrics

Generated paths shall support trajectory-level processing and quantitative execution-related metrics including, where applicable:

- smoothing;
- path length;
- duration;
- waypoint progression;
- execution error;
- constraint violation.

## REQ-27 - Robot-Aware Viewpoint Feasibility

Active-perception candidate viewpoints shall be filterable using robot reachability and safety constraints before viewpoint selection.

---

# 4. Phase 3 Requirements - Image and Stereo Perception

## REQ-28 - Explicit Camera Geometry

The perception subsystem shall implement explicit camera projection conventions and validated transformations between world and camera coordinates.

## REQ-29 - Image Processing

The project shall include genuine image-array processing rather than relying exclusively on abstract localisation-noise models.

Implemented operations may include:

- filtering;
- thresholding;
- morphology;
- contour extraction;
- image-driven localisation.

## REQ-30 - Stereo Geometry

The system shall support calibrated stereo reconstruction from corresponding left and right image observations.

## REQ-31 - 3-D Triangulation

Stereo image measurements shall be convertible into estimated 3-D positions.

## REQ-32 - Stereo Uncertainty Propagation

Image-space measurement uncertainty shall be propagated through stereo triangulation into a 3-D positional covariance.

## REQ-33 - Quantitative Stereo Validation

Stereo uncertainty shall be evaluated quantitatively across controlled synthetic conditions including changes in:

- image measurement noise;
- stereo baseline.

Predicted uncertainty shall be compared against empirical localisation behaviour.

---

# 5. Phase 4 Requirements - Learned Perception

## REQ-34 - Leakage-Resistant Dataset Splitting

Machine-learning train, validation, and test data shall be divided at an appropriate latent-scenario level so that related frames from the same generated scenario do not cross evaluation partitions.

## REQ-35 - Trainable Segmentation Model

The project shall contain a trainable PyTorch segmentation model and reproducible training pipeline.

## REQ-36 - Validation-Only Model Selection

Model selection shall use validation data only.

The held-out test partition shall not be used to select the final training epoch or tune model parameters.

## REQ-37 - Held-Out Segmentation Evaluation

The selected model shall be evaluated on held-out synthetic data using quantitative segmentation and localisation metrics.

Metrics shall include appropriate measures such as:

- Dice coefficient;
- intersection-over-union;
- detection success;
- centroid localisation error.

## REQ-38 - Classical Vision Comparator

Learned perception shall be compared against an appropriate non-learned image-processing baseline where meaningful.

## REQ-39 - Predictive Uncertainty

The learned-perception pathway shall expose quantitative predictive uncertainty information suitable for analysis and downstream propagation.

## REQ-40 - Calibration and Distribution-Shift Evaluation

Learned predictions shall be evaluated under controlled degradation and out-of-distribution conditions.

The project shall not treat poor calibration or failed OOD detection as successful uncertainty estimation.

## REQ-41 - Learned Stereo Integration

Learned image-derived centroid estimates and uncertainty shall be connectable to stereo triangulation.

## REQ-42 - Learned 3-D Covariance Propagation

Learned stereo measurement uncertainty shall propagate into the existing `PositionUncertainty` representation.

## REQ-43 - Ground-Truth Isolation During Runtime Perception

Simulation ground truth used for benchmark evaluation shall not be inserted into runtime learned-perception outputs.

---

# 6. Phase 5 Requirements - Registration and State Estimation

## REQ-44 - Corresponding-Point Rigid Registration

The system shall estimate a proper 3-D rigid transformation between corresponding landmark sets.

The implementation shall:

- estimate rotation;
- estimate translation;
- prevent improper reflection;
- return a valid homogeneous transform.

## REQ-45 - Registration Error Metrics

Registration experiments shall expose quantitative metrics including:

- fiducial registration error;
- translation error;
- rotation error;
- target registration error where reference truth is available for simulation evaluation.

## REQ-46 - Robust Registration

The system shall support robust corresponding-landmark registration in the presence of gross correspondence outliers.

## REQ-47 - Unknown-Correspondence Point-Cloud Registration

The system shall support point-cloud registration without pre-specified row-wise correspondences.

The current Phase 5 implementation uses trimmed iterative closest point.

## REQ-48 - Temporal State Estimation

The perception/navigation system shall support temporal estimation of 3-D position and velocity.

The current state representation is:

[x, y, z, vx, vy, vz]

## REQ-49 - Measurement Covariance Fusion

Temporal state estimation shall consume explicit 3-D measurement covariance rather than assuming identical observation reliability.

## REQ-50 - Prediction During Measurement Dropout

The state estimator shall support prediction during temporary observation dropout.

## REQ-51 - Innovation-Based Outlier Rejection

The temporal estimator shall support statistical innovation gating capable of rejecting inconsistent measurements.

## REQ-52 - Numerically Stable Covariance Update

The state-estimation covariance update shall preserve symmetry and positive-semidefinite behaviour within numerical tolerance.

## REQ-53 - State-Uncertainty Calibration Evaluation

State-estimator uncertainty shall be evaluated using empirical covariance coverage against nominal Gaussian confidence regions in simulation.

## REQ-54 - Validation/Held-Out Separation for State-Estimator Tuning

Any process-noise parameter selected for uncertainty calibration shall be selected using validation simulations and subsequently evaluated on disjoint held-out simulations.

## REQ-55 - Registration Uncertainty Propagation

Position covariance shall be transformed into the navigation frame.

Where registration-pose uncertainty is modelled, it shall be propagated into positional measurement covariance.

## REQ-56 - Registered Temporal Navigation Integration

The system shall support the integrated software pathway:

uncertain 3-D observation
->
rigid registration
->
registered covariance
->
temporal state estimation
->
EstimatedStructure
->
uncertainty-aware planning geometry

## REQ-57 - Integrated Planner Uncertainty Propagation

Tracked state covariance shall influence the uncertainty-aware safety geometry supplied to the planner.

## REQ-58 - Integrated Ground-Truth Isolation

Ground-truth registration transforms and target states used for simulation evaluation shall not be used as runtime measurement corrections.

---

# 7. Non-Functional Requirements

## NFR-01 - Modularity

Geometry, robotics, perception, active perception, uncertainty, registration, state estimation, planning, evaluation, and experiment logic shall remain separable software components.

## NFR-02 - Reproducibility

Recorded configuration and controlled random seeds shall support reproduction of stochastic simulation conditions within the deterministic limits of the software environment.

## NFR-03 - Traceability

Requirements, implementation, tests, experiment configurations, generated artifacts, and repository revision history shall remain traceable.

## NFR-04 - Testability

Critical mathematical, algorithmic, integration, and experimental components shall be independently testable.

## NFR-05 - Numerical Robustness

Geometric, registration, covariance, and transformation operations shall use appropriate numerical tolerances rather than inappropriate exact floating-point equality.

## NFR-06 - Quantitative Evaluation

Research conclusions shall be supported by quantitative experimental evidence rather than visual demonstrations alone.

## NFR-07 - Robustness Evaluation

The framework shall support controlled evaluation across relevant perturbations including:

- geometric variation;
- occlusion;
- localisation uncertainty;
- image degradation;
- distribution shift;
- registration outliers;
- observation dropout;
- temporal measurement outliers.

## NFR-08 - Extensibility

Alternative perception models, uncertainty representations, state estimators, viewpoint-selection methods, or motion planners should be introducible without redesigning the complete framework.

## NFR-09 - Computational Observability

Intermediate outputs required for engineering diagnosis shall remain available for analysis.

These include, where relevant:

- selected viewpoints;
- uncertainty estimates;
- registration residuals;
- inlier masks;
- state covariance;
- innovation statistics;
- planning outcomes;
- path cost;
- clearance;
- safety margins.

## NFR-10 - Interpretation Safety

Simulation outcomes shall not be represented as:

- clinically validated thresholds;
- patient-risk estimates;
- evidence of clinical effectiveness;
- evidence of medical-device safety.

---

# 8. Verification Baseline

The previously frozen Phase 1 release baseline was:

449 passed
0 failed

Following implementation of Phases 2-5, the latest completed full regression before final repository packaging was:

822 passed

A later full-suite invocation was manually interrupted and therefore is not treated as release verification.

The final consolidated Phases 2-5 release regression subsequently completed successfully with 822 tests passing.

The expanded verification suite now covers:

- coordinate transforms;
- robot kinematics;
- workspace and reachability;
- RCM constraints;
- RRT, RRT*, and A* planning;
- trajectory execution metrics;
- robot-aware viewpoint selection;
- camera projection;
- OpenCV image processing;
- stereo geometry;
- triangulation;
- stereo uncertainty propagation;
- synthetic machine-learning datasets;
- Tiny U-Net segmentation;
- held-out segmentation evaluation;
- predictive uncertainty;
- degradation and OOD behaviour;
- learned stereo integration;
- rigid registration;
- RANSAC;
- ICP;
- temporal Kalman state estimation;
- innovation gating;
- dropout prediction;
- state-estimator uncertainty calibration;
- registered perception-to-navigation integration;
- uncertainty-aware planner-margin propagation.

Requirement-to-evidence mapping is maintained in:

docs/traceability_matrix.md

---

# 9. Phase 2 Verification Summary

Phase 2 extends the original navigation framework with robot-aware kinematics, reachability, advanced planning, and trajectory execution.

Implemented evidence includes:

src/robotics/advanced_kinematics.py
src/robotics/advanced_planning.py
src/robotics/trajectory_execution.py
src/robotics/workspace_analysis.py
src/perception/robot_aware_selection.py
src/perception/robot_reachability.py

Associated verification includes:

tests/test_phase2_advanced_kinematics.py
tests/test_phase2_advanced_planning.py
tests/test_phase2_robot_aware_selection.py
tests/test_phase2_robot_reachability.py
tests/test_phase2_trajectory_execution.py

Primary quantitative artifact:

results/phase2/phase2_planning_benchmark.json

Representative observed benchmark behaviour included:

Workspace samples: 441

Reachable demonstration viewpoints: 2 / 3

RRT: 3 / 3 successful

RRT*: 3 / 3 successful

A*: 1 / 1 successful

The tested planners exhibited different path-quality, runtime, and trajectory characteristics.

No universal planner-dominance claim is made.

---

# 10. Phase 3 Verification Summary

Phase 3 introduces image-based perception, explicit camera geometry, calibrated stereo reconstruction, and image-to-3-D uncertainty propagation.

Implemented evidence includes:

src/perception/image_driven_perception.py
src/perception/image_geometry.py
src/perception/image_processing.py
src/perception/stereo_geometry.py
src/perception/stereo_uncertainty.py

Associated tests include:

tests/test_phase3_image_geometry.py
tests/test_phase3_image_processing.py
tests/test_phase3_image_uncertainty.py
tests/test_phase3_stereo_geometry.py

Primary evidence artifact:

results/phase3/phase3_stereo_uncertainty_benchmark.json

The synthetic calibrated-camera benchmark evaluated combinations of:

- stereo baseline;
- pixel measurement noise.

Observed broad behaviour was physically consistent:

- greater pixel uncertainty generally increased 3-D uncertainty;
- greater stereo baseline generally reduced depth uncertainty.

Across the tested conditions, predicted and empirical localisation uncertainty were approximately consistent.

This is simulation evidence only and does not demonstrate physical stereo-camera calibration accuracy.

---

# 11. Phase 4 Verification Summary

Phase 4 introduces trainable learned perception and explicitly evaluates both successful and failed uncertainty behaviours.

Implemented evidence includes:

src/perception/ml_dataset.py
src/perception/ml_segmentation.py
src/perception/ml_uncertainty.py
src/perception/ml_stereo_perception.py
src/simulation/phase4_train_segmentation.py
src/simulation/phase4_uncertainty_robustness_benchmark.py
src/simulation/phase4_final_integration_benchmark.py

Associated tests include:

tests/test_phase4_ml_dataset.py
tests/test_phase4_ml_segmentation.py
tests/test_phase4_ml_uncertainty.py
tests/test_phase4_ml_stereo_perception.py

Evidence artifacts include:

results/phase4/phase4_segmentation_training.json
results/phase4/phase4_uncertainty_robustness.json
results/phase4/phase4_final_integration_benchmark.json
results/phase4/tiny_unet_best.pt

## Phase 4 Segmentation Evidence

The selected Tiny U-Net was chosen using validation-only model selection.

Selected epoch:

3

Held-out synthetic Tiny U-Net results:

Dice: 0.8807

IoU: 0.8084

Detection: 100%

Centroid error: 1.017 px

Classical HSV comparator:

Dice: 0.8466

IoU: 0.7450

Detection: 100%

Centroid error: 0.982 px

The Tiny U-Net improved Dice and IoU under this evaluation, while the classical baseline produced a slightly lower centroid error.

The results therefore do not establish universal learned-perception superiority.

## Phase 4 Robustness Evidence

Representative uncertainty/degradation results included:

Clean Dice: 0.9303

Severe Dice: 0.8073

Colour-shift OOD Dice: 0.0769

The robustness evaluation also showed that:

- Brier score worsened under degradation;
- predictive entropy increased only weakly;
- perturbation variance decreased under severe and OOD conditions;
- perturbation variance therefore failed as a reliable OOD indicator;
- foreground probability calibration remained poor.

These failure modes are retained as research evidence.

## Phase 4 Final Learned-Stereo Evidence

Final integration benchmark:

Clean:

- mean 3-D localisation error = 31.794 mm
- 95% covariance coverage = 66.7%

Moderate:

- mean 3-D localisation error = 14.101 mm
- 95% covariance coverage = 66.7%

Colour-shift OOD:

- mean 3-D localisation error = 112.345 mm
- 95% covariance coverage = 0.0%

Strong held-out 2-D segmentation performance therefore did not guarantee reliable downstream stereo localisation.

The moderate condition producing lower localisation error than the clean condition must not be interpreted as degradation improving the model.

Only six targets were evaluated per condition, and the result instead demonstrates sensitivity of stereo depth estimation to image-space centroid/disparity behaviour.

---

# 12. Phase 5 Verification Summary

Phase 5 introduces rigid registration, robust registration, unknown-correspondence registration, temporal state estimation, statistical outlier gating, uncertainty calibration, and registered temporal planning integration.

Implemented evidence includes:

src/geometry/registration.py
src/geometry/robust_registration.py
src/perception/state_estimation.py
src/perception/tracked_navigation.py
src/simulation/phase5_registration_benchmark.py
src/simulation/phase5_tracking_benchmark.py
src/simulation/phase5_tracking_calibration_benchmark.py
src/simulation/phase5_final_integration_benchmark.py

Associated tests include:

tests/test_phase5_registration.py
tests/test_phase5_robust_registration.py
tests/test_phase5_state_estimation.py
tests/test_phase5_tracked_navigation.py

Evidence artifacts include:

results/phase5/phase5_registration_benchmark.json
results/phase5/phase5_tracking_benchmark.json
results/phase5/phase5_tracking_calibration_benchmark.json
results/phase5/phase5_final_integration_benchmark.json

## Robust Registration Benchmark

Across 20 synthetic correspondence-outlier trials:

Naive mean translation error: 6.860 mm

RANSAC mean translation error: 0.295 mm

Naive mean rotation error: 7.064 deg

RANSAC mean rotation error: 0.440 deg

Naive mean RMS TRE: 9.605 mm

RANSAC mean RMS TRE: 0.485 mm

RANSAC mean inlier precision: 100.0%

RANSAC mean inlier recall: 100.0%

RANSAC lower TRE than naive: 100.0% of trials

These results demonstrate robustness to the deliberately injected synthetic correspondence outliers used by this benchmark.

They do not establish performance for arbitrary physical registration conditions.

## ICP Benchmark

Across 15 controlled synthetic trials:

Mean translation error: 0.060 mm

Mean rotation error: 0.057 deg

Mean RMS TRE: 0.084 mm

Mean residual RMS: 0.376 mm

Convergence: 100.0%

ICP was evaluated from sufficiently nearby initial alignment.

These results do not establish global convergence from arbitrary initialisation.

## Temporal Tracking Benchmark

Initial temporal benchmark:

Raw position RMSE: 10.238 mm

Raw non-outlier position RMSE: 4.916 mm

Tracked position RMSE: 2.530 mm

Tracked dropout RMSE: 2.671 mm

Raw finite-difference velocity RMSE: 278.665 mm/s

Tracked velocity RMSE: 8.508 mm/s

Nominal 95% covariance coverage: 80.5%

Outlier rejection precision: 75.4%

Outlier rejection recall: 100.0%

Although temporal tracking substantially reduced position and velocity error, the initial covariance estimate was under-calibrated.

## Validation-Based Tracking Calibration

Process noise was therefore tuned using validation simulations only.

Candidate acceleration standard deviations were:

- 0.008 m/s^2
- 0.012 m/s^2
- 0.018 m/s^2
- 0.025 m/s^2
- 0.035 m/s^2
- 0.050 m/s^2

Validation selected:

acceleration_sigma = 0.025 m/s^2

The selected value was then evaluated on a disjoint held-out synthetic seed set.

Held-out results:

Raw position RMSE: 10.356 mm

Raw non-outlier position RMSE: 4.994 mm

Tracked position RMSE: 2.330 mm

Tracked dropout RMSE: 2.687 mm

Tracked velocity RMSE: 8.029 mm/s

Nominal 95% covariance coverage: 95.3%

Outlier rejection precision: 90.3%

Outlier rejection recall: 100.0%

The 95.3% coverage result is evidence only for the implemented synthetic motion and measurement-noise model.

It is not clinical uncertainty calibration.

## Final Registration-Tracking-Navigation Integration

The final Phase 5 integration benchmark used:

- an estimated RANSAC registration transform;
- residual-derived registration-pose uncertainty;
- heteroscedastic uncertain 3-D observations;
- measurement dropout;
- gross measurement outliers;
- the validation-selected Kalman process-noise parameter;
- temporal state estimation;
- uncertainty-aware planner-margin propagation.

Representative mean results:

Registration translation error: 1.005 mm

Registration rotation error: 0.340 deg

Raw registered position RMSE: 10.065 mm

Raw clean registered position RMSE: 5.016 mm

Tracked position RMSE: 2.417 mm

Tracked dropout RMSE: 2.703 mm

Tracker lower RMSE: 100.0% of trials

Uncertainty results:

Raw all-measurement 95% coverage: 93.6%

Raw clean-measurement 95% coverage: 99.1%

Tracked 95% coverage: 99.3%

Outlier gating:

Precision: 97.7%

Recall: 100.0%

Planner uncertainty propagation:

Mean raw safety margin: 14.548 mm

Mean tracked safety margin: 9.182 mm

Mean tracked dropout safety margin: 9.455 mm

The tracked covariance coverage of 99.3% is conservative relative to the nominal 95% region.

It must not be described as perfect calibration.

The final temporal benchmark uses synthetic uncertain 3-D observations conforming to the learned-stereo software interface.

The actual trained Tiny U-Net image-to-stereo pipeline is evaluated separately in Phase 4 rather than rerun inside every temporal Phase 5 trial.

---

# 13. Scientific Interpretation Boundary

Verification evidence demonstrates implementation correctness and computational behaviour relative to the current simulation requirements.

It does not demonstrate:

- clinical safety;
- clinical effectiveness;
- patient-specific performance;
- anatomical generalisation to real patients;
- clinical image segmentation performance;
- physical sensor accuracy;
- physical camera calibration;
- physical registration accuracy;
- physical robot performance;
- medical-device safety;
- regulatory compliance;
- medical-device certification;
- suitability for clinical use.

The tested scenarios are engineered simulated variations rather than patients or independent patient anatomies.

---

# 14. Current Scope Boundary

The completed Phase 1-5 research system does not claim:

- patient-specific performance;
- real anatomical generalisation;
- clinical image segmentation accuracy;
- physical stereo-camera calibration;
- physical image-to-patient registration;
- deformable registration;
- deformable tissue modelling;
- physical surgical robot performance;
- force or tactile sensing;
- autonomous cutting or suturing;
- animal or cadaver validation;
- clinical validation;
- surgical safety;
- clinical efficacy;
- regulatory certification.

Phase 4 learned perception is trained and tested on synthetic imagery.

Phase 5 registration and temporal tracking experiments use synthetic geometry and synthetic uncertain 3-D observations.

The final Phase 5 temporal integration benchmark uses a software-compatible uncertain stereo-observation interface but does not rerun the trained Tiny U-Net image pathway inside every temporal trial.

---

# 15. Remaining Roadmap Requirements

Later requirements will be introduced separately for:

- Phase 6 - robust planning under uncertainty;
- Phase 7 - safety supervision and autonomous task execution;
- Phase 8 - ROS 2 and Gazebo system integration;
- Phase 9 - control and real-time trajectory execution;
- Phase 10 - final end-to-end experiment and verification.

Capabilities belonging to those future phases are not claimed as complete by this document.

---

# Final ROS 2 Runtime Requirements

##  - Complete ROS 2 Runtime Bringup

The system shall provide reproducible ROS 2 Jazzy bringup integrating simulation, perception, navigation execution, runtime safety supervision, robot state, controllers, and research visualisation.

##  - Navigation Action Interface

The runtime shall expose a ROS 2 navigation action accepting a target pose and reporting execution progress, current tool pose, safety state, success or failure, and explicit terminal state.

##  - Runtime Perception Uncertainty

Runtime anatomical estimates shall contain explicit positional uncertainty capable of influencing autonomous execution behaviour rather than serving only as an offline metric.

##  - Runtime Safety Monitoring

Runtime safety evaluation shall use relevant live execution information including localisation uncertainty, predicted clearance, trajectory tracking error, joint state, joint-limit proximity, and execution progression.

##  - Autonomous Safety Intervention

Runtime safety conditions shall support autonomous replanning, perception reacquisition, execution recovery, and stopping. Critical conditions shall be capable of terminating an active navigation action.

##  - Gazebo and ros2_control Execution

Planner-generated trajectories shall execute through a ROS 2 joint trajectory controller connected to the simulated surgical instrument in Gazebo Harmonic.

##  - Runtime Research Visualisation

The runtime shall visualise the target, RCM, protected anatomy, safety boundary, uncertainty envelope, real planner-derived tool-tip path, and live autonomous safety state.

##  - Reproducible A/B/C Uncertainty Demonstrations

The launch system shall provide reproducible A/B/C demonstrations using the same target and execution architecture while changing localisation uncertainty: A = 0.003 m, B = 0.020 m, and C = 0.035 m. The demonstrations shall distinguish nominal execution, recoverable uncertainty handling, and critical autonomous stopping.
