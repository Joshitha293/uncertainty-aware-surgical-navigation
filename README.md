# Uncertainty-Aware Surgical Navigation with Active Perception and Autonomous Safety

A simulation-based surgical robotics research framework coupling **active perception, uncertainty-aware motion planning, autonomous safety supervision, and ROS 2 execution** for minimally invasive surgical navigation.

The project investigates how uncertainty in anatomical localisation affects perception, motion-planning feasibility, and autonomous robot behaviour.

> **Research prototype:** This repository is for simulation and engineering research only. It is not a clinical system, medical device, or validated surgical platform.

---

## Overview

The system models a minimally invasive surgical instrument operating around safety-critical anatomy under uncertain perception.

A central design principle is that deployable algorithms do **not** receive perfect simulator geometry.

```text
Hidden simulator truth
        |
        +----> synthetic observation generation
        |
        +----> independent evaluation only


Estimated anatomy + uncertainty
        |
        v
Active perception
        |
        v
Uncertainty-aware motion planning
        |
        v
Autonomous runtime safety supervision
        |
        v
ROS 2 / ros2_control execution
        |
        v
Gazebo simulation
```

The core engineering question is:

> How should a surgical-navigation system change its behaviour when its confidence in anatomical localisation changes?

---

## Key Capabilities

The repository includes:

- RCM-constrained minimally invasive instrument modelling
- forward and inverse kinematics
- collision-aware joint-space RRT planning
- edge validation and path shortcutting
- time-parameterised trajectory execution
- explicit Gaussian localisation uncertainty
- uncertainty-inflated safety geometry
- chance-constrained and risk-aware planning
- Generic and Task-Aware Active Perception
- robot-aware viewpoint feasibility filtering
- visibility and occlusion modelling
- stereo and image-driven perception
- learned segmentation and uncertainty estimation
- rigid registration and state estimation
- runtime clearance and tracking-error monitoring
- autonomous safety-state management
- ROS 2 Jazzy integration
- Gazebo Harmonic simulation
- ros2_control execution
- custom ROS 2 navigation action
- planner-derived trajectory visualisation
- live safety visualisation
- reproducible A/B/C uncertainty demonstrations
- automated verification and experimental benchmarking

---

## System Architecture

```text
                 SIMULATED PERCEPTION
                         |
                         v
             Estimated anatomy + covariance
                         |
                         v
                  Active Perception
                /                   \
               /                     \
        Generic scoring        Task-Aware scoring
               \                     /
                \                   /
                  Selected viewpoint
                         |
                         v
              Updated anatomical estimate
                         |
                         v
             Uncertainty-aware planning
                         |
                         v
                Collision-aware RRT
                         |
                  Path shortcutting
                         |
               Time parameterisation
                         |
                         v
               ROS 2 Navigation Action
                         |
                         v
                    ros2_control
                         |
                         v
                  Gazebo Harmonic
                         |
             +-----------+-----------+
             |           |           |
             v           v           v
         joint state   tracking   execution
                       error      progression
             \           |           /
              \          |          /
               +---------+---------+
                         |
                         v
               Runtime Safety Monitor
                         |
                         v
               Autonomous Supervisor
                         |
           +-------------+-------------+
           |             |             |
           v             v             v
        CONTINUE     REACQUIRE       REPLAN
                                       |
                                  RECOVER / STOP
```

The research-core algorithms remain separated from the ROS 2 integration layer.

See [`docs/architecture.md`](docs/architecture.md) for the detailed architecture.

---

## Final ROS 2 Autonomous Navigation Demo

The completed runtime integrates the research algorithms with:

- **ROS 2 Jazzy**
- **Gazebo Harmonic**
- **ros2_control**
- simulated uncertain anatomical perception
- collision-aware RRT planning
- controller-backed trajectory execution
- runtime safety monitoring
- autonomous intervention
- RViz research visualisation

The final A/B/C demonstration holds the **robot, target, planner, and execution stack constant** while changing localisation uncertainty.

### A/B/C Results

| Case | Localisation uncertainty | Autonomous behaviour | Final result |
|---|---:|---|---|
| **A** | **3 mm** | `SAFE → EXECUTE` | ✅ Completed |
| **B** | **20 mm** | `REACQUIRE` | 🟠 Safe abort when uncertainty persists |
| **C** | **35 mm** | `STOP` | 🔴 Autonomous safety stop |

The same target is used in all three cases:

```text
frame_id = robot_base
x = 0.10 m
y = 0.00 m
z = 0.00 m
```

Observed terminal behaviour:

```text
Demo A
------
success: true
terminal_state: completed
Goal finished with status: SUCCEEDED


Demo B
------
safety_state: reacquiring
success: false
terminal_state: failed
message: Safety intervention timed out without recovery.
Goal finished with status: ABORTED


Demo C
------
success: false
terminal_state: stopped
message: Navigation stopped by the autonomous safety supervisor.
Goal finished with status: ABORTED
```

This demonstrates that uncertainty is connected directly to runtime behaviour rather than being reported only as an offline metric.

Detailed runtime evidence is maintained in:

[`docs/final_ros2_runtime_evidence.md`](docs/final_ros2_runtime_evidence.md)

---

## Runtime Visualisation

The final demonstration visualises:

- surgical instrument and actuation housing
- trocar / RCM entry point
- instrument shaft
- distal tool head
- protected anatomical regions
- navigation target
- safety boundary
- uncertainty envelope
- real planner-derived tool-tip trajectory
- live autonomous safety state

The planned trajectory is published on:

```text
/navigation/planned_path_marker
```

The visualised path is generated from:

```text
RRT path
    |
    v
shortcut-smoothed joint-space path
    |
    v
instrument forward kinematics
    |
    v
tool-tip XYZ trajectory
```

It therefore represents the actual planner output rather than a decorative straight-line path.

Live scene and safety markers are published on:

```text
/navigation/demo_markers
```

---

## Research Study

The active-perception research asks:

> Given the same uncertain initial anatomical estimate and approximately equal camera-motion expenditure, does incorporating the intended surgical trajectory improve downstream simulated navigation?

The final frozen Phase 1 experiment compared:

1. Fixed View
2. Random Active
3. Generic Active
4. Movement-Budget-Matched Generic
5. Alignment-Only
6. Task-Weighted Information-Only
7. Full Task-Aware
8. Privileged Oracle

The primary comparison was:

```text
Full Task-Aware
        versus
Movement-Budget-Matched Generic
```

with safe-navigation success as the primary endpoint.

---

## Final Held-Out Research Result

The final held-out experiment did **not** establish superiority of Full Task-Aware Active Perception.

| Metric | Full Task-Aware | Budget-Matched Generic |
|---|---:|---:|
| Safe-navigation success | **61.33%** | **71.33%** |
| Difference | **−10.00 percentage points** | |
| 95% bootstrap CI | **[−17.00, −3.00] pp** | |
| Two-sided permutation p-value | **0.012390** | |

The negative result was retained without post-hoc retuning.

A further limitation was that movement-budget matching did not generalise perfectly to the held-out distribution, so the observed difference cannot be attributed cleanly to task awareness alone.

This is treated as a scientific result rather than hidden or retrospectively corrected.

For the full methodology and statistical analysis, see:

- [`docs/experimental_protocol.md`](docs/experimental_protocol.md)
- [`results/final_evidence/final_results_report.md`](results/final_evidence/final_results_report.md)
- [`docs/traceability_matrix.md`](docs/traceability_matrix.md)

---

## Quick Start

### Python Research Framework

Clone the repository:

```bash
git clone https://github.com/Joshitha293/uncertainty-aware-surgical-navigation.git
cd uncertainty-aware-surgical-navigation
```

Create and activate a Python environment, then install the required dependencies.

Run the Python test suite with:

```bash
pytest
```

---

## ROS 2 Setup

The final robot runtime was developed with:

```text
Ubuntu / WSL
ROS 2 Jazzy
Gazebo Harmonic
ros2_control
Python 3
```

Build the ROS 2 workspace:

```bash
source /opt/ros/jazzy/setup.bash

cd ~/surgical_ws

colcon build --symlink-install

source ~/surgical_ws/install/setup.bash
```

---

## Run the Final Demonstrations

### Demo A — 3 mm uncertainty

```bash
ros2 launch surgical_navigation_bringup \
  surgical_navigation.launch.py \
  demo_case:=A
```

### Demo B — 20 mm uncertainty

```bash
ros2 launch surgical_navigation_bringup \
  surgical_navigation.launch.py \
  demo_case:=B
```

### Demo C — 35 mm uncertainty

```bash
ros2 launch surgical_navigation_bringup \
  surgical_navigation.launch.py \
  demo_case:=C
```

The launch argument automatically selects both the simulated uncertainty level and the corresponding Gazebo research scene:

```text
A -> sigma = 0.003 m -> model_demo_a.sdf
B -> sigma = 0.020 m -> model_demo_b.sdf
C -> sigma = 0.035 m -> model_demo_c.sdf
```

---

## Send a Navigation Goal

With one of the demos running:

```bash
ros2 action send_goal \
  /navigation/execute_navigation \
  surgical_navigation_interfaces/action/ExecuteNavigation \
  "{target: {header: {frame_id: 'robot_base'}, pose: {position: {x: 0.10, y: 0.0, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}}" \
  --feedback
```

The action reports:

- execution progress
- current tool pose
- current safety state
- terminal state
- success or failure
- safety intervention outcome

---

## Repository Structure

```text
.
├── src/
│   ├── geometry/              # geometry, transforms and registration
│   ├── perception/            # active, stereo and learned perception
│   ├── robotics/              # planning, safety and execution algorithms
│   └── simulation/            # experiments and benchmarks
│
├── ros2/
│   └── src/
│       ├── surgical_navigation_interfaces/
│       ├── surgical_navigation_ros/
│       ├── surgical_navigation_description/
│       └── surgical_navigation_bringup/
│
├── tests/                     # research-core automated tests
├── results/                   # experimental evidence
└── docs/                      # detailed technical documentation
```

---

## Documentation

Detailed material is intentionally kept outside the README.

| Document | Purpose |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Detailed system and runtime architecture |
| [`docs/coordinate_frames.md`](docs/coordinate_frames.md) | Coordinate-frame conventions |
| [`docs/experimental_protocol.md`](docs/experimental_protocol.md) | Frozen research methodology |
| [`docs/requirements.md`](docs/requirements.md) | Engineering and research requirements |
| [`docs/traceability_matrix.md`](docs/traceability_matrix.md) | Requirement-to-evidence traceability |
| [`docs/verification_plan.md`](docs/verification_plan.md) | Verification approach |
| [`docs/final_ros2_runtime_evidence.md`](docs/final_ros2_runtime_evidence.md) | Final A/B/C ROS 2 runtime evidence |
| [`results/final_evidence/final_results_report.md`](results/final_evidence/final_results_report.md) | Experimental results |

---

## Design Goals

The project prioritises:

1. **Truth isolation**
   Deployable algorithms should reason from estimated anatomy rather than simulator ground truth.

2. **Explicit uncertainty**
   Perception uncertainty should be represented quantitatively and propagated downstream.

3. **Safety-aware planning**
   Anatomical uncertainty should influence collision margins and planning feasibility.

4. **Closed-loop supervision**
   Safety should continue to be evaluated during execution rather than ending after planning.

5. **Reproducibility**
   Experiments, random seeds, scenario splits, benchmarks, and runtime demonstration cases should be reproducible.

6. **Scientific honesty**
   Negative results and failed hypotheses are retained rather than hidden through post-hoc retuning.

---

## Limitations

This remains a simulation research platform.

Current limitations include:

- no physical surgical robot
- no cadaveric, animal, or human validation
- no clinical validation
- no medical-device certification
- simulated anatomical environments
- simulated runtime perception in the ROS 2 demonstration
- no direct clinical imaging pipeline in the final Gazebo runtime
- simplified RCM-constrained instrument model
- simplified tissue and collision geometry
- no tissue deformation or force interaction
- no guarantee that simulation performance transfers to clinical environments

The project should therefore be interpreted as a **robotics and computational research prototype**, not evidence of clinical readiness.

---

## License

See [`LICENSE`](LICENSE) for licensing information.

---

## Further Evidence

The repository contains the complete experimental outputs, verification artifacts, statistical analyses, and runtime implementation needed to inspect the work beyond this overview.

The README intentionally provides only the project-level story; detailed evidence is maintained in `docs/` and `results/`.
