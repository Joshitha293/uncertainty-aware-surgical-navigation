# Final ROS 2 Runtime Evidence

## Purpose

This document records the final end-to-end runtime verification of the simulated uncertainty-aware surgical-navigation system.

The evidence is engineering and simulation evidence only. It does not constitute clinical validation, physical-robot validation, or medical-device certification.

---

## Runtime Stack

The final runtime integrates:

- ROS 2 Jazzy
- Gazebo Harmonic
- ros2_control
- custom `ExecuteNavigation` action
- simulated uncertain anatomical perception
- collision-aware RRT planning
- trajectory shortcutting and time parameterisation
- runtime safety monitoring
- autonomous supervision
- RViz research visualisation

---

## Common Navigation Target

All final A/B/C demonstrations use the same target:

```text
frame_id = robot_base
x = 0.10 m
y = 0.00 m
z = 0.00 m
```

The robot, target, planner, controller, and execution architecture remain unchanged between cases.

Only localisation uncertainty changes.

---

## Demo A — Nominal Uncertainty

Configured uncertainty:

```text
position_sigma_m = 0.003
```

Observed runtime sequence:

```text
planning
executing
completed
```

Final action result:

```text
success: true
terminal_state: completed
message: Phase 2 trajectory executed successfully under ROS autonomous supervision.
Goal finished with status: SUCCEEDED
```

Live visual safety state:

```text
SAFE
```

**Result: PASS — normal supervised execution completed successfully.**

---

## Demo B — Recoverable Uncertainty

Configured uncertainty:

```text
position_sigma_m = 0.020
```

Observed runtime sequence:

```text
planning
executing
reacquiring
```

Final action result:

```text
success: false
terminal_state: failed
message: Safety intervention timed out without recovery.
Goal finished with status: ABORTED
```

The uncertainty was intentionally held persistently high. The supervisor therefore requested perception reacquisition but did not receive a sufficiently improved uncertainty estimate within the intervention window.

**Result: PASS — recoverable uncertainty triggered autonomous reacquisition and safe termination when uncertainty persisted.**

---

## Demo C — Critical Uncertainty

Configured uncertainty:

```text
position_sigma_m = 0.035
```

Final action result:

```text
success: false
terminal_state: stopped
message: Navigation stopped by the autonomous safety supervisor.
Goal finished with status: ABORTED
```

**Result: PASS — critical localisation uncertainty produced an autonomous safety stop.**

---

## Final A/B/C Summary

| Case | Uncertainty | Runtime response | Terminal outcome |
|---|---:|---|---|
| A | 0.003 m | SAFE / execute | Completed |
| B | 0.020 m | REACQUIRE | Safe abort |
| C | 0.035 m | STOP | Autonomous stop |

The final demonstration therefore shows:

```text
3 mm uncertainty
        |
        v
normal supervised execution

20 mm uncertainty
        |
        v
perception reacquisition
        |
        v
safe termination if uncertainty persists

35 mm uncertainty
        |
        v
critical autonomous stop
```

This verifies that localisation uncertainty directly influences runtime robot behaviour rather than functioning only as an offline reported metric.

---

## Planner-Derived Path Visualisation

The final ROS runtime publishes:

```text
/navigation/planned_path_marker
```

The marker was verified as:

```text
ns: planned_navigation_path
type: 4
```

ROS Marker type `4` is a `LINE_STRIP`.

The visualised points are generated from:

```text
collision-aware RRT result
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

The visualised trajectory therefore represents the real planner output rather than a decorative straight-line approximation.

---

## Live Research Visualisation

The final visualisation node publishes:

```text
/navigation/demo_markers
```

Verified visual elements include:

- body/workspace context
- RCM / trocar entry point
- navigation target
- protected anatomy
- safety boundary
- uncertainty envelope
- live autonomous safety state

The nominal live status was explicitly verified as:

```text
SAFE
```

The runtime also supports visible recoverable and critical intervention states.

---

## Automatic A/B/C Configuration

The complete demo is launched with:

```bash
ros2 launch surgical_navigation_bringup \
  surgical_navigation.launch.py \
  demo_case:=A
```

The same command accepts `A`, `B`, or `C`.

The selected case automatically configures both perception uncertainty and the corresponding Gazebo research scene:

```text
A -> sigma = 0.003 m -> model_demo_a.sdf
B -> sigma = 0.020 m -> model_demo_b.sdf
C -> sigma = 0.035 m -> model_demo_c.sdf
```

No separate manual Gazebo research-scene spawn is required.

---

## Runtime Safety Behaviour

The runtime architecture supports safety-relevant supervision using signals including:

- localisation uncertainty
- predicted protected-region clearance
- controller-derived tracking error
- joint state
- joint-limit proximity
- execution progression

The autonomous supervisor supports safety responses including:

```text
CONTINUE
REPLAN
REACQUIRE
RECOVER
STOP
```

Critical conditions can terminate the active navigation action with explicit stopped/aborted semantics.

---

## Targeted Final ROS Regression

Following the final planner-path, visualisation, and A/B/C runtime changes:

```text
29 tests
0 errors
0 failures
2 skipped
```

A complete repository-wide regression is performed separately as the final Phase 10 verification step.

---

## Limitations

The final runtime remains simulation-based.

It does not include:

- physical surgical-robot execution
- real patient anatomy
- cadaveric or animal validation
- clinical imaging in the final Gazebo runtime
- deformable tissue mechanics
- force or haptic interaction
- clinical validation
- medical-device certification

The results should therefore be interpreted as robotics and software-engineering evidence only.
