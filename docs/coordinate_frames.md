# Coordinate Frames and Transformation Conventions

## 1. Purpose

The simulation uses explicit coordinate frames to maintain a consistent
geometric relationship between the surgical workspace, robotic instrument,
camera and simulated anatomical structures.

Explicit frame definitions are necessary because perception outputs, robot
states, planning geometry and simulator ground truth may be represented in
different reference frames.

All transformations therefore use a consistent mathematical convention.

---

## 2. Frame Notation

A coordinate frame is denoted by:

{A}

A point expressed in frame {A} is denoted:

^A p

The rigid-body transformation:

^A T_B

represents the pose of frame {B} expressed relative to frame {A}.

Therefore, a point expressed in frame {B} can be transformed into frame {A}
using:

^A p = ^A T_B · ^B p

This notation will be used consistently throughout the project.

---

## 3. World Frame — {W}

The world frame is the fixed global reference frame of the simulation.

The convention is:

- +x_W: right
- +y_W: forward
- +z_W: upward

Ground-truth geometry is ultimately represented relative to this frame.

The world frame does not move during an experimental trial.

---

## 4. Robot Base Frame — {B}

The robot-base frame is attached to the base reference of the simulated
robotic instrument system.

Its pose relative to the world frame is represented by:

^W T_B

This frame provides the reference for subsequent robot kinematic
transformations.

---

## 5. Camera Frame — {C}

The camera frame is attached to the virtual endoscopic camera.

The camera convention is:

- +x_C: image right
- +y_C: image down
- +z_C: forward along the optical axis

The pose of the camera relative to the world frame is represented by:

^W T_C

The distinction between camera and world coordinate conventions is explicit.
Perception outputs expressed in camera coordinates must therefore be
transformed before they are used as world-frame planning geometry.

---

## 6. Tool Frame — {T}

The tool frame is attached to the simulated surgical instrument.

Unless explicitly stated otherwise, its origin is defined at the tool-tip
reference point.

Its pose relative to the world frame is represented by:

^W T_T

The tool frame will later support kinematic calculations, trajectory
representation and geometric safety evaluation.

---

## 7. Anatomical Structure Frames — {A_i}

A simulated anatomical structure may be assigned a local coordinate frame:

{A_i}

where i identifies the structure.

Its pose in the world frame is represented by:

^W T_Ai

Local anatomical geometry can therefore be mapped into the common world frame
before distance, collision or safety-margin calculations are performed.

---

## 8. Homogeneous Transformation Representation

Rigid-body transformations are represented using a 4 × 4 homogeneous
transformation matrix:

^A T_B =

[ ^A R_B   ^A t_B ]
[   0 0 0      1   ]

where:

- ^A R_B ∈ SO(3) is the orientation of frame {B} relative to frame {A};
- ^A t_B ∈ R^3 is the position of the origin of {B} expressed in {A}.

A valid rotation matrix satisfies:

R^T R = I

and:

det(R) = +1

These properties are explicitly checked by the transformation utilities.

---

## 9. Point Transformation

For a Cartesian point:

^B p = [x, y, z]^T

its homogeneous representation is:

^B p_h = [x, y, z, 1]^T

The corresponding point expressed in frame {A} is:

^A p_h = ^A T_B · ^B p_h

The Cartesian coordinates are obtained from the first three elements of the
result.

---

## 10. Transform Composition

Multiple transformations may be chained through matrix multiplication.

For example:

^W T_T = ^W T_B · ^B T_T

This maps the tool pose from the robot-base frame into the world frame.

Transform order is therefore significant and will not be treated as
commutative.

---

## 11. Transform Inversion

For the rigid transformation:

T = [ R  t ]
    [ 0  1 ]

the inverse is:

T^-1 = [ R^T   -R^T t ]
       [  0        1   ]

The implementation uses this rigid-body structure rather than a generic matrix
inverse where appropriate.

---

## 12. Design Rationale

Explicit coordinate-frame management was selected to prevent implicit mixing
of coordinates originating from different subsystems.

This becomes particularly important when integrating:

- camera-based localisation;
- active camera viewpoint changes;
- robotic instrument kinematics;
- motion planning;
- anatomical geometry;
- uncertainty propagation;
- collision detection;
- ground-truth safety evaluation.

Maintaining explicit transformations also makes geometric assumptions easier
to inspect, test and reproduce.

---

## 13. Verification Strategy

The coordinate-transformation implementation will be verified using
analytically known cases including:

1. identity transformations;
2. translation-only transformations;
3. rotation-only transformations;
4. transformation composition;
5. rigid-transform inversion;
6. point-transformation round trips;
7. rejection of invalid rotation matrices;
8. rejection of malformed homogeneous transformations.

For a valid rigid transform T and point p, the round-trip property should
satisfy:

T^-1(Tp) ≈ p

within a predefined numerical tolerance.

Similarly:

T^-1 T ≈ I

provides an independent consistency check.

---

## 14. Scope and Limitations

The coordinate frames defined here are computational conventions for the
simulation framework.

They do not represent the coordinate conventions of any specific commercial
surgical robotic platform.

Additional frame definitions may be introduced later if required by the final
simulation architecture, but any addition must be explicitly documented and
connected to the existing transformation chain.
