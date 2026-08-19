# Coordinate Frames and Transformation Conventions

## Purpose

The simulation uses explicit coordinate frames to separate the geometric
representations associated with the environment, camera, robotic system,
instrument and anatomical structures.

A consistent frame convention is required because perception outputs,
robot states and safety geometry may originate in different reference frames.

## Coordinate Frames

### World Frame — {W}

The world frame is the fixed global reference frame for the complete
simulation environment.

Convention:

- +x: right
- +y: forward
- +z: up

All ground-truth anatomical geometry is ultimately represented relative
to this frame.

### Robot Base Frame — {B}

The robot-base frame is attached to the base of the simulated surgical
instrument or manipulator.

Its pose relative to the world frame is represented by:

^W T_B

### Camera Frame — {C}

The camera frame follows the computer-vision convention:

- +x_C: image right
- +y_C: image down
- +z_C: forward along the optical axis

Its pose in the world frame is represented by:

^W T_C

### Tool Frame — {T}

The tool frame is attached to the simulated instrument.

The frame origin is defined at the tool-tip reference point unless
otherwise stated.

Its pose in the world frame is:

^W T_T

### Anatomical Frames — {A_i}

Each simulated anatomical structure may be assigned a local frame {A_i}.

The transform

^W T_Ai

maps geometry represented in the local anatomical frame into the world frame.

## Homogeneous Transform Representation

Rigid transformations are represented using a 4 × 4 homogeneous
transformation matrix:

^A T_B = [ ^A R_B   ^A p_B
            0 0 0      1 ]

where:

- ^A R_B is the rotation of frame B relative to frame A;
- ^A p_B is the position of the origin of B expressed in frame A.

A homogeneous point expressed in frame B is transformed into frame A by:

^A p = ^A T_B ^B p

## Transform Composition

Transforms are composed by matrix multiplication.

For example:

^W T_T = ^W T_B ^B T_T

maps the tool pose from robot-base coordinates into world coordinates.

## Transform Inversion

For a rigid transform:

T = [ R p
      0 1 ]

the inverse is:

T^-1 = [ R^T  -R^T p
          0       1 ]

The implementation uses this rigid-transform structure rather than a generic
matrix inverse where possible.

## Design Rationale

Explicit coordinate-frame handling prevents perception coordinates,
simulator coordinates and robot coordinates from being mixed implicitly.

This is essential for later integration of:

- camera-based localisation;
- robot kinematics;
- trajectory planning;
- safety-distance calculations;
- viewpoint changes;
- ground-truth evaluation.

## Verification

Transform utilities are verified using:

1. identity transformation tests;
2. translation-only transformations;
3. rotation-only transformations;
4. transform composition;
5. inverse consistency;
6. round-trip point transformation.

For a valid rigid transform T and point p:

T^-1 (T p) ≈ p

within numerical tolerance.
