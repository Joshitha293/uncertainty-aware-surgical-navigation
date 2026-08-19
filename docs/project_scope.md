# Project Scope

## Project Title

Uncertainty-Aware Active Perception for Safety-Critical Motion Planning in
Minimally Invasive Surgical Robotics

## Project Purpose

The purpose of this project is to develop a reproducible computational
research framework for investigating how perception uncertainty affects
downstream motion planning in a simplified minimally invasive surgical
robotics environment.

The project specifically examines whether active viewpoint selection can
reduce uncertainty before motion planning and whether conditioning sensing
decisions on task relevance can improve the safety–efficiency trade-off.

## Core Engineering Objective

To design, implement and evaluate a simulation pipeline that links:

- surgical scene simulation;
- camera-based observation;
- target and critical-structure localisation;
- perception uncertainty;
- task relevance;
- active viewpoint selection;
- motion planning;
- safety-margin evaluation;
- collision detection;
- quantitative experimental analysis.

## Experimental Comparison

Three perception strategies will be compared under matched simulated
conditions.

### 1. Fixed-View Perception

The system plans using information obtained from a fixed camera viewpoint
without intentionally acquiring an additional observation.

### 2. Generic Active Perception

The system may acquire an additional viewpoint according to expected
reduction in uncertainty across the scene, without weighting uncertainty
according to the upcoming trajectory.

### 3. Task-Aware Active Perception

The system may acquire an additional viewpoint according to expected
uncertainty reduction weighted by the relevance of each structure to the
planned trajectory and associated safety-critical regions.

## Primary Research Question

Can task-aware active perception improve the safety–efficiency trade-off of
motion planning under simulated perception uncertainty compared with fixed-view
and task-agnostic active perception?

## Primary Outcome

The primary outcome will be simulated safety-margin violation rate.

A safety-margin violation occurs when the planned or executed trajectory enters
a predefined protected region surrounding a critical simulated anatomical
structure.

## Secondary Outcomes

Secondary outcomes may include:

- simulated geometric collision rate;
- minimum surface clearance;
- minimum safety clearance;
- target localisation error;
- critical-structure localisation error;
- mean scene localisation error;
- uncertainty reduction;
- task-relevant uncertainty reduction;
- number of additional viewpoints;
- camera displacement;
- path length;
- planning time;
- task-completion success.

## Experimental Disturbances

The system will support controlled manipulation of:

- localisation noise;
- visual occlusion;
- viewpoint-dependent observation quality;
- selected visual degradation conditions where feasible.

These disturbances will be introduced using predefined parameter levels so
that methods can be compared under matched conditions.

## Scope Boundaries

The project is intentionally restricted to a computational proof-of-concept.

It will not claim to reproduce the complete physical or clinical complexity of
robot-assisted minimally invasive surgery.

The core project will not require:

- patient data;
- animal or cadaver experiments;
- physical surgical robotic hardware;
- autonomous cutting or suturing;
- force or tactile sensing;
- deformable tissue modelling;
- clinical validation;
- medical-device certification;
- regulatory approval.

Any later use of university robotics or imaging equipment will be treated as an
optional validation extension rather than a requirement for completion of the
core project.

## Engineering Assumptions

The initial simulation will assume that:

- simulator ground-truth geometry is known;
- robot and camera states are available within the simulator;
- critical structures can be represented using simplified geometry;
- the target is known or estimable;
- camera viewpoints are controllable;
- perception degradation can be introduced systematically;
- uncertainty can be represented computationally;
- geometric safety measures are sufficient for comparative evaluation.

## Interpretation Boundary

All reported collision, clearance and safety-margin outcomes will be interpreted
only as properties of the simulated environment.

They must not be interpreted as estimates of patient risk, clinical safety,
surgical complication probability, or acceptable medical-device performance.
