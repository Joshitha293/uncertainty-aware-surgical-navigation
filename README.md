# Uncertainty-Aware Surgical Navigation

A simulation-based research engineering project investigating how perception
uncertainty affects safety-critical motion planning in minimally invasive
surgical robotics.

The project develops a computational framework for evaluating whether active
perception strategies that account for task-relevant uncertainty can improve
navigation safety under imperfect visual information.

## Research Question

Can task-aware uncertainty-driven perception improve the safety and efficiency
of motion planning in simulated minimally invasive surgical environments
compared with fixed-view and task-agnostic active perception?

## Experimental Strategies

The project will compare three perception strategies:

1. **Fixed-view perception** — planning using observations from a fixed camera
   viewpoint without active viewpoint adjustment.

2. **Generic uncertainty-aware active perception** — additional viewpoints are
   selected according to global perception uncertainty.

3. **Task-aware uncertainty-driven active perception** — uncertainty is weighted
   according to its relevance to the planned trajectory and nearby
   safety-critical structures before viewpoint selection.

## Engineering Scope

The research prototype will integrate:

- simulated minimally invasive surgical environments;
- camera-based perception;
- target and critical-structure localisation;
- uncertainty estimation;
- task-relevance modelling;
- active viewpoint selection;
- risk-aware motion planning;
- ground-truth safety evaluation;
- reproducible experimental testing and quantitative analysis.

## Project Status

**In development**

## Safety and Intended Use

This repository contains a simulation-based engineering research prototype.

It is not a medical device, has not undergone clinical validation or medical
device certification, and must not be used for patient monitoring, diagnosis,
treatment, surgical guidance, or clinical decision-making.
