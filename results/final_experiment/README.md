# Final Experiment Results

This directory contains outputs from the definitive matched comparison between:

1. fixed-view perception;
2. generic active perception;
3. task-aware active perception.

## Directory Structure

### `raw/`

Immutable machine-generated per-trial experimental records.

Raw files should not be manually edited after generation.

### `processed/`

Derived datasets and statistical-analysis tables generated from the raw results.

### `figures/`

Programmatically generated figures derived from raw or processed experimental data.

### `metadata/`

Experiment configuration, random seeds, software revision, environment information, and protocol version.

### `summary/`

Aggregate human-readable and machine-readable summaries of the final experiment.

## Provenance

Every definitive experimental campaign should record:

- Git commit hash;
- experiment identifier;
- execution timestamp;
- Python/environment information;
- number of matched scenarios;
- strategy definitions;
- random-seed policy;
- uncertainty configuration;
- planning configuration.

The Git revision should be obtained using:

```bash
git rev-parse HEAD