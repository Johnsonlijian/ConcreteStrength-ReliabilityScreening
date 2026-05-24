# ConcreteStrength Reliability Screening

This repository is a clean reproducibility package for a computational study on propagating machine-learning uncertainty in concrete compressive-strength prediction to analytical reinforced-concrete reliability-screening decisions.

## Scope

The package supports a restrained reliability-screening demonstration:

- public concrete-strength benchmark as the material-input layer;
- leakage-aware and random-split ML uncertainty inputs;
- empirical residual and conformal interval envelope propagation;
- analytical RC beam and short-column resistance models;
- Monte Carlo reliability screening;
- decision-flip, confidence-interval, beta-threshold, COV-grid robustness, and sensitivity summaries.

This package does not claim to provide a design-code replacement, field-validated structural tool, or new concrete-material dataset.

## What is included

- `scripts/`: Python scripts for uncertainty preparation, member generation, resistance models, reliability simulation, decision-flip analysis, sensitivity analysis, and figure generation.
- `outputs/tables/`: selected derived summary tables used in the manuscript and supplementary information.
- `outputs/figures/`: selected generated figures.
- `DATASETS_AND_LINKS.csv`: data/source registry for public inputs and candidate external datasets.
- `REPRODUCIBLE_RUNBOOK.md`: local execution guide.

## What is excluded

- raw third-party datasets;
- active manuscripts, cover letters, reviewer-response drafts;
- private author, funding, or conflict-of-interest information;
- journal submission files;
- credentials or local environment secrets.

## Intended public repository

Intended GitHub remote:

`https://github.com/Johnsonlijian/ConcreteStrength-ReliabilityScreening`

Repository creation and push are human-only actions.
