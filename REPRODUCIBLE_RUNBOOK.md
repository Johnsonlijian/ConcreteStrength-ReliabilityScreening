# Reproducible Runbook

This runbook describes the intended local execution order. Paths assume the full private project is available. The public package includes code and selected derived outputs, but it does not include raw third-party data.

## Environment

Recommended Python packages:

- numpy
- pandas
- scipy
- matplotlib

## Script order

Run from the private project root:

```powershell
python ai_autoboost\code\round6_structural_reliability\01_prepare_ml_uncertainty_inputs.py --project-root .
python ai_autoboost\code\round6_structural_reliability\02_generate_member_population.py --project-root .
python ai_autoboost\code\round6_structural_reliability\03_resistance_models.py --project-root .
python ai_autoboost\code\round6_structural_reliability\04_monte_carlo_reliability.py --project-root . --n-mc 1000 --max-base-rows-per-protocol 160 --members-per-type 40 --target-beta 3.0
python ai_autoboost\code\round6_structural_reliability\05_decision_flip_analysis.py --project-root .
python ai_autoboost\code\round6_structural_reliability\06_sensitivity_analysis.py --project-root . --n-mc 1500 --max-base-rows-per-protocol 60 --members-per-type 16
python ai_autoboost\code\round6_structural_reliability\07_make_structural_reliability_figures.py --project-root .
python ai_autoboost\code\round6_structural_reliability\08_confirmatory_high_signal_mcs.py --project-root . --n-mc 10000 --max-base-rows-per-protocol 80 --target-beta 3.0
python ai_autoboost\code\round6_structural_reliability\09_make_confirmatory_figures.py --project-root .
python ai_autoboost\code\round6_structural_reliability\10_beta_threshold_sensitivity.py --project-root . --thresholds 2.5 3.0 3.5
python ai_autoboost\code\round6_structural_reliability\12_make_decision_boundary_figures.py --project-root . --beta-threshold 3.0
python ai_autoboost\code\round6_structural_reliability\14_make_framework_schematic.py --project-root .
python ai_autoboost\code\round6_structural_reliability\16_decision_flip_confidence_intervals.py --project-root .
python ai_autoboost\code\round6_structural_reliability\17_cov_grid_robustness.py --project-root . --n-mc 2000 --max-base-rows-per-protocol 40 --target-beta 3.0
```

## Interpretation boundaries

- `beta = 3.0` is a study screening threshold, not a code target.
- Conformal intervals are used as interval envelopes, not full predictive probability densities.
- RC formulas are ACI-type analytical propagation models, not full code-compliant design checks.
- Synthetic members are analytical case-study members, not field validation.
