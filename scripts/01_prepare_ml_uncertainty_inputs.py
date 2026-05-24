from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def classify_age(age: float) -> str:
    if age <= 7:
        return "early_1_7d"
    if age <= 28:
        return "standard_8_28d"
    if age <= 90:
        return "mature_29_90d"
    return "long_91_365d"


def classify_strength(fc: float) -> str:
    if fc < 25:
        return "low_<25MPa"
    if fc < 50:
        return "medium_25_50MPa"
    return "high_>=50MPa"


def classify_scm(slag: float, fly_ash: float) -> str:
    has_slag = slag > 0
    has_fly_ash = fly_ash > 0
    if has_slag and has_fly_ash:
        return "slag_and_flyash"
    if has_slag:
        return "slag_only"
    if has_fly_ash:
        return "flyash_only"
    return "ordinary_no_scm"


def add_group_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age_group"] = out["age"].astype(float).map(classify_age)
    out["strength_group"] = out["actual_fc"].astype(float).map(classify_strength)
    out["scm_group"] = [
        classify_scm(float(slag), float(fly_ash))
        for slag, fly_ash in zip(out["slag"], out["fly_ash"])
    ]
    return out


def build_group_predictions(project_root: Path) -> pd.DataFrame:
    src = project_root / "ai_autoboost" / "outputs" / "round3" / "predictions_with_errors.csv"
    df = pd.read_csv(src)
    out = df.rename(
        columns={
            "row_index": "mix_id",
            "actual": "actual_fc",
            "prediction": "predicted_fc",
            "error": "prediction_error",
        }
    )
    out["split_protocol"] = "group_aware_round3"
    out["model"] = "HGB_Tuned"
    out["residual_actual_minus_predicted"] = out["actual_fc"] - out["predicted_fc"]
    out["source_file"] = str(src.relative_to(project_root))
    keep = [
        "mix_id",
        "seed",
        "split_protocol",
        "model",
        "actual_fc",
        "predicted_fc",
        "prediction_error",
        "residual_actual_minus_predicted",
        "abs_error",
        "feature_group",
        "cement",
        "slag",
        "fly_ash",
        "water",
        "superplasticizer",
        "coarse_aggregate",
        "fine_aggregate",
        "age",
        "age_group",
        "strength_group",
        "scm_group",
        "source_file",
    ]
    return out[keep]


def build_random_hgb_predictions(project_root: Path) -> pd.DataFrame:
    pred_src = project_root / "ai_autoboost" / "outputs" / "round1" / "predictions.csv"
    data_src = project_root / "data" / "processed" / "uci_concrete_clean.csv"
    preds = pd.read_csv(pred_src)
    data = pd.read_csv(data_src).reset_index().rename(columns={"index": "mix_id"})
    hgb = preds[(preds["split"] == "random_split") & (preds["model"] == "HistGradientBoosting")].copy()
    hgb = hgb.rename(
        columns={
            "row_index": "mix_id",
            "actual": "actual_fc",
            "prediction": "predicted_fc",
        }
    )
    hgb["prediction_error"] = hgb["predicted_fc"] - hgb["actual_fc"]
    hgb["residual_actual_minus_predicted"] = hgb["actual_fc"] - hgb["predicted_fc"]
    hgb = hgb.merge(data, on="mix_id", how="left")
    hgb["split_protocol"] = "random_split_round1"
    hgb["source_file"] = str(pred_src.relative_to(project_root))
    hgb = add_group_labels(hgb)
    keep = [
        "mix_id",
        "seed",
        "split_protocol",
        "model",
        "actual_fc",
        "predicted_fc",
        "prediction_error",
        "residual_actual_minus_predicted",
        "abs_error",
        "feature_group",
        "cement",
        "slag",
        "fly_ash",
        "water",
        "superplasticizer",
        "coarse_aggregate",
        "fine_aggregate",
        "age",
        "age_group",
        "strength_group",
        "scm_group",
        "source_file",
    ]
    return hgb[keep]


def residual_quantiles(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split_protocol, seed), group in df.groupby(["split_protocol", "seed"]):
        residual = group["residual_actual_minus_predicted"].astype(float)
        rows.append(
            {
                "split_protocol": split_protocol,
                "seed": seed,
                "residual_q05": float(residual.quantile(0.05)),
                "residual_q50": float(residual.quantile(0.50)),
                "residual_q95": float(residual.quantile(0.95)),
                "residual_abs_q90": float(group["abs_error"].astype(float).quantile(0.90)),
                "residual_abs_q95": float(group["abs_error"].astype(float).quantile(0.95)),
                "n_predictions": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def conformal_by_seed(project_root: Path) -> pd.DataFrame:
    src = project_root / "ai_autoboost" / "outputs" / "round4" / "prediction_interval_results.csv"
    df = pd.read_csv(src)
    return df[["seed", "qhat_abs_error_mpa", "coverage", "mean_interval_width_mpa"]].copy()


def expand_uncertainty_modes(
    base: pd.DataFrame,
    quantiles: pd.DataFrame,
    conformal: pd.DataFrame,
) -> pd.DataFrame:
    merged = base.merge(quantiles, on=["split_protocol", "seed"], how="left")
    merged = merged.merge(conformal, on="seed", how="left")

    rows = []
    common_cols = [
        "mix_id",
        "seed",
        "split_protocol",
        "model",
        "actual_fc",
        "predicted_fc",
        "prediction_error",
        "residual_actual_minus_predicted",
        "abs_error",
        "feature_group",
        "cement",
        "slag",
        "fly_ash",
        "water",
        "superplasticizer",
        "coarse_aggregate",
        "fine_aggregate",
        "age",
        "age_group",
        "strength_group",
        "scm_group",
        "source_file",
    ]

    for _, row in merged.iterrows():
        record = {col: row[col] for col in common_cols}
        pred = float(row["predicted_fc"])

        deterministic = record.copy()
        deterministic.update(
            {
                "uncertainty_mode": "M1_deterministic_point_prediction",
                "lower_90": pred,
                "upper_90": pred,
                "interval_width": 0.0,
                "residual_source": "point_prediction_only",
                "coverage_reference": np.nan,
                "ood_scenario": "",
                "stress_scenario": "",
            }
        )
        rows.append(deterministic)

        empirical = record.copy()
        q05 = float(row["residual_q05"])
        q95 = float(row["residual_q95"])
        empirical.update(
            {
                "uncertainty_mode": "M2_empirical_residual_90_by_seed",
                "lower_90": pred + q05,
                "upper_90": pred + q95,
                "interval_width": q95 - q05,
                "residual_source": f"{row['split_protocol']}_seed_residual_q05_q95",
                "coverage_reference": np.nan,
                "ood_scenario": "",
                "stress_scenario": "",
            }
        )
        rows.append(empirical)

        if row["split_protocol"] == "group_aware_round3" and pd.notna(row["qhat_abs_error_mpa"]):
            qhat = float(row["qhat_abs_error_mpa"])
            conformal_row = record.copy()
            conformal_row.update(
                {
                    "uncertainty_mode": "M4_split_conformal_90_envelope",
                    "lower_90": pred - qhat,
                    "upper_90": pred + qhat,
                    "interval_width": 2.0 * qhat,
                    "residual_source": "round4_split_conformal_qhat_by_seed",
                    "coverage_reference": float(row["coverage"]),
                    "ood_scenario": "",
                    "stress_scenario": "",
                }
            )
            rows.append(conformal_row)

    out = pd.DataFrame(rows)
    out["actual_fc"] = out["actual_fc"].astype(float)
    out["predicted_fc"] = out["predicted_fc"].astype(float)
    out["lower_90"] = out["lower_90"].clip(lower=0.0)
    out["upper_90"] = out["upper_90"].clip(lower=0.0)
    out["interval_width"] = out["upper_90"] - out["lower_90"]
    return out


def scenario_summary(project_root: Path) -> pd.DataFrame:
    rows = []
    for filename, scenario_col, scenario_type in [
        ("ood_results.csv", "scenario", "ood_scenario"),
        ("stress_test_results.csv", "stress_type", "stress_scenario"),
        ("noise_robustness.csv", "noise_level", "noise_scenario"),
    ]:
        src = project_root / "ai_autoboost" / "outputs" / "round4" / filename
        df = pd.read_csv(src)
        for key, group in df.groupby(scenario_col):
            rows.append(
                {
                    "scenario_type": scenario_type,
                    "scenario": str(key),
                    "n": int(len(group)),
                    "mae_mean": float(group["mae"].mean()),
                    "rmse_mean": float(group["rmse"].mean()),
                    "r2_mean": float(group["r2"].mean()),
                    "source_file": str(src.relative_to(project_root)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=project_root_from_script())
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    out_dir = project_root / "ai_autoboost" / "outputs" / "round6_structural_reliability"
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    group_predictions = build_group_predictions(project_root)
    random_predictions = build_random_hgb_predictions(project_root)
    base = pd.concat([group_predictions, random_predictions], ignore_index=True)
    quantiles = residual_quantiles(base)
    conformal = conformal_by_seed(project_root)
    uncertainty_inputs = expand_uncertainty_modes(base, quantiles, conformal)
    scenarios = scenario_summary(project_root)

    uncertainty_inputs.to_csv(out_dir / "ml_uncertainty_inputs.csv", index=False)
    quantiles.to_csv(table_dir / "residual_quantiles_by_seed.csv", index=False)
    scenarios.to_csv(table_dir / "round4_scenario_uncertainty_summary.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "output": "ml_uncertainty_inputs.csv",
                "rows": len(uncertainty_inputs),
                "unique_mixes": uncertainty_inputs["mix_id"].nunique(),
                "uncertainty_modes": ";".join(sorted(uncertainty_inputs["uncertainty_mode"].unique())),
            },
            {
                "output": "tables/residual_quantiles_by_seed.csv",
                "rows": len(quantiles),
                "unique_mixes": "",
                "uncertainty_modes": "",
            },
            {
                "output": "tables/round4_scenario_uncertainty_summary.csv",
                "rows": len(scenarios),
                "unique_mixes": "",
                "uncertainty_modes": "",
            },
        ]
    )
    summary.to_csv(out_dir / "round6_preparation_manifest.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

