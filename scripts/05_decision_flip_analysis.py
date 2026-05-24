from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PAIR_DEFINITIONS = [
    {
        "comparison_name": "group_point_to_group_empirical",
        "reference_split_protocol": "group_aware_round3",
        "reference_uncertainty_mode": "M1_deterministic_point_prediction",
        "comparison_split_protocol": "group_aware_round3",
        "comparison_uncertainty_mode": "M2_empirical_residual_90_by_seed",
    },
    {
        "comparison_name": "group_point_to_group_conformal",
        "reference_split_protocol": "group_aware_round3",
        "reference_uncertainty_mode": "M1_deterministic_point_prediction",
        "comparison_split_protocol": "group_aware_round3",
        "comparison_uncertainty_mode": "M4_split_conformal_90_envelope",
    },
    {
        "comparison_name": "random_point_to_random_empirical",
        "reference_split_protocol": "random_split_round1",
        "reference_uncertainty_mode": "M1_deterministic_point_prediction",
        "comparison_split_protocol": "random_split_round1",
        "comparison_uncertainty_mode": "M2_empirical_residual_90_by_seed",
    },
    {
        "comparison_name": "random_empirical_to_group_empirical",
        "reference_split_protocol": "random_split_round1",
        "reference_uncertainty_mode": "M2_empirical_residual_90_by_seed",
        "comparison_split_protocol": "group_aware_round3",
        "comparison_uncertainty_mode": "M2_empirical_residual_90_by_seed",
    },
    {
        "comparison_name": "random_point_to_group_conformal",
        "reference_split_protocol": "random_split_round1",
        "reference_uncertainty_mode": "M1_deterministic_point_prediction",
        "comparison_split_protocol": "group_aware_round3",
        "comparison_uncertainty_mode": "M4_split_conformal_90_envelope",
    },
]


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def prepare_subset(df: pd.DataFrame, split_protocol: str, uncertainty_mode: str, suffix: str) -> pd.DataFrame:
    subset = df[
        (df["split_protocol"] == split_protocol)
        & (df["uncertainty_mode"] == uncertainty_mode)
    ].copy()
    keep = [
        "member_id",
        "member_type",
        "mix_id",
        "seed",
        "demand_ratio",
        "strength_group",
        "age_group",
        "scm_group",
        "pf_empirical",
        "beta",
        "decision_status",
        "mean_R",
        "std_R",
        "mean_S",
        "std_S",
        "fc_sample_mean",
        "fc_sample_std",
        "interval_width",
    ]
    subset = subset[keep]
    rename = {
        col: f"{col}_{suffix}"
        for col in [
            "pf_empirical",
            "beta",
            "decision_status",
            "mean_R",
            "std_R",
            "mean_S",
            "std_S",
            "fc_sample_mean",
            "fc_sample_std",
            "interval_width",
        ]
    }
    return subset.rename(columns=rename)


def build_pair_detail(df: pd.DataFrame, definition: dict) -> pd.DataFrame:
    ref = prepare_subset(
        df,
        definition["reference_split_protocol"],
        definition["reference_uncertainty_mode"],
        "ref",
    )
    comp = prepare_subset(
        df,
        definition["comparison_split_protocol"],
        definition["comparison_uncertainty_mode"],
        "comp",
    )
    join_cols = [
        "member_id",
        "member_type",
        "mix_id",
        "seed",
        "demand_ratio",
        "strength_group",
        "age_group",
        "scm_group",
    ]
    merged = ref.merge(comp, on=join_cols, how="inner")
    if merged.empty:
        return merged
    merged["comparison_name"] = definition["comparison_name"]
    merged["reference_mode"] = (
        definition["reference_split_protocol"]
        + "::"
        + definition["reference_uncertainty_mode"]
    )
    merged["comparison_mode"] = (
        definition["comparison_split_protocol"]
        + "::"
        + definition["comparison_uncertainty_mode"]
    )
    merged["delta_beta_comp_minus_ref"] = merged["beta_comp"] - merged["beta_ref"]
    merged["delta_pf_comp_minus_ref"] = merged["pf_empirical_comp"] - merged["pf_empirical_ref"]
    merged["pf_ratio_comp_over_ref"] = (merged["pf_empirical_comp"] + 1e-6) / (
        merged["pf_empirical_ref"] + 1e-6
    )
    merged["safe_to_unsafe"] = (
        (merged["decision_status_ref"] == "acceptable_beta_screen")
        & (merged["decision_status_comp"] == "below_beta_screen")
    )
    merged["unsafe_to_safe"] = (
        (merged["decision_status_ref"] == "below_beta_screen")
        & (merged["decision_status_comp"] == "acceptable_beta_screen")
    )
    merged["decision_changed"] = merged["decision_status_ref"] != merged["decision_status_comp"]
    return merged


def summarize_flips(detail: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "comparison_name",
        "member_type",
        "demand_ratio",
        "strength_group",
        "age_group",
    ]
    rows = []
    for keys, group in detail.groupby(group_cols):
        rows.append(
            {
                "comparison_name": keys[0],
                "member_type": keys[1],
                "demand_ratio": keys[2],
                "strength_group": keys[3],
                "age_group": keys[4],
                "n_pairs": int(len(group)),
                "flip_rate_safe_to_unsafe": float(group["safe_to_unsafe"].mean()),
                "flip_rate_unsafe_to_safe": float(group["unsafe_to_safe"].mean()),
                "decision_change_rate": float(group["decision_changed"].mean()),
                "delta_beta_mean": float(group["delta_beta_comp_minus_ref"].mean()),
                "delta_beta_median": float(group["delta_beta_comp_minus_ref"].median()),
                "delta_pf_mean": float(group["delta_pf_comp_minus_ref"].mean()),
                "pf_ratio_median": float(group["pf_ratio_comp_over_ref"].median()),
                "beta_ref_median": float(group["beta_ref"].median()),
                "beta_comp_median": float(group["beta_comp"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["comparison_name", "decision_change_rate", "n_pairs"],
        ascending=[True, False, False],
    )


def summarize_overall(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comparison_name, group in detail.groupby("comparison_name"):
        rows.append(
            {
                "comparison_name": comparison_name,
                "n_pairs": int(len(group)),
                "safe_to_unsafe_count": int(group["safe_to_unsafe"].sum()),
                "unsafe_to_safe_count": int(group["unsafe_to_safe"].sum()),
                "decision_changed_count": int(group["decision_changed"].sum()),
                "flip_rate_safe_to_unsafe": float(group["safe_to_unsafe"].mean()),
                "flip_rate_unsafe_to_safe": float(group["unsafe_to_safe"].mean()),
                "decision_change_rate": float(group["decision_changed"].mean()),
                "delta_beta_mean": float(group["delta_beta_comp_minus_ref"].mean()),
                "delta_beta_median": float(group["delta_beta_comp_minus_ref"].median()),
                "delta_pf_mean": float(group["delta_pf_comp_minus_ref"].mean()),
                "pf_ratio_median": float(group["pf_ratio_comp_over_ref"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values("decision_change_rate", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=project_root_from_script())
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    out_dir = project_root / "ai_autoboost" / "outputs" / "round6_structural_reliability"
    table_dir = out_dir / "tables"
    results_path = out_dir / "reliability_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(
            f"Missing {results_path}. Run 04_monte_carlo_reliability.py first."
        )

    df = pd.read_csv(results_path)
    detail_parts = []
    for definition in PAIR_DEFINITIONS:
        detail = build_pair_detail(df, definition)
        if not detail.empty:
            detail_parts.append(detail)

    if not detail_parts:
        raise RuntimeError("No comparable reliability result pairs were found.")

    detail_all = pd.concat(detail_parts, ignore_index=True)
    summary = summarize_flips(detail_all)
    overall = summarize_overall(detail_all)

    detail_all.to_csv(out_dir / "decision_flip_pair_details.csv", index=False)
    summary.to_csv(out_dir / "decision_flip_results.csv", index=False)
    overall.to_csv(table_dir / "decision_flip_overall_summary.csv", index=False)

    manifest = pd.DataFrame(
        [
            {
                "output": "decision_flip_pair_details.csv",
                "rows": len(detail_all),
            },
            {
                "output": "decision_flip_results.csv",
                "rows": len(summary),
            },
            {
                "output": "tables/decision_flip_overall_summary.csv",
                "rows": len(overall),
            },
        ]
    )
    manifest.to_csv(out_dir / "decision_flip_manifest.csv", index=False)

    print(manifest.to_string(index=False))
    print("\nOverall decision flip summary:")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()

