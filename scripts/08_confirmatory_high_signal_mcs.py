from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

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
]


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def load_round6_mcs_module():
    script_path = Path(__file__).with_name("04_monte_carlo_reliability.py")
    spec = importlib.util.spec_from_file_location("round6_mcs", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_high_signal_ml(
    ml: pd.DataFrame,
    max_base_rows_per_protocol: int,
    rng,
) -> pd.DataFrame:
    base = ml[ml["uncertainty_mode"] == "M1_deterministic_point_prediction"].copy()
    high_signal = base[
        (base["strength_group"] == "low_<25MPa")
        | (base["age_group"] == "early_1_7d")
    ].copy()

    if high_signal.empty:
        raise RuntimeError("No low-strength or early-age base ML rows were found.")

    key_cols = [
        "split_protocol",
        "mix_id",
        "seed",
        "predicted_fc",
        "actual_fc",
        "strength_group",
        "age_group",
        "scm_group",
    ]

    selected_keys = []
    for _, group in high_signal[key_cols].drop_duplicates().groupby("split_protocol"):
        selected_keys.append(
            mcs.stratified_sample(
                group,
                ["strength_group", "age_group", "scm_group"],
                max_base_rows_per_protocol,
                rng,
            )
        )
    keys = pd.concat(selected_keys, ignore_index=True)
    selected = ml.merge(
        keys[["split_protocol", "mix_id", "seed", "predicted_fc"]],
        on=["split_protocol", "mix_id", "seed", "predicted_fc"],
        how="inner",
    )
    return selected.reset_index(drop=True)


def select_high_signal_members(members: pd.DataFrame) -> pd.DataFrame:
    selected = members[
        (members["member_type"] == "rc_short_column_axial")
        & (members["demand_ratio"].astype(float) == 0.60)
    ].copy()
    if selected.empty:
        raise RuntimeError("No RC short-column members at demand ratio 0.60 were found.")
    return selected.reset_index(drop=True)


def prepare_subset(
    df: pd.DataFrame,
    split_protocol: str,
    uncertainty_mode: str,
    suffix: str,
) -> pd.DataFrame:
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
        "failure_count",
        "fc_sample_mean",
        "fc_sample_std",
        "interval_width",
        "n_mc",
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
            "failure_count",
            "fc_sample_mean",
            "fc_sample_std",
            "interval_width",
            "n_mc",
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
    merged["delta_pf_comp_minus_ref"] = (
        merged["pf_empirical_comp"] - merged["pf_empirical_ref"]
    )
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
    merged["decision_changed"] = (
        merged["decision_status_ref"] != merged["decision_status_comp"]
    )
    return merged


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
                "beta_ref_median": float(group["beta_ref"].median()),
                "beta_comp_median": float(group["beta_comp"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values("decision_change_rate", ascending=False)


def summarize_by_group(detail: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "comparison_name",
        "strength_group",
        "age_group",
        "scm_group",
    ]
    rows = []
    for keys, group in detail.groupby(group_cols):
        rows.append(
            {
                "comparison_name": keys[0],
                "strength_group": keys[1],
                "age_group": keys[2],
                "scm_group": keys[3],
                "n_pairs": int(len(group)),
                "safe_to_unsafe_count": int(group["safe_to_unsafe"].sum()),
                "unsafe_to_safe_count": int(group["unsafe_to_safe"].sum()),
                "decision_changed_count": int(group["decision_changed"].sum()),
                "flip_rate_safe_to_unsafe": float(group["safe_to_unsafe"].mean()),
                "flip_rate_unsafe_to_safe": float(group["unsafe_to_safe"].mean()),
                "decision_change_rate": float(group["decision_changed"].mean()),
                "delta_beta_mean": float(group["delta_beta_comp_minus_ref"].mean()),
                "delta_pf_mean": float(group["delta_pf_comp_minus_ref"].mean()),
                "beta_ref_median": float(group["beta_ref"].median()),
                "beta_comp_median": float(group["beta_comp"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["comparison_name", "decision_change_rate", "n_pairs"],
        ascending=[True, False, False],
    )


def summarize_reliability_focus(results: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["split_protocol", "uncertainty_mode", "strength_group", "age_group"]
    rows = []
    for keys, group in results.groupby(group_cols):
        rows.append(
            {
                "split_protocol": keys[0],
                "uncertainty_mode": keys[1],
                "strength_group": keys[2],
                "age_group": keys[3],
                "n_cases": int(len(group)),
                "pf_median": float(group["pf_empirical"].median()),
                "pf_p90": float(group["pf_empirical"].quantile(0.90)),
                "beta_median": float(group["beta"].median()),
                "beta_p10": float(group["beta"].quantile(0.10)),
                "below_beta_rate": float(
                    (group["decision_status"] == "below_beta_screen").mean()
                ),
                "failure_count_median": float(group["failure_count"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["split_protocol", "uncertainty_mode", "strength_group", "age_group"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=project_root_from_script())
    parser.add_argument("--n-mc", type=int, default=10000)
    parser.add_argument("--max-base-rows-per-protocol", type=int, default=80)
    parser.add_argument("--chunk-size", type=int, default=96)
    parser.add_argument("--target-beta", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260524)
    args = parser.parse_args()

    global mcs
    mcs = load_round6_mcs_module()
    rng = mcs.np.random.default_rng(args.seed)

    project_root = args.project_root.resolve()
    out_dir = project_root / "ai_autoboost" / "outputs" / "round6_structural_reliability"
    table_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    ml = pd.read_csv(out_dir / "ml_uncertainty_inputs.csv")
    members = pd.read_csv(out_dir / "member_population.csv")

    selected_ml = select_high_signal_ml(ml, args.max_base_rows_per_protocol, rng)
    selected_members = select_high_signal_members(members)
    cases = selected_ml.merge(selected_members, how="cross").reset_index(drop=True)
    residual_pools = mcs.make_residual_pools(ml)

    result_chunks = []
    for start in range(0, len(cases), args.chunk_size):
        chunk = cases.iloc[start : start + args.chunk_size].copy()
        result_chunks.append(
            mcs.simulate_chunk(
                rng,
                chunk,
                n_mc=args.n_mc,
                residual_pools=residual_pools,
                target_beta=args.target_beta,
            )
        )

    results = pd.concat(result_chunks, ignore_index=True)
    reliability_summary = summarize_reliability_focus(results)

    detail_parts = []
    for definition in PAIR_DEFINITIONS:
        detail = build_pair_detail(results, definition)
        if not detail.empty:
            detail_parts.append(detail)

    if not detail_parts:
        raise RuntimeError("No confirmatory matched pairs were found.")

    pair_detail = pd.concat(detail_parts, ignore_index=True)
    overall = summarize_overall(pair_detail)
    by_group = summarize_by_group(pair_detail)

    results.to_csv(out_dir / "confirmatory_high_signal_reliability_results.csv", index=False)
    pair_detail.to_csv(
        out_dir / "confirmatory_high_signal_decision_flip_pair_details.csv",
        index=False,
    )
    overall.to_csv(
        table_dir / "confirmatory_high_signal_decision_flip_overall.csv",
        index=False,
    )
    by_group.to_csv(
        table_dir / "confirmatory_high_signal_decision_flip_by_group.csv",
        index=False,
    )
    reliability_summary.to_csv(
        table_dir / "confirmatory_high_signal_reliability_summary.csv",
        index=False,
    )

    manifest = pd.DataFrame(
        [
            {
                "output": "confirmatory_high_signal_reliability_results.csv",
                "rows": len(results),
                "n_mc": args.n_mc,
                "max_base_rows_per_protocol": args.max_base_rows_per_protocol,
                "selected_member_rows": len(selected_members),
                "target_beta": args.target_beta,
                "seed": args.seed,
            },
            {
                "output": "confirmatory_high_signal_decision_flip_pair_details.csv",
                "rows": len(pair_detail),
                "n_mc": args.n_mc,
                "max_base_rows_per_protocol": args.max_base_rows_per_protocol,
                "selected_member_rows": len(selected_members),
                "target_beta": args.target_beta,
                "seed": args.seed,
            },
            {
                "output": "tables/confirmatory_high_signal_decision_flip_overall.csv",
                "rows": len(overall),
                "n_mc": args.n_mc,
                "max_base_rows_per_protocol": args.max_base_rows_per_protocol,
                "selected_member_rows": len(selected_members),
                "target_beta": args.target_beta,
                "seed": args.seed,
            },
            {
                "output": "tables/confirmatory_high_signal_decision_flip_by_group.csv",
                "rows": len(by_group),
                "n_mc": args.n_mc,
                "max_base_rows_per_protocol": args.max_base_rows_per_protocol,
                "selected_member_rows": len(selected_members),
                "target_beta": args.target_beta,
                "seed": args.seed,
            },
            {
                "output": "tables/confirmatory_high_signal_reliability_summary.csv",
                "rows": len(reliability_summary),
                "n_mc": args.n_mc,
                "max_base_rows_per_protocol": args.max_base_rows_per_protocol,
                "selected_member_rows": len(selected_members),
                "target_beta": args.target_beta,
                "seed": args.seed,
            },
        ]
    )
    manifest.to_csv(out_dir / "confirmatory_high_signal_manifest.csv", index=False)

    print(manifest.to_string(index=False))
    print("\nConfirmatory high-signal decision flip summary:")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
