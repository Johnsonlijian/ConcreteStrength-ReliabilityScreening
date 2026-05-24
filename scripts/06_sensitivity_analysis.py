from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtri


VARIANT_SOURCE = {
    "fixed_fc": "concrete_strength",
    "fixed_fy": "steel_strength",
    "fixed_geometry": "geometry",
    "fixed_load": "load_effect",
    "fixed_model_error": "resistance_model_error",
}


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def load_mc_module(script_dir: Path):
    path = script_dir / "04_monte_carlo_reliability.py"
    spec = importlib.util.spec_from_file_location("round6_mc", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Monte Carlo helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def deterministic_fc(cases: pd.DataFrame, n_mc: int) -> tuple[np.ndarray, np.ndarray]:
    pred = cases["predicted_fc"].astype(float).to_numpy()
    return np.repeat(pred[:, None], n_mc, axis=1), np.zeros(len(cases), dtype=float)


def simulate_variant(
    mc,
    rng: np.random.Generator,
    chunk: pd.DataFrame,
    n_mc: int,
    residual_pools: dict[tuple[str, int], np.ndarray],
    variant: str,
) -> pd.DataFrame:
    n = len(chunk)
    if variant == "fixed_fc":
        fc, fc_sampling_invalid = deterministic_fc(chunk, n_mc)
    else:
        fc, fc_sampling_invalid = mc.sample_fc(rng, chunk, n_mc, residual_pools)

    fy_mean = chunk["fy_mean_mpa"].astype(float).to_numpy()
    fy_cov = chunk["fy_cov"].astype(float).to_numpy()
    geometry_cov = chunk["geometry_cov"].astype(float).to_numpy()
    load_cov = chunk["load_cov"].astype(float).to_numpy()
    model_error_cov = chunk["model_error_cov"].astype(float).to_numpy()

    if variant == "fixed_fy":
        fy = np.repeat(fy_mean[:, None], n_mc, axis=1)
    else:
        fy = mc.lognormal_samples_from_mean_cov(rng, fy_mean, fy_cov, (n, n_mc))

    if variant == "fixed_geometry":
        geom_1 = np.ones((n, n_mc))
        geom_2 = np.ones((n, n_mc))
        geom_3 = np.ones((n, n_mc))
    else:
        geom_1 = mc.lognormal_samples_from_mean_cov(rng, np.ones(n), geometry_cov, (n, n_mc))
        geom_2 = mc.lognormal_samples_from_mean_cov(rng, np.ones(n), geometry_cov, (n, n_mc))
        geom_3 = mc.lognormal_samples_from_mean_cov(rng, np.ones(n), geometry_cov, (n, n_mc))

    if variant == "fixed_model_error":
        model_error = np.ones((n, n_mc))
    else:
        model_error = mc.lognormal_samples_from_mean_cov(
            rng, np.ones(n), model_error_cov, (n, n_mc)
        )

    resistance = np.zeros((n, n_mc), dtype=float)
    valid = np.ones((n, n_mc), dtype=bool)
    member_types = chunk["member_type"].astype(str).to_numpy()

    beam_idx = np.where(member_types == "rc_beam_flexure")[0]
    if len(beam_idx):
        b = chunk.iloc[beam_idx]["b_mm"].astype(float).to_numpy()[:, None] * geom_1[beam_idx]
        d = chunk.iloc[beam_idx]["d_mm"].astype(float).to_numpy()[:, None] * geom_2[beam_idx]
        as_area = (
            chunk.iloc[beam_idx]["As_mm2"].astype(float).to_numpy()[:, None]
            * geom_3[beam_idx]
        )
        phi = chunk.iloc[beam_idx]["phi"].astype(float).to_numpy()[:, None]
        r, ok = mc.beam_design_resistance(fc[beam_idx], fy[beam_idx], b, d, as_area, phi)
        resistance[beam_idx] = r * model_error[beam_idx]
        valid[beam_idx] = ok

    col_idx = np.where(member_types == "rc_short_column_axial")[0]
    if len(col_idx):
        b = chunk.iloc[col_idx]["b_mm"].astype(float).to_numpy()[:, None] * geom_1[col_idx]
        h = chunk.iloc[col_idx]["h_mm"].astype(float).to_numpy()[:, None] * geom_2[col_idx]
        ag = b * h
        ast = (
            chunk.iloc[col_idx]["steel_area_mm2"].astype(float).to_numpy()[:, None]
            * geom_3[col_idx]
        )
        phi = chunk.iloc[col_idx]["phi"].astype(float).to_numpy()[:, None]
        r, ok = mc.column_design_resistance(fc[col_idx], fy[col_idx], ag, ast, phi)
        resistance[col_idx] = r * model_error[col_idx]
        valid[col_idx] = ok

    base_r = mc.deterministic_design_resistance(chunk)
    demand_mean = np.maximum(
        base_r * chunk["demand_ratio"].astype(float).to_numpy(),
        1e-9,
    )
    if variant == "fixed_load":
        load = np.repeat(demand_mean[:, None], n_mc, axis=1)
    else:
        load = mc.lognormal_samples_from_mean_cov(rng, demand_mean, load_cov, (n, n_mc))

    g = resistance - load
    failures = (g <= 0).sum(axis=1)
    pf_empirical = failures / n_mc
    pf_cc = (failures + 0.5) / (n_mc + 1.0)
    beta_cc = -ndtri(pf_cc)

    out = chunk[
        [
            "member_id",
            "member_type",
            "mix_id",
            "seed",
            "split_protocol",
            "uncertainty_mode",
            "strength_group",
            "age_group",
            "scm_group",
            "demand_ratio",
            "rho",
            "fy_mean_mpa",
        ]
    ].copy()
    out["variant"] = variant
    out["n_mc"] = n_mc
    out["pf_empirical"] = pf_empirical
    out["beta"] = beta_cc
    out["mean_R"] = resistance.mean(axis=1)
    out["std_R"] = resistance.std(axis=1, ddof=1)
    out["mean_S"] = load.mean(axis=1)
    out["std_S"] = load.std(axis=1, ddof=1)
    out["invalid_resistance_sample_rate"] = 1.0 - valid.mean(axis=1)
    out["fc_sampling_invalid"] = fc_sampling_invalid
    return out


def build_cases(mc, project_root: Path, rng: np.random.Generator, args) -> pd.DataFrame:
    out_dir = project_root / "ai_autoboost" / "outputs" / "round6_structural_reliability"
    ml = pd.read_csv(out_dir / "ml_uncertainty_inputs.csv")
    members = pd.read_csv(out_dir / "member_population.csv")
    selected_ml = mc.select_ml_inputs(ml, args.max_base_rows_per_protocol, rng)
    selected_ml = selected_ml[
        selected_ml["uncertainty_mode"].isin(
            [
                "M2_empirical_residual_90_by_seed",
                "M4_split_conformal_90_envelope",
            ]
        )
    ].copy()
    selected_members = mc.select_members(members, args.members_per_type, rng)
    return selected_ml.merge(selected_members, how="cross").reset_index(drop=True)


def build_contribution_tables(all_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    key_cols = [
        "member_id",
        "member_type",
        "mix_id",
        "seed",
        "split_protocol",
        "uncertainty_mode",
        "demand_ratio",
        "strength_group",
        "age_group",
        "scm_group",
    ]
    base = all_results[all_results["variant"] == "all_uncertainty"][
        key_cols + ["pf_empirical", "beta"]
    ].rename(columns={"pf_empirical": "pf_all", "beta": "beta_all"})
    parts = []
    for variant, source in VARIANT_SOURCE.items():
        fixed = all_results[all_results["variant"] == variant][
            key_cols + ["pf_empirical", "beta"]
        ].rename(columns={"pf_empirical": "pf_fixed", "beta": "beta_fixed"})
        merged = base.merge(fixed, on=key_cols, how="inner")
        merged["source"] = source
        merged["delta_beta_fixed_minus_all"] = merged["beta_fixed"] - merged["beta_all"]
        merged["delta_pf_all_minus_fixed"] = merged["pf_all"] - merged["pf_fixed"]
        parts.append(merged)
    detail = pd.concat(parts, ignore_index=True)

    group_cols = ["member_type", "split_protocol", "uncertainty_mode", "demand_ratio", "source"]
    rows = []
    for keys, group in detail.groupby(group_cols):
        rows.append(
            {
                "member_type": keys[0],
                "split_protocol": keys[1],
                "uncertainty_mode": keys[2],
                "demand_ratio": keys[3],
                "source": keys[4],
                "n_cases": int(len(group)),
                "delta_beta_mean": float(group["delta_beta_fixed_minus_all"].mean()),
                "delta_beta_median": float(group["delta_beta_fixed_minus_all"].median()),
                "delta_pf_mean": float(group["delta_pf_all_minus_fixed"].mean()),
                "delta_pf_median": float(group["delta_pf_all_minus_fixed"].median()),
            }
        )
    summary = pd.DataFrame(rows).sort_values(
        ["member_type", "uncertainty_mode", "demand_ratio", "delta_beta_mean"],
        ascending=[True, True, True, False],
    )
    return detail, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=project_root_from_script())
    parser.add_argument("--n-mc", type=int, default=1500)
    parser.add_argument("--max-base-rows-per-protocol", type=int, default=60)
    parser.add_argument("--members-per-type", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260524)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    script_dir = Path(__file__).resolve().parent
    mc = load_mc_module(script_dir)
    rng = np.random.default_rng(args.seed)

    out_dir = project_root / "ai_autoboost" / "outputs" / "round6_structural_reliability"
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    cases = build_cases(mc, project_root, rng, args)
    ml = pd.read_csv(out_dir / "ml_uncertainty_inputs.csv")
    residual_pools = mc.make_residual_pools(ml)

    variants = [
        "all_uncertainty",
        "fixed_fc",
        "fixed_fy",
        "fixed_geometry",
        "fixed_load",
        "fixed_model_error",
    ]
    outputs = []
    for variant in variants:
        chunks = []
        # Use a deterministic per-variant seed so each variant is reproducible.
        variant_seed = args.seed + 101 * variants.index(variant)
        variant_rng = np.random.default_rng(variant_seed)
        for start in range(0, len(cases), args.chunk_size):
            chunk = cases.iloc[start : start + args.chunk_size].copy()
            chunks.append(
                simulate_variant(
                    mc,
                    variant_rng,
                    chunk,
                    n_mc=args.n_mc,
                    residual_pools=residual_pools,
                    variant=variant,
                )
            )
        outputs.append(pd.concat(chunks, ignore_index=True))

    all_results = pd.concat(outputs, ignore_index=True)
    detail, summary = build_contribution_tables(all_results)

    all_results.to_csv(out_dir / "sensitivity_variant_results.csv", index=False)
    detail.to_csv(out_dir / "sensitivity_contribution_details.csv", index=False)
    summary.to_csv(out_dir / "sensitivity_results.csv", index=False)

    overall = (
        detail.groupby("source")
        .agg(
            n_cases=("source", "size"),
            delta_beta_mean=("delta_beta_fixed_minus_all", "mean"),
            delta_beta_median=("delta_beta_fixed_minus_all", "median"),
            delta_pf_mean=("delta_pf_all_minus_fixed", "mean"),
            delta_pf_median=("delta_pf_all_minus_fixed", "median"),
        )
        .reset_index()
        .sort_values("delta_beta_mean", ascending=False)
    )
    overall.to_csv(table_dir / "sensitivity_overall_by_source.csv", index=False)

    manifest = pd.DataFrame(
        [
            {
                "output": "sensitivity_variant_results.csv",
                "rows": len(all_results),
                "n_mc": args.n_mc,
                "max_base_rows_per_protocol": args.max_base_rows_per_protocol,
                "members_per_type": args.members_per_type,
                "seed": args.seed,
            },
            {
                "output": "sensitivity_results.csv",
                "rows": len(summary),
                "n_mc": args.n_mc,
                "max_base_rows_per_protocol": args.max_base_rows_per_protocol,
                "members_per_type": args.members_per_type,
                "seed": args.seed,
            },
            {
                "output": "tables/sensitivity_overall_by_source.csv",
                "rows": len(overall),
                "n_mc": args.n_mc,
                "max_base_rows_per_protocol": args.max_base_rows_per_protocol,
                "members_per_type": args.members_per_type,
                "seed": args.seed,
            },
        ]
    )
    manifest.to_csv(out_dir / "sensitivity_manifest.csv", index=False)
    print(manifest.to_string(index=False))
    print("\nOverall sensitivity:")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()

