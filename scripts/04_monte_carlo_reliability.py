from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtri


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def stratified_sample(
    df: pd.DataFrame,
    group_cols: list[str],
    max_rows: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.copy()

    groups = list(df.groupby(group_cols, dropna=False))
    per_group = max(1, math.ceil(max_rows / len(groups)))
    chunks = []
    for _, group in groups:
        take = min(per_group, len(group))
        chunks.append(group.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))
    sampled = pd.concat(chunks, ignore_index=True)
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=int(rng.integers(0, 2**31 - 1)))
    return sampled.reset_index(drop=True)


def select_ml_inputs(
    ml: pd.DataFrame,
    max_base_rows_per_protocol: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
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
    base = (
        ml[ml["uncertainty_mode"] == "M1_deterministic_point_prediction"][key_cols]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    selected_keys = []
    for protocol, group in base.groupby("split_protocol"):
        sampled = stratified_sample(
            group,
            ["strength_group", "age_group", "scm_group"],
            max_base_rows_per_protocol,
            rng,
        )
        selected_keys.append(sampled)
    keys = pd.concat(selected_keys, ignore_index=True)

    selected = ml.merge(
        keys[["split_protocol", "mix_id", "seed", "predicted_fc"]],
        on=["split_protocol", "mix_id", "seed", "predicted_fc"],
        how="inner",
    )
    return selected.reset_index(drop=True)


def select_members(
    members: pd.DataFrame,
    members_per_type: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    chunks = []
    for _, group in members.groupby("member_type"):
        chunks.append(
            stratified_sample(
                group,
                ["demand_ratio", "rho", "fy_mean_mpa"],
                members_per_type,
                rng,
            )
        )
    return pd.concat(chunks, ignore_index=True)


def lognormal_samples_from_mean_cov(
    rng: np.random.Generator,
    mean: np.ndarray,
    cov: np.ndarray | float,
    size: tuple[int, int],
) -> np.ndarray:
    cov_arr = np.asarray(cov, dtype=float)
    sigma = np.sqrt(np.log1p(cov_arr**2))
    mu = np.log(np.maximum(mean, 1e-12)) - 0.5 * sigma**2
    return rng.lognormal(mean=mu[:, None], sigma=sigma[:, None], size=size)


def make_residual_pools(ml: pd.DataFrame) -> dict[tuple[str, int], np.ndarray]:
    base = ml[ml["uncertainty_mode"] == "M1_deterministic_point_prediction"].copy()
    pools: dict[tuple[str, int], np.ndarray] = {}
    for (protocol, seed), group in base.groupby(["split_protocol", "seed"]):
        residuals = group["residual_actual_minus_predicted"].astype(float).to_numpy()
        if len(residuals) == 0:
            continue
        pools[(protocol, int(seed))] = residuals
    return pools


def sample_fc(
    rng: np.random.Generator,
    cases: pd.DataFrame,
    n_mc: int,
    residual_pools: dict[tuple[str, int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    n = len(cases)
    pred = cases["predicted_fc"].astype(float).to_numpy()
    lower = cases["lower_90"].astype(float).to_numpy()
    upper = cases["upper_90"].astype(float).to_numpy()
    modes = cases["uncertainty_mode"].astype(str).to_numpy()
    protocols = cases["split_protocol"].astype(str).to_numpy()
    seeds = cases["seed"].astype(int).to_numpy()

    fc = np.empty((n, n_mc), dtype=float)
    fc_sampling_invalid = np.zeros(n, dtype=float)

    for i in range(n):
        mode = modes[i]
        if mode == "M1_deterministic_point_prediction":
            fc[i, :] = pred[i]
        elif mode == "M2_empirical_residual_90_by_seed":
            pool = residual_pools.get((protocols[i], int(seeds[i])))
            if pool is None or len(pool) == 0:
                fc[i, :] = pred[i]
                fc_sampling_invalid[i] = 1.0
            else:
                sampled_residual = rng.choice(pool, size=n_mc, replace=True)
                fc[i, :] = pred[i] + sampled_residual
        elif mode == "M4_split_conformal_90_envelope":
            lo = min(lower[i], upper[i])
            hi = max(lower[i], upper[i])
            fc[i, :] = rng.uniform(lo, hi, size=n_mc)
        else:
            raise ValueError(f"Unsupported uncertainty_mode: {mode}")

    fc = np.clip(fc, 1.0, None)
    return fc, fc_sampling_invalid


def beam_design_resistance(
    fc: np.ndarray,
    fy: np.ndarray,
    b: np.ndarray,
    d: np.ndarray,
    as_area: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    a = as_area * fy / (0.85 * fc * b)
    lever = d - a / 2.0
    valid = (a > 0) & (lever > 0) & (a < d)
    mn_kNm = as_area * fy * lever / 1_000_000.0
    resistance = np.where(valid, phi * mn_kNm, 0.0)
    return resistance, valid


def column_design_resistance(
    fc: np.ndarray,
    fy: np.ndarray,
    ag: np.ndarray,
    ast: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ac = ag - ast
    valid = (fc > 0) & (fy > 0) & (ag > 0) & (ast >= 0) & (ac > 0)
    pn0_kN = (0.85 * fc * ac + fy * ast) / 1000.0
    resistance = np.where(valid, phi * 0.80 * pn0_kN, 0.0)
    return resistance, valid


def deterministic_design_resistance(cases: pd.DataFrame) -> np.ndarray:
    fc = cases["predicted_fc"].astype(float).to_numpy()
    fy = cases["fy_mean_mpa"].astype(float).to_numpy()
    phi = cases["phi"].astype(float).to_numpy()
    out = np.empty(len(cases), dtype=float)

    beam_mask = cases["member_type"].to_numpy() == "rc_beam_flexure"
    if beam_mask.any():
        b = cases.loc[beam_mask, "b_mm"].astype(float).to_numpy()
        d = cases.loc[beam_mask, "d_mm"].astype(float).to_numpy()
        as_area = cases.loc[beam_mask, "As_mm2"].astype(float).to_numpy()
        res, _ = beam_design_resistance(
            fc[beam_mask],
            fy[beam_mask],
            b,
            d,
            as_area,
            phi[beam_mask],
        )
        out[beam_mask] = res

    column_mask = cases["member_type"].to_numpy() == "rc_short_column_axial"
    if column_mask.any():
        ag = cases.loc[column_mask, "Ag_mm2"].astype(float).to_numpy()
        ast = cases.loc[column_mask, "steel_area_mm2"].astype(float).to_numpy()
        res, _ = column_design_resistance(
            fc[column_mask],
            fy[column_mask],
            ag,
            ast,
            phi[column_mask],
        )
        out[column_mask] = res
    return out


def simulate_chunk(
    rng: np.random.Generator,
    chunk: pd.DataFrame,
    n_mc: int,
    residual_pools: dict[tuple[str, int], np.ndarray],
    target_beta: float,
) -> pd.DataFrame:
    n = len(chunk)
    fc, fc_sampling_invalid = sample_fc(rng, chunk, n_mc, residual_pools)

    fy_mean = chunk["fy_mean_mpa"].astype(float).to_numpy()
    fy_cov = chunk["fy_cov"].astype(float).to_numpy()
    geometry_cov = chunk["geometry_cov"].astype(float).to_numpy()
    load_cov = chunk["load_cov"].astype(float).to_numpy()
    model_error_cov = chunk["model_error_cov"].astype(float).to_numpy()

    fy = lognormal_samples_from_mean_cov(rng, fy_mean, fy_cov, (n, n_mc))
    geom_1 = lognormal_samples_from_mean_cov(rng, np.ones(n), geometry_cov, (n, n_mc))
    geom_2 = lognormal_samples_from_mean_cov(rng, np.ones(n), geometry_cov, (n, n_mc))
    geom_3 = lognormal_samples_from_mean_cov(rng, np.ones(n), geometry_cov, (n, n_mc))
    model_error = lognormal_samples_from_mean_cov(
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
        r, ok = beam_design_resistance(fc[beam_idx], fy[beam_idx], b, d, as_area, phi)
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
        r, ok = column_design_resistance(fc[col_idx], fy[col_idx], ag, ast, phi)
        resistance[col_idx] = r * model_error[col_idx]
        valid[col_idx] = ok

    base_r = deterministic_design_resistance(chunk)
    demand_mean = np.maximum(
        base_r * chunk["demand_ratio"].astype(float).to_numpy(),
        1e-9,
    )
    load = lognormal_samples_from_mean_cov(rng, demand_mean, load_cov, (n, n_mc))

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
            "actual_fc",
            "predicted_fc",
            "lower_90",
            "upper_90",
            "interval_width",
            "strength_group",
            "age_group",
            "scm_group",
            "demand_ratio",
            "rho",
            "fy_mean_mpa",
            "resistance_model",
        ]
    ].copy()
    out["n_mc"] = n_mc
    out["target_beta"] = target_beta
    out["mean_R"] = resistance.mean(axis=1)
    out["std_R"] = resistance.std(axis=1, ddof=1)
    out["mean_S"] = load.mean(axis=1)
    out["std_S"] = load.std(axis=1, ddof=1)
    out["pf_empirical"] = pf_empirical
    out["failure_count"] = failures
    out["pf_for_beta_continuity_corrected"] = pf_cc
    out["beta"] = beta_cc
    out["decision_status"] = np.where(
        beta_cc >= target_beta,
        "acceptable_beta_screen",
        "below_beta_screen",
    )
    out["invalid_resistance_sample_rate"] = 1.0 - valid.mean(axis=1)
    out["fc_sampling_invalid"] = fc_sampling_invalid
    out["fc_sample_mean"] = fc.mean(axis=1)
    out["fc_sample_std"] = fc.std(axis=1, ddof=1)
    return out


def summarize_reliability(results: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["member_type", "split_protocol", "uncertainty_mode", "demand_ratio"]
    rows = []
    for keys, group in results.groupby(group_cols):
        rows.append(
            {
                "member_type": keys[0],
                "split_protocol": keys[1],
                "uncertainty_mode": keys[2],
                "demand_ratio": keys[3],
                "n_cases": int(len(group)),
                "pf_median": float(group["pf_empirical"].median()),
                "pf_p90": float(group["pf_empirical"].quantile(0.90)),
                "beta_median": float(group["beta"].median()),
                "beta_p10": float(group["beta"].quantile(0.10)),
                "below_beta_rate": float((group["decision_status"] == "below_beta_screen").mean()),
                "mean_R_median": float(group["mean_R"].median()),
                "std_R_median": float(group["std_R"].median()),
                "mean_S_median": float(group["mean_S"].median()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=project_root_from_script())
    parser.add_argument("--n-mc", type=int, default=1000)
    parser.add_argument("--max-base-rows-per-protocol", type=int, default=160)
    parser.add_argument("--members-per-type", type=int, default=40)
    parser.add_argument("--chunk-size", type=int, default=384)
    parser.add_argument("--target-beta", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260523)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    project_root = args.project_root.resolve()
    out_dir = project_root / "ai_autoboost" / "outputs" / "round6_structural_reliability"
    table_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    ml_path = out_dir / "ml_uncertainty_inputs.csv"
    member_path = out_dir / "member_population.csv"
    if not ml_path.exists():
        raise FileNotFoundError(f"Missing {ml_path}. Run 01_prepare_ml_uncertainty_inputs.py first.")
    if not member_path.exists():
        raise FileNotFoundError(f"Missing {member_path}. Run 02_generate_member_population.py first.")

    ml = pd.read_csv(ml_path)
    members = pd.read_csv(member_path)
    selected_ml = select_ml_inputs(ml, args.max_base_rows_per_protocol, rng)
    selected_members = select_members(members, args.members_per_type, rng)

    cases = selected_ml.merge(selected_members, how="cross")
    cases = cases.reset_index(drop=True)
    residual_pools = make_residual_pools(ml)

    result_chunks = []
    for start in range(0, len(cases), args.chunk_size):
        chunk = cases.iloc[start : start + args.chunk_size].copy()
        result_chunks.append(
            simulate_chunk(
                rng,
                chunk,
                n_mc=args.n_mc,
                residual_pools=residual_pools,
                target_beta=args.target_beta,
            )
        )

    results = pd.concat(result_chunks, ignore_index=True)
    summary = summarize_reliability(results)

    results.to_csv(out_dir / "reliability_results.csv", index=False)
    summary.to_csv(table_dir / "reliability_summary_by_mode.csv", index=False)

    manifest = pd.DataFrame(
        [
            {
                "output": "reliability_results.csv",
                "rows": len(results),
                "n_mc": args.n_mc,
                "max_base_rows_per_protocol": args.max_base_rows_per_protocol,
                "members_per_type": args.members_per_type,
                "target_beta": args.target_beta,
                "seed": args.seed,
            },
            {
                "output": "tables/reliability_summary_by_mode.csv",
                "rows": len(summary),
                "n_mc": args.n_mc,
                "max_base_rows_per_protocol": args.max_base_rows_per_protocol,
                "members_per_type": args.members_per_type,
                "target_beta": args.target_beta,
                "seed": args.seed,
            },
        ]
    )
    manifest.to_csv(out_dir / "monte_carlo_reliability_manifest.csv", index=False)

    print(manifest.to_string(index=False))
    print("\nReliability summary preview:")
    print(summary.head(12).to_string(index=False))


if __name__ == "__main__":
    main()

