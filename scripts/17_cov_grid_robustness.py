from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOAD_COV_GRID = [0.05, 0.10, 0.15]
MODEL_ERROR_COV_GRID = [0.05, 0.10, 0.15]


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def load_module(script_name: str, module_name: str):
    script_path = Path(__file__).with_name(script_name)
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wilson_interval(count: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    phat = count / n
    denom = 1.0 + z**2 / n
    center = (phat + z**2 / (2.0 * n)) / denom
    half = z * np.sqrt((phat * (1.0 - phat) / n) + z**2 / (4.0 * n**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def summarize_with_ci(detail: pd.DataFrame, hs) -> pd.DataFrame:
    summary = hs.summarize_overall(detail)
    rows = []
    for _, row in summary.iterrows():
        n = int(row["n_pairs"])
        changed = int(row["decision_changed_count"])
        safe_to_unsafe = int(row["safe_to_unsafe_count"])
        changed_low, changed_high = wilson_interval(changed, n)
        stu_low, stu_high = wilson_interval(safe_to_unsafe, n)
        enriched = row.to_dict()
        enriched.update(
            {
                "decision_change_rate_wilson95_low": changed_low,
                "decision_change_rate_wilson95_high": changed_high,
                "flip_rate_safe_to_unsafe_wilson95_low": stu_low,
                "flip_rate_safe_to_unsafe_wilson95_high": stu_high,
            }
        )
        rows.append(enriched)
    return pd.DataFrame(rows)


def run_one_scenario(
    scenario_id: str,
    load_cov: float,
    model_error_cov: float,
    ml: pd.DataFrame,
    members: pd.DataFrame,
    mcs,
    hs,
    rng: np.random.Generator,
    n_mc: int,
    max_base_rows_per_protocol: int,
    chunk_size: int,
    target_beta: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_ml = hs.select_high_signal_ml(ml, max_base_rows_per_protocol, rng)
    selected_members = hs.select_high_signal_members(members).copy()
    selected_members["load_cov"] = load_cov
    selected_members["model_error_cov"] = model_error_cov

    cases = selected_ml.merge(selected_members, how="cross").reset_index(drop=True)
    residual_pools = mcs.make_residual_pools(ml)

    result_chunks = []
    for start in range(0, len(cases), chunk_size):
        chunk = cases.iloc[start : start + chunk_size].copy()
        result_chunks.append(
            mcs.simulate_chunk(
                rng,
                chunk,
                n_mc=n_mc,
                residual_pools=residual_pools,
                target_beta=target_beta,
            )
        )

    results = pd.concat(result_chunks, ignore_index=True)
    results["cov_scenario_id"] = scenario_id
    results["load_cov_scenario"] = load_cov
    results["model_error_cov_scenario"] = model_error_cov

    detail_parts = []
    for definition in hs.PAIR_DEFINITIONS:
        detail = hs.build_pair_detail(results, definition)
        if not detail.empty:
            detail_parts.append(detail)

    if not detail_parts:
        raise RuntimeError(f"No matched pairs for scenario {scenario_id}.")

    detail_all = pd.concat(detail_parts, ignore_index=True)
    summary = summarize_with_ci(detail_all, hs)
    summary["cov_scenario_id"] = scenario_id
    summary["load_cov_scenario"] = load_cov
    summary["model_error_cov_scenario"] = model_error_cov
    return results, summary


def make_heatmap(summary: pd.DataFrame, figures_dir: Path) -> Path:
    comparisons = [
        "random_point_to_random_empirical",
        "group_point_to_group_empirical",
        "group_point_to_group_conformal",
    ]
    labels = {
        "random_point_to_random_empirical": "Random point -> empirical",
        "group_point_to_group_empirical": "Group point -> empirical",
        "group_point_to_group_conformal": "Group point -> conformal envelope",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
    for ax, comparison in zip(axes, comparisons):
        subset = summary[summary["comparison_name"] == comparison].copy()
        pivot = subset.pivot(
            index="model_error_cov_scenario",
            columns="load_cov_scenario",
            values="decision_change_rate",
        ).sort_index(ascending=False)
        im = ax.imshow(pivot.to_numpy(), vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(labels[comparison], fontsize=9)
        ax.set_xlabel("Load COV")
        ax.set_ylabel("Model-error COV")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{c:.2f}" for c in pivot.index])
        for row_i, model_cov in enumerate(pivot.index):
            for col_i, load_cov in enumerate(pivot.columns):
                value = pivot.loc[model_cov, load_cov]
                ax.text(
                    col_i,
                    row_i,
                    f"{100.0 * value:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if value < 0.25 or value > 0.55 else "black",
                    fontsize=8,
                )
    cbar = fig.colorbar(im, ax=axes, shrink=0.82)
    cbar.set_label("Decision-change rate")
    out_path = figures_dir / "fig_r11_cov_grid_decision_change_heatmap.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=project_root_from_script())
    parser.add_argument("--n-mc", type=int, default=2000)
    parser.add_argument("--max-base-rows-per-protocol", type=int, default=40)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--target-beta", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260524)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    out_dir = project_root / "ai_autoboost" / "outputs" / "round6_structural_reliability"
    table_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    mcs = load_module("04_monte_carlo_reliability.py", "round6_mcs")
    hs = load_module("08_confirmatory_high_signal_mcs.py", "round6_high_signal")
    hs.mcs = mcs

    rng = np.random.default_rng(args.seed)
    ml = pd.read_csv(out_dir / "ml_uncertainty_inputs.csv")
    members = pd.read_csv(out_dir / "member_population.csv")

    all_results = []
    all_summaries = []
    for load_cov in LOAD_COV_GRID:
        for model_error_cov in MODEL_ERROR_COV_GRID:
            scenario_id = f"load{load_cov:.2f}_model{model_error_cov:.2f}"
            results, summary = run_one_scenario(
                scenario_id=scenario_id,
                load_cov=load_cov,
                model_error_cov=model_error_cov,
                ml=ml,
                members=members,
                mcs=mcs,
                hs=hs,
                rng=rng,
                n_mc=args.n_mc,
                max_base_rows_per_protocol=args.max_base_rows_per_protocol,
                chunk_size=args.chunk_size,
                target_beta=args.target_beta,
            )
            all_results.append(results)
            all_summaries.append(summary)

    results_all = pd.concat(all_results, ignore_index=True)
    summary_all = pd.concat(all_summaries, ignore_index=True)
    summary_all = summary_all.sort_values(
        ["comparison_name", "load_cov_scenario", "model_error_cov_scenario"]
    )

    results_path = out_dir / "cov_grid_high_signal_reliability_results.csv"
    summary_path = table_dir / "cov_grid_high_signal_decision_flip_summary.csv"
    manifest_path = out_dir / "cov_grid_high_signal_manifest.csv"
    figure_path = make_heatmap(summary_all, figures_dir)

    results_all.to_csv(results_path, index=False)
    summary_all.to_csv(summary_path, index=False)

    manifest = pd.DataFrame(
        [
            {
                "output": "cov_grid_high_signal_reliability_results.csv",
                "rows": len(results_all),
                "n_mc": args.n_mc,
                "max_base_rows_per_protocol": args.max_base_rows_per_protocol,
                "target_beta": args.target_beta,
                "seed": args.seed,
            },
            {
                "output": "tables/cov_grid_high_signal_decision_flip_summary.csv",
                "rows": len(summary_all),
                "n_mc": args.n_mc,
                "max_base_rows_per_protocol": args.max_base_rows_per_protocol,
                "target_beta": args.target_beta,
                "seed": args.seed,
            },
            {
                "output": "figures/fig_r11_cov_grid_decision_change_heatmap.png",
                "rows": 1,
                "n_mc": args.n_mc,
                "max_base_rows_per_protocol": args.max_base_rows_per_protocol,
                "target_beta": args.target_beta,
                "seed": args.seed,
            },
        ]
    )
    manifest.to_csv(manifest_path, index=False)

    print(manifest.to_string(index=False))
    print("\nCOV-grid decision-change summary:")
    cols = [
        "comparison_name",
        "load_cov_scenario",
        "model_error_cov_scenario",
        "decision_change_rate",
        "decision_change_rate_wilson95_low",
        "decision_change_rate_wilson95_high",
        "delta_beta_median",
    ]
    print(summary_all[cols].to_string(index=False))


if __name__ == "__main__":
    main()
