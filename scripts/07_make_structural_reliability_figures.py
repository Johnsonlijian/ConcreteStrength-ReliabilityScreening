from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def mode_label(mode: str) -> str:
    return {
        "M1_deterministic_point_prediction": "Point prediction",
        "M2_empirical_residual_90_by_seed": "Empirical residual",
        "M4_split_conformal_90_envelope": "Conformal envelope",
    }.get(mode, mode)


def comparison_label(name: str) -> str:
    return {
        "random_point_to_random_empirical": "Random point -> empirical",
        "group_point_to_group_empirical": "Group point -> empirical",
        "group_point_to_group_conformal": "Group point -> conformal",
        "random_point_to_group_conformal": "Random point -> group conformal",
        "random_empirical_to_group_empirical": "Random empirical -> group empirical",
    }.get(name, name)


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_beta_by_demand(summary: pd.DataFrame, fig_dir: Path) -> Path:
    df = summary.copy()
    df["mode_label"] = df["uncertainty_mode"].map(mode_label)
    member_types = list(df["member_type"].drop_duplicates())
    fig, axes = plt.subplots(1, len(member_types), figsize=(12, 4.5), sharey=True)
    if len(member_types) == 1:
        axes = [axes]
    styles = {
        "group_aware_round3": "-",
        "random_split_round1": "--",
    }
    colors = {
        "Point prediction": "#355070",
        "Empirical residual": "#B56576",
        "Conformal envelope": "#6D597A",
    }
    for ax, member_type in zip(axes, member_types):
        sub = df[df["member_type"] == member_type]
        for (protocol, label), group in sub.groupby(["split_protocol", "mode_label"]):
            group = group.sort_values("demand_ratio")
            ax.plot(
                group["demand_ratio"],
                group["beta_p10"],
                linestyle=styles.get(protocol, "-"),
                marker="o",
                color=colors.get(label, "#333333"),
                label=f"{protocol.replace('_', ' ')} | {label}",
            )
        ax.axhline(3.0, color="#333333", linestyle="--", linewidth=1)
        ax.set_ylabel("10th-percentile reliability index beta")
        ax.set_xlabel("Demand ratio")
        ax.set_title(member_type.replace("_", " "))
        ax.grid(True, alpha=0.25)
    axes[-1].legend(fontsize=7, loc="upper right")
    path = fig_dir / "fig_r6_beta_p10_by_demand.png"
    savefig(path)
    return path


def plot_below_beta(summary: pd.DataFrame, fig_dir: Path) -> Path:
    df = summary.copy()
    df["mode_label"] = df["uncertainty_mode"].map(mode_label)
    filtered = df[df["split_protocol"] == "group_aware_round3"].copy()
    pivot = filtered.pivot_table(
        index=["member_type", "demand_ratio"],
        columns="mode_label",
        values="below_beta_rate",
        aggfunc="mean",
    ).fillna(0.0)
    ax = pivot.plot(kind="bar", figsize=(11, 5.5), width=0.8)
    ax.set_ylabel("Fraction below beta screen")
    ax.set_xlabel("Member type and demand ratio")
    ax.set_title("Below-beta screening rate across uncertainty modes")
    ax.legend(title="Uncertainty mode", loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.grid(axis="y", alpha=0.25)
    path = fig_dir / "fig_r6_below_beta_rate_by_mode.png"
    savefig(path)
    return path


def plot_decision_flip_overall(overall: pd.DataFrame, fig_dir: Path) -> Path:
    df = overall.copy()
    df["comparison_label"] = df["comparison_name"].map(comparison_label)
    df = df.sort_values("decision_change_rate", ascending=True)
    plt.figure(figsize=(9, 5.8))
    plt.barh(df["comparison_label"], df["decision_change_rate"], color="#476A6F")
    plt.xlabel("Decision-change rate")
    plt.ylabel("")
    plt.title("Reliability screening decisions change when uncertainty is propagated")
    for i, value in enumerate(df["decision_change_rate"]):
        plt.text(value + 0.004, i, f"{value:.1%}", va="center")
    path = fig_dir / "fig_r6_decision_flip_overall.png"
    savefig(path)
    return path


def plot_decision_flip_heatmap(flip: pd.DataFrame, fig_dir: Path) -> Path:
    df = flip[flip["comparison_name"] == "group_point_to_group_empirical"].copy()
    df["column"] = df["member_type"].str.replace("_", " ") + " | DR " + df["demand_ratio"].astype(str)
    table = df.pivot_table(
        index="strength_group",
        columns="column",
        values="flip_rate_safe_to_unsafe",
        aggfunc="mean",
        fill_value=0.0,
    )
    fig, ax = plt.subplots(figsize=(12, 4.8))
    values = table.to_numpy()
    im = ax.imshow(values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(0.2, values.max()))
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index, fontsize=8)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Safe-to-unsafe flip rate")
    plt.xlabel("Member type and demand ratio")
    plt.ylabel("Strength group")
    plt.title("Safe-to-unsafe flip rate: group point -> group empirical residual")
    path = fig_dir / "fig_r6_decision_flip_heatmap.png"
    savefig(path)
    return path


def plot_sensitivity(overall: pd.DataFrame, fig_dir: Path) -> Path:
    df = overall.copy()
    df["source"] = df["source"].str.replace("_", " ")
    df = df.sort_values("delta_beta_mean", ascending=True)
    plt.figure(figsize=(8, 5.2))
    plt.barh(df["source"], df["delta_beta_mean"], color="#5B8C5A")
    plt.xlabel("Mean beta increase when source uncertainty is fixed")
    plt.ylabel("Uncertainty source fixed")
    plt.title("First-pass sensitivity of reliability index to uncertainty sources")
    for i, value in enumerate(df["delta_beta_mean"]):
        plt.text(value + 0.004, i, f"{value:.3f}", va="center")
    path = fig_dir / "fig_r6_sensitivity_delta_beta.png"
    savefig(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=project_root_from_script())
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    out_dir = project_root / "ai_autoboost" / "outputs" / "round6_structural_reliability"
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    reliability_summary = pd.read_csv(table_dir / "reliability_summary_by_mode.csv")
    flip_overall = pd.read_csv(table_dir / "decision_flip_overall_summary.csv")
    flip = pd.read_csv(out_dir / "decision_flip_results.csv")
    sensitivity_overall = pd.read_csv(table_dir / "sensitivity_overall_by_source.csv")

    outputs = [
        plot_beta_by_demand(reliability_summary, fig_dir),
        plot_below_beta(reliability_summary, fig_dir),
        plot_decision_flip_overall(flip_overall, fig_dir),
        plot_decision_flip_heatmap(flip, fig_dir),
        plot_sensitivity(sensitivity_overall, fig_dir),
    ]
    manifest = pd.DataFrame(
        [{"figure": str(path.relative_to(project_root)), "bytes": path.stat().st_size} for path in outputs]
    )
    manifest.to_csv(out_dir / "figure_manifest.csv", index=False)
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
