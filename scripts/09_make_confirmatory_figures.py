from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def short_label(name: str) -> str:
    mapping = {
        "random_point_to_random_empirical": "Random point ->\nrandom empirical",
        "group_point_to_group_empirical": "Group point ->\ngroup empirical",
        "group_point_to_group_conformal": "Group point ->\ngroup conformal",
    }
    return mapping.get(name, name.replace("_", " "))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=project_root_from_script())
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    out_dir = project_root / "ai_autoboost" / "outputs" / "round6_structural_reliability"
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    overall = pd.read_csv(table_dir / "confirmatory_high_signal_decision_flip_overall.csv")
    by_group = pd.read_csv(table_dir / "confirmatory_high_signal_decision_flip_by_group.csv")

    overall = overall.sort_values("decision_change_rate", ascending=True)
    labels = [short_label(v) for v in overall["comparison_name"]]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.barh(labels, overall["flip_rate_safe_to_unsafe"] * 100, color="#1f77b4")
    ax.set_xlabel("Safe-to-unsafe flip rate (%)")
    ax.set_title("Confirmatory MCS: high-signal short-column regime")
    ax.set_xlim(0, 100)
    for idx, value in enumerate(overall["flip_rate_safe_to_unsafe"] * 100):
        ax.text(value + 1, idx, f"{value:.1f}%", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig_path_1 = fig_dir / "fig_r6_confirmatory_high_signal_flip_overall.png"
    fig.savefig(fig_path_1, dpi=300)
    plt.close(fig)

    group_emp = by_group[
        by_group["comparison_name"] == "group_point_to_group_empirical"
    ].copy()
    group_emp = group_emp.sort_values("decision_change_rate", ascending=False).head(12)
    group_emp["label"] = (
        group_emp["strength_group"].astype(str)
        + "\n"
        + group_emp["age_group"].astype(str)
        + "\n"
        + group_emp["scm_group"].astype(str)
    )
    group_emp = group_emp.sort_values("decision_change_rate", ascending=True)

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ax.barh(group_emp["label"], group_emp["decision_change_rate"] * 100, color="#d95f02")
    ax.set_xlabel("Decision-change rate (%)")
    ax.set_title("Top subgroup decision shifts: group point -> empirical")
    ax.set_xlim(0, 105)
    for idx, value in enumerate(group_emp["decision_change_rate"] * 100):
        ax.text(value + 1, idx, f"{value:.1f}%", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig_path_2 = fig_dir / "fig_r6_confirmatory_high_signal_subgroups.png"
    fig.savefig(fig_path_2, dpi=300)
    plt.close(fig)

    manifest_path = out_dir / "confirmatory_high_signal_figure_manifest.csv"
    manifest = pd.DataFrame(
        [
            {"figure": str(fig_path_1.relative_to(project_root)), "bytes": fig_path_1.stat().st_size},
            {"figure": str(fig_path_2.relative_to(project_root)), "bytes": fig_path_2.stat().st_size},
        ]
    )
    manifest.to_csv(manifest_path, index=False)
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
