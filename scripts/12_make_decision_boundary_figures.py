from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


LABELS = {
    "random_point_to_random_empirical": "Random point -> random empirical",
    "group_point_to_group_empirical": "Group point -> group empirical",
    "group_point_to_group_conformal": "Group point -> group conformal envelope",
}


COLORS = {
    "random_point_to_random_empirical": "#1f77b4",
    "group_point_to_group_empirical": "#d95f02",
    "group_point_to_group_conformal": "#7570b3",
}


def plot_one(detail: pd.DataFrame, comparison_name: str, beta_threshold: float, out_path: Path) -> None:
    subset = detail[detail["comparison_name"] == comparison_name].copy()
    if subset.empty:
        raise RuntimeError(f"No rows for {comparison_name}")

    safe_to_unsafe = (
        (subset["beta_ref"] >= beta_threshold)
        & (subset["beta_comp"] < beta_threshold)
    )
    no_change = ~safe_to_unsafe

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    if no_change.any():
        ax.scatter(
            subset.loc[no_change, "beta_ref"],
            subset.loc[no_change, "beta_comp"],
            s=13,
            alpha=0.26,
            color="#8c8c8c",
            edgecolors="none",
            label="No safe-to-unsafe flip",
        )
    if safe_to_unsafe.any():
        ax.scatter(
            subset.loc[safe_to_unsafe, "beta_ref"],
            subset.loc[safe_to_unsafe, "beta_comp"],
            s=16,
            alpha=0.58,
            color=COLORS.get(comparison_name, "#1f77b4"),
            edgecolors="none",
            label="Safe-to-unsafe",
        )

    ax.axvline(beta_threshold, color="#222222", lw=1.2, ls="--")
    ax.axhline(beta_threshold, color="#222222", lw=1.2, ls="--")
    low = min(subset["beta_ref"].min(), subset["beta_comp"].min(), beta_threshold) - 0.08
    high = max(subset["beta_ref"].max(), subset["beta_comp"].max(), beta_threshold) + 0.08
    ax.plot([low, high], [low, high], color="#aaaaaa", lw=1.0, ls=":")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Point-prediction beta")
    ax.set_ylabel("Uncertainty-aware beta")
    ax.set_title(LABELS.get(comparison_name, comparison_name))
    ax.text(
        beta_threshold + 0.03,
        low + 0.03,
        "safe-to-unsafe quadrant",
        fontsize=8,
        color="#333333",
        va="bottom",
    )
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=project_root_from_script())
    parser.add_argument("--beta-threshold", type=float, default=3.0)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    out_dir = project_root / "ai_autoboost" / "outputs" / "round6_structural_reliability"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    detail_path = out_dir / "confirmatory_high_signal_decision_flip_pair_details.csv"
    detail = pd.read_csv(detail_path)

    rows = []
    for comparison_name in LABELS:
        out_path = fig_dir / f"fig_r6_decision_boundary_{comparison_name}.png"
        plot_one(detail, comparison_name, args.beta_threshold, out_path)
        rows.append(
            {
                "figure": str(out_path.relative_to(project_root)),
                "comparison_name": comparison_name,
                "beta_threshold": args.beta_threshold,
                "bytes": out_path.stat().st_size,
            }
        )

    manifest = pd.DataFrame(rows)
    manifest_path = out_dir / "decision_boundary_figure_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
