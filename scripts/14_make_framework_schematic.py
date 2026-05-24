from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def add_box(ax, xy, w, h, title, body, facecolor, edgecolor="#2b2b2b"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.08, title, ha="center", va="top", fontsize=10, weight="bold")
    ax.text(x + w / 2, y + h / 2 - 0.03, body, ha="center", va="center", fontsize=8.2, linespacing=1.2)
    return patch


def add_arrow(ax, start, end, color="#444444"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.3,
        color=color,
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arrow)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=project_root_from_script())
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    out_dir = project_root / "ai_autoboost" / "outputs" / "round6_structural_reliability"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.6, 7.2))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    colors = {
        "data": "#e8f3ff",
        "ml": "#f1f7e8",
        "uncertainty": "#fff0dc",
        "structure": "#f4e9ff",
        "decision": "#ffe8e8",
        "stress": "#e8f8f5",
    }

    boxes = {}
    boxes["data"] = add_box(
        ax,
        (0.45, 4.75),
        2.15,
        1.35,
        "Public material data",
        "UCI concrete strength\n1030 records\n8 inputs -> f'c",
        colors["data"],
    )
    boxes["audit"] = add_box(
        ax,
        (3.05, 4.75),
        2.25,
        1.35,
        "Leakage-aware ML",
        "random split vs\ngroup-aware split\nstress/OOD checks",
        colors["ml"],
    )
    boxes["unc"] = add_box(
        ax,
        (5.85, 4.75),
        2.35,
        1.35,
        "Strength uncertainty",
        "point prediction\nempirical residuals\nconformal interval envelope",
        colors["uncertainty"],
    )
    boxes["member"] = add_box(
        ax,
        (8.75, 4.75),
        2.35,
        1.35,
        "Analytical RC members",
        "beam flexure\nshort-column axial\nACI-type equations",
        colors["structure"],
    )

    boxes["mcs"] = add_box(
        ax,
        (3.05, 2.25),
        2.4,
        1.35,
        "Monte Carlo limit state",
        "g(X)=R(X)-S(X)\nPf and beta\nfinite-sample correction",
        colors["structure"],
    )
    boxes["decision"] = add_box(
        ax,
        (5.95, 2.25),
        2.4,
        1.35,
        "Reliability screening",
        "beta screen\nsafe/unsafe status\ndecision flip metrics",
        colors["decision"],
    )
    boxes["boundary"] = add_box(
        ax,
        (8.9, 2.25),
        2.4,
        1.35,
        "Boundary-regime stress test",
        "RC short columns\nDR = 0.60\nlow-strength / early-age",
        colors["stress"],
    )

    add_arrow(ax, (2.60, 5.42), (3.05, 5.42))
    add_arrow(ax, (5.30, 5.42), (5.85, 5.42))
    add_arrow(ax, (8.20, 5.42), (8.75, 5.42))
    add_arrow(ax, (9.93, 4.75), (4.25, 3.60))
    add_arrow(ax, (5.45, 2.92), (5.95, 2.92))
    add_arrow(ax, (8.35, 2.92), (8.90, 2.92))

    ax.text(
        6.3,
        6.78,
        "Leakage-controlled uncertainty propagation from concrete-strength ML to structural reliability screening",
        ha="center",
        va="center",
        fontsize=13,
        weight="bold",
    )

    ax.text(
        6.3,
        0.72,
        "Scope boundary: computational reliability-screening demonstration; not a new material, design-code replacement, or field-validated structural tool.",
        ha="center",
        va="center",
        fontsize=9.2,
        color="#333333",
    )

    fig_path_png = fig_dir / "fig_r9_framework_schematic.png"
    fig_path_pdf = fig_dir / "fig_r9_framework_schematic.pdf"
    fig.savefig(fig_path_png, dpi=300, bbox_inches="tight")
    fig.savefig(fig_path_pdf, bbox_inches="tight")
    plt.close(fig)

    manifest = pd.DataFrame(
        [
            {"figure": str(fig_path_png.relative_to(project_root)), "bytes": fig_path_png.stat().st_size},
            {"figure": str(fig_path_pdf.relative_to(project_root)), "bytes": fig_path_pdf.stat().st_size},
        ]
    )
    manifest.to_csv(out_dir / "framework_schematic_manifest.csv", index=False)
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
