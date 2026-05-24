from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def add_ci(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "n_pairs" not in out.columns:
        raise ValueError("Expected n_pairs column")

    count_cols = [
        ("safe_to_unsafe_count", "flip_rate_safe_to_unsafe"),
        ("unsafe_to_safe_count", "flip_rate_unsafe_to_safe"),
        ("decision_changed_count", "decision_change_rate"),
    ]
    for count_col, rate_col in count_cols:
        if count_col not in out.columns or rate_col not in out.columns:
            continue
        lows = []
        highs = []
        for k, n in zip(out[count_col].astype(int), out["n_pairs"].astype(int)):
            lo, hi = wilson_interval(k, n)
            lows.append(lo)
            highs.append(hi)
        out[f"{rate_col}_wilson95_low"] = lows
        out[f"{rate_col}_wilson95_high"] = highs
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=project_root_from_script())
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    out_dir = project_root / "ai_autoboost" / "outputs" / "round6_structural_reliability"
    table_dir = out_dir / "tables"

    inputs = [
        (
            table_dir / "decision_flip_overall_summary.csv",
            table_dir / "decision_flip_overall_summary_with_ci.csv",
            "whole_grid",
        ),
        (
            table_dir / "confirmatory_high_signal_decision_flip_overall.csv",
            table_dir / "confirmatory_high_signal_decision_flip_overall_with_ci.csv",
            "confirmatory_high_signal",
        ),
    ]

    manifest_rows = []
    for in_path, out_path, label in inputs:
        if not in_path.exists():
            continue
        df = pd.read_csv(in_path)
        ci = add_ci(df)
        ci.insert(0, "analysis_scope", label)
        ci.to_csv(out_path, index=False)
        manifest_rows.append(
            {
                "input": str(in_path.relative_to(project_root)),
                "output": str(out_path.relative_to(project_root)),
                "rows": len(ci),
            }
        )

    if not manifest_rows:
        raise FileNotFoundError("No decision-flip summary tables found.")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out_dir / "decision_flip_ci_manifest.csv", index=False)
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
