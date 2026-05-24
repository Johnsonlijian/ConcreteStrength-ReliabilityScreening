from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def summarize_for_threshold(detail: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for comparison_name, group in detail.groupby("comparison_name"):
        ref_safe = group["beta_ref"] >= threshold
        comp_safe = group["beta_comp"] >= threshold
        safe_to_unsafe = ref_safe & ~comp_safe
        unsafe_to_safe = ~ref_safe & comp_safe
        changed = ref_safe != comp_safe
        rows.append(
            {
                "beta_threshold": threshold,
                "comparison_name": comparison_name,
                "n_pairs": int(len(group)),
                "safe_to_unsafe_count": int(safe_to_unsafe.sum()),
                "unsafe_to_safe_count": int(unsafe_to_safe.sum()),
                "decision_changed_count": int(changed.sum()),
                "flip_rate_safe_to_unsafe": float(safe_to_unsafe.mean()),
                "flip_rate_unsafe_to_safe": float(unsafe_to_safe.mean()),
                "decision_change_rate": float(changed.mean()),
                "beta_ref_median": float(group["beta_ref"].median()),
                "beta_comp_median": float(group["beta_comp"].median()),
                "delta_beta_mean": float(group["delta_beta_comp_minus_ref"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=project_root_from_script())
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[2.5, 3.0, 3.5],
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    out_dir = project_root / "ai_autoboost" / "outputs" / "round6_structural_reliability"
    table_dir = out_dir / "tables"
    detail_path = out_dir / "confirmatory_high_signal_decision_flip_pair_details.csv"
    if not detail_path.exists():
        raise FileNotFoundError(
            f"Missing {detail_path}. Run 08_confirmatory_high_signal_mcs.py first."
        )

    detail = pd.read_csv(detail_path)
    summaries = [summarize_for_threshold(detail, threshold) for threshold in args.thresholds]
    all_summary = pd.concat(summaries, ignore_index=True).sort_values(
        ["beta_threshold", "decision_change_rate"],
        ascending=[True, False],
    )

    out_path = table_dir / "confirmatory_beta_threshold_sensitivity.csv"
    all_summary.to_csv(out_path, index=False)

    manifest = pd.DataFrame(
        [
            {
                "output": "tables/confirmatory_beta_threshold_sensitivity.csv",
                "rows": len(all_summary),
                "thresholds": ";".join(str(v) for v in args.thresholds),
            }
        ]
    )
    manifest.to_csv(out_dir / "confirmatory_beta_threshold_sensitivity_manifest.csv", index=False)

    print(manifest.to_string(index=False))
    print("\nBeta-threshold sensitivity:")
    print(all_summary.to_string(index=False))


if __name__ == "__main__":
    main()
