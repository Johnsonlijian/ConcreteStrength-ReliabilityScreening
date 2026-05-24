from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import pandas as pd


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def beam_members() -> list[dict]:
    rows: list[dict] = []
    b_values = [250, 300, 350]
    h_values = [450, 550, 650]
    rho_values = [0.006, 0.012, 0.018, 0.024]
    fy_values = [400, 500]
    demand_ratios = [0.40, 0.60, 0.80, 0.95]

    member_i = 1
    for b_mm, h_mm, rho, fy_mean_mpa, demand_ratio in product(
        b_values, h_values, rho_values, fy_values, demand_ratios
    ):
        d_mm = round(0.9 * h_mm, 3)
        as_mm2 = round(rho * b_mm * d_mm, 3)
        ag_mm2 = b_mm * h_mm
        rows.append(
            {
                "member_id": f"RCB-{member_i:04d}",
                "member_type": "rc_beam_flexure",
                "b_mm": b_mm,
                "h_mm": h_mm,
                "d_mm": d_mm,
                "As_mm2": as_mm2,
                "Ag_mm2": ag_mm2,
                "Ac_mm2": "",
                "steel_area_mm2": as_mm2,
                "rho": rho,
                "fy_mean_mpa": fy_mean_mpa,
                "fy_cov": 0.05,
                "geometry_cov": 0.01,
                "load_cov": 0.10,
                "model_error_cov": 0.10,
                "phi": 0.90,
                "demand_ratio": demand_ratio,
                "load_model": "lognormal_load_effect_cov_0.10",
                "resistance_model": "singly_reinforced_rectangular_beam_aci_type",
                "fc_source": "ml_uncertainty_inputs",
                "assumption_note": "Synthetic analytical case-study member; not a real project design.",
            }
        )
        member_i += 1
    return rows


def short_column_members() -> list[dict]:
    rows: list[dict] = []
    b_values = [300, 400, 500, 600]
    rho_values = [0.010, 0.020, 0.030, 0.040]
    fy_values = [400, 500]
    demand_ratios = [0.40, 0.60, 0.80, 0.95]

    member_i = 1
    for b_mm, rho, fy_mean_mpa, demand_ratio in product(
        b_values, rho_values, fy_values, demand_ratios
    ):
        h_mm = b_mm
        ag_mm2 = b_mm * h_mm
        steel_area_mm2 = round(rho * ag_mm2, 3)
        ac_mm2 = round(ag_mm2 - steel_area_mm2, 3)
        rows.append(
            {
                "member_id": f"RCC-{member_i:04d}",
                "member_type": "rc_short_column_axial",
                "b_mm": b_mm,
                "h_mm": h_mm,
                "d_mm": "",
                "As_mm2": "",
                "Ag_mm2": ag_mm2,
                "Ac_mm2": ac_mm2,
                "steel_area_mm2": steel_area_mm2,
                "rho": rho,
                "fy_mean_mpa": fy_mean_mpa,
                "fy_cov": 0.05,
                "geometry_cov": 0.01,
                "load_cov": 0.10,
                "model_error_cov": 0.10,
                "phi": 0.65,
                "demand_ratio": demand_ratio,
                "load_model": "lognormal_axial_load_effect_cov_0.10",
                "resistance_model": "short_tied_rc_column_concentric_aci_type",
                "fc_source": "ml_uncertainty_inputs",
                "assumption_note": "Synthetic analytical case-study member; concentric short-column model only.",
            }
        )
        member_i += 1
    return rows


def summarize_population(population: pd.DataFrame) -> pd.DataFrame:
    grouped = []
    for member_type, group in population.groupby("member_type"):
        grouped.append(
            {
                "member_type": member_type,
                "n_members": int(len(group)),
                "b_min_mm": float(group["b_mm"].min()),
                "b_max_mm": float(group["b_mm"].max()),
                "h_min_mm": float(group["h_mm"].min()),
                "h_max_mm": float(group["h_mm"].max()),
                "rho_min": float(group["rho"].min()),
                "rho_max": float(group["rho"].max()),
                "fy_values_mpa": ";".join(str(int(v)) for v in sorted(group["fy_mean_mpa"].unique())),
                "demand_ratio_values": ";".join(f"{v:.2f}" for v in sorted(group["demand_ratio"].unique())),
            }
        )
    return pd.DataFrame(grouped)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=project_root_from_script())
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    out_dir = project_root / "ai_autoboost" / "outputs" / "round6_structural_reliability"
    table_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    population = pd.DataFrame(beam_members() + short_column_members())
    population.to_csv(out_dir / "member_population.csv", index=False)

    summary = summarize_population(population)
    summary.to_csv(table_dir / "member_population_summary.csv", index=False)

    print(summary.to_string(index=False))
    print(f"\nWrote {len(population)} synthetic analytical members.")


if __name__ == "__main__":
    main()

