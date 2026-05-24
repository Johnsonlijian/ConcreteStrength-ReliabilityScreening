from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FC_GRID_MPA = [20, 30, 40, 50, 60, 80]


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def beam_flexural_capacity_kNm(
    fc_mpa: float,
    fy_mpa: float,
    b_mm: float,
    d_mm: float,
    as_mm2: float,
    phi: float = 0.90,
) -> dict:
    """ACI-type singly reinforced rectangular beam capacity.

    Units: MPa = N/mm2; output moment is kN*m.
    This is a transparent analytical propagation model, not a design-code
    replacement. Compression reinforcement, shear, serviceability, bar layout,
    strain limits, and minimum/maximum reinforcement checks are not evaluated.
    """
    if fc_mpa <= 0 or fy_mpa <= 0 or b_mm <= 0 or d_mm <= 0 or as_mm2 <= 0:
        return {
            "nominal_resistance": np.nan,
            "design_resistance": np.nan,
            "valid_model_region": False,
            "failure_reason": "nonpositive_input",
        }

    a_mm = as_mm2 * fy_mpa / (0.85 * fc_mpa * b_mm)
    lever_arm_mm = d_mm - a_mm / 2.0
    valid = a_mm > 0 and lever_arm_mm > 0 and a_mm < d_mm
    mn_nmm = as_mm2 * fy_mpa * lever_arm_mm if valid else np.nan
    mn_kNm = mn_nmm / 1_000_000.0 if valid else np.nan
    return {
        "nominal_resistance": mn_kNm,
        "design_resistance": phi * mn_kNm if valid else np.nan,
        "valid_model_region": bool(valid),
        "failure_reason": "" if valid else "compression_block_exceeds_effective_depth",
        "a_mm": a_mm,
        "lever_arm_mm": lever_arm_mm,
    }


def short_column_axial_capacity_kN(
    fc_mpa: float,
    fy_mpa: float,
    ag_mm2: float,
    steel_area_mm2: float,
    phi: float = 0.65,
    tied_column_factor: float = 0.80,
) -> dict:
    """ACI-type concentric short tied RC column axial capacity.

    Units: MPa = N/mm2; output axial force is kN.
    This is a simplified analytical reliability case. Slenderness, eccentricity,
    interaction diagrams, confinement, second-order effects, and detailing
    checks are outside this first-pass propagation model.
    """
    if fc_mpa <= 0 or fy_mpa <= 0 or ag_mm2 <= 0 or steel_area_mm2 < 0:
        return {
            "nominal_resistance": np.nan,
            "design_resistance": np.nan,
            "valid_model_region": False,
            "failure_reason": "invalid_input",
        }
    if steel_area_mm2 >= ag_mm2:
        return {
            "nominal_resistance": np.nan,
            "design_resistance": np.nan,
            "valid_model_region": False,
            "failure_reason": "steel_area_exceeds_gross_area",
        }

    concrete_area_mm2 = ag_mm2 - steel_area_mm2
    pn0_n = 0.85 * fc_mpa * concrete_area_mm2 + fy_mpa * steel_area_mm2
    pn0_kN = pn0_n / 1000.0
    return {
        "nominal_resistance": pn0_kN,
        "design_resistance": phi * tied_column_factor * pn0_kN,
        "valid_model_region": True,
        "failure_reason": "",
        "concrete_area_mm2": concrete_area_mm2,
        "tied_column_factor": tied_column_factor,
    }


def evaluate_member(row: pd.Series, fc_mpa: float) -> dict:
    member_type = row["member_type"]
    if member_type == "rc_beam_flexure":
        cap = beam_flexural_capacity_kNm(
            fc_mpa=fc_mpa,
            fy_mpa=float(row["fy_mean_mpa"]),
            b_mm=float(row["b_mm"]),
            d_mm=float(row["d_mm"]),
            as_mm2=float(row["As_mm2"]),
            phi=float(row["phi"]),
        )
        resistance_unit = "kN_m"
    elif member_type == "rc_short_column_axial":
        cap = short_column_axial_capacity_kN(
            fc_mpa=fc_mpa,
            fy_mpa=float(row["fy_mean_mpa"]),
            ag_mm2=float(row["Ag_mm2"]),
            steel_area_mm2=float(row["steel_area_mm2"]),
            phi=float(row["phi"]),
        )
        resistance_unit = "kN"
    else:
        raise ValueError(f"Unknown member_type: {member_type}")

    nominal = cap["nominal_resistance"]
    design = cap["design_resistance"]
    demand = design * float(row["demand_ratio"]) if pd.notna(design) else np.nan
    return {
        "member_id": row["member_id"],
        "member_type": member_type,
        "resistance_model": row["resistance_model"],
        "fc_mpa": fc_mpa,
        "fy_mpa": float(row["fy_mean_mpa"]),
        "demand_ratio": float(row["demand_ratio"]),
        "nominal_resistance": nominal,
        "design_resistance": design,
        "deterministic_demand_at_ratio": demand,
        "resistance_unit": resistance_unit,
        "valid_model_region": cap["valid_model_region"],
        "failure_reason": cap["failure_reason"],
        "aux_a_mm": cap.get("a_mm", np.nan),
        "aux_lever_arm_mm": cap.get("lever_arm_mm", np.nan),
        "aux_concrete_area_mm2": cap.get("concrete_area_mm2", np.nan),
    }


def evaluate_population(population: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, member in population.iterrows():
        for fc_mpa in FC_GRID_MPA:
            rows.append(evaluate_member(member, fc_mpa))
    return pd.DataFrame(rows)


def summarize_snapshot(snapshot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for member_type, group in snapshot.groupby("member_type"):
        valid = group[group["valid_model_region"] == True]
        rows.append(
            {
                "member_type": member_type,
                "n_rows": int(len(group)),
                "n_valid": int(len(valid)),
                "n_invalid": int(len(group) - len(valid)),
                "nominal_min": float(valid["nominal_resistance"].min()),
                "nominal_median": float(valid["nominal_resistance"].median()),
                "nominal_max": float(valid["nominal_resistance"].max()),
                "unit": valid["resistance_unit"].iloc[0] if len(valid) else "",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=project_root_from_script())
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    out_dir = project_root / "ai_autoboost" / "outputs" / "round6_structural_reliability"
    table_dir = out_dir / "tables"
    population_path = out_dir / "member_population.csv"
    if not population_path.exists():
        raise FileNotFoundError(
            f"Missing {population_path}. Run 02_generate_member_population.py first."
        )

    population = pd.read_csv(population_path)
    snapshot = evaluate_population(population)
    summary = summarize_snapshot(snapshot)
    snapshot.to_csv(table_dir / "resistance_model_capacity_snapshot.csv", index=False)
    summary.to_csv(table_dir / "resistance_model_capacity_summary.csv", index=False)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

