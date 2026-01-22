#!/usr/bin/env python3
# scripts/sim/setup.py
from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Helpers
# ----------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _lower_cols(df: pd.DataFrame) -> dict:
    return {c.lower(): c for c in df.columns}


def _pick_col(df: pd.DataFrame, *cands: str) -> Optional[str]:
    m = _lower_cols(df)
    for c in cands:
        if c.lower() in m:
            return m[c.lower()]
    return None


def _read_obs_csv_to_arrays(input_csv: Path, elev_default: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a raw observation CSV and return easting, northing, elevation, obs arrays.

    Flexible detection:
      easting:      easting, east, x, Easting
      northing:     northing, north, y, Northing
      elevation:    elevation, elev, z, altitude, alt   (optional)
      observations: observations, obs, mass, load, thickness, mass_value, total(kg/m2), Observations

    NOTE:
      - We accept site_id columns if present, but DO NOT output them (legacy Tephra2 format).
    """
    df = pd.read_csv(input_csv)

    c_e = _pick_col(df, "easting", "east", "x", "Easting")
    c_n = _pick_col(df, "northing", "north", "y", "Northing")
    c_z = _pick_col(df, "elevation", "elev", "z", "altitude", "alt", "Elevation")
    c_obs = _pick_col(
        df,
        "observations", "obs", "mass", "load", "thickness",
        "mass_value", "total(kg/m2)", "Observations"
    )

    missing = [name for name, col in [("easting", c_e), ("northing", c_n), ("observations", c_obs)] if col is None]
    if missing:
        raise ValueError(
            f"Missing required columns in {input_csv.name}: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    e = df[c_e].to_numpy(dtype=float)
    n = df[c_n].to_numpy(dtype=float)

    if c_z is None:
        z = np.full_like(e, float(elev_default), dtype=float)
    else:
        z = df[c_z].to_numpy(dtype=float)

    obs = df[c_obs].to_numpy(dtype=float)
    return e, n, z, obs


def _write_sites_observations_legacy(
    sites_path: Path,
    obs_path: Path,
    e: np.ndarray,
    n: np.ndarray,
    z: np.ndarray,
    obs: np.ndarray,
) -> None:
    """
    Legacy Tephra2-style:
      - sites.csv:        E N Z (3 cols), whitespace-delimited, no header
      - observations.csv: obs (1 col),   whitespace-delimited, no header
    """
    sites_arr = np.column_stack([e, n, z])
    np.savetxt(sites_path, sites_arr, fmt="%.3f")   # matches common Tephra2 input conventions
    np.savetxt(obs_path, obs, fmt="%.6f")

    print(f"[OK] wrote sites        -> {sites_path} (rows={len(sites_arr)})")
    print(f"[OK] wrote observations -> {obs_path} (rows={len(obs)})")


def _write_tephra2_conf_full_legacy(
    out_path: Path,
    vent_easting: float,
    vent_northing: float,
    vent_elevation: float,
    plume_height: float,
    eruption_mass_kg: float,
    median_grainsize: float,
    std_grainsize: float,
    alpha: float,
    beta: float,
    min_grainsize: float,
    max_grainsize: float,
) -> None:
    """
    Write tephra2.conf with the FULL legacy template (including defaults and comments),
    matching the style your lab expects.

    Note: We round vent easting/northing/elev to integers like typical Tephra2 examples.
    """
    ve = int(round(vent_easting))
    vn = int(round(vent_northing))
    vz = int(round(vent_elevation))

    txt = f"""VENT_EASTING {ve}
VENT_NORTHING {vn}
VENT_ELEVATION {vz}
#
# Note: UTM coordinates are used (add 10,000,000 m in 
#      northern hemisphere
#
PLUME_HEIGHT   {plume_height:.6f}
ALPHA {alpha:g}
BETA {beta:g}
ERUPTION_MASS  {eruption_mass_kg:.6f}
MAX_GRAINSIZE {int(round(max_grainsize))}
MIN_GRAINSIZE {int(round(min_grainsize))}
MEDIAN_GRAINSIZE {median_grainsize:g}
STD_GRAINSIZE {std_grainsize:g}

/*eddy diff for small particles in m2/s (400 cm2/s) */
EDDY_CONST  0.04

# diffusion coeff for large particles (m2/s)
DIFFUSION_COEFFICIENT 1000

# threshold for change in diffusion (seconds fall time)
FALL_TIME_THRESHOLD 1e9

# density model for the pyroclasts
LITHIC_DENSITY \t2700.0
PUMICE_DENSITY \t1024.0

#define column integration steps
COL_STEPS 100
PART_STEPS 100

# Note: 
# 0 = uniform distribution using threshold at PLUME_RATIO (no longer used)
# 1 = log-normal distribution using beta (no longer used)
# 2 = beta distribution using parameters alpha and beta (set below)
PLUME_MODEL 2
"""
    out_path.write_text(txt)
    print(f"[OK] wrote tephra2.conf -> {out_path}")


@dataclass
class SimMeta:
    created_at_utc: str
    input_csv: str
    wind_source: Optional[str]

    vent_easting: float
    vent_northing: float
    vent_elevation: float

    plume_height: float
    eruption_mass_kg: float
    ln_mass: float

    median_grainsize: float
    std_grainsize: float
    alpha: float
    beta: float
    min_grainsize: float
    max_grainsize: float


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build a simulation-ready input tree from one observation CSV.")

    parser.add_argument("input_csv", type=str, help="Raw observation CSV (e.g., data/input/observations/cn92_a.csv)")
    parser.add_argument("--out-root", type=str, required=True, help="Scenario output root (creates input/ and config/ inside).")

    # Vent / source parameters
    parser.add_argument("--vent-easting", type=float, required=True)
    parser.add_argument("--vent-northing", type=float, required=True)
    parser.add_argument("--vent-elev", type=float, required=True)

    parser.add_argument("--plume-height", type=float, required=True, help="Plume height in meters.")
    parser.add_argument("--eruption-mass", type=float, default=None, help="Eruption mass in kg (physical units).")

    # Natural log mass (alias: --log-mass, but it is ln(kg), not base-10)
    parser.add_argument(
        "--ln-mass", "--log-mass", dest="ln_mass", type=float, default=None,
        help="Natural log of eruption mass in kg. (Alias --log-mass kept for compatibility; it is ln, not log10.)"
    )

    # Grain-size and plume distribution params
    parser.add_argument("--median-gs", type=float, required=True)
    parser.add_argument("--std-gs", type=float, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--min-gs", type=float, required=True)
    parser.add_argument("--max-gs", type=float, required=True)

    # Wind
    parser.add_argument("--wind-src", type=str, default=None, help="Wind file to copy into input/wind.txt")

    # Observation ingestion defaults
    parser.add_argument("--elev-default", type=float, default=0.0, help="If elevation missing, fill with this value.")

    args = parser.parse_args(argv)

    input_csv = Path(args.input_csv)
    out_root = Path(args.out_root)
    input_dir = out_root / "input"
    config_dir = out_root / "config"
    _ensure_dir(input_dir)
    _ensure_dir(config_dir)

    # Decide mass (kg) and ln mass (natural log)
    if args.eruption_mass is None and args.ln_mass is None:
        raise ValueError("Provide either --eruption-mass (kg) or --ln-mass/--log-mass (ln(kg)).")

    if args.eruption_mass is not None:
        eruption_mass_kg = float(args.eruption_mass)
        if eruption_mass_kg <= 0:
            raise ValueError("eruption mass must be positive.")
        ln_mass = float(math.log(eruption_mass_kg))
    else:
        ln_mass = float(args.ln_mass)
        eruption_mass_kg = float(math.exp(ln_mass))

    # Convert raw obs into legacy Tephra2 input files (no IDs in output)
    e, n, z, obs = _read_obs_csv_to_arrays(input_csv, elev_default=float(args.elev_default))
    _write_sites_observations_legacy(
        sites_path=input_dir / "sites.csv",
        obs_path=input_dir / "observations.csv",
        e=e, n=n, z=z, obs=obs,
    )

    # Wind file -> input/wind.txt
    wind_source = None
    if args.wind_src:
        wsrc = Path(args.wind_src)
        if not wsrc.exists():
            raise FileNotFoundError(f"Wind source not found: {wsrc}")
        shutil.copyfile(wsrc, input_dir / "wind.txt")
        wind_source = str(wsrc)
        print(f"[OK] copied wind -> {input_dir/'wind.txt'} from {wind_source}")

    # Full legacy tephra2.conf
    _write_tephra2_conf_full_legacy(
        out_path=input_dir / "tephra2.conf",
        vent_easting=float(args.vent_easting),
        vent_northing=float(args.vent_northing),
        vent_elevation=float(args.vent_elev),
        plume_height=float(args.plume_height),
        eruption_mass_kg=float(eruption_mass_kg),
        median_grainsize=float(args.median_gs),
        std_grainsize=float(args.std_gs),
        alpha=float(args.alpha),
        beta=float(args.beta),
        min_grainsize=float(args.min_gs),
        max_grainsize=float(args.max_gs),
    )

    # Meta (used by simulate.py to fetch “true” values cleanly)
    meta = SimMeta(
        created_at_utc=_now_iso(),
        input_csv=str(input_csv),
        wind_source=wind_source,
        vent_easting=float(args.vent_easting),
        vent_northing=float(args.vent_northing),
        vent_elevation=float(args.vent_elev),
        plume_height=float(args.plume_height),
        eruption_mass_kg=float(eruption_mass_kg),
        ln_mass=float(ln_mass),
        median_grainsize=float(args.median_gs),
        std_grainsize=float(args.std_gs),
        alpha=float(args.alpha),
        beta=float(args.beta),
        min_grainsize=float(args.min_gs),
        max_grainsize=float(args.max_gs),
    )
    (config_dir / "sim_meta.json").write_text(json.dumps(asdict(meta), indent=2))
    print(f"[OK] wrote sim_meta.json -> {config_dir/'sim_meta.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
