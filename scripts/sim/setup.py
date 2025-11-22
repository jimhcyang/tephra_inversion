#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Helper: interactive float prompt
# ---------------------------------------------------------------------
def prompt_float(name: str, default: float | None = None) -> float:
    """
    Ask the user for a float value on stdin if not provided via CLI.
    """
    while True:
        if default is not None:
            raw = input(f"{name} [{default}]: ").strip()
            if raw == "":
                return float(default)
        else:
            raw = input(f"{name}: ").strip()

        try:
            return float(raw)
        except ValueError:
            print(f"Could not parse '{raw}' as a number, please try again.", file=sys.stderr)


# ---------------------------------------------------------------------
# 1. sites.csv & observations.csv from aggregated CSV
# ---------------------------------------------------------------------
def make_sites_and_obs(
    input_csv: Path,
    sites_path: Path,
    obs_path: Path,
    default_elev: float = 100.0,
) -> None:
    """
    Read easting, northing, mass_value from input_csv and write:
      - sites.csv (E, N, Z) with Z = default_elev
      - observations.csv (single column of mass_value)
    """
    df = pd.read_csv(input_csv)

    required = {"easting", "northing", "mass_value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{input_csv} is missing required columns: {', '.join(sorted(missing))}"
        )

    e = df["easting"].to_numpy(dtype=float)
    n = df["northing"].to_numpy(dtype=float)
    z = np.full_like(e, float(default_elev), dtype=float)
    sites_arr = np.column_stack([e, n, z])

    obs_arr = df["mass_value"].to_numpy(dtype=float)

    # space-delimited, no header – Tephra2 expects this format
    np.savetxt(sites_path, sites_arr, fmt="%.3f")
    np.savetxt(obs_path, obs_arr, fmt="%.6f")

    print(f"Wrote sites to        {sites_path}  (rows={len(sites_arr)})")
    print(f"Wrote observations to {obs_path}  (rows={len(obs_arr)})")


# ---------------------------------------------------------------------
# 2. wind.txt – zero wind every 500 m up to 25 km
# ---------------------------------------------------------------------
def make_wind(
    wind_path: Path,
    dz: int = 500,
    z_max: int = 25000,
    speed: float = 0.0,
    direction: int = 0,
) -> None:
    """
    Create a simple wind.txt:
      #HEIGHT SPEED DIRECTION
      500   0.00  0
      1000  0.00  0
      ...
      25000 0.00  0
    """
    heights = np.arange(dz, z_max + 1, dz, dtype=int)

    with wind_path.open("w") as f:
        f.write("#HEIGHT\tSPEED\tDIRECTION\n")
        for h in heights:
            f.write(f"{h:d} {speed:.2f} {direction:d}\n")

    print(f"Wrote wind profile to {wind_path}  (levels={len(heights)})")


# ---------------------------------------------------------------------
# 3. tephra2.conf – template with user-specified vent & plume params
# ---------------------------------------------------------------------
def make_tephra2_conf(
    conf_path: Path,
    vent_easting: float,
    vent_northing: float,
    plume_height: float,
    eruption_mass: float,
    vent_elev: float = 100.0,
) -> None:
    """
    Write a tephra2.conf identical to your example except for the user-supplied
    VENT_EASTING, VENT_NORTHING, PLUME_HEIGHT, ERUPTION_MASS (vent elevation
    optional, default 100 m).
    """
    txt = f"""VENT_EASTING {vent_easting:.0f}
VENT_NORTHING {vent_northing:.0f}
VENT_ELEVATION {vent_elev:.0f}
#
# Note: UTM coordinates are used (add 10,000,000 m in 
#      northern hemisphere
#
PLUME_HEIGHT   {plume_height:.0f}
ALPHA 2.0
BETA 2.0
ERUPTION_MASS  {eruption_mass:.0f}
MAX_GRAINSIZE -5
MIN_GRAINSIZE 5
MEDIAN_GRAINSIZE 0
STD_GRAINSIZE 2.0

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
    conf_path.write_text(txt)
    print(f"Wrote tephra2.conf to {conf_path}")


# ---------------------------------------------------------------------
# 4. sim_meta.json – minimal scenario description
# ---------------------------------------------------------------------
def write_meta(
    meta_path: Path,
    *,
    input_csv: Path,
    vent_easting: float,
    vent_northing: float,
    vent_elev: float,
    plume_height: float,
    eruption_mass: float,
) -> None:
    meta = {
        "input_csv": str(input_csv),
        "vent_easting": float(vent_easting),
        "vent_northing": float(vent_northing),
        "vent_elev": float(vent_elev),
        "plume_height": float(plume_height),
        "eruption_mass": float(eruption_mass),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote simulation metadata to {meta_path}")


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Set up a simulation input directory (data_sim-style) for Tephra2 inversion."
    )
    parser.add_argument(
        "input_csv",
        type=str,
        help="Path to CSV with columns: easting, northing, mass_value "
             "(e.g. data_std/cn_std_agg.csv).",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="data_sim",
        help="Root directory for simulation data (default: data_sim). "
             "Input files will go in <out-root>/input.",
    )
    parser.add_argument(
        "--vent-easting",
        type=float,
        default=None,
        help="VENT_EASTING (UTM easting). If omitted, will be prompted.",
    )
    parser.add_argument(
        "--vent-northing",
        type=float,
        default=None,
        help="VENT_NORTHING (UTM northing). If omitted, will be prompted.",
    )
    parser.add_argument(
        "--vent-elev",
        type=float,
        default=100.0,
        help="VENT_ELEVATION (m). Default: 100.",
    )
    parser.add_argument(
        "--plume-height",
        type=float,
        default=None,
        help="Initial PLUME_HEIGHT (m) to embed in tephra2.conf. "
             "If omitted, will be prompted.",
    )
    parser.add_argument(
        "--eruption-mass",
        type=float,
        default=None,
        help="Initial ERUPTION_MASS (kg) to embed in tephra2.conf. "
             "If omitted, will be prompted.",
    )
    parser.add_argument(
        "--elev-default",
        type=float,
        default=100.0,
        help="Default station elevation (m) for sites.csv (3rd column). Default: 100.",
    )

    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    out_root = Path(args.out_root).resolve()
    input_dir = out_root / "input"
    output_dir = out_root / "output"
    config_dir = out_root / "config"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    # 1) sites.csv & observations.csv
    sites_path = input_dir / "sites.csv"
    obs_path = input_dir / "observations.csv"
    make_sites_and_obs(
        input_csv,
        sites_path,
        obs_path,
        default_elev=args.elev_default,
    )

    # 2) wind.txt (zero wind)
    wind_path = input_dir / "wind.txt"
    make_wind(wind_path)

    # 3) Ask for vent/plume parameters if not provided
    vent_easting = (
        args.vent_easting
        if args.vent_easting is not None
        else prompt_float("VENT_EASTING (UTM easting)")
    )
    vent_northing = (
        args.vent_northing
        if args.vent_northing is not None
        else prompt_float("VENT_NORTHING (UTM northing)")
    )
    plume_height = (
        args.plume_height
        if args.plume_height is not None
        else prompt_float("PLUME_HEIGHT (m)", default=7500.0)
    )
    eruption_mass = (
        args.eruption_mass
        if args.eruption_mass is not None
        else prompt_float("ERUPTION_MASS (kg)", default=5e10)
    )

    # 4) tephra2.conf
    conf_path = input_dir / "tephra2.conf"
    make_tephra2_conf(
        conf_path,
        vent_easting=vent_easting,
        vent_northing=vent_northing,
        plume_height=plume_height,
        eruption_mass=eruption_mass,
        vent_elev=args.vent_elev,
    )

    # 5) sim_meta.json
    meta_path = config_dir / "sim_meta.json"
    write_meta(
        meta_path,
        input_csv=input_csv,
        vent_easting=vent_easting,
        vent_northing=vent_northing,
        vent_elev=args.vent_elev,
        plume_height=plume_height,
        eruption_mass=eruption_mass,
    )

    print("\nDone.")
    print(f"Input directory  : {input_dir}")
    print(f"Output directory : {output_dir}")
    print(f"Metadata         : {meta_path}")
    print(
        "Later, simulate.py can:\n"
        "  • load config/default_config.py via load_config()\n"
        "  • override DEFAULT_CONFIG['paths'] to point to this data_sim tree\n"
        "  • optionally override priors / run lengths, then call TephraInversion(config=...)."
    )


if __name__ == "__main__":
    main()

"""
python -m scripts.sim.setup \
  data_std/cn_std_agg.csv \
  --out-root data_sim_cerro \
  --vent-easting 532400 \
  --vent-northing 1382525 \
  --vent-elev 100 \
  --plume-height 7000 \
  --eruption-mass 2.4e10
"""