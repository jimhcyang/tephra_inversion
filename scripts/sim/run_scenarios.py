#!/usr/bin/env python3
# scripts/sim/run_scenarios.py
from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
import importlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, List


# ============================================================
# DEFAULT CONSTANTS (ALL VISIBLE HERE)
# ============================================================

# Repo layout (assumes this file is scripts/sim/run_scenarios.py)
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_INPUT_DIR = REPO_ROOT / "data" / "input"
OBS_DIR = DATA_INPUT_DIR / "observations"
WINDS_DIR = DATA_INPUT_DIR / "winds"

SCENARIO_ROOT = REPO_ROOT / "data" / "scenarios"        # created inputs live here
EXPERIMENT_ROOT = REPO_ROOT / "data" / "experiments"    # simulate.py outputs live here

# Fixed vent elevation (your request)
VENT_ELEVATION_M = 100.0

# Grain-size & diffusion defaults (kept identical across CN and KL unless overridden)
MEDIAN_GS = 0.0
STD_GS = 1.0
ALPHA = 2.0
BETA = 2.0
MIN_GS = -5.0
MAX_GS = 5.0

# CN92 location (your provided coords)
CN_LAT = 12.506
CN_LON = -86.702

# KL24 location (NOTE: Kīlauea longitude is ~155 W; 115 W is almost certainly a typo)
KL_LAT = 19.421
KL_LON = -155.287

# CN92 DRE constants (used to compute mass at top)
CN_DRE_VOL_KM3 = 0.03
CN_DRE_DENSITY_KG_M3 = 1024.0
CN_TOTAL_MASS_KG = CN_DRE_VOL_KM3 * 1e9 * CN_DRE_DENSITY_KG_M3  # 0.03 km^3 -> m^3 -> kg

# Split ratio file (optional; if present we parse it, else we use your provided values)
CN_SPLIT_RATIO_TXT = DATA_INPUT_DIR / "cn92_splitm_ratio.txt"
CN_P_A_DEFAULT = 0.571779
CN_P_B_DEFAULT = 0.428221

# Literature-informed plume heights (m)
CN_PLUME_A_M = 7500.0   # energetic pulses to ~7.5 km
CN_PLUME_B_M = 3500.0   # later weaker phase ~3.5 km
CN_PLUME_AB_M = 7500.0  # use max as a single representative “true” for total

KL_PLUME_M = 9000.0     # USGS summary max ~9 km

# KL mass (pure constant; put your preferred number here)
KL_TOTAL_MASS_KG = 1.0e10

# UTC targets used to pick nearest wind file (script selects nearest available file)
CN_A_WIND_TARGET_UTC = "1992-04-10T06:00:00Z"
CN_B_WIND_TARGET_UTC = "1992-04-14T06:00:00Z"
KL_WIND_TARGET_UTC   = "1924-05-18T21:00:00Z"

# Wind filename patterns expected:
#   cerro_YYYYMMDD_HHMM.dat
#   kilauea_YYYYMMDD_HHMM.dat
WIND_TS_RE = re.compile(r"^(?P<prefix>[a-zA-Z0-9]+)_(?P<ymd>\d{8})_(?P<hm>\d{4})\.(dat|txt)$")


# ============================================================
# UTM conversion (correct) — requires pyproj
# ============================================================

def latlon_to_utm(lat: float, lon: float) -> Tuple[float, float, int]:
    """
    Convert WGS84 lat/lon to UTM easting/northing.
    Returns (easting_m, northing_m, epsg_code).
    """
    try:
        from pyproj import CRS, Transformer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "pyproj is required for correct lat/lon -> UTM conversion.\n"
            "Install with: pip install pyproj\n"
            f"Original import error: {e}"
        )

    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    is_north = lat >= 0.0
    epsg = (32600 + zone) if is_north else (32700 + zone)

    crs_wgs84 = CRS.from_epsg(4326)
    crs_utm = CRS.from_epsg(epsg)
    transformer = Transformer.from_crs(crs_wgs84, crs_utm, always_xy=True)

    easting, northing = transformer.transform(lon, lat)
    return float(easting), float(northing), int(epsg)


# ============================================================
# Wind selection
# ============================================================

def _parse_utc_z(ts: str) -> datetime:
    # ts like 1992-04-10T06:00:00Z
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)

def _wind_file_datetime(p: Path) -> Optional[datetime]:
    m = WIND_TS_RE.match(p.name)
    if not m:
        return None
    ymd = m.group("ymd")
    hm = m.group("hm")
    dt = datetime.strptime(ymd + hm, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    return dt

def find_nearest_wind(prefix: str, target_utc: str, winds_dir: Path) -> Path:
    target = _parse_utc_z(target_utc)
    winds_dir = Path(winds_dir)

    candidates = sorted([p for p in winds_dir.glob(f"{prefix}_*") if p.is_file()])
    scored: List[Tuple[float, Path]] = []
    for p in candidates:
        dt = _wind_file_datetime(p)
        if dt is None:
            continue
        scored.append((abs((dt - target).total_seconds()), p))

    if not scored:
        raise FileNotFoundError(f"No wind files matching prefix '{prefix}_*' in {winds_dir}")

    scored.sort(key=lambda x: x[0])
    return scored[0][1]


# ============================================================
# CN split ratio parsing
# ============================================================

def read_cn_split_ratio() -> Tuple[float, float]:
    if CN_SPLIT_RATIO_TXT.exists():
        txt = CN_SPLIT_RATIO_TXT.read_text()
        # look for lines like: p_a = 0.571779
        m_a = re.search(r"p_a\s*=\s*([0-9]*\.[0-9]+)", txt)
        m_b = re.search(r"p_b\s*=\s*([0-9]*\.[0-9]+)", txt)
        if m_a and m_b:
            return float(m_a.group(1)), float(m_b.group(1))
    return CN_P_A_DEFAULT, CN_P_B_DEFAULT


# ============================================================
# Scenario definitions
# ============================================================

@dataclass(frozen=True)
class Scenario:
    name: str
    obs_csv: Path
    volcano: str
    lat: float
    lon: float
    plume_height_m: float
    eruption_mass_kg: float
    wind_prefix: str
    wind_target_utc: str


def build_scenarios() -> List[Scenario]:
    p_a, p_b = read_cn_split_ratio()
    cn_mass_a = CN_TOTAL_MASS_KG * p_a
    cn_mass_b = CN_TOTAL_MASS_KG * p_b

    return [
        Scenario(
            name="cn92_a",
            obs_csv=OBS_DIR / "cn92_a.csv",
            volcano="CN",
            lat=CN_LAT, lon=CN_LON,
            plume_height_m=CN_PLUME_A_M,
            eruption_mass_kg=cn_mass_a,
            wind_prefix="cerro",
            wind_target_utc=CN_A_WIND_TARGET_UTC,
        ),
        Scenario(
            name="cn92_a_plus_splitm",
            obs_csv=OBS_DIR / "cn92_a_plus_splitm.csv",
            volcano="CN",
            lat=CN_LAT, lon=CN_LON,
            plume_height_m=CN_PLUME_A_M,
            eruption_mass_kg=cn_mass_a,
            wind_prefix="cerro",
            wind_target_utc=CN_A_WIND_TARGET_UTC,
        ),
        Scenario(
            name="cn92_ab_total",
            obs_csv=OBS_DIR / "cn92_ab_total.csv",
            volcano="CN",
            lat=CN_LAT, lon=CN_LON,
            plume_height_m=CN_PLUME_AB_M,
            eruption_mass_kg=CN_TOTAL_MASS_KG,
            wind_prefix="cerro",
            wind_target_utc=CN_A_WIND_TARGET_UTC,  # your earlier rule: total uses A wind
        ),
        Scenario(
            name="cn92_b",
            obs_csv=OBS_DIR / "cn92_b.csv",
            volcano="CN",
            lat=CN_LAT, lon=CN_LON,
            plume_height_m=CN_PLUME_B_M,
            eruption_mass_kg=cn_mass_b,
            wind_prefix="cerro",
            wind_target_utc=CN_B_WIND_TARGET_UTC,
        ),
        Scenario(
            name="cn92_b_plus_splitm",
            obs_csv=OBS_DIR / "cn92_b_plus_splitm.csv",
            volcano="CN",
            lat=CN_LAT, lon=CN_LON,
            plume_height_m=CN_PLUME_B_M,
            eruption_mass_kg=cn_mass_b,
            wind_prefix="cerro",
            wind_target_utc=CN_B_WIND_TARGET_UTC,
        ),
        Scenario(
            name="kl24_1924",
            obs_csv=OBS_DIR / "kl24_1924_profiles_tephra2.csv",
            volcano="KL",
            lat=KL_LAT, lon=KL_LON,
            plume_height_m=KL_PLUME_M,
            eruption_mass_kg=KL_TOTAL_MASS_KG,
            wind_prefix="kilauea",
            wind_target_utc=KL_WIND_TARGET_UTC,
        ),
    ]


# ============================================================
# Orchestration: setup + run simulate
# ============================================================

def run_setup(s: Scenario, dry_run: bool = False) -> Tuple[Path, Path]:
    if not s.obs_csv.exists():
        raise FileNotFoundError(f"Missing observation file: {s.obs_csv}")

    easting, northing, epsg = latlon_to_utm(s.lat, s.lon)

    wind_path = find_nearest_wind(s.wind_prefix, s.wind_target_utc, WINDS_DIR)
    scenario_dir = SCENARIO_ROOT / s.name
    out_root = scenario_dir  # setup.py will create input/ and config/ under this

    cmd = [
        "python", "-m", "scripts.sim.setup",
        str(s.obs_csv),
        "--out-root", str(out_root),

        "--vent-easting", f"{easting}",
        "--vent-northing", f"{northing}",
        "--vent-elev", f"{VENT_ELEVATION_M}",

        "--plume-height", f"{s.plume_height_m}",
        "--eruption-mass", f"{s.eruption_mass_kg}",

        "--median-gs", f"{MEDIAN_GS}",
        "--std-gs", f"{STD_GS}",
        "--alpha", f"{ALPHA}",
        "--beta", f"{BETA}",
        "--min-gs", f"{MIN_GS}",
        "--max-gs", f"{MAX_GS}",

        "--wind-src", str(wind_path),

        "--elev-default", f"{VENT_ELEVATION_M}",
    ]

    print(f"\n[SCENARIO] {s.name}")
    print(f"  lat/lon: ({s.lat}, {s.lon}) -> UTM EPSG:{epsg} e={easting:.1f} n={northing:.1f}")
    print(f"  plume_height_m: {s.plume_height_m}")
    print(f"  eruption_mass_kg: {s.eruption_mass_kg:.6e}  (lnM={math.log(s.eruption_mass_kg):.3f})")
    print(f"  wind target: {s.wind_target_utc}")
    print(f"  wind chosen: {wind_path.name}")

    if dry_run:
        print("  [DRY RUN] setup command:", " ".join(cmd))
    else:
        scenario_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, check=True)

    return (out_root / "input", EXPERIMENT_ROOT / s.name)


def run_simulate(
    input_dir: Path,
    output_dir: Path,
    config_module: str,
    plot_winds: bool,
    dry_run: bool = False,
    invert_n_params: int | None = None,
    logm_prior_std: float | None = None,
) -> None:
    cmd = [
        sys.executable, "-m", "scripts.sim.simulate",
        "--config-module", config_module,
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
    ]
    if plot_winds:
        cmd.append("--plot-winds")

    if invert_n_params is not None:
        cmd += ["--invert-n-params", str(int(invert_n_params))]

    if logm_prior_std is not None:
        cmd += ["--logm-prior-std", str(float(logm_prior_std))]

    if dry_run:
        print("  [DRY RUN] simulate command:", " ".join(cmd))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Create scenario input trees and run inversion simulations for CN92 + KL24.")
    parser.add_argument("--config-module", type=str, default="scripts.sim.exp_config", help="Experiment grid config module.")
    parser.add_argument("--only", type=str, default=None, help="Run only one scenario by name (e.g., cn92_a).")
    parser.add_argument("--setup-only", action="store_true", help="Only build scenario input trees; do not run simulate.py.")
    parser.add_argument("--plot-winds", action="store_true", help="Pass through to simulate.py to generate wind plots.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands but do not execute.")
    args = parser.parse_args(argv)

    # Load the experiment config module so we can pass through default
    # simulation settings (e.g. 2- vs 4-parameter inversion) consistently.
    EXP = importlib.import_module(args.config_module)
    invert_n_params = getattr(EXP, "INVERT_N_PARAMS", None)
    logm_prior_std = getattr(EXP, "LOGM_PRIOR_STD", None)

    scenarios = build_scenarios()
    if args.only:
        scenarios = [s for s in scenarios if s.name == args.only]
        if not scenarios:
            raise ValueError(f"--only={args.only} not found among scenarios.")

    for s in scenarios:
        input_dir, out_dir = run_setup(s, dry_run=args.dry_run)
        if not args.setup_only:
            run_simulate(
                input_dir=input_dir,
                output_dir=out_dir,
                config_module=args.config_module,
                plot_winds=args.plot_winds,
                dry_run=args.dry_run,
                invert_n_params=invert_n_params,
                logm_prior_std=logm_prior_std,
            )

    print("\n[DONE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
