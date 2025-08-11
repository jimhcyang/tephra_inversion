# ─────────────────────────────────────────────────────────────
# scripts/core/tephra2_utils.py  · safe exp/log and IO guards
# ─────────────────────────────────────────────────────────────
from __future__ import annotations
import os
import subprocess
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Union, Optional

logger = logging.getLogger(__name__)

# Physical-ish safety rails (broad on purpose)
_PLUME_MIN = 100.0          # meters
_PLUME_MAX = 5.0e4          # meters
_LOGM_MIN  = np.log(1e6)    # ln(kg)
_LOGM_MAX  = np.log(1e14)   # ln(kg)

def _safe_float(x: float, default: float) -> float:
    try:
        xf = float(x)
        if not np.isfinite(xf):
            return default
        return xf
    except Exception:
        return default

def update_config_file(plume_vec: np.ndarray, conf_path: Union[Path, str]) -> None:
    """
    Update *only* PLUME_HEIGHT and ERUPTION_MASS in `tephra2.conf`.

    plume_vec[0] = plume height [m]
    plume_vec[1] = ln(eruption mass [kg])
    """
    # sanitize inputs
    plume_height = _safe_float(plume_vec[0], 7500.0)
    plume_height = float(np.clip(plume_height, _PLUME_MIN, _PLUME_MAX))

    # protect exp() from overflow and non-finite
    log_mass = _safe_float(plume_vec[1], np.log(5e10))
    log_mass = float(np.clip(log_mass, _LOGM_MIN, _LOGM_MAX))
    eruption_mass = float(np.exp(log_mass))

    conf_path = Path(conf_path)
    lines = conf_path.read_text().splitlines(keepends=True)

    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("/*"):
            continue
        key = s.split()[0]
        if key == "PLUME_HEIGHT":
            lines[i] = f"PLUME_HEIGHT   {plume_height:.6f}\n"
        elif key == "ERUPTION_MASS":
            lines[i] = f"ERUPTION_MASS  {eruption_mass:.6f}\n"

    conf_path.write_text("".join(lines))
    logger.debug("→ conf updated: height=%.1f m, lnM=%.2f", plume_height, log_mass)

def ensure_sites_format(sites_csv: Path) -> None:
    """Re-write sites file as space-delimited E N Z."""
    df = pd.read_csv(sites_csv, sep=r"[,\s]+", engine="python", header=None)
    if df.shape[1] != 3:
        raise ValueError(f"{sites_csv} must have 3 columns (E,N,Z); got {df.shape[1]}")
    df.to_csv(sites_csv, sep=" ", header=False, index=False, float_format="%.3f")

def run_tephra2(plume_vec: np.ndarray,
                conf_path: Union[Path, str],
                sites_csv: Union[Path, str],
                tephra2_path: Optional[Union[Path, str]] = None,
                wind_path: Optional[Union[Path, str]] = None,
                output_path: Optional[Union[Path, str]] = None,
                silent: bool = True) -> np.ndarray:
    """
    Edit tephra2.conf, ensure sites file OK, run Tephra2 executable,
    return deposit column (kg m⁻²).
    """
    conf_path = Path(conf_path)
    sites_csv = Path(sites_csv)

    # Defaults
    if tephra2_path is None:
        tephra2_path = Path(__file__).resolve().parents[2] / "Tephra2" / "tephra2_2020"
    else:
        tephra2_path = Path(tephra2_path)
    if wind_path is None:
        wind_path = conf_path.parent / "wind.txt"
    else:
        wind_path = Path(wind_path)
    if output_path is None:
        output_path = conf_path.parent / "tephra2_output_mcmc.txt"
    else:
        output_path = Path(output_path)

    # Update configuration and ensure sites format
    update_config_file(plume_vec, conf_path)
    ensure_sites_format(sites_csv)

    # Check if files exist
    for path, desc in [
        (tephra2_path, "Tephra2 executable"),
        (conf_path, "Config file"),
        (sites_csv, "Sites file"),
        (wind_path, "Wind file"),
    ]:
        if not os.path.exists(path):
            logger.error(f"{desc} not found: {path}")
            raise FileNotFoundError(f"{desc} not found: {path}")

    # Run tephra2
    with open(output_path, "w") as fout:
        res = subprocess.run([str(tephra2_path), str(conf_path), str(sites_csv), str(wind_path)],
                             stdout=fout, stderr=subprocess.PIPE, text=True)

    # Check if the run was successful
    if res.returncode != 0 or output_path.stat().st_size == 0:
        raise RuntimeError(f"Tephra2 failed (exit {res.returncode}).\n--- STDERR ---\n{res.stderr}")

    # Read and return the output
    data = np.genfromtxt(output_path)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return data[:, 3].astype(float)  # mass-loading column
