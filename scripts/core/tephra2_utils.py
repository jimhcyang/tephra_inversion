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
    """Update key Tephra2 ESP lines in a ``tephra2.conf`` file.

    Parameter conventions used throughout this repository:

    - plume_vec[0] = plume height [m]
    - plume_vec[1] = **ln(eruption mass [kg])**  (natural log)

    Optional (if provided; enables 4-parameter inversion):

    - plume_vec[2] = TGSD median grain size (PHI)  -> ``MEDIAN_GRAINSIZE``
    - plume_vec[3] = TGSD std grain size (PHI, >0) -> ``STD_GRAINSIZE``

    Notes
    -----
    * If ``MEDIAN_GRAINSIZE`` / ``STD_GRAINSIZE`` are missing from the config file,
      they will be appended.
    * We also try to respect ``MAX_GRAINSIZE`` and ``MIN_GRAINSIZE`` (PHI) if present,
      by clipping the median into that range. ``STD_GRAINSIZE`` is lower-bounded
      to avoid invalid values.
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

    # Attempt to read grain bounds from file (PHI)
    max_phi = None
    min_phi = None
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("/*"):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        if parts[0] == "MAX_GRAINSIZE":
            try:
                max_phi = float(parts[1])
            except Exception:
                max_phi = None
        elif parts[0] == "MIN_GRAINSIZE":
            try:
                min_phi = float(parts[1])
            except Exception:
                min_phi = None

    phi_lo = None
    phi_hi = None
    if max_phi is not None and min_phi is not None:
        phi_lo = float(min(max_phi, min_phi))
        phi_hi = float(max(max_phi, min_phi))

    # Optional grain params (PHI)
    med_phi = None
    std_phi = None
    if len(plume_vec) >= 3:
        med_phi = _safe_float(plume_vec[2], 0.0)
        if phi_lo is not None and phi_hi is not None:
            med_phi = float(np.clip(med_phi, phi_lo, phi_hi))
    if len(plume_vec) >= 4:
        std_phi = _safe_float(plume_vec[3], 2.0)
        # Tephra2 expects STD_GRAINSIZE > 0
        std_phi = float(max(std_phi, 0.05))

    # Replace existing keys if present
    found = {"PLUME_HEIGHT": False, "ERUPTION_MASS": False,
             "MEDIAN_GRAINSIZE": False, "STD_GRAINSIZE": False}

    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("/*"):
            continue
        key = s.split()[0]
        if key == "PLUME_HEIGHT":
            lines[i] = f"PLUME_HEIGHT   {plume_height:.6f}\n"
            found["PLUME_HEIGHT"] = True
        elif key == "ERUPTION_MASS":
            lines[i] = f"ERUPTION_MASS  {eruption_mass:.6f}\n"
            found["ERUPTION_MASS"] = True
        elif key == "MEDIAN_GRAINSIZE" and med_phi is not None:
            lines[i] = f"MEDIAN_GRAINSIZE {med_phi:.6f}\n"
            found["MEDIAN_GRAINSIZE"] = True
        elif key == "STD_GRAINSIZE" and std_phi is not None:
            lines[i] = f"STD_GRAINSIZE {std_phi:.6f}\n"
            found["STD_GRAINSIZE"] = True

    # Append missing optional keys (keeps forward-compatible with templates)
    if med_phi is not None and not found["MEDIAN_GRAINSIZE"]:
        lines.append(f"MEDIAN_GRAINSIZE {med_phi:.6f}\n")
    if std_phi is not None and not found["STD_GRAINSIZE"]:
        lines.append(f"STD_GRAINSIZE {std_phi:.6f}\n")

    conf_path.write_text("".join(lines))
    logger.debug(
        "→ conf updated: height=%.1f m, lnM=%.2f%s%s",
        plume_height,
        log_mass,
        "" if med_phi is None else f", medPhi={med_phi:.2f}",
        "" if std_phi is None else f", stdPhi={std_phi:.2f}",
    )

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
