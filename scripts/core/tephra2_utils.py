"""
Tephra2 utilities for file management and interface functions.
Consolidated functionality from tephra2_interface.py and mcmc_utils.py
"""
import os
import subprocess
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Union, Dict, Optional, List, Tuple, Any

logger = logging.getLogger(__name__)

def update_config_file(plume_vec: np.ndarray, conf_path: Union[Path, str]) -> None:
    """
    Update *only* PLUME_HEIGHT and ERUPTION_MASS in `tephra2.conf`.

    Parameters
    ----------
    plume_vec : np.ndarray
        1-D array whose first element is plume height [m] and whose
        second element is **ln(eruption mass in kg)**.
        Extra elements are ignored here (they stay fixed in the file).
    conf_path : str or Path
        Full path to the Tephra2 configuration file.
    """
    plume_height = float(plume_vec[0])
    eruption_mass = float(np.exp(plume_vec[1]))  # ln → kg

    conf_path = Path(conf_path)
    lines = conf_path.read_text().splitlines(keepends=True)

    for i, ln in enumerate(lines):
        key = ln.split()[0]
        if key == "PLUME_HEIGHT":
            lines[i] = f"PLUME_HEIGHT   {plume_height:.6f}\n"
        elif key == "ERUPTION_MASS":
            lines[i] = f"ERUPTION_MASS  {eruption_mass:.6f}\n"

    conf_path.write_text("".join(lines))
    logger.debug("→ conf updated: height=%.1f m, mass=%.2e kg", plume_height, eruption_mass)


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

    Parameters
    ----------
    plume_vec : np.ndarray
        Vector of parameters with plume height and ln(mass)
    conf_path : Path or str
        Path to the configuration file
    sites_csv : Path or str
        Path to the sites file
    tephra2_path : Path or str, optional
        Path to the tephra2 executable. If not provided, defaults to repo/Tephra2/tephra2_2020
    wind_path : Path or str, optional
        Path to the wind file. If not provided, looks for wind.txt in the same dir as conf_path
    output_path : Path or str, optional
        Path for the output file. If not provided, uses tephra2_output_mcmc.txt
    silent : bool, default=True
        Whether to run silently

    Returns
    -------
    np.ndarray
        Model predictions (mass loading column)
    """
    # Convert all paths to Path objects
    conf_path = Path(conf_path)
    sites_csv = Path(sites_csv)
    
    # Default values
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
    cmd = [str(tephra2_path), str(conf_path), str(sites_csv), str(wind_path)]
    res = subprocess.run(cmd, stdout=open(output_path, "w"),
                        stderr=subprocess.PIPE, text=True)

    # Check if the run was successful
    if res.returncode != 0 or output_path.stat().st_size == 0:
        raise RuntimeError(f"Tephra2 failed (exit {res.returncode}).\n--- STDERR ---\n{res.stderr}")

    # Read and return the output
    data = np.genfromtxt(output_path)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return data[:, 3]  # mass-loading column 