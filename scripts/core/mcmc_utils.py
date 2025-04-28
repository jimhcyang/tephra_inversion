# ─────────────────────────────────────────────────────────────
# scripts/core/mcmc_utils.py
# Keyword-safe editor for tephra2.conf (plume-only version)
# ─────────────────────────────────────────────────────────────
from pathlib import Path
import numpy as np
import logging

log = logging.getLogger(__name__)

def changing_variable(plume_vec: np.ndarray,
                      conf_path: Path | str):
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
    plume_height  = float(plume_vec[0])
    eruption_mass = float(np.exp(plume_vec[1]))   # ln → kg

    conf_path = Path(conf_path)
    lines = conf_path.read_text().splitlines(keepends=True)

    for i, ln in enumerate(lines):
        key = ln.split()[0]
        if key == "PLUME_HEIGHT":
            lines[i] = f"PLUME_HEIGHT   {plume_height:.6f}\n"
        elif key == "ERUPTION_MASS":
            lines[i] = f"ERUPTION_MASS  {eruption_mass:.6f}\n"

    conf_path.write_text("".join(lines))
    log.debug("→ conf updated: height=%.1f m, mass=%.2e kg", plume_height, eruption_mass)
