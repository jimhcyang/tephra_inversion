from pathlib import Path
from typing  import Dict, Union
import numpy as np
import pandas as pd


_FIXED_ROWS: Dict[str, float] = {
    "alpha":                3.4,   # beta-alpha
    "beta":                 2.0,   # beta-beta
    "max_grainsize":       -6,
    "min_grainsize":        6,
    "median_grainsize":    -1,
    "std_grainsize":        1.5,
    "eddy_const":           0.04,
    "diffusion_coefficient":5000,
    "fall_time_threshold": 1419,
    "lithic_density":      2500,
    "pumice_density":      1100,
    "col_steps":            100,
    "part_steps":           100,
    "plume_model":            2,
}

def write_tephra2_conf(easting: float,
                       northing: float,
                       elevation: float,
                       plume_height: float,
                       eruption_mass: float,
                       out_path: Union[str, Path] = "data/input/tephra2.conf"
                       ) -> Path:
    """
    Build a *19-row* `tephra2.conf`, now including VENT_* lines.
    Values are written at full precision; Tephra2 ignores comments.
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w") as f:
        for k, v in _FIXED_ROWS.items():
            f.write(f"{k.upper()} {v:.6f}\n")

        f.write(f"VENT_EASTING   {easting:.3f}\n")
        f.write(f"VENT_NORTHING  {northing:.3f}\n")
        f.write(f"VENT_ELEVATION {elevation:.3f}\n")

        f.write(f"PLUME_HEIGHT   {plume_height:.1f}\n")
        f.write(f"ERUPTION_MASS  {eruption_mass:.6f}\n")

    print(f"tephra2.conf written → {p}")
    return p


def write_esp_input(plume_height: float,
                    log_m_e: float,
                    out_path: Union[str, Path] = "data/input/esp_input.csv"
                    ) -> Path:
    """
    Make `esp_input.csv` with natural-log mass and Gaussian priors.
    """
    df = pd.DataFrame(
        [
            ["column_height", plume_height, "Gaussian", plume_height, 2000, 200],
            ["log_m",        log_m_e,      "Gaussian", log_m_e,      1,  0.1],
        ],
        columns=[
            "variable_name", "initial_val", "prior_type",
            "prior_para_a", "prior_para_b", "draw_scale"
        ],
    )
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print(f"esp_input.csv written → {p}")
    return p

def get_mcmc_setup(plume_height: float,
                   log_m_e: float) -> Dict[str, np.ndarray]:
    """
    Return the arrays that `scripts/core/mcmc.py` expects, *in natural log*.
    """
    return {
        "initial_values":   np.array([plume_height, log_m_e]),
        "prior_type":       np.array(["Gaussian",   "Gaussian"]),
        "prior_parameters": np.array([[plume_height, 2000],   # μ, σ
                                      [log_m_e,      1 ]]),
        "draw_scale":       np.array([200, 0.1]),
    }
