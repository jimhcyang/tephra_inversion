# scripts/config_io.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Union
import importlib.util
import numpy as np
import pandas as pd

# Number of decimals when writing human-facing files
DECIMAL_PRECISION = 2

# ---------------------------------------------------------------------
# Config loader (reads config/default_config.py -> DEFAULT_CONFIG)
# ---------------------------------------------------------------------
def load_config():
    spec = importlib.util.spec_from_file_location(
        "default_config",
        "config/default_config.py"
    )
    config_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(config_module)  # type: ignore[attr-defined]
    return config_module.DEFAULT_CONFIG

# Cache the config so helpers don’t keep re-reading the file
CONFIG = load_config()

# Pull commonly-used sections
_FIXED_ROWS: Dict[str, float] = CONFIG["parameters"]["fixed"]
_VARIABLE_PARAMS: Dict[str, dict] = CONFIG["parameters"]["variable"]

# ---------------------------------------------------------------------
# Writers for Tephra2 inputs
# ---------------------------------------------------------------------
def write_tephra2_conf(
    easting: float,
    northing: float,
    elevation: float,
    out_path: Union[str, Path] = "data/input/tephra2.conf"
) -> Path:
    """
    Build a minimal `tephra2.conf` containing:
      - fixed rows from config
      - VENT_* (from args)
      - PLUME_HEIGHT and ERUPTION_MASS (from variable params initial values)
    Values are formatted with DECIMAL_PRECISION for readability.
    """
    plume_height = float(_VARIABLE_PARAMS["column_height"]["initial_val"])
    eruption_mass = float(_VARIABLE_PARAMS["eruption_mass"]["initial_val"])

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w") as f:
        # fixed section (keys are already Tephra2-style in your config)
        for k, v in _FIXED_ROWS.items():
            f.write(f"{k.upper():<20} {v:.{DECIMAL_PRECISION}f}\n")

        # vent + core ESPs
        f.write(f"{'VENT_EASTING':<20} {easting:.{DECIMAL_PRECISION}f}\n")
        f.write(f"{'VENT_NORTHING':<20} {northing:.{DECIMAL_PRECISION}f}\n")
        f.write(f"{'VENT_ELEVATION':<20} {elevation:.{DECIMAL_PRECISION}f}\n")
        f.write(f"{'PLUME_HEIGHT':<20} {plume_height:.{DECIMAL_PRECISION}f}\n")
        f.write(f"{'ERUPTION_MASS':<20} {eruption_mass:.{DECIMAL_PRECISION}f}\n")

    print(f"tephra2.conf written → {p}")
    return p


def write_esp_input(
    easting: float | None = None,
    northing: float | None = None,
    elevation: float | None = None,
    out_path: Union[str, Path] = "data/input/esp_input.csv"
) -> Path:
    """
    Write an `esp_input.csv` reflecting current config.
    - Uses 'log_m' (natural log of eruption_mass initial_val) as a variable row.
    - Maps fixed parameter keys to legacy ESP column names.
    """
    rows = []

    # Variable params: convert 'eruption_mass' to 'log_m'
    for name, cfg in _VARIABLE_PARAMS.items():
        if name == "eruption_mass":
            log_m = float(np.log(cfg["initial_val"]))
            if cfg["prior_type"] == "Gaussian":
                a = log_m
                b = float(cfg["prior_para_b"])
            else:
                a = float(cfg["prior_para_a"])
                b = float(cfg["prior_para_b"])
            rows.append([
                "log_m",
                round(log_m, DECIMAL_PRECISION),
                cfg["prior_type"],
                round(a, DECIMAL_PRECISION),
                round(b, DECIMAL_PRECISION),
                round(float(cfg["draw_scale"]), DECIMAL_PRECISION),
            ])
        else:
            a = float(cfg["prior_para_a"]) if cfg["prior_type"] == "Gaussian" else float(cfg["prior_para_a"])
            b = float(cfg["prior_para_b"])
            rows.append([
                name,
                round(float(cfg["initial_val"]), DECIMAL_PRECISION),
                cfg["prior_type"],
                round(a, DECIMAL_PRECISION),
                round(b, DECIMAL_PRECISION),
                round(float(cfg["draw_scale"]), DECIMAL_PRECISION),
            ])

    # Fixed rows mapping to ESP names (legacy expected headers)
    param_mapping = {
        "alpha": "alpha",
        "beta": "beta",
        "max_grainsize": "gs_max",
        "min_grainsize": "gs_min",
        "median_grainsize": "gs_med",
        "std_grainsize": "gs_sd",
        "eddy_const": "edy_const",
        "diffusion_coefficient": "diffu_coef",
        "fall_time_threshold": "fall_time_thre",
        "lithic_density": "lithic_rou",
        "pumice_density": "pumice_rou",
        "col_steps": "column_steps",
        "part_steps": "particle_bins",
        "plume_model": "plume_model",
    }
    for internal_name, value in _FIXED_ROWS.items():
        esp_name = param_mapping.get(internal_name, internal_name)
        rows.append([esp_name, round(float(value), DECIMAL_PRECISION), "Fixed", "", "", ""])

    # Optional: vent xyz
    if easting is not None:
        rows.append(["vent_x", round(float(easting), DECIMAL_PRECISION), "Fixed", "", "", ""])
    if northing is not None:
        rows.append(["vent_y", round(float(northing), DECIMAL_PRECISION), "Fixed", "", "", ""])
    if elevation is not None:
        rows.append(["vent_z", round(float(elevation), DECIMAL_PRECISION), "Fixed", "", "", ""])

    df = pd.DataFrame(
        rows,
        columns=["variable_name", "initial_val", "prior_type", "prior_para_a", "prior_para_b", "draw_scale"],
    )
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print(f"esp_input.csv written → {p}")
    return p

# ---------------------------------------------------------------------
# Helper for MCMC arrays (natural‑log convention for mass)
# ---------------------------------------------------------------------
def get_mcmc_setup() -> Dict[str, np.ndarray]:
    """
    Build the arrays that scripts/core/mcmc.py expects.
      - Eruption mass is provided as log_m
      - Preserves your prior types and scales
    Returns:
      {
        "initial_values": np.ndarray,
        "prior_type": np.ndarray(dtype=object),
        "prior_parameters": np.ndarray(shape=(P,2)),
        "draw_scale": np.ndarray
      }
    """
    initial_values: list[float] = []
    prior_types: list[str] = []
    prior_parameters: list[list[float]] = []
    draw_scales: list[float] = []

    for name, cfg in _VARIABLE_PARAMS.items():
        if name == "eruption_mass":
            log_m0 = float(np.log(cfg["initial_val"]))
            initial_values.append(log_m0)
            prior_types.append(cfg["prior_type"])
            if cfg["prior_type"] == "Gaussian":
                prior_parameters.append([log_m0, float(cfg["prior_para_b"])])
            else:
                prior_parameters.append([float(cfg["prior_para_a"]), float(cfg["prior_para_b"])])
            draw_scales.append(float(cfg["draw_scale"]))
        else:
            initial_values.append(float(cfg["initial_val"]))
            prior_types.append(cfg["prior_type"])
            prior_parameters.append([float(cfg["prior_para_a"]), float(cfg["prior_para_b"])])
            draw_scales.append(float(cfg["draw_scale"]))

    return {
        "initial_values": np.array(initial_values, dtype=float),
        "prior_type": np.array(prior_types, dtype=object),
        "prior_parameters": np.array(prior_parameters, dtype=float),
        "draw_scale": np.array(draw_scales, dtype=float),
    }