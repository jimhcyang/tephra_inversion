from pathlib import Path
from typing  import Dict, Union
import numpy as np
import pandas as pd
import importlib.util
import sys

# Global configuration
DECIMAL_PRECISION = 2  # Number of decimal places for output values

# Import DEFAULT_CONFIG from config/default_config.py
def load_config():
    spec = importlib.util.spec_from_file_location(
        "default_config", 
        "config/default_config.py"
    )
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.DEFAULT_CONFIG

# Load the configuration
CONFIG = load_config()

# Get parameter dictionaries from config
_FIXED_ROWS = CONFIG["parameters"]["fixed"]
_VARIABLE_PARAMS = CONFIG["parameters"]["variable"]

def write_tephra2_conf(easting: float,
                       northing: float,
                       elevation: float,
                       out_path: Union[str, Path] = "data/input/tephra2.conf"
                       ) -> Path:
    """
    Build a *19-row* `tephra2.conf`, now including VENT_* lines.
    Values are written at full precision; Tephra2 ignores comments.
    """
    # Get plume_height and eruption_mass from variable params
    plume_height = _VARIABLE_PARAMS["column_height"]['initial_val']
    eruption_mass = _VARIABLE_PARAMS["eruption_mass"]['initial_val']
    
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w") as f:
        for k, v in _FIXED_ROWS.items():
            f.write(f"{k.upper()} {v:.{DECIMAL_PRECISION}f}\n")

        f.write(f"VENT_EASTING   {easting:.{DECIMAL_PRECISION}f}\n")
        f.write(f"VENT_NORTHING  {northing:.{DECIMAL_PRECISION}f}\n")
        f.write(f"VENT_ELEVATION {elevation:.{DECIMAL_PRECISION}f}\n")

        f.write(f"PLUME_HEIGHT   {plume_height:.{DECIMAL_PRECISION}f}\n")
        f.write(f"ERUPTION_MASS  {eruption_mass:.{DECIMAL_PRECISION}f}\n")

    print(f"tephra2.conf written → {p}")
    return p


def write_esp_input(easting: float = None,
                    northing: float = None,
                    elevation: float = None,
                    out_path: Union[str, Path] = "data/input/esp_input.csv"
                    ) -> Path:
    """
    Make `esp_input.csv` with all parameters from the config file.
    """
    rows = []
    
    # Add all variable parameters from config
    for param_name, param_config in _VARIABLE_PARAMS.items():
        # Special case for log_m (calculated from eruption_mass)
        if param_name == "eruption_mass":
            log_m_value = np.log(param_config["initial_val"])
            # Use log_m as the parameter name for esp_input
            rows.append([
                "log_m",
                round(log_m_value, DECIMAL_PRECISION),
                param_config["prior_type"],
                round(log_m_value, DECIMAL_PRECISION) if param_config["prior_type"] == "Gaussian" else round(param_config["prior_para_a"], DECIMAL_PRECISION),
                round(param_config["prior_para_b"], DECIMAL_PRECISION),
                round(param_config["draw_scale"], DECIMAL_PRECISION)
            ])
        else:
            rows.append([
                param_name,
                round(param_config["initial_val"], DECIMAL_PRECISION),
                param_config["prior_type"],
                round(param_config["prior_para_a"], DECIMAL_PRECISION) if param_config["prior_type"] == "Gaussian" else param_config["prior_para_a"],
                round(param_config["prior_para_b"], DECIMAL_PRECISION),
                round(param_config["draw_scale"], DECIMAL_PRECISION)
            ])
    
    # Add fixed rows from _FIXED_ROWS dictionary
    # Map internal parameter names to ESP input names
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
        rows.append([esp_name, round(value, DECIMAL_PRECISION), "Fixed", "", "", ""])
    
    # Add vent location parameters if provided
    if easting is not None:
        rows.append(["vent_x", round(easting, DECIMAL_PRECISION), "Fixed", "", "", ""])
    if northing is not None:
        rows.append(["vent_y", round(northing, DECIMAL_PRECISION), "Fixed", "", "", ""])
    if elevation is not None:
        rows.append(["vent_z", round(elevation, DECIMAL_PRECISION), "Fixed", "", "", ""])
    
    df = pd.DataFrame(
        rows,
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

def get_mcmc_setup() -> Dict[str, np.ndarray]:
    """
    Return the arrays that `scripts/core/mcmc.py` expects, *in natural log*.
    """
    initial_values = []
    prior_types = []
    prior_parameters = []
    draw_scales = []
    
    # Process all variable parameters
    for param_name, param_config in _VARIABLE_PARAMS.items():
        # Special case for eruption_mass -> log_m conversion
        if param_name == "eruption_mass":
            log_m_value = np.log(param_config["initial_val"])
            initial_values.append(log_m_value)
            prior_types.append(param_config["prior_type"])
            
            # Handle different prior types properly
            if param_config["prior_type"] == "Gaussian":
                prior_parameters.append([log_m_value, param_config["prior_para_b"]])
            else:  # Uniform or other prior types
                prior_parameters.append([param_config["prior_para_a"], param_config["prior_para_b"]])
                
            draw_scales.append(param_config["draw_scale"])
        else:
            initial_values.append(param_config["initial_val"])
            prior_types.append(param_config["prior_type"])
            
            # Handle different prior types properly
            if param_config["prior_type"] == "Gaussian":
                prior_parameters.append([param_config["prior_para_a"], param_config["prior_para_b"]])
            else:  # Uniform or other prior types
                prior_parameters.append([param_config["prior_para_a"], param_config["prior_para_b"]])
                
            draw_scales.append(param_config["draw_scale"])
    
    return {
        "initial_values": np.array(initial_values),
        "prior_type": np.array(prior_types),
        "prior_parameters": np.array(prior_parameters),
        "draw_scale": np.array(draw_scales),
    }
