# scripts/data_handling/esp_config.py

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Optional

class ESPConfig:
    """
    Eruption Source Parameter configuration handler.
    Simplified to focus on plume height and eruption mass.
    """
    def __init__(self, output_dir: Union[str, Path] = "data/input"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default parameter ranges
        self.default_ranges = {
            "plume_height": [5000, 20000],  # meters
            "eruption_mass": [1e20, 1e25],  # kg
        }
        
        # Default fixed parameters (from Kirishima example)
        self.default_fixed = {
            "alpha": 3.0,                  # Beta distribution parameter alpha
            "beta": 2.0,                   # Beta distribution parameter beta
            "max_grainsize": -6,           # Maximum grain size (phi)
            "min_grainsize": 6,            # Minimum grain size (phi)
            "median_grainsize": -1,        # Median grain size (phi)
            "std_grainsize": 1.5,          # Std deviation of grain size (phi)
            "eddy_const": 0.04,            # Eddy constant
            "diffusion_coefficient": 5000, # Diffusion coefficient
            "fall_time_threshold": 1419,   # Fall time threshold
            "lithic_density": 2500,        # Lithic density (kg/m³)
            "pumice_density": 1100,        # Pumice density (kg/m³)
            "col_steps": 100,              # Column steps
            "part_steps": 100,             # Particle steps
            "plume_model": 2               # Plume model
        }
    
    def create_default_config(self, config_path: Union[str, Path] = "data/input/tephra2.conf") -> None:
        """
        Create a default Tephra2 configuration file.
        
        Parameters
        ----------
        config_path : str or Path
            Path to save the configuration file
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            # Write fixed parameters
            for param, value in self.default_fixed.items():
                f.write(f"{param.upper()} {value:.6f}\n")
            
            # Add placeholder values for parameters to be estimated
            f.write(f"PLUME_HEIGHT 10000.000000\n")
            f.write(f"ERUPTION_MASS 1.000000e+22\n")
    
    def get_mcmc_parameters(self) -> Dict:
        """
        Get parameters for MCMC inversion.
        
        Returns
        -------
        Dict
            Dictionary with initial values, prior types, prior parameters, and draw scales
        """
        return {
            "initial_values": np.array([10000, np.log10(1e22)]),  # plume_height, log_mass
            "prior_type": np.array(["Uniform", "Uniform"]),
            "prior_parameters": np.array([
                [5000, 20000],    # plume_height: min, max
                [20, 25]          # log_mass: min, max
            ]),
            "draw_scale": np.array([500, 0.5])  # proposal step sizes
        }