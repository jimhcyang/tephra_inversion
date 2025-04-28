# scripts/core/tephra2_interface.py

import os
import subprocess
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Union, Optional, List, Tuple

# Configure logging
logger = logging.getLogger(__name__)

class Tephra2Interface:
    def __init__(self, 
                 tephra2_path: Union[str, Path] = "Tephra2/tephra2_2020",
                 config_dir: Union[str, Path] = "data/input",
                 output_dir: Union[str, Path] = "data/output"):
        self.tephra2_path = Path(tephra2_path)
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_forward_model(self,
                         vent_location: Tuple[float, float, float],
                         wind_data: pd.DataFrame,
                         params: Dict[str, float],
                         config_path: Optional[Union[str, Path]] = None,
                         sites_path: Optional[Union[str, Path]] = None,
                         wind_path: Optional[Union[str, Path]] = None,
                         output_path: Optional[Union[str, Path]] = None) -> np.ndarray:
        """
        Run forward model with the given parameters.
        
        Parameters
        ----------
        vent_location : Tuple[float, float, float]
            (easting, northing, elevation) coordinates of vent
        wind_data : pd.DataFrame
            Wind data with HEIGHT, SPEED, DIRECTION columns
        params : Dict[str, float]
            Model parameters
        config_path, sites_path, wind_path, output_path : Optional paths
            If not provided, default paths will be used
            
        Returns
        -------
        np.ndarray
            Model predictions
        """
        # Use default paths if not provided
        if config_path is None:
            config_path = self.config_dir / "tephra2.conf"
        if sites_path is None:
            sites_path = self.config_dir / "sites.csv"
        if wind_path is None:
            wind_path = self.config_dir / "wind.txt"
        if output_path is None:
            output_path = self.output_dir / "tephra2_output.txt"
            
        # Ensure paths are Path objects
        config_path = Path(config_path)
        sites_path = Path(sites_path)
        wind_path = Path(wind_path)
        output_path = Path(output_path)
        
        # Check if we need to create/update the config file
        self._write_config_file(config_path, vent_location, params)
        
        # Save wind data if it's a DataFrame
        if isinstance(wind_data, pd.DataFrame) and not wind_path.exists():
            self._write_wind_file(wind_path, wind_data)
            
        # Run tephra2
        return run_tephra2(config_path, sites_path, wind_path, output_path, self.tephra2_path)
    
    def _write_config_file(self, 
                         config_path: Path, 
                         vent_location: Tuple[float, float, float],
                         params: Dict[str, float]) -> None:
        """Write Tephra2 configuration file."""
        with open(config_path, 'w') as f:
            # Write vent location
            f.write(f"VENT_EASTING {vent_location[0]:.6f}\n")
            f.write(f"VENT_NORTHING {vent_location[1]:.6f}\n")
            f.write(f"VENT_ELEVATION {vent_location[2]:.6f}\n")
            
            # Write other parameters
            for name, value in params.items():
                if isinstance(value, float) and value > 1e4:
                    f.write(f"{name.upper()} {value:.6e}\n")
                else:
                    f.write(f"{name.upper()} {value:.6f}\n")
    
    def _write_wind_file(self, wind_path: Path, wind_data: pd.DataFrame) -> None:
        """Write wind data file from DataFrame."""
        with open(wind_path, 'w') as f:
            f.write("#HEIGHT SPEED DIRECTION\n")
            for _, row in wind_data.iterrows():
                f.write(f"{row['HEIGHT']:.1f} {row['SPEED']:.1f} {row['DIRECTION']:.1f}\n")


def changing_variable(input_params, config_path):
    """
    Update the tephra2 configuration file with new parameters.
    
    Args:
        input_params (np.ndarray): Parameters for tephra2
        config_path (str): Path to the configuration file
    """
    try:
        with open(config_path, 'r') as fid:
            s = fid.readlines()
            
        # Update parameters in config file
        s[0] = f'PLUME_HEIGHT {float(input_params[0])}\n'
        s[1] = f'ERUPTION_MASS {float(np.exp(input_params[1]))}\n'
        s[2] = f'ALPHA {float(input_params[2])}\n'
        s[3] = f'BETA {float(input_params[3])}\n'
        s[4] = f'MAX_GRAINSIZE {float(input_params[4])}\n'
        s[5] = f'MIN_GRAINSIZE {float(input_params[5])}\n'
        s[6] = f'MEDIAN_GRAINSIZE {float(input_params[6])}\n'
        s[7] = f'STD_GRAINSIZE {float(input_params[7])}\n'
        s[8] = f'VENT_EASTING {float(input_params[8])}\n'
        s[9] = f'VENT_NORTHING {float(input_params[9])}\n'
        s[10] = f'VENT_ELEVATION {float(input_params[10])}\n'
        s[11] = f'EDDY_CONST {float(input_params[11])}\n'
        s[12] = f'DIFFUSION_COEFFICIENT {float(input_params[12])}\n'
        s[13] = f'FALL_TIME_THRESHOLD {float(input_params[13])}\n'
        s[14] = f'LITHIC_DENSITY {float(input_params[14])}\n'
        s[15] = f'PUMICE_DENSITY {float(input_params[15])}\n'
        s[16] = f'COL_STEPS {float(input_params[16])}\n'
        s[17] = f'PART_STEPS {float(input_params[17])}\n'
        s[18] = f'PLUME_MODEL {int(input_params[18])}\n'
        
        with open(config_path, 'w') as out:
            for i in range(len(s)):
                out.write(s[i])
                
        logger.debug(f"Updated config file at {config_path}")
    
    except Exception as e:
        logger.error(f"Error updating config file: {str(e)}")
        raise


def run_tephra2(config_path, sites_path, wind_path, output_path, tephra2_path, silent=True):
    """
    Run Tephra2 with given inputs.
    
    Args:
        config_path (str): Path to tephra2 config file
        sites_path (str): Path to tephra2 sites file
        wind_path (str): Path to tephra2 wind file
        output_path (str): Path for tephra2 output
        tephra2_path (str): Path to tephra2 executable
        silent (bool): Whether to run tephra2 silently
    
    Returns:
        np.ndarray: Model predictions (tephra thickness)
    """
    # Ensure paths are strings
    config_path = str(config_path)
    sites_path = str(sites_path)
    wind_path = str(wind_path)
    output_path = str(output_path)
    tephra2_path = str(tephra2_path)
    
    # Add debug output
    logger.info(f"Running tephra2 with:")
    logger.info(f"  tephra2_path: {tephra2_path}")
    logger.info(f"  config_path: {config_path}")
    logger.info(f"  sites_path: {sites_path}")
    logger.info(f"  wind_path: {wind_path}")
    logger.info(f"  output_path: {output_path}")
    
    # Check if files exist
    for path, desc in [
        (tephra2_path, "Tephra2 executable"),
        (config_path, "Config file"),
        (sites_path, "Sites file"),
        (wind_path, "Wind file"),
    ]:
        if not os.path.exists(path):
            logger.error(f"{desc} not found: {path}")
            raise FileNotFoundError(f"{desc} not found: {path}")
    
    if os.path.exists(tephra2_path):
        # Use shell=True approach with command string
        cmd_str = f"{tephra2_path} {config_path} {sites_path} {wind_path}"
        
        with open(output_path, 'w') as f:
            if silent:
                result = subprocess.run(f"{cmd_str} > {output_path}", 
                                        shell=True, 
                                        stderr=subprocess.PIPE,
                                        stdout=subprocess.PIPE,
                                        check=False)
            else:
                result = subprocess.run(cmd_str, 
                                        shell=True,
                                        stdout=f,
                                        stderr=subprocess.PIPE,
                                        check=False)
        
        # Check if the command was successful
        if result.returncode != 0:
            error_msg = result.stderr.decode() if result.stderr else "No error message available"
            logger.error(f"Tephra2 execution failed with error code {result.returncode}: {error_msg}")
            raise subprocess.CalledProcessError(result.returncode, cmd_str, stderr=error_msg)
        
        # Read and return the output
        try:
            prediction = np.genfromtxt(output_path, delimiter=' ')
            prediction = np.nan_to_num(prediction, nan=0.001)
            return prediction[:, 3]  # Extract the relevant column
        except Exception as e:
            logger.error(f"Error reading tephra2 output: {str(e)}")
            raise