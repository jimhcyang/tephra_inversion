# scripts/core/tephra2_interface.py

import os
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Union, Optional, List, Tuple

class Tephra2Interface:
    def __init__(self, 
                 tephra2_path: Union[str, Path] = "tephra2/tephra2_2020",
                 config_dir: Union[str, Path] = "data/input",
                 output_dir: Union[str, Path] = "data/output"):
        self.tephra2_path = Path(tephra2_path)
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_tephra2(self, 
                   config_path: Union[str, Path],
                   sites_path: Union[str, Path],
                   wind_path: Union[str, Path],
                   output_path: Union[str, Path],
                   silent: bool = True) -> np.ndarray:
        """
        Execute Tephra2 with the given input files.
        
        Parameters
        ----------
        config_path : str or Path
            Path to Tephra2 configuration file
        sites_path : str or Path
            Path to sites file
        wind_path : str or Path
            Path to wind profile file
        output_path : str or Path
            Path where Tephra2 output will be saved
        silent : bool
            If True, suppresses Tephra2 console output
            
        Returns
        -------
        np.ndarray
            Tephra deposit predictions
        """
        # Verify all input files exist
        for path, name in [(config_path, "config"), 
                          (sites_path, "sites"),
                          (wind_path, "wind")]:
            if not Path(path).exists():
                raise FileNotFoundError(f"{name} file not found: {path}")
                
        if not self.tephra2_path.exists():
            raise FileNotFoundError(f"Tephra2 executable not found: {self.tephra2_path}")

        # Construct the command
        cmd = [str(self.tephra2_path), str(config_path), str(sites_path), str(wind_path)]
        
        # Run Tephra2
        try:
            with open(output_path, 'w') as f:
                if silent:
                    subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL, check=True)
                else:
                    subprocess.run(cmd, stdout=f, check=True)
            
            # Read results (assuming columns: EAST, NORTH, ELEVATION, MASS, ...)
            output_data = np.genfromtxt(output_path, delimiter=' ')
            
            # Check if output data has at least 4 columns (EAST, NORTH, ELEV, MASS)
            if output_data.shape[1] >= 4:
                return output_data[:, 3]  # Return the mass column
            else:
                raise ValueError(f"Tephra2 output has fewer than 4 columns: {output_data.shape}")
                
        except subprocess.CalledProcessError as e:
            print(f"Tephra2 execution failed: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error in run_tephra2: {e}")
            raise
    
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
        return self.run_tephra2(config_path, sites_path, wind_path, output_path)
    
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