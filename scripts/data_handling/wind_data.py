# scripts/data_handling/wind_data.py

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Union, Optional

class WindDataHandler:
    def __init__(self, output_dir: Union[str, Path] = "data/input"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_wind_data(self, params: Dict) -> pd.DataFrame:
        """
        Generate synthetic wind data using Gaussian model.
        
        Parameters
        ----------
        params : Dict
            Wind parameters with keys:
            - wind_direction: predominant wind direction in degrees
            - max_wind_speed: maximum wind speed in m/s
            - elevation_max_speed: elevation of maximum speed in meters
            - zero_elevation: elevation where wind speed is zero in meters
            
        Returns
        -------
        pd.DataFrame
            Wind data with HEIGHT, SPEED, DIRECTION columns
        """
        heights = np.linspace(0, 30000, 20)  # 20 points from 0 to 30000m
        
        # Gaussian wind speed profile
        if "elevation_max_speed" in params:
            max_elev = params["elevation_max_speed"]
            sigma = max_elev / 2  # standard deviation based on max elevation
        else:
            max_elev = 5000
            sigma = 2500
            
        speeds = params.get("max_wind_speed", 30) * np.exp(
            -((heights - max_elev)**2) / (2 * sigma**2)
        )
        
        # Optional: vary wind direction with height
        if "direction_variation" in params and params["direction_variation"]:
            # Wind direction that varies with height
            base_dir = params.get("wind_direction", 180)
            directions = base_dir + 20 * np.sin(heights / 5000)
        else:
            # Constant wind direction
            directions = np.full_like(heights, params.get("wind_direction", 180))
        
        # Create DataFrame
        wind_df = pd.DataFrame({
            'HEIGHT': heights,
            'SPEED': speeds,
            'DIRECTION': directions
        })
        
        return wind_df
    
    def save_wind_data(self, data: Union[np.ndarray, pd.DataFrame], 
                     filename: str = "wind.txt") -> Path:
        """Save wind data to file and return the file path."""
        # If filename is a full path, use it directly
        if os.path.isabs(filename):
            output_path = Path(filename)
        else:
            output_path = self.output_dir / filename
            
        # Ensure the directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Format data for saving
        if isinstance(data, pd.DataFrame):
            header = "#HEIGHT SPEED DIRECTION"
            np.savetxt(
                output_path,
                data.values,
                fmt="%.1f",
                header=header,
                comments=""
            )
        else:
            header = "#HEIGHT SPEED DIRECTION"
            np.savetxt(
                output_path,
                data,
                fmt="%.1f",
                header=header,
                comments=""
            )
        
        print(f"Wind data saved to: {output_path}")
        return output_path

    def load_wind_data(self, filename: Union[str, Path]) -> pd.DataFrame:
        """Load wind data from a file."""
        filepath = Path(filename)
        if not filepath.exists():
            raise FileNotFoundError(f"Wind data file not found: {filepath}")
            
        try:
            # Read the wind data file
            df = pd.read_csv(
                filepath,
                delim_whitespace=True,
                comment='#',
                names=['HEIGHT', 'SPEED', 'DIRECTION']
            )
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error loading wind data: {e}")