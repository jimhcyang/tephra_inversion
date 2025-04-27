# scripts/data_handling/wind_data.py

import os
import numpy as np
from pathlib import Path
from typing import Dict, Union, Optional
import requests
import pandas as pd
import yaml

class WindDataHandler:
    def __init__(self, output_dir: Union[str, Path] = "data/input"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def get_user_input(self) -> Dict:
        """Get user input for wind data source."""
        print("\nWind Data Source Options:")
        print("1. Download from ERA5")
        print("2. Generate synthetic data")
        print("3. Input custom data")
        
        choice = input("Select option (1-3): ")
        
        if choice == "1":
            return self._get_era5_input()
        elif choice == "2":
            return self._get_synthetic_input()
        elif choice == "3":
            return self._get_custom_input()
        else:
            print("Invalid choice. Please try again.")
            return self.get_user_input()
    
    def _get_era5_input(self) -> Dict:
        """Get input for ERA5 download."""
        print("\nERA5 Download Parameters:")
        lat = float(input("Enter vent latitude: "))
        lon = float(input("Enter vent longitude: "))
        time = input("Enter eruption time (YYYY-MM-DD HH:MM): ").split()
        
        return {
            "source": "era5",
            "lat": lat,
            "lon": lon,
            "time": time
        }
    
    def _get_synthetic_input(self) -> Dict:
        """Get input for synthetic data generation."""
        print("\nSynthetic Wind Parameters:")
        wind_dir = float(input("Enter wind direction (degrees): "))
        max_speed = float(input("Enter maximum wind speed (m/s): "))
        max_elev = float(input("Enter elevation of maximum speed (m): "))
        zero_elev = float(input("Enter zero elevation (m): "))
        
        return {
            "source": "synthetic",
            "wind_direction": wind_dir,
            "max_wind_speed": max_speed,
            "elevation_max_speed": max_elev,
            "zero_elevation": zero_elev
        }
    
    def _get_custom_input(self) -> Dict:
        """Get custom wind data input."""
        print("\nEnter wind data (height speed direction):")
        print("Enter 'done' when finished")
        
        data = []
        while True:
            line = input("> ")
            if line.lower() == 'done':
                break
            try:
                height, speed, direction = map(float, line.split())
                data.append([height, speed, direction])
            except ValueError:
                print("Invalid input. Please enter three numbers separated by spaces.")
        
        return {
            "source": "custom",
            "data": np.array(data)
        }
    
    def generate_wind_data(self, params: Dict) -> np.ndarray:
        """Generate wind data based on parameters."""
        if params["source"] == "synthetic":
            return self._generate_synthetic(params)
        elif params["source"] == "custom":
            return params["data"]
        elif params["source"] == "era5":
            return self._download_era5(params)
        else:
            raise ValueError(f"Unknown source: {params['source']}")
    
    def _generate_synthetic(self, params: Dict) -> np.ndarray:
        """Generate synthetic wind data using Gaussian model."""
        heights = np.linspace(0, 10000, 11)  # 11 points from 0 to 10000m
        
        # Gaussian wind speed profile
        speeds = params["max_wind_speed"] * np.exp(
            -((heights - params["elevation_max_speed"])**2) / 
            (2 * (params["elevation_max_speed"]/2)**2)
        )
        
        # Constant wind direction
        directions = np.full_like(heights, params["wind_direction"])
        
        return np.column_stack((heights, speeds, directions))
    
    def _download_era5(self, params: Dict) -> np.ndarray:
        """Download wind data from ERA5."""
        # TODO: Implement ERA5 API call
        # For now, return synthetic data as placeholder
        return self._generate_synthetic({
            "source": "synthetic",
            "wind_direction": 150,
            "max_wind_speed": 30,
            "elevation_max_speed": 5000,
            "zero_elevation": 0
        })
    
    def save_wind_data(self, data: np.ndarray, filename: str = "wind.txt") -> None:
        """Save wind data to file."""
        # If filename is a full path, use it directly
        if os.path.isabs(filename):
            output_path = Path(filename)
        else:
            output_path = self.output_dir / filename
            
        # Ensure the directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Format data for saving
        header = "#HEIGHT SPEED DIRECTION"
        np.savetxt(
            output_path,
            data,
            fmt="%.1f",
            header=header,
            comments=""
        )
        
        print(f"\nWind data saved to: {output_path}")

    def load_wind_data(self, filename: Union[str, Path]) -> pd.DataFrame:
        """
        Load wind data from a file.
        
        Parameters
        ----------
        filename : str or Path
            Path to the wind data file
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing the wind data with columns:
            - HEIGHT: Elevation in meters
            - SPEED: Wind speed in m/s
            - DIRECTION: Wind direction in degrees
        """
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
            
            # Validate the data
            if df.shape[1] != 3:
                raise ValueError(f"Expected 3 columns in wind data, got {df.shape[1]}")
                
            if not all(col in df.columns for col in ['HEIGHT', 'SPEED', 'DIRECTION']):
                raise ValueError("Wind data must contain HEIGHT, SPEED, and DIRECTION columns")
                
            return df
            
        except Exception as e:
            raise ValueError(f"Error loading wind data: {e}")

    def load_params_from_yaml(self, yaml_path: Union[str, Path]) -> Dict:
        """
        Load wind parameters from a YAML file.
        
        Parameters
        ----------
        yaml_path : str or Path
            Path to the YAML configuration file
            
        Returns
        -------
        Dict
            Dictionary containing wind parameters
        """
        filepath = Path(yaml_path)
        if not filepath.exists():
            raise FileNotFoundError(f"YAML configuration file not found: {filepath}")
            
        try:
            with open(filepath, 'r') as f:
                params = yaml.safe_load(f)
                
            # Validate required parameters
            required_params = ['source', 'wind_direction', 'max_wind_speed', 
                             'elevation_max_speed', 'zero_elevation']
            if not all(param in params for param in required_params):
                raise ValueError(f"Missing required parameters in YAML file. Required: {required_params}")
                
            return params
            
        except Exception as e:
            raise ValueError(f"Error loading YAML configuration: {e}")

def main():
    """Main function to handle wind data generation/download."""
    handler = WindDataHandler()
    
    # Get user input
    params = handler.get_user_input()
    
    # Generate/download data
    wind_data = handler.generate_wind_data(params)
    
    # Save data
    handler.save_wind_data(wind_data)
    
    return wind_data

if __name__ == "__main__":
    main()