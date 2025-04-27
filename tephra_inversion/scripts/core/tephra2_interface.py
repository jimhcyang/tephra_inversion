import os
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, Union, Optional, List, Tuple
import yaml

class Tephra2Interface:
    def __init__(self, 
                 tephra2_path: Union[str, Path] = "tephra2/tephra2_2020",
                 config_dir: Union[str, Path] = "config",
                 output_dir: Union[str, Path] = "output"):
        self.tephra2_path = Path(tephra2_path)
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_config(self, 
                     vent_location: Tuple[float, float, float],
                     wind_data: np.ndarray,
                     output_path: Union[str, Path]) -> None:
        """Create Tephra2 configuration file."""
        output_path = Path(output_path)
        
        config = {
            "VENT_LOCATION": {
                "EASTING": vent_location[0],
                "NORTHING": vent_location[1],
                "ELEVATION": vent_location[2]
            },
            "WIND_DATA": {
                "HEIGHTS": wind_data[:, 0].tolist(),
                "SPEEDS": wind_data[:, 1].tolist(),
                "DIRECTIONS": wind_data[:, 2].tolist()
            },
            "OUTPUT": {
                "PATH": str(output_path.parent),
                "FILENAME": output_path.name
            }
        }
        
        config_path = self.config_dir / "tephra2_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path
    
    def run_tephra2(self, 
                   config_path: Union[str, Path],
                   params: Dict[str, float]) -> np.ndarray:
        """Run Tephra2 with given parameters."""
        # Create parameter file
        param_path = self.config_dir / "tephra2_params.txt"
        with open(param_path, 'w') as f:
            for name, value in params.items():
                f.write(f"{name} {value}\n")
        
        # Run Tephra2
        cmd = [
            str(self.tephra2_path),
            str(config_path),
            str(param_path)
        ]
        
        try:
            result = subprocess.run(cmd, 
                                  capture_output=True, 
                                  text=True, 
                                  check=True)
            print("Tephra2 run successful")
        except subprocess.CalledProcessError as e:
            print(f"Tephra2 run failed: {e.stderr}")
            raise
        
        # Parse output
        output_path = Path(config_path).parent / "output.txt"
        if not output_path.exists():
            raise FileNotFoundError(f"Tephra2 output file not found: {output_path}")
        
        return self._parse_output(output_path)
    
    def _parse_output(self, output_path: Union[str, Path]) -> np.ndarray:
        """Parse Tephra2 output file."""
        data = []
        with open(output_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                try:
                    easting, northing, loading = map(float, line.split())
                    data.append([easting, northing, loading])
                except ValueError:
                    continue
        
        return np.array(data)
    
    def calculate_likelihood(self, 
                           model_output: np.ndarray,
                           observations: np.ndarray,
                           sites: np.ndarray,
                           sigma: float = 0.1) -> float:
        """Calculate log-likelihood of model output given observations."""
        # Interpolate model output to observation sites
        from scipy.interpolate import griddata
        
        # Create grid points from model output
        grid_x = model_output[:, 0]
        grid_y = model_output[:, 1]
        grid_z = model_output[:, 2]
        
        # Interpolate to observation sites
        interpolated = griddata(
            (grid_x, grid_y),
            grid_z,
            (sites[:, 0], sites[:, 1]),
            method='linear',
            fill_value=0.0
        )
        
        # Calculate log-likelihood
        residuals = np.log(observations) - np.log(interpolated)
        log_likelihood = -0.5 * np.sum(residuals**2) / (sigma**2)
        
        return log_likelihood
    
    def run_forward_model(self,
                         vent_location: Tuple[float, float, float],
                         wind_data: np.ndarray,
                         params: Dict[str, float],
                         output_path: Union[str, Path] = "output/forward_model.txt") -> Tuple[np.ndarray, float]:
        """Run complete forward model with likelihood calculation."""
        # Create config
        config_path = self.create_config(vent_location, wind_data, output_path)
        
        # Run Tephra2
        model_output = self.run_tephra2(config_path, params)
        
        return model_output

def main():
    """Main function to test Tephra2 interface."""
    # Example usage
    interface = Tephra2Interface()
    
    # Example parameters
    vent_location = (0.0, 0.0, 0.0)  # easting, northing, elevation
    wind_data = np.array([
        [1000, 10, 150],
        [2000, 20, 140],
        [3000, 25, 130],
        [4000, 30, 110]
    ])
    params = {
        "column_height": 7500,
        "log_m": 27.5,
        "alpha": 4.0,
        "beta": 2.0
    }
    
    # Run forward model
    model_output = interface.run_forward_model(
        vent_location,
        wind_data,
        params
    )
    
    print(f"Model output shape: {model_output.shape}")
    print(f"First few points:\n{model_output[:5]}")

if __name__ == "__main__":
    main()
