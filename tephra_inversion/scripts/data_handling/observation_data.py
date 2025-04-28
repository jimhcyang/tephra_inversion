# scripts/data_handling/observation_data.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Union, Optional, Tuple

class ObservationHandler:
    def __init__(self, output_dir: Union[str, Path] = "data/input"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default observation parameters
        self.default_params = {
            "n_points": 100,
            "noise_level": 0.1,
            "grid_spacing": 1000,  # meters
            "max_distance": 50000  # meters
        }
    
    def generate_observations(self, params: Dict, 
                            vent_location: Tuple[float, float] = (0, 0)) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic observation data."""
        # Generate grid of points
        n_side = int(np.sqrt(params["n_points"]))
        x = np.linspace(-params["max_distance"], params["max_distance"], n_side)
        y = np.linspace(-params["max_distance"], params["max_distance"], n_side)
        X, Y = np.meshgrid(x, y)
        
        # Adjust grid to center on vent location
        X = X + vent_location[0]
        Y = Y + vent_location[1]
        
        # Calculate distances from vent
        distances = np.sqrt((X - vent_location[0])**2 + (Y - vent_location[1])**2)
        
        # Generate synthetic observations using inverse square law
        base_loading = 1000  # kg/m² at vent
        observations = base_loading / (1 + (distances/10000)**2)
        
        # Add noise
        noise = np.random.normal(0, params["noise_level"] * observations)
        observations += noise
        
        # Create site coordinates
        sites = np.column_stack((X.flatten(), Y.flatten(), np.ones_like(X.flatten()) * 1000))
        
        return observations.flatten(), sites
    
    def save_observations(self, observations: np.ndarray, 
                         sites: np.ndarray, 
                         obs_filename: str = "observations.csv",
                         sites_filename: str = "sites.csv") -> None:
        """Save observation data to files."""
        # Save observations
        obs_path = self.output_dir / obs_filename
        np.savetxt(obs_path, observations, fmt='%.6f')
        
        # Save sites
        sites_path = self.output_dir / sites_filename
        np.savetxt(sites_path, sites, fmt='%.1f')
        
        print(f"Observations saved to: {obs_path}")
        print(f"Sites saved to: {sites_path}")
    
    def load_observations(self, 
                        obs_filename: str = "observations.csv",
                        sites_filename: str = "sites.csv") -> Tuple[np.ndarray, np.ndarray]:
        """Load observation data from files."""
        # Load observations
        obs_path = self.output_dir / obs_filename
        sites_path = self.output_dir / sites_filename
        
        if not obs_path.exists():
            raise FileNotFoundError(f"Observation file not found: {obs_path}")
        if not sites_path.exists():
            raise FileNotFoundError(f"Sites file not found: {sites_path}")
            
        observations = np.loadtxt(obs_path)
        sites = np.loadtxt(sites_path)
        
        return observations, sites
    
    def plot_observations(self, observations: np.ndarray, 
                         sites: np.ndarray,
                         output_path: Union[str, Path] = "data/output/plots/observations.png") -> None:
        """Plot observation data."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(sites[:, 0], sites[:, 1], 
                            c=observations, 
                            cmap='viridis',
                            s=50,
                            norm=plt.Normalize(vmin=np.min(observations), vmax=np.max(observations)))
        plt.colorbar(scatter, label='Mass Loading (kg/m²)')
        plt.xlabel('Easting (m)')
        plt.ylabel('Northing (m)')
        plt.title('Tephra Deposit Observations')
        plt.grid(True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to: {output_path}")