import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Union, Optional, Tuple
import yaml

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
    
    def get_user_input(self) -> Dict:
        """Get user input for observation data source."""
        print("\nObservation Data Source Options:")
        print("1. Generate synthetic data")
        print("2. Load from file")
        
        choice = input("Select option (1-2): ")
        
        if choice == "1":
            return self._get_synthetic_input()
        elif choice == "2":
            return self._get_file_input()
        else:
            print("Invalid choice. Please try again.")
            return self.get_user_input()
    
    def _get_synthetic_input(self) -> Dict:
        """Get input for synthetic data generation."""
        print("\nSynthetic Observation Parameters:")
        n_points = int(input(f"Enter number of points (default: {self.default_params['n_points']}): ") 
                      or self.default_params['n_points'])
        noise_level = float(input(f"Enter noise level (default: {self.default_params['noise_level']}): ") 
                          or self.default_params['noise_level'])
        grid_spacing = float(input(f"Enter grid spacing in meters (default: {self.default_params['grid_spacing']}): ") 
                           or self.default_params['grid_spacing'])
        max_distance = float(input(f"Enter maximum distance in meters (default: {self.default_params['max_distance']}): ") 
                           or self.default_params['max_distance'])
        
        return {
            "source": "synthetic",
            "n_points": n_points,
            "noise_level": noise_level,
            "grid_spacing": grid_spacing,
            "max_distance": max_distance
        }
    
    def _get_file_input(self) -> Dict:
        """Get input for file loading."""
        print("\nFile Input Parameters:")
        obs_file = input("Enter path to observations file: ")
        sites_file = input("Enter path to sites file: ")
        
        return {
            "source": "file",
            "obs_file": obs_file,
            "sites_file": sites_file
        }
    
    def generate_observations(self, params: Dict, 
                            vent_location: Tuple[float, float] = (0, 0)) -> Tuple[np.ndarray, np.ndarray]:
        """Generate observation data based on parameters."""
        if params["source"] == "synthetic":
            return self._generate_synthetic(params, vent_location)
        elif params["source"] == "file":
            return self._load_from_file(params)
        else:
            raise ValueError(f"Unknown source: {params['source']}")
    
    def _generate_synthetic(self, params: Dict, 
                          vent_location: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic observation data."""
        # Generate grid of points
        n_side = int(np.sqrt(params["n_points"]))
        x = np.linspace(-params["max_distance"], params["max_distance"], n_side)
        y = np.linspace(-params["max_distance"], params["max_distance"], n_side)
        X, Y = np.meshgrid(x, y)
        
        # Calculate distances from vent
        distances = np.sqrt((X - vent_location[0])**2 + (Y - vent_location[1])**2)
        
        # Generate synthetic observations using inverse square law
        # This is a simplified model - in practice, you'd use Tephra2
        base_loading = 1000  # kg/m² at vent
        observations = base_loading / (1 + (distances/10000)**2)
        
        # Add noise
        noise = np.random.normal(0, params["noise_level"] * observations)
        observations += noise
        
        # Create site coordinates
        sites = np.column_stack((X.flatten(), Y.flatten()))
        
        return observations.flatten(), sites
    
    def _load_from_file(self, params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Load observation data from files."""
        # Load observations
        obs_df = pd.read_csv(params["obs_file"], header=None, names=['mass_loading'])
        observations = obs_df['mass_loading'].values
        
        # Load sites
        sites_df = pd.read_csv(params["sites_file"], sep=r'\s+', header=None, 
                             names=['easting', 'northing'])
        sites = sites_df[['easting', 'northing']].values
        
        return observations, sites
    
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
        
        print(f"\nObservations saved to: {obs_path}")
        print(f"Sites saved to: {sites_path}")
    
    def plot_observations(self, observations: np.ndarray, 
                         sites: np.ndarray,
                         output_path: Union[str, Path] = "output/plots/observations.png") -> None:
        """Plot observation data."""
        import matplotlib.pyplot as plt
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(sites[:, 0], sites[:, 1], 
                            c=observations, 
                            cmap='viridis',
                            s=50)
        plt.colorbar(scatter, label='Mass Loading (kg/m²)')
        plt.xlabel('Easting (m)')
        plt.ylabel('Northing (m)')
        plt.title('Tephra Deposit Observations')
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        
        print(f"\nPlot saved to: {output_path}")

def main():
    """Main function to handle observation data."""
    handler = ObservationHandler()
    
    # Get user input
    params = handler.get_user_input()
    
    # Generate/load data
    observations, sites = handler.generate_observations(params)
    
    # Save data
    handler.save_observations(observations, sites)
    
    # Plot data
    handler.plot_observations(observations, sites)
    
    return observations, sites

if __name__ == "__main__":
    main()
