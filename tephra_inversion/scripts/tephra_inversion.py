# scripts/tephra_inversion.py

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List

# Import utility modules
from scripts.data_handling.coordinate_utils import latlon_to_utm
from scripts.data_handling.wind_data import WindDataHandler
from scripts.data_handling.observation_data import ObservationHandler
from scripts.data_handling.esp_config import ESPConfig
from scripts.core.tephra2_interface import Tephra2Interface
from scripts.core.mcmc import metropolis_hastings
from scripts.visualization.wind_plots import WindPlotter
from scripts.visualization.observation_plots import ObservationPlotter
from scripts.visualization.diagnostic_plots import DiagnosticPlotter

# Import default configuration
from config.default_config import DEFAULT_CONFIG

class TephraInversion:
    def __init__(self):
        """Initialize TephraInversion with default configuration."""
        self.config = DEFAULT_CONFIG
        
        # Create output directories if they don't exist
        for dir_path in [
            self.config["paths"]["input_dir"],
            self.config["paths"]["output_dir"],
            self.config["paths"]["plots_dir"],
            self.config["paths"]["mcmc_dir"]
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize handlers
        self.wind_handler = WindDataHandler(self.config["paths"]["input_dir"])
        self.obs_handler = ObservationHandler(self.config["paths"]["input_dir"])
        self.esp_config = ESPConfig(self.config["paths"]["input_dir"])
        self.tephra2 = Tephra2Interface(
            tephra2_path=self.config["tephra2"]["executable"],
            config_dir=self.config["paths"]["input_dir"],
            output_dir=self.config["paths"]["output_dir"]
        )
        
        # Initialize visualization tools
        self.wind_plotter = WindPlotter(self.config["paths"]["plots_dir"])
        self.obs_plotter = ObservationPlotter(self.config["paths"]["plots_dir"])
        self.diag_plotter = DiagnosticPlotter(self.config["paths"]["plots_dir"])
        
        # Initialize state
        self.vent_lat = None
        self.vent_lon = None
        self.vent_elev = 0.0  # Default elevation
        self.vent_easting = None
        self.vent_northing = None
        self.eruption_time = None
        self.wind_data = None
        self.observations = None
        self.sites = None
        
    def setup_vent_location(self, lat: float, lon: float, elevation: float = 0.0) -> None:
        """
        Set vent location from latitude and longitude.
        
        Parameters
        ----------
        lat : float
            Vent latitude in decimal degrees
        lon : float
            Vent longitude in decimal degrees
        elevation : float, optional
            Vent elevation in meters, defaults to 0.0
        """
        self.vent_lat = lat
        self.vent_lon = lon
        self.vent_elev = elevation
        
        # Convert to UTM
        self.vent_easting, self.vent_northing, _ = latlon_to_utm(lat, lon, elevation)
        
        print(f"Vent location set: {lat}°N, {lon}°E, {elevation}m elevation")
        print(f"UTM coordinates: Easting {self.vent_easting:.1f}m, Northing {self.vent_northing:.1f}m")
    
    def setup_eruption_time(self, eruption_time: List[str]) -> None:
        """
        Set eruption time.
        
        Parameters
        ----------
        eruption_time : List[str]
            Eruption time as ["YYYY", "MM", "DD", "HH:MM"]
        """
        self.eruption_time = eruption_time
        
        time_str = f"{eruption_time[0]}-{eruption_time[1]}-{eruption_time[2]} {eruption_time[3]}"
        print(f"Eruption time set: {time_str}")
    
    def setup_wind_data(self, ask_user: bool = True) -> None:
        """
        Set up wind data by checking for existing data first, then API, then synthetic.
        
        Parameters
        ----------
        ask_user : bool, optional
            Whether to ask the user for confirmation, defaults to True
        """
        wind_file = Path(self.config["paths"]["input_dir"]) / "wind.txt"
        
        # Check if wind file exists
        if wind_file.exists():
            print(f"Found existing wind data at: {wind_file}")
            try:
                self.wind_data = self.wind_handler.load_wind_data(wind_file)
                print("Successfully loaded wind data.")
            except Exception as e:
                print(f"Error loading wind data: {e}")
                ask_user = True
        else:
            ask_user = True
        
        # If needed, ask user about wind data
        if ask_user:
            has_wind = input("Do you have wind data? (y/n): ").lower().startswith('y')
            
            if has_wind:
                # User has wind data
                wind_file_input = input("Enter path to wind data file: ")
                try:
                    self.wind_data = self.wind_handler.load_wind_data(wind_file_input)
                    print(f"Wind data loaded from: {wind_file_input}")
                    
                    # Save to standard location if needed
                    if wind_file_input != str(wind_file):
                        self.wind_handler.save_wind_data(self.wind_data, str(wind_file))
                except Exception as e:
                    print(f"Error loading wind data: {e}")
                    self._get_wind_data_api_or_synthetic()
            else:
                # Try API first, then synthetic
                self._get_wind_data_api_or_synthetic()
        
        # Plot wind data
        if self.wind_data is not None:
            heights = self.wind_data["HEIGHT"].values
            speeds = self.wind_data["SPEED"].values
            directions = self.wind_data["DIRECTION"].values
            
            # Create profile plot
            profile_path = self.wind_plotter.plot_wind_profile(
                heights, 
                speeds, 
                directions,
                save_path=Path(self.config["paths"]["plots_dir"]) / "wind_profile.png"
            )
            
            # Create wind rose plot
            rose_path = self.wind_plotter.plot_wind_rose(
                directions,
                speeds,
                heights,
                save_path=Path(self.config["paths"]["plots_dir"]) / "wind_rose.png"
            )
            
            print(f"Wind profile plot saved to: {profile_path}")
            print(f"Wind rose plot saved to: {rose_path}")
        else:
            raise ValueError("Failed to set up wind data.")
    
    def _get_wind_data_api_or_synthetic(self) -> None:
        """Try to get wind data from API first, then fallback to synthetic."""
        if self.vent_lat is None or self.vent_lon is None or self.eruption_time is None:
            print("Missing vent location or eruption time for API wind data.")
            self._generate_synthetic_wind()
            return
            
        try:
            print("Attempting to fetch wind data from reanalysis API...")
            # This would be the API call - for now we'll just use synthetic data
            # TODO: Implement actual API call
            raise NotImplementedError("API wind data retrieval not yet implemented")
            
        except Exception as e:
            print(f"Error fetching wind data from API: {e}")
            print("Falling back to synthetic wind data.")
            self._generate_synthetic_wind()
    
    def _generate_synthetic_wind(self) -> None:
        """Generate synthetic wind data."""
        print("Generating synthetic wind data...")
        params = {
            "wind_direction": 150,  # Default wind direction
            "max_wind_speed": 30,   # Default max wind speed
            "elevation_max_speed": 5000,  # Default elevation of max speed
        }
        self.wind_data = self.wind_handler.generate_wind_data(params)
        wind_path = self.wind_handler.save_wind_data(
            self.wind_data, 
            str(Path(self.config["paths"]["input_dir"]) / "wind.txt")
        )
        print(f"Synthetic wind data saved to: {wind_path}")
        
    def setup_observation_data(self, ask_user: bool = True) -> None:
        """
        Set up observation data by checking for existing data first, then synthetic.
        
        Parameters
        ----------
        ask_user : bool, optional
            Whether to ask the user for confirmation, defaults to True
        """
        obs_file = Path(self.config["paths"]["input_dir"]) / "observations.csv"
        sites_file = Path(self.config["paths"]["input_dir"]) / "sites.csv"
        
        # Check if observation files exist
        if obs_file.exists() and sites_file.exists():
            print(f"Found existing observation data at: {obs_file} and {sites_file}")
            try:
                # Load observations and sites
                self.observations = np.loadtxt(obs_file)
                self.sites = np.loadtxt(sites_file)
                print("Successfully loaded observation data.")
                
                # If we don't have vent location, try to infer from sites
                if (self.vent_easting is None or self.vent_northing is None) and ask_user:
                    use_inferred = input("Vent location not set. Infer from observation data? (y/n): ").lower().startswith('y')
                    if use_inferred:
                        # Simple inference - use maximum deposit location as vent
                        max_idx = np.argmax(self.observations)
                        self.vent_easting = self.sites[max_idx, 0]
                        self.vent_northing = self.sites[max_idx, 1]
                        print(f"Inferred vent location: Easting {self.vent_easting:.1f}m, Northing {self.vent_northing:.1f}m")
            except Exception as e:
                print(f"Error loading observation data: {e}")
                ask_user = True
        else:
            ask_user = True
        
        # If needed, ask user about observation data
        if ask_user and (self.observations is None or self.sites is None):
            has_obs = input("Do you have observation data? (y/n): ").lower().startswith('y')
            
            if has_obs:
                # User has observation data
                obs_file_input = input("Enter path to observations file: ")
                sites_file_input = input("Enter path to sites file: ")
                
                try:
                    # Load observations
                    self.observations = np.loadtxt(obs_file_input)
                    
                    # Load sites
                    self.sites = np.loadtxt(sites_file_input)
                    
                    print(f"Observation data loaded from: {obs_file_input}")
                    print(f"Sites data loaded from: {sites_file_input}")
                    
                    # Save to standard location if needed
                    if obs_file_input != str(obs_file):
                        np.savetxt(obs_file, self.observations, fmt='%.6f')
                    if sites_file_input != str(sites_file):
                        np.savetxt(sites_file, self.sites, fmt='%.1f')
                except Exception as e:
                    print(f"Error loading observation data: {e}")
                    self._generate_synthetic_observations()
            else:
                # Generate synthetic observations
                self._generate_synthetic_observations()
        
        # Plot observation data
        if self.observations is not None and self.sites is not None:
            # Make sure vent location is set for plotting
            if self.vent_easting is None or self.vent_northing is None:
                # Infer vent location from sites if not set
                max_idx = np.argmax(self.observations)
                vent_x = self.sites[max_idx, 0]
                vent_y = self.sites[max_idx, 1]
                print(f"Using inferred vent location for plotting: ({vent_x:.1f}, {vent_y:.1f})")
            else:
                vent_x = self.vent_easting
                vent_y = self.vent_northing
                
            plot_path = self.obs_plotter.plot_tephra_distribution(
                eastings=self.sites[:, 0],
                northings=self.sites[:, 1],
                thicknesses=self.observations,
                vent_location=(vent_x, vent_y),
                save_path=Path(self.config["paths"]["plots_dir"]) / "tephra_distribution.png"
            )
            print(f"Tephra distribution plot saved to: {plot_path}")
        else:
            raise ValueError("Failed to set up observation data.")
    
    def _generate_synthetic_observations(self) -> None:
        """Generate synthetic observation data."""
        print("Generating synthetic observation data...")
        if self.vent_easting is None or self.vent_northing is None:
            raise ValueError("Vent location not set. Use setup_vent_location() first.")
                
        params = {
            "n_points": 100,
            "noise_level": 0.1,
            "grid_spacing": 1000,
            "max_distance": 50000
        }
        vent_location = (self.vent_easting, self.vent_northing)
        self.observations, self.sites = self.obs_handler.generate_observations(params, vent_location)
        
        # Save observations
        self.obs_handler.save_observations(
            self.observations, 
            self.sites,
            obs_filename="observations.csv",
            sites_filename="sites.csv"
        )
    
    def setup_parameters(self) -> Dict:
        """
        Set up parameters for inversion.
        
        Returns
        -------
        Dict
            Dictionary with parameter configuration
        """
        # Create default configuration file
        config_path = Path(self.config["paths"]["input_dir"]) / "tephra2.conf"
        self.esp_config.create_default_config(config_path)
        
        # Get MCMC parameters
        mcmc_params = self.esp_config.get_mcmc_parameters()
        
        # Update with vent location
        with open(config_path, 'r') as f:
            lines = f.readlines()
            
        with open(config_path, 'w') as f:
            for line in lines:
                if line.startswith("VENT_EASTING"):
                    f.write(f"VENT_EASTING {self.vent_easting:.6f}\n")
                elif line.startswith("VENT_NORTHING"):
                    f.write(f"VENT_NORTHING {self.vent_northing:.6f}\n")
                elif line.startswith("VENT_ELEVATION"):
                    f.write(f"VENT_ELEVATION {self.vent_elev:.6f}\n")
                else:
                    f.write(line)
                    
        print(f"Parameter configuration file created at: {config_path}")
        return mcmc_params
    
    def run_inversion(self) -> Dict:
        """
        Run the tephra inversion to estimate plume height and eruption mass.
        
        Returns
        -------
        Dict
            Results of the inversion, including best parameters and diagnostics
        """
        # Check that we have all required data
        if self.vent_easting is None or self.vent_northing is None:
            raise ValueError("Vent location not set. Use setup_vent_location() first.")
        if self.wind_data is None:
            raise ValueError("Wind data not set. Use setup_wind_data() first.")
        if self.observations is None or self.sites is None:
            raise ValueError("Observation data not set. Use setup_observation_data() first.")
        
        # Setup parameters
        mcmc_params = self.setup_parameters()
        
        # Get paths to files
        config_path = str(Path(self.config["paths"]["input_dir"]) / "tephra2.conf")
        sites_path = str(Path(self.config["paths"]["input_dir"]) / "sites.csv")
        wind_path = str(Path(self.config["paths"]["input_dir"]) / "wind.txt")
        output_path = str(Path(self.config["paths"]["output_dir"]) / "tephra2_output.txt")
        
        # Save wind and sites data in proper format if needed
        if not Path(wind_path).exists():
            self.wind_handler.save_wind_data(self.wind_data, wind_path)
        if not Path(sites_path).exists():
            np.savetxt(sites_path, self.sites, fmt='%.1f')
        
        # Run MCMC
        print("\nRunning MCMC parameter estimation...")
        chain, post_chain, acceptance_count, prior_array, likeli_array = metropolis_hastings(
            mcmc_params["initial_values"], 
            mcmc_params["prior_type"], 
            mcmc_params["draw_scale"], 
            mcmc_params["prior_parameters"],
            config_path, 
            sites_path, 
            wind_path, 
            output_path,
            self.config["tephra2"]["executable"],
            self.config["mcmc"]["n_iterations"],
            0.1,  # Likelihood scale
            self.observations,
            check_snapshot=100,
            silent=True
        )
        
        # Process results
        best_idx = np.argmax(post_chain)
        best_params = chain[best_idx]
        
        # Convert to dictionary of named parameters
        param_names = ["plume_height", "log_m"]
        best_params_dict = {name: value for name, value in zip(param_names, best_params)}
        
        # Create chain dictionary for plotting
        chain_dict = {name: chain[:, i] for i, name in enumerate(param_names)}
        
        # Plot diagnostic plots
        burnin = self.config["mcmc"]["n_burnin"]
        
        # Trace plots
        trace_path = self.diag_plotter.plot_trace(
            chain_dict, 
            burnin=burnin,
            save_path=Path(self.config["paths"]["plots_dir"]) / "trace_plots.png"
        )
        
        # Distribution plots
        dist_path = self.diag_plotter.plot_parameter_distributions(
            chain_dict, 
            burnin=burnin,
            save_path=Path(self.config["paths"]["plots_dir"]) / "parameter_distributions.png"
        )
        
        # Correlation plots
        corr_path = self.diag_plotter.plot_parameter_correlations(
            chain_dict, 
            burnin=burnin,
            save_path=Path(self.config["paths"]["plots_dir"]) / "parameter_correlations.png"
        )
        
        print(f"Diagnostic plots saved to:\n{trace_path}\n{dist_path}\n{corr_path}")
        
        # Run forward model with best parameters to get predictions
        with open(config_path, 'r') as f:
            lines = f.readlines()
            
        with open(config_path, 'w') as f:
            for line in lines:
                if line.startswith("PLUME_HEIGHT"):
                    f.write(f"PLUME_HEIGHT {best_params_dict['plume_height']:.6f}\n")
                elif line.startswith("ERUPTION_MASS"):
                    f.write(f"ERUPTION_MASS {10 ** best_params_dict['log_m']:.6e}\n")
                else:
                    f.write(line)
        
        # Run forward model with best parameters
        print("\nRunning forward model with best parameters...")
        predictions = self.tephra2.run_tephra2(
            config_path,
            sites_path,
            wind_path,
            output_path,
            silent=True
        )
        
        # Plot tephra distribution comparison
        comp_path = self.diag_plotter.plot_tephra_distribution_comparison(
            self.observations,
            predictions,
            save_path=Path(self.config["paths"]["plots_dir"]) / "tephra_comparison.png"
        )
        print(f"Tephra distribution comparison saved to: {comp_path}")
        
        # Save results to file
        results_file = Path(self.config["paths"]["mcmc_dir"]) / "inversion_results.txt"
        with open(results_file, 'w') as f:
            f.write("Tephra Inversion Results\n")
            f.write("=======================\n\n")
            f.write(f"Plume Height: {best_params_dict['plume_height']:.1f} m\n")
            f.write(f"Log10 Eruption Mass: {best_params_dict['log_m']:.2f}\n")
            f.write(f"Eruption Mass: {10 ** best_params_dict['log_m']:.2e} kg\n\n")
            f.write(f"MCMC Acceptance Rate: {acceptance_count / self.config['mcmc']['n_iterations']:.4f}\n")
            f.write(f"Posterior Value: {post_chain[best_idx]:.4f}\n")
        
        # Extract relevant results
        results = {
            "best_params": best_params_dict,
            "chain": chain_dict,
            "posterior": post_chain,
            "acceptance_rate": acceptance_count / self.config["mcmc"]["n_iterations"],
            "predictions": predictions,
            "results_file": str(results_file)
        }
        
        # Print results
        print("\nInversion Results:")
        print(f"Plume Height: {best_params_dict['plume_height']:.1f} m")
        print(f"Eruption Mass: {10 ** best_params_dict['log_m']:.2e} kg")
        print(f"Acceptance Rate: {results['acceptance_rate']:.2f}")
        print(f"Results saved to: {results_file}")
        
        return results
    
    def run_workflow(self) -> Dict:
        """
        Run the complete workflow with user inputs.
        
        Returns
        -------
        Dict
            Results of the inversion
        """
        # Get vent location
        lat = float(input("Enter vent latitude (degrees): "))
        lon = float(input("Enter vent longitude (degrees): "))
        elev_input = input("Enter vent elevation in meters (default: 0.0): ")
        elevation = float(elev_input) if elev_input.strip() else 0.0
        
        self.setup_vent_location(lat, lon, elevation)
        
        # Get eruption time
        year = input("Enter eruption year (YYYY): ")
        month = input("Enter eruption month (MM): ")
        day = input("Enter eruption day (DD): ")
        time = input("Enter eruption time (HH:MM): ")
        self.setup_eruption_time([year, month, day, time])
        
        # Setup wind data and observation data with automatic checks for existing files
        self.setup_wind_data()
        self.setup_observation_data()
        
        # Run inversion
        results = self.run_inversion()
        
        return results