import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import stat
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('tephra_inversion.log')
    ]
)

# Import core modules
from scripts.core.mcmc import metropolis_hastings, prior_function, likelihood_function
from scripts.core.tephra2_interface import run_tephra2, changing_variable

class TephraInversion:
    """
    Class for performing inversion of tephra2 model using MCMC.
    """
    
    def __init__(self, config, observations=None):
        """
        Initialize the TephraInversion class.
        
        Args:
            config (dict): Configuration dictionary for the inversion
            observations (pd.DataFrame, optional): Observed deposit data
        """
        self.config = config
        self.observations = observations
        
        # Validate configuration
        self._validate_configuration()
        
        # Load observations if not provided
        if observations is None and "observation_file" in self.config:
            self.load_observations(self.config["observation_file"])
            
        logging.info("TephraInversion initialized")
    
    def _validate_configuration(self):
        """
        Validate the configuration settings and correct paths if needed.
        """
        # Check tephra2 executable path
        tephra2_path = self.config["tephra2"]["executable"]
        if not os.path.exists(tephra2_path):
            # Try alternative paths
            alternatives = [
                "./Tephra2/tephra2_2020",
                "../Tephra2/tephra2_2020",
                "~/tephra2/tephra2_2020"
            ]
            
            for alt_path in alternatives:
                expanded_path = os.path.expanduser(alt_path)
                if os.path.exists(expanded_path):
                    self.config["tephra2"]["executable"] = expanded_path
                    logging.info(f"Updated tephra2 executable path to {expanded_path}")
                    break
            else:
                logging.warning(f"Could not find tephra2 executable at {tephra2_path} or alternative locations")
        
        # Ensure input/output directories exist
        os.makedirs("data/input", exist_ok=True)
        os.makedirs("data/output", exist_ok=True)
        
        # Check required parameters
        required_sections = ["tephra2", "mcmc", "parameters"]
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
    
    def _ensure_executable(self, path):
        """
        Ensure the tephra2 executable has proper permissions.
        
        Args:
            path (str): Path to the tephra2 executable
        """
        if os.path.exists(path):
            current_mode = os.stat(path).st_mode
            os.chmod(path, current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            logging.info(f"Set executable permissions for {path}")
        else:
            logging.error(f"Tephra2 executable not found at {path}")
            raise FileNotFoundError(f"Tephra2 executable not found at {path}")
    
    def load_observations(self, file_path):
        """
        Load observations from file.
        
        Args:
            file_path (str): Path to the observation file
        """
        try:
            observations = pd.read_csv(file_path)
            required_columns = ["easting", "northing", "elevation"]
            
            # Check required columns
            for col in required_columns:
                if col not in observations.columns:
                    raise ValueError(f"Observation file missing required column: {col}")
            
            self.observations = observations
            logging.info(f"Loaded {len(observations)} observations from {file_path}")
        except Exception as e:
            logging.error(f"Failed to load observations: {str(e)}")
            raise
    
    def _prepare_input_files(self):
        """
        Prepare the input files for tephra2.
        
        Returns:
            tuple: Paths to the input files (config_path, sites_path, wind_path, output_path)
        """
        # Create output directory if it doesn't exist
        os.makedirs("data/output", exist_ok=True)
        
        # Define paths
        config_path = "data/input/tephra2.conf"
        sites_path = "data/input/sites.csv"
        wind_path = "data/input/wind.txt"
        output_path = "data/output/tephra2_output_mcmc.txt"
        
        # Verify files exist
        for path in [config_path, sites_path, wind_path]:
            if not os.path.exists(path):
                logging.error(f"Input file not found: {path}")
                raise FileNotFoundError(f"Input file not found: {path}")
        
        # Log paths
        logging.info(f"Using config file: {config_path}")
        logging.info(f"Using sites file: {sites_path}")
        logging.info(f"Using wind file: {wind_path}")
        logging.info(f"Output will be written to: {output_path}")
        
        return config_path, sites_path, wind_path, output_path
    
    def _prepare_mcmc_parameters(self):
        """
        Prepare parameters for the MCMC algorithm.
        
        Returns:
            dict: Dictionary of MCMC parameters
        """
        param_config = self.config["parameters"]
        
        # Initialize parameter arrays
        n_params = len(param_config)
        initial_values = np.zeros(n_params)
        prior_type = [""] * n_params
        draw_scale = np.zeros(n_params)
        prior_parameters = [None] * n_params
        
        # Fill parameter arrays from config
        for i, (param_name, param_info) in enumerate(param_config.items()):
            initial_values[i] = param_info["initial_value"]
            prior_type[i] = param_info["prior_type"]
            draw_scale[i] = param_info["draw_scale"]
            
            if prior_type[i] == "Gaussian":
                prior_parameters[i] = [param_info["prior_mean"], param_info["prior_std"]]
            elif prior_type[i] == "Uniform":
                prior_parameters[i] = [param_info["prior_min"], param_info["prior_max"]]
            elif prior_type[i] == "Fixed":
                prior_parameters[i] = [param_info["initial_value"], 0]
            else:
                raise ValueError(f"Unknown prior type: {prior_type[i]}")
        
        # Log parameter info
        logging.info(f"MCMC initialized with {n_params} parameters")
        for i, param_name in enumerate(param_config.keys()):
            logging.info(f"  {param_name}: initial={initial_values[i]}, prior={prior_type[i]}")
        
        return {
            "initial_values": initial_values,
            "prior_type": prior_type,
            "draw_scale": draw_scale,
            "prior_parameters": prior_parameters
        }
    
    def create_tephra2_config(self):
        """
        Create a tephra2 configuration file from template.
        
        Returns:
            str: Path to the created config file
        """
        # Define paths
        config_path = "data/input/tephra2.conf"
        template_path = self.config.get("tephra2", {}).get("template", "templates/tephra2.conf.template")
        
        # Ensure template exists
        if not os.path.exists(template_path):
            logging.error(f"Template file not found: {template_path}")
            raise FileNotFoundError(f"Template file not found: {template_path}")
        
        # Read template
        with open(template_path, 'r') as f:
            template = f.read()
        
        # Replace placeholders with values from config
        placeholders = {
            "PLUME_HEIGHT": self.config["parameters"]["plume_height"]["initial_value"],
            "ERUPTION_MASS": np.exp(self.config["parameters"]["log_mass"]["initial_value"]),
            "VENT_EASTING": self.config["parameters"]["vent_easting"]["initial_value"],
            "VENT_NORTHING": self.config["parameters"]["vent_northing"]["initial_value"],
            "VENT_ELEVATION": self.config["parameters"]["vent_elevation"]["initial_value"],
            "MAX_GRAINSIZE": self.config["parameters"]["max_grainsize"]["initial_value"],
            "MIN_GRAINSIZE": self.config["parameters"]["min_grainsize"]["initial_value"],
            "MEDIAN_GRAINSIZE": self.config["parameters"]["median_grainsize"]["initial_value"],
            "STD_GRAINSIZE": self.config["parameters"]["std_grainsize"]["initial_value"],
            "DIFFUSION_COEFFICIENT": self.config["parameters"]["diffusion_coefficient"]["initial_value"]
        }
        
        for key, value in placeholders.items():
            template = template.replace(f"${key}$", str(value))
        
        # Write config file
        with open(config_path, 'w') as f:
            f.write(template)
        
        logging.info(f"Parameter configuration file created at: {config_path}")
        
        return config_path
    
    def create_sites_file(self):
        """
        Create a sites file for tephra2 from observations.
        
        Returns:
            str: Path to the created sites file
        """
        if self.observations is None:
            logging.error("No observations loaded")
            raise ValueError("No observations loaded")
        
        # Define path
        sites_path = "data/input/sites.csv"
        
        # Extract coordinates from observations
        sites = self.observations[["easting", "northing", "elevation"]]
        
        # Write sites file
        with open(sites_path, 'w') as f:
            for i, row in sites.iterrows():
                f.write(f"{row['easting']} {row['northing']} {row['elevation']}\n")
        
        logging.info(f"Sites file created at: {sites_path} with {len(sites)} sites")
        
        return sites_path
    
    def create_wind_file(self):
        """
        Create a wind file for tephra2.
        
        Returns:
            str: Path to the created wind file
        """
        # Define path
        wind_path = "data/input/wind.txt"
        
        # Get wind parameters from config
        wind_params = self.config.get("wind", {})
        max_height = wind_params.get("max_height", 30000)
        interval = wind_params.get("interval", 1000)
        wind_speed = wind_params.get("speed", 10)
        wind_direction = wind_params.get("direction", 0)
        
        # Generate wind profile
        heights = np.arange(0, max_height + interval, interval)
        
        # Write wind file
        with open(wind_path, 'w') as f:
            for h in heights:
                f.write(f"{h} {wind_speed} {wind_direction}\n")
        
        logging.info(f"Wind file created at: {wind_path} with {len(heights)} levels")
        
        return wind_path
    
    def prepare_inputs(self):
        """
        Prepare all input files for tephra2.
        
        Returns:
            tuple: Paths to the input files (config_path, sites_path, wind_path)
        """
        config_path = self.create_tephra2_config()
        sites_path = self.create_sites_file()
        wind_path = self.create_wind_file()
        
        return config_path, sites_path, wind_path
    
    def run_inversion(self):
        """
        Run the inversion using MCMC.
        
        Returns:
            dict: Results of the inversion containing the parameter chain and statistics
        """
        # Prepare input files
        config_path, sites_path, wind_path, output_path = self._prepare_input_files()
        
        # Ensure tephra2 executable has proper permissions
        tephra2_path = self.config["tephra2"]["executable"]
        self._ensure_executable(tephra2_path)
        
        # Prepare MCMC parameters
        mcmc_params = self._prepare_mcmc_parameters()
        
        # Run MCMC
        print("\nRunning MCMC parameter estimation...")
        try:
            chain, post_chain, acceptance_count, prior_array, likeli_array = metropolis_hastings(
                mcmc_params["initial_values"], 
                mcmc_params["prior_type"], 
                mcmc_params["draw_scale"], 
                mcmc_params["prior_parameters"],
                config_path, 
                sites_path, 
                wind_path, 
                output_path,
                tephra2_path,
                self.config["mcmc"]["n_iterations"],
                0.1,  # Likelihood scale
                self.observations["thickness"].values,
                check_snapshot=100,
                silent=True
            )
            
            # Process results
            best_idx = np.argmax(post_chain)
            best_params = chain[best_idx]
            
            # Store results
            results = {
                "chain": chain,
                "post_chain": post_chain,
                "acceptance_rate": acceptance_count / self.config["mcmc"]["n_iterations"],
                "best_params": best_params,
                "best_posterior": post_chain[best_idx],
                "prior_array": prior_array,
                "likelihood_array": likeli_array
            }
            
            # Log results summary
            logging.info(f"MCMC completed with {self.config['mcmc']['n_iterations']} iterations")
            logging.info(f"Acceptance rate: {results['acceptance_rate']:.2f}")
            
            return results
        
        except Exception as e:
            logging.error(f"Fatal error in MCMC: {str(e)}")
            raise

    def save_results(self, results, output_dir="results"):
        """
        Save inversion results to files.
        
        Args:
            results (dict): Results of the inversion
            output_dir (str, optional): Directory to save results to
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save parameter chain
        chain_df = pd.DataFrame(results["chain"])
        chain_df.columns = list(self.config["parameters"].keys())
        chain_df.to_csv(f"{output_dir}/chain_{timestamp}.csv", index=False)
        
        # Save posterior values
        post_df = pd.DataFrame({
            "posterior": results["post_chain"],
            "prior": results["prior_array"],
            "likelihood": results["likelihood_array"]
        })
        post_df.to_csv(f"{output_dir}/posterior_{timestamp}.csv", index=False)
        
        # Save best parameters
        best_params = pd.DataFrame({
            "parameter": list(self.config["parameters"].keys()),
            "value": results["best_params"]
        })
        best_params.to_csv(f"{output_dir}/best_params_{timestamp}.csv", index=False)
        
        # Save run information
        run_info = {
            "timestamp": timestamp,
            "n_iterations": self.config["mcmc"]["n_iterations"],
            "n_parameters": len(self.config["parameters"]),
            "acceptance_rate": results["acceptance_rate"],
            "best_posterior": results["best_posterior"]
        }
        
        with open(f"{output_dir}/run_info_{timestamp}.txt", 'w') as f:
            for key, value in run_info.items():
                f.write(f"{key}: {value}\n")
        
        logging.info(f"Results saved to {output_dir}")
        
        return f"{output_dir}/chain_{timestamp}.csv"
    
    def plot_results(self, results, output_dir="results", burnin=None):
        """
        Plot inversion results.
        
        Args:
            results (dict): Results of the inversion
            output_dir (str, optional): Directory to save plots to
            burnin (int, optional): Number of samples to discard as burn-in
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set burnin if not provided
        if burnin is None:
            burnin = int(self.config["mcmc"].get("n_burnin", 0))
        
        # Get parameter names
        param_names = list(self.config["parameters"].keys())
        
        # Plot traces
        plt.figure(figsize=(12, 8))
        for i, param in enumerate(param_names):
            plt.subplot(len(param_names), 1, i+1)
            plt.plot(results["chain"][burnin:, i])
            plt.ylabel(param)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/traces_{timestamp}.png")
        
        # Plot posterior
        plt.figure(figsize=(10, 6))
        plt.plot(results["post_chain"][burnin:])
        plt.xlabel("Iterations")
        plt.ylabel("Posterior")
        plt.savefig(f"{output_dir}/posterior_{timestamp}.png")
        
        # Plot parameter distributions
        n_params = len(param_names)
        ncols = min(3, n_params)
        nrows = (n_params + ncols - 1) // ncols
        
        plt.figure(figsize=(ncols*4, nrows*3))
        for i, param in enumerate(param_names):
            plt.subplot(nrows, ncols, i+1)
            plt.hist(results["chain"][burnin:, i], bins=30)
            plt.xlabel(param)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/distributions_{timestamp}.png")
        
        # Plot parameter correlations
        plt.figure(figsize=(12, 12))
        for i in range(n_params):
            for j in range(n_params):
                if i != j:
                    plt.subplot(n_params, n_params, i*n_params + j + 1)
                    plt.plot(results["chain"][burnin:, i], results["chain"][burnin:, j], 'o', markersize=1)
                    plt.xlabel(param_names[i])
                    plt.ylabel(param_names[j])
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlations_{timestamp}.png")
        
        logging.info(f"Plots saved to {output_dir}")