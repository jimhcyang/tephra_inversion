import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import stat
from datetime import datetime
from pathlib import Path

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
from scripts.core.mcmc import metropolis_hastings
from scripts.data_handling.esp_config import load_config

class TephraInversion:
    """
    Class for performing inversion of tephra2 model using MCMC.
    """
    
    def __init__(self, config={}, observations=None):
        """
        Initialize the TephraInversion class.
        
        Args:
            config (dict): Configuration dictionary for the inversion
            observations (pd.DataFrame, optional): Observed deposit data
        """
        self.config = config
        
        # Load default configuration
        self.default_config = load_config()
        
        # If config doesn't contain mcmc settings, use default
        if "mcmc" not in self.config:
            self.config["mcmc"] = self.default_config["mcmc"]
        
        # Ensure directories exist
        os.makedirs("data/input", exist_ok=True)
        os.makedirs("data/output", exist_ok=True)
        
        # If observations not provided, load them from files
        if observations is None:
            try:
                # Load observation mass from CSV
                obs_vec = np.loadtxt("data/input/observations.csv")  # 1-D array
                
                # Check the file format first
                with open("data/input/sites.csv", "r") as f:
                    first_line = f.readline().strip()
                
                # Determine separator based on first line
                if "," in first_line:
                    sites_ar = pd.read_csv("data/input/sites.csv", sep=",", header=None).values
                    logging.info("Using comma separator for sites.csv")
                else:
                    sites_ar = pd.read_csv("data/input/sites.csv", sep=r"\s+", header=None).values
                    logging.info("Using whitespace separator for sites.csv")
                
                # Create observations dataframe
                self.observations = pd.DataFrame({
                    "easting":      sites_ar[:, 0],
                    "northing":     sites_ar[:, 1],
                    "elevation":    sites_ar[:, 2],
                    "observation":  obs_vec,
                })
                
                logging.info(f"Loaded {len(self.observations)} observations automatically")
            except Exception as e:
                logging.error(f"Error loading observations: {str(e)}")
                raise ValueError("Observations must be provided or available in data/input/ directory")
        else:
            self.observations = observations
        
        # Load ESP input parameters
        self._load_esp_parameters()
        
        logging.info("TephraInversion initialized")
    
    def _load_esp_parameters(self):
        """
        Load parameters from esp_input.csv file
        """
        try:
            # Check if esp_input.csv exists
            esp_input_path = Path("data/input/esp_input.csv")
            if not esp_input_path.exists():
                logging.warning("esp_input.csv not found, will use provided config only")
                return
                
            # Read parameters from file
            esp_params = pd.read_csv(esp_input_path)
            logging.info(f"Loaded {len(esp_params)} parameters from esp_input.csv")
            
            # If we have a parameters section in config already, update it with ESP parameters
            if "parameters" not in self.config:
                self.config["parameters"] = {}
                
            # Get variable parameters (non-Fixed)
            for _, row in esp_params[esp_params['prior_type'] != 'Fixed'].iterrows():
                param_name = row['variable_name']
                self.config["parameters"][param_name] = {
                    "initial_value": row['initial_val'],
                    "prior_type": row['prior_type'],
                    "prior_mean" if row['prior_type'] == "Gaussian" else "prior_min": row['prior_para_a'],
                    "prior_std" if row['prior_type'] == "Gaussian" else "prior_max": row['prior_para_b'],
                    "draw_scale": row['draw_scale']
                }
                
            # Get fixed parameters
            for _, row in esp_params[esp_params['prior_type'] == 'Fixed'].iterrows():
                param_name = row['variable_name']
                self.config["parameters"][param_name] = {
                    "initial_value": row['initial_val'],
                    "prior_type": "Fixed",
                    "draw_scale": ""
                }
                
        except Exception as e:
            logging.error(f"Error loading ESP parameters: {str(e)}")
            raise
    
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
    
    def _prepare_input_files(self):
        """
        Prepare the input files for tephra2.
        
        Returns:
            tuple: Paths to the input files (config_path, sites_path, wind_path, output_path)
        """
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
    
    def run_inversion(self):
        """
        Run the Metropolis-Hastings inversion and return a results dict.
        """
        # 0. Input files + exec permissions
        conf_path, sites_path, wind_path, _ = self._prepare_input_files()
        tephra2_exec = self.default_config["tephra2"]["executable"]
        self._ensure_executable(tephra2_exec)
    
        # 1. Assemble MCMC vectors
        pcfg = self.config["parameters"]
        names = list(pcfg.keys())
    
        init_vals = np.array([pcfg[k]["initial_value"] for k in names])
        prior_typ = np.array([pcfg[k]["prior_type"] for k in names])
        draw_scl = np.array([
            float(pcfg[k].get("draw_scale", 0) or 0.0)  # robust numeric
            for k in names
        ])
        prior_par = np.array([
            [pcfg[k].get("prior_mean", pcfg[k].get("prior_min", 0)),
             pcfg[k].get("prior_std", pcfg[k].get("prior_max", 0))]
            if pcfg[k]["prior_type"] == "Gaussian" else
            [pcfg[k].get("prior_min", 0),
             pcfg[k].get("prior_max", 0)]
            for k in names
        ])
    
        # 2. Runtime options
        mcmc_cfg = self.config.get("mcmc", {})
        n_iter = mcmc_cfg.get("n_iterations", 10000)
        n_burn = mcmc_cfg.get("n_burnin", 2000)
        like_sig = mcmc_cfg.get("likelihood_sigma", 0.25)
        silent = mcmc_cfg.get("silent", True)
        snapshot = mcmc_cfg.get("snapshot", 1000)
    
        # 3. Run MH (will raise on fatal Tephra2 errors)
        mh = metropolis_hastings(
            initial_plume=init_vals,
            prior_type=prior_typ,
            prior_para=prior_par,
            draw_scale=draw_scl,
            runs=n_iter,
            obs_load=self.observations["observation"].values,
            likelihood_sigma=like_sig,
            conf_path=Path(conf_path),
            sites_csv=Path(sites_path),
            burnin=n_burn,
            silent=silent,
            snapshot=snapshot,
        )
    
        # 4. Post-process
        chain_np = mh["chain"]                       # (n_iter+1, n_param)
        chain_df = pd.DataFrame(chain_np, columns=names)
    
        best_idx = np.argmax(mh["posterior"])
        best_row = chain_df.iloc[best_idx]
    
        results = {
            "chain": chain_df,                     # DataFrame → notebook-friendly
            "posterior": mh["posterior"],
            "prior_array": mh["prior"],
            "likelihood_array": mh["likelihood"],
            "acceptance_rate": mh["accept_rate"],
            "burnin": n_burn,
            "best_params": best_row,
            "best_posterior": mh["posterior"][best_idx],
        }
    
        logging.info(f"MCMC finished: {n_iter} iters, accept={results['acceptance_rate']:.2f}")
    
        return results

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