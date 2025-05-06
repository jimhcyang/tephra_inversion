# tephra_inversion.py
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from scripts.core.mcmc import metropolis_hastings
from scripts.data_handling.build_inputs import build_all
from scripts.data_handling.observation_data import ObservationHandler
from scripts.data_handling.esp_config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("tephra_inversion.log")
    ]
)
LOGGER = logging.getLogger(__name__)


class TephraInversion:
    """
    Class for performing inversion of Tephra2 model using MCMC.
    """

    def __init__(
        self,
        vent_lat: float,
        vent_lon: float,
        vent_elev: float,
        config: Dict = None
    ):
        # 1) Load DEFAULT_CONFIG from config/default_config.py
        self.default_config = load_config()

        # 2) Merge in user-supplied config
        self.config = {}
        # First take all default keys
        for section, vals in self.default_config.items():
            self.config[section] = vals.copy()
        # Then override with any user-provided top-level keys
        config = config or {}
        for k, v in config.items():
            if isinstance(v, dict) and k in self.config:
                # merge sub-dict
                self.config[k].update(v)
            else:
                # override entire section
                self.config[k] = v

        # Ensure directories exist
        self.base_input  = Path(self.config["paths"]["input_dir"])
        self.base_output = Path(self.config["paths"]["output_dir"])
        self.base_input.mkdir(parents=True, exist_ok=True)
        self.base_output.mkdir(parents=True, exist_ok=True)

        # Build or load all inputs (and diagnostic plots if desired)
        self.conf_path, self.esp_path, self.wind_path = build_all(
            vent_lat           = vent_lat,
            vent_lon           = vent_lon,
            vent_elev          = vent_elev,
            base_dir           = self.base_input,
            load_observations  = self.config["mcmc"].get("load_observations", True),
            load_wind          = self.config["mcmc"].get("load_wind", True),
            obs_params         = self.config["mcmc"].get("obs_params", {}),
            wind_params        = self.config["mcmc"].get("wind_params", {}),
            show_plots         = self.config["mcmc"].get("show_plots", False),
        )

        # Load observations into a DataFrame
        obs_handler = ObservationHandler(self.base_input)
        obs_vec, sites = obs_handler.load_observations()
        self.observations = pd.DataFrame({
            "easting":     sites[:, 0],
            "northing":    sites[:, 1],
            "elevation":   sites[:, 2],
            "observation": obs_vec,
        })
        LOGGER.info("Loaded %d observations", len(self.observations))

        # Load ESP parameters into self.config["parameters"]
        self._load_esp_parameters()
        LOGGER.info("TephraInversion initialized")

    def _load_esp_parameters(self) -> None:
        """
        Read esp_input.csv and populate self.config["parameters"].
        """
        esp_path = self.base_input / "esp_input.csv"
        if not esp_path.exists():
            LOGGER.warning("esp_input.csv not found; using default parameters")
            return

        df = pd.read_csv(esp_path)
        LOGGER.info("Loaded %d ESP rows", len(df))

        params: Dict[str, Dict] = {}
        for _, row in df.iterrows():
            name = row["variable_name"]
            if row["prior_type"] == "Fixed":
                params[name] = {
                    "initial_value": float(row["initial_val"]),
                    "prior_type":     "Fixed",
                    "draw_scale":     0.0,
                }
            else:
                entry: Dict = {
                    "initial_value": float(row["initial_val"]),
                    "prior_type":     row["prior_type"],
                    "draw_scale":     float(row.get("draw_scale") or 0.0),
                }
                if row["prior_type"] == "Gaussian":
                    entry["prior_mean"] = float(row["prior_para_a"])
                    entry["prior_std"]  = float(row["prior_para_b"])
                else:
                    entry["prior_min"] = float(row["prior_para_a"])
                    entry["prior_max"] = float(row["prior_para_b"])
                params[name] = entry

        self.config["parameters"] = params

    def _prepare_input_files(self) -> Tuple[str, str, str, str]:
        """
        Confirm existence of input files and return their paths plus the
        Tephra2 output path.
        """
        cfg   = str(self.conf_path)
        sites = str(self.base_input / "sites.csv")
        wind  = str(self.wind_path)
        out   = str(self.config["tephra2"]["output_file"])

        for p in (cfg, sites, wind):
            if not os.path.exists(p):
                LOGGER.error("Input file missing: %s", p)
                raise FileNotFoundError(p)

        LOGGER.info("Using config: %s", cfg)
        LOGGER.info("Using sites:  %s", sites)
        LOGGER.info("Using wind:   %s", wind)
        LOGGER.info("Tephra2 will write to: %s", out)
        return cfg, sites, wind, out

    def run_inversion(self) -> Dict:
        """
        Execute MCMC inversion and return a results dict.
        """
        cfg_path, sites_path, wind_path, out_path = self._prepare_input_files()
        tephra2_exec = self.config["tephra2"]["executable"]

        # Ensure executable permissions
        if os.path.exists(tephra2_exec):
            os.chmod(
                tephra2_exec,
                os.stat(tephra2_exec).st_mode | 0o111
            )
            LOGGER.info("Tephra2 exec ready: %s", tephra2_exec)
        else:
            LOGGER.error("Tephra2 exec not found at %s", tephra2_exec)
            raise FileNotFoundError(tephra2_exec)

        # Prepare parameter vectors
        names      = list(self.config["parameters"].keys())
        pcfg       = self.config["parameters"]
        initial    = np.array([pcfg[n]["initial_value"] for n in names], dtype=float)
        ptype      = np.array([pcfg[n]["prior_type"]    for n in names], dtype=object)
        draw_scale = np.array([
            float(pcfg[n].get("draw_scale") or 0.0)
            for n in names
        ], dtype=float)

        # Build prior parameter array
        prior_para = []
        for n in names:
            entry = pcfg[n]
            if entry["prior_type"] == "Gaussian":
                prior_para.append([entry["prior_mean"], entry["prior_std"]])
            else:
                prior_para.append([
                    entry.get("prior_min", 0.0),
                    entry.get("prior_max", 0.0)
                ])
        prior_para = np.array(prior_para, dtype=float)

        # Merge MCMC defaults + overrides
        mcmc_def = self.default_config["mcmc"]
        mcmc_usr = self.config.get("mcmc", {})
        mcmc     = {**mcmc_def, **mcmc_usr}

        runs     = int(mcmc["n_iterations"])
        burnin   = int(mcmc["n_burnin"])
        log_sigma= float(mcmc["likelihood_sigma"])
        silent   = bool(mcmc["silent"])
        snapshot = int(mcmc["snapshot"])
        # thinning, seed, etc could also be pulled here if needed

        # Run Metropolis–Hastings
        mh = metropolis_hastings(
            initial_plume    = initial,
            prior_type       = ptype,
            prior_para       = prior_para,
            draw_scale       = draw_scale,
            runs             = runs,
            obs_load         = self.observations["observation"].values,
            likelihood_sigma = log_sigma,
            conf_path        = Path(cfg_path),
            sites_csv        = Path(sites_path),
            tephra2_path     = tephra2_exec,
            wind_path        = wind_path,
            burnin           = burnin,
            silent           = silent,
            snapshot         = snapshot,
        )

        # Wrap results
        chain_df  = pd.DataFrame(mh["chain"], columns=names)
        posterior = mh["posterior"]
        best_idx  = int(np.argmax(posterior))
        best_row  = chain_df.iloc[best_idx]

        results = {
            "chain":            chain_df,
            "posterior":        posterior,
            "prior_array":      mh["prior"],
            "likelihood_array": mh["likelihood"],
            "acceptance_rate":  mh["accept_rate"],
            "burnin":           burnin,
            "best_params":      best_row,
            "best_posterior":   float(posterior[best_idx]),
        }
        LOGGER.info(
            "MCMC finished: %d iters, accept=%.2f",
            runs, results["acceptance_rate"]
        )
        return results

    def save_results(self, results: Dict, output_dir: str = None) -> None:
        """
        Save chain, posterior/prior/likelihood, best_params, and run_info.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        od = Path(output_dir or self.config["paths"]["mcmc_dir"])
        od.mkdir(parents=True, exist_ok=True)

        results["chain"].to_csv(od / f"chain_{ts}.csv", index=False)
        pd.DataFrame({
            "posterior":  results["posterior"],
            "prior":      results["prior_array"],
            "likelihood": results["likelihood_array"],
        }).to_csv(od / f"posterior_{ts}.csv", index=False)
        pd.DataFrame({
            "parameter": results["best_params"].index,
            "value":     results["best_params"].values,
        }).to_csv(od / f"best_params_{ts}.csv", index=False)

        # Write run info, pulling n_iterations from the merged config
        mcmc_cfg = self.config.get("mcmc", {})
        n_iter   = mcmc_cfg.get("n_iterations", None)
        with open(od / f"run_info_{ts}.txt", "w") as fh:
            fh.write(f"timestamp: {ts}\n")
            fh.write(f"n_iterations: {n_iter}\n")
            fh.write(f"acceptance_rate: {results['acceptance_rate']:.4f}\n")
            fh.write(f"best_posterior: {results['best_posterior']:.4f}\n")

        LOGGER.info("Results saved to %s", od)

    def plot_results(self, results: Dict, output_dir: str = None) -> None:
        """
        Produce trace, posterior curve, histograms, and correlation plots.
        """
        import matplotlib.pyplot as plt

        od = Path(output_dir or self.config["paths"]["plots_dir"])
        od.mkdir(parents=True, exist_ok=True)

        chain = results["chain"].values
        names = list(results["chain"].columns)
        burnin= results["burnin"]

        # Trace plots
        fig, axes = plt.subplots(len(names), 1, figsize=(8, 2*len(names)), sharex=True)
        for i, n in enumerate(names):
            axes[i].plot(chain[burnin:, i], lw=0.8)
            axes[i].set_ylabel(n)
        axes[-1].set_xlabel("Iteration")
        fig.tight_layout()
        fig.savefig(od / "trace_plots.png", dpi=300)
        plt.close(fig)

        # Posterior curve
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(results["posterior"][burnin:])
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Posterior")
        fig.tight_layout()
        fig.savefig(od / "posterior_curve.png", dpi=300)
        plt.close(fig)

        # Distributions
        n = len(names)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        axs = axs.flat if hasattr(axs, "flat") else axs.flatten()
        for i, param in enumerate(names):
            axs[i].hist(chain[burnin:, i], bins=30)
            axs[i].set_title(param)
        fig.tight_layout()
        fig.savefig(od / "distributions.png", dpi=300)
        plt.close(fig)

        # Pairwise correlations
        if n > 1:
            fig, axes = plt.subplots(n, n, figsize=(3*n, 3*n))
            for i in range(n):
                for j in range(n):
                    axes[i, j].scatter(chain[burnin:, i], chain[burnin:, j], s=1)
                    axes[i, j].set_xticks([])
                    axes[i, j].set_yticks([])
            fig.tight_layout()
            fig.savefig(od / "correlations.png", dpi=300)
            plt.close(fig)

        LOGGER.info("Diagnostic plots saved to %s", od)
