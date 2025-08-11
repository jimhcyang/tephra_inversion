from __future__ import annotations
import os, sys, logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from scripts.core.mcmc import metropolis_hastings
from scripts.core.sa import simulated_annealing
from scripts.core.enkf import ensemble_smoother_mda
from scripts.data_handling.config_io import load_config


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("tephra_inversion.log")]
)
LOGGER = logging.getLogger(__name__)


class TephraInversion:
    """
    Unified runner for three inversion methods:
      - Metropolis-Hastings (MCMC)
      - Simulated Annealing (SA)
      - Ensemble Kalman (ES-MDA style)

    Assumes the working inputs already exist under data/input/:
      tephra2.conf, wind.txt, sites.csv, observations.csv
    (Use:  python scripts/download_cerro_negro.py --prepare  to create them.)
    """

    def __init__(self, vent_lat: float | None = None, vent_lon: float | None = None,
                 vent_elev: float | None = None, config: Dict | None = None):
        # Load defaults and merge user config
        self.default_config = load_config()
        self.config: Dict = {}
        for section, vals in self.default_config.items():
            self.config[section] = vals.copy() if isinstance(vals, dict) else vals
        config = config or {}
        for k, v in config.items():
            if isinstance(v, dict) and k in self.config and isinstance(self.config[k], dict):
                self.config[k].update(v)
            else:
                self.config[k] = v

        # Base paths
        self.base_input = Path(self.config.get("paths", {}).get("input_dir", "data/input"))
        self.base_output = Path(self.config.get("paths", {}).get("output_dir", "data/output"))
        self.base_input.mkdir(parents=True, exist_ok=True)
        self.base_output.mkdir(parents=True, exist_ok=True)

        # Required working files (prepared by the downloader/standardizer)
        self.conf_path = self.base_input / "tephra2.conf"
        self.wind_path = self.base_input / "wind.txt"
        self.sites_csv = self.base_input / "sites.csv"
        obs_csv = self.base_input / "observations.csv"

        missing = [p.name for p in (self.conf_path, self.wind_path, self.sites_csv, obs_csv) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing working inputs under data/input/: "
                + ", ".join(missing)
                + "\nRun:  python scripts/download_cerro_negro.py --prepare"
            )

        # Load sites + observations
        sites_arr = np.loadtxt(self.sites_csv)
        obs_arr = np.loadtxt(obs_csv)
        if sites_arr.ndim == 1:
            sites_arr = sites_arr[None, :]
        if obs_arr.ndim == 0:
            obs_arr = np.array([float(obs_arr)])
        if len(obs_arr) != len(sites_arr):
            raise ValueError(
                f"observations.csv length ({len(obs_arr)}) != sites.csv rows ({len(sites_arr)})"
            )

        self.observations = pd.DataFrame({
            "easting":   sites_arr[:, 0],
            "northing":  sites_arr[:, 1],
            "elevation": sites_arr[:, 2],
            "observation": obs_arr,
        })

        # (Re)write clean working copies
        np.savetxt(self.base_input / "observations.csv", obs_arr, fmt="%.6f")
        np.savetxt(self.sites_csv, sites_arr, fmt="%.3f")

        # Load parameters
        self._load_parameters()

        LOGGER.info("TephraInversion initialized")

    # --------------------------------------------------------------------- #
    def _load_parameters(self) -> None:
        """
        Populate self.config['parameters'] with keys:
          'plume_height' (float),
          'log_mass'     (float) = ln(eruption_mass [kg])
        """
        esp_path = self.base_input / "esp_input.csv"
        params: Dict[str, Dict] = {}

        if esp_path.exists():
            df = pd.read_csv(esp_path)
            csv_vals = {row["variable_name"]: row for _, row in df.iterrows()}

            def _maybe_gauss(row, fallback_mean):
                p = {"initial_value": float(row["initial_val"]),
                     "prior_type": str(row["prior_type"]),
                     "draw_scale": float(row.get("draw_scale") or 0.0)}
                if p["prior_type"] == "Gaussian":
                    p["prior_mean"] = float(row.get("prior_para_a", fallback_mean))
                    p["prior_std"]  = float(row.get("prior_para_b", 1.0))
                elif p["prior_type"] == "Uniform":
                    p["prior_min"]  = float(row.get("prior_para_a", fallback_mean))
                    p["prior_max"]  = float(row.get("prior_para_b", fallback_mean))
                else:  # Fixed
                    p["prior_mean"] = p["initial_value"]
                    p["prior_std"]  = 0.0
                return p

            if "column_height" in csv_vals:
                ch = csv_vals["column_height"]
                params["plume_height"] = _maybe_gauss(ch, ch["initial_val"])
            elif "plume_height" in csv_vals:
                ph = csv_vals["plume_height"]
                params["plume_height"] = _maybe_gauss(ph, ph["initial_val"])

            if "log_m" in csv_vals:
                lm = csv_vals["log_m"]
                params["log_mass"] = _maybe_gauss(lm, lm["initial_val"])
            elif "log_mass" in csv_vals:
                lm = csv_vals["log_mass"]
                params["log_mass"] = _maybe_gauss(lm, lm["initial_val"])

        # Fall back to default_config if missing
        need_defaults = any(k not in params for k in ("plume_height", "log_mass"))
        if need_defaults:
            var = self.default_config["parameters"]["variable"]
            params.setdefault("plume_height", {
                "initial_value": float(var["column_height"]["initial_val"]),
                "prior_type": "Gaussian",
                "prior_mean": float(var["column_height"]["prior_para_a"]),
                "prior_std":  float(var["column_height"]["prior_para_b"]),
                "draw_scale": float(var["column_height"]["draw_scale"]),
            })
            em = float(var["eruption_mass"]["initial_val"])
            params.setdefault("log_mass", {
                "initial_value": float(np.log(em)),
                "prior_type": "Gaussian",
                "prior_mean":  float(np.log(em)),
                "prior_std":   float(max(var["eruption_mass"]["prior_para_b"], 1e-6)),
                "draw_scale":  float(var["eruption_mass"]["draw_scale"]),
            })

        self.config["parameters"] = params

    # --------------------------------------------------------------------- #
    def _prepare_input_files(self) -> Tuple[str, str, str]:
        cfg = str(self.conf_path)
        sites = str(self.sites_csv)
        wind = str(self.wind_path)
        for p in (cfg, sites, wind):
            if not os.path.exists(p):
                LOGGER.error("Input file missing: %s", p)
                raise FileNotFoundError(p)
        return cfg, sites, wind

    # --------------------------------------------------------------------- #
    def run_inversion(self) -> Dict:
        cfg_path, sites_path, wind_path = self._prepare_input_files()
        tephra2_exec = self.config["tephra2"]["executable"]
        if os.path.exists(tephra2_exec):
            os.chmod(tephra2_exec, os.stat(tephra2_exec).st_mode | 0o111)
        else:
            raise FileNotFoundError(tephra2_exec)

        # Parameter arrays
        names = list(self.config["parameters"].keys())
        pcfg  = self.config["parameters"]
        initial   = np.array([pcfg[n]["initial_value"] for n in names], dtype=float)
        ptype     = np.array([pcfg[n]["prior_type"] for n in names], dtype=object)
        draw_scale= np.array([float(pcfg[n].get("draw_scale") or 0.0) for n in names], dtype=float)
        prior_para= []
        for n in names:
            entry = pcfg[n]
            if entry["prior_type"] == "Gaussian":
                prior_para.append([entry.get("prior_mean", entry["initial_value"]),
                                   entry.get("prior_std",  1.0)])
            elif entry["prior_type"] == "Uniform":
                prior_para.append([entry.get("prior_min", entry["initial_value"]),
                                   entry.get("prior_max", entry["initial_value"])])
            else:  # Fixed
                prior_para.append([entry.get("initial_value", 0.0),
                                   entry.get("initial_value", 0.0)])
        prior_para = np.array(prior_para, dtype=float)

        method = str(self.config.get("method", "mcmc")).lower()

        if method == "mcmc":
            mcmc_def = self.default_config["mcmc"]
            mcmc_usr = self.config.get("mcmc", {})
            mcmc = {**mcmc_def, **mcmc_usr}
            runs     = int(mcmc["n_iterations"])
            burnin   = int(mcmc["n_burnin"])
            log_sigma= float(mcmc["likelihood_sigma"])
            silent   = bool(mcmc["silent"])
            snapshot = int(mcmc["snapshot"])
            mh = metropolis_hastings(
                initial_plume=initial, prior_type=ptype, prior_para=prior_para, draw_scale=draw_scale,
                runs=runs, obs_load=self.observations["observation"].values, likelihood_sigma=log_sigma,
                conf_path=Path(cfg_path), sites_csv=Path(sites_path),
                tephra2_path=tephra2_exec, wind_path=wind_path,
                burnin=burnin, silent=silent, snapshot=snapshot
            )
            chain_df = pd.DataFrame(mh["chain"], columns=names)
            posterior = mh["posterior"]
            best_idx  = int(np.argmax(posterior))
            best_row  = chain_df.iloc[best_idx]
            return {
                "chain": chain_df,
                "posterior": posterior,
                "prior_array": mh["prior"],
                "likelihood_array": mh["likelihood"],
                "acceptance_rate": mh["accept_rate"],
                "burnin": burnin,
                "best_params": best_row,
                "best_posterior": float(posterior[best_idx]),
            }

        if method == "sa":
            sa_def = self.default_config.get("sa", {})
            sa_usr = self.config.get("sa", {})
            sa = {**sa_def, **sa_usr}
            # adaptive cooling handled inside simulated_annealing when alpha is None
            res = simulated_annealing(
                initial=initial, prior_type=ptype, prior_para=prior_para, draw_scale=draw_scale,
                runs=int(sa.get("runs", 2000)),
                obs=self.observations["observation"].values,
                sigma=float(sa.get("likelihood_sigma", 0.6)),
                conf_path=Path(cfg_path), sites_csv=Path(sites_path),
                tephra2_exec=Path(tephra2_exec), wind_path=Path(wind_path),
                T0=float(sa.get("T0", 1.0)),
                alpha=sa.get("alpha", None),
                restarts=int(sa.get("restarts", 0)),
                seed=sa.get("seed", None),
                silent=bool(sa.get("silent", False)),
                print_every=int(sa.get("print_every", 200)),
                T_end=float(sa.get("T_end", 0.2)),
            )
            chain_df = pd.DataFrame(res["chain"], columns=names)
            best_idx = int(np.nanargmax(res["posterior"]))
            best_row = chain_df.iloc[best_idx]
            return {
                "chain": chain_df,
                "posterior": res["posterior"],
                "prior_array": res["prior"],
                "likelihood_array": res["likelihood"],
                "acceptance_rate": np.nan,
                "burnin": 0,
                "best_params": best_row,
                "best_posterior": float(res["posterior"][best_idx]),
            }

        if method in ("enkf", "es-mda", "ens"):
            ek_def = self.default_config.get("enkf", {})
            ek_usr = self.config.get("enkf", {})
            ek = {**ek_def, **ek_usr}
            out = ensemble_smoother_mda(
                prior_type=ptype, prior_para=prior_para,
                obs=self.observations["observation"].values,
                sigma=float(ek.get("likelihood_sigma", 0.5)),
                conf_path=Path(cfg_path), sites_csv=Path(sites_path),
                tephra2_exec=tephra2_exec, wind_path=wind_path,
                n_ens=int(ek.get("n_ens", 60)),
                n_assimilations=int(ek.get("n_assimilations", 4)),
                inflation=float(ek.get("inflation", 1.02)),
                seed=ek.get("seed", 42),
                silent=bool(ek.get("silent", False)),
                print_every=int(ek.get("print_every", 1)),
                member_update_every=int(ek.get("member_update_every", 0)),  # <- NEW
                # new robustness/exploration knobs (all optional)
                obs_logspace=bool(ek.get("obs_logspace", True)),
                sd_scale=float(ek.get("sd_scale", 1.0)),
                jitter_after_pass=float(ek.get("jitter_after_pass", 0.0)),
                step_trust=float(ek.get("step_trust", 3.0)),
                winsor_k=float(ek.get("winsor_k", 6.0)),
                ridge_cyy=float(ek.get("ridge_cyy", 1e-6)),
                suppress_runtime_warnings=bool(ek.get("suppress_runtime_warnings", True)),
            )
            chain_df = pd.DataFrame(out["ensemble"], columns=names)
            return {
                "chain": chain_df,
                "posterior": np.full(len(chain_df), np.nan),
                "prior_array": np.full(len(chain_df), np.nan),
                "likelihood_array": np.full(len(chain_df), np.nan),
                "acceptance_rate": np.nan,
                "burnin": 0,
                "best_params": chain_df.mean(axis=0),
                "best_posterior": float("nan"),
                "ensemble_history": out.get("ensemble_history", []),
            }

        raise ValueError(f"Unknown method: {method}")

    # --------------------------------------------------------------------- #
    def save_results(self, results: Dict, output_dir: str | None = None) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        od = Path(output_dir or self.config["paths"]["mcmc_dir"])
        od.mkdir(parents=True, exist_ok=True)

        results["chain"].to_csv(od / f"chain_{ts}.csv", index=False)
        pd.DataFrame({
            "posterior": results["posterior"],
            "prior": results["prior_array"],
            "likelihood": results["likelihood_array"],
        }).to_csv(od / f"posterior_{ts}.csv", index=False)
        best = results["best_params"]
        if isinstance(best, pd.Series):
            pd.DataFrame({"parameter": best.index, "value": best.values}).to_csv(
                od / f"best_params_{ts}.csv", index=False
            )
        else:
            pd.DataFrame({"parameter": list(best.keys()), "value": list(best.values())}).to_csv(
                od / f"best_params_{ts}.csv", index=False
            )
        with open(od / f"run_info_{ts}.txt", "w") as fh:
            n_iter = self.config.get("mcmc", {}).get("n_iterations", None)
            fh.write(f"timestamp: {ts}\n")
            fh.write(f"n_iterations: {n_iter}\n")
            fh.write(f"acceptance_rate: {results.get('acceptance_rate', float('nan'))}\n")
            fh.write(f"best_posterior: {results.get('best_posterior', float('nan'))}\n")
        LOGGER.info("Results saved to %s", od)
