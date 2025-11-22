# scripts/sim/simulate.py
from __future__ import annotations

import argparse
import importlib
import time
import copy
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from scripts.data_handling.config_io import load_config
from scripts.tephra_inversion import TephraInversion

# Example usage:
#   python -m scripts.sim.simulate --config-module scripts.sim.exp_config_test
#   python -m scripts.sim.simulate --config-module scripts.sim.exp_config


def mass_scale_from_factor(scale: float) -> float:
    """
    Map a 'normal' scale k to a log-style scaling factor for eruption mass.

    Let k be the same type of factor used for plume height (normal scaling).
    We define a corresponding 'L-scale' factor f(k) such that:

        k = 1/3  ->  f = 0.01
        k = 1/2  ->  f = 0.1
        k = 1    ->  f = 1
        k = 2    ->  f = 10
        k = 3    ->  f = 100

    Using:

        if k >= 1:  f = 10**(k - 1)
        if k <= 1:  f = 10**(1 - 1/k)

    Here, f is the multiplier applied to the *true eruption mass*.
    """
    k = float(scale)
    if k <= 0:
        raise ValueError(f"scale must be positive, got {k}")

    if k >= 1.0:
        return 10.0 ** (k - 1.0)
    else:
        return 10.0 ** (1.0 - 1.0 / k)


def _build_param_config(
    base_cfg: Dict[str, Any],
    scale_height: float,
    scale_mass: float,
) -> Dict[str, Dict[str, Any]]:
    """
    Build a TephraInversion-style parameter config where we ONLY shift the
    centers (initial value and prior mean) of:

      - plume_height
      - log_mass  (log of eruption_mass)

    and leave the spreads (prior_para_b) and draw_scale unchanged.

    For plume_height:
      new_center_h = scale_height * true_plume_height

    For eruption_mass:
      mass_factor  = mass_scale_from_factor(scale_mass)
      new_mass     = mass_factor * true_eruption_mass
      log_mass_center = ln(new_mass)

    Notes:
      - 'true_values' in DEFAULT_CONFIG["parameters"] are used as ground truth.
      - If 'true_values' is absent, we fall back to 'variable' initial_val.
    """
    params_cfg = base_cfg["parameters"]
    var = params_cfg["variable"]
    col = var["column_height"]
    em  = var["eruption_mass"]

    true_vals = params_cfg.get("true_values", {})

    # --- True plume height & eruption mass ---------------------------------
    true_h = float(true_vals.get("plume_height", col["initial_val"]))
    true_m = float(true_vals.get("eruption_mass", em["initial_val"]))

    # --- New centers --------------------------------------------------------
    # Normal scaling for plume height
    h_center = float(scale_height) * true_h

    # L-scaling for eruption mass
    mass_factor = mass_scale_from_factor(scale_mass)
    m_center = mass_factor * true_m
    log_m_center = float(np.log(m_center))

    # --- Original spreads & draw scales (unchanged) ------------------------
    h_std = float(col["prior_para_b"])
    h_draw = float(col["draw_scale"])

    # For eruption_mass we follow the existing convention used in TephraInversion:
    # treat prior_para_b as a log-space sigma.
    m_log_std = float(max(em["prior_para_b"], 1e-6))
    m_draw = float(em["draw_scale"])

    params = {
        "plume_height": {
            "initial_value": h_center,
            "prior_type": "Gaussian",
            "prior_mean": h_center,
            "prior_std": h_std,
            "draw_scale": h_draw,
        },
        "log_mass": {
            "initial_value": log_m_center,
            "prior_type": "Gaussian",
            "prior_mean": log_m_center,
            "prior_std": m_log_std,
            "draw_scale": m_draw,
        },
    }
    return params


def _iter_method_configs(model: str, EXP) -> Dict[str, Any]:
    """
    Yield dicts of method-specific config overrides for a given model.
    Each dict is meant to merge into DEFAULT_CONFIG[model].
    """
    if model == "sa":
        for runs in EXP.SA_RUNS:
            for restarts in EXP.SA_RESTARTS:
                yield {
                    "runs": int(runs),
                    "restarts": int(restarts),
                    "print_every": max(int(runs) // 10, 1),
                }

    elif model == "pso":
        for runs in EXP.PSO_RUNS:
            for restarts in EXP.PSO_RESTARTS:
                yield {
                    "runs": int(runs),
                    "restarts": int(restarts),
                    "print_every": max(int(runs) // 10, 1),
                }

    elif model == "es":
        for n_ens in EXP.ES_N_ENS:
            for n_assim in EXP.ES_N_ASSIM:
                yield {
                    "n_ens": int(n_ens),
                    "n_assimilations": int(n_assim),
                    "print_every": int(EXP.ES_PRINT_EVERY),
                }

    elif model == "mcmc":
        for n_iter in EXP.MCMC_N_ITER:
            yield {
                "n_iterations": int(n_iter),
                # keep existing burn-in; you can override here if desired
                "snapshot": max(int(n_iter) // 10, 1),
            }

    else:
        raise ValueError(f"Unknown model: {model}")


def _summarize_run(
    model: str,
    run_id: int,
    simulation_id: int,
    seed: int,
    scale_h: float,
    scale_m: float,
    param_cfg: Dict[str, Dict[str, Any]],
    method_cfg: Dict[str, Any],
    duration: float,
    results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a single row dict with all metadata + outcome of one run.

    Note: 'scale_h' and 'scale_m' are *center* scaling factors:
      - plume_height center → scale_h * true_plume_height
      - eruption_mass center → mass_scale_from_factor(scale_m) * true_eruption_mass
    """
    best = results["best_params"]

    if isinstance(best, pd.Series):
        plume_est = float(best.get("plume_height", np.nan))
        logm_est  = float(best.get("log_mass", np.nan))
    else:  # dict-like
        plume_est = float(best.get("plume_height", np.nan))
        logm_est  = float(best.get("log_mass", np.nan))

    # Prior means used for this run
    plume_prior_mean = float(param_cfg["plume_height"]["prior_mean"])
    logm_prior_mean  = float(param_cfg["log_mass"]["prior_mean"])

    row: Dict[str, Any] = {
        "run_id": run_id,
        "simulation_id": simulation_id,
        "model": model,
        "seed": seed,
        "scale_plume_factor": float(scale_h),
        "scale_mass_factor": float(scale_m),
        "plume_prior_mean": plume_prior_mean,
        "logm_prior_mean": logm_prior_mean,
        "plume_estimate": plume_est,
        "logm_estimate": logm_est,
        "best_posterior": float(results.get("best_posterior", np.nan)),
        "runtime_sec": float(duration),
    }

    # Attach shared method hyperparameters (if present)
    for key in (
        "runs",
        "restarts",
        "T0",
        "alpha",
        "T_end",
        "n_iterations",
        "n_burnin",
        "n_ens",
        "n_assimilations",
        "likelihood_sigma",
        "print_every",
        "snapshot",
    ):
        if key in method_cfg:
            row[key] = method_cfg[key]

    # Add algorithm-specific stats if available
    if model == "mcmc":
        row["acceptance_rate"] = float(results.get("acceptance_rate", np.nan))

    return row


def run_all_experiments(EXP) -> None:
    """
    Main loop over:
      - simulation_id in [0, N_REPEATS)
      - center scaling factors (height × mass)
      - models
      - each model's hyperparameter grid

    Writes:
      - one CSV per model under EXP.SIM_OUTPUT_DIR:
          results_mcmc.csv, results_sa.csv, results_pso.csv, results_es.csv
      - one chain CSV per run under:
          EXP.SIM_OUTPUT_DIR/chains/<model>/<model>_run<run_id>.csv
    """
    # Ensure output dirs exist
    out_dir = Path(EXP.SIM_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    chains_root = out_dir / "chains"
    chains_root.mkdir(parents=True, exist_ok=True)

    # Base config from config/default_config.py
    base_cfg = load_config()

    # Collect rows per model
    rows_by_model: Dict[str, list[Dict[str, Any]]] = {m: [] for m in EXP.MODELS}

    run_counter = 0

    for sim_id in range(EXP.N_REPEATS):
        for scale_h in EXP.PRIOR_FACTORS:
            for scale_m in EXP.PRIOR_FACTORS:
                # Build parameter (prior center) config for this factor pair
                param_cfg = _build_param_config(base_cfg, scale_h, scale_m)

                for model in EXP.MODELS:
                    for method_cfg_override in _iter_method_configs(model, EXP):
                        run_counter += 1
                        run_id = run_counter

                        # Unique seed per run
                        seed = int(EXP.BASE_SEED + run_counter)
                        np.random.seed(seed)

                        # --- Build full config dict for TephraInversion -------
                        cfg = copy.deepcopy(base_cfg)

                        # override paths for this simulation experiment
                        cfg.setdefault("paths", {})
                        cfg["paths"]["input_dir"] = EXP.SIM_INPUT_DIR
                        # other paths stay as in default_config; we don't call save_results()

                        # set method selector (matches the 'method' logic in TephraInversion)
                        cfg["method"] = model

                        # merge method-specific overrides
                        method_cfg = cfg.get(model, {}).copy()
                        method_cfg.update(method_cfg_override)
                        cfg[model] = method_cfg

                        # --- Run inversion ------------------------------------
                        inv = TephraInversion(config=cfg)
                        # Overwrite parameters with our scaled centers
                        inv.config["parameters"] = param_cfg

                        t0 = time.perf_counter()
                        results = inv.run_inversion()
                        t1 = time.perf_counter()

                        duration = t1 - t0

                        # Save chain for later trace / marginal plots
                        model_chain_dir = chains_root / model
                        model_chain_dir.mkdir(parents=True, exist_ok=True)
                        chain_path = model_chain_dir / f"{model}_run{run_id}.csv"
                        results["chain"].to_csv(chain_path, index=False)

                        # Summarize single run
                        row = _summarize_run(
                            model=model,
                            run_id=run_id,
                            simulation_id=sim_id,
                            seed=seed,
                            scale_h=scale_h,
                            scale_m=scale_m,
                            param_cfg=param_cfg,
                            method_cfg=method_cfg,
                            duration=duration,
                            results=results,
                        )
                        rows_by_model[model].append(row)

                        # Minimal progress print
                        print(
                            f"[SIM] run_id={run_id} sim={sim_id} model={model} "
                            f"h_scale={scale_h:.3g} m_scale={scale_m:.3g} "
                            f"runtime={duration:.2f}s"
                        )

    # --- Write per-model CSVs -----------------------------------------------
    for model, rows in rows_by_model.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        out_file = out_dir / f"results_{model}.csv"

        # Append if file exists, else create with header
        if out_file.exists():
            df.to_csv(out_file, mode="a", header=False, index=False)
        else:
            df.to_csv(out_file, index=False)

        print(f"[SIM] wrote {len(rows)} rows to {out_file}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run Tephra2 inversion experiment grid.")
    parser.add_argument(
        "--config-module",
        type=str,
        default="scripts.sim.exp_config",
        help=(
            "Python module path for experiment config "
            "(e.g. scripts.sim.exp_config or scripts.sim.exp_config_test)"
        ),
    )
    args = parser.parse_args(argv)

    # Dynamic import of experiment config
    EXP = importlib.import_module(args.config_module)

    run_all_experiments(EXP)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
