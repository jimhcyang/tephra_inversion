# scripts/sim/results_io.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

from .sim_types import GroupSpec

# Columns that indicate we have the full metadata produced by simulate._summarize_run
_META_REQUIRED = {
    "scale_plume_factor",
    "scale_mass_factor",
    "plume_estimate",
    "logm_estimate",
    "runtime_sec",
}


def has_full_metadata(df: pd.DataFrame) -> bool:
    """
    Return True if df looks like a full results_<model>.csv produced by
    scripts.sim.simulate (i.e., not an old minimal CSV).
    """
    return _META_REQUIRED.issubset(df.columns)


def load_results_df(
    model: str,
    sim_output_dir: str | Path,
    allow_minimal: bool = True,  # kept for backward compatibility; not used
) -> Tuple[pd.DataFrame, bool]:
    """
    Load results_<model>.csv from sim_output_dir.

    Returns:
        (df, rebuilt_from_chains)

    Behaviour:
    - If results_<model>.csv exists and has full metadata columns, we just load
      and return it with rebuilt_from_chains = False.
    - If it does not exist, or exists but is a 'minimal' CSV (missing metadata),
      we reconstruct a full metadata CSV from the chains and write it back to disk.

      For these reconstructed rows:
        - plume_estimate / logm_estimate are taken from the LAST row of each chain.
        - scale_plume_factor / scale_mass_factor are inferred from the experiment
          config loops (same order as scripts.sim.simulate).
        - runtime_sec is set to NaN, because we cannot recover wall-clock time.
    """
    sim_output_dir = Path(sim_output_dir)
    results_path = sim_output_dir / f"results_{model}.csv"

    # 1) Existing full-metadata CSV
    if results_path.exists():
        df = pd.read_csv(results_path)
        if has_full_metadata(df):
            return df, False

        print(
            f"[WARN] {results_path} exists but is missing metadata columns; "
            "rebuilding from chains."
        )
    else:
        print(
            f"[WARN] {results_path} not found; attempting to rebuild from chains."
        )

    # 2) Rebuild full metadata from chains and write file once
    df_full = _rebuild_full_results_from_chains(model, sim_output_dir)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    df_full.to_csv(results_path, index=False)

    return df_full, True


def _rebuild_full_results_from_chains(
    model: str,
    sim_output_dir: str | Path,
) -> pd.DataFrame:
    """
    Rebuild a full results_<model>.csv from chains, using the same loop order
    as scripts.sim.simulate.run_all_experiments.

    We DO NOT recompute posteriors or runtimes; 'best_posterior' and
    'acceptance_rate' are set to NaN, and 'runtime_sec' is NaN.
    """
    from scripts.data_handling.config_io import load_config
    from . import simulate
    from . import exp_config as EXP

    sim_output_dir = Path(sim_output_dir)
    chains_root = sim_output_dir / "chains"
    model_dir = chains_root / model

    if not model_dir.exists():
        raise FileNotFoundError(
            f"No chains directory for model={model}: {model_dir}"
        )

    base_cfg = load_config()

    rows: List[Dict[str, Any]] = []
    run_counter = 0

    for sim_id in range(EXP.N_REPEATS):
        for scale_h in EXP.PRIOR_FACTORS:
            for scale_m in EXP.PRIOR_FACTORS:
                # Build parameter config for this prior centre pair
                param_cfg = simulate._build_param_config(base_cfg, scale_h, scale_m)

                for m in EXP.MODELS:
                    for method_cfg_override in simulate._iter_method_configs(m, EXP):
                        run_counter += 1

                        # We only care about the requested model; but we must
                        # increment run_counter in the SAME way simulate.py did.
                        if m != model:
                            continue

                        run_id = run_counter
                        chain_path = model_dir / f"{model}_run{run_id}.csv"
                        if not chain_path.exists():
                            # Could happen if you killed runs midway; skip.
                            continue

                        df_chain = pd.read_csv(chain_path)
                        if df_chain.empty:
                            continue

                        # Use the last row of the chain as the "best" point
                        best = df_chain.iloc[-1]
                        plume_est = float(best.get("plume_height", np.nan))
                        logm_est = float(best.get("log_mass", np.nan))

                        plume_prior_mean = float(
                            param_cfg["plume_height"]["prior_mean"]
                        )
                        logm_prior_mean = float(param_cfg["log_mass"]["prior_mean"])

                        seed = int(EXP.BASE_SEED + run_counter)

                        # Merge default method config with override, like simulate.py
                        base_method_cfg = base_cfg.get(m, {})
                        if isinstance(base_method_cfg, dict):
                            method_cfg: Dict[str, Any] = base_method_cfg.copy()
                        else:
                            method_cfg = {}
                        method_cfg.update(method_cfg_override)

                        row: Dict[str, Any] = {
                            "run_id": run_id,
                            "simulation_id": sim_id,
                            "model": model,
                            "seed": seed,
                            "scale_plume_factor": float(scale_h),
                            "scale_mass_factor": float(scale_m),
                            "plume_prior_mean": plume_prior_mean,
                            "logm_prior_mean": logm_prior_mean,
                            "plume_estimate": plume_est,
                            "logm_estimate": logm_est,
                            "best_posterior": np.nan,
                            "runtime_sec": np.nan,  # cannot recover after the fact
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

                        if model == "mcmc":
                            row["acceptance_rate"] = np.nan

                        rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"Could not rebuild results for model={model}: no usable chains "
            f"found under {model_dir}"
        )

    df = pd.DataFrame(rows)

    sort_cols = [
        c for c in ["simulation_id", "scale_plume_factor", "scale_mass_factor", "run_id"]
        if c in df.columns
    ]
    if sort_cols:
        df.sort_values(sort_cols, inplace=True)

    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Hyperparameter config helpers
# ---------------------------------------------------------------------------

def _hyperparam_columns_for_model(model: str) -> List[str]:
    """
    Decide which columns define a 'config' for each model.
    """
    model = model.lower()
    if model == "mcmc":
        return ["n_iterations"]
    if model in ("sa", "pso"):
        return ["runs", "restarts"]
    if model == "es":
        return ["n_ens", "n_assimilations"]
    raise ValueError(f"Unknown model: {model}")


def list_model_configs(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """
    Return a small DataFrame of unique hyperparameter combos for this model.

    Rows are sorted; index 0..(n_configs-1) can be used as config_index.
    """
    model = model.lower()
    hyper_cols = _hyperparam_columns_for_model(model)
    for c in hyper_cols:
        if c not in df.columns:
            raise ValueError(
                f"results_{model}.csv is missing hyperparameter column '{c}'. "
                "You probably have an old minimal CSV; delete it and re-run "
                "scripts.sim.simulate, or let load_results_df() rebuild it "
                "from chains."
            )

    configs = (
        df[hyper_cols]
        .drop_duplicates()
        .sort_values(hyper_cols)
        .reset_index(drop=True)
    )
    return configs


def get_config_hyperparams(
    df: pd.DataFrame,
    model: str,
    config_index: int,
) -> Dict[str, Any]:
    """
    Return a dict of hyperparameters for the given model/config index.

    Example:
        cfg = get_config_hyperparams(df, "sa", 2)
        # -> {"runs": 1000, "restarts": 4}
    """
    configs = list_model_configs(df, model)
    if not (0 <= config_index < len(configs)):
        raise IndexError(
            f"config_index={config_index} out of range for model={model}. "
            f"Available configs: {len(configs)}."
        )
    row = configs.iloc[config_index]
    return row.to_dict()


# ---------------------------------------------------------------------------
# Group filtering + chain loading
# ---------------------------------------------------------------------------

def filter_group_rows(
    df: pd.DataFrame,
    spec: GroupSpec,
    allow_empty: bool = False,
) -> pd.DataFrame:
    """
    Filter df for rows matching the GroupSpec.

    - If scale_plume_factor / scale_mass_factor are set in spec, we match them
      against the corresponding columns (if present).
    - If hyperparameters are set (n_iter, runs, restarts, n_ens, n_assimilations),
      we also enforce matches when the columns exist.

    If allow_empty is False and no rows match, we raise a ValueError.
    """
    if df.empty:
        raise ValueError("Empty results DataFrame passed to filter_group_rows.")

    mask = np.ones(len(df), dtype=bool)

    # Prior-centre scaling
    if spec.scale_plume_factor is not None and "scale_plume_factor" in df.columns:
        mask &= np.isclose(df["scale_plume_factor"], spec.scale_plume_factor)

    if spec.scale_mass_factor is not None and "scale_mass_factor" in df.columns:
        mask &= np.isclose(df["scale_mass_factor"], spec.scale_mass_factor)

    # Hyperparameters (if spec has them and df has matching columns)
    if spec.n_iter is not None and "n_iterations" in df.columns:
        mask &= df["n_iterations"] == spec.n_iter

    if spec.runs is not None and "runs" in df.columns:
        mask &= df["runs"] == spec.runs

    if spec.restarts is not None and "restarts" in df.columns:
        mask &= df["restarts"] == spec.restarts

    if spec.n_ens is not None and "n_ens" in df.columns:
        mask &= df["n_ens"] == spec.n_ens

    if spec.n_assimilations is not None and "n_assimilations" in df.columns:
        mask &= df["n_assimilations"] == spec.n_assimilations

    subset = df.loc[mask].copy()

    sort_cols = [c for c in ["simulation_id", "run_id"] if c in subset.columns]
    if sort_cols:
        subset.sort_values(sort_cols, inplace=True)

    if subset.empty and not allow_empty:
        raise ValueError(f"No runs found for spec={spec} in this results file.")

    return subset


def load_chains_for_group(
    model: str,
    subset: pd.DataFrame,
    sim_output_dir: str | Path,
) -> Tuple[List[pd.DataFrame], int]:
    """
    Given a set of rows (one per run) for a model, load all corresponding
    chain CSVs from sim_output_dir/chains/<model>/.

    Returns:
        chains: list of DataFrames (one per run)
        burnin: inferred burn-in index (min over 'burnin' column if present).
    """
    sim_output_dir = Path(sim_output_dir)
    chains_root = sim_output_dir / "chains"
    model_dir = chains_root / model

    if "burnin" in subset.columns:
        burnin_vals = subset["burnin"].dropna().astype(int).tolist()
        burnin = min(burnin_vals) if burnin_vals else 0
    else:
        burnin = 0

    chains: List[pd.DataFrame] = []

    for _, row in subset.iterrows():
        run_id = int(row["run_id"])
        chain_path = model_dir / f"{model}_run{run_id}.csv"
        if not chain_path.exists():
            raise FileNotFoundError(f"Missing chain file: {chain_path}")
        df_chain = pd.read_csv(chain_path)
        chains.append(df_chain)

    if not chains:
        raise ValueError(
            f"No chain files loaded for model={model}; subset may be empty."
        )

    # Trim all chains to the shortest length for safety
    min_len = min(len(df) for df in chains)
    chains = [df.iloc[:min_len].reset_index(drop=True) for df in chains]

    burnin = max(0, min(burnin, min_len - 1))
    return chains, burnin
