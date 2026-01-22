# scripts/sim/results_io.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

from .sim_types import GroupSpec


# ---------------------------------------------------------------------------
# Column normalization + metadata detection
# ---------------------------------------------------------------------------

_META_REQUIRED_BASE = {
    "scale_plume_factor",
    "scale_mass_factor",
    "plume_estimate",
    "runtime_sec",
}


def _normalize_results_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize legacy column names to the canonical naming used now:

        logm_estimate    -> lnM_estimate
        logm_prior_mean  -> lnM_prior_mean

    Note:
    - Both represent NATURAL LOG in this project; the 'logm_*' name is legacy.
    """
    if df is None or df.empty:
        return df

    rename: Dict[str, str] = {}

    if "logm_estimate" in df.columns and "lnM_estimate" not in df.columns:
        rename["logm_estimate"] = "lnM_estimate"

    if "logm_prior_mean" in df.columns and "lnM_prior_mean" not in df.columns:
        rename["logm_prior_mean"] = "lnM_prior_mean"

    if rename:
        df = df.rename(columns=rename)

    return df


def has_full_metadata(df: pd.DataFrame) -> bool:
    """
    Return True if df looks like a full results_<model>.csv produced by
    scripts.sim.simulate (or rebuilt from chains with metadata columns).

    Accepts either:
      - canonical: lnM_estimate
      - legacy:    logm_estimate (normalized by loader)
    """
    if df is None or df.empty:
        return False

    # Mass estimate column must exist (either canonical or legacy)
    has_mass = ("lnM_estimate" in df.columns) or ("logm_estimate" in df.columns)
    if not has_mass:
        return False

    return _META_REQUIRED_BASE.issubset(df.columns)


# ---------------------------------------------------------------------------
# Public: load results df (auto-normalize, rebuild if needed)
# ---------------------------------------------------------------------------

def load_results_df(
    model: str,
    sim_output_dir: str | Path,
    allow_minimal: bool = True,  # kept for backward compatibility; ignored now
    EXP: Optional[object] = None,
    input_dir_hint: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, bool]:
    """
    Load results_<model>.csv from sim_output_dir.

    Returns:
        (df, rebuilt_from_chains)

    Behaviour:
    - If results_<model>.csv exists and has full metadata columns, we load it,
      normalize legacy columns -> canonical lnM_*, and return rebuilt=False.
    - If it does not exist, or exists but is missing metadata columns,
      we reconstruct a full metadata CSV from the chains and write it back.

    Rebuild assumptions:
    - uses the same run_id ordering as scripts.sim.simulate.run_all_experiments
    - plume_estimate and lnM_estimate are read from the LAST row of each chain
    - runtime_sec cannot be recovered -> NaN
    """
    sim_output_dir = Path(sim_output_dir)
    model = model.lower()
    results_path = sim_output_dir / f"results_{model}.csv"

    # 1) Existing CSV
    if results_path.exists():
        df = pd.read_csv(results_path)
        df = _normalize_results_columns(df)

        if has_full_metadata(df):
            # Optional: write back normalized headers so downstream code is consistent
            try:
                df.to_csv(results_path, index=False)
            except Exception:
                pass
            return df, False

        print(
            f"[WARN] {results_path} exists but is missing metadata columns; "
            "rebuilding from chains."
        )
    else:
        print(
            f"[WARN] {results_path} not found; attempting to rebuild from chains."
        )

    # 2) Rebuild full metadata from chains
    df_full = _rebuild_full_results_from_chains(
        model=model,
        sim_output_dir=sim_output_dir,
        EXP=EXP,
        input_dir_hint=input_dir_hint,
    )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    df_full.to_csv(results_path, index=False)

    return df_full, True


# ---------------------------------------------------------------------------
# Internal: rebuild from chains
# ---------------------------------------------------------------------------

def _guess_input_dir(
    sim_output_dir: Path,
    EXP: Optional[object],
    input_dir_hint: Optional[str | Path],
) -> Path:
    """
    Best-effort guess of the scenario input_dir (used only to read sim_meta.json).
    If guessing fails, returns sim_output_dir (safe fallback).
    """
    # 0) explicit hint
    if input_dir_hint is not None:
        p = Path(input_dir_hint)
        if p.exists():
            return p

    # 1) if experiment folder has a copied input/
    p1 = sim_output_dir / "input"
    if p1.exists():
        return p1

    # 2) sibling layout:
    #   .../data/experiments/<scenario>
    #   .../data/scenarios/<scenario>/input
    # this assumes experiments_root and scenarios_root share the same parent "data"
    try:
        data_root = sim_output_dir.parent.parent  # .../data
        p2 = data_root / "scenarios" / sim_output_dir.name / "input"
        if p2.exists():
            return p2
    except Exception:
        pass

    # 3) EXP.SIM_INPUT_DIR if defined
    if EXP is not None and hasattr(EXP, "SIM_INPUT_DIR"):
        p3 = Path(getattr(EXP, "SIM_INPUT_DIR"))
        if p3.exists():
            return p3

    # 4) fallback
    return sim_output_dir


def _rebuild_full_results_from_chains(
    model: str,
    sim_output_dir: str | Path,
    EXP: Optional[object] = None,
    input_dir_hint: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Rebuild a full results_<model>.csv from chains.

    Notes:
    - Does NOT recompute posteriors or runtimes; best_posterior is NaN and runtime_sec is NaN.
    - Uses the same run_id increment ordering as scripts.sim.simulate.run_all_experiments.
    """
    from scripts.data_handling.config_io import load_config
    from . import simulate  # uses simulate._iter_method_configs and _build_param_config

    # Default EXP if not provided (kept for backward compatibility)
    if EXP is None:
        from . import exp_config as EXP  # type: ignore

    sim_output_dir = Path(sim_output_dir)
    model = model.lower()

    chains_root = sim_output_dir / "chains"
    model_dir = chains_root / model

    if not model_dir.exists():
        raise FileNotFoundError(
            f"No chains directory for model={model}: {model_dir}"
        )

    base_cfg = load_config()

    # log-mass prior std in ln-space: allow EXP override, else default ~ “10x at ~2σ”
    logm_prior_std = float(getattr(EXP, "LOGM_PRIOR_STD", np.log(10.0) / 2.0))

    # best-effort input_dir for sim_meta.json true values
    input_dir = _guess_input_dir(sim_output_dir, EXP, input_dir_hint)

    rows: List[Dict[str, Any]] = []
    run_counter = 0

    for sim_id in range(int(getattr(EXP, "N_REPEATS", 1))):
        for scale_h in getattr(EXP, "PRIOR_FACTORS", [1.0]):
            for scale_m in getattr(EXP, "PRIOR_FACTORS", [1.0]):

                # Build parameter config for this prior centre pair
                # NEW signature: (base_cfg, input_dir, scale_height, scale_mass, logm_prior_std)
                param_cfg = simulate._build_param_config(
                    base_cfg=base_cfg,
                    input_dir=Path(input_dir),
                    scale_height=float(scale_h),
                    scale_mass=float(scale_m),
                    logm_prior_std=float(logm_prior_std),
                )

                for m in getattr(EXP, "MODELS", ["mcmc", "sa", "pso", "es"]):
                    for method_cfg_override in simulate._iter_method_configs(str(m), EXP):
                        run_counter += 1

                        # must preserve run_counter increments exactly,
                        # but only keep rows for the requested model
                        if str(m).lower() != model:
                            continue

                        run_id = run_counter
                        chain_path = model_dir / f"{model}_run{run_id}.csv"
                        if not chain_path.exists():
                            continue

                        df_chain = pd.read_csv(chain_path)
                        if df_chain.empty:
                            continue

                        # Use the last row of the chain as the "best" point
                        best = df_chain.iloc[-1]
                        plume_est = float(best.get("plume_height", np.nan))

                        # Chain uses 'log_mass' parameter name; store in results as lnM_*
                        lnM_est = float(best.get("log_mass", np.nan))

                        plume_prior_mean = float(param_cfg["plume_height"]["prior_mean"])
                        lnM_prior_mean = float(param_cfg["log_mass"]["prior_mean"])

                        seed = int(getattr(EXP, "BASE_SEED", 0) + run_counter)

                        # Merge default method config with override (like simulate.py)
                        base_method_cfg = base_cfg.get(str(m).lower(), {})
                        method_cfg: Dict[str, Any] = base_method_cfg.copy() if isinstance(base_method_cfg, dict) else {}
                        method_cfg.update(method_cfg_override)

                        row: Dict[str, Any] = {
                            "run_id": run_id,
                            "simulation_id": sim_id,
                            "model": model,
                            "seed": seed,
                            "scale_plume_factor": float(scale_h),
                            "scale_mass_factor": float(scale_m),
                            "plume_prior_mean": plume_prior_mean,
                            "lnM_prior_mean": lnM_prior_mean,
                            "plume_estimate": plume_est,
                            "lnM_estimate": lnM_est,
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
            f"Could not rebuild results for model={model}: no usable chains found under {model_dir}"
        )

    df = pd.DataFrame(rows)
    df = _normalize_results_columns(df)

    sort_cols = [c for c in ["simulation_id", "scale_plume_factor", "scale_mass_factor", "run_id"] if c in df.columns]
    if sort_cols:
        df.sort_values(sort_cols, inplace=True)

    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Hyperparameter config helpers
# ---------------------------------------------------------------------------

def _hyperparam_columns_for_model(model: str) -> List[str]:
    model = model.lower()
    if model == "mcmc":
        return ["n_iterations"]
    if model in ("sa", "pso"):
        return ["runs", "restarts"]
    if model == "es":
        return ["n_ens", "n_assimilations"]
    raise ValueError(f"Unknown model: {model}")


def list_model_configs(df: pd.DataFrame, model: str) -> pd.DataFrame:
    model = model.lower()
    hyper_cols = _hyperparam_columns_for_model(model)
    for c in hyper_cols:
        if c not in df.columns:
            raise ValueError(
                f"results_{model}.csv is missing hyperparameter column '{c}'. "
                "You probably have a partial/minimal CSV; delete it and re-run "
                "scripts.sim.simulate, or let load_results_df() rebuild it from chains."
            )

    configs = (
        df[hyper_cols]
        .drop_duplicates()
        .sort_values(hyper_cols)
        .reset_index(drop=True)
    )
    return configs


def get_config_hyperparams(df: pd.DataFrame, model: str, config_index: int) -> Dict[str, Any]:
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

def filter_group_rows(df: pd.DataFrame, spec: GroupSpec, allow_empty: bool = False) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Empty results DataFrame passed to filter_group_rows.")

    # normalize just in case caller bypassed load_results_df
    df = _normalize_results_columns(df)

    mask = np.ones(len(df), dtype=bool)

    # Prior-centre scaling
    if spec.scale_plume_factor is not None and "scale_plume_factor" in df.columns:
        mask &= np.isclose(df["scale_plume_factor"], spec.scale_plume_factor)

    if spec.scale_mass_factor is not None and "scale_mass_factor" in df.columns:
        mask &= np.isclose(df["scale_mass_factor"], spec.scale_mass_factor)

    # Hyperparameters
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
    sim_output_dir = Path(sim_output_dir)
    model = model.lower()
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
        raise ValueError(f"No chain files loaded for model={model}; subset may be empty.")

    # Trim to the shortest length across runs
    min_len = min(len(df) for df in chains)
    chains = [df.iloc[:min_len].reset_index(drop=True) for df in chains]

    burnin = max(0, min(burnin, min_len - 1))
    return chains, burnin
