#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate simulation visuals for multiple repeats.

We assume experiments have already been run via scripts/sim/simulate.py
using exp_config.py, so that we have:

  SIM_OUTPUT_DIR/
    results_mcmc.csv
    results_sa.csv
    results_pso.csv
    results_es.csv
    chains/
      mcmc/mcmc_run<run_id>.csv
      sa/sa_run<run_id>.csv
      pso/pso_run<run_id>.csv
      es/es_run<run_id>.csv

The main entry-point is `make_demo_plots()`, which generates aggregate
plots for a user-specified subset of:

  - Prior factors (default: 2.5, 1.0, 0.4)
  - SA:  runs, restarts (default: 1000, 3)
  - PSO: runs, restarts (default: 100,  3)
  - ES:  n_ens, n_assimilations (default: 100, 3)
  - MCMC: n_iter (default: 1000)

Each (model, prior_factor, hyperparam combo) gets:

  - <model>_config_XX_trace.png
  - <model>_config_XX_marginals.png

and a small dict (GroupResult) in the returned `summary` list.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Use same general style as your existing notebooks
plt.style.use("seaborn-v0_8")
plt.rcParams["font.family"] = "DejaVu Sans"

# ---------------------------------------------------------------------------
# Config import
# ---------------------------------------------------------------------------

try:
    # Your main experiment config (not the _test one)
    from scripts.sim.exp_config import SIM_OUTPUT_DIR
except Exception:
    # Fallback: let user override paths manually if needed
    SIM_OUTPUT_DIR = "data_sim_cerro/experiments"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GroupSpec:
    model: str
    prior_factor: float
    # Only some of these will be used depending on model
    n_iter: Optional[int] = None  # MCMC
    runs: Optional[int] = None    # SA / PSO
    restarts: Optional[int] = None
    n_ens: Optional[int] = None   # ES
    n_assimilations: Optional[int] = None
    # Internal index for naming
    config_index: int = 0


@dataclass
class GroupResult:
    """Bookkeeping for what we generated (for use in the notebook)."""
    model: str
    prior_factor: float
    config_index: int
    trace_path: Path
    marginals_path: Path
    n_runs: int
    n_steps: int
    param_cols: Sequence[str]


# ---------------------------------------------------------------------------
# Helpers to filter results rows and load chains
# ---------------------------------------------------------------------------

def _filter_group_rows(
    df: pd.DataFrame,
    spec: GroupSpec,
) -> pd.DataFrame:
    """
    Return subset of df matching the desired prior factor and hyperparams.

    If some metadata columns (e.g. scale_plume_factor, scale_mass_factor)
    are missing because results_*.csv was reconstructed from chains only,
    we skip those filters and use whatever runs are available.
    """
    if df.empty:
        raise ValueError("Empty results DataFrame passed into _filter_group_rows.")

    # Start from "everything included"
    mask = np.ones(len(df), dtype=bool)

    # Prior factors (only if columns exist)
    if "scale_plume_factor" in df.columns:
        mask &= df["scale_plume_factor"] == spec.prior_factor
    if "scale_mass_factor" in df.columns:
        mask &= df["scale_mass_factor"] == spec.prior_factor

    # Hyperparameters (guarded by column existence)
    if spec.n_iter is not None and "n_iterations" in df.columns:
        mask &= df["n_iterations"] == spec.n_iter
    if spec.n_iter is not None and "n_iter" in df.columns:
        mask &= df["n_iter"] == spec.n_iter

    if spec.runs is not None and "runs" in df.columns:
        mask &= df["runs"] == spec.runs

    if spec.restarts is not None and "restarts" in df.columns:
        mask &= df["restarts"] == spec.restarts

    if spec.n_ens is not None and "n_ens" in df.columns:
        mask &= df["n_ens"] == spec.n_ens

    if spec.n_assimilations is not None and "n_assimilations" in df.columns:
        mask &= df["n_assimilations"] == spec.n_assimilations

    # Use only successful runs if status column exists
    if "status" in df.columns:
        mask &= df["status"] == "OK"

    subset = df.loc[mask].copy()
    # sort if columns are present
    sort_cols = [c for c in ["simulation_id", "run_id"] if c in subset.columns]
    if sort_cols:
        subset.sort_values(sort_cols, inplace=True)

    if subset.empty:
        raise ValueError(f"No runs found for spec={spec} in this results file.")

    return subset


def _load_chains_for_group(
    model: str,
    subset: pd.DataFrame,
    chains_root: Path,
) -> Tuple[List[pd.DataFrame], int]:
    """
    Load all chain CSVs for the runs in subset.

    Returns:
        chains: list of DataFrames (one per run)
        burnin: integer burn-in index (assumed shared across runs, else min)
    """
    # Assume burnin is constant across these runs; fallback to 0
    if "burnin" in subset.columns:
        burnin_vals = subset["burnin"].dropna().astype(int).tolist()
        burnin = min(burnin_vals) if burnin_vals else 0
    else:
        burnin = 0

    chains: List[pd.DataFrame] = []
    for _, row in subset.iterrows():
        run_id = int(row["run_id"])
        chain_path = chains_root / model / f"{model}_run{run_id}.csv"
        if not chain_path.exists():
            raise FileNotFoundError(f"Missing chain file: {chain_path}")
        df_chain = pd.read_csv(chain_path)
        chains.append(df_chain)

    # Trim all chains to the shortest length for safety
    min_len = min(len(df) for df in chains)
    chains = [df.iloc[:min_len].reset_index(drop=True) for df in chains]

    # Cap burnin at min_len-1
    burnin = max(0, min(burnin, min_len - 1))

    return chains, burnin

def _rebuild_results_from_chains(
    model: str,
    chains_root: Path,
) -> pd.DataFrame:
    """
    Best-effort reconstruction of a minimal results_<model>.csv
    from existing chain files.

    We only recover:
      - run_id
      - model
      - (optionally) a dummy simulation_id = 0

    All other metadata (scale factors, hyperparams, etc.) will be absent
    and thus cannot be filtered on. The aggregator will still work and
    will simply treat all available runs for that model as one pool.
    """
    model_dir = chains_root / model
    if not model_dir.exists():
        raise FileNotFoundError(f"No chains directory for model={model}: {model_dir}")

    rows = []
    for chain_path in sorted(model_dir.glob(f"{model}_run*.csv")):
        stem = chain_path.stem  # e.g. "sa_run12"
        try:
            run_id_str = stem.split("run")[-1]
            run_id = int(run_id_str)
        except Exception:
            # Ignore any weirdly-named files
            continue
        rows.append(
            {
                "run_id": run_id,
                "simulation_id": 0,
                "model": model,
                # scale_plume_factor / scale_mass_factor etc. are unknown here
            }
        )

    if not rows:
        raise FileNotFoundError(
            f"No usable chain files found under {model_dir} for model={model}"
        )

    df = pd.DataFrame(rows)
    df.sort_values(["simulation_id", "run_id"], inplace=True)
    return df

# ---------------------------------------------------------------------------
# Plot routines (multi-trace + averaged marginals)
# ---------------------------------------------------------------------------

def _plot_traces_with_mean(
    model: str,
    spec: GroupSpec,
    chains: List[pd.DataFrame],
    burnin: int,
    out_path: Path,
    show: bool = False,
) -> None:
    """
    For each parameter column:
      - plot all chains (light, transparent),
      - plot the per-index mean across runs (bold).
    """
    if not chains:
        raise ValueError("No chains to plot.")

    param_cols = list(chains[0].columns)
    n_params = len(param_cols)
    n_runs = len(chains)
    n_steps = len(chains[0])

    # Stack into 3D array: (n_runs, n_steps, n_params)
    arr = np.stack([df[param_cols].to_numpy() for df in chains], axis=0)
    mean_arr = arr.mean(axis=0)  # shape: (n_steps, n_params)

    fig, axes = plt.subplots(
        n_params, 1,
        figsize=(8, 2.4 * n_params),
        sharex=True,
        constrained_layout=True,
    )
    if n_params == 1:
        axes = [axes]

    x = np.arange(n_steps)
    for j, (ax, col) in enumerate(zip(axes, param_cols)):
        # Light traces for each run
        for r in range(n_runs):
            ax.plot(
                x,
                arr[r, :, j],
                alpha=0.18,
                linewidth=0.7,
                zorder=1,
            )

        # Mean trajectory across runs
        ax.plot(
            x,
            mean_arr[:, j],
            linewidth=2.0,
            alpha=0.95,
            zorder=3,
            label="mean across runs",
        )

        # Optional: mark burn-in
        if burnin > 0:
            ax.axvline(burnin, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.text(
                burnin,
                ax.get_ylim()[1],
                " burn-in",
                va="top",
                ha="left",
                fontsize=8,
                alpha=0.7,
            )

        ax.set_ylabel(col)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Iteration / step")

    title = (
        f"{model.upper()} | prior σ×={spec.prior_factor} "
        f"| config {spec.config_index:02d}"
    )
    fig.suptitle(title, fontsize=12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def _plot_marginals_with_mean_hist(
    model: str,
    spec: GroupSpec,
    chains: List[pd.DataFrame],
    burnin: int,
    out_path: Path,
    show: bool = False,
) -> None:
    """
    For each parameter:

      - collect all post-burn-in samples from all runs and plot as a very
        transparent histogram,
      - compute the per-index mean across runs (over time) and plot a
        second histogram of those means.

    This keeps all original “data points” visible while highlighting
    what the ensemble is doing on average.
    """
    if not chains:
        raise ValueError("No chains to plot.")

    param_cols = list(chains[0].columns)
    n_params = len(param_cols)
    n_runs = len(chains)
    n_steps = len(chains[0])

    # Stack into (n_runs, n_steps, n_params)
    arr = np.stack([df[param_cols].to_numpy() for df in chains], axis=0)
    mean_arr = arr.mean(axis=0)  # (n_steps, n_params)

    # Restrict to post-burn-in samples
    start = burnin
    if start < 0:
        start = 0
    if start >= n_steps:
        start = 0  # fallback if burnin was degenerate

    fig, axes = plt.subplots(
        n_params, 1,
        figsize=(8, 2.4 * n_params),
        constrained_layout=True,
    )
    if n_params == 1:
        axes = [axes]

    for j, (ax, col) in enumerate(zip(axes, param_cols)):
        # All raw samples across runs
        raw_samples = arr[:, start:, j].reshape(-1)

        # Mean across runs at each index in chain
        mean_samples = mean_arr[start:, j]

        # Base histogram: all samples, low alpha
        ax.hist(
            raw_samples,
            bins=50,
            alpha=0.15,
            edgecolor="none",
            label="all runs",
        )

        # Overlay: histogram of per-index means
        ax.hist(
            mean_samples,
            bins=40,
            histtype="step",
            linewidth=2.0,
            alpha=0.9,
            label="mean-by-index",
        )

        ax.set_ylabel(col)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Parameter value")
    axes[0].legend(fontsize=8)

    title = (
        f"{model.upper()} marginals | prior σ×={spec.prior_factor} "
        f"| config {spec.config_index:02d}"
    )
    fig.suptitle(title, fontsize=12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# High-level driver for the *demo* configs
# ---------------------------------------------------------------------------

def make_demo_plots(
    sim_output_dir: str | Path | None = None,
    show: bool = False,
    demo_priors: Optional[Sequence[float]] = None,
    sa_runs: int = 1000,
    sa_restarts: int = 0,
    pso_runs: int = 100,
    pso_restarts: int = 0,
    es_n_ens: int = 100,
    es_n_assimilations: int = 1,
    mcmc_n_iter: int = 1000,
) -> List[GroupResult]:
    """
    Aggregate and plot a demo subset.

    Args:
        sim_output_dir: Folder with results_*.csv and chains/ subfolder.
                        Defaults to SIM_OUTPUT_DIR from exp_config.py.
        show:           If True, display plots inline as well as saving.
        demo_priors:    Iterable of prior-std scaling factors to include.
                        Default: [2.5, 1.0, 0.4].
        sa_runs, sa_restarts:
                        SA hyperparameters for the subset.
        pso_runs, pso_restarts:
                        PSO hyperparameters for the subset.
        es_n_ens, es_n_assimilations:
                        ES-MDA hyperparameters for the subset.
        mcmc_n_iter:    MCMC iterations for the subset.

    Returns:
        A list of GroupResult entries describing what was saved.
    """
    if sim_output_dir is None:
        sim_output_dir = SIM_OUTPUT_DIR
    sim_output_dir = Path(sim_output_dir)

    root_out = sim_output_dir.parent / "output"
    trace_root = root_out / "trace"
    marg_root = root_out / "marginals"
    trace_root.mkdir(parents=True, exist_ok=True)
    marg_root.mkdir(parents=True, exist_ok=True)

    chains_root = sim_output_dir / "chains"

    # Default priors if not provided
    if demo_priors is None:
        demo_priors = [2.5, 1.0, 0.4]

    # Spec per model, using the passed-in hyperparams
    demo_specs: Dict[str, List[GroupSpec]] = {
        "sa": [
            GroupSpec(
                model="sa",
                prior_factor=pf,
                runs=sa_runs,
                restarts=sa_restarts,
                config_index=i,
            )
            for i, pf in enumerate(demo_priors)
        ],
        "pso": [
            GroupSpec(
                model="pso",
                prior_factor=pf,
                runs=pso_runs,
                restarts=pso_restarts,
                config_index=i,
            )
            for i, pf in enumerate(demo_priors)
        ],
        "es": [
            GroupSpec(
                model="es",
                prior_factor=pf,
                n_ens=es_n_ens,
                n_assimilations=es_n_assimilations,
                config_index=i,
            )
            for i, pf in enumerate(demo_priors)
        ],
        "mcmc": [
            GroupSpec(
                model="mcmc",
                prior_factor=pf,
                n_iter=mcmc_n_iter,
                config_index=i,  # 00, 01, 02 within this model
            )
            for i, pf in enumerate(demo_priors)
        ],
    }

    results: List[GroupResult] = []

    chains_root = sim_output_dir / "chains"

    for model, specs in demo_specs.items():
        results_path = sim_output_dir / f"results_{model}.csv"

        # Try primary: results_<model>.csv
        if results_path.exists():
            df_results = pd.read_csv(results_path)
        else:
            # Fallback: attempt to reconstruct from chains
            print(f"[WARN] Missing {results_path}; attempting to rebuild from chains/...")
            try:
                df_results = _rebuild_results_from_chains(model, chains_root)
                print(
                    f"[WARN] Rebuilt minimal results for model={model} from "
                    f"{len(df_results)} chain file(s)."
                )
            except FileNotFoundError as e:
                # No CSV and no chains: nothing we can do → skip this model
                print(
                    f"[WARN] {e}. Skipping model={model} in make_demo_plots()."
                )
                continue

        for spec in specs:
            subset = _filter_group_rows(df_results, spec)
            chains, burnin = _load_chains_for_group(model, subset, chains_root)

            param_cols = list(chains[0].columns)
            n_runs = len(chains)
            n_steps = len(chains[0])

            base_tag = f"{model}_config_{spec.config_index:04d}"
            trace_path = trace_root / f"{base_tag}_trace.png"
            marg_path  = marg_root  / f"{base_tag}_marginals.png"

            _plot_traces_with_mean(
                model=model,
                spec=spec,
                chains=chains,
                burnin=burnin,
                out_path=trace_path,
                show=show,
            )

            _plot_marginals_with_mean_hist(
                model=model,
                spec=spec,
                chains=chains,
                burnin=burnin,
                out_path=marg_path,
                show=show,
            )

            results.append(
                GroupResult(
                    model=model,
                    prior_factor=spec.prior_factor,
                    config_index=spec.config_index,
                    trace_path=trace_path,
                    marginals_path=marg_path,
                    n_runs=n_runs,
                    n_steps=n_steps,
                    param_cols=param_cols,
                )
            )

    return results


if __name__ == "__main__":
    # CLI entry: simple demo run with defaults
    summary = make_demo_plots(show=False)
    print("Generated demo plots:")
    for gr in summary:
        print(
            f"  {gr.model.upper()} | prior={gr.prior_factor} | "
            f"config={gr.config_index:02d} | runs={gr.n_runs} | steps={gr.n_steps}"
        )
        print(f"    trace:     {gr.trace_path}")
        print(f"    marginals: {gr.marginals_path}")
