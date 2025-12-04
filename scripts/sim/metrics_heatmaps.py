# scripts/sim/metrics_heatmaps.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from .sim_types import GroupSpec
from .results_io import (
    load_results_df,
    list_model_configs,
    get_config_hyperparams,
    filter_group_rows,
)

# ---------------------------------------------------------------------
# Core: compute signed relative-error & runtime grids
# ---------------------------------------------------------------------

def compute_accuracy_time_grids(
    sim_output_dir: str | Path,
    model: str,
    config_index: int,
    true_plume_height: float,
    true_eruption_mass: float,
    EXP,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    For a given model + config_index, compute 2D grids over
    (height_factor, mass_factor) of

        - signed relative error (average of plume & mass, as fraction),
        - runtime (seconds).

    The relative errors are always computed as (prediction - true) / true,
    using the true_plume_height and true_eruption_mass passed in.

    Returns
    -------
    height_factors, mass_factors : 1D np.ndarray
        Prior scaling factors on the Y (height) and X (mass) axes.
    err_grid : 2D np.ndarray
        Median signed relative error for each (height_factor, mass_factor).
    time_grid : 2D np.ndarray
        Median runtime (seconds) for each (height_factor, mass_factor).
    hyper_cfg : dict
        Hyperparameter configuration for this config_index.
    """
    sim_output_dir = Path(sim_output_dir)
    model = model.lower()

    # Load results and restrict to this model
    df, _ = load_results_df(model=model, sim_output_dir=sim_output_dir)
    df_model = df[df["model"].str.lower() == model].copy()
    if df_model.empty:
        raise ValueError(f"No rows for model={model} in results file.")

    # Hyperparameters for this config (n_iterations / runs / etc.)
    hyper_cfg = get_config_hyperparams(df_model, model, config_index)

    # 1D list of prior scaling factors (length 7 in exp_config.PRIOR_FACTORS)
    height_factors = np.asarray(EXP.PRIOR_FACTORS, dtype=float)
    mass_factors = np.asarray(EXP.PRIOR_FACTORS, dtype=float)

    n_h = len(height_factors)
    n_m = len(mass_factors)

    err_grid = np.full((n_h, n_m), np.nan, dtype=float)
    time_grid = np.full((n_h, n_m), np.nan, dtype=float)

    # Loop over the 7×7 prior-factor grid
    for i, scale_h in enumerate(height_factors):
        for j, scale_m in enumerate(mass_factors):
            spec = GroupSpec(
                model=model,
                scale_plume_factor=scale_h,
                scale_mass_factor=scale_m,
                n_iter=hyper_cfg.get("n_iterations"),
                runs=hyper_cfg.get("runs"),
                restarts=hyper_cfg.get("restarts"),
                n_ens=hyper_cfg.get("n_ens"),
                n_assimilations=hyper_cfg.get("n_assimilations"),
                config_index=config_index,
            )

            subset = filter_group_rows(df_model, spec, allow_empty=True)
            if subset.empty:
                continue

            # Predictions
            plume_est = subset["plume_estimate"].to_numpy(dtype=float)
            logm_est = subset["logm_estimate"].to_numpy(dtype=float)
            mass_est = np.exp(logm_est)

            # Signed relative errors: (prediction - true) / true
            rel_plume = (plume_est - true_plume_height) / true_plume_height
            rel_mass = (mass_est - true_eruption_mass) / true_eruption_mass

            # Combine height + mass into a single scalar error per run
            rel_combined = 0.5 * (rel_plume + rel_mass)

            # Median across repeats
            err_grid[i, j] = np.nanmedian(rel_combined)

            # Median runtime
            if "runtime_sec" in subset.columns:
                runtime_vals = subset["runtime_sec"].to_numpy(dtype=float)
                if np.isfinite(runtime_vals).any():
                    time_grid[i, j] = np.nanmedian(runtime_vals)

    return height_factors, mass_factors, err_grid, time_grid, hyper_cfg

# ---------------------------------------------------------------------
# Per-config accuracy/time heatmaps
# ---------------------------------------------------------------------

def plot_accuracy_time_heatmaps_for_config(
    sim_output_dir: str | Path,
    model: str,
    config_index: int,
    true_plume_height: float,
    true_eruption_mass: float,
    EXP,
    cmap: str = "bwr",
    show: bool = False,
) -> Path:
    """
    Plot a 7×7 accuracy heatmap (and optionally time heatmap) for one
    model + config_index, over (height_factor, mass_factor).

    Accuracy is signed relative error, averaged over plume & mass,
    plotted in percent. All errors are (prediction - true)/true.
    """
    sim_output_dir = Path(sim_output_dir)
    model = model.lower()

    (
        h_factors,
        m_factors,
        err_grid,
        time_grid,
        hyper_cfg,
    ) = compute_accuracy_time_grids(
        sim_output_dir=sim_output_dir,
        model=model,
        config_index=config_index,
        true_plume_height=true_plume_height,
        true_eruption_mass=true_eruption_mass,
        EXP=EXP,
    )

    err_pct = err_grid * 100.0

    # Clip visualised errors to [-100, 100] so colours saturate at ±100%
    err_pct_plot = np.clip(err_pct, -100.0, 100.0)

    root_out = sim_output_dir.parent / "output_heatmaps"
    root_out.mkdir(parents=True, exist_ok=True)

    has_time = np.isfinite(time_grid).any()

    if has_time:
        fig, (ax_acc, ax_time) = plt.subplots(
            1, 2, figsize=(10, 4), constrained_layout=True
        )
    else:
        fig, ax_acc = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)
        ax_time = None

    # Fixed, symmetric diverging norm at ±100%
    norm = TwoSlopeNorm(vmin=-100.0, vcenter=0.0, vmax=100.0)

    # Accuracy panel
    im_acc = ax_acc.imshow(
        err_pct_plot,
        cmap=cmap,
        norm=norm,
        origin="lower",
        aspect="auto",
    )
    
    cbar = fig.colorbar(im_acc, ax=ax_acc, fraction=0.046, pad=0.04)
    cbar.set_label("avg((plume_true, mass_true)) error [%]")
    cbar.set_ticks([-100, -50, 0, 50, 100])
    
    ax_acc.grid(False, which="both")
    ax_acc.tick_params(axis="both", which="both", length=0, width=0)
    ax_acc.set_xticks(np.arange(len(m_factors)))
    ax_acc.set_yticks(np.arange(len(h_factors)))
    ax_acc.set_xticklabels([f"{v:g}" for v in m_factors])
    ax_acc.set_yticklabels([f"{v:g}" for v in h_factors])
    ax_acc.set_xlabel("mass prior factor")
    ax_acc.set_ylabel("height prior factor")
    ax_acc.set_title("Signed relative error [%]")
    ax_acc.tick_params(axis="both", which="major", length=0, width=0)
    
    # --- Draw gridlines on cell edges for accuracy panel ---
    n_h, n_m = err_pct.shape

    # Minor ticks at cell boundaries: -0.5, 0.5, 1.5, ..., n_m-0.5
    ax_acc.set_xticks(np.arange(-0.5, n_m, 1.0), minor=True)
    ax_acc.set_yticks(np.arange(-0.5, n_h, 1.0), minor=True)

    # Grid on the minor ticks = borders of the cells
    ax_acc.grid(which="minor", color="black", linestyle="-", linewidth=0.5)

    # Hide minor tick marks themselves
    ax_acc.tick_params(which="minor", bottom=False, left=False)

    # Ensure the image extends exactly to the outer borders
    ax_acc.set_xlim(-0.5, n_m - 0.5)
    ax_acc.set_ylim(-0.5, n_h - 0.5)

    cbar = fig.colorbar(im_acc, ax=ax_acc, fraction=0.046, pad=0.04)
    cbar.set_label("avg((plume_true, mass_true)) error [%]")

    # Time panel (if available)
    if has_time and ax_time is not None:
        im_time = ax_time.imshow(
            time_grid,
            cmap="Greens",
            origin="lower",
            aspect="auto",
        )
        ax_time.grid(False, which="both")
        ax_time.tick_params(axis="both", which="both", length=0, width=0)
        ax_time.set_xticks(np.arange(len(m_factors)))
        ax_time.set_yticks(np.arange(len(h_factors)))
        ax_time.set_xticklabels([f"{v:g}" for v in m_factors])
        ax_time.set_yticklabels([f"{v:g}" for v in h_factors])
        ax_time.set_xlabel("mass prior factor")
        ax_time.set_ylabel("height prior factor")
        ax_time.set_title("Runtime [s] (median)")
        ax_time.tick_params(axis="both", which="major", length=0, width=0)

        # --- Draw gridlines on cell edges for runtime panel ---
        n_h, n_m = time_grid.shape

        ax_time.set_xticks(np.arange(-0.5, n_m, 1.0), minor=True)
        ax_time.set_yticks(np.arange(-0.5, n_h, 1.0), minor=True)

        ax_time.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
        ax_time.tick_params(which="minor", bottom=False, left=False)

        ax_time.set_xlim(-0.5, n_m - 0.5)
        ax_time.set_ylim(-0.5, n_h - 0.5)
        
        cbar_t = fig.colorbar(im_time, ax=ax_time, fraction=0.046, pad=0.04)
        cbar_t.set_label("runtime_sec")

    # Config label
    cfg_label = ""
    if model == "mcmc" and "n_iterations" in hyper_cfg:
        cfg_label = f"n_iter={hyper_cfg['n_iterations']}"
    elif model in ("sa", "pso") and "runs" in hyper_cfg:
        cfg_label = f"runs={hyper_cfg.get('runs')}, restarts={hyper_cfg.get('restarts')}"
    elif model == "es" and "n_ens" in hyper_cfg:
        cfg_label = f"n_ens={hyper_cfg.get('n_ens')}, n_assim={hyper_cfg.get('n_assimilations')}"

    fig.suptitle(
        f"{model.upper()} — signed relative error vs true [%] | config {config_index}: {cfg_label}",
        fontsize=12,
    )

    png_path = root_out / f"{model}_cfg{config_index:02d}_accuracy.png"
    fig.savefig(png_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return png_path

# ---------------------------------------------------------------------
# Per-model 2x2 grid of accuracy heatmaps (configs 0–3)
# ---------------------------------------------------------------------

def plot_accuracy_time_heatmaps_for_model(
    sim_output_dir: str | Path,
    model: str,
    true_plume_height: float,
    true_eruption_mass: float,
    EXP,
    cmap: str = "bwr",
    show: bool = False,
) -> Path:
    """
    For a given model (mcmc / sa / pso / es), plot a 2×2 grid of
    accuracy heatmaps (no time) for up to the first 4 config_index values.

    Each subplot is a 7×7 heatmap over (height_factor, mass_factor).

    All accuracy values are signed relative errors (prediction - true)/true,
    and the colour scale is shared across configs and data-driven, so
    changing the true values actually changes the colour map.
    """
    sim_output_dir = Path(sim_output_dir)
    model = model.lower()

    # Load once to figure out how many configs exist
    df, _ = load_results_df(model=model, sim_output_dir=sim_output_dir, allow_minimal=True)
    df_model = df[df["model"].str.lower() == model].copy()
    if df_model.empty:
        raise ValueError(f"No rows for model={model} in results file.")

    configs = list_model_configs(df_model, model)
    n_cfg = min(4, len(configs))
    if n_cfg == 0:
        raise ValueError(f"No hyperparameter configs found for model={model}.")

    root_out = sim_output_dir.parent / "output_heatmaps"
    root_out.mkdir(parents=True, exist_ok=True)

    # First pass: compute grids and collect all error values
    stored = []
    all_err_values = []

    for cfg_idx in range(n_cfg):
        h_factors, m_factors, err_grid, _time_grid, hyper_cfg = compute_accuracy_time_grids(
            sim_output_dir=sim_output_dir,
            model=model,
            config_index=cfg_idx,
            true_plume_height=true_plume_height,
            true_eruption_mass=true_eruption_mass,
            EXP=EXP,
        )
        err_pct = err_grid * 100.0
        stored.append((h_factors, m_factors, err_pct, hyper_cfg))

        valid = err_pct[np.isfinite(err_pct)]
        if valid.size:
            all_err_values.append(valid)
            
        norm = TwoSlopeNorm(vmin=-100.0, vcenter=0.0, vmax=100.0)
        
    # Second pass: draw with shared norm
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes = axes.ravel()

    last_im = None

    for idx in range(n_cfg):
        ax = axes[idx]
        h_factors, m_factors, err_pct, hyper_cfg = stored[idx]

        # Clip visualised errors to [-100, 100] so colours saturate at ±100%
        err_pct_plot = np.clip(err_pct, -100.0, 100.0)

        im = ax.imshow(
            err_pct_plot,
            cmap=cmap,
            norm=norm,
            origin="lower",
            aspect="auto",
        )
        last_im = im
        
        ax.grid(False, which="both")
        ax.tick_params(axis="both", which="both", length=0, width=0)
        ax.set_xticks(np.arange(len(m_factors)))
        ax.set_yticks(np.arange(len(h_factors)))
        ax.set_xticklabels(
            [f"{v:g}" for v in m_factors],
            rotation=45,
            ha="right",
            fontsize=7,
        )
        ax.set_yticklabels([f"{v:g}" for v in h_factors], fontsize=7)
        ax.tick_params(axis="both", which="major", length=0, width=0)
        
        # --- Draw gridlines on cell edges for each config panel ---
        n_h, n_m = err_pct.shape

        ax.set_xticks(np.arange(-0.5, n_m, 1.0), minor=True)
        ax.set_yticks(np.arange(-0.5, n_h, 1.0), minor=True)

        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.4)
        ax.tick_params(which="minor", bottom=False, left=False)

        ax.set_xlim(-0.5, n_m - 0.5)
        ax.set_ylim(-0.5, n_h - 0.5)
        
        if idx % 2 == 0:
            ax.set_ylabel("height prior factor")
        if idx >= 2:
            ax.set_xlabel("mass prior factor")

        cfg_label = ""
        if model == "mcmc" and "n_iterations" in hyper_cfg:
            cfg_label = f"n_iter={hyper_cfg['n_iterations']}"
        elif model in ("sa", "pso") and "runs" in hyper_cfg:
            cfg_label = f"runs={hyper_cfg.get('runs')}, restarts={hyper_cfg.get('restarts')}"
        elif model == "es" and "n_ens" in hyper_cfg:
            cfg_label = f"n_ens={hyper_cfg.get('n_ens')}, n_assim={hyper_cfg.get('n_assimilations')}"

        ax.set_title(f"config {idx}: {cfg_label}", fontsize=9)

    # Remove unused axes if < 4 configs
    for k in range(n_cfg, 4):
        fig.delaxes(axes[k])

    # Shared colourbar
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes[:n_cfg], fraction=0.046, pad=0.04)
        cbar.set_label("avg((plume_true, mass_true)) error [%]")
        cbar.set_ticks([-100, -50, 0, 50, 100])
        
    fig.suptitle(
        f"{model.upper()} — signed relative error vs true [%] (first {n_cfg} configs)",
        fontsize=12,
    )

    png_path = root_out / f"{model}_all_configs_accuracy.png"
    fig.savefig(png_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return png_path