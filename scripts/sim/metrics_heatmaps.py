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
    (height_factor, mass_factor) of:

        - signed relative error (average of plume & mass, as fraction),
        - runtime (seconds).

    The relative errors are always computed as (prediction - true) / true.
    """
    sim_output_dir = Path(sim_output_dir)
    model = model.lower()

    df, _ = load_results_df(model=model, sim_output_dir=sim_output_dir, allow_minimal=True)
    df_model = df[df["model"].str.lower() == model].copy()
    if df_model.empty:
        raise ValueError(f"No rows for model={model} in results file.")

    hyper_cfg = get_config_hyperparams(df_model, model, config_index)

    height_factors = np.asarray(EXP.PRIOR_FACTORS, dtype=float)
    mass_factors = np.asarray(EXP.PRIOR_FACTORS, dtype=float)

    n_h = len(height_factors)
    n_m = len(mass_factors)

    err_grid = np.full((n_h, n_m), np.nan, dtype=float)
    time_grid = np.full((n_h, n_m), np.nan, dtype=float)

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

            plume_est = subset["plume_estimate"].to_numpy(dtype=float)
            logm_est = subset["lnM_estimate"].to_numpy(dtype=float)
            mass_est = np.exp(logm_est)

            rel_plume = (plume_est - true_plume_height) / true_plume_height
            rel_mass = (mass_est - true_eruption_mass) / true_eruption_mass
            rel_combined = 0.5 * (rel_plume + rel_mass)

            err_grid[i, j] = np.nanmedian(rel_combined)

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

    SAVES INSIDE the scenario experiment folder:
        <sim_output_dir>/output_heatmaps/
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
    err_pct_plot = np.clip(err_pct, -100.0, 100.0)

    # per-scenario output
    root_out = sim_output_dir / "output_heatmaps"
    root_out.mkdir(parents=True, exist_ok=True)

    has_time = np.isfinite(time_grid).any()

    if has_time:
        fig, (ax_acc, ax_time) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    else:
        fig, ax_acc = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)
        ax_time = None

    norm = TwoSlopeNorm(vmin=-100.0, vcenter=0.0, vmax=100.0)

    im_acc = ax_acc.imshow(
        err_pct_plot,
        cmap=cmap,
        norm=norm,
        origin="lower",
        aspect="auto",
    )

    # Ticks/labels
    ax_acc.set_xticks(np.arange(len(m_factors)))
    ax_acc.set_yticks(np.arange(len(h_factors)))
    ax_acc.set_xticklabels([f"{v:g}" for v in m_factors])
    ax_acc.set_yticklabels([f"{v:g}" for v in h_factors])
    ax_acc.set_xlabel("mass prior factor")
    ax_acc.set_ylabel("height prior factor")
    ax_acc.set_title("Signed relative error [%]")
    ax_acc.tick_params(axis="both", which="both", length=0, width=0)

    # Cell gridlines
    n_h, n_m = err_pct.shape
    ax_acc.set_xticks(np.arange(-0.5, n_m, 1.0), minor=True)
    ax_acc.set_yticks(np.arange(-0.5, n_h, 1.0), minor=True)
    ax_acc.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
    ax_acc.tick_params(which="minor", bottom=False, left=False)
    ax_acc.set_xlim(-0.5, n_m - 0.5)
    ax_acc.set_ylim(-0.5, n_h - 0.5)

    cbar = fig.colorbar(im_acc, ax=ax_acc, fraction=0.046, pad=0.04)
    cbar.set_label("avg((plume_true, mass_true)) error [%]")
    cbar.set_ticks([-100, -50, 0, 50, 100])

    if has_time and ax_time is not None:
        im_time = ax_time.imshow(
            time_grid,
            cmap="Greens",
            origin="lower",
            aspect="auto",
        )
        ax_time.set_xticks(np.arange(len(m_factors)))
        ax_time.set_yticks(np.arange(len(h_factors)))
        ax_time.set_xticklabels([f"{v:g}" for v in m_factors])
        ax_time.set_yticklabels([f"{v:g}" for v in h_factors])
        ax_time.set_xlabel("mass prior factor")
        ax_time.set_ylabel("height prior factor")
        ax_time.set_title("Runtime [s] (median)")
        ax_time.tick_params(axis="both", which="both", length=0, width=0)

        n_h2, n_m2 = time_grid.shape
        ax_time.set_xticks(np.arange(-0.5, n_m2, 1.0), minor=True)
        ax_time.set_yticks(np.arange(-0.5, n_h2, 1.0), minor=True)
        ax_time.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
        ax_time.tick_params(which="minor", bottom=False, left=False)
        ax_time.set_xlim(-0.5, n_m2 - 0.5)
        ax_time.set_ylim(-0.5, n_h2 - 0.5)

        cbar_t = fig.colorbar(im_time, ax=ax_time, fraction=0.046, pad=0.04)
        cbar_t.set_label("runtime_sec")

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
    For a given model, plot a 2×2 grid of accuracy heatmaps (first 4 configs).

    SAVES INSIDE the scenario experiment folder:
        <sim_output_dir>/output_heatmaps/
    """
    sim_output_dir = Path(sim_output_dir)
    model = model.lower()

    df, _ = load_results_df(model=model, sim_output_dir=sim_output_dir, allow_minimal=True)
    df_model = df[df["model"].str.lower() == model].copy()
    if df_model.empty:
        raise ValueError(f"No rows for model={model} in results file.")

    configs = list_model_configs(df_model, model)
    n_cfg = min(4, len(configs))
    if n_cfg == 0:
        raise ValueError(f"No hyperparameter configs found for model={model}.")

    # per-scenario output
    root_out = sim_output_dir / "output_heatmaps"
    root_out.mkdir(parents=True, exist_ok=True)

    stored = []

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

    norm = TwoSlopeNorm(vmin=-100.0, vcenter=0.0, vmax=100.0)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes = axes.ravel()

    last_im = None

    for idx in range(n_cfg):
        ax = axes[idx]
        h_factors, m_factors, err_pct, hyper_cfg = stored[idx]

        err_pct_plot = np.clip(err_pct, -100.0, 100.0)

        im = ax.imshow(
            err_pct_plot,
            cmap=cmap,
            norm=norm,
            origin="lower",
            aspect="auto",
        )
        last_im = im

        ax.set_xticks(np.arange(len(m_factors)))
        ax.set_yticks(np.arange(len(h_factors)))
        ax.set_xticklabels([f"{v:g}" for v in m_factors], rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels([f"{v:g}" for v in h_factors], fontsize=7)
        ax.tick_params(axis="both", which="both", length=0, width=0)

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

    for k in range(n_cfg, 4):
        fig.delaxes(axes[k])

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
