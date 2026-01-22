# scripts/sim/simple_plots.py
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .sim_types import GroupSpec
from .results_io import (
    load_results_df,
    get_config_hyperparams,
    filter_group_rows,
    load_chains_for_group,
)


def _normalize_prior_factors(
    prior_factors: float | Sequence[float] | None,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Turn prior_factors into (scale_plume_factor, scale_mass_factor).

    - If None       -> (None, None)  (no filtering on factors)
    - If scalar     -> (val, val)
    - If len == 2   -> (val[0], val[1])
    """
    if prior_factors is None:
        return None, None

    if isinstance(prior_factors, (int, float)):
        v = float(prior_factors)
        return v, v

    prior_factors = list(prior_factors)
    if len(prior_factors) != 2:
        raise ValueError(
            "prior_factors must be a scalar or length-2 iterable "
            f"(got length {len(prior_factors)})"
        )
    return float(prior_factors[0]), float(prior_factors[1])


def _pretty_param_name(col: str) -> str:
    """
    Make chain parameter names more interpretable.
    """
    if col == "log_mass":
        return "ln(mass_kg)  [log_mass]"
    if col in ("ln_mass",):
        return "ln(mass_kg)  [ln_mass]"
    return col


def _true_value_for_param(
    param_name: str,
    true_plume_height: float | None,
    true_eruption_mass: float | None,
) -> Optional[float]:
    """
    Map parameter name to its 'true' value, if provided.

    - plume_height -> true_plume_height
    - log_mass / ln_mass -> ln(true_eruption_mass)
    """
    if param_name == "plume_height":
        return float(true_plume_height) if true_plume_height is not None else None

    if param_name in ("log_mass", "ln_mass", "log_m", "ln_m"):
        if true_eruption_mass is None:
            return None
        return float(np.log(true_eruption_mass))

    return None


def _load_results_df_compat(
    model: str,
    sim_output_dir: Path,
    EXP: Any | None,
) -> tuple[pd.DataFrame, bool]:
    """
    Backward/forward compatible wrapper:
    - Some versions of results_io.load_results_df accept EXP=...
    - Older ones don't.
    """
    try:
        return load_results_df(  # type: ignore
            model=model,
            sim_output_dir=sim_output_dir,
            allow_minimal=False,
            EXP=EXP,
        )
    except TypeError:
        return load_results_df(  # type: ignore
            model=model,
            sim_output_dir=sim_output_dir,
            allow_minimal=False,
        )


def plot_single_config_traces_and_marginals(
    sim_output_dir: str | Path,
    model: str,
    prior_factors: float | Sequence[float] | None,
    config_index: int,
    run_index: int,
    true_plume_height: float | None = None,
    true_eruption_mass: float | None = None,
    show: bool = False,
    EXP: Any | None = None,
) -> tuple[Path, Path]:
    """
    Plot *one* chain (trace + marginal) for a given model/config/prior setup.

    NOTE: EXP is optional but important if results CSV must be rebuilt from chains,
    because the rebuild needs the correct loop order (PRIOR_FACTORS, MODELS, etc.)
    from the config module used in the simulation.
    """
    sim_output_dir = Path(sim_output_dir)
    model = model.lower()

    df, _rebuilt = _load_results_df_compat(model=model, sim_output_dir=sim_output_dir, EXP=EXP)
    df_model = df[df["model"].str.lower() == model].copy()
    if df_model.empty:
        raise ValueError(f"No rows for model={model} in results CSV.")

    hyper_cfg = get_config_hyperparams(df_model, model, config_index)

    scale_h, scale_m = _normalize_prior_factors(prior_factors)

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

    subset = filter_group_rows(df_model, spec, allow_empty=False)

    if run_index < 0 or run_index >= len(subset):
        raise IndexError(
            f"run_index={run_index} out of range for this group; "
            f"found {len(subset)} matching runs."
        )

    subset_run = subset.iloc[[run_index]].copy()
    chains, _burnin = load_chains_for_group(model, subset_run, sim_output_dir)
    chain = chains[0]

    param_cols = list(chain.columns)
    n_steps = len(chain)
    burnin50 = n_steps // 2

    arr = chain[param_cols].to_numpy()
    post_arr = arr[burnin50:, :]

    post_mean = post_arr.mean(axis=0)
    post_low = np.quantile(post_arr, 0.025, axis=0)
    post_high = np.quantile(post_arr, 0.975, axis=0)

    true_vals = [
        _true_value_for_param(col, true_plume_height, true_eruption_mass)
        for col in param_cols
    ]

    # IMPORTANT CHANGE: write inside THIS scenario folder
    root_out = sim_output_dir / "output_simple"
    root_out.mkdir(parents=True, exist_ok=True)

    h_label = f"{scale_h:g}" if scale_h is not None else "NA"
    m_label = f"{scale_m:g}" if scale_m is not None else "NA"

    run_id = int(subset_run.iloc[0]["run_id"])
    base_tag = f"{model}_h{h_label}_m{m_label}_cfg{config_index:02d}_run{run_id}"

    trace_path = root_out / f"{base_tag}_trace.png"
    marg_path = root_out / f"{base_tag}_marginals.png"

    # ---------------- Trace plot ----------------
    x = np.arange(n_steps)

    fig_tr, axes_tr = plt.subplots(
        len(param_cols), 1,
        figsize=(8, 2.4 * len(param_cols)),
        sharex=True,
        constrained_layout=True,
    )
    if len(param_cols) == 1:
        axes_tr = [axes_tr]

    for j, (ax, col, tval) in enumerate(zip(axes_tr, param_cols, true_vals)):
        y = arr[:, j]
        ax.plot(x, y, lw=1.0, alpha=0.9, label="chain")

        if tval is not None:
            ax.axhline(tval, color="red", linestyle=":", linewidth=1.3, label="true")

        ax.axhline(
            post_mean[j],
            color="green",
            linestyle=":",
            linewidth=1.3,
            label="post-mean (last 50%)" if j == 0 else None,
        )
        ax.axhline(
            post_low[j],
            color="green",
            linestyle="-",
            linewidth=1.0,
            alpha=0.8,
            label="95% interval" if j == 0 else None,
        )
        ax.axhline(post_high[j], color="green", linestyle="-", linewidth=1.0, alpha=0.8)

        ax.set_ylabel(_pretty_param_name(col))
        ax.grid(True, alpha=0.2)
        ax.axvline(burnin50, color="k", linestyle="--", linewidth=0.8, alpha=0.5)

    axes_tr[-1].set_xlabel("Iteration")
    axes_tr[0].legend(fontsize=8, loc="upper right")

    cfg_label = ""
    if model == "mcmc" and "n_iterations" in hyper_cfg:
        cfg_label = f"n_iter={hyper_cfg['n_iterations']}"
    elif model in ("sa", "pso") and "runs" in hyper_cfg:
        cfg_label = f"runs={hyper_cfg.get('runs')}, restarts={hyper_cfg.get('restarts')}"
    elif model == "es" and "n_ens" in hyper_cfg:
        cfg_label = f"n_ens={hyper_cfg.get('n_ens')}, n_assim={hyper_cfg.get('n_assimilations')}"

    fig_tr.suptitle(
        f"{model.upper()} trace | h×={h_label}, m×={m_label} | {cfg_label} | run {run_id}",
        fontsize=11,
    )

    fig_tr.savefig(trace_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig_tr)

    # ---------------- Marginals (last 50%) ----------------
    fig_m, axes_m = plt.subplots(
        len(param_cols), 1,
        figsize=(8, 2.4 * len(param_cols)),
        constrained_layout=True,
    )
    if len(param_cols) == 1:
        axes_m = [axes_m]

    for j, (ax, col, tval) in enumerate(zip(axes_m, param_cols, true_vals)):
        samples = np.asarray(post_arr[:, j], dtype=float)
        finite = np.isfinite(samples)
        finite_samples = samples[finite]

        can_hist = (
            finite_samples.size >= 2
            and not np.isclose(finite_samples.min(), finite_samples.max())
        )

        if can_hist:
            ax.hist(
                finite_samples,
                bins=40,
                alpha=0.35,
                edgecolor="none",
                label="post samples (last 50%)",
            )
        elif finite_samples.size > 0:
            v = float(finite_samples[0])
            ax.axvline(
                v,
                color="C0",
                linestyle="-",
                linewidth=2.0,
                alpha=0.7,
                label="post samples (degenerate)" if j == 0 else None,
            )

        if tval is not None:
            ax.axvline(tval, color="red", linestyle=":", linewidth=1.3, label="true")

        ax.axvline(
            post_mean[j],
            color="green",
            linestyle=":",
            linewidth=1.3,
            label="post-mean (last 50%)" if j == 0 else None,
        )
        ax.axvline(post_low[j], color="green", linestyle="-", linewidth=1.0, alpha=0.8,
                   label="95% interval" if j == 0 else None)
        ax.axvline(post_high[j], color="green", linestyle="-", linewidth=1.0, alpha=0.8)

        ax.set_ylabel(_pretty_param_name(col))
        ax.grid(True, alpha=0.2)

    axes_m[-1].set_xlabel("Parameter value")
    axes_m[0].legend(fontsize=8, loc="upper right")

    fig_m.suptitle(
        f"{model.upper()} marginals | h×={h_label}, m×={m_label} | {cfg_label} | run {run_id}",
        fontsize=11,
    )

    fig_m.savefig(marg_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig_m)

    return trace_path, marg_path
