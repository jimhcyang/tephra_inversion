# scripts/sim/plot_traces.py
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _pretty_param_name(col: str) -> str:
    """
    Make chain column names easier to interpret on plots.
    """
    if col == "log_mass":
        return "ln(mass_kg)  [log_mass]"
    return col


def plot_traces_with_mean(
    chains: List[pd.DataFrame],
    burnin: int,
    out_path: Path,
    title: str,
    show: bool = False,
) -> None:
    """
    Multi-run trace plot:
      - light lines: each run
      - bold line: mean across runs at each iteration
    """
    if not chains:
        raise ValueError("No chains to plot.")

    param_cols = list(chains[0].columns)
    n_params = len(param_cols)
    n_runs = len(chains)
    n_steps = len(chains[0])

    arr = np.stack([df[param_cols].to_numpy() for df in chains], axis=0)
    mean_arr = arr.mean(axis=0)
    x = np.arange(n_steps)

    fig, axes = plt.subplots(
        n_params, 1,
        figsize=(8, 2.4 * n_params),
        sharex=True,
        constrained_layout=True,
    )
    if n_params == 1:
        axes = [axes]

    for j, (ax, col) in enumerate(zip(axes, param_cols)):
        for r in range(n_runs):
            ax.plot(x, arr[r, :, j], alpha=0.18, linewidth=0.7, zorder=1)

        ax.plot(
            x,
            mean_arr[:, j],
            linewidth=2.0,
            alpha=0.95,
            zorder=3,
            label="mean across runs",
        )

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

        ax.set_ylabel(_pretty_param_name(col))
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Iteration / step")
    fig.suptitle(title, fontsize=12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_marginals_with_mean_hist(
    chains: List[pd.DataFrame],
    burnin: int,
    out_path: Path,
    title: str,
    show: bool = False,
) -> None:
    """
    Multi-run marginal histograms:
      - filled histogram of all post-burn-in samples across runs
      - step histogram of the mean-by-index series.
    """
    if not chains:
        raise ValueError("No chains to plot.")

    param_cols = list(chains[0].columns)
    n_params = len(param_cols)
    n_steps = len(chains[0])

    arr = np.stack([df[param_cols].to_numpy() for df in chains], axis=0)
    mean_arr = arr.mean(axis=0)

    start = max(0, min(burnin, n_steps - 1))

    fig, axes = plt.subplots(
        n_params, 1,
        figsize=(8, 2.4 * n_params),
        constrained_layout=True,
    )
    if n_params == 1:
        axes = [axes]

    for j, (ax, col) in enumerate(zip(axes, param_cols)):
        raw_samples = arr[:, start:, j].reshape(-1)
        mean_samples = mean_arr[start:, j]

        ax.hist(raw_samples, bins=50, alpha=0.15, edgecolor="none", label="all runs")
        ax.hist(mean_samples, bins=40, histtype="step", linewidth=2.0, alpha=0.9, label="mean-by-index")
        ax.set_ylabel(_pretty_param_name(col))
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Parameter value")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(title, fontsize=12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_single_chain_trace(
    chain: pd.DataFrame,
    out_path: Path,
    title: str,
    show: bool = False,
) -> None:
    """
    Trace plot for a single chain.
    """
    param_cols = list(chain.columns)
    n_params = len(param_cols)
    x = np.arange(len(chain))

    fig, axes = plt.subplots(
        n_params, 1,
        figsize=(8, 2.4 * n_params),
        sharex=True,
        constrained_layout=True,
    )
    if n_params == 1:
        axes = [axes]

    for ax, col in zip(axes, param_cols):
        ax.plot(x, chain[col].to_numpy(), linewidth=1.0)
        ax.set_ylabel(_pretty_param_name(col))
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Iteration / step")
    fig.suptitle(title, fontsize=12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_single_chain_marginals(
    chain: pd.DataFrame,
    out_path: Path,
    title: str,
    show: bool = False,
) -> None:
    """
    Marginal histograms for a single chain.
    """
    param_cols = list(chain.columns)
    n_params = len(param_cols)

    fig, axes = plt.subplots(
        n_params, 1,
        figsize=(8, 2.4 * n_params),
        constrained_layout=True,
    )
    if n_params == 1:
        axes = [axes]

    for ax, col in zip(axes, param_cols):
        ax.hist(chain[col].to_numpy(), bins=50, alpha=0.7, edgecolor="none")
        ax.set_ylabel(_pretty_param_name(col))
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Parameter value")
    fig.suptitle(title, fontsize=12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)
