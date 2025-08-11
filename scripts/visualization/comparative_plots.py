from __future__ import annotations
from typing import Dict, Optional, List
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _norm_range(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    lo, hi = np.nanmin(values), np.nanmax(values)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return np.ones_like(values)
    return (values - lo) / (hi - lo)

def _sample_rows(df: pd.DataFrame, k: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(df) <= k:
        return df
    idx = rng.choice(len(df), size=k, replace=False)
    return df.iloc[idx].reset_index(drop=True)

def scatter_2d_progress(
    mcmc: Optional[dict],
    sa: Optional[dict],
    enkf: Optional[dict],
    xparam: str = "plume_height",
    yparam: str = "log_mass",
    title: str = "Inversion trajectories",
    save_path: str | Path = "data/output/plots/method_trajectories.png",
    show: bool = True,
    *,
    max_points_per_method: int = 256,
    base_alpha: float = 0.01,
    seed: int = 1234,
) -> str:
    """
    Overlay 2D clouds for MCMC, SA, and EnKF in (xparam, yparam) space.
    - No connecting lines (less clutter).
    - Very low alpha by default.
    - Randomly subsample up to `max_points_per_method` points per method.
    """
    save_path = Path(save_path); save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 7))
    ax = plt.gca()
    rng = np.random.default_rng(seed)

    def _plot_cloud(df: pd.DataFrame, color: str, label: str):
        df = df[[xparam, yparam]].dropna()
        df = _sample_rows(df, max_points_per_method, rng)
        idx = np.arange(len(df))
        w = _norm_range(idx)
        alpha = np.clip(base_alpha + 0.6 * w, 0.0, 1.0)
        ax.scatter(df[xparam], df[yparam], s=18, c=color, alpha=np.clip(alpha, 0.01, 0.35),
                   edgecolors="none", label=label)

    if mcmc and xparam in mcmc["chain"].columns and yparam in mcmc["chain"].columns:
        _plot_cloud(mcmc["chain"].copy(), "tab:blue", "MCMC")

    if sa and xparam in sa["chain"].columns and yparam in sa["chain"].columns:
        _plot_cloud(sa["chain"].copy(), "tab:orange", "SA")

    if enkf and xparam in enkf["chain"].columns and yparam in enkf["chain"].columns:
        _plot_cloud(enkf["chain"].copy(), "tab:green", "EnKF")

    ax.set_xlabel("Plume Height (m)")
    ax.set_ylabel("Log Eruption Mass (ln kg)")
    ax.set_title(title.replace("\u2011", "-"))
    ax.grid(True, ls="--", alpha=0.35)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show: plt.show()
    else: plt.close()
    return str(save_path)
