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

def _sort_timewise(df: pd.DataFrame) -> pd.DataFrame:
    # Try to respect any explicit iteration/time columns if present
    for col in ["iter", "iteration", "step", "time", "t"]:
        if col in df.columns:
            return df.sort_values(col, kind="mergesort").reset_index(drop=True)
    return df.reset_index(drop=True)

def _enkf_group0(df: pd.DataFrame) -> pd.DataFrame:
    # Prefer common group/member columns; fall back to unfiltered if none exist
    for col in ["group", "ensemble", "ens", "member", "grp", "gid"]:
        if col in df.columns:
            try:
                return df[df[col] == 0].reset_index(drop=True)
            except Exception:
                pass
    return df.reset_index(drop=True)

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
    max_points_per_method: int = 10000,  # used for MCMC only
    sa_step: int = 10,                 # plot every Nth SA point, connected
    seed: int = 20250812,
) -> str:
    """
    Overlay 2D trajectories/clouds for MCMC, SA, and EnKF:
      • MCMC: very tiny, super low-opacity circles (randomly subsampled).
      • SA:   every `sa_step` points, connected with lines; higher opacity; diamonds.
      • EnKF: only the 0-th group; squares, same size as SA, low opacity.

    Returns the saved image path.
    """
    save_path = Path(save_path); save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 7))
    ax = plt.gca()
    rng = np.random.default_rng(seed)

    # ─────────── MCMC: tiny, super faint circles ───────────
    if (
        isinstance(mcmc, dict)
        and isinstance(mcmc.get("chain", None), pd.DataFrame)
        and {xparam, yparam}.issubset(mcmc["chain"].columns)
    ):
        df = mcmc["chain"][[xparam, yparam]].dropna().copy()
        if len(df) > 0:
            df = _sample_rows(df, max_points_per_method, rng)
            ax.scatter(
                df[xparam], df[yparam],
                s=9, marker="o", linewidths=0,
                alpha=0.09, c="tab:blue", edgecolors="none", label="MCMC"
            )

    # ─────────── SA: every Nth point, line + diamonds (start=X, end=★) ───────────
    if (
        isinstance(sa, dict)
        and isinstance(sa.get("chain", None), pd.DataFrame)
        and {xparam, yparam}.issubset(sa["chain"].columns)
    ):
        df = sa["chain"][[xparam, yparam] + [c for c in ["iter","iteration","step","time","t"] if c in sa["chain"].columns]].dropna().copy()
        df = _sort_timewise(df)
        df_step = df.iloc[::max(1, int(sa_step))].reset_index(drop=True)
        if len(df_step) > 0:
            # line first (so points sit on top)
            ax.plot(
                df_step[xparam].values, df_step[yparam].values,
                linestyle="-", linewidth=0.25, alpha=0.75, color="tab:orange", label="SA", zorder=1
            )

            # diamonds for intermediate nodes
            if len(df_step) > 2:
                inter = df_step.iloc[1:-1]
                ax.scatter(
                    inter[xparam], inter[yparam],
                    s=25, marker="D", linewidths=0.25,
                    alpha=0.25, c="tab:orange", edgecolors="none", zorder=2
                )

            # first point: X
            start = df_step.iloc[0]
            ax.scatter(
                [start[xparam]], [start[yparam]],
                s=25, marker="X", linewidths=0.0,
                alpha=0.75, c="tab:orange", zorder=3
            )

            # last point: star
            end = df_step.iloc[-1]
            ax.scatter(
                [end[xparam]], [end[yparam]],
                s=30, marker="*", linewidths=0,
                alpha=0.95, c="tab:orange", edgecolors="none", zorder=4
            )

    # ─────────── EnKF: group 0 only, subsample every sa_step, squares ───────────
    if (
        isinstance(enkf, dict)
        and isinstance(enkf.get("chain", None), pd.DataFrame)
        and {xparam, yparam}.issubset(enkf["chain"].columns)
    ):
        df = enkf["chain"].copy()
        df = _enkf_group0(df)
        # keep any time/iteration column so _sort_timewise can use it
        time_cols = [c for c in ["iter", "iteration", "step", "time", "t"] if c in df.columns]
        df = df[[xparam, yparam] + time_cols].dropna().copy()
        if len(df) > 0:
            df = _sort_timewise(df)
            df_step = df.iloc[::max(1, int(sa_step))].reset_index(drop=True)
            if len(df_step) > 0:
                ax.scatter(
                    df_step[xparam], df_step[yparam],
                    s=16, marker="s", linewidths=0,
                    alpha=0.16, c="tab:green", edgecolors="none", label="EnKF (group 0)"
                )

    ax.set_xlabel("Plume Height (m)")
    ax.set_ylabel("Log Eruption Mass (ln kg)")
    ax.set_title(title.replace("\u2011", "-"))
    ax.grid(True, ls="--", alpha=0.35)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
    return str(save_path)
