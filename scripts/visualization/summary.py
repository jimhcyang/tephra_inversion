from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd

def posterior_summary_from_chain(chain_df: pd.DataFrame, burnin: int = 0) -> pd.DataFrame:
    """95% CI summary from an MCMC-like chain (rows are samples)."""
    post = chain_df.iloc[burnin:]
    stats = []
    for col in post.columns:
        x = post[col].to_numpy()
        stats.append({
            "parameter": col,
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
            "ci2.5": float(np.percentile(x, 2.5)),
            "ci97.5": float(np.percentile(x, 97.5)),
        })
    return pd.DataFrame(stats).set_index("parameter")

def summary_for_methods(mcmc: dict, sa: dict, enkf: dict, vary_only: Tuple[str, ...] = ("plume_height","log_mass")) -> pd.DataFrame:
    """
    Build a side-by-side table with:
      - MCMC: mean + 95% CI
      - SA: point estimate only
      - EnKF: ensemble mean + 95% empirical CI
    Only shows `vary_only` parameters.
    """
    rows = []
    # MCMC
    if mcmc:
        sm = posterior_summary_from_chain(mcmc["chain"], mcmc.get("burnin", 0))
        for p in vary_only:
            if p in sm.index:
                rows.append({
                    "parameter": p,
                    "MCMC mean": sm.loc[p,"mean"],
                    "MCMC 95% CI": f"[{sm.loc[p,'ci2.5']:.3g}, {sm.loc[p,'ci97.5']:.3g}]"
                })
    # SA (point)
    if sa:
        bp = sa.get("best_params", None)
        if bp is not None:
            for r in rows:
                p = r["parameter"]
                if p in bp.index:
                    r["SA point"] = float(bp[p])
            # if a parameter not yet in rows (edge case), add it
            for p in vary_only:
                if p not in [rr["parameter"] for rr in rows] and p in bp.index:
                    rows.append({"parameter": p, "SA point": float(bp[p])})
    # EnKF: treat final ensemble as samples
    if enkf:
        ens = enkf["chain"]
        for p in vary_only:
            if p in ens.columns:
                x = ens[p].to_numpy()
                mean = float(np.mean(x))
                ci = (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
                # find row
                hit = next((rr for rr in rows if rr["parameter"] == p), None)
                if hit is None:
                    hit = {"parameter": p}
                    rows.append(hit)
                hit["EnKF mean"] = mean
                hit["EnKF 95% CI"] = f"[{ci[0]:.3g}, {ci[1]:.3g}]"
    df = pd.DataFrame(rows).set_index("parameter")
    # order columns
    cols = ["MCMC mean","MCMC 95% CI","SA point","EnKF mean","EnKF 95% CI"]
    return df.reindex(columns=[c for c in cols if c in df.columns])
