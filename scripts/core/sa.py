# ─────────────────────────────────────────────────────────────
# scripts/core/sa.py  · simulated annealing (plume only)
# Plain single-line progress like MCMC; adaptive cooling (alpha=None)
# ─────────────────────────────────────────────────────────────
from __future__ import annotations
import sys
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

from .tephra2_utils import run_tephra2
from .mcmc import log_likelihood, log_prior, draw_plume

def _compute_alpha(T0: float, T_end: float, runs: int) -> float:
    T0 = max(float(T0), 1e-9)
    T_end = max(float(T_end), 1e-9)
    runs = max(int(runs), 1)
    # geometric: T_end = T0 * alpha^runs
    return float((T_end / T0) ** (1.0 / runs))

def simulated_annealing(
    *,
    initial: np.ndarray,
    prior_type: np.ndarray,
    prior_para: np.ndarray,
    draw_scale: np.ndarray,
    runs: int,
    obs: np.ndarray,
    sigma: float,
    conf_path: Path,
    sites_csv: Path,
    tephra2_exec: Path,
    wind_path: Path,
    T0: float = 1.0,
    alpha: Optional[float] = None,   # None -> adaptive to reach T_end
    restarts: int = 0,
    seed: Optional[int] = None,
    silent: bool = False,
    print_every: int = 200,          # progress cadence, like MCMC "snapshot"
    T_end: float = 0.2,              # only used when alpha is None
) -> Dict[str, Any]:

    rng = np.random.default_rng(seed)

    npar = len(initial)
    total = runs * (restarts + 1)

    chain = np.zeros((total + 1, npar))
    post  = np.zeros(total + 1)
    prior_arr = np.zeros(total + 1)
    like_arr  = np.zeros(total + 1)

    # initial state
    chain[0] = initial
    pred0 = run_tephra2(chain[0], conf_path, sites_csv, tephra2_exec, wind_path, silent=True)
    like0 = log_likelihood(pred0, obs, sigma)
    prior0 = log_prior(chain[0], prior_type, prior_para)
    post[0] = like0 + prior0
    prior_arr[0] = prior0
    like_arr[0]  = like0

    idx_global = 1
    best_x = chain[0].copy()
    best_p = post[0]

    # For a familiar "acc=" readout (fraction of accepted proposals)
    accepted_total = 0
    attempted_total = 0

    for r in range(restarts + 1):
        # decide cooling schedule for this restart
        a = float(alpha) if (alpha is not None) else _compute_alpha(T0, T_end, runs)
        T = float(T0)

        if not silent:
            # restart banner (fixed‑point numbers; no sci‑notation)
            sys.stdout.write(f"\n[SA] Restart {r+1}/{restarts+1}  (T0={T0:.4f}, alpha={a:.6f})\n")
            sys.stdout.flush()

        accepted_restart = 0  # optional per‑restart if you ever want it

        for it in range(1, runs + 1):
            attempted_total += 1

            prop = draw_plume(chain[idx_global-1], prior_type, draw_scale, prior_para)
            pred_p  = run_tephra2(prop, conf_path, sites_csv, tephra2_exec, wind_path, silent=True)
            like_p  = log_likelihood(pred_p, obs, sigma)
            prior_p = log_prior(prop, prior_type, prior_para)
            post_p  = like_p + prior_p

            dE = post_p - post[idx_global-1]
            if dE >= 0 or rng.random() < np.exp(dE / max(T, 1e-12)):
                chain[idx_global] = prop
                post[idx_global]  = post_p
                prior_arr[idx_global], like_arr[idx_global] = prior_p, like_p
                accepted_total += 1
                accepted_restart += 1
            else:
                chain[idx_global] = chain[idx_global-1]
                post[idx_global]  = post[idx_global-1]
                prior_arr[idx_global], like_arr[idx_global] = prior_arr[idx_global-1], like_arr[idx_global-1]

            if post[idx_global] > best_p:
                best_p = post[idx_global]
                best_x = chain[idx_global].copy()

            T *= a

            # MCMC‑style single‑line progress every print_every iters
            if not silent and (it % max(int(print_every), 1) == 0):
                acc = accepted_total / max(attempted_total, 1)
                h_val   = chain[idx_global, 0]
                lnM_val = chain[idx_global, 1]
                sys.stdout.write(
                    f"\r[SA] iter {idx_global:5d}/{total:5d}  T={T:.4f}  acc={acc:.3f}  "
                    f"h={h_val:.1f}  lnM={lnM_val:.3f}  best={best_p:.4f}"
                )
                sys.stdout.flush()

            idx_global += 1

        # newline after each restart’s inner loop if we were printing
        if not silent and (runs % max(int(print_every), 1) != 0):
            sys.stdout.write("\n")

    if not silent:
        sys.stdout.write("\n")

    return {
        "chain": chain,
        "posterior": post,
        "prior": prior_arr,
        "likelihood": like_arr,
    }
