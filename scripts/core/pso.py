# ─────────────────────────────────────────────────────────────
# scripts/core/pso.py  · particle swarm optimization (plume only)
# Plain single-line progress like MCMC/SA
# Signature, inputs, outputs, and printing cadence mirror sa.py
# ─────────────────────────────────────────────────────────────
from __future__ import annotations
import sys
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

from .tephra2_utils import run_tephra2
from .mcmc import log_likelihood, log_prior

def particle_swarm_optimization(
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
    T0: float = 1.0,                # kept for signature parity (unused)
    alpha: Optional[float] = None,  # kept for signature parity (unused)
    restarts: int = 0,
    seed: Optional[int] = None,
    silent: bool = False,
    print_every: int = 200,
    T_end: float = 0.2,             # kept for signature parity (unused)
) -> Dict[str, Any]:
    """
    PSO for Tephra2 plume parameters.
    Matches sa.py I/O: returns a 'chain' timeline (global-best per iter),
    and arrays for posterior, prior, likelihood with identical shapes.
    """

    rng = np.random.default_rng(seed)

    npar = int(len(initial))
    total = int(runs) * (int(restarts) + 1)

    # OUTPUT ARRAYS — same shapes as sa.py
    chain      = np.zeros((total + 1, npar))
    post_arr   = np.zeros(total + 1)
    prior_arr  = np.zeros(total + 1)
    like_arr   = np.zeros(total + 1)

    # ——— PSO HYPERPARAMETERS (internal defaults; no API change) ———
    # Reasonable, literature-backed defaults (Clerc/Kennedy-ish)
    # - swarm size scales with dimension; uses draw_scale to set velocity limits.
    n_particles = max(12, 6 * npar)              # swarm size
    w_inertia   = 0.72                            # inertia
    c_cog       = 1.49                            # cognitive weight
    c_soc       = 1.49                            # social weight
    # Velocity clamp taken from parameter scales, fallback to |initial| magnitude
    v_clip = np.maximum(3.0 * np.abs(draw_scale.astype(float)), 1e-3)

    # Helper: prior bounds for Uniform / Fixed to keep swarm feasible
    def _bounds_for_dim(j: int) -> Optional[tuple[float, float]]:
        ptype = str(prior_type[j])
        if ptype == "Uniform":
            a, b = float(prior_para[j, 0]), float(prior_para[j, 1])
            lo, hi = (min(a, b), max(a, b))
            return lo, hi
        if ptype == "Fixed":
            val = float(prior_para[j, 0])
            return val, val
        return None

    bounds = [ _bounds_for_dim(j) for j in range(npar) ]
    fixed_mask = np.array([ (b is not None and b[0] == b[1]) for b in bounds ], dtype=bool)

    # Evaluate a single parameter vector -> (prior, like, post)
    def _eval(x: np.ndarray) -> tuple[float, float, float]:
        pred  = run_tephra2(x, conf_path, sites_csv, tephra2_exec, wind_path, silent=True)
        like  = log_likelihood(pred, obs, sigma)
        prior = log_prior(x, prior_type, prior_para)
        return prior, like, prior + like

    # Initialize timeline with the starting point (like SA)
    chain[0] = initial
    pr0, lk0, po0 = _eval(chain[0])
    prior_arr[0], like_arr[0], post_arr[0] = pr0, lk0, po0
    gbest_x = chain[0].copy()
    gbest_p = float(po0)

    idx_global = 1

    # For a familiar "acc=" readout: fraction of iterations that improved gbest
    improved_total = 0
    attempted_total = 0

    # ——— PSO outer restarts ———
    for r in range(int(restarts) + 1):

        # Swarm init around initial using draw_scale
        X = np.tile(initial, (n_particles, 1)).astype(float)
        # Gaussian scatter scaled by draw_scale; respect fixed params
        X += rng.normal(size=X.shape) * np.maximum(draw_scale, 1e-6)
        X[:, fixed_mask] = initial[fixed_mask]

        # Enforce bounds for Uniform/Fixed at init
        for j in range(npar):
            if bounds[j] is not None:
                lo, hi = bounds[j]
                X[:, j] = np.clip(X[:, j], lo, hi)

        # Initialize velocities proportional to draw_scale
        V = rng.normal(size=X.shape) * np.maximum(draw_scale, 1e-6)
        # Clamp velocities
        V = np.clip(V, -v_clip, v_clip)
        # Ensure fixed params don't move
        V[:, fixed_mask] = 0.0

        # Evaluate swarm
        pbest_x = X.copy()
        pbest_pr = np.zeros(n_particles)
        pbest_lk = np.zeros(n_particles)
        pbest_po = np.zeros(n_particles)

        for i in range(n_particles):
            pr, lk, po = _eval(X[i])
            pbest_pr[i], pbest_lk[i], pbest_po[i] = pr, lk, po
            # Update global best vs running gbest across restarts
            if po > gbest_p:
                gbest_p = po
                gbest_x = X[i].copy()

        if not silent:
            sys.stdout.write(f"\n[PSO] Restart {r+1}/{int(restarts)+1}  (n={n_particles}, w={w_inertia:.2f})\n")
            sys.stdout.flush()

        # ——— Main PSO loop ———
        for it in range(1, int(runs) + 1):
            attempted_total += 1

            # Random weights
            r1 = rng.random(size=(n_particles, npar))
            r2 = rng.random(size=(n_particles, npar))

            # Velocity & position update
            cognitive = c_cog * r1 * (pbest_x - X)
            social    = c_soc * r2 * (gbest_x  - X)
            V = w_inertia * V + cognitive + social

            # Clamp velocities elementwise
            V = np.clip(V, -v_clip, v_clip)

            # Update positions
            X = X + V

            # Apply bounds; reflect at edges for Uniform, freeze for Fixed
            for j in range(npar):
                b = bounds[j]
                if b is None:
                    continue
                lo, hi = b
                if lo == hi:  # Fixed
                    X[:, j] = lo
                    V[:, j] = 0.0
                else:
                    # Reflective walls
                    over_lo = X[:, j] < lo
                    over_hi = X[:, j] > hi
                    if np.any(over_lo):
                        X[over_lo, j] = lo + (lo - X[over_lo, j])  # reflect
                        V[over_lo, j] = -V[over_lo, j]
                    if np.any(over_hi):
                        X[over_hi, j] = hi - (X[over_hi, j] - hi)  # reflect
                        V[over_hi, j] = -V[over_hi, j]
                    # After reflection, still clip for numeric safety
                    X[:, j] = np.clip(X[:, j], lo, hi)

            # Evaluate, update personal and global bests
            gbest_improved = False
            for i in range(n_particles):
                pr, lk, po = _eval(X[i])
                if po > pbest_po[i]:
                    pbest_po[i] = po
                    pbest_x[i]  = X[i].copy()
                    pbest_pr[i] = pr
                    pbest_lk[i] = lk
                if po > gbest_p:
                    gbest_p = po
                    gbest_x = X[i].copy()
                    gbest_improved = True

            # Record timeline as global-best (to match sa.py array lengths)
            chain[idx_global]     = gbest_x
            post_arr[idx_global]  = gbest_p
            # For prior/like arrays, compute at gbest_x explicitly
            pr_g, lk_g, _ = _eval(gbest_x)
            prior_arr[idx_global] = pr_g
            like_arr[idx_global]  = lk_g

            if gbest_improved:
                improved_total += 1

            # Single-line progress every print_every iters
            if not silent and (it % max(int(print_every), 1) == 0):
                acc = improved_total / max(attempted_total, 1)  # "acc" as improvement rate
                h_val   = chain[idx_global, 0]
                lnM_val = chain[idx_global, 1] if npar > 1 else np.nan
                sys.stdout.write(
                    f"\r[PSO] iter {idx_global:5d}/{total:5d}  w={w_inertia:.2f}  acc={acc:.3f}  "
                    f"h={h_val:.1f}  lnM={lnM_val:.3f}  best={gbest_p:.4f}"
                )
                sys.stdout.flush()

            idx_global += 1

        # newline after each restart if we were printing and didn’t end on a print boundary
        if not silent and (runs % max(int(print_every), 1) != 0):
            sys.stdout.write("\n")

    if not silent:
        sys.stdout.write("\n")

    return {
        "chain": chain,
        "posterior": post_arr,
        "prior": prior_arr,
        "likelihood": like_arr,
    }
