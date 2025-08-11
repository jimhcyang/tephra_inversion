# ─────────────────────────────────────────────────────────────
# scripts/core/enkf.py  · ES‑MDA in log‑space with robust numerics
# unified plain-text progress (MCMC-style)
# ─────────────────────────────────────────────────────────────
from __future__ import annotations
import sys
import warnings
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

from .tephra2_utils import run_tephra2

# Broad physical safety rails
_PLUME_MIN = 100.0
_PLUME_MAX = 5.0e4
_LOGM_MIN  = np.log(1e6)
_LOGM_MAX  = np.log(1e14)


# ----------------------------- helpers ----------------------------- #
def _state_bounds(prior_type: np.ndarray, prior_para: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = len(prior_type)
    lo = np.full(p, -np.inf, dtype=float)
    hi = np.full(p,  np.inf, dtype=float)
    for j in range(p):
        if prior_type[j] == "Gaussian":
            mu, sd = prior_para[j, 0], max(prior_para[j, 1], 1e-6)
            lo[j], hi[j] = mu - 5.0 * sd, mu + 5.0 * sd
        elif prior_type[j] == "Uniform":
            lo[j], hi[j] = prior_para[j, 0], prior_para[j, 1]
        else:  # Fixed -> point mass around given value
            lo[j] = hi[j] = prior_para[j, 0]

    # enforce broad physical bounds for first two dims if present
    if p >= 1:
        lo[0] = max(lo[0], _PLUME_MIN)
        hi[0] = min(hi[0], _PLUME_MAX)
    if p >= 2:
        lo[1] = max(lo[1], _LOGM_MIN)
        hi[1] = min(hi[1], _LOGM_MAX)
    return lo, hi


def _apply_bounds(X: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    Xc = np.clip(X, lo, hi)
    # replace any remaining non-finites with midpoint (if both finite) or zero
    mid = np.where(np.isfinite(lo) & np.isfinite(hi), 0.5 * (lo + hi), 0.0)
    bad = ~np.isfinite(Xc)
    if np.any(bad):
        # broadcast mid across columns for selection
        col_idx = np.where(bad)[1]
        Xc[bad] = mid[col_idx]
    return Xc


def _winsorize(A: np.ndarray, k_sigma: float = 6.0) -> np.ndarray:
    s = A.std(axis=0, ddof=1)
    s = np.where(s <= 0, 1.0, s)
    lim = k_sigma * s
    return np.clip(A, -lim, lim)


def _forward_batch(
    params: np.ndarray,
    conf_path: Path, sites_csv: Path, tephra2_exec: Path, wind_path: Path,
    obs_dim: int, eps_log: float,
    on_progress=None
) -> np.ndarray:
    """
    Evaluate Tephra2 for each ensemble member; return log-space predictions.
    If on_progress is provided, it is called with the 1-based member index.
    """
    out = np.full((params.shape[0], obs_dim), np.nan, dtype=float)
    for i, x in enumerate(params, start=1):
        try:
            y = run_tephra2(x, conf_path, sites_csv, tephra2_exec, wind_path, silent=True)
            y = np.asarray(y, dtype=float).ravel()
            if y.size == obs_dim:
                y = np.log(np.maximum(y, eps_log))
                out[i - 1, :] = y
        except Exception:
            # leave as NaN; robust handling downstream
            pass
        if on_progress is not None:
            on_progress(i)
    return out


# ----------------------------- main ES‑MDA ----------------------------- #
def ensemble_smoother_mda(
    *,
    prior_type: np.ndarray,
    prior_para: np.ndarray,
    obs: np.ndarray,
    sigma: float,
    conf_path: Path,
    sites_csv: Path,
    tephra2_exec: Path,
    wind_path: Path,
    n_ens: int = 60,
    n_assimilations: int = 4,
    inflation: float = 1.02,
    seed: Optional[int] = 42,
    silent: bool = False,
    print_every: int = 1,
    # exploration / robustness
    obs_logspace: bool = True,
    sd_scale: float = 1.0,
    jitter_after_pass: float = 0.0,
    step_trust: float = 3.0,
    winsor_k: float = 6.0,
    eps_log: float = 1e-12,
    ridge_cyy: float = 1e-6,
    suppress_runtime_warnings: bool = True,
    # progress cadence for inner forward loop
    member_update_every: int = 0,   # 0 = disable "iter i/N" lines
) -> Dict[str, Any]:
    """
    ES‑MDA (Emerick & Reynolds style) with robust numerics in log-space.

    Returns:
        {
          "ensemble": X_final,                # (Nens x p)
          "ensemble_history": [X^1, ...],     # list length ≤ n_assimilations
        }
    """
    rng = np.random.default_rng(seed)

    # ---------- initial ensemble from priors ----------
    p = len(prior_type)
    lo_b, hi_b = _state_bounds(prior_type, prior_para)

    X = np.zeros((n_ens, p), dtype=float)
    prior_sd = np.zeros(p, dtype=float)

    for j in range(p):
        if prior_type[j] == "Gaussian":
            mu, sd = prior_para[j, 0], max(prior_para[j, 1], 1e-6)
            prior_sd[j] = sd * max(sd_scale, 1.0)
            X[:, j] = rng.normal(mu, prior_sd[j], size=n_ens)
        elif prior_type[j] == "Uniform":
            lo, hi = prior_para[j, 0], prior_para[j, 1]
            # fall back to bounded support if ±inf
            if not np.isfinite(lo): lo = lo_b[j]
            if not np.isfinite(hi): hi = hi_b[j]
            prior_sd[j] = (hi - lo) / np.sqrt(12.0) if np.isfinite(hi - lo) else 1.0
            X[:, j] = rng.uniform(lo, hi, size=n_ens)
        else:  # Fixed
            X[:, j] = prior_para[j, 0]
            prior_sd[j] = max(abs(prior_para[j, 0]) * 0.05, 1.0)

    X = _apply_bounds(X, lo_b, hi_b)

    # ---------- observations (optionally in log-space) ----------
    y_obs = np.asarray(obs, dtype=float).ravel()
    if obs_logspace:
        y_obs = np.log(np.maximum(y_obs, eps_log))
    m = y_obs.size

    # ES‑MDA: split noise across passes
    R_scalar = float(sigma) ** 2 * float(n_assimilations)
    Rk = np.eye(m) * R_scalar

    ensemble_history: List[np.ndarray] = []

    # ---------- optional warning silencer ----------
    warn_ctx = warnings.catch_warnings()
    if suppress_runtime_warnings:
        warn_ctx.__enter__()
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message=".*encountered in matmul.*")
        warnings.filterwarnings("ignore", message=".*divide by zero encountered.*")
        warnings.filterwarnings("ignore", module="numpy.linalg")

    try:
        for k in range(1, n_assimilations + 1):
            # --- inner per-member progress (optional) ---
            prog_used = (not silent) and (int(member_update_every) > 0)
            n_total = X.shape[0]

            def _progress(i: int):
                if prog_used and (i % member_update_every == 0 or i == n_total):
                    sys.stdout.write(f"\r[ENKF] pass {k}/{n_assimilations}  iter {i}/{n_total}")
                    sys.stdout.flush()

            # forward model in log space
            with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
                Y = _forward_batch(
                    X, conf_path, sites_csv, tephra2_exec, wind_path,
                    obs_dim=m, eps_log=eps_log, on_progress=_progress
                )

            if prog_used:
                sys.stdout.write("\n")  # finish iter line before summary

            # keep only valid rows
            valid = np.isfinite(Y).all(axis=1) & np.isfinite(X).all(axis=1)
            if valid.sum() < max(3, p + 1):
                if not silent:
                    sys.stdout.write(
                        f"[ENKF] pass {k}/{n_assimilations}  stopped early (valid={int(valid.sum())})\n"
                    )
                break

            Xv, Yv = X[valid], Y[valid]

            # anomalies (winsorized to tame outliers)
            Xm = Xv.mean(axis=0, keepdims=True)
            Ym = Yv.mean(axis=0, keepdims=True)
            Xc = _winsorize(Xv - Xm, k_sigma=winsor_k)
            Yc = _winsorize(Yv - Ym, k_sigma=winsor_k)

            # mild inflation on state anomalies
            if inflation and inflation > 1.0:
                Xc *= float(inflation)

            denom = max(Xv.shape[0] - 1, 1)

            with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
                Cxy = (Xc.T @ Yc) / denom
                Cyy = (Yc.T @ Yc) / denom
                Cyy_reg = Cyy + np.eye(m) * float(ridge_cyy) + Rk
                # use pseudo-inverse; more stable than direct inv for noisy Cyy
                K = Cxy @ np.linalg.pinv(Cyy_reg)

            # draw observation jitter ~ N(0, Rk)
            try:
                L = np.linalg.cholesky(Rk)
                E = rng.normal(size=(Xv.shape[0], m)) @ L.T
            except np.linalg.LinAlgError:
                E = rng.normal(scale=np.sqrt(R_scalar), size=(Xv.shape[0], m))

            innov = (y_obs + E) - Yv

            with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
                delta = innov @ K.T  # (Nvalid x p)

            # trust region cap per-dimension
            step_cap = np.maximum(step_trust * np.maximum(prior_sd, 1e-6), 1e-6)
            delta = np.clip(delta, -step_cap, step_cap)

            Xv_new = Xv + delta

            # small jitter after pass to maintain diversity if desired
            if jitter_after_pass and jitter_after_pass > 0:
                Xv_new += rng.normal(scale=jitter_after_pass, size=Xv_new.shape)

            # write back + clamp
            X[valid] = _apply_bounds(Xv_new, lo_b, hi_b)

            # end-of-pass summary (plain ASCII, no sci-notation)
            if not silent and (k % max(int(print_every), 1) == 0):
                h_mean = float(np.nanmean(X[valid, 0])) if p >= 1 else float("nan")
                lnM_mean = float(np.nanmean(X[valid, 1])) if p >= 2 else float("nan")
                sys.stdout.write(
                    f"[ENKF] pass {k}/{n_assimilations}  valid={int(valid.sum())}  "
                    f"h_mean={h_mean:,.1f}  lnM_mean={lnM_mean:.3f}\n"
                )

            ensemble_history.append(X.copy())

    finally:
        if suppress_runtime_warnings:
            warn_ctx.__exit__(None, None, None)

    return {"ensemble": X, "ensemble_history": ensemble_history}
