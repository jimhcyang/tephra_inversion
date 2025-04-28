# ─────────────────────────────────────────────────────────────
# scripts/core/mcmc.py      · wind-free · ln(mass) · tqdm · 2025-04-28
# ─────────────────────────────────────────────────────────────
"""
Metropolis-Hastings inversion for Tephra2 when ONLY plume parameters vary.

Parameter order we assume in the vector:
  [0] plume_height  (m)
  [1] log_mass      (natural log, kg)
  [2] alpha, [3] beta, … any extra plume params.

Wind is fixed (wind.txt already present).
"""
from __future__ import annotations
import logging, subprocess, os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, uniform

from .mcmc_utils import changing_variable   # edits tephra2.conf

# ─────────────────────────────────────────────────────────────
# tqdm (progress-bar) – graceful fallback if library missing
# ─────────────────────────────────────────────────────────────
try:
    from tqdm import trange
except ImportError:                       # tqdm not installed
    def trange(n, **kwargs):              # dummy iterator
        return range(n)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Proposal helpers
# ---------------------------------------------------------------------------
def propose_gaussian(current, mask, scale):
    """Random-walk Gaussian step where mask == True."""
    proposal = current.copy()
    # numeric draw_scale; empty strings / None → 0.0
    scale_f = np.array([float(s) if str(s).strip() not in ("", "None") else 0.0
                        for s in scale], dtype=float)
    proposal[mask] = np.random.normal(loc=current[mask], scale=scale_f[mask])
    return proposal


def propose_uniform(current, mask, lower, upper):
    proposal = current.copy()
    proposal[mask] = np.random.uniform(lower, upper)
    return proposal


def draw_plume(current, prior_type, draw_scale, prior_para):
    """One unified sampler for the full plume vector."""
    proposal = current.copy()
    # Gaussian params
    mask_g = prior_type == "Gaussian"
    if mask_g.any():
        proposal = propose_gaussian(proposal, mask_g, draw_scale)
    # Uniform params
    mask_u = prior_type == "Uniform"
    if mask_u.any():
        lower, upper = prior_para[mask_u, 0], prior_para[mask_u, 1]
        proposal = propose_uniform(proposal, mask_u, lower, upper)
    return proposal

# ---------------------------------------------------------------------------
#  Priors & likelihood
# ---------------------------------------------------------------------------
def log_prior(plume, prior_type, prior_para) -> float:
    """Return Σ log10(prior_pdf)."""
    logp = np.zeros_like(plume, dtype=float)

    mask_g = prior_type == "Gaussian"
    if mask_g.any():
        mu, sigma = prior_para[mask_g, 0], prior_para[mask_g, 1]
        logp[mask_g] = np.log10(np.clip(norm.pdf(plume[mask_g], mu, sigma), 1e-300, None))

    mask_u = prior_type == "Uniform"
    if mask_u.any():
        lo, hi = prior_para[mask_u, 0], prior_para[mask_u, 1]
        logp[mask_u] = np.log10(np.clip(uniform.pdf(plume[mask_u], lo, hi-lo), 1e-300, None))

    return float(logp.sum())


def log_likelihood(pred, obs, sigma) -> float:
    """Gaussian likelihood in log10 space."""
    pred = np.clip(pred, 1e-3, None)
    obs  = np.clip(obs,  1e-3, None)
    log_ratio = np.log10(obs / pred)
    return float(np.log10(np.clip(norm.pdf(log_ratio, 0, sigma), 1e-300, None)).sum())

# ---------------------------------------------------------------------------
#  Tephra2 runner (auto-fix sites delimiter)
# ---------------------------------------------------------------------------
def _ensure_sites_format(sites_csv: Path):
    """Re-write sites file as space-delimited E N Z."""
    df = pd.read_csv(sites_csv, sep=r"[,\s]+", engine="python", header=None)
    if df.shape[1] != 3:
        raise ValueError(f"{sites_csv} must have 3 columns (E,N,Z); got {df.shape[1]}")
    df.to_csv(sites_csv, sep=" ", header=False, index=False, float_format="%.3f")


def run_tephra2(plume_vec,
                conf_path: Path,
                sites_csv: Path,
                silent=True) -> np.ndarray:
    """
    Edit tephra2.conf, ensure sites file OK, run Tephra2 executable located at
    <repo_root>/Tephra2/tephra2_2020, return deposit column (kg m⁻²).
    """
    changing_variable(plume_vec, conf_path)
    _ensure_sites_format(sites_csv)

    exe = Path(__file__).resolve().parents[2] / "Tephra2" / "tephra2_2020"
    out_file = conf_path.parent / "tephra2_output_mcmc.txt"

    cmd = [str(exe), str(conf_path), str(sites_csv), str(conf_path.parent / "wind.txt")]
    res = subprocess.run(cmd, stdout=open(out_file, "w"),
                         stderr=subprocess.PIPE, text=True)

    if res.returncode != 0 or out_file.stat().st_size == 0:
        raise RuntimeError(f"Tephra2 failed (exit {res.returncode}).\n--- STDERR ---\n{res.stderr}")

    data = np.genfromtxt(out_file)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return data[:, 3]     # mass-loading column

# ---------------------------------------------------------------------------
#  Main MH driver
# ---------------------------------------------------------------------------
def metropolis_hastings(
        initial_plume, prior_type, prior_para, draw_scale,
        runs, obs_load, likelihood_sigma,
        conf_path, sites_csv,
        burnin=0, silent=True, snapshot=100) -> dict:
    """
    Returns dict with keys:
        chain, posterior, prior, likelihood, accept_rate, burnin
    """
    npar   = len(initial_plume)
    chain  = np.zeros((runs + 1, npar))
    post   = np.zeros(runs + 1)
    prior_arr = np.zeros(runs + 1)
    like_arr  = np.zeros(runs + 1)

    # ── initial state ─────────────────────────────────────────
    chain[0] = initial_plume
    pred0    = run_tephra2(chain[0], conf_path, sites_csv, silent)
    like0    = log_likelihood(pred0, obs_load, likelihood_sigma)
    prior0   = log_prior(chain[0], prior_type, prior_para)
    post[0]  = like0 + prior0
    like_arr[0], prior_arr[0] = like0, prior0

    accept = 0
    bar = trange(1, runs + 1, disable=silent, desc="MCMC")

    # ── MH loop ───────────────────────────────────────────────
    for i in bar:
        proposal = draw_plume(chain[i-1], prior_type, draw_scale, prior_para)

        pred_p  = run_tephra2(proposal, conf_path, sites_csv, silent)
        like_p  = log_likelihood(pred_p, obs_load, likelihood_sigma)
        prior_p = log_prior(proposal, prior_type, prior_para)
        post_p  = like_p + prior_p

        if np.random.rand() < np.exp((post_p - post[i-1]) * np.log(10)):     # log10→ln
            chain[i] = proposal
            post[i]  = post_p
            prior_arr[i], like_arr[i] = prior_p, like_p
            accept += 1
        else:
            chain[i] = chain[i-1]
            post[i]  = post[i-1]
            prior_arr[i], like_arr[i] = prior_arr[i-1], like_arr[i-1]

        if not silent and i % snapshot == 0:
            bar.set_postfix(acc=accept/i, plume=proposal[0], lnM=proposal[1])

    return {
        "chain": chain,
        "posterior": post,
        "prior": prior_arr,
        "likelihood": like_arr,
        "accept_rate": accept / runs,
        "burnin": burnin
    }

# ---------------------------------------------------------------------------
#  Deprecated wrappers for legacy imports
# ---------------------------------------------------------------------------
def prior_function(prior_type, plume_vec, prior_para):
    return log_prior(plume_vec, prior_type, prior_para)

def likelihood_function(prediction, observation, sigma):
    return log_likelihood(prediction, observation, sigma)
