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
import logging
from pathlib import Path

import numpy as np
from scipy.stats import norm, uniform

from .tephra2_utils import run_tephra2

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
    """Uniform proposal distribution where mask == True."""
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
    """Return Σ ln(prior_pdf)."""
    logp = np.zeros_like(plume, dtype=float)

    mask_g = prior_type == "Gaussian"
    if mask_g.any():
        mu, sigma = prior_para[mask_g, 0], prior_para[mask_g, 1]
        logp[mask_g] = np.log(np.clip(norm.pdf(plume[mask_g], mu, sigma), 1e-300, None))

    mask_u = prior_type == "Uniform"
    if mask_u.any():
        lo, hi = prior_para[mask_u, 0], prior_para[mask_u, 1]
        logp[mask_u] = np.log(np.clip(uniform.pdf(plume[mask_u], lo, hi-lo), 1e-300, None))

    return float(logp.sum())


def log_likelihood(pred, obs, sigma) -> float:
    """Gaussian likelihood in natural log space."""
    pred = np.clip(pred, 1e-3, None)
    obs  = np.clip(obs,  1e-3, None)
    log_ratio = np.log(obs / pred)
    return float(np.log(np.clip(norm.pdf(log_ratio, 0, sigma), 1e-300, None)).sum())

# ---------------------------------------------------------------------------
#  Main MH driver
# ---------------------------------------------------------------------------
def metropolis_hastings(
        initial_plume, prior_type, prior_para, draw_scale,
        runs, obs_load, likelihood_sigma,
        conf_path, sites_csv,
        tephra2_path=None, wind_path=None,
        burnin=0, silent=True, snapshot=100) -> dict:
    """
    Run Metropolis-Hastings MCMC algorithm for tephra inversion.
    
    Parameters
    ----------
    initial_plume : np.ndarray
        Initial parameter vector (plume height, log mass, etc.)
    prior_type : np.ndarray
        Array of prior types ("Gaussian", "Uniform", "Fixed")
    prior_para : np.ndarray
        Array of prior parameters (mean/std for Gaussian, min/max for Uniform)
    draw_scale : np.ndarray
        Array of proposal distribution scale factors
    runs : int
        Number of MCMC iterations to run
    obs_load : np.ndarray
        Observed tephra deposit thicknesses
    likelihood_sigma : float
        Standard deviation for likelihood function
    conf_path : Path or str
        Path to tephra2 configuration file
    sites_csv : Path or str
        Path to sites file
    tephra2_path : Path or str, optional
        Path to tephra2 executable
    wind_path : Path or str, optional
        Path to wind file
    burnin : int, default=0
        Number of samples to discard as burn-in
    silent : bool, default=True
        Whether to suppress output
    snapshot : int, default=100
        Frequency to print status updates
        
    Returns
    -------
    dict
        Dictionary with keys:
        chain, posterior, prior, likelihood, accept_rate, burnin
    """
    npar   = len(initial_plume)
    chain  = np.zeros((runs + 1, npar))
    post   = np.zeros(runs + 1)
    prior_arr = np.zeros(runs + 1)
    like_arr  = np.zeros(runs + 1)

    # ── initial state ─────────────────────────────────────────
    chain[0] = initial_plume
    pred0    = run_tephra2(chain[0], conf_path, sites_csv, tephra2_path, wind_path, silent=silent)
    like0    = log_likelihood(pred0, obs_load, likelihood_sigma)
    prior0   = log_prior(chain[0], prior_type, prior_para)
    post[0]  = like0 + prior0
    like_arr[0], prior_arr[0] = like0, prior0

    accept = 0
    bar = trange(1, runs + 1, disable=silent, desc="MCMC")

    # ── MH loop ───────────────────────────────────────────────
    for i in bar:
        proposal = draw_plume(chain[i-1], prior_type, draw_scale, prior_para)

        pred_p  = run_tephra2(proposal, conf_path, sites_csv, tephra2_path, wind_path, silent=silent)
        like_p  = log_likelihood(pred_p, obs_load, likelihood_sigma)
        prior_p = log_prior(proposal, prior_type, prior_para)
        post_p  = like_p + prior_p

        if np.random.rand() < np.exp(post_p - post[i-1]):     # Using natural log directly
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