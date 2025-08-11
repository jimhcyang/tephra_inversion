from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

# IMPORTANT: we reuse your existing Tephra2 runner
from .tephra2_utils import run_tephra2 as _run_tephra2

def log_prior(x: np.ndarray, prior_type: np.ndarray, prior_para: np.ndarray) -> float:
    lp = 0.0
    for i,t in enumerate(prior_type):
        a,b = prior_para[i]
        if t == "Gaussian":
            mu, sd = a, b
            if sd <= 0: return -np.inf
            lp += -0.5 * ((x[i]-mu)/sd)**2 - np.log(sd) - 0.5*np.log(2*np.pi)
        elif t == "Uniform":
            lo, hi = a, b
            if not (lo <= x[i] <= hi): return -np.inf
            lp += -np.log(max(hi - lo, 1e-300))
        elif t == "Fixed":
            if abs(x[i] - a) > 1e-12:
                return -np.inf
        else:
            return -np.inf
    return float(lp)

# Likelihood in *log-space* to match your MCMC convention
# (works well for heavy-tailed masses)
def log_likelihood(obs: np.ndarray, pred: np.ndarray, sigma: float) -> float:
    pred = np.clip(pred, 1e-3, None)
    obs  = np.clip(obs,  1e-3, None)
    log_ratio = np.log(obs / pred)
    return float(-0.5*np.sum((log_ratio/sigma)**2) - log_ratio.size*np.log(sigma) - 0.5*log_ratio.size*np.log(2*np.pi))

def log_posterior(x: np.ndarray,
                  prior_type: np.ndarray,
                  prior_para: np.ndarray,
                  obs: np.ndarray,
                  sigma: float,
                  conf_path: Path,
                  sites_csv: Path,
                  tephra2_exec: str,
                  wind_path: str) -> tuple[float, dict]:
    lp = log_prior(x, prior_type, prior_para)
    if not np.isfinite(lp):
        return -np.inf, {"log_prior": -np.inf, "log_lik": -np.inf}
    pred = _run_tephra2(x, conf_path, sites_csv, tephra2_exec, wind_path, silent=True)
    ll   = log_likelihood(obs, pred, sigma)
    return lp + ll, {"log_prior": lp, "log_lik": ll, "pred": pred}