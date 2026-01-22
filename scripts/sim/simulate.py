#!/usr/bin/env python3
# scripts/sim/simulate.py
from __future__ import annotations

import argparse
import copy
import importlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.data_handling.config_io import load_config
from scripts.tephra_inversion import TephraInversion

# =============================================================================
# PARAMETER / PRIOR POLICY (EDIT HERE)
# =============================================================================
#
# This simulation runner constructs the per-run parameter priors used by
# TephraInversion. It supports:
#   - 2-parameter inversion: plume_height, log_mass
#   - 4-parameter inversion: plume_height, log_mass, median_grain, std_grain
#
# Key idea:
#   - prior_std controls how "costly" it is to leave the prior mean
#   - draw_scale controls typical step sizes (proposal scale / perturbation scale)
#
# IMPORTANT:
#   - log_mass is NATURAL LOG of mass in kg, i.e., ln(kg).
#   - median_grain / std_grain are PHI units (Tephra2 TGSD convention)
#
# -----------------------------------------------------------------------------
# PLUME HEIGHT (meters)
# -----------------------------------------------------------------------------
PLUME_TRUE_WITHIN_NSIGMA: float = 2.0
# Base fractional uncertainty (relative), used when we don't want a super-wide prior
PLUME_PRIOR_STD_FRAC: float = 0.20     # e.g., 20% of a typical scale
PLUME_PRIOR_STD_MIN_M: float = 500.0   # never narrower than this
PLUME_PRIOR_STD_MAX_M: float = 5000.0  # never wider than this
# Draw scale as fraction of std
PLUME_DRAW_FRAC_OF_STD: float = 0.25
PLUME_DRAW_MIN_M: float = 125.0
PLUME_DRAW_MAX_M: float = 1250.0

# -----------------------------------------------------------------------------
# LOG MASS (natural log kg)
# -----------------------------------------------------------------------------
LOGM_TRUE_WITHIN_NSIGMA: float = 2.0
LOGM_PRIOR_STD_DEFAULT: float = 1.0
LOGM_PRIOR_STD_MIN: float = 0.2
LOGM_PRIOR_STD_MAX: float = 2.0
LOGM_DRAW_FRAC_OF_STD: float = 0.5
LOGM_DRAW_MIN: float = 0.1
LOGM_DRAW_MAX: float = 1.0

# -----------------------------------------------------------------------------
# GRAIN SIZE (PHI)
# -----------------------------------------------------------------------------
# Used only if invert_n_params == 4.
GRAIN_MED_MIN: float = -6.0
GRAIN_MED_MAX: float = 6.0
GRAIN_STD_MIN: float = 0.1
GRAIN_STD_MAX: float = 3.0

# Prior widths (phi)
GRAIN_MED_PRIOR_STD: float = 1.0
GRAIN_STD_PRIOR_STD: float = 0.5

# Proposal / draw scales (phi)
GRAIN_MED_DRAW_FRAC_OF_STD: float = 0.5
GRAIN_STD_DRAW_FRAC_OF_STD: float = 0.5

# Parameter names expected by TephraInversion / forward wrapper.
GRAIN_MED_PARAM_NAME: str = "median_grain"
GRAIN_STD_PARAM_NAME: str = "std_grain"


# =============================================================================
# Helpers
# =============================================================================

def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), float(lo)), float(hi)))


def _find_wind_file(input_dir: Path) -> Optional[Path]:
    input_dir = Path(input_dir)

    preferred = [
        input_dir / "wind.txt",
        input_dir / "wind.dat",
        input_dir / "wind_profile.txt",
        input_dir / "wind_profile.dat",
    ]
    for p in preferred:
        if p.exists() and p.is_file():
            return p

    for pat in ("wind*.txt", "wind*.dat", "*wind*.txt", "*wind*.dat"):
        hits = sorted(input_dir.glob(pat))
        for h in hits:
            if h.is_file():
                return h

    return None


def _maybe_plot_winds(input_dir: Path, out_dir: Path) -> None:
    """Best-effort wind sanity plot."""
    try:
        from scripts.visualization.wind_plots import plot_wind_file  # type: ignore

        wind_path = _find_wind_file(input_dir)
        if wind_path is None:
            print(f"[WARN] --plot-winds set, but no wind file found in: {input_dir}")
            return

        plot_dir = out_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        save_path = plot_wind_file(
            wind_path=wind_path,
            output_dir=plot_dir,
            title="Wind profile used by simulation (speed & direction vs altitude)",
            save_name=f"wind_profile_{wind_path.stem}.png",
            show_plot=False,
        )
        print(f"[OK] wind plot saved: {save_path}")

    except Exception as e:
        print(f"[WARN] Failed to plot winds: {e}")


def _load_sim_meta(input_dir: Path) -> dict:
    """Reads <scenario_root>/config/sim_meta.json where scenario_root = input_dir.parent."""
    input_dir = Path(input_dir)
    meta_path = input_dir.parent / "config" / "sim_meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return {}


def _compute_plume_std_draw(center: float, true_h: Optional[float]) -> Tuple[float, float]:
    """Compute prior_std and draw_scale for plume height."""
    scale = abs(center) if abs(center) > 1e-9 else 1.0
    base_std = PLUME_PRIOR_STD_FRAC * scale
    std = max(base_std, PLUME_PRIOR_STD_MIN_M)

    if true_h is not None and np.isfinite(true_h):
        cover_std = abs(center - float(true_h)) / max(PLUME_TRUE_WITHIN_NSIGMA, 1e-6)
        std = max(std, cover_std)

    std = _clip(std, PLUME_PRIOR_STD_MIN_M, PLUME_PRIOR_STD_MAX_M)

    draw = std * PLUME_DRAW_FRAC_OF_STD
    draw = _clip(draw, PLUME_DRAW_MIN_M, PLUME_DRAW_MAX_M)
    return float(std), float(draw)


def _compute_logm_std_draw(center: float, true_lnM: Optional[float], base_std: float) -> Tuple[float, float]:
    """Compute prior_std and draw_scale for ln(mass)."""
    std = float(base_std)
    std = _clip(std, LOGM_PRIOR_STD_MIN, LOGM_PRIOR_STD_MAX)

    if true_lnM is not None and np.isfinite(true_lnM):
        cover_std = abs(center - float(true_lnM)) / max(LOGM_TRUE_WITHIN_NSIGMA, 1e-6)
        std = max(std, cover_std)
        std = _clip(std, LOGM_PRIOR_STD_MIN, LOGM_PRIOR_STD_MAX)

    draw = std * LOGM_DRAW_FRAC_OF_STD
    draw = _clip(draw, LOGM_DRAW_MIN, LOGM_DRAW_MAX)
    return float(std), float(draw)


def _safe_float(d: dict, key: str, default: float) -> float:
    try:
        return float(d.get(key, default))
    except Exception:
        return float(default)


def _build_param_config(
    base_cfg: Dict[str, Any],
    input_dir: Path,
    scale_height: float,
    scale_mass: float,
    logm_prior_std: float,
    invert_n_params: int,
) -> Dict[str, Dict[str, Any]]:
    """Build a TephraInversion-style parameter config for simulation runs."""

    params_cfg = base_cfg.get("parameters", {})
    var = params_cfg.get("variable", {})
    true_vals = params_cfg.get("true_values", {})
    meta = _load_sim_meta(input_dir)

    # -- True plume height (m)
    true_h = float(meta.get("plume_height", true_vals.get("plume_height", 0.0)))
    if not np.isfinite(true_h) or true_h <= 0:
        col = var.get("column_height", {})
        true_h = _safe_float(col, "initial_val", 0.0)

    h_center = float(scale_height) * float(true_h)
    h_std, h_draw = _compute_plume_std_draw(center=h_center, true_h=true_h)

    # -- True mass in ln(kg)
    if "ln_mass" in meta:
        true_lnM = float(meta["ln_mass"])
    elif "eruption_mass_kg" in meta:
        true_lnM = float(np.log(float(meta["eruption_mass_kg"])))
    else:
        true_lnM = float(true_vals.get("log_mass", np.nan))

    if not np.isfinite(true_lnM):
        em = var.get("eruption_mass", {})
        true_lnM = _safe_float(em, "initial_val", np.nan)

    if float(scale_mass) <= 0:
        raise ValueError(f"scale_mass must be positive, got {scale_mass}")
    center_lnM = float(true_lnM) + float(np.log(float(scale_mass)))

    m_std, m_draw = _compute_logm_std_draw(center=center_lnM, true_lnM=true_lnM, base_std=logm_prior_std)

    params: Dict[str, Dict[str, Any]] = {
        "plume_height": {
            "initial_value": h_center,
            "prior_type": "Gaussian",
            "prior_mean": h_center,
            "prior_std": h_std,
            "draw_scale": h_draw,
        },
        "log_mass": {
            "initial_value": center_lnM,
            "prior_type": "Gaussian",
            "prior_mean": center_lnM,
            "prior_std": m_std,
            "draw_scale": m_draw,
        },
    }

    # Optional grain parameters
    if int(invert_n_params) == 4:
        true_med = float(meta.get(GRAIN_MED_PARAM_NAME, true_vals.get(GRAIN_MED_PARAM_NAME, 1.0)))
        true_std = float(meta.get(GRAIN_STD_PARAM_NAME, true_vals.get(GRAIN_STD_PARAM_NAME, 1.0)))

        true_med = _clip(true_med, GRAIN_MED_MIN, GRAIN_MED_MAX)
        true_std = _clip(true_std, GRAIN_STD_MIN, GRAIN_STD_MAX)

        med_center = true_med
        std_center = true_std

        med_std = float(GRAIN_MED_PRIOR_STD)
        std_std = float(GRAIN_STD_PRIOR_STD)

        med_draw = _clip(med_std * GRAIN_MED_DRAW_FRAC_OF_STD, 0.05, 2.0)
        std_draw = _clip(std_std * GRAIN_STD_DRAW_FRAC_OF_STD, 0.02, 1.0)

        params[GRAIN_MED_PARAM_NAME] = {
            "initial_value": med_center,
            "prior_type": "Gaussian",
            "prior_mean": med_center,
            "prior_std": med_std,
            "draw_scale": med_draw,
        }
        params[GRAIN_STD_PARAM_NAME] = {
            "initial_value": std_center,
            "prior_type": "Gaussian",
            "prior_mean": std_center,
            "prior_std": std_std,
            "draw_scale": std_draw,
        }

    return params


def _iter_method_configs(model: str, EXP):
    if model == "sa":
        for runs in EXP.SA_RUNS:
            for restarts in EXP.SA_RESTARTS:
                yield {"runs": int(runs), "restarts": int(restarts), "print_every": max(int(runs) // 10, 1)}

    elif model == "pso":
        for runs in EXP.PSO_RUNS:
            for restarts in EXP.PSO_RESTARTS:
                yield {"runs": int(runs), "restarts": int(restarts), "print_every": max(int(runs) // 10, 1)}

    elif model == "es":
        for n_ens in EXP.ES_N_ENS:
            for n_assim in EXP.ES_N_ASSIM:
                yield {"n_ens": int(n_ens), "n_assimilations": int(n_assim), "print_every": int(EXP.ES_PRINT_EVERY)}

    elif model == "mcmc":
        for n_iter in EXP.MCMC_N_ITER:
            yield {"n_iterations": int(n_iter), "snapshot": max(int(n_iter) // 10, 1)}

    else:
        raise ValueError(f"Unknown model: {model}")


def _summarize_run(
    model: str,
    run_id: int,
    simulation_id: int,
    seed: int,
    scale_h: float,
    scale_m: float,
    param_cfg: Dict[str, Dict[str, Any]],
    method_cfg: Dict[str, Any],
    duration: float,
    results: Dict[str, Any],
) -> Dict[str, Any]:
    best = results["best_params"]

    def _get_best(name: str) -> float:
        if isinstance(best, pd.Series):
            return float(best.get(name, np.nan))
        return float(best.get(name, np.nan))

    plume_est = _get_best("plume_height")
    lnM_est = _get_best("log_mass")

    plume_prior_mean = float(param_cfg["plume_height"]["prior_mean"])
    lnM_prior_mean = float(param_cfg["log_mass"]["prior_mean"])

    row: Dict[str, Any] = {
        "run_id": run_id,
        "simulation_id": simulation_id,
        "model": model,
        "seed": seed,
        "scale_plume_factor": float(scale_h),
        "scale_mass_factor": float(scale_m),
        "plume_prior_mean": plume_prior_mean,
        "lnM_prior_mean": lnM_prior_mean,
        "plume_estimate": plume_est,
        "lnM_estimate": lnM_est,
        "best_posterior": float(results.get("best_posterior", np.nan)),
        "runtime_sec": float(duration),
    }

    if GRAIN_MED_PARAM_NAME in param_cfg:
        row[f"{GRAIN_MED_PARAM_NAME}_prior_mean"] = float(param_cfg[GRAIN_MED_PARAM_NAME]["prior_mean"])
        row[f"{GRAIN_MED_PARAM_NAME}_estimate"] = _get_best(GRAIN_MED_PARAM_NAME)

    if GRAIN_STD_PARAM_NAME in param_cfg:
        row[f"{GRAIN_STD_PARAM_NAME}_prior_mean"] = float(param_cfg[GRAIN_STD_PARAM_NAME]["prior_mean"])
        row[f"{GRAIN_STD_PARAM_NAME}_estimate"] = _get_best(GRAIN_STD_PARAM_NAME)

    for key in (
        "runs", "restarts", "T0", "alpha", "T_end",
        "n_iterations", "n_burnin",
        "n_ens", "n_assimilations",
        "likelihood_sigma", "print_every", "snapshot",
    ):
        if key in method_cfg:
            row[key] = method_cfg[key]

    if model == "mcmc":
        row["acceptance_rate"] = float(results.get("acceptance_rate", np.nan))

    return row


def run_all_experiments(
    EXP,
    input_dir: Path,
    out_dir: Path,
    plot_winds: bool = False,
    invert_n_params: int = 2,
    logm_prior_std: Optional[float] = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chains_root = out_dir / "chains"
    chains_root.mkdir(parents=True, exist_ok=True)

    if plot_winds:
        _maybe_plot_winds(Path(input_dir), out_dir)

    base_cfg = load_config()

    if logm_prior_std is None:
        logm_prior_std = float(getattr(EXP, "LOGM_PRIOR_STD", LOGM_PRIOR_STD_DEFAULT))

    invert_n_params = int(invert_n_params)
    if invert_n_params not in (2, 4):
        raise ValueError(f"invert_n_params must be 2 or 4, got {invert_n_params}")

    rows_by_model: Dict[str, list[Dict[str, Any]]] = {m: [] for m in EXP.MODELS}
    run_counter = 0

    print(f"[SIM] input_dir       = {Path(input_dir).resolve()}")
    print(f"[SIM] output_dir      = {Path(out_dir).resolve()}")
    print(f"[SIM] invert_n_params = {invert_n_params}")
    print(f"[SIM] logm_prior_std  = {float(logm_prior_std):.4f} (ln-space)")

    for sim_id in range(EXP.N_REPEATS):
        for scale_h in EXP.PRIOR_FACTORS:
            for scale_m in EXP.PRIOR_FACTORS:
                param_cfg = _build_param_config(
                    base_cfg=base_cfg,
                    input_dir=Path(input_dir),
                    scale_height=scale_h,
                    scale_mass=scale_m,
                    logm_prior_std=float(logm_prior_std),
                    invert_n_params=invert_n_params,
                )

                for model in EXP.MODELS:
                    for method_cfg_override in _iter_method_configs(model, EXP):
                        run_counter += 1
                        run_id = run_counter

                        seed = int(EXP.BASE_SEED + run_counter)
                        np.random.seed(seed)

                        cfg = copy.deepcopy(base_cfg)
                        cfg.setdefault("paths", {})
                        cfg["paths"]["input_dir"] = str(input_dir)
                        cfg["method"] = model

                        method_cfg = cfg.get(model, {}).copy()
                        method_cfg.update(method_cfg_override)
                        cfg[model] = method_cfg

                        inv = TephraInversion(config=cfg)
                        inv.config["parameters"] = param_cfg

                        t0 = time.perf_counter()
                        results = inv.run_inversion()
                        t1 = time.perf_counter()
                        duration = t1 - t0

                        model_chain_dir = chains_root / model
                        model_chain_dir.mkdir(parents=True, exist_ok=True)
                        chain_path = model_chain_dir / f"{model}_run{run_id}.csv"
                        results["chain"].to_csv(chain_path, index=False)

                        row = _summarize_run(
                            model=model,
                            run_id=run_id,
                            simulation_id=sim_id,
                            seed=seed,
                            scale_h=scale_h,
                            scale_m=scale_m,
                            param_cfg=param_cfg,
                            method_cfg=method_cfg,
                            duration=duration,
                            results=results,
                        )
                        rows_by_model[model].append(row)

                        print(
                            f"[SIM] run_id={run_id} sim={sim_id} model={model} "
                            f"h_scale={float(scale_h):.3g} m_scale={float(scale_m):.3g} "
                            f"runtime={duration:.2f}s"
                        )

    for model, rows in rows_by_model.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        out_file = out_dir / f"results_{model}.csv"
        if out_file.exists():
            df.to_csv(out_file, mode="a", header=False, index=False)
        else:
            df.to_csv(out_file, index=False)
        print(f"[SIM] wrote {len(rows)} rows to {out_file}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run Tephra2 inversion experiment grid.")
    parser.add_argument(
        "--config-module",
        type=str,
        default="scripts.sim.exp_config",
        help="Python module path for experiment config (e.g. scripts.sim.exp_config_test)",
    )
    parser.add_argument("--plot-winds", action="store_true", help="Save a wind sanity plot under <output>/plots/.")

    parser.add_argument("--input-dir", type=str, default=None, help="Override EXP.SIM_INPUT_DIR")
    parser.add_argument("--output-dir", type=str, default=None, help="Override EXP.SIM_OUTPUT_DIR")

    parser.add_argument(
        "--invert-n-params",
        type=int,
        choices=[2, 4],
        default=None,
        help=(
            "Invert 2 params (plume_height, log_mass) or 4 params (+ median_grain, std_grain). "
            "If omitted, uses EXP.INVERT_N_PARAMS if present, else defaults to 2."
        ),
    )

    parser.add_argument(
        "--logm-prior-std",
        type=float,
        default=None,
        help=(
            "Override ln-mass prior_std (in natural log space). If omitted, uses EXP.LOGM_PRIOR_STD "
            "if present, else default ~ ln(10)/2."
        ),
    )

    args = parser.parse_args(argv)
    EXP = importlib.import_module(args.config_module)

    input_dir = Path(args.input_dir) if args.input_dir else Path(EXP.SIM_INPUT_DIR)
    output_dir = Path(args.output_dir) if args.output_dir else Path(EXP.SIM_OUTPUT_DIR)

    invert_n_params = args.invert_n_params
    if invert_n_params is None:
        invert_n_params = int(getattr(EXP, "INVERT_N_PARAMS", 2))

    run_all_experiments(
        EXP,
        input_dir=input_dir,
        out_dir=output_dir,
        plot_winds=args.plot_winds,
        invert_n_params=invert_n_params,
        logm_prior_std=args.logm_prior_std,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
