#!/usr/bin/env python3
# scripts/visualization/postprocess_sim_results.py
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from scripts.sim.metrics_heatmaps import plot_accuracy_time_heatmaps_for_model
from scripts.sim.interactive_scatter import (
    make_interactive_scatter_all_models,
    write_hover_interactive_html,
)
from scripts.sim.simple_plots import plot_single_config_traces_and_marginals
from scripts.sim.results_io import load_results_df, get_config_hyperparams


def _load_sim_meta(scenario_dir: Path) -> Dict[str, Any]:
    """
    Reads <scenario_dir>/config/sim_meta.json if present.
    """
    meta_path = scenario_dir / "config" / "sim_meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return {}


def _true_values_from_meta(meta: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (true_plume_height_m, true_mass_kg) if available.
    Accepts:
      - plume_height
      - eruption_mass_kg
      - ln_mass
    """
    true_plume = None
    true_mass = None

    if "plume_height" in meta:
        try:
            true_plume = float(meta["plume_height"])
        except Exception:
            true_plume = None

    if "eruption_mass_kg" in meta:
        try:
            true_mass = float(meta["eruption_mass_kg"])
        except Exception:
            true_mass = None
    elif "ln_mass" in meta:
        try:
            true_mass = float(np.exp(float(meta["ln_mass"])))
        except Exception:
            true_mass = None

    return true_plume, true_mass


def _list_scenarios(scenarios_root: Path, only: Optional[str] = None) -> List[str]:
    scenarios_root = Path(scenarios_root)
    if only is not None:
        return [only]

    if not scenarios_root.exists():
        return []
    # scenario dirs: data/scenarios/<scenario_name>/
    return sorted([p.name for p in scenarios_root.iterdir() if p.is_dir()])


def _available_prior_pairs_for_model_config(
    exp_dir: Path,
    model: str,
    EXP: Any,
    config_index: int = 0,
) -> Set[Tuple[float, float]]:
    """
    Return {(scale_h, scale_m), ...} that actually exist in results_<model>.csv
    for the given model/config_index.
    """
    model = model.lower()
    df, _ = load_results_df(model=model, sim_output_dir=exp_dir, allow_minimal=True, EXP=EXP)

    dfm = df[df["model"].str.lower() == model].copy()
    if dfm.empty:
        return set()

    # Identify the hyperparameter combo for this config_index,
    # then filter down to those rows.
    hyper_cfg = get_config_hyperparams(dfm, model, config_index)

    mask = np.ones(len(dfm), dtype=bool)
    if model == "mcmc" and "n_iterations" in hyper_cfg and "n_iterations" in dfm.columns:
        mask &= (dfm["n_iterations"] == hyper_cfg["n_iterations"])

    if model in ("sa", "pso"):
        if "runs" in hyper_cfg and "runs" in dfm.columns:
            mask &= (dfm["runs"] == hyper_cfg["runs"])
        if "restarts" in hyper_cfg and "restarts" in dfm.columns:
            mask &= (dfm["restarts"] == hyper_cfg["restarts"])

    if model == "es":
        if "n_ens" in hyper_cfg and "n_ens" in dfm.columns:
            mask &= (dfm["n_ens"] == hyper_cfg["n_ens"])
        if "n_assimilations" in hyper_cfg and "n_assimilations" in dfm.columns:
            mask &= (dfm["n_assimilations"] == hyper_cfg["n_assimilations"])

    dfc = dfm.loc[mask].copy()
    if dfc.empty:
        return set()

    if "scale_plume_factor" not in dfc.columns or "scale_mass_factor" not in dfc.columns:
        return set()

    pairs = set(
        (float(h), float(m))
        for h, m in zip(dfc["scale_plume_factor"].to_numpy(), dfc["scale_mass_factor"].to_numpy())
    )
    return pairs


def _choose_trace_pairs(
    available: Set[Tuple[float, float]],
    preferred: Sequence[Tuple[float, float]] = ((1.0, 1.0), (2.0, 2.0), (0.5, 0.5)),
    max_pairs: int = 3,
) -> List[Tuple[float, float]]:
    """
    Pick trace pairs to plot:
      1) keep preferred pairs if they exist
      2) else fall back to first available pairs
    """
    chosen: List[Tuple[float, float]] = []
    for p in preferred:
        if p in available:
            chosen.append(p)
        if len(chosen) >= max_pairs:
            return chosen

    if not chosen:
        # fallback: arbitrary stable order
        for p in sorted(list(available)):
            chosen.append(p)
            if len(chosen) >= max_pairs:
                break

    return chosen


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Postprocess simulation results: heatmaps, interactive scatter, traces."
    )
    parser.add_argument("--experiments-root", type=str, required=True)
    parser.add_argument("--scenarios-root", type=str, required=True)
    parser.add_argument("--config-module", type=str, required=True)

    parser.add_argument("--only", type=str, default=None, help="Only postprocess one scenario name.")

    parser.add_argument("--plot-heatmaps", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--traces", action="store_true")

    parser.add_argument("--max-trace-pairs", type=int, default=3)
    parser.add_argument("--trace-config-index", type=int, default=0)
    parser.add_argument("--trace-run-index", type=int, default=0)

    args = parser.parse_args(argv)

    experiments_root = Path(args.experiments_root).resolve()
    scenarios_root = Path(args.scenarios_root).resolve()

    EXP = importlib.import_module(args.config_module)

    scenarios = _list_scenarios(scenarios_root, only=args.only)
    models = tuple(getattr(EXP, "MODELS", ("mcmc", "sa", "pso", "es")))

    print(f"[POST] experiments_root = {experiments_root}")
    print(f"[POST] scenarios_root   = {scenarios_root}")
    print(f"[POST] config_module    = {args.config_module}")
    print(f"[POST] scenarios        = {scenarios}")
    print(f"[POST] models           = {models}")

    for sname in scenarios:
        scenario_dir = scenarios_root / sname
        exp_dir = experiments_root / sname

        if not exp_dir.exists():
            print(f"\n[SCENARIO] {sname}")
            print(f"  [SKIP] exp_dir does not exist: {exp_dir}")
            continue

        meta = _load_sim_meta(scenario_dir)
        true_plume, true_mass = _true_values_from_meta(meta)

        print(f"\n[SCENARIO] {sname}")
        print(f"  exp_dir     = {exp_dir}")
        print(f"  true_plume  = {true_plume}")
        print(f"  true_mass   = {true_mass}")

        if true_plume is None or true_mass is None:
            print("  [WARN] Missing true values (plume_height / eruption_mass_kg or ln_mass) in sim_meta.json;")
            print("         heatmaps/errors may be meaningless. (Traces will still plot chains.)")

        # --------------------------
        # Heatmaps
        # --------------------------
        if args.plot_heatmaps:
            for m in models:
                print(f"  [HEATMAP] {m.upper()}")
                try:
                    png = plot_accuracy_time_heatmaps_for_model(
                        sim_output_dir=exp_dir,
                        model=m,
                        true_plume_height=float(true_plume) if true_plume is not None else 1.0,
                        true_eruption_mass=float(true_mass) if true_mass is not None else 1.0,
                        EXP=EXP,
                        show=False,
                    )
                    print(f"    saved: {png}")
                except Exception as e:
                    print(f"    [WARN] heatmap failed for {m}: {e}")

        # --------------------------
        # Interactive scatter
        # --------------------------
        if args.interactive:
            print("  [INTERACTIVE] building scatter...")
            try:
                fig = make_interactive_scatter_all_models(
                    sim_output_dir=exp_dir,
                    EXP=EXP,
                    true_plume_height=float(true_plume) if true_plume is not None else 1.0,
                    true_eruption_mass=float(true_mass) if true_mass is not None else 1.0,
                    models=models,
                    max_configs=4,
                )
                out_html = exp_dir / "interactive_scatter_all_models_hover.html"
                write_hover_interactive_html(fig, out_html)
                print(f"    saved: {out_html}")
            except Exception as e:
                print(f"    [WARN] interactive scatter failed: {e}")

        # --------------------------
        # Traces (robust to missing prior-factor pairs)
        # --------------------------
        if args.traces:
            cfg_idx = int(args.trace_config_index)
            run_idx = int(args.trace_run_index)

            # Choose trace pairs based on availability in results
            avail = _available_prior_pairs_for_model_config(
                exp_dir=exp_dir,
                model="mcmc",   # use mcmc as the reference for "what priors exist"
                EXP=EXP,
                config_index=cfg_idx,
            )

            if not avail:
                print("  [TRACE] [SKIP] No available prior-factor pairs found in results (did you run simulate?).")
                continue

            trace_pairs = _choose_trace_pairs(
                available=avail,
                max_pairs=int(args.max_trace_pairs),
            )

            for (hx, mx) in trace_pairs:
                # For each model, attempt to plot the chain for this prior pair
                for m in models:
                    print(f"  [TRACE] model={m} hx={hx:g}, mx={mx:g} cfg={cfg_idx} run={run_idx}")
                    try:
                        trace_path, marg_path = plot_single_config_traces_and_marginals(
                            sim_output_dir=exp_dir,
                            model=m,
                            prior_factors=(hx, mx),
                            config_index=cfg_idx,
                            run_index=run_idx,
                            true_plume_height=true_plume,
                            true_eruption_mass=true_mass,
                            show=False,
                            EXP=EXP,
                        )
                        print(f"    trace: {trace_path}")
                        print(f"    marg : {marg_path}")
                    except Exception as e:
                        print(f"    [SKIP] {m} hx={hx:g} mx={mx:g} not available (or failed): {e}")

    print("\n[POST] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
