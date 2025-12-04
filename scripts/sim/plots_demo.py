# scripts/sim/plots_demo.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .sim_types import GroupSpec, GroupResult
from .results_io import load_results_df, filter_group_rows, load_chains_for_group
from .plot_traces import plot_traces_with_mean, plot_marginals_with_mean_hist

try:
    # Default from your experiment config
    from scripts.sim.exp_config import SIM_OUTPUT_DIR as DEFAULT_SIM_OUTPUT_DIR
except Exception:
    DEFAULT_SIM_OUTPUT_DIR = "data_sim_cerro/experiments"


def _build_demo_specs(
    demo_priors: Sequence[float],
    sa_runs: int,
    sa_restarts: int,
    pso_runs: int,
    pso_restarts: int,
    es_n_ens: int,
    es_n_assimilations: int,
    mcmc_n_iter: int,
) -> Dict[str, List[GroupSpec]]:
    """
    Build the demo subset of GroupSpecs for each model.

    Here we keep plume and mass scaling equal (diagonal of the 7x7 grid),
    i.e. scale_plume_factor = scale_mass_factor = pf.
    """
    specs: Dict[str, List[GroupSpec]] = {
        "sa": [],
        "pso": [],
        "es": [],
        "mcmc": [],
    }

    for i, pf in enumerate(demo_priors):
        specs["sa"].append(
            GroupSpec(
                model="sa",
                scale_plume_factor=pf,
                scale_mass_factor=pf,
                runs=sa_runs,
                restarts=sa_restarts,
                config_index=i,
            )
        )
        specs["pso"].append(
            GroupSpec(
                model="pso",
                scale_plume_factor=pf,
                scale_mass_factor=pf,
                runs=pso_runs,
                restarts=pso_restarts,
                config_index=i,
            )
        )
        specs["es"].append(
            GroupSpec(
                model="es",
                scale_plume_factor=pf,
                scale_mass_factor=pf,
                n_ens=es_n_ens,
                n_assimilations=es_n_assimilations,
                config_index=i,
            )
        )
        specs["mcmc"].append(
            GroupSpec(
                model="mcmc",
                scale_plume_factor=pf,
                scale_mass_factor=pf,
                n_iter=mcmc_n_iter,
                config_index=i,
            )
        )

    return specs


def make_demo_plots(
    sim_output_dir: str | Path | None = None,
    show: bool = False,
    demo_priors: Optional[Sequence[float]] = None,
    sa_runs: int = 1000,
    sa_restarts: int = 0,
    pso_runs: int = 100,
    pso_restarts: int = 0,
    es_n_ens: int = 100,
    es_n_assimilations: int = 1,
    mcmc_n_iter: int = 1000,
) -> List[GroupResult]:
    """
    Aggregate and plot a demo subset:

      - multi-trace plots with mean trajectory,
      - marginal histograms with mean-by-index overlay.

    Returns:
        List[GroupResult] describing each (model, prior_factor, config_index)
        combination that was plotted.
    """
    if sim_output_dir is None:
        sim_output_dir = DEFAULT_SIM_OUTPUT_DIR
    sim_output_dir = Path(sim_output_dir)

    # Where to save
    root_out = sim_output_dir.parent / "output"
    trace_root = root_out / "trace"
    marg_root = root_out / "marginals"
    trace_root.mkdir(parents=True, exist_ok=True)
    marg_root.mkdir(parents=True, exist_ok=True)

    # Default prior factors if not provided
    if demo_priors is None:
        demo_priors = [2.5, 1.0, 0.4]

    demo_specs = _build_demo_specs(
        demo_priors=demo_priors,
        sa_runs=sa_runs,
        sa_restarts=sa_restarts,
        pso_runs=pso_runs,
        pso_restarts=pso_restarts,
        es_n_ens=es_n_ens,
        es_n_assimilations=es_n_assimilations,
        mcmc_n_iter=mcmc_n_iter,
    )

    results: List[GroupResult] = []

    for model, specs in demo_specs.items():
        df_results, is_minimal = load_results_df(model, sim_output_dir, allow_minimal=True)
        if df_results.empty:
            continue

        for spec in specs:
            subset = filter_group_rows(df_results, spec, allow_empty=False)
            chains, burnin = load_chains_for_group(model, subset, sim_output_dir)

            param_cols = list(chains[0].columns)
            n_runs = len(chains)
            n_steps = len(chains[0])

            base_tag = f"{model}_config_{spec.config_index:04d}"
            trace_path = trace_root / f"{base_tag}_trace.png"
            marg_path = marg_root / f"{base_tag}_marginals.png"

            prior_label = (
                f"σ×={spec.prior_factor:g}"
                if spec.prior_factor is not None
                else f"h×={spec.scale_plume_factor:g}, m×={spec.scale_mass_factor:g}"
            )
            title_trace = f"{model.upper()} | {prior_label} | config {spec.config_index:02d}"
            title_marg = f"{model.upper()} marginals | {prior_label} | config {spec.config_index:02d}"

            plot_traces_with_mean(
                chains=chains,
                burnin=burnin,
                out_path=trace_path,
                title=title_trace,
                show=show,
            )

            plot_marginals_with_mean_hist(
                chains=chains,
                burnin=burnin,
                out_path=marg_path,
                title=title_marg,
                show=show,
            )

            results.append(
                GroupResult(
                    model=model,
                    scale_plume_factor=spec.scale_plume_factor or np.nan,
                    scale_mass_factor=spec.scale_mass_factor or np.nan,
                    config_index=spec.config_index,
                    trace_path=trace_path,
                    marginals_path=marg_path,
                    n_runs=n_runs,
                    n_steps=n_steps,
                    param_cols=param_cols,
                )
            )

    return results
