# scripts/sim/aggregate_sim_visuals.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, List, Dict, Any

from .sim_types import GroupSpec, GroupResult
from .plots_demo import make_demo_plots
from .metrics_heatmaps import (
    compute_accuracy_time_grids,
    plot_accuracy_time_heatmaps_for_config,
    plot_accuracy_time_heatmaps_for_model,
)
from .simple_plots import plot_single_config_traces_and_marginals

__all__ = [
    "GroupSpec",
    "GroupResult",
    "make_demo_plots",
    "compute_accuracy_time_grids",
    "plot_accuracy_time_heatmaps_for_config",
    "plot_accuracy_time_heatmaps_for_model",
    "plot_single_config_traces_and_marginals",
]


if __name__ == "__main__":
    # Simple CLI demo using exp_config.SIM_OUTPUT_DIR
    try:
        from scripts.sim.exp_config import SIM_OUTPUT_DIR as DEFAULT_SIM_OUTPUT_DIR
    except Exception:
        DEFAULT_SIM_OUTPUT_DIR = "data_sim_cerro/experiments"

    summary = make_demo_plots(sim_output_dir=DEFAULT_SIM_OUTPUT_DIR, show=False)
    print("Generated demo plots:")
    for gr in summary:
        print(
            f"  {gr.model.upper()} | "
            f"h×={gr.scale_plume_factor:g}, m×={gr.scale_mass_factor:g} | "
            f"cfg={gr.config_index:02d} | runs={gr.n_runs} | steps={gr.n_steps}"
        )
        print(f"    trace:     {gr.trace_path}")
        print(f"    marginals: {gr.marginals_path}")
