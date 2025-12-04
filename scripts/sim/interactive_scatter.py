# scripts/sim/interactive_scatter.py
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .sim_types import GroupSpec
from .results_io import (
    load_results_df,
    list_model_configs,
    get_config_hyperparams,
    filter_group_rows,
)

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """
    Convert a hex color like '#1f77b4' to an 'rgba(r,g,b,a)' CSS string.
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        # Fallback: just return black with given alpha
        return f"rgba(0,0,0,{alpha})"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def _collect_summary_points(
    sim_output_dir: str | Path,
    EXP,
    models: Sequence[str] = ("mcmc", "sa", "pso", "es"),
    max_configs: int = 4,
) -> pd.DataFrame:
    """
    Build a DataFrame with one row per (model, prior_h, prior_m, config_index),
    summarising the *median* plume and mass estimates across repeats.

    Columns:
        model, config_index, prior_h, prior_m,
        plume_est, mass_est, group_id
    """
    sim_output_dir = Path(sim_output_dir)

    records: list[dict] = []

    for model in models:
        model_l = model.lower()

        # Load once per model
        df, _ = load_results_df(model=model_l, sim_output_dir=sim_output_dir, allow_minimal=True)
        df_model = df[df["model"].str.lower() == model_l].copy()
        if df_model.empty:
            continue

        configs = list_model_configs(df_model, model_l)
        n_cfg = min(max_configs, len(configs))
        if n_cfg == 0:
            continue

        for config_index in range(n_cfg):
            hyper_cfg = get_config_hyperparams(df_model, model_l, config_index)

            for scale_h in EXP.PRIOR_FACTORS:
                for scale_m in EXP.PRIOR_FACTORS:
                    spec = GroupSpec(
                        model=model_l,
                        scale_plume_factor=scale_h,
                        scale_mass_factor=scale_m,
                        n_iter=hyper_cfg.get("n_iterations"),
                        runs=hyper_cfg.get("runs"),
                        restarts=hyper_cfg.get("restarts"),
                        n_ens=hyper_cfg.get("n_ens"),
                        n_assimilations=hyper_cfg.get("n_assimilations"),
                        config_index=config_index,
                    )

                    subset = filter_group_rows(df_model, spec, allow_empty=True)
                    if subset.empty:
                        continue

                    plume_est = subset["plume_estimate"].to_numpy(dtype=float)
                    logm_est = subset["logm_estimate"].to_numpy(dtype=float)
                    mass_est = np.exp(logm_est)

                    # Median across repeats for plotting
                    median_plume = float(np.nanmedian(plume_est))
                    median_mass = float(np.nanmedian(mass_est))

                    records.append(
                        dict(
                            model=model_l,
                            config_index=int(config_index),
                            prior_h=float(scale_h),
                            prior_m=float(scale_m),
                            plume_est=median_plume,
                            mass_est=median_mass,
                            group_id=f"{model_l}_h{scale_h}_m{scale_m}",
                        )
                    )

    if not records:
        return pd.DataFrame()

    return pd.DataFrame.from_records(records)


def make_interactive_scatter_all_models(
    sim_output_dir: str | Path,
    EXP,
    true_plume_height: float,
    true_eruption_mass: float,
    models: Sequence[str] = ("mcmc", "sa", "pso", "es"),
    max_configs: int = 4,
) -> go.Figure:
    """
    Build an interactive scatter plot:

    - x-axis: plume height estimate (m)
    - y-axis: eruption mass estimate (kg)
    - One point per (model, prior_h, prior_m, config_index) — up to
      49 priors × 4 configs per model.
    - Colour: model
    - Size: config_index (0–3), with small absolute sizes
    - For each (model, prior_h, prior_m), the 4 configs are connected
      by a line (so hovering highlights the little trajectory).
    - The true source is shown as a red star at (true_plume_height, true_eruption_mass).
    """
    sim_output_dir = Path(sim_output_dir)

    df_points = _collect_summary_points(
        sim_output_dir=sim_output_dir,
        EXP=EXP,
        models=models,
        max_configs=max_configs,
    )

    if df_points.empty:
        raise ValueError("No summary points could be constructed for the requested models.")

    # Map config_index -> marker size (all relatively small)
    size_map = {0: 6, 1: 8, 2: 10, 3: 12}

    # Colours per model
    color_map = {
        "mcmc": "#1f77b4",  # blue
        "sa": "#ff7f0e",    # orange
        "pso": "#2ca02c",   # green
        "es": "#d62728",    # red
    }

    fig = go.Figure()

    # One trace per (model, prior_h, prior_m) group: markers + line
    for (model, prior_h, prior_m), g in df_points.groupby(["model", "prior_h", "prior_m"]):
        g_sorted = g.sort_values("config_index")

        xs = g_sorted["plume_est"].to_numpy()
        ys = g_sorted["mass_est"].to_numpy()
        config_idx = g_sorted["config_index"].to_numpy(dtype=int)

        sizes = [size_map.get(int(ci), 8) for ci in config_idx]
        model_color = color_map.get(model, "gray")
        hover_bg = _hex_to_rgba(model_color, 0.2)  # same color, low alpha

        # customdata used in hovertemplate
        customdata = np.column_stack(
            [
                g_sorted["model"].to_numpy(),
                g_sorted["config_index"].to_numpy(),
                g_sorted["prior_h"].to_numpy(),
                g_sorted["prior_m"].to_numpy(),
            ]
        )

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",  # no line by default
                marker=dict(
                    size=sizes,
                    color=model_color,
                    symbol="circle",
                    opacity=0.8,
                ),
                showlegend=False,   # legend handled separately
                legendgroup=model,
                hovertemplate=(
                    "Model: %{customdata[0]}<br>"
                    "Config idx: %{customdata[1]}<br>"
                    "Prior h×: %{customdata[2]}<br>"
                    "Prior m×: %{customdata[3]}<br>"
                    "Plume est: %{x:.1f} m<br>"
                    "Mass est: %{y:.3e} kg"
                    "<extra></extra>"
                ),
                customdata=customdata,
                hoverlabel=dict(
                    bgcolor=hover_bg,
                    bordercolor=hover_bg,  # same color, faint border
                ),
            )
        )

    # Add a single legend entry per model (dummy points)
    for model in models:
        model_l = model.lower()
        model_color = color_map.get(model_l, "gray")
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    size=10,
                    color=model_color,
                    symbol="circle",
                    opacity=0.9,
                ),
                showlegend=True,
                name=model_l.upper(),
                legendgroup=model_l,
                hoverinfo="skip",
            )
        )

    # True source point as red star
    fig.add_trace(
        go.Scatter(
            x=[true_plume_height],
            y=[true_eruption_mass],
            mode="markers",
            marker=dict(
                size=14,
                color="black",
                symbol="star",
                line=dict(color="black", width=1),
            ),
            name="True source",
            hovertemplate=(
                "True plume/mass<br>"
                "Plume: %{x:.1f} m<br>"
                "Mass: %{y:.3e} kg"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Plume height vs eruption mass — all models, priors, and configs",
        xaxis_title="Plume height estimate (m)",
        yaxis_title="Eruption mass estimate (kg)",
        legend_title="Model",
        hovermode="closest",
        template="plotly_white",
    )

    # Default: log-scale for eruption mass
    fig.update_yaxes(type="log")

    return fig

import plotly.io as pio


def write_hover_interactive_html(fig: go.Figure, out_html: Path):
    """
    Write the given figure to an HTML file where:
      - group traces are markers only by default,
      - on hover, the hovered trace gets a connecting line and full opacity,
      - all other group traces fade to low opacity.

    This uses Plotly JS event handlers injected via `post_script`.
    """
    div_id = "interactive-scatter-div"

    # JS: add hover/unhover callbacks
    post_script = f"""
    var gd = document.getElementById('{div_id}');
    if (!gd) {{
        console.warn("Plotly div not found for id {div_id}");
    }} else {{

        // Save default modes and opacities
        var defaultModes = [];
        var defaultOpacities = [];
        for (var i = 0; i < gd.data.length; i++) {{
            defaultModes.push(gd.data[i].mode || "markers");
            var m = gd.data[i].marker || {{}};
            defaultOpacities.push(
                (m.opacity !== undefined && m.opacity !== null) ? m.opacity : 0.8
            );
        }}

        gd.on('plotly_hover', function(ev) {{
            if (!ev || !ev.points || !ev.points.length) return;
            var traceIndex = ev.points[0].curveNumber;

            var newModes = [];
            var newOpacities = [];

            for (var i = 0; i < gd.data.length; i++) {{
                var d = gd.data[i];

                // Keep "True source" and legend-only traces unchanged
                var isTrue = (d.name === "True source");
                var isLegendOnly = (d.x && d.x.length === 1 && d.x[0] === null);

                if (isTrue || isLegendOnly) {{
                    newModes.push(defaultModes[i]);
                    newOpacities.push(defaultOpacities[i]);
                    continue;
                }}

                if (i === traceIndex) {{
                    newModes.push("markers+lines");
                    newOpacities.push(1.0);
                }} else {{
                    newModes.push("markers");
                    newOpacities.push(0.1);
                }}
            }}

            Plotly.restyle(gd, {{
                mode: newModes,
                'marker.opacity': newOpacities
            }});
        }});

        gd.on('plotly_unhover', function(ev) {{
            Plotly.restyle(gd, {{
                mode: defaultModes,
                'marker.opacity': defaultOpacities
            }});
        }});
    }}
    """

    html_str = pio.to_html(
        fig,
        include_plotlyjs="cdn",
        full_html=True,
        div_id=div_id,
        post_script=post_script,
    )

    out_html = Path(out_html)
    out_html.write_text(html_str, encoding="utf-8")
    print(f"Wrote interactive HTML with hover behavior to {out_html}")

if __name__ == "__main__":
    from . import exp_config as EXP  # or exp_config_test

    sim_output_dir = Path(EXP.SIM_OUTPUT_DIR)
    true_plume_height = 7000.0
    true_eruption_mass = 2.4e10

    fig = make_interactive_scatter_all_models(
        sim_output_dir=sim_output_dir,
        EXP=EXP,
        true_plume_height=true_plume_height,
        true_eruption_mass=true_eruption_mass,
        models=("mcmc", "sa", "pso", "es"),
        max_configs=4,
    )

    out_html = sim_output_dir.parent / "interactive_scatter_all_models_hover.html"
    write_hover_interactive_html(fig, out_html)

# python -m scripts.sim.interactive_scatter