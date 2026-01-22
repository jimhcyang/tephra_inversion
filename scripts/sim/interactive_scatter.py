# scripts/sim/interactive_scatter.py
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

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
        return f"rgba(0,0,0,{alpha})"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _get_lnM_estimate_array(subset: pd.DataFrame) -> np.ndarray:
    """
    Canonical column is lnM_estimate.
    For backward compatibility, accept legacy logm_estimate too.
    """
    if "lnM_estimate" in subset.columns:
        return subset["lnM_estimate"].to_numpy(dtype=float)
    if "logm_estimate" in subset.columns:
        return subset["logm_estimate"].to_numpy(dtype=float)
    raise KeyError("subset is missing lnM_estimate/logm_estimate.")


def _collect_summary_points(
    sim_output_dir: str | Path,
    EXP,
    models: Sequence[str] = ("mcmc", "sa", "pso", "es"),
    max_configs: int = 4,
) -> pd.DataFrame:
    """
    One row per (model, prior_h, prior_m, config_index), summarising the MEDIAN
    plume and mass estimates across repeats.

    Columns:
        model, config_index, prior_h, prior_m, plume_est, mass_est, group_id
    """
    sim_output_dir = Path(sim_output_dir)
    records: list[dict] = []

    for model in models:
        model_l = model.lower()

        # IMPORTANT: pass EXP so any "rebuild from chains" uses the right config loops
        df, _ = load_results_df(model=model_l, sim_output_dir=sim_output_dir, EXP=EXP)
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
                    lnM_est = _get_lnM_estimate_array(subset)
                    mass_est = np.exp(lnM_est)

                    records.append(
                        dict(
                            model=model_l,
                            config_index=int(config_index),
                            prior_h=float(scale_h),
                            prior_m=float(scale_m),
                            plume_est=float(np.nanmedian(plume_est)),
                            mass_est=float(np.nanmedian(mass_est)),
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
    Interactive scatter:

    - x: plume height estimate (m)
    - y: eruption mass estimate (kg), log-scale
    - One point per (model, prior_h, prior_m, config_index)
    - Colour: model
    - Size: config_index
    - True source: star
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

    size_map = {0: 6, 1: 8, 2: 10, 3: 12}

    color_map = {
        "mcmc": "#1f77b4",
        "sa": "#ff7f0e",
        "pso": "#2ca02c",
        "es": "#d62728",
    }

    fig = go.Figure()

    # One trace per (model, prior_h, prior_m). Hover JS will toggle markers+lines on that trace.
    for (model, prior_h, prior_m), g in df_points.groupby(["model", "prior_h", "prior_m"]):
        g_sorted = g.sort_values("config_index")

        xs = g_sorted["plume_est"].to_numpy()
        ys = g_sorted["mass_est"].to_numpy()
        config_idx = g_sorted["config_index"].to_numpy(dtype=int)

        sizes = [size_map.get(int(ci), 8) for ci in config_idx]
        model_color = color_map.get(model, "gray")
        hover_bg = _hex_to_rgba(model_color, 0.2)

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
                mode="markers",
                marker=dict(
                    size=sizes,
                    color=model_color,
                    symbol="circle",
                    opacity=0.8,
                ),
                showlegend=False,
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
                    bordercolor=hover_bg,
                ),
            )
        )

    # Legend entries (dummy points)
    for model in models:
        model_l = model.lower()
        model_color = color_map.get(model_l, "gray")
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=model_color, symbol="circle", opacity=0.9),
                showlegend=True,
                name=model_l.upper(),
                legendgroup=model_l,
                hoverinfo="skip",
            )
        )

    # True source point
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

    fig.update_yaxes(type="log")
    return fig


def write_hover_interactive_html(fig: go.Figure, out_html: Path) -> None:
    """
    Write an HTML file with hover behavior:
      - on hover: hovered trace becomes markers+lines and full opacity
      - others fade
    """
    div_id = "interactive-scatter-div"

    post_script = f"""
    var gd = document.getElementById('{div_id}');
    if (!gd) {{
        console.warn("Plotly div not found for id {div_id}");
    }} else {{

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

    out_html = Path(out_html)
    html_str = pio.to_html(
        fig,
        include_plotlyjs="cdn",
        full_html=True,
        div_id=div_id,
        post_script=post_script,
    )
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

    # Keep outputs *inside* the experiment directory so multi-scenario runs
    # don't overwrite each other.
    out_html = sim_output_dir / "interactive_scatter_all_models_hover.html"
    write_hover_interactive_html(fig, out_html)

    # python -m scripts.sim.interactive_scatter
