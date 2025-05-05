"""
build_inputs.py
===============

Create the full Tephra-2 input bundle (`tephra2.conf`, `esp_input.csv`,
`wind.txt`) in one call, *always* producing:

1. Observation scatter (log-scale colour).
2. Three-panel wind-profile figure.

Existing files are reused unless `load_observations=False`
or `load_wind=False`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .coordinate_utils import latlon_to_utm
from .esp_config import write_esp_input, write_tephra2_conf
from .observation_data import ObservationHandler
from .wind_data import WindDataHandler

LOGGER = logging.getLogger(__name__)
PLOTS_DIR = Path("data/output/plots")


def _plot_wind(df: np.ndarray, save_path: Path, show: bool = True) -> None:
    """Re-create the notebook’s 3-panel wind diagnostic."""
    elev = df["HEIGHT"].to_numpy()
    spd  = df["SPEED"].to_numpy()
    dirc = df["DIRECTION"].to_numpy()

    order = np.argsort(elev)
    elev, spd, dirc = elev[order], spd[order], dirc[order]

    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(15, 5))

    # 1) Speed vs Altitude
    ax1 = fig.add_subplot(1, 3, 1)
    sns.lineplot(x=elev, y=spd, marker="o", linewidth=2, color="steelblue", ax=ax1)
    ax1.set_xlabel("Altitude (m)")
    ax1.set_ylabel("Wind speed (m/s)")
    ax1.set_title("Wind speed vs altitude")

    # 2) Direction vs Altitude
    ax2 = fig.add_subplot(1, 3, 2)
    sns.lineplot(x=elev, y=dirc, marker="o", linewidth=2, color="steelblue", ax=ax2)
    ax2.set_xlabel("Altitude (m)")
    ax2.set_ylabel("Wind direction (°)")
    ax2.set_title("Wind direction vs altitude")

    # 3) Polar plot
    ax3 = fig.add_subplot(1, 3, 3, projection="polar")
    ax3.plot(np.radians(dirc), elev, marker="o", color="steelblue")
    ax3.set_theta_zero_location("N")
    ax3.set_theta_direction(-1)
    ax3.set_title("Polar (direction vs altitude)")

    # Radial grid labels
    max_alt = float(elev.max())
    step = max_alt / 10 if max_alt else 1000
    circles = np.arange(step, max_alt + step, step)
    ax3.set_rgrids(circles, angle=60, labels=[f"{int(c)}" for c in circles])

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)


def build_all(
    *,
    vent_lat: float,
    vent_lon: float,
    vent_elev: float,
    base_dir: str | Path = "data/input",
    load_observations: bool = True,
    load_wind: bool = True,
    obs_params: Optional[Dict] = None,
    wind_params: Optional[Dict] = None,
    show_plots: bool = True,
) -> Tuple[Path, Path, Path]:
    """
    Returns (tephra2_conf, esp_input_csv, wind_txt) after ensuring:
    - observations.csv & sites.csv exist (or are synthesized)
    - wind.txt exists (or is synthesized)
    - diagnostic plots are saved/shown.
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # 1) Observations
    obs_handler = ObservationHandler(base_dir)
    if load_observations and (base_dir / "observations.csv").exists():
        obs_vec, sites = obs_handler.load_observations()
    else:
        obs_vec, sites = obs_handler.generate_synthetic(**(obs_params or {}))
        obs_handler.save_observations(obs_vec, sites)

    if show_plots:
        obs_handler.plot_observations(
            obs_vec,
            sites,
            output_path=PLOTS_DIR / "observations.png",
            show_plot=True,
        )

    # 2) Wind
    wind_handler = WindDataHandler(base_dir)
    if load_wind and (base_dir / "wind.txt").exists():
        wind_df = wind_handler.load_wind_data()
    else:
        wind_df = wind_handler.generate_wind_data(**(wind_params or {}))
        wind_handler.save_wind_data(wind_df)

    if show_plots:
        _plot_wind(wind_df, PLOTS_DIR / "wind_profile.png", show=True)

    # 3) Tephra2 config & ESP inputs
    easting, northing, _ = latlon_to_utm(vent_lat, vent_lon)
    conf_path = write_tephra2_conf(easting, northing, vent_elev)
    esp_path  = write_esp_input(easting, northing, vent_elev)

    LOGGER.info("Built inputs at %s", base_dir)
    return conf_path, esp_path, base_dir / "wind.txt"
