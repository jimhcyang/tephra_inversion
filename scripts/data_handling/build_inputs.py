"""
build_inputs.py – minimal, Victor‑safe, *now with* load_observations / load_wind
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import utm
import numpy as np

from .coordinate_utils import latlon_to_utm
from .esp_config import write_esp_input, write_tephra2_conf
from .observation_data import ObservationHandler
from .wind_data import WindDataHandler
from scripts.visualization.observation_plots import ObservationPlotter
from scripts.visualization.wind_plots import WindPlotter

from .dem_utils import (
    download_dem,
    get_elevation_at_point,
    VICTOR_AVAILABLE,
)

LOGGER = logging.getLogger(__name__)
PLOTS_DIR = Path("data/output/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def build_all(
    *,
    vent_lat: float,
    vent_lon: float,
    vent_elev: Optional[float] = None,
    base_dir: str | Path = "data/input",
    # DEM options ------------------------------------------------------------
    use_dem: bool = True,
    dem_buffer: float = 1.0,
    dem_dataset: str = "COP30",
    dem_path: Optional[str | Path] = None,
    # new flags expected by TephraInversion ----------------------------------
    load_observations: bool = True,
    load_wind: bool = True,
    # synthetic‑data parameters ----------------------------------------------
    obs_params: Optional[Dict] = None,
    wind_params: Optional[Dict] = None,
    show_plots: bool = True,
) -> Tuple[Path, Path, Path]:
    """
    Build Tephra‑2 inputs and return paths to (tephra2.conf, esp_input.csv, wind.txt).
    """
    base_dir = Path(base_dir).expanduser()
    base_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- DEM
    actual_dem_path: Optional[Path] = None
    if use_dem and VICTOR_AVAILABLE:
        if dem_path is None:
            actual_dem_path = download_dem(
                vent_lat + dem_buffer,
                vent_lat - dem_buffer,
                vent_lon - dem_buffer,
                vent_lon + dem_buffer,
                output_dir=base_dir,
                filename="tephra2.tiff",
                dataset=dem_dataset,
            )
        else:
            actual_dem_path = Path(dem_path).expanduser()
        if vent_elev is None and actual_dem_path.exists():
            vent_elev = get_elevation_at_point(actual_dem_path, vent_lat, vent_lon)
            LOGGER.info("Vent elevation from DEM: %.1f m", vent_elev)
    else:
        if use_dem and not VICTOR_AVAILABLE:
            LOGGER.warning("Victor not available – skipping DEM download")
        if vent_elev is None:
            vent_elev = 0.0
            LOGGER.warning("No elevation provided – defaulting to 0 m")

    # ---------------------------------------------------------------- OBSERVATIONS
    obs_hdl = ObservationHandler(base_dir)
    if load_observations and (base_dir / "observations.csv").exists():
        obs_vec, sites = obs_hdl.load_observations()
    else:
        obs_vec, sites = obs_hdl.generate_synthetic(**(obs_params or {}))
        obs_hdl.save_observations(obs_vec, sites)

    # ---------------------------------------------------------------- WIND
    wind_hdl = WindDataHandler(base_dir)
    if load_wind and (base_dir / "wind.txt").exists():
        wind_df = wind_hdl.load_wind_data()
    else:
        wind_df = wind_hdl.generate_wind_data(**(wind_params or {}))
        wind_hdl.save_wind_data(wind_df)

    # ---------------------------------------------------------------- PLOTS
    vent_east, vent_north, _ = latlon_to_utm(vent_lat, vent_lon)
    obs_plot = ObservationPlotter(PLOTS_DIR)
    obs_plot.plot_tephra_distribution(
        eastings=sites[:, 0],
        northings=sites[:, 1],
        thicknesses=obs_vec,
        vent_location=(vent_east, vent_north),
        save_path=PLOTS_DIR / "observations.png",
        show_plot=show_plots,
    )

    zone_num, zone_letter = utm.from_latlon(vent_lat, vent_lon)[2:]
    lats, lons = utm.to_latlon(sites[:, 0], sites[:, 1], zone_num, zone_letter)
    obs_plot.plot_isomass_map(
        lon=lons,
        lat=lats,
        mass=obs_vec,
        vent_lon=vent_lon,
        vent_lat=vent_lat,
        dem_path=str(actual_dem_path) if actual_dem_path else "none",
        save_path=PLOTS_DIR / "isomass_map.png",
        show_plot=show_plots,
    )

    wp = WindPlotter(PLOTS_DIR)
    wp.plot_wind_profile(
        heights=wind_df["HEIGHT"].to_numpy(),
        speeds=wind_df["SPEED"].to_numpy(),
        directions=wind_df["DIRECTION"].to_numpy(),
        save_path=PLOTS_DIR / "wind_profile.png",
        show_plot=show_plots,
    )

    # ----------------------------------------------------------- CONFIG FILES
    conf_path = write_tephra2_conf(vent_east, vent_north, vent_elev)
    esp_path = write_esp_input(vent_east, vent_north, vent_elev)
    wind_path = base_dir / "wind.txt"

    return conf_path, esp_path, wind_path
