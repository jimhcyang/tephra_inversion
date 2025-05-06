"""
build_inputs.py
===============

Creates a complete Tephra‑2 input bundle and three diagnostic plots:
  1. UTM scatter of observations
  2. Three‑panel wind profile with speed-colored scatter
  3. Isomass contour map with DEM background

Outputs land in data/input/  and  data/output/plots/.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import utm                                # only for UTM→lat/lon conversion

from .coordinate_utils import latlon_to_utm
from .esp_config          import write_esp_input, write_tephra2_conf
from .observation_data    import ObservationHandler
from .wind_data           import WindDataHandler
from .dem_utils           import download_dem, get_elevation_at_point
from scripts.visualization.observation_plots import ObservationPlotter
from scripts.visualization.wind_plots        import WindPlotter

LOGGER = logging.getLogger(__name__)
PLOTS_DIR = Path("data/output/plots")


def build_all(
    *,
    vent_lat: float,
    vent_lon: float,
    vent_elev: Optional[float] = None,
    base_dir: str | Path = "data/input",
    load_observations: bool = True,
    load_wind: bool = True,
    download_dem_data: bool = True,
    dem_buffer: float = 1.0,
    dem_dataset: str = "SRTMGL3",
    obs_params: Optional[Dict] = None,
    wind_params: Optional[Dict] = None,
    show_plots: bool = True,
    dem_path: Optional[str] = None,
) -> Tuple[Path, Path, Path]:
    """
    Return paths to (tephra2.conf, esp_input.csv, wind.txt).
    
    Args:
        vent_lat: Vent latitude
        vent_lon: Vent longitude
        vent_elev: Vent elevation (optional - if None, will be extracted from DEM)
        base_dir: Base directory for input/output files
        load_observations: Whether to load existing observation data
        load_wind: Whether to load existing wind data
        download_dem_data: Whether to download DEM data for the area
        dem_buffer: Buffer in degrees around vent for DEM download
        dem_dataset: DEM dataset to use (default: SRTMGL3 - 30m resolution SRTM)
        obs_params: Parameters for synthetic observation generation
        wind_params: Parameters for synthetic wind generation
        show_plots: Whether to display plots
        dem_path: Custom path to DEM file
        
    Returns:
        Paths to tephra2.conf, esp_input.csv, and wind.txt
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Download DEM data if requested and extract vent elevation if not provided
    if download_dem_data:
        if dem_path is None:
            dem_filename = "tephra2.tiff"
            dem_path = str(download_dem(
                vent_lat + dem_buffer, vent_lat - dem_buffer,
                vent_lon - dem_buffer, vent_lon + dem_buffer,
                output_dir=base_dir,
                filename=dem_filename,
                dataset=dem_dataset
            ))
            LOGGER.info(f"DEM downloaded to {dem_path}")
        
        # Extract vent elevation from DEM if not provided
        if vent_elev is None:
            vent_elev = get_elevation_at_point(dem_path, vent_lat, vent_lon)
            LOGGER.info(f"Vent elevation extracted from DEM: {vent_elev:.2f} m")

    # Ensure vent elevation is defined
    if vent_elev is None:
        LOGGER.warning("No vent elevation provided and DEM extraction disabled. Using 0 m.")
        vent_elev = 0.0

    obs_hdl = ObservationHandler(base_dir)
    if load_observations and (base_dir / "observations.csv").exists():
        obs_vec, sites = obs_hdl.load_observations()
    else:
        obs_vec, sites = obs_hdl.generate_synthetic(**(obs_params or {}))
        obs_hdl.save_observations(obs_vec, sites)

    # Vent UTM & lon/lat
    vent_east, vent_north, _ = latlon_to_utm(vent_lat, vent_lon)
    LOGGER.info(f"Vent location: ({vent_east:.2f}, {vent_north:.2f}), Elevation: {vent_elev:.2f} m")

    # ----------------  PLOTTERS  -------------------------------------- #
    obs_plot = ObservationPlotter(PLOTS_DIR)
    
    # Tephra distribution (UTM scatter)
    obs_plot.plot_tephra_distribution(
        eastings      = sites[:, 0],
        northings     = sites[:, 1],
        thicknesses   = obs_vec,
        vent_location = (vent_east, vent_north),
        title         = "Observed tephra distribution",
        save_path     = PLOTS_DIR / "observations.png",
        show_plot     = show_plots,
    )

    # Isomass map (lat/lon contours)
    zone_num, zone_letter = utm.from_latlon(vent_lat, vent_lon)[2:]
    lat_vals, lon_vals = utm.to_latlon(
        sites[:, 0], sites[:, 1], zone_num, zone_letter
    )
    obs_plot.plot_isomass_map(
        lon         = lon_vals, 
        lat         = lat_vals, 
        mass        = obs_vec,
        vent_lon    = vent_lon, 
        vent_lat    = vent_lat,
        dem_path    = dem_path or str(base_dir / "tephra2.tiff"),
        title       = "Tephra isomass contours",
        save_path   = PLOTS_DIR / "isomass_map.png",
        show_plot   = show_plots,
    )

    # ----------------  WIND  ----------------------------------------- #
    wind_hdl = WindDataHandler(base_dir)
    if load_wind and (base_dir / "wind.txt").exists():
        wind_df = wind_hdl.load_wind_data()
    else:
        wind_df = wind_hdl.generate_wind_data(**(wind_params or {}))
        wind_hdl.save_wind_data(wind_df)

    # Wind profile with speed-colored rose
    wp = WindPlotter(PLOTS_DIR)
    wp.plot_wind_profile(
        heights     = wind_df["HEIGHT"].to_numpy(),
        speeds      = wind_df["SPEED"].to_numpy(),
        directions  = wind_df["DIRECTION"].to_numpy(),
        title       = "Wind profile",
        save_path   = PLOTS_DIR / "wind_profile.png",
        show_plot   = show_plots,
    )

    # ----------------  CONF & ESP  ------------------------------------ #
    conf_path = write_tephra2_conf(vent_east, vent_north, vent_elev)
    esp_path  = write_esp_input(vent_east, vent_north, vent_elev)

    LOGGER.info("All inputs & plots written to %s", base_dir.resolve())
    return conf_path, esp_path, base_dir / "wind.txt"
