"""
DEM (Digital Elevation Model) utilities for Tephra2.

Provides functions to:
- Download DEM data for a given geographical area
- Extract elevation at specific coordinates

Note: This module requires the 'victor' package to be installed
for DEM downloads.
"""

from pathlib import Path
import logging
import os
from typing import Tuple, Optional, Union

# Core imports for DEM processing
import numpy as np
import pandas as pd
import rioxarray as rxr
import utm

# Visualization imports
import matplotlib.pyplot as plt
import matplotlib.tri as tri 
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from cartopy.io.img_tiles import Stamen
import seaborn as sns

# Direct import of Victor for DEM downloads
import victor

LOGGER = logging.getLogger(__name__)


def download_dem(
    lat_north: float,
    lat_south: float,
    lon_west: float, 
    lon_east: float,
    output_dir: Union[str, Path] = "data/input",
    filename: str = "tephra2.tiff",
    dataset: str = "SRTMGL3",
    format: str = "tiff"
) -> Path:
    """
    Download Digital Elevation Model (DEM) for the specified area.
    
    Note: Requires the 'victor' package to be installed.
    
    Args:
        lat_north: Northern latitude boundary
        lat_south: Southern latitude boundary  
        lon_west: Western longitude boundary
        lon_east: Eastern longitude boundary
        output_dir: Directory to save the DEM file
        filename: Output filename
        dataset: DEM dataset to use (default: SRTMGL3 - 30m resolution SRTM)
        format: Output format (default: tiff)
        
    Returns:
        Path to the downloaded DEM file
        
    Raises:
        ValueError: If coordinates are invalid
        RuntimeError: If DEM download fails
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / filename
    
    # If file already exists, return its path
    if output_path.exists():
        LOGGER.info(f"Using existing DEM file: {output_path}")
        return output_path
    
    LOGGER.info(f"Downloading DEM ({dataset}) for area: N={lat_north}, S={lat_south}, W={lon_west}, E={lon_east}")
    
    try:
        # Download the DEM using victor
        dem_file = victor.download_dem(
            lat_north, lat_south,
            lon_west, lon_east,
            format, dataset, 
            filename=str(output_path)
        )
        
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"DEM download appeared to succeed but file is missing or empty")
            
        LOGGER.info(f"DEM downloaded to: {output_path}")
        return output_path
        
    except Exception as e:
        LOGGER.error(f"DEM download failed: {e}")
        # Re-raise with more context
        raise RuntimeError(f"Failed to download DEM: {e}") from e


def get_elevation_at_point(
    dem_path: Union[str, Path],
    latitude: float,
    longitude: float
) -> float:
    """
    Extract elevation at specific coordinates from a DEM file.
    
    Args:
        dem_path: Path to the DEM file
        latitude: Point latitude
        longitude: Point longitude
        
    Returns:
        Elevation in meters
    """
    try:
        dem = rxr.open_rasterio(dem_path)
        elevation = float(dem.sel(x=longitude, y=latitude, method="nearest").values[0])
        return elevation
    except Exception as e:
        LOGGER.error(f"Error extracting elevation: {e}")
        return 0.0 