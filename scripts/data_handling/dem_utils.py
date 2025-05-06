"""
DEM (Digital Elevation Model) utilities for Tephra2.

Provides functions to:
- Download DEM data for a given geographical area
- Extract elevation at specific coordinates
"""

from pathlib import Path
import logging
from typing import Tuple, Optional, Union

import rioxarray as rxr
import numpy as np

# Check if victor is available (for DEM downloads)
try:
    import victor
    VICTOR_AVAILABLE = True
except ImportError:
    VICTOR_AVAILABLE = False
    logging.warning("Victor package not available; DEM downloads disabled.")

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
        ImportError: If victor package is not available
    """
    if not VICTOR_AVAILABLE:
        raise ImportError(
            "Victor package required for DEM downloads. "
            "Install with 'pip install victor-api'."
        )
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / filename
    
    # If file already exists, return its path
    if output_path.exists():
        LOGGER.info(f"Using existing DEM file: {output_path}")
        return output_path
    
    LOGGER.info(f"Downloading DEM ({dataset}) for area: N={lat_north}, S={lat_south}, W={lon_west}, E={lon_east}")
    
    # Download the DEM using victor
    dem_file = victor.download_dem(
        lat_north, lat_south,
        lon_west, lon_east,
        format, dataset, 
        filename=str(output_path)
    )
    
    LOGGER.info(f"DEM downloaded to: {output_path}")
    return output_path


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