"""
Minimal DEM helpers for Tephra‑2
--------------------------------
Every call to `import victor` is guarded, so merely importing this module
is safe even on systems where the Victor package is not installed.
"""

from pathlib import Path
import logging
import sys
from typing import Union

import rioxarray as rxr  # pip install rioxarray

# ----------------------------------------------------------------- Victor
sys.path.insert(0, "/home/jovyan/shared/Libraries/")
try:
    import victor  # noqa: E402
    VICTOR_AVAILABLE = True
except ImportError:        # Victor not installed
    VICTOR_AVAILABLE = False

LOGGER = logging.getLogger(__name__)


def download_dem(
    lat_north: float,
    lat_south: float,
    lon_west: float,
    lon_east: float,
    output_dir: str | Path = "data/input",
    filename: str = "tephra2.tiff",
    dataset: str = "COP30",
    file_format: str = "tiff",
    overwrite: bool = False,
) -> Path:
    """
    Wrapper around `victor.download_dem`.  Raises a RuntimeError if
    Victor is unavailable *and* someone attempts to call this function.
    """
    if not VICTOR_AVAILABLE:  # import is safe, call is not
        raise RuntimeError(
            "Victor package is not available – DEM download cannot proceed."
        )

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    if out_path.exists() and not overwrite:
        LOGGER.info("Using cached DEM at %s", out_path)
        return out_path

    victor.download_dem(
        lat_north,
        lat_south,
        lon_west,
        lon_east,
        file_format,
        dataset,
        filename=str(out_path),
    )
    return out_path


def get_elevation_at_point(dem_path: str | Path, lat: float, lon: float) -> float:
    """Nearest‑neighbour elevation lookup (metres)."""
    dem = rxr.open_rasterio(dem_path)
    return float(dem.sel(x=lon, y=lat, method="nearest").values[0])
