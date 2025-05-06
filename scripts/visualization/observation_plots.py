"""
ObservationPlotter
------------------
• plot_tephra_distribution : UTM scatter, log‑coloured, vent triangle
• plot_isomass_map         : lon/lat contour map (solid + dashed), DEM
"""

from pathlib import Path
from typing  import Optional, Union, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.tri    as mtri
import rioxarray as rxr


class ObservationPlotter:
    def __init__(self, output_dir: str = "data/output/plots") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # UTM scatter                                                        #
    # ------------------------------------------------------------------ #
    def plot_tephra_distribution(
        self,
        eastings: np.ndarray,
        northings: np.ndarray,
        thicknesses: np.ndarray,
        vent_location: Tuple[float, float],
        title: str = "Tephra deposit distribution",
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = True,
    ) -> str:
        fig, ax = plt.subplots(figsize=(10, 8))

        vmin = max(thicknesses[thicknesses > 0].min(), 1e-3)
        vmax = thicknesses.max()

        sc = ax.scatter(
            eastings, northings,
            c=thicknesses,
            cmap="RdBu_r",
            norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
            s=45, alpha=0.75,
            edgecolors="k", linewidth=0.3,
        )
        cbar = plt.colorbar(sc, ax=ax, pad=0.01)
        cbar.set_label("Deposit thickness (kg/m²)")

        # Vent marker
        ax.plot(*vent_location, "^", ms=14, color="red",
                mec="black", mew=1.2, label="Vent")

        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_title(title)
        ax.legend()
        ax.grid(ls="--", alpha=0.4)

        # limits with 10% padding
        xpad = (eastings.max() - eastings.min()) * 0.1
        ypad = (northings.max() - northings.min()) * 0.1
        ax.set_xlim(eastings.min() - xpad, eastings.max() + xpad)
        ax.set_ylim(northings.min() - ypad, northings.max() + ypad)

        out = self._resolve(save_path, "tephra_distribution.png")
        plt.tight_layout()
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.show() if show_plot else plt.close(fig)
        return str(out)

    # ------------------------------------------------------------------ #
    # Isomass contour map                                                #
    # ------------------------------------------------------------------ #
    def plot_isomass_map(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        mass: np.ndarray,
        vent_lon: float,
        vent_lat: float,
        locations: Optional[Sequence[Tuple[float, float]]] = None,
        dem_path: str = "tephra2.tiff",
        title: str = "Isomass map",
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = True,
    ) -> str:
        """Plot tephra dispersion map with contour lines and vent location."""
        fig, ax = plt.subplots(figsize=(8, 8))

        # DEM background (optional)
        try:
            dem = rxr.open_rasterio(dem_path)
            dem.plot(ax=ax, cmap="gray_r", add_colorbar=False)
        except Exception as exc:
            print(f"[WARNING] Could not open '{dem_path}' for background map.")

        # Contours for solid and dashed lines
        lv_solid = [2 ** (2 * i + 1) for i in range(7)]   # 2, 8, 32, 128, 512, 2048, 8192
        lv_dashed = [2 ** (2 * i + 2) for i in range(6)]  # 4, 16, 64, 256, 1024, 4096
        
        # Create triangulation
        tri_obj = mtri.Triangulation(lon, lat)

        # Draw solid contours
        cs1 = ax.tricontour(tri_obj, mass, levels=lv_solid,
                            colors="k", linestyles="-", linewidths=1)
        ax.clabel(cs1, fmt="%2.1d", colors="w", fontsize=8)

        # Draw dashed contours
        cs2 = ax.tricontour(tri_obj, mass, levels=lv_dashed,
                            colors="k", linestyles="--", linewidths=0.5)
        ax.clabel(cs2, fmt="%2.1d", colors="k", fontsize=8)

        # Add points of interest if provided
        if locations:
            loc_lat = np.array(locations)[:, 0]
            loc_lon = np.array(locations)[:, 1]
            ax.scatter(loc_lon, loc_lat, marker="*", s=50, color="blue", label="POI")

        # Add vent marker
        ax.plot(vent_lon, vent_lat, "^", ms=10, color="red",
                mec="black", mew=1.0, label="Vent")

        # Zoom on non-zero mass with 10% padding
        threshold = 1
        mask = mass >= threshold
        if mask.any():
            lon_min, lon_max = lon[mask].min(), lon[mask].max()
            lat_min, lat_max = lat[mask].min(), lat[mask].max()
            lon_pad = (lon_max - lon_min) * 0.1
            lat_pad = (lat_max - lat_min) * 0.1
            ax.set_xlim(lon_min - lon_pad, lon_max + lon_pad)
            ax.set_ylim(lat_min - lat_pad, lat_max + lat_pad)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.grid(True, ls="--", alpha=0.3)

        out = self._resolve(save_path, "isomass_map.png")
        plt.tight_layout()
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.show() if show_plot else plt.close(fig)
        return str(out)

    # ------------------------------------------------------------------ #
    # helper                                                             #
    # ------------------------------------------------------------------ #
    def _resolve(self, path: Optional[Union[str, Path]], default: str) -> Path:
        if path is None:
            return self.output_dir / default
        return Path(path)
