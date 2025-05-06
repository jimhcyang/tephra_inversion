"""
Wind‑profile visualisation.

• plot_wind_profile : three‑panel figure
    (speed–altitude, direction–altitude, polar scatter coloured by speed).

Both PNGs land in the output directory you pass (default: data/output/plots).
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


class WindPlotter:
    """Utility for plotting wind diagnostics."""

    def __init__(self, output_dir: str = "data/output/plots") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  THREE‑IN‑ONE PROFILE                                              #
    # ------------------------------------------------------------------ #
    def plot_wind_profile(
        self,
        heights: np.ndarray,
        speeds: np.ndarray,
        directions: np.ndarray,
        title: str = "Wind profile",
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = True,
    ) -> str:
        """Speed, direction, and rose in a single PNG."""
        # Sort by altitude
        order = np.argsort(heights)
        h = heights[order]
        s = speeds[order]
        d = directions[order] % 360

        fig = plt.figure(figsize=(18, 6))

        # ------------ panel 1 (speed) ---------------------------------- #
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(s, h, "s-", color="steelblue", lw=2)
        ax1.set_xlabel("Wind speed (m/s)")
        ax1.set_ylabel("Altitude (m)")
        ax1.set_xlim(left=0)
        ax1.grid(True, ls="--", alpha=0.4)
        ax1.set_title("Speed vs altitude")
        _square_axes(ax1)

        # ------------ panel 2 (direction) ------------------------------ #
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(d, h, "s-", color="steelblue", lw=2)
        ax2.set_xlabel("Wind direction (deg)")
        ax2.set_ylabel("Altitude (m)")
        ax2.set_xlim(0, 360)
        ax2.set_xticks([0, 90, 180, 270, 360])
        ax2.grid(True, ls="--", alpha=0.4)
        ax2.set_title("Direction vs altitude")
        _square_axes(ax2)

        # ------------ panel 3 (polar rose scatter) ------------------- #
        ax3 = fig.add_subplot(1, 3, 3, projection="polar")

        theta = np.deg2rad(d)
        norm = mpl.colors.Normalize(vmin=s.min(), vmax=s.max())
        sc = ax3.scatter(theta, h, c=s, cmap=mpl.cm.Blues,
                         s=45, alpha=0.8, norm=norm, edgecolors="k", linewidth=0.3)
        cbar = plt.colorbar(sc, ax=ax3, pad=0.1, shrink=0.7)
        cbar.set_label("Wind speed (m/s)")

        ax3.set_theta_zero_location("N")
        ax3.set_theta_direction(-1)
        ax3.set_title("Polar scatter (dir vs alt, colored by speed)")

        # 10 concentric circles – but label only mid-radius and max
        max_r = float(h.max())
        circles = np.linspace(max_r / 10, max_r, 10)
        mid_r = circles[4]
        labels = [""] * len(circles)
        labels[4] = f"{int(mid_r)} m"
        labels[-1] = f"{int(max_r)} m"
        ax3.set_rgrids(circles, angle=60, labels=labels)

        # Save / show
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0.04, 1, 0.92])

        if save_path is None:
            save_path = self.output_dir / "wind_profile.png"
        else:
            save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return str(save_path)
    
# ---------------------------------------------------------------------- #
#  Helper – make subplot square (mpl < 3.4 fallback)                     #
# ---------------------------------------------------------------------- #
def _square_axes(ax: mpl.axes.Axes) -> None:
    """Force a square drawing box without distorting data."""
    try:                           # mpl ≥ 3.4
        ax.set_box_aspect(1)
    except AttributeError:
        # fallback: let matplotlib pick aspect automatically
        ax.set_aspect("auto")
