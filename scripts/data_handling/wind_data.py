"""wind_data.py
~~~~~~~~~~~~
Helper for *wind profile* handling:

- Loads whitespace-separated files, skipping commented lines.
- Synthesises a Gaussian/linear profile when missing.
- Optional 3-panel diagnostic plot.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


class WindDataHandler:
    default_params: Dict = {
        "direction_mean": 180.0,
        "direction_sd": 30.0,
        "max_speed": 50.0,
        "elevation_max_speed": 15000.0,
        "zero_elevation": 40000.0,
        "n_levels": 50,
        "seed": 42,
    }

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    def load_wind_data(
        self,
        filename: str = "wind.txt"
    ) -> pd.DataFrame:
        path = self.base_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Wind file not found: {filename}")
        # Skip commented lines starting with '#'
        df = pd.read_csv(
            path,
            sep=r"\s+",
            comment="#",
            header=None,
            names=["HEIGHT", "SPEED", "DIRECTION"],
            engine="python",
            dtype=float,
        )
        LOGGER.info("Loaded wind profile from %s", path)
        return df

    def save_wind_data(
        self, df: pd.DataFrame, filename: str = "wind.txt"
    ) -> Path:
        path = self.base_dir / filename
        df.to_csv(path, sep=" ", header=False, index=False, float_format="%.6f")
        LOGGER.info("Wind profile saved → %s", path)
        return path

    def generate_wind_data(
        self,
        *,
        direction_mean: float = 180.0,
        direction_sd: float = 30.0,
        max_speed: float = 50.0,
        elevation_max_speed: float = 15000.0,
        zero_elevation: float = 40000.0,
        n_levels: int = 50,
        seed: int = 42,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        heights = np.linspace(1000, 50000, n_levels)
        speeds = np.where(
            heights <= elevation_max_speed,
            max_speed * heights / elevation_max_speed,
            max_speed
            * np.clip(
                (zero_elevation - heights) / (zero_elevation - elevation_max_speed),
                0,
                1,
            ),
        )
        speeds += rng.normal(0, max_speed * 0.05, n_levels)
        speeds = np.clip(speeds, 0, None)

        directions = np.mod(rng.normal(direction_mean, direction_sd, n_levels), 360)

        df = pd.DataFrame({"HEIGHT": heights, "SPEED": speeds, "DIRECTION": directions})
        LOGGER.info("Generated synthetic wind profile (%d levels)", n_levels)
        return df

    def plot_wind_profile(self, df: pd.DataFrame) -> None:
        """Optional: local 3-panel diagnostic."""
        import matplotlib.pyplot as plt

        elev = df["HEIGHT"].to_numpy()
        spd = df["SPEED"].to_numpy()
        dirc = df["DIRECTION"].to_numpy()

        order = np.argsort(elev)
        elev, spd, dirc = elev[order], spd[order], dirc[order]

        fig = plt.figure(figsize=(15, 5))
        # speed vs altitude
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(elev, spd, "-o")
        ax1.set_xlabel("Altitude (m)")
        ax1.set_ylabel("Speed (m/s)")
        ax1.set_title("Speed vs altitude")

        # direction vs altitude
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(elev, dirc, "-o")
        ax2.set_xlabel("Altitude (m)")
        ax2.set_ylabel("Direction (°)")
        ax2.set_title("Direction vs altitude")

        # polar
        ax3 = fig.add_subplot(1, 3, 3, projection="polar")
        ax3.plot(np.radians(dirc), elev, "-o")
        ax3.set_theta_zero_location("N")
        ax3.set_theta_direction(-1)
        ax3.set_title("Polar")

        fig.tight_layout()
        plt.show()
