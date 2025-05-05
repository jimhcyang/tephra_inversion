"""
observation_data.py
~~~~~~~~~~~~~~~~~~~
Helper for *sites* + *observations*.

- Auto-detects comma vs whitespace in sites.csv.
- Synthesises random observations if files are missing.
- Plots scatter with log-colour.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


class ObservationHandler:
    default_params = {
        "n_points": 100,
        "seed": 42,
        "obs_min": 0.1,
        "obs_max": 1000.0,
    }

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    def load_observations(
        self,
        obs_file: str = "observations.csv",
        sites_file: str = "sites.csv",
    ) -> Tuple[np.ndarray, np.ndarray]:
        obs_path  = self.base_dir / obs_file
        sites_path = self.base_dir / sites_file
        if not obs_path.exists() or not sites_path.exists():
            raise FileNotFoundError("Missing observations or sites file.")
        observations = np.loadtxt(obs_path, dtype=float)
        # detect delimiter
        with open(sites_path) as fh:
            head = fh.readline()
        if "," in head:
            LOGGER.info("Using comma separator for sites.csv")
            df = pd.read_csv(sites_path, sep=",", header=None)
        else:
            LOGGER.info("Using whitespace separator for sites.csv")
            df = pd.read_csv(sites_path, sep=r"\s+", header=None, engine="python")
        sites = df.values.astype(float)
        LOGGER.info("Loaded %d observations", observations.size)
        return observations, sites

    def save_observations(
        self, observations: np.ndarray, sites: np.ndarray
    ) -> Tuple[Path, Path]:
        obs_path  = self.base_dir / "observations.csv"
        sites_path = self.base_dir / "sites.csv"
        np.savetxt(obs_path, observations, fmt="%.6f")
        np.savetxt(sites_path, sites, fmt="%.3f")
        LOGGER.info("Saved obs → %s and sites → %s", obs_path, sites_path)
        return obs_path, sites_path

    def generate_synthetic(
        self,
        n_points: int = 100,
        seed: int = 42,
        obs_min: float = 0.1,
        obs_max: float = 1000.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        obs = rng.uniform(obs_min, obs_max, n_points)
        e  = rng.uniform(-50_000, 50_000, n_points)
        n  = rng.uniform(-50_000, 50_000, n_points)
        sites = np.column_stack([e, n, np.full(n_points, 1000.0)])
        LOGGER.info("Generated %d synthetic observations", n_points)
        return obs, sites

    def plot_observations(
        self,
        observations: np.ndarray,
        sites: np.ndarray,
        *,
        output_path: Path | None = None,
        show_plot: bool = True,
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 8))
        norm = mcolors.LogNorm(vmin=observations.min(), vmax=observations.max())
        sc = ax.scatter(
            sites[:, 0], sites[:, 1],
            c=observations, cmap="RdBu_r", norm=norm, alpha=0.7
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Deposit thickness (log scale)")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_title("Tephra observations")
        ax.grid(True)
        fig.tight_layout()

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300)
            LOGGER.info("Observation plot saved → %s", output_path)
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
