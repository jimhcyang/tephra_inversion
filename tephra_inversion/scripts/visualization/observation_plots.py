# scripts/visualization/observation_plots.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List, Union
import matplotlib.colors as mcolors

class ObservationPlotter:
    def __init__(self, output_dir: str = "data/output/plots"):
        """Initialize observation plotter with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_tephra_distribution(self,
                            eastings: np.ndarray,
                            northings: np.ndarray,
                            thicknesses: np.ndarray,
                            vent_location: Tuple[float, float],
                            title: str = "Tephra Deposit Distribution",
                            save_path: Optional[Union[str, Path]] = None,
                            show_plot: bool = True) -> str:
        """
        Plot tephra deposit thickness distribution.
        
        Parameters
        ----------
        eastings : np.ndarray
            Array of easting coordinates
        northings : np.ndarray
            Array of northing coordinates
        thicknesses : np.ndarray
            Array of deposit thicknesses (kg/m²)
        vent_location : Tuple[float, float]
            (easting, northing) coordinates of vent
        title : str
            Plot title
        save_path : Optional[str or Path]
            Path to save the plot. If None, uses default output directory
        show_plot : bool
            Whether to display the plot interactively (default: True)
            
        Returns
        -------
        str
            Path where the plot was saved
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create scatter plot with log-scaled colors
        norm = mcolors.LogNorm(vmin=max(thicknesses.min(), 0.001), vmax=thicknesses.max())
        scatter = ax.scatter(eastings, northings, c=thicknesses,
                        cmap='RdBu_r', norm=norm, alpha=0.7, s=30)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Deposit Thickness (kg/m²)')
        
        # Add vent location
        ax.plot(vent_location[0], vent_location[1], 'r^', markersize=12,
                label='Vent Location')
        
        # Add labels and title
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        # Set save path
        if save_path is None:
            save_path = self.output_dir / "tephra_distribution.png"
        else:
            save_path = Path(save_path)
            
        # Make sure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
            
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show the plot if requested (default is now True)
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        return str(save_path)