# scripts/visualization/wind_plots.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List, Union

class WindPlotter:
    def __init__(self, output_dir: str = "data/output/plots"):
        """Initialize wind plotter with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_wind_profile(self, 
                         heights: np.ndarray,
                         speeds: np.ndarray,
                         directions: np.ndarray,
                         title: str = "Wind Profile",
                         save_path: Optional[Union[str, Path]] = None,
                         show_plot: bool = False) -> str:
        """
        Plot wind speed and direction profiles.
        
        Parameters
        ----------
        heights : np.ndarray
            Array of heights in meters
        speeds : np.ndarray
            Array of wind speeds in m/s
        directions : np.ndarray
            Array of wind directions in degrees
        title : str
            Plot title
        save_path : Optional[str or Path]
            Path to save the plot. If None, uses default output directory
        show_plot : bool
            Whether to display the plot interactively
            
        Returns
        -------
        str
            Path where the plot was saved
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Wind speed profile
        ax1.plot(speeds, heights, color='skyblue', linewidth=2, label='Wind Speed')
        ax1.set_xlabel('Wind Speed (m/s)')
        ax1.set_ylabel('Height (m)')
        ax1.set_title('Wind Speed Profile')
        ax1.grid(True)
        ax1.legend()
        
        # Wind direction profile
        ax2.plot(directions, heights, color='skyblue', linewidth=2, label='Wind Direction')
        ax2.set_xlabel('Wind Direction (degrees)')
        ax2.set_ylabel('Height (m)')
        ax2.set_title('Wind Direction Profile')
        ax2.grid(True)
        ax2.legend()
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Set save path
        if save_path is None:
            save_path = self.output_dir / "wind_profile.png"
        else:
            save_path = Path(save_path)
            
        # Make sure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show the plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        return str(save_path)
    
    def plot_wind_rose(self,
                      directions: np.ndarray,
                      speeds: np.ndarray,
                      heights: np.ndarray,
                      title: str = "Wind Rose",
                      save_path: Optional[Union[str, Path]] = None,
                      show_plot: bool = False) -> str:
        """
        Plot wind rose diagram with elevation on radial axis and speed as color.
        
        Parameters
        ----------
        directions : np.ndarray
            Array of wind directions in degrees
        speeds : np.ndarray
            Array of wind speeds in m/s
        heights : np.ndarray
            Array of heights in meters
        title : str
            Plot title
        save_path : Optional[str or Path]
            Path to save the plot. If None, uses default output directory
        show_plot : bool
            Whether to display the plot interactively
            
        Returns
        -------
        str
            Path where the plot was saved
        """
        # Convert directions to radians
        directions_rad = np.radians(directions)
        
        # Create figure
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='polar')
        
        # Create color map based on wind speeds
        norm = plt.Normalize(speeds.min(), speeds.max())
        cmap = plt.cm.Blues
        
        # Plot wind directions and heights with speed-based coloring
        scatter = ax.scatter(directions_rad, heights, c=speeds, cmap=cmap,
                           s=50, alpha=0.7, norm=norm)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Wind Speed (m/s)')
        
        # Customize plot
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_title(title)
        
        # Set save path
        if save_path is None:
            save_path = self.output_dir / "wind_rose.png"
        else:
            save_path = Path(save_path)
            
        # Make sure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show the plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        return str(save_path)