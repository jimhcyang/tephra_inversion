# scripts/visualization/wind_plots.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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
                         save_path: Optional[str] = None) -> None:
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
        save_path : Optional[str]
            Path to save the plot. If None, uses default output directory
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
        
        # Show the plot
        plt.show()
        
        # Save the plot
        if save_path is None:
            save_path = self.output_dir / "wind_profile.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_wind_field(self,
                       lats: np.ndarray,
                       lons: np.ndarray,
                       u_components: np.ndarray,
                       v_components: np.ndarray,
                       title: str = "Wind Field",
                       save_path: Optional[str] = None) -> None:
        """
        Plot wind field using quiver plot.
        
        Parameters
        ----------
        lats : np.ndarray
            Array of latitudes
        lons : np.ndarray
            Array of longitudes
        u_components : np.ndarray
            U components of wind vectors
        v_components : np.ndarray
            V components of wind vectors
        title : str
            Plot title
        save_path : Optional[str]
            Path to save the plot. If None, uses default output directory
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, color='lightgray')
        ax.add_feature(cfeature.OCEAN, color='lightblue')
        
        # Plot wind vectors
        q = ax.quiver(lons, lats, u_components, v_components,
                     transform=ccrs.PlateCarree(),
                     scale=100, color='red')
        
        # Add quiver key
        ax.quiverkey(q, 0.9, 0.1, 20, '20 m/s',
                    labelpos='E', coordinates='figure')
        
        ax.set_title(title)
        
        if save_path is None:
            save_path = self.output_dir / "wind_field.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_wind_rose(self,
                      directions: np.ndarray,
                      speeds: np.ndarray,
                      heights: np.ndarray,
                      title: str = "Wind Rose",
                      save_path: Optional[str] = None) -> None:
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
        save_path : Optional[str]
            Path to save the plot. If None, uses default output directory
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
        
        # Show the plot
        plt.show()
        
        # Save the plot
        if save_path is None:
            save_path = self.output_dir / "wind_rose.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Example usage of WindPlotter."""
    # Create plotter instance
    plotter = WindPlotter()
    
    # Example wind data
    heights = np.linspace(0, 10000, 11)
    speeds = np.linspace(0, 30, 11)
    directions = np.linspace(0, 360, 11)
    
    # Plot wind profile
    plotter.plot_wind_profile(heights, speeds, directions)
    
    # Example wind field data
    lats = np.linspace(30, 35, 10)
    lons = np.linspace(130, 135, 10)
    u = np.random.normal(0, 10, (10, 10))
    v = np.random.normal(0, 10, (10, 10))
    
    # Plot wind field
    plotter.plot_wind_field(lats, lons, u, v)
    
    # Plot wind rose
    plotter.plot_wind_rose(directions, speeds, heights)

if __name__ == "__main__":
    main()