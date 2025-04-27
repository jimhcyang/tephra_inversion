import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.cm as cm

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
                               save_path: Optional[str] = None) -> None:
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
        save_path : Optional[str]
            Path to save the plot. If None, uses default output directory
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create scatter plot with log-scaled colors
        norm = mcolors.LogNorm(vmin=thicknesses.min(), vmax=thicknesses.max())
        scatter = ax.scatter(eastings, northings, c=thicknesses,
                           cmap='viridis', norm=norm, alpha=0.7)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Deposit Thickness (kg/m²)')
        
        # Add vent location
        ax.plot(vent_location[0], vent_location[1], 'r^', markersize=10,
                label='Vent Location')
        
        # Add labels and title
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        if save_path is None:
            save_path = self.output_dir / "tephra_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()  # Display the plot
        plt.close()
    
    def plot_tephra_map(self,
                       lats: np.ndarray,
                       lons: np.ndarray,
                       thicknesses: np.ndarray,
                       vent_lat: float,
                       vent_lon: float,
                       title: str = "Tephra Deposit Map",
                       save_path: Optional[str] = None) -> None:
        """
        Plot tephra deposit distribution on a map.
        
        Parameters
        ----------
        lats : np.ndarray
            Array of latitudes
        lons : np.ndarray
            Array of longitudes
        thicknesses : np.ndarray
            Array of deposit thicknesses (kg/m²)
        vent_lat : float
            Vent latitude
        vent_lon : float
            Vent longitude
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
        
        # Create scatter plot with log-scaled colors
        norm = mcolors.LogNorm(vmin=thicknesses.min(), vmax=thicknesses.max())
        scatter = ax.scatter(lons, lats, c=thicknesses,
                           cmap='viridis', norm=norm, alpha=0.7,
                           transform=ccrs.PlateCarree())
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Deposit Thickness (kg/m²)')
        
        # Add vent location
        ax.plot(vent_lon, vent_lat, 'r^', markersize=10,
                transform=ccrs.PlateCarree(), label='Vent Location')
        
        # Add labels and title
        ax.set_title(title)
        ax.legend()
        
        if save_path is None:
            save_path = self.output_dir / "tephra_map.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()  # Display the plot
        plt.close()
    
    def plot_thickness_histogram(self,
                               thicknesses: np.ndarray,
                               title: str = "Deposit Thickness Distribution",
                               save_path: Optional[str] = None) -> None:
        """
        Plot histogram of deposit thicknesses.
        
        Parameters
        ----------
        thicknesses : np.ndarray
            Array of deposit thicknesses (kg/m²)
        title : str
            Plot title
        save_path : Optional[str]
            Path to save the plot. If None, uses default output directory
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create histogram with log scale
        ax.hist(thicknesses, bins=50, log=True, alpha=0.7)
        
        # Add labels and title
        ax.set_xlabel('Deposit Thickness (kg/m²)')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.grid(True)
        
        if save_path is None:
            save_path = self.output_dir / "thickness_histogram.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()  # Display the plot
        plt.close()

def main():
    """Example usage of ObservationPlotter."""
    # Create plotter instance
    plotter = ObservationPlotter()
    
    # Example data
    n_points = 100
    eastings = np.random.normal(0, 10000, n_points)
    northings = np.random.normal(0, 10000, n_points)
    thicknesses = np.random.lognormal(0, 1, n_points)
    vent_location = (0, 0)
    
    # Plot tephra distribution
    plotter.plot_tephra_distribution(eastings, northings, thicknesses, vent_location)
    
    # Example map data
    lats = np.random.normal(31.93, 0.1, n_points)
    lons = np.random.normal(130.93, 0.1, n_points)
    vent_lat, vent_lon = 31.93, 130.93
    
    # Plot tephra map
    plotter.plot_tephra_map(lats, lons, thicknesses, vent_lat, vent_lon)
    
    # Plot thickness histogram
    plotter.plot_thickness_histogram(thicknesses)

if __name__ == "__main__":
    main()
