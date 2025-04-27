#!/usr/bin/env python3
"""
wind_utils.py

Utilities for handling wind data in the Tephra2 inversion workflow.
This module provides functions for loading, generating, and visualizing wind profiles.

Functions:
- read_wind_file: Read a wind file into a structured format
- write_wind_file: Write wind data to a file in Tephra2 format
- generate_wind_profile: Generate a synthetic wind profile
- fetch_reanalysis_wind: Fetch wind data from ERA5 reanalysis
- plot_wind_profile: Visualize wind data

Dependencies:
- numpy
- pandas
- matplotlib
- seaborn (for visualization)
- netCDF4, cdsapi (for reanalysis data, optional)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional, Union

# Set default style for all plots
sns.set_style("whitegrid")


def read_wind_file(wind_file_path: str) -> pd.DataFrame:
    """
    Read a wind file in Tephra2 format into a pandas DataFrame.
    
    Parameters
    ----------
    wind_file_path : str
        Path to the wind file
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: HEIGHT, SPEED, DIRECTION
        
    Raises
    ------
    FileNotFoundError
        If the wind file doesn't exist
    ValueError
        If the wind file has invalid format
    """
    try:
        # Try reading with pandas
        df = pd.read_csv(wind_file_path, sep=r'\s+', header=None, 
                         comment='#', names=['HEIGHT', 'SPEED', 'DIRECTION'])
        
        # Validate data
        if df.shape[1] != 3:
            raise ValueError(f"Wind file must have 3 columns, found {df.shape[1]}")
        
        # Ensure heights are in ascending order
        df = df.sort_values('HEIGHT').reset_index(drop=True)
        
        return df
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Wind file not found: {wind_file_path}")
    except Exception as e:
        raise ValueError(f"Failed to read wind file: {str(e)}")


def write_wind_file(wind_data: Union[pd.DataFrame, np.ndarray], 
                   output_path: str, 
                   header: bool = True) -> str:
    """
    Write wind data to a file in Tephra2 format.
    
    Parameters
    ----------
    wind_data : pd.DataFrame or np.ndarray
        Wind data with columns/dimensions: HEIGHT, SPEED, DIRECTION
    output_path : str
        Path where to save the wind file
    header : bool, optional
        Whether to include a header comment (default: True)
        
    Returns
    -------
    str
        Path to the created wind file
        
    Raises
    ------
    ValueError
        If wind_data has invalid format
    """
    try:
        # Convert numpy array to DataFrame if needed
        if isinstance(wind_data, np.ndarray):
            if wind_data.shape[1] != 3:
                raise ValueError(f"Wind data array must have 3 columns, found {wind_data.shape[1]}")
            wind_df = pd.DataFrame(wind_data, columns=['HEIGHT', 'SPEED', 'DIRECTION'])
        else:
            wind_df = wind_data
        
        # Ensure heights are in ascending order
        wind_df = wind_df.sort_values('HEIGHT').reset_index(drop=True)
        
        # Write to file
        with open(output_path, 'w') as f:
            if header:
                f.write("#HEIGHT SPEED DIRECTION\n")
            
            for _, row in wind_df.iterrows():
                f.write(f"{row['HEIGHT']:.6f} {row['SPEED']:.6f} {row['DIRECTION']:.6f}\n")
                
        return output_path
        
    except Exception as e:
        raise ValueError(f"Failed to write wind file: {str(e)}")


def generate_wind_profile(min_height: int = 1000, 
                         max_height: int = 40000, 
                         height_step: int = 1000,
                         wind_direction_mean: float = 180,
                         wind_direction_sd: float = 30,
                         wind_direction_gaussian: bool = True,
                         max_wind_speed: float = 50,
                         wind_speed_sd: float = 5,
                         elevation_max_speed: float = 15000,
                         zero_elevation: float = 40000,
                         seed: Optional[int] = None) -> pd.DataFrame:
    """
    Generate a synthetic wind profile with a piecewise linear speed profile.
    
    Parameters
    ----------
    min_height : int
        Minimum height in meters
    max_height : int
        Maximum height in meters
    height_step : int
        Height increment in meters
    wind_direction_mean : float
        Mean wind direction in degrees
    wind_direction_sd : float
        Standard deviation of wind direction in degrees
    wind_direction_gaussian : bool
        If True, use Gaussian distribution for direction,
        otherwise use uniform distribution
    max_wind_speed : float
        Maximum wind speed in m/s
    wind_speed_sd : float
        Standard deviation of wind speed in m/s
    elevation_max_speed : float
        Height at which wind speed reaches maximum
    zero_elevation : float
        Height at which wind speed returns to zero
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: HEIGHT, SPEED, DIRECTION
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Generate height levels
    heights = np.arange(min_height, max_height + 1, height_step)
    
    # Initialize arrays for speed and direction
    speeds = np.zeros_like(heights, dtype=float)
    directions = np.zeros_like(heights, dtype=float)
    
    # Generate wind speeds using piecewise linear model
    for i, height in enumerate(heights):
        # Phase 1: Linear increase up to max_speed at elevation_max_speed
        if height <= elevation_max_speed:
            speed = (max_wind_speed / elevation_max_speed) * height
        # Phase 2: Linear decrease to 0 at zero_elevation
        else:
            speed = max(0, max_wind_speed * (1 - (height - elevation_max_speed) / 
                                             (zero_elevation - elevation_max_speed)))
        
        # Add random perturbation
        speeds[i] = max(0, speed + np.random.normal(0, wind_speed_sd))
        
        # Generate wind direction
        if wind_direction_gaussian:
            directions[i] = np.random.normal(wind_direction_mean, wind_direction_sd) % 360
        else:
            directions[i] = np.random.uniform(wind_direction_mean - wind_direction_sd,
                                            wind_direction_mean + wind_direction_sd) % 360
    
    # Create DataFrame
    wind_df = pd.DataFrame({
        'HEIGHT': heights,
        'SPEED': speeds,
        'DIRECTION': directions
    })
    
    return wind_df


def plot_wind_profile(wind_data: Union[pd.DataFrame, str], 
                     figsize: Tuple[int, int] = (15, 5),
                     color: str = 'steelblue',
                     title: str = 'Wind Profile',
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot a wind profile showing speed vs. height, direction vs. height,
    and a polar plot.
    
    Parameters
    ----------
    wind_data : pd.DataFrame or str
        Wind data DataFrame or path to wind file
    figsize : tuple
        Figure size (width, height) in inches
    color : str
        Color for the plotted lines
    title : str
        Title for the overall figure
    save_path : str, optional
        If provided, save figure to this path
        
    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    # Load data if a string path is provided
    if isinstance(wind_data, str):
        wind_df = read_wind_file(wind_data)
    else:
        wind_df = wind_data
        
    # Sort by height for smooth lines
    wind_df = wind_df.sort_values('HEIGHT')
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Extract data
    heights = wind_df['HEIGHT'].values
    speeds = wind_df['SPEED'].values
    directions = wind_df['DIRECTION'].values
    
    # 1) Subplot: Wind Speed vs. Height
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(speeds, heights, marker='o', color=color, linewidth=2)
    ax1.set_title('Wind Speed vs. Height', fontsize=12)
    ax1.set_xlabel('Wind Speed (m/s)', fontsize=10)
    ax1.set_ylabel('Height (m)', fontsize=10)
    ax1.grid(True)
    
    # 2) Subplot: Wind Direction vs. Height
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(directions, heights, marker='o', color=color, linewidth=2)
    ax2.set_title('Wind Direction vs. Height', fontsize=12)
    ax2.set_xlabel('Wind Direction (°)', fontsize=10)
    ax2.set_ylabel('Height (m)', fontsize=10)
    ax2.set_xlim(0, 360)
    ax2.grid(True)
    
    # 3) Subplot: Polar plot
    ax3 = fig.add_subplot(1, 3, 3, projection='polar')
    theta_radians = np.radians(directions)
    ax3.scatter(theta_radians, heights, color=color, s=30)
    
    # Add connecting lines with increasing transparency by height
    colormap = plt.cm.get_cmap(color)
    for i in range(1, len(heights)):
        # Calculate transparency based on height
        alpha = 0.8 * (1 - heights[i] / heights.max())
        ax3.plot([theta_radians[i-1], theta_radians[i]], 
                [heights[i-1], heights[i]], 
                color=color, alpha=max(0.2, alpha))
    
    # Customize polar plot
    ax3.set_title('Polar Wind Profile', fontsize=12)
    ax3.set_theta_zero_location('N')  # 0° at North
    ax3.set_theta_direction(-1)       # Clockwise
    
    # Set radial ticks at 5-10 equally spaced heights
    max_height = heights.max()
    n_ticks = min(10, len(heights) // 3)
    r_ticks = np.linspace(0, max_height, n_ticks + 1)[1:]
    ax3.set_rticks(r_ticks)
    ax3.set_rlabel_position(112.5)  # Place labels at 112.5° (ESE)
    
    # Add colorbar for wind speed
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, 
                              norm=plt.Normalize(vmin=speeds.min(), vmax=speeds.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax3, pad=0.1)
    cbar.set_label('Wind Speed (m/s)')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def process_wind_data(wind_file: str, output_format: str = 'dataframe') -> Union[pd.DataFrame, Dict]:
    """
    Process wind data from a file and return in specified format.
    
    Parameters
    ----------
    wind_file : str
        Path to the wind file
    output_format : str
        'dataframe' or 'dict' for output format
        
    Returns
    -------
    pd.DataFrame or dict
        Wind data in requested format
    """
    # Read wind data
    wind_df = read_wind_file(wind_file)
    
    if output_format.lower() == 'dataframe':
        return wind_df
    
    elif output_format.lower() == 'dict':
        wind_dict = {
            'heights': wind_df['HEIGHT'].values,
            'speeds': wind_df['SPEED'].values,
            'directions': wind_df['DIRECTION'].values
        }
        return wind_dict
    
    else:
        raise ValueError(f"Unknown output format: {output_format}")


def parse_wind_text(wind_text: str) -> pd.DataFrame:
    """
    Parse wind data from a text string.
    
    Parameters
    ----------
    wind_text : str
        Text containing wind data in HEIGHT SPEED DIRECTION format
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: HEIGHT, SPEED, DIRECTION
    """
    lines = wind_text.strip().split('\n')
    heights = []
    speeds = []
    directions = []
    
    for line in lines:
        # Skip empty lines and comments
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # Parse values
        try:
            parts = line.split()
            if len(parts) >= 3:
                height = float(parts[0])
                speed = float(parts[1])
                direction = float(parts[2])
                
                heights.append(height)
                speeds.append(speed)
                directions.append(direction)
        except ValueError:
            # Skip lines that can't be parsed
            continue
    
    # Create DataFrame
    wind_df = pd.DataFrame({
        'HEIGHT': heights,
        'SPEED': speeds,
        'DIRECTION': directions
    }).sort_values('HEIGHT').reset_index(drop=True)
    
    return wind_df


def interpolate_wind_profile(wind_df: pd.DataFrame, 
                           target_heights: np.ndarray) -> pd.DataFrame:
    """
    Interpolate a wind profile to specific height levels.
    
    Parameters
    ----------
    wind_df : pd.DataFrame
        Wind data with HEIGHT, SPEED, DIRECTION columns
    target_heights : np.ndarray
        Array of heights to interpolate to
        
    Returns
    -------
    pd.DataFrame
        Interpolated wind profile
    """
    from scipy.interpolate import interp1d
    
    # Extract source data
    heights = wind_df['HEIGHT'].values
    speeds = wind_df['SPEED'].values
    
    # For directions, we need to handle the circular nature (0/360 degrees)
    sin_dirs = np.sin(np.radians(wind_df['DIRECTION'].values))
    cos_dirs = np.cos(np.radians(wind_df['DIRECTION'].values))
    
    # Create interpolation functions
    speed_interp = interp1d(heights, speeds, kind='linear', 
                           bounds_error=False, fill_value='extrapolate')
    sin_interp = interp1d(heights, sin_dirs, kind='linear', 
                         bounds_error=False, fill_value='extrapolate')
    cos_interp = interp1d(heights, cos_dirs, kind='linear', 
                         bounds_error=False, fill_value='extrapolate')
    
    # Interpolate values
    interp_speeds = speed_interp(target_heights)
    interp_sin = sin_interp(target_heights)
    interp_cos = cos_interp(target_heights)
    
    # Convert back to directions in degrees
    interp_dirs = (np.degrees(np.arctan2(interp_sin, interp_cos)) + 360) % 360
    
    # Create new DataFrame
    interp_df = pd.DataFrame({
        'HEIGHT': target_heights,
        'SPEED': interp_speeds,
        'DIRECTION': interp_dirs
    })
    
    return interp_df


def try_fetch_reanalysis_wind(lat: float, lon: float, 
                            date: List[str], 
                            output_file: str = "wind.txt") -> Optional[str]:
    """
    Try to fetch wind data from ERA5 reanalysis.
    Requires cdsapi and netCDF4 packages.
    
    Parameters
    ----------
    lat : float
        Latitude
    lon : float
        Longitude
    date : List[str]
        [year, month, day, hour] as strings
    output_file : str
        Path to save wind file
        
    Returns
    -------
    str or None
        Path to created wind file if successful, None if failed
    """
    try:
        import cdsapi
        import netCDF4
        
        # Check if required packages are available
        required_packages = ['cdsapi', 'netCDF4']
        for pkg in required_packages:
            if pkg not in globals():
                print(f"[WARNING] Package {pkg} not available for reanalysis")
                return None
        
        # Round coordinates to nearest 0.25 degrees
        north = round(lat*4)/4
        south = north + 0.1
        east = round(lon*4)/4
        west = east - 0.1
        
        # Extract date components
        years, months, days, hours = date
        
        # Pressure levels to gather
        pressures = [
            '1', '2', '3', '5', '7', '10', '20', '30', '50', '70', '100', '125', '150', '175',
            '200', '225', '250', '300', '350', '400', '450', '500', '550', '600', '650', '700',
            '750', '775', '800', '825', '850', '875', '900', '925', '950', '975', '1000'
        ]
        
        print(f"[INFO] Fetching ERA5 reanalysis data for {lat:.2f}°N, {lon:.2f}°E on {years}-{months}-{days} {hours}:00...")
        
        # Download data using CDS API
        c = cdsapi.Client()
        temp_file = "download_temp.nc"
        c.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type": ["reanalysis"],
                "data_format": "netcdf",
                "variable": [
                    "geopotential", "u_component_of_wind", "v_component_of_wind",
                ],
                "pressure_level": pressures,
                "year": years,
                "month": months,
                "day": days,
                "time": hours,
                "download_format": "unarchived",
                "area": [
                    north, west, south, east
                ],
            },
            temp_file
        )
        
        # Process the NetCDF file
        wind_nc = netCDF4.Dataset(temp_file)
        uwnd = wind_nc["u"][0, :, 0, 0]
        vwnd = wind_nc["v"][0, :, 0, 0]
        
        # Calculate speed and direction
        speed = np.sqrt(vwnd**2 + uwnd**2)
        direction = -180/np.pi * np.arctan2(vwnd, uwnd)
        
        # Adjust direction
        for d in range(len(direction)):
            if uwnd[d] > 0:
                direction[d] += 90
            else:
                direction[d] += 270
        
        # Convert geopotential to height
        heights = wind_nc["z"][0, :, 0, 0] / 9.80665
        
        # Reverse arrays to go from bottom to top
        speed = speed[::-1]
        direction = direction[::-1]
        heights = heights[::-1]
        
        # Create DataFrame
        wind_df = pd.DataFrame({
            'HEIGHT': heights,
            'SPEED': speed,
            'DIRECTION': direction
        })
        
        # Write to file
        write_wind_file(wind_df, output_file)
        
        # Clean up
        os.remove(temp_file)
        
        print(f"[INFO] Successfully fetched reanalysis wind data -> {output_file}")
        return output_file
        
    except ImportError:
        print("[ERROR] Missing required packages for reanalysis (cdsapi, netCDF4)")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to fetch reanalysis data: {str(e)}")
        return None


if __name__ == "__main__":
    # Example usage
    print("Wind utilities for Tephra2 inversion.")
    
    # Example 1: Generate a synthetic wind profile
    wind_df = generate_wind_profile(seed=42)
    print("\nGenerated wind profile:")
    print(wind_df.head())
    
    # Example 2: Write to file
    test_file = "example_wind.txt"
    write_wind_file(wind_df, test_file)
    print(f"\nWrote example wind file to: {test_file}")
    
    # Example 3: Read and plot
    loaded_df = read_wind_file(test_file)
    print("\nRead wind file:")
    print(loaded_df.head())
    
    fig = plot_wind_profile(loaded_df, title="Example Wind Profile")
    plt.show()
    
    # Clean up example file
    if os.path.exists(test_file):
        os.remove(test_file)