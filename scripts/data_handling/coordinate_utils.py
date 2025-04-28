# scripts/data_handling/coordinate_utils.py

import utm
from typing import Tuple

def latlon_to_utm(lat: float, lon: float, elevation: float = 0.0) -> Tuple[float, float, float]:
    """
    Convert latitude and longitude to UTM coordinates.
    
    Parameters
    ----------
    lat : float
        Latitude in decimal degrees
    lon : float
        Longitude in decimal degrees
    elevation : float, optional
        Elevation in meters, defaults to 0.0
        
    Returns
    -------
    Tuple[float, float, float]
        (easting, northing, elevation) in meters
    """
    # Convert to UTM
    easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
    
    return easting, northing, elevation

def utm_to_latlon(easting: float, northing: float, zone_number: int, zone_letter: str) -> Tuple[float, float]:
    """
    Convert UTM coordinates back to latitude and longitude.
    
    Parameters
    ----------
    easting : float
        UTM easting coordinate in meters
    northing : float
        UTM northing coordinate in meters
    zone_number : int
        UTM zone number
    zone_letter : str
        UTM zone letter
        
    Returns
    -------
    Tuple[float, float]
        (latitude, longitude) in decimal degrees
    """
    # Convert to lat/lon
    lat, lon = utm.to_latlon(easting, northing, zone_number, zone_letter)
    
    return lat, lon