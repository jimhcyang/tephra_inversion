"""
Glue the four data-handling helpers together.

Given:
  data/input/observations.csv   (1-col mass-loading, kg/m²)
  data/input/sites.csv          (E,N,Elev; comma-sep)
  data/input/wind.txt           (#HEIGHT SPEED DIRECTION)

produce:
  data/input/tephra2.conf
  data/input/esp_input.csv
"""

from pathlib import Path
import numpy as np

from .coordinate_utils   import latlon_to_utm
from .observation_data   import ObservationHandler
from .wind_data          import WindDataHandler
from .esp_config         import write_tephra2_conf, write_esp_input


def build_all(vent_lat: float,
              vent_lon: float,
              vent_elev: float,
              plume_height: float = 10000,
              eruption_mass: float = 2.5e10,  # kg
              base_dir: str = "data/input"):
    base = Path(base_dir)

    # 1. Verify the three mandatory plain-text inputs exist
    for fn in ("observations.csv", "sites.csv", "wind.txt"):
        if not (base / fn).exists():
            raise FileNotFoundError(f"Missing {fn} under {base}")

    # 2. VENT → UTM
    easting, northing, _ = latlon_to_utm(vent_lat, vent_lon, vent_elev)

    # 3. Write tephra2.conf (19 rows)
    write_tephra2_conf(easting, northing, vent_elev,
                       plume_height, eruption_mass)

    # 4. Natural-log mass (base-e!)
    log_m_e = np.log(eruption_mass)

    # 5. Write esp_input.csv
    write_esp_input(plume_height, log_m_e)

    print("[INFO] Input bundle ready.")