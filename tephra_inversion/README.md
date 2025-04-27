# Tephra Inversion Framework

A Python framework for tephra inversion using Tephra2.

## Installation

1. Create a virtual environment:
```bash
python -m venv tephra_env
source tephra_env/bin/activate  # On Windows: tephra_env\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Wind Data

The framework provides utilities for handling wind data:

1. Fetching wind data from ERA5 reanalysis:
```python
from tephra_inversion.scripts.utils import wind_utils

# Fetch wind data
wind_path = wind_utils.try_fetch_reanalysis_wind(
    lat=40.0,  # Vent latitude
    lon=-120.0,  # Vent longitude
    date='2020-01-01',  # Eruption date
    output_file='output/wind.txt'
)

# Read wind data
wind_data = wind_utils.read_wind_file(wind_path)

# Plot wind profile
wind_utils.plot_wind_profile(wind_data, save_path='output/plots/wind_profile.png')
```

2. Generating synthetic wind profiles:
```python
# Generate default wind profile
wind_data = wind_utils.generate_wind_profile()

# Write to file
wind_path = wind_utils.write_wind_file(wind_data, 'output/wind.txt')
```

### Demo Notebook

A demo notebook is provided to demonstrate the complete workflow:
1. Getting wind data
2. Loading observation data
3. Running parameter estimation
4. Analyzing results

To run the demo:
```bash
jupyter notebook demo.ipynb
```

## Directory Structure

```
tephra_inversion/
├── config/
│   └── default_config.yaml       # Default configuration
├── data/
│   ├── input/
│   │   ├── wind.txt             # Wind data
│   │   ├── observations.csv     # Tephra observations
│   │   └── sites.csv           # Site coordinates
│   └── output/
│       ├── plots/              # Generated plots
│       └── mcmc/              # MCMC results
├── scripts/
│   ├── data_handling/
│   │   ├── wind_data.py       # Wind data generation/download
│   │   ├── observation_data.py # Observation data generation
│   │   └── parameter_input.py  # Parameter input handling
│   ├── core/
│   │   ├── tephra2_interface.py # Tephra2 model interface
│   │   ├── mcmc.py            # MCMC implementation
│   │   └── lhs.py             # Latin Hypercube Sampling
│   └── visualization/
│       ├── wind_plots.py      # Wind data visualization
│       ├── observation_plots.py # Observation plots
│       └── diagnostic_plots.py # MCMC diagnostics
├── tephra2/
│   └── tephra2_2020       # Tephra2 Executable
├── demo.ipynb                    # Main demo script
└── requirements.txt           # Dependencies
└── README.md                     # This file
```

## Requirements

- Python 3.8+
- See requirements.txt for package dependencies

## License

MIT License