# Tephra Inversion Framework

A Python framework for tephra inversion using Tephra2. This framework provides a streamlined workflow for estimating eruption source parameters (plume height and eruption mass) from tephra deposit data.

## Features

- Automatic UTM conversion from latitude/longitude coordinates
- Intelligent handling of existing data files (wind, observations, sites)
- Fallback strategy for wind data: existing file → API source → synthetic generation
- Simple MCMC implementation focused on estimating plume height and eruption mass
- Visualization tools for wind profiles, tephra deposits, and inversion results
- Interactive or scripted workflow options

## Installation

### 1. Clone the repository

```bash
cd your/working/directory
git clone https://github.com/jimhcyang/tephra_inversion.git
cd tephra_inversion/tephra_inversion
```

### 2. Set up Tephra2

Clone and compile the Tephra2 executable:

```bash
# Clone Tephra2 repository
git clone https://github.com/geoscience-community-codes/Tephra2.git

# Install required dependencies (if using conda)
conda install -y -c conda-forge boehm-gc libatomic_ops

# Set environment variables for compilation
export C_INCLUDE_PATH=$CONDA_PREFIX/include
export LIBRARY_PATH=$CONDA_PREFIX/lib

# Compile Tephra2
cd Tephra2
make clean
make

# Return to main directory
cd ..
```

### 3. Set up Python environment

```bash
# Create and activate a virtual environment
python -m venv tephra_env
source tephra_env/bin/activate  # On Windows: tephra_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Directory Structure

```
tephra_inversion/
├── config/
│   └── default_config.py        # Default configuration
├── data/
│   ├── input/
│   │   ├── observations.csv     # Tephra observations
│   │   ├── sites.csv            # Site coordinates
│   │   └── wind.txt             # Wind data
│   └── output/
│       ├── plots/               # Generated plots
│       └── mcmc/                # MCMC results
├── scripts/
│   ├── core/
│   │   ├── mcmc.py              # MCMC implementation
│   │   └── tephra2_interface.py # Tephra2 interface
│   ├── data_handling/
│   │   ├── coordinate_utils.py  # Coordinate conversion
│   │   ├── esp_config.py        # Eruption source parameters
│   │   ├── observation_data.py  # Observation data handling
│   │   └── wind_data.py         # Wind data handling
│   ├── tephra_inversion.py      # Main workflow
│   └── visualization/
│       ├── diagnostic_plots.py  # MCMC diagnostic plots
│       ├── observation_plots.py # Observation plots
│       └── wind_plots.py        # Wind data plots
├── Tephra2/
│   └── tephra2_2020             # Tephra2 executable (compiled)
├── demo.ipynb                   # Demo notebook
└── requirements.txt             # Dependencies
```

## Usage

The framework can be used in two ways:

### 1. Interactive Workflow

Run the framework interactively from the command line or a notebook:

```python
from scripts.tephra_inversion import TephraInversion

# Create inversion instance
inversion = TephraInversion()

# Run interactive workflow
results = inversion.run_workflow()
```

The framework will prompt for required inputs and check for existing data files.

### 2. Scripted Workflow

For more control, use the individual setup functions:

```python
from scripts.tephra_inversion import TephraInversion

# Create inversion instance
inversion = TephraInversion()

# Set vent location
inversion.setup_vent_location(lat=31.93, lon=130.93, elevation=1000)

# Set eruption time
inversion.setup_eruption_time(["2011", "01", "26", "15:00"])

# Set up wind data (checks for existing data first)
inversion.setup_wind_data()

# Set up observation data (checks for existing data first)
inversion.setup_observation_data()

# Run inversion
results = inversion.run_inversion()

# Access results
print(f"Plume Height: {results['best_params']['plume_height']:.1f} m")
print(f"Eruption Mass: {10 ** results['best_params']['log_m']:.2e} kg")
```

## Demo Notebook

A demo notebook (`demo.ipynb`) is provided to demonstrate the complete workflow:
1. Initializing the framework
2. Setting vent location and eruption time
3. Loading or generating wind data
4. Loading or generating observation data
5. Running parameter estimation
6. Analyzing and visualizing results

To run the demo:
```bash
jupyter notebook demo.ipynb
```

## Workflow Overview

1. **Vent Location**: Set the vent location using latitude and longitude. The framework automatically converts these to UTM coordinates.
2. **Eruption Time**: Set the eruption time in the format ["YYYY", "MM", "DD", "HH:MM"].
3. **Wind Data**: The framework checks for existing wind data in `data/input/wind.txt`. If not found, it tries to fetch from API sources or generates synthetic data.
4. **Observation Data**: The framework checks for existing observation data in `data/input/observations.csv` and `data/input/sites.csv`. If not found, it generates synthetic data.
5. **Parameter Estimation**: MCMC is used to estimate plume height and eruption mass by comparing model predictions to observations.
6. **Results Analysis**: The framework provides diagnostic plots and result summaries.

## Requirements

- Python 3.8+
- NumPy, Pandas, Matplotlib, SciPy
- utm (for coordinate conversion)
- Tephra2 executable (in Tephra2 directory)
- Additional compilation dependencies for Tephra2: boehm-gc, libatomic_ops

See `requirements.txt` for full list of dependencies.