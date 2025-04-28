# Tephra Inversion Framework

A Python framework for tephra inversion using Tephra2. This framework provides a streamlined workflow for estimating eruption source parameters (plume height and eruption mass) from tephra deposit data.

## Features

- Automatic UTM conversion from latitude/longitude coordinates
- Intelligent handling of existing data files (wind, observations, sites)
- Fallback strategy for wind data: existing file → API source → synthetic generation
- Metropolis-Hastings MCMC implementation focused on estimating plume height and eruption mass
- Real-time feedback during MCMC sampling with progress bars and parameter updates
- Comprehensive visualization tools for wind profiles, tephra deposits, and inversion results
- Interactive or scripted workflow options

## Installation

### 1. Clone the repository

```bash
cd your/working/directory
git clone https://github.com/jimhcyang/tephra_inversion.git
cd tephra_inversion
```

### 2. Set up Tephra2

Clone and compile the Tephra2 executable:

```bash
# Clone Tephra2 repository
git clone https://github.com/geoscience-community-codes/Tephra2.git

# Install required dependencies (if using conda, for example, on VICTOR)
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

or on Mac:

```bash
# Clone Tephra2 repository
git clone https://github.com/geoscience-community-codes/Tephra2.git

# Install required dependencies
brew install bdw-gc libatomic_ops

export C_INCLUDE_PATH=/opt/homebrew/include
export LIBRARY_PATH=/opt/homebrew/lib

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

## Primary Workflow: Example Notebook

The primary way to use this framework is through the `example.ipynb` notebook, which demonstrates a complete tephra inversion workflow:

1. **Setup**: Loading configuration, initializing volcano parameters
2. **Data Preparation**: Loading or generating wind data and observation data
3. **Forward Model**: Running Tephra2 with initial parameters
4. **Inversion**: MCMC sampling to estimate plume height and eruption mass
5. **Visualization**: Plotting parameter traces, distributions, and joint posteriors
6. **Analysis**: Statistical summary of posterior distributions

To run the example notebook:
```bash
jupyter notebook example.ipynb
```

### Example Code

Here's a simplified version of the workflow from the example notebook:

```python
import numpy as np
import matplotlib.pyplot as plt
from config.default_config import DEFAULT_CONFIG
from scripts.tephra_inversion import TephraInversion

# Initialize the inversion framework with Kirishima volcano
config = DEFAULT_CONFIG.copy()
config["volcano"]["default"] = "kirishima"

# Create inversion object
inversion = TephraInversion(config)

# Prepare input files
inversion.prepare_inputs()

# Run the inversion
results = inversion.run_inversion()

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(results["chain"]["plume_height"][1000:], alpha=0.7)
plt.axhline(results["best_params"]["plume_height"], color='r', ls='--')
plt.xlabel("Iteration"); plt.ylabel("Plume height (m)")
plt.title("Trace – plume height"); plt.grid(True); plt.show()

# Convert log mass to linear scale
mass_vals = np.exp(results["chain"]["log_mass"][1000:])
best_mass = np.exp(results["best_params"]["log_mass"])

plt.figure(figsize=(10, 6))
plt.hist(mass_vals, bins=30, alpha=0.7)
plt.axvline(best_mass, color='r', ls='--',
            label=f"Best: {best_mass: .2e} kg")
plt.xlabel("Eruption mass (kg)"); plt.xscale("log"); plt.legend(); plt.grid(True); plt.show()
```

## Alternative Usage: Scripted API

For more control, you can use the programmatic API directly:

```python
from scripts.tephra_inversion import TephraInversion
from config.default_config import DEFAULT_CONFIG

# Create inversion instance with custom configuration
config = DEFAULT_CONFIG.copy()
config["mcmc"]["n_iterations"] = 5000  # Customize MCMC settings
inversion = TephraInversion(config)

# Set vent location
inversion.setup_vent_location(lat=31.93, lon=130.93, elevation=1000)

# Set eruption time
inversion.setup_eruption_time(["2011", "01", "26", "15:00"])

# Set up wind data
inversion.setup_wind_data()

# Set up observation data
inversion.setup_observation_data()

# Run inversion
results = inversion.run_inversion()

# Access results
print(f"Plume Height: {results['best_params']['plume_height']:.1f} m")
print(f"Eruption Mass: {np.exp(results['best_params']['log_mass']):.2e} kg")

# Visualize results
from scripts.visualization.mcmc_plots import MCMCPlotter
plotter = MCMCPlotter()
plotter.summarize_mcmc_results(
    results["chain"],
    param_names=["plume_height", "log_mass"],
    burnin=1000
)
```

## Directory Structure

tephra_inversion/
├── config/
│ └── default_config.py # Default configuration
├── data/
│ ├── input/
│ │ ├── observations.csv # Tephra observations
│ │ ├── sites.csv # Site coordinates
│ │ └── wind.txt # Wind data
│ └── output/
│ ├── plots/ # Generated plots
│ └── mcmc/ # MCMC results
├── scripts/
│ ├── core/
│ │ ├── mcmc.py # MCMC implementation
│ │ ├── mcmc_utils.py # MCMC utilities
│ │ └── tephra2_interface.py # Tephra2 interface
│ ├── data_handling/
│ │ ├── coordinate_utils.py # Coordinate conversion
│ │ ├── esp_config.py # Eruption source parameters
│ │ ├── observation_data.py # Observation data handling
│ │ └── wind_data.py # Wind data handling
│ ├── tephra_inversion.py # Main workflow
│ └── visualization/
│ ├── diagnostic_plots.py # MCMC diagnostic plots
│ ├── mcmc_plots.py # MCMC result plots
│ ├── observation_plots.py # Observation plots
│ └── wind_plots.py # Wind data plots
├── Tephra2/
│ └── tephra2_2020 # Tephra2 executable (compiled)
├── example.ipynb # Example notebook
└── requirements.txt # Dependencies


## Requirements

- Python 3.8+
- NumPy, Pandas, Matplotlib, SciPy, Seaborn
- tqdm (for progress bars)
- utm (for coordinate conversion)
- Tephra2 executable (in Tephra2 directory)

See `requirements.txt` for full list of dependencies.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.