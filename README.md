# Tephra2 Inversion Framework

A comprehensive Python toolkit for volcanic tephra dispersion modeling using Tephra2 and estimating eruption source parameters (ESPs) via Markov Chain Monte Carlo (MCMC) inversion. This framework provides tools to:

1. Generate synthetic or process real tephra observation data
2. Download and process Digital Elevation Models (DEMs) using the Victor API (optional)
3. Build wind profiles from various sources or generate synthetic ones
4. Automatically generate all necessary Tephra2 input files (`tephra2.conf`, `esp_input.csv`, `wind.txt`)
5. Run Tephra2 forward models to simulate tephra deposition
6. Perform MCMC inversions to estimate ESPs like plume height and erupted mass
7. Generate publication-quality visualizations of inputs, outputs, and inversion results

## Installation & Setup

### Requirements

- Python 3.8+
- Tephra2 executable (compiled, see step 2)
- Victor API access and package installation for DEM downloads (optional)

### Installation

1. **Clone the repository**
   ```bash
   cd your/working/directory
   git clone https://github.com/jimhcyang/tephra_inversion.git
   cd tephra_inversion
   ```

2. **Set up Tephra2**

   Clone and compile the Tephra2 executable:

   * **Linux/Victor Environment:**
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

   * **macOS:**
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
   *Note: Ensure the compiled `tephra2` executable is accessible, potentially by adding its location to your system's PATH or placing it within the project structure (e.g., in a `bin/` directory).* 

3. **Set up Python environment**
   ```bash
   # Create and activate a virtual environment
   python -m venv tephra_env
   source tephra_env/bin/activate  # On Windows: tephra_env\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

4. **(Optional) Configure Victor for DEMs**
   If you intend to use DEM features, ensure the Victor package is installed and accessible. The `scripts/data_handling/dem_utils.py` script currently expects it to be available via `/home/jovyan/shared/Libraries/` but can be modified if needed.

## Repository Structure

```
├── README.md                 # This file
├── config/
│   └── default_config.py     # Default parameters for inversion runs
├── data/
│   ├── input/                # Default location for input files
│   │   ├── observations.csv  # Tephra thickness/mass observations
│   │   ├── sites.csv         # Sampling site coordinates (UTM)
│   │   └── wind.txt          # Tephra2-formatted wind profile
│   └── output/               # Default location for results
│       └── plots/            # Generated visualization images
├── example.ipynb             # Jupyter notebook demonstrating workflow
├── requirements.txt          # Python dependencies
├── scripts/
│   ├── core/
│   │   ├── mcmc.py           # MCMC sampler implementation
│   │   └── tephra2_utils.py  # Tephra2 execution and output parsing
│   ├── data_handling/
│   │   ├── build_inputs.py   # Main script to generate all inputs & plots
│   │   ├── coordinate_utils.py # UTM/Lat-Lon conversions
│   │   ├── dem_utils.py      # DEM download and elevation extraction (Victor)
│   │   ├── esp_config.py     # Tephra2 config file generation
│   │   ├── observation_data.py # Observation loading/generation
│   │   └── wind_data.py      # Wind data loading/generation
│   ├── tephra_inversion.py   # Main script for running MCMC inversions
│   └── visualization/
│       ├── diagnostic_plots.py # MCMC diagnostic plots (convergence, etc.)
│       ├── observation_plots.py # Observation/Isomass map plotting
│       └── wind_plots.py     # Wind profile plotting
└── tephra_inversion.log      # Log file for inversion runs (example)
```

### Configuration

- **config/default_config.py**: Contains default parameters for MCMC runs, Tephra2 execution paths, grid parameters, plotting settings, and default volcano information. Modify this file or provide a custom configuration via command-line arguments to tailor runs.

### Core Scripts

- **scripts/tephra_inversion.py**: The main command-line entry point for performing MCMC inversions. It handles:
  * Parsing command-line arguments (e.g., vent location, iterations, custom config).
  * Setting up logging.
  * Loading configurations.
  * Preparing input data using the `data_handling` modules.
  * Instantiating and running the MCMC sampler (`scripts/core/mcmc.py`).
  * Saving results and generating final diagnostic plots.

- **scripts/core/mcmc.py**: Implements the Metropolis-Hastings MCMC algorithm tailored for Tephra2 inversion. Key components include:
  * `MCMCRunner` class managing the sampling loop.
  * Parameter proposal distributions (e.g., Gaussian random walk).
  * Likelihood calculation based on misfit between Tephra2 output and observations (using `tephra2_utils`).
  * Acceptance/rejection logic.
  * Support for parallel execution of Tephra2 forward runs.
  * Checkpointing MCMC state.

- **scripts/core/tephra2_utils.py**: Provides functions to interact with the external Tephra2 executable:
  * `run_tephra2`: Executes the Tephra2 model with specified input files.
  * `parse_tephra2_output`: Reads and parses the output files generated by Tephra2 (e.g., `tephra2.csv`).
  * `calculate_misfit`: Computes the difference (e.g., RMS error) between simulated deposition and actual observations.

### Data Handling

- **scripts/data_handling/build_inputs.py**: Orchestrates the creation of all necessary input files for a Tephra2 run or inversion setup. It calls other `data_handling` modules to:
  * Load or generate observation data (`observation_data.py`).
  * Load or generate wind data (`wind_data.py`).
  * Download DEM and get elevation if needed (`dem_utils.py`).
  * Generate Tephra2 configuration files (`esp_config.py`).
  * It also generates initial diagnostic plots (wind profile, observation map, isomass map) using the `visualization` modules.

- **scripts/data_handling/coordinate_utils.py**: Contains helper functions for converting between Latitude/Longitude and UTM coordinates, essential for Tephra2 which operates in UTM.

- **scripts/data_handling/dem_utils.py**: Handles interactions with Digital Elevation Models. Requires the `victor` package.
  * `download_dem`: Downloads DEM data for a specified bounding box.
  * `get_elevation_at_point`: Extracts the elevation from a DEM file for given coordinates.
  * Includes checks for Victor availability, allowing graceful fallback.

- **scripts/data_handling/esp_config.py**: Functions to create the `tephra2.conf` and `esp_input.csv` files based on eruption source parameters (plume height, mass, particle size distribution, etc.) and grid settings.

- **scripts/data_handling/observation_data.py**: Manages tephra observation data:
  * Loading observations from standard file formats (`observations.csv`, `sites.csv`).
  * Generating synthetic observation datasets for testing.
  * Converting site coordinates to UTM.

- **scripts/data_handling/wind_data.py**: Manages atmospheric wind data:
  * Loading wind data from Tephra2-formatted files (`wind.txt`).
  * Generating synthetic wind profiles based on specified parameters.
  * Formatting data correctly for Tephra2.

### Visualization

- **scripts/visualization/diagnostic_plots.py**: Generates plots specifically for analyzing MCMC inversion results:
  * Parameter trace plots to assess chain mixing and convergence.
  * Posterior distribution histograms.
  * Joint distribution plots (scatter/contour) to visualize parameter correlations.

- **scripts/visualization/observation_plots.py**: Creates plots related to the tephra observations and model output:
  * `plot_tephra_distribution`: Scatter plot of observation sites coloured by deposit thickness/mass (UTM coordinates).
  * `plot_isomass_map`: Contour map showing lines of equal mass deposition (isomass lines) on a Latitude/Longitude grid, optionally overlayed on a DEM background.

- **scripts/visualization/wind_plots.py**: Generates plots to visualize the wind profile used in the simulation:
  * `plot_wind_profile`: Creates a multi-panel figure showing wind speed vs. altitude, wind direction vs. altitude, and a polar scatter plot (wind rose) showing direction vs. altitude, colored by speed.

## Example Usage

The primary way to explore the framework's capabilities is via the `example.ipynb` Jupyter notebook. It provides a step-by-step guide through a typical workflow, including data preparation, running a forward model, performing an MCMC inversion, and visualizing the results.

To run the notebook:

```bash
jupyter notebook example.ipynb
```

### Basic Command-Line Usage (Inversion)

The main script for running inversions is `scripts/tephra_inversion.py`.

1. **Run an inversion with default parameters (defined in `config/default_config.py`):**
   ```bash
   python scripts/tephra_inversion.py
   ```
   This will typically use the default volcano settings, load/generate data in the `data/input/` directory, run the MCMC sampler, and save results/plots to `data/output/`.

2. **Run with custom vent location and MCMC iterations:**
   ```bash
   python scripts/tephra_inversion.py --vent-lat 46.8 --vent-lon -121.7 --iterations 5000
   ```

3. **Run using a custom configuration file:**
   ```bash
   python scripts/tephra_inversion.py --config path/to/your_custom_config.py
   ```

4. **See all available command-line options:**
   ```bash
   python scripts/tephra_inversion.py --help
   ```

## Data Structure

The framework expects and generates data in the following default structure:

- **`data/input/`**: Contains input files required by Tephra2 and the inversion script.
  - `observations.csv`: A single column file containing tephra mass loading (kg/m²) or thickness (m) values corresponding to the sites.
  - `sites.csv`: A file with site coordinates, typically Easting and Northing (UTM). Ensure the coordinate system matches Tephra2 expectations.
  - `wind.txt`: A text file defining the wind profile. Each line should contain `HEIGHT (m) SPEED (m/s) DIRECTION (degrees from N)`. See Tephra2 documentation for details.
  - *(Optionally)* `tephra2.tiff`: Default filename for downloaded DEM data.
- **`data/output/`**: Stores results from model runs and analysis.
  - `plots/`: Directory where visualization scripts save generated figures (e.g., wind profiles, isomass maps, MCMC diagnostics).
  - *(Potentially)* MCMC chain files, best parameter files, etc., depending on configuration.

## Key Modules Explained

### Inversion Workflow

The typical MCMC inversion workflow orchestrated by `scripts/tephra_inversion.py` involves:

1. **Initialization**: Load configuration (default or custom), parse arguments.
2. **Data Preparation**: Use `scripts/data_handling/build_inputs.py` to ensure all necessary input data (`observations.csv`, `sites.csv`, `wind.txt`) is present or generated. This step might involve DEM download/processing if enabled and Victor is available.
3. **Tephra2 Setup**: Generate the `tephra2.conf` file based on grid settings in the configuration.
4. **MCMC Sampling**: Initialize the `MCMCRunner` from `scripts/core/mcmc.py`. The runner iteratively:
   a. Proposes new ESPs.
   b. Generates an `esp_input.csv` file.
   c. Runs Tephra2 using `scripts/core/tephra2_utils.py`.
   d. Parses Tephra2 output.
   e. Calculates the misfit/likelihood.
   f. Accepts or rejects the proposed ESPs based on the Metropolis-Hastings criterion.
   g. Logs the chain state.
5. **Post-processing**: After sampling, analyze the MCMC chain (e.g., calculate posterior means, credible intervals).
6. **Visualization**: Generate diagnostic plots using `scripts/visualization/diagnostic_plots.py`.

### DEM Download Rules

When downloading DEM data using the `dem_utils.py` module (which relies on the Victor API):

1. **API Limits**: Be mindful that external APIs like Victor often have usage limits. Repeatedly downloading the same area is inefficient.
2. **Caching**: The `download_dem` function includes basic caching. If a DEM file with the target filename already exists in the output directory, it will be reused unless `overwrite=True` is specified.
3. **Fallback**: If the Victor package is not available in the Python environment, `dem_utils.py` will raise an error *only if* `download_dem` is called. The `build_inputs.py` script handles this gracefully by checking `VICTOR_AVAILABLE` and skipping DEM download/processing steps if necessary, allowing the rest of the workflow to proceed without DEM features.
4. **Terms of Service**: Always respect the terms of service of the DEM data provider (e.g., SRTM, Copernicus DEM via Victor).

### MCMC Sampling Details

The MCMC implementation in `scripts/core/mcmc.py` features:

- **Metropolis-Hastings Algorithm**: The core sampling logic.
- **Parameter Space**: Typically configured to sample key ESPs like plume height and total erupted mass (often in log-space).
- **Proposal Distribution**: Uses a Gaussian random walk by default, potentially with adaptive step sizes to tune acceptance rates.
- **Likelihood Function**: Based on the misfit (e.g., RMS error) between Tephra2 model predictions and the observation data. Assumes Gaussian error structure.
- **Parallelism**: Can utilize multiple processor cores (`--parallel` argument) to run Tephra2 forward models concurrently for different proposed parameters within the MCMC loop, significantly speeding up the inversion.
- **Checkpointing**: Can periodically save the MCMC chain state, allowing resumption of interrupted runs.
- **Convergence Diagnostics**: While basic trace plots are generated, users may want to implement more formal convergence diagnostics (e.g., Gelman-Rubin statistic if running multiple chains).

## Advanced Usage

### Custom ESP Ranges & Priors

Modify the `param_config` dictionary within your configuration file (or directly in `scripts/tephra_inversion.py`) to change the prior ranges and proposal step sizes for sampled parameters.

Example snippet for `config/default_config.py`:

```python
PARAM_CONFIG = {
    "plume_height": {
        "range": (1000, 30000),  # Prior range (meters)
        "proposal_std": 500,    # Std dev for Gaussian proposal
        "initial_value": 15000
    },
    "log_mass": {
        "range": (np.log(1e9), np.log(1e13)), # Prior range (log kg)
        "proposal_std": 0.2,
        "initial_value": np.log(1e11)
    },
    # ... add other parameters like median_phi, sigma_phi if needed
}
```

### Running Tephra2 Forward Model Only

While the main script focuses on inversion, you can adapt `scripts/core/tephra2_utils.py` or use `scripts/data_handling/build_inputs.py` followed by a manual call to Tephra2 if you only need to run a single forward simulation:

1. **Prepare Inputs:** Use `build_inputs.py` logic or manually create `tephra2.conf`, `esp_input.csv`, `wind.txt`, `sites.csv`.
2. **Run Tephra2:**
   ```bash
   path/to/Tephra2/tephra2 path/to/tephra2.conf path/to/esp_input.csv path/to/wind.txt path/to/sites.csv
   ```
3. **Parse Output:** Use `parse_tephra2_output` from `tephra2_utils.py`.

## Development

Contributions to improve the framework are welcome!

1. Fork the repository.
2. Create a new branch for your feature or bug fix (`git checkout -b feature/my-new-feature`).
3. Make your changes and commit them (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/my-new-feature`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details (if one exists, otherwise assume standard MIT License).

## Acknowledgements

This project builds significantly upon the foundational work of **Qingyuan Yang**, who developed the original Metropolis-Hastings algorithm for Tephra2 inversion. His codebases ([`tephra2_mcmc`](https://github.com/yiqioyang/tephra2_mcmc) and [`mh_tephra2`](https://github.com/yiqioyang/mh_tephra2)) served as the core inspiration and basis for this framework, and his mentorship throughout the project was invaluable.

Special thanks are extended to **Professor Einat Lev** (Principal Investigator) and **Software Engineer Samuel Krasnoff** at Columbia University's Lamont-Doherty Earth Observatory (LDEO). Their close collaboration, guidance, and support during the Spring 2025 semester were instrumental in the development of this project.

This work was conducted as part of the **Columbia Data Science Institute Scholar Program**, and their support is gratefully acknowledged.

We also thank:

- The developers and maintainers of the Tephra2 model.
- The providers of the Victor API and the underlying DEM datasets (e.g., NASA SRTM, Copernicus DEM).
- Contributors to the open-source Python libraries used in this project (NumPy, Matplotlib, SciPy, Pandas, Rioxarray, Cartopy, UTM, etc.).