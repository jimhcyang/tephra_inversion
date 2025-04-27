# Tephra Inversion Framework

## Overview

This framework provides tools for performing Bayesian inversion of tephra deposit data using the Tephra2 forward model. Scientists can estimate eruption source parameters (ESPs) and their uncertainties from observed tephra deposits using Markov Chain Monte Carlo (MCMC) methods.

## Installation

### Prerequisites

- Python 3.8+
- NumPy, Pandas, Matplotlib, Seaborn
- SciPy
- tqdm
- UTM
- Tephra2 executable (`tephra2_2020`)

### Optional Dependencies

- statsmodels (for ACF/PACF plots)
- arviz (for effective sample size calculation)
- netCDF4, cdsapi (for ERA5 reanalysis wind data)

### Setup

```bash
# Clone the repository
git clone https://github.com/jimhcyang/tephra_inversion.git
cd tephra_inversion

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

Ensure that the `tephra2_2020` executable is available in the `tephra2` directory or update the path in your configuration.

## Directory Structure

```
tephra_inversion/
├── config/
│   └── default_config.yaml       # Default configuration settings
├── data/
│   ├── input/
│   │   ├── wind.txt              # Direct wind data input file
│   │   ├── esp_input.csv         # ESP parameters input file
│   │   └── observations.csv      # Observed tephra data
│   └── output/
│       ├── plots/                # Visualization outputs
│       └── mcmc/                 # MCMC run results
├── scripts/
│   ├── tephra2_interface.py      # Interface with Tephra2 model
│   ├── data_io.py                # Data loading and saving functions
│   ├── wind_utils.py             # Wind data handling utilities
│   ├── mcmc.py                   # Core MCMC implementation
│   ├── diagnostics.py            # Analysis of MCMC results
│   ├── visualization.py          # Plotting functions
│   ├── filtering.py              # Data filtering and sampling
│   └── lhs_search.py             # Latin Hypercube Sampling
├── templates/
│   ├── config_template.txt       # Template for Tephra2 config file
│   └── wind_template.txt         # Template for wind data file
├── demo.ipynb                    # Main demo notebook
└── README.md                     # This file
```

## Core Components

### Tephra2 Interface (`tephra2_interface.py`)
Interface to the Tephra2 forward model for configuration and execution.

### Wind Utilities (`wind_utils.py`)
Tools for wind data management, generation, and visualization.

### MCMC Implementation (`mcmc.py`)
Core Metropolis-Hastings algorithm for Bayesian inversion.

### Diagnostics (`diagnostics.py`)
Tools for analyzing MCMC results with statistical summaries and visualizations.

### Latin Hypercube Sampling (`lhs_search.py`)
Parameter space exploration to optimize MCMC starting points.

### Filtering (`filtering.py`)
Data preprocessing tools for tephra observations.

## Workflow

1. **Data Preparation**:
   - Provide tephra deposit observations
   - Specify a wind profile
   - Define initial parameter ranges

2. **Parameter Space Exploration** (optional):
   - Use Latin Hypercube Sampling
   - Visualize parameter sensitivities

3. **MCMC Inversion**:
   - Run Metropolis-Hastings algorithm
   - Sample from the posterior distribution

4. **Analysis and Visualization**:
   - Generate statistical summaries
   - Create plots and calculate credible intervals

## Input File Formats

### ESP Parameters (`esp_input.csv`)

CSV file with columns:
- `variable_name`: Parameter name
- `initial_val`: Initial value
- `prior_type`: Distribution type ("Gaussian", "Uniform", or "Fixed")
- `prior_para_a`, `prior_para_b`: Distribution parameters
- `draw_scale`: Standard deviation for proposal distribution

Example:
```
variable_name,initial_val,prior_type,prior_para_a,prior_para_b,draw_scale
column_height,7500,Gaussian,12500,1000,100
log_m,27.5,Gaussian,22.5,1,0.1
alpha,4,Gaussian,4,2,0.2
beta,2.0,Fixed,,,
```

### Wind Data (`wind.txt`)

Text file with space-separated columns:
- Height (m)
- Wind speed (m/s)
- Wind direction (degrees)

Example:
```
#HEIGHT SPEED DIRECTION
1000 10 150
2000 20 140
3000 25 130
4000 30 110
```

### Observations (`observations.csv`)

CSV file with one column containing tephra deposit measurements (kg/m²) at sampling locations.

## Demo Notebook

The `demo.ipynb` notebook provides a comprehensive guide to using the framework, including data preparation, MCMC inversion, and results visualization.

## Contributing



## License



## Acknowledgments



## Contact

For questions or support, please contact Jim Yang at [hy2867@columbia.edu](mailto:hy2867@columbia.edu).