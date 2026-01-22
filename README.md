# Tephra2 Inversion Framework (VICTOR-style workflow)

A compact, reproducible workflow for **estimating eruption source parameters (ESPs)** for tephra dispersion using **Tephra2** as the forward model and multiple inversion backends:

- **MCMC** (Metropolis–Hastings)
- **SA** (Simulated Annealing)
- **PSO** (Particle Swarm Optimization)
- **ES** (Ensemble Smoother / ES-MDA-style updates)

The repo supports:

- A **scenario system** (each scenario has its own inputs + truth metadata)
- A **simulation grid runner** (priors × hyperparameters × repeats × methods)
- A **postprocessing pipeline** that crawls experiment outputs and produces:
  - 7×7 prior-grid **accuracy heatmaps**
  - per-scenario **interactive scatter** (hover highlights trajectories)
  - **trace + marginal** diagnostics for available runs

---

## Quick Start

### 1) Requirements

- Python **3.9+**
- A compiled **Tephra2** executable available on your PATH *or* referenced in config
- macOS: Homebrew recommended
- Linux/HPC: Conda recommended

> Optional: Victor platform access (if you’re running on VICTOR)

---

## Install

```bash
git clone https://github.com/jimhcyang/tephra_inversion.git
cd tephra_inversion

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
````

---

## Build Tephra2

### macOS

```bash
git clone https://github.com/geoscience-community-codes/Tephra2.git
brew install bdw-gc libatomic_ops

export C_INCLUDE_PATH=/opt/homebrew/include
export LIBRARY_PATH=/opt/homebrew/lib

cd Tephra2
make clean
make
cd ..
```

### Linux / Conda (e.g., Victor/HPC)

```bash
git clone https://github.com/geoscience-community-codes/Tephra2.git
conda install -y -c conda-forge boehm-gc libatomic_ops

export C_INCLUDE_PATH=$CONDA_PREFIX/include
export LIBRARY_PATH=$CONDA_PREFIX/lib

cd Tephra2
make clean
make
cd ..
```

Make sure the `tephra2` binary is accessible. Two common options:

* Put it in `./bin/` and reference it in config
* Add it to your PATH

---

# The New Workflow (Multi-Scenario, Multi-Experiment)

This repo separates:

1. **Scenarios** (inputs + ground truth)
2. **Experiments** (simulation output grids for each scenario)
3. **Postprocessing** (plots + summaries computed from experiment logs/chains)

---

## Directory Layout (Current)

```text
tephra_inversion/
├── demo.ipynb
├── requirements.txt
├── config/
│   └── default_config.py
├── scripts/
│   ├── tephra_inversion.py
│   ├── data_handling/
│   │   └── config_io.py
│   ├── sim/
│   │   ├── setup.py
│   │   ├── simulate.py
│   │   ├── exp_config.py
│   │   ├── exp_config_test.py
│   │   ├── results_io.py
│   │   ├── metrics_heatmaps.py
│   │   ├── interactive_scatter.py
│   │   └── simple_plots.py
│   └── visualization/
│       ├── postprocess_sim_results.py
│       └── wind_plots.py
└── data/
    ├── scenarios/
    │   ├── cn92_a/
    │   │   ├── input/
    │   │   └── config/sim_meta.json
    │   └── ...
    └── experiments/
        ├── cn92_a/
        │   ├── results_mcmc.csv
        │   ├── chains/mcmc/mcmc_run1.csv
        │   ├── output_heatmaps/
        │   ├── output_simple/
        │   └── interactive_scatter_all_models_hover.html
        └── ...
```

> Each scenario produces outputs into its *own* folder under `data/experiments/<scenario>/`.

---

# Step-by-Step: Run a Full Experiment

## Step 0 — Choose / Create a scenario

A **scenario** is a folder under `data/scenarios/<scenario_name>/` containing:

* `input/` (Tephra2-ready input files + observations)
* `config/sim_meta.json` (truth metadata used for plotting errors)

### Example: Build a synthetic Cerro Negro dataset (one-time)

This creates the `data/scenarios/<scenario>/` tree from an aggregated CSV.

```bash
python -m scripts.sim.setup \
  data_std/cn_std_agg.csv \
  --out-root data/scenarios \
  --scenario cn92_a \
  --vent-easting 532400 \
  --vent-northing 1382525 \
  --vent-elev 100 \
  --plume-height 7500 \
  --eruption-mass 4.288342e10
```

This will generate:

* `data/scenarios/cn92_a/input/`
* `data/scenarios/cn92_a/config/sim_meta.json`

> `sim_meta.json` is what postprocessing uses for “true_plume” and “true_mass”.

---

## Step 1 — Run the simulation grid for that scenario

The simulation grid loops over:

* **PRIOR_FACTORS** (7×7 prior centers for height & mass)
* **method hyperparameter configs** (config index 0..K per method)
* **N_REPEATS** (repeated stochastic runs per grid cell)
* **MODELS** = (`mcmc`, `sa`, `pso`, `es`)

### Run with the main config

```bash
python -m scripts.sim.simulate \
  --config-module scripts.sim.exp_config \
  --input-dir data/scenarios/cn92_a/input \
  --output-dir data/experiments/cn92_a \
  --plot-winds
```

### Run with the smaller test config

```bash
python -m scripts.sim.simulate \
  --config-module scripts.sim.exp_config_test \
  --input-dir data/scenarios/cn92_a/input \
  --output-dir data/experiments/cn92_a_test \
  --plot-winds
```

Outputs (per scenario experiment folder):

* `results_<model>.csv` (one row per run)
* `chains/<model>/<model>_run<id>.csv` (the full chain/trajectory)
* `plots/wind_profile_*.png` (if `--plot-winds`)

---

## Step 2 — Postprocess across scenarios (one command)

This scans every scenario and produces:

* heatmaps (per model)
* interactive scatter (per scenario)
* trace+marginals diagnostics (for available runs)

```bash
python -m scripts.visualization.postprocess_sim_results \
  --experiments-root data/experiments \
  --scenarios-root data/scenarios \
  --config-module scripts.sim.exp_config \
  --plot-heatmaps \
  --interactive \
  --traces
```

### What gets written where?

For each scenario `X`, outputs go to:

* `data/experiments/X/output_heatmaps/`
* `data/experiments/X/output_simple/`
* `data/experiments/X/interactive_scatter_all_models_hover.html`

So scenarios no longer overwrite each other.

---

# Configuration

## Core inversion defaults

Defaults live in:

* `config/default_config.py`
* Loaded by: `scripts.data_handling.config_io.load_config()`

This base config defines:

* **paths** (input_dir, output_dir, tephra2 executable)
* **fixed plume/model parameters**
* **parameter definitions** (priors + proposal scales)
* **method defaults** (`mcmc`, `sa`, `pso`, `es` blocks)

## Experiment grid configuration

The experiment grid is defined in:

* `scripts/sim/exp_config.py` (full)
* `scripts/sim/exp_config_test.py` (small)

These modules define:

* `PRIOR_FACTORS` (7 values → 7×7 grid)
* `MODELS` (mcmc, sa, pso, es)
* `N_REPEATS`
* method hyperparameter grids:

  * `MCMC_N_ITER`
  * `SA_RUNS`, `SA_RESTARTS`
  * `PSO_RUNS`, `PSO_RESTARTS`
  * `ES_N_ENS`, `ES_N_ASSIM`
* optional overrides (e.g., `LOGM_PRIOR_STD`)

---

# What the Methods Are Doing (in this repo)

* **MCMC**: posterior sampling (good uncertainty quantification)
* **SA**: mode-seeking trajectory (fast optimization; still produces “chain-like” logs)
* **PSO**: swarm-based optimization across the parameter space
* **ES**: ensemble smoother with multiple assimilations (EnKF-style update, iterative passes)

All methods are run through the shared entry point:

* `scripts/tephra_inversion.py` → `TephraInversion.run_inversion()`

---

# Priors, “Mean/Std”, and “Draw Scale” (Where they come from)

There are two layers:

## 1) Base parameter definitions

Defined in `config/default_config.py` under the `parameters` section (per-parameter):

* `initial_value`
* `prior_type` (typically Gaussian)
* `prior_mean`
* `prior_std`
* `draw_scale` (proposal step size)

## 2) Simulation-grid prior re-centering

When running the simulation grid, priors are *re-centered* per grid cell by:

* `scripts/sim/simulate.py` → `_build_param_config(...)`

That function sets the **prior_mean** (and usually prior_std/draw_scale) based on:

* `sim_meta.json` truth (preferred)
* config `true_values` (fallback)
* config `initial_val` (last resort)

This is what enables the **7×7 “prior factor” grid** experiments.

---

# Extending to 4-Parameter Inversion (How to do it cleanly)

You currently invert 2 parameters (typically `plume_height`, `log_mass`). To add two more:

1. **Add parameters to the inversion config**

   * In `config/default_config.py` add two new parameter entries with
     `initial_value, prior_mean, prior_std, draw_scale`.

2. **Ensure the forward model actually uses them**

   * If they are Tephra2 inputs (e.g., `median_grain_size`, `sigma_grain`),
     you must thread them into whatever writes Tephra2 input files.

3. **Update the simulation grid logic**

   * `_build_param_config(...)` currently builds a 2D grid (height × mass).
   * For 4 parameters you have two common options:

   **Option A (recommended): keep 2D grid, fix the extra 2**

   * Still sweep priors over height×mass
   * Add fixed priors for the extra two (or a single factor)
   * This preserves the 7×7 heatmap concept.

   **Option B: expand to higher-dimensional grids**

   * A full 7×7×7×7 grid explodes quickly.
   * If you do this, switch to:

     * random/LHS sampling over prior centers, or
     * a smaller factor set for extra parameters.

4. **Update postprocessing**

   * Heatmaps remain 2D (choose which two dims define the grid)
   * Traces/marginals will automatically include extra parameters as long as
     they appear in the chain CSV columns.

---

# Outputs

Per scenario experiment folder (e.g. `data/experiments/cn92_a/`):

* `results_<model>.csv`
* `chains/<model>/<model>_run<id>.csv`
* `output_heatmaps/<model>_all_configs_accuracy.png`
* `output_simple/*_trace.png` and `*_marginals.png`
* `interactive_scatter_all_models_hover.html`

---

# Demo Notebook

`demo.ipynb` is intentionally minimal. It typically:

* shows input visuals (observations map & wind profile)
* runs a small inversion example (single config)
* calls postprocessing utilities rather than re-implementing plotting inline

---

## License

MIT License.

---

## Acknowledgements

This project builds significantly upon foundational work by **Qingyuan Yang**, who developed the original Metropolis–Hastings Tephra2 inversion workflow. His codebases served as core inspiration, and his mentorship was invaluable.

Special thanks to my PI **Professor Einat Lev** and **Software Engineer Samuel Krasnoff** at Columbia University’s Lamont-Doherty Earth Observatory (LDEO) for guidance and collaboration.

This work was conducted as part of the **Columbia Data Science Institute Scholar Program**, and their support is gratefully acknowledged.

We also thank:

* The developers and maintainers of Tephra2
* The Victor Platform team