# scripts/sim/exp_config.py
from __future__ import annotations

# --- Global experiment controls ---------------------------------------------

# Base seed used to generate per-run seeds
BASE_SEED: int = 20251111

# How many *independent repetitions* of the whole grid to run
N_REPEATS: int = 10

# Directory that contains tephra2.conf, wind.txt, sites.csv, observations.csv
# For Cerro Negro "standard" workflow, this is "data/input".
# For your simulated setup, you might change this to "data_sim/input" etc.
SIM_INPUT_DIR: str = "data_sim_cerro/input"

# Where to write summary CSVs for each model
SIM_OUTPUT_DIR: str = "data_sim_cerro/experiments"

# Which inversion methods to include in this experiment
MODELS = ["mcmc", "sa", "pso", "es"]  # subset this list if you like

# --- Prior-std scaling factors ----------------------------------------------
# Factors are applied to the *prior standard deviation* of:
#   - column_height prior std
#   - eruption_mass prior std
# Means stay the same; only spread changes.
PRIOR_FACTORS = [
    4.0,
    3.0,
    2.0,
    1.5,
    1.25,
    1.0,
    0.8,
    2.0 / 3.0,
    0.5,
    1.0 / 3.0,
    0.25,
]

# --- SA grid ----------------------------------------------------------------
SA_RUNS = [100, 1000, 10000]
SA_RESTARTS = [0, 3, 9]
# print_every will be set in code as runs // 10 (at least 1)

# --- PSO grid ---------------------------------------------------------------
PSO_RUNS = [10, 100, 1000]
PSO_RESTARTS = [0, 3, 9]
# print_every will be set in code as runs // 10 (at least 1)

# --- ES-MDA grid ------------------------------------------------------------
ES_N_ENS = [10, 100, 1000]
ES_N_ASSIM = [1, 3, 9]
ES_PRINT_EVERY = 1 # print every assimilation step

# --- MCMC grid --------------------------------------------------------------
MCMC_N_ITER = [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]
# snapshot (print_every) will be set in code as n_iter // 10 (at least 1)
