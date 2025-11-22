# scripts/sim/exp_config_test.py
from __future__ import annotations

# --- Global experiment controls ---------------------------------------------

# Base seed used to generate per-run seeds
BASE_SEED: int = 20251122

# How many *independent repetitions* of the whole grid to run
N_REPEATS: int = 1

# Directory that contains tephra2.conf, wind.txt, sites.csv, observations.csv
SIM_INPUT_DIR: str = "data_sim_cerro/input"

# Where to write summary CSVs for each model
SIM_OUTPUT_DIR: str = "data_sim_cerro/experiments_test"

# Which inversion methods to include in this experiment
MODELS = ["mcmc", "sa", "pso", "es"]

# --- Prior-std scaling factors ----------------------------------------------
# Smaller smoke-test set: 2x, 1x, 0.5x
PRIOR_FACTORS = [
    2.0,
    1.0,
    0.5,
]

# --- SA grid ----------------------------------------------------------------
# Just first two for smoke test
SA_RUNS = [1000]
SA_RESTARTS = [0]
# print_every will be set in code as runs // 10 (at least 1)

# --- PSO grid ---------------------------------------------------------------
PSO_RUNS = [100]
PSO_RESTARTS = [0]
# print_every will be set in code as runs // 10 (at least 1)

# --- ES-MDA grid ------------------------------------------------------------
ES_N_ENS = [100]
ES_N_ASSIM = [1]
ES_PRINT_EVERY = 1  # print every assimilation step

# --- MCMC grid --------------------------------------------------------------
MCMC_N_ITER = [1000]
# snapshot (print_every) will be set in code as n_iter // 10 (at least 1)
