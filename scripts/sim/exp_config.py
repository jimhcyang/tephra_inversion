# scripts/sim/exp_config.py
from __future__ import annotations

# --- Global experiment controls ---------------------------------------------

# Base seed used to generate per-run seeds
BASE_SEED: int = 20260111

# How many *independent repetitions* of the whole grid to run
N_REPEATS: int = 1

# Directory that contains tephra2.conf, wind.txt, sites.csv, observations.csv
# For Cerro Negro "standard" workflow, this is "data/input".
# For your simulated setup, you might change this to "data_sim/input" etc.
SIM_INPUT_DIR: str = "data_sim_cerro/input"

# Where to write summary CSVs for each model
SIM_OUTPUT_DIR: str = "data_sim_cerro/experiments"

# Which inversion methods to include in this experiment
MODELS = ["mcmc", "sa", "pso", "es"]  # subset this list if you like

# --- Prior scaling factors -------------------------------------------------
# Each factor k is applied to the *ground-truth center* of:
#   - plume_height
#   - eruption_mass
#
# For a given k, the simulation layer sets
#   prior_mean = initial_value = k * true_value
# and then enforces
#   prior_std  = prior_mean / 5
#   draw_scale = prior_std / 5    (for plume_height only; log_mass keeps default draw_scale)
PRIOR_FACTORS = [
    2.00,
    1.60,
    1.25,
    1.00,
    0.80,
    0.64,
    0.50,
]

# --- SA grid ----------------------------------------------------------------
SA_RUNS = [1000, 10000]
SA_RESTARTS = [0, 4]
# print_every will be set in code as runs // 10 (at least 1)

# --- PSO grid ---------------------------------------------------------------
PSO_RUNS = [100, 1000]
PSO_RESTARTS = [0, 4]
# print_every will be set in code as runs // 10 (at least 1)

# --- ES-MDA grid ------------------------------------------------------------
ES_N_ENS = [100, 1000]
ES_N_ASSIM = [1, 5]
ES_PRINT_EVERY = 1 # print every assimilation step

# --- MCMC grid --------------------------------------------------------------
MCMC_N_ITER = [100, 1000, 10000, 100000]
# snapshot (print_every) will be set in code as n_iter // 10 (at least 1)
