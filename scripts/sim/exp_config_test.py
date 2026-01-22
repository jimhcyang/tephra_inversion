"""scripts/sim/exp_config_test.py

Fast smoke-test experiment configuration.

Used by:
  - python -m scripts.sim.simulate --config-module scripts.sim.exp_config_test
  - python -m scripts.sim.run_scenarios --config-module scripts.sim.exp_config_test

Notes
-----
- `run_scenarios` overrides SIM_INPUT_DIR/SIM_OUTPUT_DIR per scenario.
- `simulate` will use these defaults if you run it directly.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Where to read scenario inputs and where to write outputs.
# (When using run_scenarios, these are overridden per scenario.)
# -----------------------------------------------------------------------------
SIM_INPUT_DIR = "data/scenarios/cn92_a/input"
SIM_OUTPUT_DIR = "data/experiments/cn92_a"

# -----------------------------------------------------------------------------
# What to invert
# -----------------------------------------------------------------------------
# 2 = plume_height + log_mass (ln kg)
# 4 = plume_height + log_mass + median_grain + std_grain (phi)
INVERT_N_PARAMS = 4

# Optional override for ln-mass prior std (natural log space).
# If you omit it, simulate.py will use its internal default (~ln(10)/2).
# LOGM_PRIOR_STD = 1.15

# -----------------------------------------------------------------------------
# Grid structure
# -----------------------------------------------------------------------------
N_REPEATS = 1
BASE_SEED = 20261122

# Prior mean mis-centering factors applied to true values.
# simulate.py uses a 2D grid of (scale_height, scale_mass).
PRIOR_FACTORS = [1.0]

# -----------------------------------------------------------------------------
# Methods to run
# -----------------------------------------------------------------------------
# Keep this small for smoke tests.
MODELS = [
    "sa",
    "pso",
    "es",
    "mcmc",
]

# --- SA grid ----------------------------------------------------------------
# Just first two for smoke test
SA_RUNS = [10000]
SA_RESTARTS = [9]
# print_every will be set in code as runs // 10 (at least 1)

# --- PSO grid ---------------------------------------------------------------
PSO_RUNS = [1000]
PSO_RESTARTS = [9]
# print_every will be set in code as runs // 10 (at least 1)

# --- ES-MDA grid ------------------------------------------------------------
ES_N_ENS = [10000]
ES_N_ASSIM = [10]
ES_PRINT_EVERY = 1  # print every assimilation step

# --- MCMC grid --------------------------------------------------------------
MCMC_N_ITER = [100000]
# snapshot (print_every) will be set in code as n_iter // 10 (at least 1)