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
BASE_SEED = 123

# Prior mean mis-centering factors applied to true values.
# simulate.py uses a 2D grid of (scale_height, scale_mass).
PRIOR_FACTORS = [1.0]

# -----------------------------------------------------------------------------
# Methods to run
# -----------------------------------------------------------------------------
# Keep this small for smoke tests.
MODELS = [
    "mcmc",
    # "sa",
    # "pso",
    # "es",
]

# ---- SA ----
SA_RUNS = [200]
SA_RESTARTS = [1]

# ---- PSO ----
PSO_RUNS = [150]
PSO_RESTARTS = [1]

# ---- ES-MDA ----
ES_N_ENS = [50]
ES_N_ASSIM = [4]
ES_PRINT_EVERY = 10

# ---- MCMC ----
# Keep low for smoke tests; increase for real runs.
MCMC_N_ITER = [1000]

