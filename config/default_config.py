# config/default_config.py

DEFAULT_CONFIG = {
    "tephra2": {
        "executable": "Tephra2/tephra2_2020",
        "output_file": "data/output/tephra2_output.txt",
    },

    "parameters": {
        "true_values": {          # used only for plotting, if present
            "plume_height": 7500,
            "eruption_mass": 5e10,
        },
        "variable": {
            "column_height": {    # plume height (m)
                "initial_val": 7500,
                "prior_type": "Gaussian",
                "prior_para_a": 7500,   # mean
                "prior_para_b": 1000,   # std
                "draw_scale": 500       # SA/MH step (m)
            },
            "eruption_mass": {    # mass (kg); code converts to ln(mass) for MCMC
                "initial_val": 5e10,
                "prior_type": "Gaussian",
                "prior_para_a": 5e10,    # mean (linear space)
                "prior_para_b": 0.5e10,  # std (linear space)
                "draw_scale": 0.25       # step in ln-space for samplers
            },
        },
        "fixed": {
            "alpha": 2.0,
            "beta": 2.0,
            "max_grainsize": -5,
            "min_grainsize": 5,
            "median_grainsize": 0,
            "std_grainsize": 2.0,
            "eddy_const": 0.04,
            "diffusion_coefficient": 1000,
            "fall_time_threshold": 1e9,
            "lithic_density": 2700,
            "pumice_density": 1024,
            "col_steps": 100,
            "part_steps": 100,
            "plume_model": 2,
        },
    },

    # MCMC (unchanged)
    "mcmc": {
        "n_iterations": 50000,
        "n_burnin": 0,
        "n_thin": 1,
        "proposal_scale": 0.5,
        "seed": 20250812,
        "likelihood_sigma": 0.6,   # log-space residual sigma
        "silent": False,
        "snapshot": 100,
        "save_chain": True,
        "output_dir": "data/output/mcmc",
    },

    # SA comparable to MCMC but with adaptive cooling available by default.
    # If alpha is None, the code chooses alpha so T decays from T0 -> T_end across 'runs'.
    "sa": {
        "runs": 5000,           # ~match MCMC iterations
        "T0": 1.0,              # initial temperature
        "alpha": 0.999,          # ADAPTIVE: let code compute from T0 -> T_end
        "T_end": 0.01,          # target terminal temperature (used when alpha=None)
        "restarts": 0,
        "likelihood_sigma": 0.6,
        "seed": 20250812,
        "silent": False,
        "print_every": 100,
    },

    # EnKF/ES‑MDA: small ensemble over 5 passes; gentle inflation; surfaced robustness knobs.
    "enkf": {
        "n_ens": 10000,
        "n_assimilations": 5,
        "inflation": 1.01,
        "likelihood_sigma": 0.6,
        "seed": 20250812,
        "silent": False,
        "print_every": 1,
        "member_update_every": 100,
        # exposed stability/exploration options (defaults match enkf.py)
        "obs_logspace": True,
        "sd_scale": 1.0,
        "jitter_after_pass": 0.0,
        "step_trust": 3.0,
        "winsor_k": 6.0,
        "ridge_cyy": 1e-6,
        "suppress_runtime_warnings": True,
    },

    "paths": {
        "input_dir": "data/input",
        "output_dir": "data/output",
        "plots_dir": "data/output/plots",
        "mcmc_dir": "data/output/mcmc",
        "work_dir": "data/work",
        "cerro_dir": "data/input/cerro_negro",
    },
}
