# config/default_config.py

DEFAULT_CONFIG = {
    # Tephra2 model parameters
    "tephra2": {
        "executable": "Tephra2/tephra2_2020",
        "output_file": "data/output/tephra2_output.txt"
    },
    
    # Default parameters for Tephra2
    "parameters": {
        "true_values": {
            "plume_height": 7313,
            "eruption_mass": 1.4e10  # kg
        },
        "variable": {
            "column_height": {
                "initial_val": 7500,
                "prior_type": "Gaussian",
                "prior_para_a": 7500,  # mean
                "prior_para_b": 1500,   # std
                "draw_scale": 50
            },
            "eruption_mass": {
                "initial_val": 1e10,  # will be calculated from eruption_mass
                "prior_type": "Gaussian",
                "prior_para_a": 1e10,  # mean (will be replaced)
                "prior_para_b": 1,        # std
                "draw_scale": 0.1
            }
        },
        "fixed": {
            "alpha": 3.4,
            "beta": 2.0,
            "max_grainsize": -5,
            "min_grainsize": 5,
            "median_grainsize": 0.87,
            "std_grainsize": 2.0,
            "eddy_const": 0.04,
            "diffusion_coefficient": 500,
            "fall_time_threshold": 1e9,
            "lithic_density": 2700,
            "pumice_density": 1024,
            "col_steps": 100,
            "part_steps": 100,
            "plume_model": 2
        }
    },
    
    # MCMC settings
    "mcmc": {
        "n_iterations": 25000,     # Total number of MCMC iterations to run
        "n_burnin": 0,             # Number of initial samples to discard as burn-in
        "n_thin": 10,              # Keep every n_thin samples (thinning the chain)
        "proposal_scale": 0.1,     # Scale factor for proposal distribution (similar to draw_scale)
        "seed": 42,                # Random seed for reproducibility
        "likelihood_sigma": 0.25,  # Standard deviation for Gaussian likelihood in log-space (equivalent to likelihood_scale)
        "silent": False,           # If True, suppresses command-line output from Tephra2
        "snapshot": 100,           # Frequency to print status updates (equivalent to check_snapshot)
        "save_chain": True,        # Whether to save the MCMC chain to disk
        "output_dir": "data/output/mcmc"  # Directory to save MCMC results
    },
    
    # LHS settings
    "lhs": {
        "n_samples": 1000,
        "top_fraction": 0.05,
        "n_grid_points": 20,
        "seed": 42
    },
    
    # File paths
    "paths": {
        "input_dir": "data/input",
        "output_dir": "data/output",
        "plots_dir": "data/output/plots",
        "mcmc_dir": "data/output/mcmc"
    }
}