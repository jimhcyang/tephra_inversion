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
            "plume_height": 10000,
            "eruption_mass": 2.5e10  # kg
        },
        "variable": {
            "column_height": {
                "initial_val": 10000,
                "prior_type": "Gaussian",
                "prior_para_a": 10000,  # mean
                "prior_para_b": 2000,   # std
                "draw_scale": 50
            },
            "eruption_mass": {
                "initial_val": 2.5e10,  # will be calculated from eruption_mass
                "prior_type": "Gaussian",
                "prior_para_a": 2.5e10,  # mean (will be replaced)
                "prior_para_b": 1,        # std
                "draw_scale": 0.25
            }
        },
        "fixed": {
            "alpha": 3.4,
            "beta": 2.0,
            "max_grainsize": -6,
            "min_grainsize": 6,
            "median_grainsize": -1,
            "std_grainsize": 1.5,
            "eddy_const": 0.04,
            "diffusion_coefficient": 5000,
            "fall_time_threshold": 1419,
            "lithic_density": 2500,
            "pumice_density": 1100,
            "col_steps": 100,
            "part_steps": 100,
            "plume_model": 2
        }
    },
    
    # MCMC settings
    "mcmc": {
        "n_iterations": 10000,
        "n_burnin": 1000,
        "n_thin": 10,
        "proposal_scale": 0.1,
        "seed": 42
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