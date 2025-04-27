#!/usr/bin/env python3
"""
mcmc.py

Implementation of the Metropolis-Hastings algorithm for Tephra2 parameter inversion.
This module focuses solely on the core MCMC functionality, excluding diagnostic and 
visualization functions.

Functions:
- changing_variable: Update Tephra2 config file with new parameters
- draw_input_parameter: Draw a new sample for parameters
- prior_function: Compute prior probability for parameters
- likelihood_function: Compute likelihood of observations given predictions
- run_tephra2: Interface with Tephra2 model
- compute_posterior: Run forward model and compute posterior
- metropolis_hastings: Main MCMC algorithm

Dependencies:
- numpy
- scipy.stats
- subprocess (for calling Tephra2)
- tqdm (for progress bars)
"""

import os
import math
import numpy as np
import subprocess
from scipy.stats import norm, uniform
from tqdm import tqdm
import warnings

# Suppress warnings for clarity
warnings.filterwarnings("ignore")


def changing_variable(input_params, config_path):
    """
    Updates the Tephra2 configuration file with new eruption source parameters.
    
    Parameters
    ----------
    input_params : array-like of float
        Current values of the eruption source parameters (ESPs).
    config_path : str
        Path to the Tephra2 configuration file to update.

    Returns
    -------
    None
        (Modifies config file in place)
    """
    # Map of parameter indices to config file parameter names
    param_names = [
        'PLUME_HEIGHT',         # 0
        'ERUPTION_MASS',        # 1 (input is log, we convert with exp)
        'ALPHA',                # 2
        'BETA',                 # 3
        'MAX_GRAINSIZE',        # 4
        'MIN_GRAINSIZE',        # 5
        'MEDIAN_GRAINSIZE',     # 6
        'STD_GRAINSIZE',        # 7
        'VENT_EASTING',         # 8
        'VENT_NORTHING',        # 9
        'VENT_ELEVATION',       # 10
        'EDDY_CONST',           # 11
        'DIFFUSION_COEFFICIENT',# 12
        'FALL_TIME_THRESHOLD',  # 13
        'LITHIC_DENSITY',       # 14
        'PUMICE_DENSITY',       # 15
        'COL_STEPS',            # 16
        'PART_STEPS',           # 17
        'PLUME_MODEL'           # 18
    ]
    
    # Special handling for certain parameters
    param_formatters = {
        'ERUPTION_MASS': lambda x: math.exp(x),   # Convert log mass to linear
        'PLUME_MODEL': lambda x: int(x)           # Ensure plume model is an integer
    }
    
    # Read the entire config file
    try:
        with open(config_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    # Create a dictionary of parameters to update
    updates = {}
    for i, param_name in enumerate(param_names):
        if i < len(input_params):
            # Apply special formatting if needed
            if param_name in param_formatters:
                value = param_formatters[param_name](input_params[i])
            else:
                value = input_params[i]
            updates[param_name] = value
    
    # Update the config file lines
    updated_lines = []
    params_found = set()
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            # Keep comments and empty lines unchanged
            updated_lines.append(line + '\n')
            continue
        
        parts = line.split()
        if len(parts) >= 1 and parts[0] in updates:
            # Update this parameter
            param = parts[0]
            updated_lines.append(f"{param} {updates[param]:.6f}\n")
            params_found.add(param)
        else:
            # Keep line unchanged
            updated_lines.append(line + '\n')
    
    # Add any parameters that weren't found in the original file
    for param, value in updates.items():
        if param not in params_found:
            updated_lines.append(f"{param} {value:.6f}\n")
    
    # Write the updated file
    with open(config_path, 'w') as file:
        file.writelines(updated_lines)


def draw_input_parameter(input_params, prior_type, draw_scale, prior_para):
    """
    Proposes new samples for eruption source parameters.

    Parameters
    ----------
    input_params : array-like
        Current parameter vector.
    prior_type : array of str
        Distribution type for each parameter: "Gaussian", "Uniform", or "Fixed".
    draw_scale : array-like
        Standard deviations for the proposal distribution.
    prior_para : 2D array-like
        Prior parameters for each parameter.
        - If "Gaussian", then [mean, std].
        - If "Uniform", then [min, max].

    Returns
    -------
    param_draw : np.array
        Proposed new parameters.
    """
    param_draw = np.array(input_params).copy()  # Avoid in-place changes
    
    # For parameters flagged as "Gaussian"
    mask_gaussian = prior_type == "Gaussian"
    if np.any(mask_gaussian):
        param_draw[mask_gaussian] = np.random.normal(
            loc=input_params[mask_gaussian], 
            scale=draw_scale[mask_gaussian]
        )
    
    # For parameters flagged as "Uniform"
    mask_uniform = prior_type == "Uniform"
    if np.any(mask_uniform):
        # Extract lower and upper bounds for each parameter
        indices = np.where(mask_uniform)[0]
        for i in indices:
            lower, upper = prior_para[i][0], prior_para[i][1]
            param_draw[i] = np.random.uniform(lower, upper)
    
    # "Fixed" parameters remain unchanged
    return param_draw


def prior_function(prior_type, input_params, prior_para):
    """
    Computes the log-prior for parameters.

    Parameters
    ----------
    prior_type : array of str
        "Gaussian", "Uniform", or "Fixed".
    input_params : np.array
        Current parameter vector.
    prior_para : 2D array-like
        Prior parameters.
        - If "Gaussian", then [mean, std].
        - If "Uniform", then [min, max].

    Returns
    -------
    priors : np.array
        Log10 of the prior pdf for each parameter.
    """
    priors = np.zeros_like(input_params, dtype=float)
    
    # Gaussian prior
    mask_gaussian = prior_type == "Gaussian"
    if np.any(mask_gaussian):
        indices = np.where(mask_gaussian)[0]
        for i in indices:
            mean, std = prior_para[i][0], prior_para[i][1]
            pdf_value = norm.pdf(input_params[i], loc=mean, scale=std)
            priors[i] = np.log10(np.clip(pdf_value, 1e-300, None))
    
    # Uniform prior
    mask_uniform = prior_type == "Uniform"
    if np.any(mask_uniform):
        indices = np.where(mask_uniform)[0]
        for i in indices:
            lower, upper = prior_para[i][0], prior_para[i][1]
            # Check if parameter is within bounds
            if lower <= input_params[i] <= upper:
                pdf_value = 1.0 / (upper - lower)
                priors[i] = np.log10(np.clip(pdf_value, 1e-300, None))
            else:
                # Zero probability (or very small in log space) if outside bounds
                priors[i] = -300  # log10(1e-300)
    
    # "Fixed" parameters remain zero in log space (= probability of 1)
    return priors


def likelihood_function(prediction, observation, likelihood_scale):
    """
    Computes the log-likelihood comparing predicted vs. observed deposit loads.

    Parameters
    ----------
    prediction : np.array
        Model-predicted deposit load (kg/m2) at observation sites.
    observation : np.array
        Observed deposit load (kg/m2) from field measurements.
    likelihood_scale : float
        Standard deviation (σ) for the Gaussian likelihood in log10-space.

    Returns
    -------
    likelihood_array : np.array
        Log10 likelihood contributions from each observation.
    """
    # Avoid division by zero or log(0)
    prediction = np.clip(prediction, 0.001, None)
    observation = np.clip(observation, 0.001, None)

    # Compute log ratio of observation to prediction
    log_ratio = np.log10(observation / prediction)
    
    # Evaluate normal PDF with mean=0, std=likelihood_scale
    pdf_values = norm.pdf(log_ratio, loc=0, scale=likelihood_scale)
    
    # Convert to log10 space
    likelihood_array = np.log10(np.maximum(pdf_values, 1e-300))
    return likelihood_array


def run_tephra2(config_path, sites_path, wind_path, output_path, tephra2_path, silent=True):
    """
    Executes Tephra2 with specified input files.

    Parameters
    ----------
    config_path : str
        Path to Tephra2 configuration file
    sites_path : str
        Path to sites file
    wind_path : str
        Path to wind profile file
    output_path : str
        Path where Tephra2 output will be saved
    tephra2_path : str
        Path to Tephra2 executable
    silent : bool
        If True, suppresses Tephra2 console output

    Returns
    -------
    np.array
        The tephra deposit predictions from the output file
    """
    # Construct the command
    cmd = [tephra2_path, config_path, sites_path, wind_path]
    
    # Run Tephra2
    with open(output_path, 'w') as f:
        if silent:
            subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(cmd, stdout=f)
    
    # Read results (assuming columns: EAST, NORTH, ELEVATION, MASS, ...)
    try:
        # Try to read the output file
        output_data = np.genfromtxt(output_path, delimiter=' ')
        
        # Check if output data has at least 4 columns (EAST, NORTH, ELEV, MASS)
        if output_data.shape[1] >= 4:
            return output_data[:, 3]  # Return the mass column
        else:
            raise ValueError(f"Tephra2 output has fewer than 4 columns: {output_data.shape}")
    except Exception as e:
        print(f"Error reading Tephra2 output: {e}")
        # Return zeros as a fallback (but this indicates a problem)
        return np.zeros(100)  # Arbitrary size, should be replaced with actual expected size


def compute_posterior(input_params, config_path, sites_path, wind_path, 
                      output_path, tephra2_path, observation, likelihood_scale,
                      prior_type, prior_para, silent=True):
    """
    Computes the log-posterior = log-likelihood + log-prior.

    Parameters
    ----------
    input_params : np.array
        Proposed parameters.
    config_path, sites_path, wind_path, output_path : str
        Paths to Tephra2 input/output files
    tephra2_path : str
        Path to Tephra2 executable
    observation : np.array
        Observed deposit load at field sites.
    likelihood_scale : float
        Standard deviation for the log-likelihood function.
    prior_type : array of str
        Prior distribution types.
    prior_para : array-like
        Prior distribution parameters.
    silent : bool
        If True, suppresses Tephra2 console output.

    Returns
    -------
    posterior_val : float
        Sum of log-likelihood and log-prior values.
    likelihood_temp : np.array
        Individual observation log-likelihood contributions.
    prior_temp : np.array
        Individual parameter log-prior contributions.
    """
    # 1. Update config file with new parameters
    changing_variable(input_params, config_path)
    
    # 2. Run Tephra2 forward model
    prediction = run_tephra2(config_path, sites_path, wind_path, output_path, tephra2_path, silent)
    
    # 3. Calculate likelihood
    likelihood_temp = likelihood_function(prediction, observation, likelihood_scale)
    
    # 4. Calculate prior
    prior_temp = prior_function(prior_type, input_params, prior_para)
    
    # 5. Calculate posterior (sum in log space = product in normal space)
    posterior_val = np.sum(likelihood_temp) + np.sum(prior_temp)
    
    return posterior_val, likelihood_temp, prior_temp


def metropolis_hastings(input_params, prior_type, draw_scale, prior_para,
                       config_path, sites_path, wind_path, output_path, tephra2_path,
                       runs, likelihood_scale, observation, check_snapshot=100, silent=True):
    """
    Implements the Metropolis-Hastings algorithm for Tephra2 parameter inversion.

    Parameters
    ----------
    input_params : array-like
        Initial parameter vector.
    prior_type : array of str
        Distribution types for parameters.
    draw_scale : np.array
        Step sizes for parameter proposals.
    prior_para : array-like
        Prior distribution parameters.
    config_path, sites_path, wind_path, output_path : str
        Paths to Tephra2 input/output files
    tephra2_path : str
        Path to Tephra2 executable
    runs : int
        Number of MCMC iterations.
    likelihood_scale : float
        Standard deviation for the likelihood function.
    observation : np.array
        Observed deposit loads at each site.
    check_snapshot : int
        Frequency to print status updates.
    silent : bool
        If True, suppresses Tephra2 console output.

    Returns
    -------
    chain : (runs+1) x n_params np.array
        Sample chain of all accepted states.
    post_chain : (runs+1) np.array
        Posterior values for each sample in the chain.
    acceptance_count : int
        How many proposed states were accepted.
    prior_array : (runs+1) np.array
        Log-prior sums for each iteration.
    likeli_array : (runs+1) np.array
        Log-likelihood sums for each iteration.
    """
    # 1. Initialize storage
    n_params = len(input_params)
    chain = np.zeros((runs + 1, n_params))
    post_chain = np.zeros(runs + 1)
    prior_array = np.zeros(runs + 1)
    likeli_array = np.zeros(runs + 1)
    
    # 2. Set initial state
    chain[0] = input_params
    
    # 3. Compute initial posterior
    posterior_temp, likelihood_temp, prior_temp = compute_posterior(
        input_params, config_path, sites_path, wind_path, output_path,
        tephra2_path, observation, likelihood_scale, prior_type, prior_para, silent
    )
    
    post_chain[0] = posterior_temp
    prior_array[0] = np.sum(prior_temp)
    likeli_array[0] = np.sum(likelihood_temp)
    
    acceptance_count = 0
    
    # 4. Main MCMC loop
    for i in tqdm(range(1, runs + 1), desc="MCMC Progress"):
        # 4.1. Propose new sample
        params_temp = draw_input_parameter(
            chain[i-1], prior_type, draw_scale, prior_para
        )
        
        # 4.2. Compute new posterior
        posterior_new, likelihood_new, prior_new = compute_posterior(
            params_temp, config_path, sites_path, wind_path, output_path,
            tephra2_path, observation, likelihood_scale, prior_type, prior_para, silent
        )
        
        # 4.3. Accept/reject step
        if not np.isfinite(posterior_new):
            # Automatically reject invalid proposals
            acceptance_prob = 0
        else:
            # Standard Metropolis acceptance probability
            acceptance_prob = min(1.0, 10**(posterior_new - posterior_temp))
        
        if np.random.rand() < acceptance_prob:
            # Accept the proposal
            chain[i] = params_temp
            post_chain[i] = posterior_new
            posterior_temp = posterior_new
            prior_array[i] = np.sum(prior_new)
            likeli_array[i] = np.sum(likelihood_new)
            acceptance_count += 1
        else:
            # Reject the proposal, keep previous state
            chain[i] = chain[i-1]
            post_chain[i] = post_chain[i-1]
            prior_array[i] = prior_array[i-1]
            likeli_array[i] = likeli_array[i-1]
        
        # 4.4. Periodic reporting
        if i % check_snapshot == 0:
            print(f"Iteration {i}/{runs}: Acceptance Rate = {acceptance_count / i:.4f}")
    
    return chain, post_chain, acceptance_count, prior_array, likeli_array


if __name__ == "__main__":
    print("This module provides MCMC functionality for tephra inversion.")
    print("Import this module in your scripts or notebooks to use its functions.")