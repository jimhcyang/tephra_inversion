#!/usr/bin/env python3
"""
mcmc.py

Implementation of the Metropolis-Hastings algorithm for Tephra2 parameter inversion.
Simplified version that focuses on estimating only plume height and eruption mass.

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
import logging
from pathlib import Path
import stat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for clarity
warnings.filterwarnings("ignore")


def ensure_executable(path):
    """
    Ensure the tephra2 executable has proper permissions.
    
    Args:
        path (str): Path to the tephra2 executable
    """
    path = Path(path)
    if path.exists():
        current_mode = os.stat(path).st_mode
        os.chmod(path, current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        logger.info(f"Set executable permissions for {path}")
    else:
        logger.error(f"Warning: {path} does not exist")
        raise FileNotFoundError(f"Tephra2 executable not found: {path}")


def changing_variable(input_params, config_path):
    """
    Update the tephra2 configuration file with new parameters.
    
    Args:
        input_params (np.ndarray): Parameters for tephra2
        config_path (str): Path to the configuration file
    """
    try:
        with open(config_path, 'r') as fid:
            s = fid.readlines()
            
        # Update parameters in config file
        s[0] = f'PLUME_HEIGHT {float(input_params[0])}\n'
        s[1] = f'ERUPTION_MASS {float(np.exp(input_params[1]))}\n'
        s[2] = f'ALPHA {float(input_params[2])}\n'
        s[3] = f'BETA {float(input_params[3])}\n'
        s[4] = f'MAX_GRAINSIZE {float(input_params[4])}\n'
        s[5] = f'MIN_GRAINSIZE {float(input_params[5])}\n'
        s[6] = f'MEDIAN_GRAINSIZE {float(input_params[6])}\n'
        s[7] = f'STD_GRAINSIZE {float(input_params[7])}\n'
        s[8] = f'VENT_EASTING {float(input_params[8])}\n'
        s[9] = f'VENT_NORTHING {float(input_params[9])}\n'
        s[10] = f'VENT_ELEVATION {float(input_params[10])}\n'
        s[11] = f'EDDY_CONST {float(input_params[11])}\n'
        s[12] = f'DIFFUSION_COEFFICIENT {float(input_params[12])}\n'
        s[13] = f'FALL_TIME_THRESHOLD {float(input_params[13])}\n'
        s[14] = f'LITHIC_DENSITY {float(input_params[14])}\n'
        s[15] = f'PUMICE_DENSITY {float(input_params[15])}\n'
        s[16] = f'COL_STEPS {float(input_params[16])}\n'
        s[17] = f'PART_STEPS {float(input_params[17])}\n'
        s[18] = f'PLUME_MODEL {int(input_params[18])}\n'
        
        with open(config_path, 'w') as out:
            for i in range(len(s)):
                out.write(s[i])
                
        logger.debug(f"Updated config file at {config_path}")
    
    except Exception as e:
        logger.error(f"Error updating config file: {str(e)}")
        raise


def draw_input_parameter(input_params, prior_type, draw_scale, prior_para):
    """
    Proposes new samples for eruption source parameters.
    Simplified to handle only plume height and eruption mass.

    Parameters
    ----------
    input_params : array-like
        Current parameter vector [plume_height, log_mass].
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
    mask_gaussian = np.array(prior_type) == "Gaussian"
    if np.any(mask_gaussian):
        param_draw[mask_gaussian] = np.random.normal(
            loc=input_params[mask_gaussian], 
            scale=draw_scale[mask_gaussian]
        )
    
    # For parameters flagged as "Uniform"
    mask_uniform = np.array(prior_type) == "Uniform"
    if np.any(mask_uniform):
        # Extract lower and upper bounds for each parameter
        indices = np.where(mask_uniform)[0]
        for i in indices:
            lower, upper = prior_para[i][0], prior_para[i][1]
            param_draw[i] = np.random.normal(
                loc=input_params[i], 
                scale=draw_scale[i]
            )
            # Ensure the drawn value is within the bounds
            while param_draw[i] < lower or param_draw[i] > upper:
                param_draw[i] = np.random.normal(
                    loc=input_params[i], 
                    scale=draw_scale[i]
                )
    
    # "Fixed" parameters remain unchanged
    return param_draw


def prior_function(prior_type, input_params, prior_para):
    """
    Computes the log-prior for parameters.
    Simplified to handle only plume height and eruption mass.

    Parameters
    ----------
    prior_type : array of str
        "Gaussian", "Uniform", or "Fixed".
    input_params : np.array
        Current parameter vector [plume_height, log_mass].
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
    mask_gaussian = np.array(prior_type) == "Gaussian"
    if np.any(mask_gaussian):
        indices = np.where(mask_gaussian)[0]
        for i in indices:
            mean, std = prior_para[i][0], prior_para[i][1]
            pdf_value = norm.pdf(input_params[i], loc=mean, scale=std)
            priors[i] = np.log10(np.clip(pdf_value, 1e-300, None))
    
    # Uniform prior
    mask_uniform = np.array(prior_type) == "Uniform"
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
    config_path : str or Path
        Path to Tephra2 configuration file
    sites_path : str or Path
        Path to sites file
    wind_path : str or Path
        Path to wind profile file
    output_path : str or Path
        Path where Tephra2 output will be saved
    tephra2_path : str or Path
        Path to Tephra2 executable
    silent : bool
        If True, suppresses Tephra2 console output

    Returns
    -------
    np.array
        The tephra deposit predictions from the output file
    """
    # Ensure paths are strings
    config_path = str(config_path)
    sites_path = str(sites_path)
    wind_path = str(wind_path)
    output_path = str(output_path)
    tephra2_path = str(tephra2_path)
    
    # Add debug output
    logger.debug(f"Running tephra2 with:")
    logger.debug(f"  tephra2_path: {tephra2_path}")
    logger.debug(f"  config_path: {config_path}")
    logger.debug(f"  sites_path: {sites_path}")
    logger.debug(f"  wind_path: {wind_path}")
    logger.debug(f"  output_path: {output_path}")
    
    # Check if files exist
    for path, desc in [
        (tephra2_path, "Tephra2 executable"),
        (config_path, "Config file"),
        (sites_path, "Sites file"),
        (wind_path, "Wind file"),
    ]:
        if not os.path.exists(path):
            logger.error(f"{desc} not found: {path}")
            raise FileNotFoundError(f"{desc} not found: {path}")
    
    # Ensure executable permission
    ensure_executable(tephra2_path)
    
    try:
        # Use shell=True approach with command string
        cmd_str = f"{tephra2_path} {config_path} {sites_path} {wind_path}"
        
        if silent:
            # Redirect output to file
            result = subprocess.run(f"{cmd_str} > {output_path}", 
                                    shell=True, 
                                    stderr=subprocess.PIPE,
                                    stdout=subprocess.PIPE,
                                    check=False)
        else:
            # Open file for writing output
            with open(output_path, 'w') as f:
                result = subprocess.run(cmd_str, 
                                        shell=True,
                                        stdout=f,
                                        stderr=subprocess.PIPE,
                                        check=False)
        
        # Check if the command was successful
        if result.returncode != 0:
            error_msg = result.stderr.decode() if result.stderr else "No error message available"
            logger.error(f"Tephra2 execution failed with error code {result.returncode}: {error_msg}")
            raise subprocess.CalledProcessError(result.returncode, cmd_str, stderr=error_msg)
        
        # Read and return the output
        try:
            prediction = np.genfromtxt(output_path, delimiter=' ')
            prediction = np.nan_to_num(prediction, nan=0.001)
            return prediction[:, 3]  # Extract the relevant column (mass)
        except Exception as e:
            logger.error(f"Error reading tephra2 output: {str(e)}")
            raise
        
    except Exception as e:
        logger.error(f"Error running tephra2: {str(e)}")
        raise


def compute_posterior(input_params, config_path, sites_path, wind_path, output_path, 
                     tephra2_path, observation, likelihood_scale, prior_type, prior_para, silent):
    """
    Compute the posterior value for the given parameters.
    
    Args:
        input_params (np.ndarray): Parameters to evaluate
        config_path (str): Path to tephra2 config file
        sites_path (str): Path to tephra2 sites file
        wind_path (str): Path to tephra2 wind file
        output_path (str): Path for tephra2 output
        tephra2_path (str): Path to tephra2 executable
        observation (np.ndarray): Observed deposit thicknesses
        likelihood_scale (float): Scale parameter for likelihood function
        prior_type (list): Prior distribution types for each parameter
        prior_para (list): Prior distribution parameters for each parameter
        silent (bool): Whether to run tephra2 silently
    
    Returns:
        tuple: (posterior, likelihood, prior) values
    """
    # 1. Update config file with input parameters
    changing_variable(input_params, config_path)
    
    # 2. Run Tephra2 forward model
    try:
        prediction = run_tephra2(config_path, sites_path, wind_path, output_path, tephra2_path, silent)
        
        # 3. Calculate likelihood
        likelihood_temp = likelihood_function(prediction, observation, likelihood_scale)
        
        # 4. Calculate prior
        prior_temp = prior_function(prior_type, input_params, prior_para)
        
        # 5. Calculate posterior
        posterior_temp = np.sum(likelihood_temp) + np.sum(prior_temp)
        
        return posterior_temp, likelihood_temp, prior_temp
    
    except Exception as e:
        logger.error(f"Error in compute_posterior: {str(e)}")
        raise


def metropolis_hastings(input_params, prior_type, draw_scale, prior_para,
                       config_path, sites_path, wind_path, output_path, tephra2_path,
                       runs, likelihood_scale, observation, check_snapshot=100, silent=True):
    """
    Implements the Metropolis-Hastings algorithm for Tephra2 parameter inversion.
    Simplified to handle only plume height and eruption mass.

    Parameters
    ----------
    input_params : array-like
        Initial parameter vector [plume_height, log_mass].
    prior_type : array of str
        Distribution types for parameters.
    draw_scale : np.array
        Step sizes for parameter proposals.
    prior_para : array-like
        Prior distribution parameters.
    config_path, sites_path, wind_path, output_path : str or Path
        Paths to Tephra2 input/output files
    tephra2_path : str or Path
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
    try:
        # 1. Initialize storage
        n_params = len(input_params)
        chain = np.zeros((runs + 1, n_params))
        post_chain = np.zeros(runs + 1)
        prior_array = np.zeros(runs + 1)
        likeli_array = np.zeros(runs + 1)
        
        # 2. Set initial state
        chain[0] = input_params
        
        # Print current directory and file existence status for debugging
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info("Checking if key files exist:")
        for path in [tephra2_path, config_path, sites_path, wind_path]:
            logger.info(f"  {path}: {'exists' if os.path.exists(path) else 'does not exist'}")
        
        # Ensure tephra2 is executable
        ensure_executable(tephra2_path)
        
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
            try:
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
                    logger.warning(f"Invalid proposal at iteration {i}: non-finite posterior")
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
                    logger.info(f"Iteration {i}/{runs}: Acceptance Rate = {acceptance_count / i:.4f}")
                    
            except Exception as e:
                logger.error(f"Error in MCMC iteration {i}: {e}")
                # Keep previous state on error
                chain[i] = chain[i-1]
                post_chain[i] = post_chain[i-1]
                prior_array[i] = prior_array[i-1]
                likeli_array[i] = likeli_array[i-1]
        
        logger.info(f"MCMC completed with final acceptance rate: {acceptance_count / runs:.4f}")
        return chain, post_chain, acceptance_count, prior_array, likeli_array
        
    except Exception as e:
        logger.error(f"Fatal error in MCMC: {e}")
        raise


if __name__ == "__main__":
    print("This module provides MCMC functionality for tephra inversion.")
    print("Import this module in your scripts or notebooks to use its functions.")