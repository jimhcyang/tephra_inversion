#!/usr/bin/env python3
"""
diagnostics.py

Diagnostic and analysis tools for MCMC results from tephra inversion.
This module provides functions for analyzing, visualizing, and summarizing
MCMC chains generated from the metropolis_hastings algorithm.

Functions:
- summarize_posterior: Statistical summary of MCMC chains
- trace_plots: Generate trace plots for MCMC parameters
- acf_pacf_plots: Generate autocorrelation and partial autocorrelation plots
- effective_sample_size: Calculate effective sample size for MCMC chains
- convergence_metrics: Calculate convergence metrics like Gelman-Rubin
- parameter_correlation: Analyze parameter correlations
- credible_intervals: Calculate credible intervals for parameters

Dependencies:
- numpy
- pandas
- matplotlib
- seaborn
- statsmodels (for ACF/PACF)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import warnings

# Try to import statsmodels for ACF/PACF plots
try:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available; ACF/PACF plots will be disabled")


def summarize_posterior(chain: np.ndarray, 
                       var_names: List[str] = None,
                       burn_in_fraction: float = 0.5,
                       credible_interval: float = 0.95,
                       true_values: Dict[str, float] = None,
                       show_plots: bool = True,
                       plot_kde: bool = True,
                       figsize: Tuple[int, int] = (10, 8),
                       save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Generate summary statistics and plots for MCMC samples.

    Parameters
    ----------
    chain : np.ndarray
        MCMC chain with shape (n_samples, n_parameters)
    var_names : List[str], optional
        List of parameter names (default: "param_0", "param_1", etc.)
    burn_in_fraction : float, optional
        Fraction of samples to discard as burn-in (default: 0.5)
    credible_interval : float, optional
        Credible interval probability (default: 0.95)
    true_values : Dict[str, float], optional
        Dictionary of true parameter values for reference
    show_plots : bool, optional
        Whether to generate and display plots (default: True)
    plot_kde : bool, optional
        Whether to include kernel density estimates in histograms (default: True)
    figsize : Tuple[int, int], optional
        Figure size in inches (width, height) (default: (10, 8))
    save_path : str, optional
        Path to save the summary plot (default: None, no saving)

    Returns
    -------
    pd.DataFrame
        Summary statistics for each parameter
    """
    # Validate inputs
    if not 0 <= burn_in_fraction < 1:
        raise ValueError("burn_in_fraction must be between 0 and 1")
    if not 0 < credible_interval < 1:
        raise ValueError("credible_interval must be between 0 and 1")
    
    # Get dimensions
    n_samples, n_params = chain.shape
    
    # Create default parameter names if not provided
    if var_names is None:
        var_names = [f"param_{i}" for i in range(n_params)]
    elif len(var_names) != n_params:
        raise ValueError(f"Length of var_names ({len(var_names)}) must match number of parameters ({n_params})")
    
    # Discard burn-in samples
    burn_in = int(n_samples * burn_in_fraction)
    post_burn_chain = chain[burn_in:, :]
    n_post = len(post_burn_chain)
    
    # Calculate summary statistics
    means = np.mean(post_burn_chain, axis=0)
    medians = np.median(post_burn_chain, axis=0)
    
    # Calculate credible intervals
    alpha = (1 - credible_interval) / 2
    lower_percentile = alpha * 100
    upper_percentile = (1 - alpha) * 100
    
    ci_lower = np.percentile(post_burn_chain, lower_percentile, axis=0)
    ci_upper = np.percentile(post_burn_chain, upper_percentile, axis=0)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame({
        "Parameter": var_names,
        "Mean": means,
        "Median": medians,
        f"CI_{lower_percentile:.1f}%": ci_lower,
        f"CI_{upper_percentile:.1f}%": ci_upper
    })
    
    # Identify which parameters varied during MCMC
    valid_mask = summary_df[f"CI_{lower_percentile:.1f}%"] != summary_df[f"CI_{upper_percentile:.1f}%"]
    summary_df = summary_df[valid_mask].reset_index(drop=True)
    varying_params = summary_df["Parameter"].tolist()
    
    # Generate plots if requested
    if show_plots and len(varying_params) > 0:
        # Calculate number of rows and columns for subplots
        n_plots = len(varying_params)
        n_cols = min(2, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Handle case of single subplot
        if n_plots == 1:
            axes = np.array([axes])
        
        # Flatten axes for easier iteration
        axes = np.ravel(axes)
        
        # Create plots
        for i, param in enumerate(varying_params):
            if i < len(axes):
                # Get parameter index
                param_idx = var_names.index(param)
                
                # Get samples
                samples = post_burn_chain[:, param_idx]
                
                # Generate histogram/KDE
                if plot_kde:
                    sns.histplot(samples, kde=True, ax=axes[i], color="steelblue")
                else:
                    sns.histplot(samples, ax=axes[i], color="steelblue")
                
                # Add reference lines
                median = summary_df.loc[summary_df["Parameter"] == param, "Median"].values[0]
                ci_low = summary_df.loc[summary_df["Parameter"] == param, f"CI_{lower_percentile:.1f}%"].values[0]
                ci_high = summary_df.loc[summary_df["Parameter"] == param, f"CI_{upper_percentile:.1f}%"].values[0]
                
                axes[i].axvline(median, color="red", linestyle="--", label=f"Median={median:.4g}")
                axes[i].axvline(ci_low, color="green", linestyle=":", label=f"{lower_percentile:.1f}%={ci_low:.4g}")
                axes[i].axvline(ci_high, color="green", linestyle=":", label=f"{upper_percentile:.1f}%={ci_high:.4g}")
                
                # Add true value if provided
                if true_values and param in true_values:
                    true_val = true_values[param]
                    axes[i].axvline(true_val, color="blue", linestyle="-", linewidth=1.5, 
                                    label=f"True={true_val:.4g}")
                
                # Add labels
                axes[i].set_title(f"Posterior: {param}")
                axes[i].set_xlabel(param)
                axes[i].set_ylabel("Density")
                axes[i].legend(fontsize=8)
        
        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        # Add overall title
        plt.suptitle(f"Posterior Distributions (Burn-in: {burn_in_fraction*100:.0f}%, " +
                   f"CI: {credible_interval*100:.0f}%)", fontsize=14)
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            
        plt.show()
    
    # Print summary message
    print(f"Burn-in: {burn_in} samples discarded. {n_post} samples remain.")
    
    return summary_df


def trace_plots(chain: np.ndarray, 
              var_names: List[str] = None,
              true_values: Dict[str, float] = None,
              fixed_params: List[str] = None,
              n_cols: int = 1,
              figsize: Tuple[int, int] = (10, 8),
              save_path: Optional[str] = None) -> plt.Figure:
    """
    Generate trace plots for MCMC chains.

    Parameters
    ----------
    chain : np.ndarray
        MCMC chain with shape (n_samples, n_parameters)
    var_names : List[str], optional
        List of parameter names (default: "param_0", "param_1", etc.)
    true_values : Dict[str, float], optional
        Dictionary of true parameter values for reference
    fixed_params : List[str], optional
        List of parameters that were fixed during MCMC (not to be plotted)
    n_cols : int, optional
        Number of columns for subplot grid (default: 1)
    figsize : Tuple[int, int], optional
        Figure size in inches (width, height) (default: (10, 8))
    save_path : str, optional
        Path to save the trace plots (default: None, no saving)

    Returns
    -------
    plt.Figure
        The generated figure object
    """
    # Get dimensions
    n_samples, n_params = chain.shape
    
    # Create default parameter names if not provided
    if var_names is None:
        var_names = [f"param_{i}" for i in range(n_params)]
    elif len(var_names) != n_params:
        raise ValueError(f"Length of var_names ({len(var_names)}) must match number of parameters ({n_params})")
    
    # Identify fixed parameters
    if fixed_params is None:
        fixed_params = []
    
    # Identify varying parameters (min != max)
    varying_params = []
    for i, param in enumerate(var_names):
        if param not in fixed_params and np.min(chain[:, i]) != np.max(chain[:, i]):
            varying_params.append(param)
    
    # Calculate number of rows and columns for subplots
    n_plots = len(varying_params)
    n_cols = min(n_cols, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
    
    # Handle case of single subplot
    if n_plots == 1:
        axes = np.array([axes])
    
    # Flatten axes for easier iteration
    axes = np.ravel(axes)
    
    # Create plots
    for i, param in enumerate(varying_params):
        if i < len(axes):
            # Get parameter index
            param_idx = var_names.index(param)
            
            # Plot trace
            axes[i].plot(chain[:, param_idx], color="steelblue")
            
            # Calculate running mean
            window_size = max(1, n_samples // 20)
            running_mean = np.convolve(chain[:, param_idx], 
                                       np.ones(window_size)/window_size, 
                                       mode='valid')
            x_vals = np.arange(window_size-1, n_samples)
            axes[i].plot(x_vals, running_mean, color="red", linewidth=1.5)
            
            # Add true value if provided
            if true_values and param in true_values:
                true_val = true_values[param]
                axes[i].axhline(true_val, color="green", linestyle="--", linewidth=1.5)
            
            # Add median
            median = np.median(chain[:, param_idx])
            axes[i].axhline(median, color="black", linestyle=":", linewidth=1)
            
            # Add labels
            axes[i].set_ylabel(param)
            axes[i].grid(True, alpha=0.3)
    
    # Add x-label to bottom subplot
    axes[-1].set_xlabel("Iteration")
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    # Add overall title
    plt.suptitle("Trace Plots for MCMC Chains", fontsize=14)
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def acf_pacf_plots(chain: np.ndarray, 
                 var_names: List[str] = None,
                 fixed_params: List[str] = None,
                 burn_in_fraction: float = 0.5,
                 max_lags: int = 40,
                 figsize: Tuple[int, int] = (12, 8),
                 save_path: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Generate ACF and PACF plots for MCMC chains.

    Parameters
    ----------
    chain : np.ndarray
        MCMC chain with shape (n_samples, n_parameters)
    var_names : List[str], optional
        List of parameter names (default: "param_0", "param_1", etc.)
    fixed_params : List[str], optional
        List of parameters that were fixed during MCMC (not to be plotted)
    burn_in_fraction : float, optional
        Fraction of samples to discard as burn-in (default: 0.5)
    max_lags : int, optional
        Maximum number of lags to include in plots (default: 40)
    figsize : Tuple[int, int], optional
        Figure size in inches (width, height) (default: (12, 8))
    save_path : str, optional
        Path to save the ACF/PACF plots (default: None, no saving)

    Returns
    -------
    plt.Figure or None
        The generated figure object, or None if statsmodels is not available
    """
    # Check if statsmodels is available
    if not STATSMODELS_AVAILABLE:
        warnings.warn("statsmodels is required for ACF/PACF plots")
        return None
    
    # Get dimensions
    n_samples, n_params = chain.shape
    
    # Create default parameter names if not provided
    if var_names is None:
        var_names = [f"param_{i}" for i in range(n_params)]
    elif len(var_names) != n_params:
        raise ValueError(f"Length of var_names ({len(var_names)}) must match number of parameters ({n_params})")
    
    # Identify fixed parameters
    if fixed_params is None:
        fixed_params = []
    
    # Identify varying parameters (min != max)
    varying_params = []
    for i, param in enumerate(var_names):
        if param not in fixed_params and np.min(chain[:, i]) != np.max(chain[:, i]):
            varying_params.append(param)
    
    # Discard burn-in samples
    burn_in = int(n_samples * burn_in_fraction)
    post_burn_chain = chain[burn_in:, :]
    
    # Create figure
    fig, axes = plt.subplots(len(varying_params), 2, figsize=figsize)
    
    # Handle case of single parameter
    if len(varying_params) == 1:
        axes = np.array([axes])
    
    # Create plots
    for i, param in enumerate(varying_params):
        # Get parameter index
        param_idx = var_names.index(param)
        
        # Extract chain for this parameter
        param_chain = post_burn_chain[:, param_idx]
        
        # ACF plot
        plot_acf(param_chain, ax=axes[i, 0], lags=max_lags, alpha=0.05)
        axes[i, 0].set_title(f"ACF: {param}")
        
        # PACF plot
        plot_pacf(param_chain, ax=axes[i, 1], lags=max_lags, alpha=0.05, method="ywm")
        axes[i, 1].set_title(f"PACF: {param}")
    
    # Add overall title
    plt.suptitle("Autocorrelation and Partial Autocorrelation Functions", fontsize=14)
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    return fig


def effective_sample_size(chain: np.ndarray, 
                        var_names: List[str] = None,
                        burn_in_fraction: float = 0.5) -> pd.DataFrame:
    """
    Calculate effective sample size for MCMC chains.

    Parameters
    ----------
    chain : np.ndarray
        MCMC chain with shape (n_samples, n_parameters)
    var_names : List[str], optional
        List of parameter names (default: "param_0", "param_1", etc.)
    burn_in_fraction : float, optional
        Fraction of samples to discard as burn-in (default: 0.5)

    Returns
    -------
    pd.DataFrame
        DataFrame with effective sample sizes for each parameter
    """
    try:
        import arviz as az
    except ImportError:
        warnings.warn("arviz package is required for effective sample size calculation")
        return pd.DataFrame({"Parameter": var_names, "ESS": np.nan})
    
    # Get dimensions
    n_samples, n_params = chain.shape
    
    # Create default parameter names if not provided
    if var_names is None:
        var_names = [f"param_{i}" for i in range(n_params)]
    elif len(var_names) != n_params:
        raise ValueError(f"Length of var_names ({len(var_names)}) must match number of parameters ({n_params})")
    
    # Discard burn-in samples
    burn_in = int(n_samples * burn_in_fraction)
    post_burn_chain = chain[burn_in:, :]
    
    # Convert to arviz InferenceData
    data = az.convert_to_inference_data(post_burn_chain, var_names=var_names)
    
    # Calculate ESS
    ess = az.ess(data)
    
    # Convert to DataFrame
    ess_df = pd.DataFrame({
        "Parameter": var_names,
        "ESS": [ess.data_vars[param].values[()] for param in var_names],
        "ESS/N": [ess.data_vars[param].values[()] / (n_samples - burn_in) for param in var_names]
    })
    
    return ess_df


def parameter_correlation(chain: np.ndarray, 
                         var_names: List[str] = None,
                         burn_in_fraction: float = 0.5,
                         figsize: Tuple[int, int] = (10, 8),
                         save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Analyze parameter correlations in MCMC chains.

    Parameters
    ----------
    chain : np.ndarray
        MCMC chain with shape (n_samples, n_parameters)
    var_names : List[str], optional
        List of parameter names (default: "param_0", "param_1", etc.)
    burn_in_fraction : float, optional
        Fraction of samples to discard as burn-in (default: 0.5)
    figsize : Tuple[int, int], optional
        Figure size in inches (width, height) (default: (10, 8))
    save_path : str, optional
        Path to save the correlation plot (default: None, no saving)

    Returns
    -------
    pd.DataFrame
        Correlation matrix
    """
    # Get dimensions
    n_samples, n_params = chain.shape
    
    # Create default parameter names if not provided
    if var_names is None:
        var_names = [f"param_{i}" for i in range(n_params)]
    elif len(var_names) != n_params:
        raise ValueError(f"Length of var_names ({len(var_names)}) must match number of parameters ({n_params})")
    
    # Discard burn-in samples
    burn_in = int(n_samples * burn_in_fraction)
    post_burn_chain = chain[burn_in:, :]
    
    # Create DataFrame
    chain_df = pd.DataFrame(post_burn_chain, columns=var_names)
    
    # Calculate correlation matrix
    corr_matrix = chain_df.corr()
    
    # Plot correlation matrix
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0,
               fmt=".2f", linewidths=0.5)
    plt.title("Parameter Correlation Matrix", fontsize=14)
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()
    
    return corr_matrix


def gelman_rubin(chains: List[np.ndarray], 
                var_names: List[str] = None,
                burn_in_fraction: float = 0.5) -> pd.DataFrame:
    """
    Calculate Gelman-Rubin statistic for multiple MCMC chains.

    Parameters
    ----------
    chains : List[np.ndarray]
        List of MCMC chains, each with shape (n_samples, n_parameters)
    var_names : List[str], optional
        List of parameter names (default: "param_0", "param_1", etc.)
    burn_in_fraction : float, optional
        Fraction of samples to discard as burn-in (default: 0.5)

    Returns
    -------
    pd.DataFrame
        DataFrame with Gelman-Rubin statistics for each parameter
    """
    try:
        import arviz as az
    except ImportError:
        warnings.warn("arviz package is required for Gelman-Rubin calculation")
        return pd.DataFrame({"Parameter": var_names, "R_hat": np.nan})
    
    if len(chains) < 2:
        warnings.warn("At least 2 chains are required for Gelman-Rubin statistic")
        if var_names is None:
            var_names = [f"param_{i}" for i in range(chains[0].shape[1])]
        return pd.DataFrame({"Parameter": var_names, "R_hat": np.nan})
    
    # Get dimensions from first chain
    n_samples, n_params = chains[0].shape
    
    # Create default parameter names if not provided
    if var_names is None:
        var_names = [f"param_{i}" for i in range(n_params)]
    elif len(var_names) != n_params:
        raise ValueError(f"Length of var_names ({len(var_names)}) must match number of parameters ({n_params})")
    
    # Discard burn-in samples from all chains
    burn_in = int(n_samples * burn_in_fraction)
    post_burn_chains = [chain[burn_in:, :] for chain in chains]
    
    # Convert to arviz InferenceData
    data_dict = {var_names[i]: np.array([chain[:, i] for chain in post_burn_chains]) 
                for i in range(n_params)}
    data = az.convert_to_inference_data(data_dict)
    
    # Calculate Gelman-Rubin statistic
    rhat = az.rhat(data)
    
    # Convert to DataFrame
    rhat_df = pd.DataFrame({
        "Parameter": var_names,
        "R_hat": [rhat.data_vars[param].values[()] for param in var_names]
    })
    
    return rhat_df


def analyze_mcmc_results(chain: np.ndarray,
                       var_names: List[str],
                       burn_in_fraction: float = 0.5,
                       true_values: Dict[str, float] = None,
                       output_dir: Optional[str] = None) -> Dict:
    """
    Comprehensive analysis of MCMC results.

    Parameters
    ----------
    chain : np.ndarray
        MCMC chain with shape (n_samples, n_parameters)
    var_names : List[str]
        List of parameter names
    burn_in_fraction : float, optional
        Fraction of samples to discard as burn-in (default: 0.5)
    true_values : Dict[str, float], optional
        Dictionary of true parameter values for reference
    output_dir : str, optional
        Directory to save plots and results (default: None, no saving)

    Returns
    -------
    Dict
        Dictionary with analysis results
    """
    # Create output directory if needed
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    # Summary
    print("\n--- MCMC Analysis Summary ---\n")
    
    print("1. Posterior Summary Statistics")
    summary_path = os.path.join(output_dir, "posterior_summary.png") if output_dir else None
    summary_df = summarize_posterior(chain, var_names, burn_in_fraction, 
                                    true_values=true_values, save_path=summary_path)
    print(summary_df)
    
    # Trace Plots
    print("\n2. Trace Plots")
    trace_path = os.path.join(output_dir, "trace_plots.png") if output_dir else None
    fig_trace = trace_plots(chain, var_names, true_values=true_values, save_path=trace_path)
    
    # ACF/PACF Plots
    print("\n3. Autocorrelation Analysis")
    acf_path = os.path.join(output_dir, "acf_pacf_plots.png") if output_dir else None
    fig_acf = acf_pacf_plots(chain, var_names, burn_in_fraction=burn_in_fraction, save_path=acf_path)
    
    # Parameter Correlations
    print("\n4. Parameter Correlations")
    corr_path = os.path.join(output_dir, "parameter_correlation.png") if output_dir else None
    corr_matrix = parameter_correlation(chain, var_names, burn_in_fraction=burn_in_fraction, save_path=corr_path)
    
    # Save summary to CSV if requested
    if output_dir:
        summary_df.to_csv(os.path.join(output_dir, "posterior_summary.csv"), index=False)
        corr_matrix.to_csv(os.path.join(output_dir, "parameter_correlation.csv"))
    
    # Return results dictionary
    results = {
        "summary": summary_df,
        "correlation_matrix": corr_matrix,
        "n_samples": chain.shape[0],
        "burn_in": int(chain.shape[0] * burn_in_fraction),
        "trace_plot": fig_trace,
        "acf_pacf_plot": fig_acf
    }
    
    # Try to calculate effective sample size
    try:
        import arviz as az
        ess_df = effective_sample_size(chain, var_names, burn_in_fraction)
        print("\n5. Effective Sample Size")
        print(ess_df)
        results["ess"] = ess_df
        if output_dir:
            ess_df.to_csv(os.path.join(output_dir, "effective_sample_size.csv"), index=False)
    except ImportError:
        print("\nNote: Install arviz package for effective sample size calculation")
    
    print("\n--- Analysis Complete ---")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Diagnostic tools for MCMC analysis in Tephra2 inversion.")
    
    # Example: Simulate MCMC chain
    np.random.seed(42)
    n_samples = 5000
    n_params = 3
    
    # Parameters with different autocorrelation properties
    chain = np.zeros((n_samples, n_params))
    chain[0] = np.random.normal(0, 1, n_params)
    
    # AR(1) process with different coefficients
    ar_coeffs = [0.9, 0.5, 0.1]
    for i in range(1, n_samples):
        for j in range(n_params):
            chain[i, j] = ar_coeffs[j] * chain[i-1, j] + np.random.normal(0, np.sqrt(1 - ar_coeffs[j]**2))
    
    # Add drift to first parameter
    chain[:, 0] += np.linspace(0, 2, n_samples)
    
    # Names and true values
    var_names = ["Plume Height", "Log Mass", "Alpha"]
    true_values = {"Plume Height": 1.0, "Log Mass": 0.0, "Alpha": 0.0}
    
    # Analyze
    results = analyze_mcmc_results(chain, var_names, burn_in_fraction=0.3, true_values=true_values)