import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import matplotlib.gridspec as gridspec
from scipy import stats
import arviz as az
import matplotlib.colors as mcolors
import matplotlib.cm as cm

class DiagnosticPlotter:
    def __init__(self, output_dir: str = "data/output/plots"):
        """Initialize diagnostic plotter with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn')
        plt.rc('font', size=12)
        plt.rc('figure', titlesize=14)
        plt.rc('axes', labelsize=12)
        plt.rc('axes', titlesize=14)
    
    def plot_trace(self,
                  samples: Dict[str, np.ndarray],
                  burnin: int = 0,
                  title: str = "MCMC Trace Plots",
                  save_path: Optional[str] = None) -> None:
        """
        Plot trace plots for MCMC parameters.
        
        Parameters
        ----------
        samples : Dict[str, np.ndarray]
            Dictionary of parameter names and their MCMC samples
        burnin : int
            Number of burn-in samples to exclude
        title : str
            Plot title
        save_path : Optional[str]
            Path to save the plot
        """
        n_params = len(samples)
        fig, axes = plt.subplots(n_params, 1, figsize=(10, 3*n_params))
        if n_params == 1:
            axes = [axes]
        
        for ax, (param_name, param_samples) in zip(axes, samples.items()):
            ax.plot(param_samples[burnin:], alpha=0.7)
            ax.set_ylabel(param_name)
            ax.grid(True)
        
        axes[-1].set_xlabel("Iteration")
        fig.suptitle(title)
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "trace_plots.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_autocorrelation(self,
                           samples: Dict[str, np.ndarray],
                           max_lag: int = 100,
                           title: str = "Autocorrelation Plots",
                           save_path: Optional[str] = None) -> None:
        """
        Plot autocorrelation for MCMC parameters.
        
        Parameters
        ----------
        samples : Dict[str, np.ndarray]
            Dictionary of parameter names and their MCMC samples
        max_lag : int
            Maximum lag to compute autocorrelation
        title : str
            Plot title
        save_path : Optional[str]
            Path to save the plot
        """
        n_params = len(samples)
        fig, axes = plt.subplots(n_params, 1, figsize=(10, 3*n_params))
        if n_params == 1:
            axes = [axes]
        
        for ax, (param_name, param_samples) in zip(axes, samples.items()):
            acf = np.correlate(param_samples, param_samples, mode='full')[len(param_samples)-1:]
            acf = acf[:max_lag] / acf[0]
            ax.bar(range(max_lag), acf, alpha=0.7)
            ax.set_ylabel(param_name)
            ax.grid(True)
        
        axes[-1].set_xlabel("Lag")
        fig.suptitle(title)
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "autocorrelation_plots.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_parameter_distributions(self,
                                   samples: Dict[str, np.ndarray],
                                   burnin: int = 0,
                                   title: str = "Parameter Distributions",
                                   save_path: Optional[str] = None) -> None:
        """
        Plot parameter distributions with KDE and histograms.
        
        Parameters
        ----------
        samples : Dict[str, np.ndarray]
            Dictionary of parameter names and their MCMC samples
        burnin : int
            Number of burn-in samples to exclude
        title : str
            Plot title
        save_path : Optional[str]
            Path to save the plot
        """
        n_params = len(samples)
        fig, axes = plt.subplots(n_params, 1, figsize=(10, 3*n_params))
        if n_params == 1:
            axes = [axes]
        
        for ax, (param_name, param_samples) in zip(axes, samples.items()):
            sns.histplot(param_samples[burnin:], ax=ax, kde=True, stat="density")
            ax.set_xlabel(param_name)
            ax.grid(True)
        
        fig.suptitle(title)
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "parameter_distributions.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_parameter_correlations(self,
                                  samples: Dict[str, np.ndarray],
                                  burnin: int = 0,
                                  title: str = "Parameter Correlations",
                                  save_path: Optional[str] = None) -> None:
        """
        Plot parameter correlation matrix and scatter plots.
        
        Parameters
        ----------
        samples : Dict[str, np.ndarray]
            Dictionary of parameter names and their MCMC samples
        burnin : int
            Number of burn-in samples to exclude
        title : str
            Plot title
        save_path : Optional[str]
            Path to save the plot
        """
        # Create DataFrame from samples
        df = pd.DataFrame({k: v[burnin:] for k, v in samples.items()})
        
        # Create figure with two subplots
        fig = plt.figure(figsize=(15, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])
        
        # Correlation matrix
        ax1 = fig.add_subplot(gs[0])
        corr = df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax1)
        ax1.set_title("Correlation Matrix")
        
        # Scatter plot matrix
        ax2 = fig.add_subplot(gs[1])
        pd.plotting.scatter_matrix(df, alpha=0.2, ax=ax2)
        
        fig.suptitle(title)
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "parameter_correlations.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_tephra_distribution_comparison(self,
                                          observed: np.ndarray,
                                          predicted: np.ndarray,
                                          title: str = "Tephra Distribution Comparison",
                                          save_path: Optional[str] = None) -> None:
        """
        Plot comparison between observed and predicted tephra distributions.
        
        Parameters
        ----------
        observed : np.ndarray
            Array of observed deposit thicknesses
        predicted : np.ndarray
            Array of predicted deposit thicknesses
        title : str
            Plot title
        save_path : Optional[str]
            Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram comparison
        bins = np.logspace(np.log10(min(observed.min(), predicted.min())),
                          np.log10(max(observed.max(), predicted.max())),
                          50)
        ax1.hist(observed, bins=bins, alpha=0.5, label='Observed', density=True)
        ax1.hist(predicted, bins=bins, alpha=0.5, label='Predicted', density=True)
        ax1.set_xscale('log')
        ax1.set_xlabel('Deposit Thickness (kg/m²)')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True)
        
        # QQ plot
        observed_sorted = np.sort(observed)
        predicted_sorted = np.sort(predicted)
        ax2.scatter(observed_sorted, predicted_sorted, alpha=0.5)
        min_val = min(observed.min(), predicted.min())
        max_val = max(observed.max(), predicted.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('Observed Thickness (kg/m²)')
        ax2.set_ylabel('Predicted Thickness (kg/m²)')
        ax2.grid(True)
        
        fig.suptitle(title)
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "tephra_distribution_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_parameter_evolution(self,
                               samples: Dict[str, np.ndarray],
                               window_size: int = 100,
                               title: str = "Parameter Evolution",
                               save_path: Optional[str] = None) -> None:
        """
        Plot parameter evolution with rolling mean and confidence intervals.
        
        Parameters
        ----------
        samples : Dict[str, np.ndarray]
            Dictionary of parameter names and their MCMC samples
        window_size : int
            Size of rolling window for mean and std calculation
        title : str
            Plot title
        save_path : Optional[str]
            Path to save the plot
        """
        n_params = len(samples)
        fig, axes = plt.subplots(n_params, 1, figsize=(10, 3*n_params))
        if n_params == 1:
            axes = [axes]
        
        for ax, (param_name, param_samples) in zip(axes, samples.items()):
            # Calculate rolling statistics
            df = pd.DataFrame(param_samples)
            rolling_mean = df.rolling(window=window_size).mean()
            rolling_std = df.rolling(window=window_size).std()
            
            # Plot
            ax.plot(rolling_mean, label='Rolling Mean', color='blue')
            ax.fill_between(range(len(param_samples)),
                          rolling_mean[0] - 2*rolling_std[0],
                          rolling_mean[0] + 2*rolling_std[0],
                          alpha=0.2, color='blue')
            ax.set_ylabel(param_name)
            ax.grid(True)
            ax.legend()
        
        axes[-1].set_xlabel("Iteration")
        fig.suptitle(title)
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "parameter_evolution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_effective_sample_size(self,
                                 samples: Dict[str, np.ndarray],
                                 title: str = "Effective Sample Size",
                                 save_path: Optional[str] = None) -> None:
        """
        Plot effective sample size for each parameter.
        
        Parameters
        ----------
        samples : Dict[str, np.ndarray]
            Dictionary of parameter names and their MCMC samples
        title : str
            Plot title
        save_path : Optional[str]
            Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate effective sample size
        ess = {param: az.ess(sample) for param, sample in samples.items()}
        
        # Plot
        ax.bar(ess.keys(), ess.values())
        ax.set_ylabel("Effective Sample Size")
        ax.set_title(title)
        ax.grid(True)
        
        if save_path is None:
            save_path = self.output_dir / "effective_sample_size.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_r_hat(self,
                  samples: Dict[str, np.ndarray],
                  title: str = "R-hat Diagnostic",
                  save_path: Optional[str] = None) -> None:
        """
        Plot R-hat diagnostic for each parameter.
        
        Parameters
        ----------
        samples : Dict[str, np.ndarray]
            Dictionary of parameter names and their MCMC samples
        title : str
            Plot title
        save_path : Optional[str]
            Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate R-hat
        r_hat = {param: az.rhat(sample) for param, sample in samples.items()}
        
        # Plot
        ax.bar(r_hat.keys(), r_hat.values())
        ax.axhline(y=1.1, color='r', linestyle='--', label='Threshold')
        ax.set_ylabel("R-hat")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        if save_path is None:
            save_path = self.output_dir / "r_hat.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Example usage of DiagnosticPlotter."""
    # Create plotter instance
    plotter = DiagnosticPlotter()
    
    # Generate example MCMC samples
    n_samples = 10000
    samples = {
        'column_height': np.random.normal(10000, 1000, n_samples),
        'log_m': np.random.normal(27.5, 0.5, n_samples),
        'alpha': np.random.normal(4, 0.5, n_samples),
        'beta': np.random.normal(2, 0.3, n_samples)
    }
    
    # Generate example tephra data
    n_points = 1000
    observed = np.random.lognormal(0, 1, n_points)
    predicted = np.random.lognormal(0.1, 1.1, n_points)
    
    # Create all diagnostic plots
    plotter.plot_trace(samples)
    plotter.plot_autocorrelation(samples)
    plotter.plot_parameter_distributions(samples)
    plotter.plot_parameter_correlations(samples)
    plotter.plot_tephra_distribution_comparison(observed, predicted)
    plotter.plot_parameter_evolution(samples)
    plotter.plot_effective_sample_size(samples)
    plotter.plot_r_hat(samples)

if __name__ == "__main__":
    main()
