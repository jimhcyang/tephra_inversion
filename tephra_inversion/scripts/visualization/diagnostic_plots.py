# scripts/visualization/diagnostic_plots.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import matplotlib.gridspec as gridspec

class DiagnosticPlotter:
    def __init__(self, output_dir: str = "data/output/plots"):
        """Initialize diagnostic plotter with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use("seaborn-v0_8")
        plt.rc('font', size=12)
        plt.rc('figure', titlesize=14)
        plt.rc('axes', labelsize=12)
        plt.rc('axes', titlesize=14)
        
    def plot_trace(self,
                samples: Dict[str, np.ndarray],
                burnin: int = 0,
                title: str = "MCMC Trace Plots",
                save_path: Optional[Union[str, Path]] = None,
                show_plot: bool = True) -> str:
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
        save_path : Optional[str or Path]
            Path to save the plot
        show_plot : bool
            Whether to display the plot interactively (default: True)
            
        Returns
        -------
        str
            Path where the plot was saved
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
        
        # Set save path
        if save_path is None:
            save_path = self.output_dir / "trace_plots.png"
        else:
            save_path = Path(save_path)
            
        # Make sure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show the plot if requested (default is now True)
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        return str(save_path)


    def plot_parameter_distributions(self,
                                samples: Dict[str, np.ndarray],
                                burnin: int = 0,
                                title: str = "Parameter Distributions",
                                save_path: Optional[Union[str, Path]] = None,
                                show_plot: bool = True) -> str:
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
        save_path : Optional[str or Path]
            Path to save the plot
        show_plot : bool
            Whether to display the plot interactively (default: True)
            
        Returns
        -------
        str
            Path where the plot was saved
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
        
        # Set save path
        if save_path is None:
            save_path = self.output_dir / "parameter_distributions.png"
        else:
            save_path = Path(save_path)
            
        # Make sure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show the plot if requested (default is now True)
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        return str(save_path)


    def plot_parameter_correlations(self,
                                samples: Dict[str, np.ndarray],
                                burnin: int = 0,
                                title: str = "Parameter Correlations",
                                save_path: Optional[Union[str, Path]] = None,
                                show_plot: bool = True) -> str:
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
        save_path : Optional[str or Path]
            Path to save the plot
        show_plot : bool
            Whether to display the plot interactively (default: True)
            
        Returns
        -------
        str
            Path where the plot was saved
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
        
        # Set save path
        if save_path is None:
            save_path = self.output_dir / "parameter_correlations.png"
        else:
            save_path = Path(save_path)
            
        # Make sure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show the plot if requested (default is now True)
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        return str(save_path)


    def plot_tephra_distribution_comparison(self,
                                        observed: np.ndarray,
                                        predicted: np.ndarray,
                                        title: str = "Tephra Distribution Comparison",
                                        save_path: Optional[Union[str, Path]] = None,
                                        show_plot: bool = True) -> str:
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
        save_path : Optional[str or Path]
            Path to save the plot
        show_plot : bool
            Whether to display the plot interactively (default: True)
            
        Returns
        -------
        str
            Path where the plot was saved
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
        
        # Set save path
        if save_path is None:
            save_path = self.output_dir / "tephra_distribution_comparison.png"
        else:
            save_path = Path(save_path)
            
        # Make sure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show the plot if requested (default is now True)
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        return str(save_path)