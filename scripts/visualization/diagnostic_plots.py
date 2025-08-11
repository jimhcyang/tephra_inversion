# scripts/visualization/diagnostic_plots.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import matplotlib.gridspec as gridspec
from scripts.data_handling.config_io import load_config

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
        
        # Load default config to get true values
        self.default_config = load_config()
        self.true_values = self.default_config["parameters"]["true_values"]
        
        # Dictionary to map parameter names to pretty labels
        self.pretty_labels = {
            'column_height': "Plume Height (m)",
            'log_m': "Log Eruption Mass (ln)",
            'max_grainsize': "Max Grain Size (φ)",
            'min_grainsize': "Min Grain Size (φ)",
            'median_grainsize': "Median Grain Size (φ)",
            'std_grainsize': "Std. Dev. Grain Size (φ)",
            'eddy_const': "Eddy Constant",
            'diffusion_coefficient': "Diffusion Coefficient",
            'fall_time_threshold': "Fall Time Threshold",
            'lithic_density': "Lithic Density (kg/m³)",
            'pumice_density': "Pumice Density (kg/m³)",
            'col_steps': "Column Steps (#)",
            'part_steps': "Particle Bins (#)",
            'plume_model': "Plume Model",
            'alpha': "Alpha",
            'beta': "Beta"
        }
        
    def _get_relevant_parameters(self, samples: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Filter parameters to only include those in true_values and those that vary.
        
        Parameters
        ----------
        samples : Dict[str, np.ndarray]
            Dictionary of parameter names and their MCMC samples
            
        Returns
        -------
        Dict[str, np.ndarray]
            Filtered parameter dictionary
        """
        # Map between esp parameters and config parameters
        param_mapping = {
            'column_height': 'plume_height',
            'log_m': 'eruption_mass'
        }
        
        # Get all keys from true_values
        true_keys = set(self.true_values.keys())
        
        # Filter parameters
        filtered_params = {}
        for param_name, param_samples in samples.items():
            # Check if parameter is in true_values directly or through mapping
            mapped_name = param_mapping.get(param_name, param_name)
            
            if mapped_name in true_keys:
                # Also check if this parameter actually varies
                if np.min(param_samples) != np.max(param_samples):
                    filtered_params[param_name] = param_samples
        
        return filtered_params
    
    def _get_true_value(self, param_name: str) -> float:
        """
        Get true value for a parameter, handling special cases like log_m.
        
        Parameters
        ----------
        param_name : str
            Parameter name
            
        Returns
        -------
        float
            True value or None if not found
        """
        if param_name == 'column_height':
            return self.true_values.get('plume_height')
        elif param_name == 'log_m':
            eruption_mass = self.true_values.get('eruption_mass')
            if eruption_mass is not None:
                return np.log(eruption_mass)
        
        return self.true_values.get(param_name)
    
    def _get_pretty_label(self, param_name: str) -> str:
        """
        Get a pretty label for a parameter.
        
        Parameters
        ----------
        param_name : str
            Parameter name
            
        Returns
        -------
        str
            Pretty label or the original name if not found
        """
        return self.pretty_labels.get(param_name, param_name)
        
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
        # Filter to relevant parameters
        relevant_params = self._get_relevant_parameters(samples)
        
        if not relevant_params:
            print("No relevant parameters found for trace plots.")
            return None
            
        n_params = len(relevant_params)
        fig, axes = plt.subplots(n_params, 1, figsize=(10, 3*n_params))
        if n_params == 1:
            axes = [axes]
        
        for ax, (param_name, param_samples) in zip(axes, relevant_params.items()):
            post_burn_samples = param_samples[burnin:]
            ax.plot(post_burn_samples, alpha=0.7)
            
            # Add median line
            median_val = np.median(post_burn_samples)
            ax.axhline(median_val, color='red', linestyle='--', 
                       label=f"Median: {median_val:.2f}")
            
            # Add true value if available
            true_val = self._get_true_value(param_name)
            if true_val is not None:
                ax.axhline(true_val, color='green', linestyle='-', 
                         label=f"True: {true_val:.2f}")
            
            ax.set_ylabel(self._get_pretty_label(param_name))
            ax.legend(loc='best')
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
        
        # Show the plot if requested
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
        Plot parameter distributions with KDE, histograms, true values and credible intervals.
        
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
        # Filter to relevant parameters
        relevant_params = self._get_relevant_parameters(samples)
        
        if not relevant_params:
            print("No relevant parameters found for distribution plots.")
            return None
            
        n_params = len(relevant_params)
        fig, axes = plt.subplots(n_params, 1, figsize=(10, 3*n_params))
        if n_params == 1:
            axes = [axes]
        
        # Create DataFrame for summary statistics
        summary_data = []
        
        for ax, (param_name, param_samples) in zip(axes, relevant_params.items()):
            post_burn_samples = param_samples[burnin:]
            
            # Calculate statistics
            median = np.median(post_burn_samples)
            ci_low = np.percentile(post_burn_samples, 2.5)
            ci_high = np.percentile(post_burn_samples, 97.5)
            mean = np.mean(post_burn_samples)
            
            # Add to summary data
            summary_data.append({
                'Parameter': param_name,
                'Mean': mean,
                'Median': median,
                'CI_2.5%': ci_low,
                'CI_97.5%': ci_high
            })
            
            # Plot histogram with KDE
            sns.histplot(post_burn_samples, kde=True, stat="density", ax=ax, color="steelblue")
            
            # Add median and CI lines
            ax.axvline(median, color='red', linestyle='--', 
                       label=f"Median: {median:.2f}")
            ax.axvline(ci_low, color='orange', linestyle=':', 
                       label=f"2.5%: {ci_low:.2f}")
            ax.axvline(ci_high, color='orange', linestyle=':', 
                       label=f"97.5%: {ci_high:.2f}")
            
            # Add true value if available
            true_val = self._get_true_value(param_name)
            if true_val is not None:
                ax.axvline(true_val, color='green', linestyle='-', linewidth=2, 
                         label=f"True: {true_val:.2f}")
            
            ax.set_xlabel(self._get_pretty_label(param_name))
            ax.set_ylabel("Density")
            ax.set_title(f"Posterior Distribution: {self._get_pretty_label(param_name)}")
            ax.legend(loc='best')
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
        
        # Create and display summary table
        summary_df = pd.DataFrame(summary_data)
        print("\nPosterior Summary Statistics:")
        print(summary_df)
        
        # Show the plot if requested
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
        Plot parameter correlation matrix and joint posteriors for parameters with true values.
        
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
        # Filter to relevant parameters
        relevant_params = self._get_relevant_parameters(samples)
        
        if not relevant_params:
            print("No relevant parameters found for correlation plots.")
            return None
        
        if len(relevant_params) < 2:
            print("Need at least 2 parameters for correlation plots.")
            return None
            
        # Create DataFrame from samples with pretty labels as column names
        df = pd.DataFrame({
            self._get_pretty_label(k): v[burnin:] for k, v in relevant_params.items()
        })
        
        # Create figure with two subplots
        fig = plt.figure(figsize=(15, 8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])
        
        # Correlation matrix
        ax1 = fig.add_subplot(gs[0])
        corr = df.corr()
        sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, ax=ax1, fmt=".2f")
        ax1.set_title("Correlation Matrix")
        
        # Joint posteriors - create pairplot on the right side
        ax2 = fig.add_subplot(gs[1])
        
        param_names = list(relevant_params.keys())
        param_names_pretty = [self._get_pretty_label(p) for p in param_names]
        
        # If we have exactly 2 parameters, plot a single joint posterior
        if len(relevant_params) == 2:
            x_name, y_name = param_names
            x_samples = relevant_params[x_name][burnin:]
            y_samples = relevant_params[y_name][burnin:]
            
            # Scatter plot of samples
            ax2.scatter(x_samples, y_samples, alpha=0.3, s=10)
            
            # Add true values
            x_true = self._get_true_value(x_name)
            y_true = self._get_true_value(y_name)
            
            if x_true is not None and y_true is not None:
                ax2.scatter([x_true], [y_true], color='green', s=100, marker='s', 
                           label="True Value")
            
            # Add axis labels
            ax2.set_xlabel(self._get_pretty_label(x_name))
            ax2.set_ylabel(self._get_pretty_label(y_name))
            ax2.set_title("Joint Posterior")
            ax2.legend(loc='best')
            ax2.grid(True)
        else:
            # For more than 2 parameters, we'll use seaborn pairplot in a separate figure
            plt.close(fig)
            
            # Create a new figure for the pairplot
            pairgrid = sns.PairGrid(df)
            pairgrid.map_upper(sns.scatterplot, alpha=0.3, s=10)
            pairgrid.map_lower(sns.kdeplot)
            pairgrid.map_diag(sns.histplot, kde=True)
            
            # Add true values as markers
            for i, param1 in enumerate(param_names_pretty):
                for j, param2 in enumerate(param_names_pretty):
                    if i > j:  # Lower triangle only
                        orig_param1 = param_names[j]  # Note the swap for axis
                        orig_param2 = param_names[i]
                        true1 = self._get_true_value(orig_param1)
                        true2 = self._get_true_value(orig_param2)
                        
                        if true1 is not None and true2 is not None:
                            pairgrid.axes[i, j].scatter([true1], [true2], 
                                                       color='green', s=100, marker='s')
            
            pairgrid.fig.suptitle(title)
            plt.tight_layout()
            
            # Set save path
            if save_path is None:
                save_path = self.output_dir / "parameter_correlations.png"
            else:
                save_path = Path(save_path)
                
            # Make sure directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the plot
            pairgrid.fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
            # Show the plot if requested
            if show_plot:
                plt.show()
            else:
                plt.close()
                
            return str(save_path)
        
        # If we only had 2 parameters, save the original figure
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
        
        # Show the plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        return str(save_path)
        
    def summarize_posterior(self, 
                          samples: Dict[str, np.ndarray],
                          burnin: int = 0,
                          save_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Generate a summary table of posterior statistics and save it to a CSV file.
        
        Parameters
        ----------
        samples : Dict[str, np.ndarray]
            Dictionary of parameter names and their MCMC samples
        burnin : int
            Number of burn-in samples to exclude
        save_path : Optional[str or Path]
            Path to save the summary table
            
        Returns
        -------
        pd.DataFrame
            Summary statistics table
        """
        # Filter to relevant parameters
        relevant_params = self._get_relevant_parameters(samples)
        
        if not relevant_params:
            print("No relevant parameters found for summary.")
            return None
            
        # Calculate statistics for each parameter
        summary_data = []
        
        for param_name, param_samples in relevant_params.items():
            post_burn_samples = param_samples[burnin:]
            
            # Calculate statistics
            mean = np.mean(post_burn_samples)
            median = np.median(post_burn_samples)
            std_dev = np.std(post_burn_samples)
            ci_low = np.percentile(post_burn_samples, 2.5)
            ci_high = np.percentile(post_burn_samples, 97.5)
            
            # Get true value if available
            true_val = self._get_true_value(param_name)
            
            # Add to summary data
            summary_data.append({
                'Parameter': param_name,
                'Pretty Label': self._get_pretty_label(param_name),
                'Mean': mean,
                'Median': median,
                'Std Dev': std_dev,
                'CI_2.5%': ci_low,
                'CI_97.5%': ci_high,
                'True Value': true_val if true_val is not None else 'N/A'
            })
        
        # Create DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Set save path
        if save_path is None:
            save_path = self.output_dir / "posterior_summary.csv"
        else:
            save_path = Path(save_path)
            
        # Make sure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        summary_df.to_csv(save_path, index=False)
        print(f"Summary saved to {save_path}")
        
        return summary_df


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
        bins = np.logspace(np.log(min(observed.min(), predicted.min())),
                        np.log(max(observed.max(), predicted.max())),
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