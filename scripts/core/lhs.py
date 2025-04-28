import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
from tqdm import tqdm

# Try importing pyDOE for LHS
try:
    from pyDOE import lhs
    PYDOE_AVAILABLE = True
except ImportError:
    PYDOE_AVAILABLE = False
    import warnings
    warnings.warn("pyDOE not available; using numpy for LHS sampling")

class LatinHypercubeSampler:
    def __init__(self, 
                 output_dir: Union[str, Path] = "output/lhs",
                 seed: Optional[int] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def generate_samples(self,
                        n_samples: int,
                        param_ranges: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
        """Generate Latin Hypercube samples from parameter space."""
        param_names = list(param_ranges.keys())
        n_params = len(param_names)
        
        # Generate LHS samples
        if PYDOE_AVAILABLE:
            samples = lhs(n_params, samples=n_samples, criterion='maximin', iterations=10)
        else:
            cut = np.linspace(0, 1, n_samples + 1)
            pts = (cut[1:] + cut[:-1]) / 2
            samples = np.zeros((n_samples, n_params))
            for i in range(n_params):
                samples[:, i] = np.random.permutation(pts)
        
        # Scale samples to parameter ranges
        for i, param_name in enumerate(param_names):
            min_val, max_val = param_ranges[param_name]
            samples[:, i] = min_val + samples[:, i] * (max_val - min_val)
        
        # Convert to DataFrame
        df = pd.DataFrame(samples, columns=param_names)
        return df
    
    def evaluate_samples(self,
                        samples_df: pd.DataFrame,
                        forward_model: callable,
                        fixed_params: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """Evaluate forward model on LHS samples."""
        if fixed_params:
            for param, value in fixed_params.items():
                samples_df[param] = value
        
        # Add posterior column
        samples_df['posterior'] = np.nan
        
        # Evaluate each sample
        for i, row in tqdm(samples_df.iterrows(), total=len(samples_df)):
            params = row.to_dict()
            if 'posterior' in params:
                del params['posterior']
            
            posterior = forward_model(params)
            samples_df.at[i, 'posterior'] = posterior
        
        return samples_df
    
    def find_best_samples(self,
                         samples_df: pd.DataFrame,
                         top_fraction: float = 0.05) -> pd.DataFrame:
        """Identify samples with highest posterior probabilities."""
        sorted_df = samples_df.sort_values('posterior', ascending=False)
        n_top = max(1, int(len(sorted_df) * top_fraction))
        return sorted_df.head(n_top)
    
    def refine_parameter_space(self,
                             top_samples: pd.DataFrame,
                             n_points: int = 20) -> Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]:
        """Generate refined grid around best samples."""
        # Identify varying parameters
        varying_params = []
        for col in top_samples.columns:
            if col != 'posterior' and top_samples[col].nunique() > 1:
                varying_params.append(col)
        
        if len(varying_params) < 2:
            return {}
        
        # Focus on two most important parameters
        if len(varying_params) > 2:
            param_ranges = {}
            for param in varying_params:
                min_val = top_samples[param].min()
                max_val = top_samples[param].max()
                range_val = max_val - min_val
                global_min = top_samples[param].min()
                global_max = top_samples[param].max()
                if abs(global_max - global_min) > 1e-10:
                    normalized_range = range_val / (global_max - global_min)
                else:
                    normalized_range = 0
                param_ranges[param] = normalized_range
            
            sorted_params = sorted(param_ranges.items(), key=lambda x: x[1], reverse=True)
            varying_params = [p[0] for p in sorted_params[:2]]
        
        # Create grid for parameters
        grid_dict = {}
        for i, param1 in enumerate(varying_params):
            for param2 in varying_params[i+1:]:
                min1, max1 = top_samples[param1].min(), top_samples[param1].max()
                min2, max2 = top_samples[param2].min(), top_samples[param2].max()
                
                pad1 = 0.05 * (max1 - min1)
                pad2 = 0.05 * (max2 - min2)
                min1 -= pad1
                max1 += pad1
                min2 -= pad2
                max2 += pad2
                
                grid1 = np.linspace(min1, max1, n_points)
                grid2 = np.linspace(min2, max2, n_points)
                
                grid_dict[(param1, param2)] = (grid1, grid2)
        
        return grid_dict
    
    def evaluate_grid(self,
                     grid_dict: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]],
                     base_params: Dict[str, float],
                     forward_model: callable) -> Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Evaluate posterior on refined grid."""
        result_dict = {}
        
        for (param1, param2), (grid1, grid2) in grid_dict.items():
            X, Y = np.meshgrid(grid1, grid2)
            Z = np.zeros_like(X)
            
            for i in range(len(grid1)):
                for j in range(len(grid2)):
                    params = base_params.copy()
                    params[param1] = grid1[i]
                    params[param2] = grid2[j]
                    Z[j, i] = forward_model(params)
            
            result_dict[(param1, param2)] = (grid1, grid2, Z)
        
        return result_dict
    
    def find_best_parameters(self,
                           grid_results: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray, np.ndarray]],
                           base_params: Dict[str, float]) -> Tuple[Dict[str, float], float]:
        """Find best parameters from grid results."""
        best_posterior = -np.inf
        best_params = base_params.copy()
        
        for (param1, param2), (grid1, grid2, Z) in grid_results.items():
            best_idx = np.unravel_index(np.argmax(Z), Z.shape)
            j, i = best_idx
            grid_posterior = Z[j, i]
            
            if grid_posterior > best_posterior:
                best_posterior = grid_posterior
                best_params[param1] = grid1[i]
                best_params[param2] = grid2[j]
        
        return best_params, best_posterior

def main():
    """Example usage of LatinHypercubeSampler."""
    # Example parameter ranges
    param_ranges = {
        'column_height': (5000, 20000),
        'log_m': (20, 25),
        'alpha': (1, 5),
        'beta': (2, 5)
    }
    
    # Fixed parameters
    fixed_params = {
        'vent_easting': 0.0,
        'vent_northing': 0.0,
        'vent_elevation': 0.0
    }
    
    # Create sampler
    sampler = LatinHypercubeSampler(seed=42)
    
    # Generate samples
    samples_df = sampler.generate_samples(n_samples=1000, param_ranges=param_ranges)
    
    # Example forward model (replace with actual Tephra2 interface)
    def forward_model(params):
        return np.random.normal(0, 1)  # Placeholder
    
    # Evaluate samples
    samples_df = sampler.evaluate_samples(samples_df, forward_model, fixed_params)
    
    # Find best samples
    top_samples = sampler.find_best_samples(samples_df)
    
    # Refine parameter space
    grid_dict = sampler.refine_parameter_space(top_samples)
    
    if grid_dict:
        # Base parameters: median of top samples
        base_params = {}
        for col in top_samples.columns:
            if col != 'posterior':
                base_params[col] = top_samples[col].median()
        
        # Evaluate grid
        grid_results = sampler.evaluate_grid(grid_dict, base_params, forward_model)
        
        # Find best parameters
        best_params, best_posterior = sampler.find_best_parameters(grid_results, base_params)
        
        print("\nBest parameters:")
        for param, value in best_params.items():
            print(f"{param}: {value:.4f}")
        print(f"Log posterior: {best_posterior:.4f}")

if __name__ == "__main__":
    main()
