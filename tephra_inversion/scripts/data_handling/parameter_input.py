# scripts/data_handling/parameter_input.py

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Optional
import yaml

class ParameterHandler:
    def __init__(self, output_dir: Union[str, Path] = "data/input"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default parameter ranges
        self.default_ranges = {
            "column_height": [5000, 20000],  # meters
            "log_m": [20, 25],              # log10(kg)
            "alpha": [1, 5],                # unitless
            "beta": [1, 3],                 # unitless
        }
        
        # Default fixed parameters
        self.default_fixed = {
            "diffusion_coefficient": 1000,
            "fall_time_threshold": 100
        }
    
    def get_user_input(self) -> Dict:
        """Get user input for parameter configuration."""
        print("\nParameter Configuration Options:")
        print("1. Use default parameters")
        print("2. Input custom parameters")
        print("3. Run LHS search to find optimal parameters")
        
        choice = input("Select option (1-3): ")
        
        if choice == "1":
            return self._get_default_params()
        elif choice == "2":
            return self._get_custom_params()
        elif choice == "3":
            return self._get_lhs_params()
        else:
            print("Invalid choice. Please try again.")
            return self.get_user_input()
    
    def _get_default_params(self) -> Dict:
        """Get default parameter configuration."""
        return {
            "params": self._create_param_dict(self.default_ranges),
            "fixed": self.default_fixed,
            "use_lhs": False
        }
    
    def _get_custom_params(self) -> Dict:
        """Get custom parameter configuration from user."""
        params = {}
        fixed = {}
        
        print("\nEnter parameters (one per line, format: name value [fixed]):")
        print("Enter 'done' when finished")
        print("Example: column_height 7500")
        print("Example: diffusion_coefficient 1000 fixed")
        
        while True:
            line = input("> ")
            if line.lower() == 'done':
                break
            
            try:
                parts = line.split()
                name = parts[0]
                value = float(parts[1])
                
                if len(parts) > 2 and parts[2].lower() == 'fixed':
                    fixed[name] = value
                else:
                    params[name] = {
                        "initial_val": value,
                        "prior_type": "Gaussian",
                        "prior_para_a": value,
                        "prior_para_b": value * 0.1,  # 10% of value
                        "draw_scale": value * 0.01    # 1% of value
                    }
            except (ValueError, IndexError):
                print("Invalid input. Please try again.")
        
        return {
            "params": params,
            "fixed": fixed,
            "use_lhs": False
        }
    
    def _get_lhs_params(self) -> Dict:
        """Get parameters through LHS search."""
        print("\nLHS Search Configuration:")
        n_samples = int(input("Enter number of LHS samples (default: 1000): ") or "1000")
        top_frac = float(input("Enter fraction of top samples to keep (default: 0.05): ") or "0.05")
        
        return {
            "params": self._create_param_dict(self.default_ranges),
            "fixed": self.default_fixed,
            "use_lhs": True,
            "lhs_config": {
                "n_samples": n_samples,
                "top_frac": top_frac
            }
        }
    
    def _create_param_dict(self, ranges: Dict) -> Dict:
        """Create parameter dictionary from ranges."""
        params = {}
        for name, (min_val, max_val) in ranges.items():
            mid_val = (min_val + max_val) / 2
            params[name] = {
                "initial_val": mid_val,
                "prior_type": "Uniform",
                "prior_para_a": min_val,
                "prior_para_b": max_val,
                "draw_scale": (max_val - min_val) * 0.01
            }
        return params
    
    def save_parameters(self, config: Dict, filename: str = "esp_input.csv") -> None:
        """Save parameter configuration to file."""
        # If filename is a full path, use it directly
        if os.path.isabs(filename):
            output_path = Path(filename)
        else:
            output_path = self.output_dir / filename
            
        # Ensure the directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create DataFrame for parameters
        rows = []
        for name, param in config["params"].items():
            rows.append({
                "variable_name": name,
                "initial_val": param["initial_val"],
                "prior_type": param["prior_type"],
                "prior_para_a": param["prior_para_a"],
                "prior_para_b": param["prior_para_b"],
                "draw_scale": param["draw_scale"]
            })
        
        # Add fixed parameters
        for name, value in config["fixed"].items():
            rows.append({
                "variable_name": name,
                "initial_val": value,
                "prior_type": "Fixed",
                "prior_para_a": "",
                "prior_para_b": "",
                "draw_scale": ""
            })
        
        # Save to CSV
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        print(f"\nParameters saved to: {output_path}")
    
    def save_config(self, config: Dict, filename: str = "config.yaml") -> None:
        """Save configuration to YAML file."""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f)
        
        print(f"\nConfiguration saved to: {output_path}")

    def load_parameters(self, filename: str = "esp_input.csv") -> Dict:
        """
        Load parameter configuration from file.
        
        Parameters
        ----------
        filename : str
            Name of the parameter file
            
        Returns
        -------
        Dict
            Dictionary containing parameter configuration
        """
        filepath = self.output_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Parameter file not found: {filepath}")
            
        try:
            # Read the parameter file
            df = pd.read_csv(filepath)
            
            # Separate parameters and fixed values
            params = {}
            fixed = {}
            
            for _, row in df.iterrows():
                param_dict = {
                    "initial_val": row["initial_val"],
                    "prior_type": row["prior_type"],
                    "prior_para_a": row["prior_para_a"],
                    "prior_para_b": row["prior_para_b"],
                    "draw_scale": row["draw_scale"]
                }
                
                if row["prior_type"] == "Fixed":
                    fixed[row["variable_name"]] = row["initial_val"]
                else:
                    params[row["variable_name"]] = param_dict
            
            return {
                "params": params,
                "fixed": fixed,
                "use_lhs": False
            }
            
        except Exception as e:
            raise ValueError(f"Error loading parameters: {e}")

def main():
    """Main function to handle parameter configuration."""
    handler = ParameterHandler()
    
    # Get user input
    config = handler.get_user_input()
    
    # Save parameters
    handler.save_parameters(config)
    
    # Save configuration
    handler.save_config(config)
    
    return config

if __name__ == "__main__":
    main()