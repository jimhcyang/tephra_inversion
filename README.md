# Tephra2 Inversion Framework

A compact Python workflow to estimate eruption source parameters (ESPs) for tephra dispersion using **Tephra2** as the forward model and three inversion backends:

- **MCMC (Metropolis–Hastings)**
- **Simulated Annealing (SA)**
- **Ensemble Smoother with Multiple Data Assimilations (EnKF)**

It includes a one‑click dataset prep for the _Cerro Negro_ short‑course bundle and modular plotting utilities so the demo notebook stays minimal.

---

## Quick Start

### 1. Dependencies

    - Python 3.9+
    - A compiled **Tephra2** executable
    - Victor API access and package installation (Optional)

### 2. Build / point to Tephra2

   Clone and compile the Tephra2 executable:

   * **Linux/Victor Environment:**
     ```bash
     # Clone Tephra2 repository
     git clone https://github.com/geoscience-community-codes/Tephra2.git

     # Install required dependencies (if using conda, for example, on VICTOR)
     conda install -y -c conda-forge boehm-gc libatomic_ops

     # Set environment variables for compilation
     export C_INCLUDE_PATH=$CONDA_PREFIX/include
     export LIBRARY_PATH=$CONDA_PREFIX/lib

     # Compile Tephra2
     cd Tephra2
     make clean
     make

     # Return to main directory
     cd ..
     ```

   * **macOS:**
     ```bash
     # Clone Tephra2 repository
     git clone https://github.com/geoscience-community-codes/Tephra2.git

     # Install required dependencies
     brew install bdw-gc libatomic_ops

     export C_INCLUDE_PATH=/opt/homebrew/include
     export LIBRARY_PATH=/opt/homebrew/lib

     # Compile Tephra2
     cd Tephra2
     make clean
     make

     # Return to main directory
     cd ..
     ```
   *Note: Ensure the compiled `tephra2` executable is accessible, potentially by adding its location to your system's PATH or placing it within the project structure (e.g., in a `bin/` directory).* 

3. **Set up Python environment**
    Create and activate a python virtual environment

    ```bash
    cd your/working/directory
    git clone https://github.com/jimhcyang/tephra_inversion.git
    cd tephra_inversion
    python -m venv .venv
    source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

4. **Run the Demo**
    Open the notebook:

    ```bash
    jupyter notebook demo.ipynb
    ```

    The notebook:

    * Shows basic input visuals (observations map & wind profile)
    * Runs **MCMC**, **SA**, and **EnKF** with defaults from `config/default_config.py`
    * Produces diagnostic plots and per‑pass EnKF figures


## Repository Structure

```plaintext
tephra-inversion-workflow/
├── README.md
├── demo.ipynb
├── requirements.txt
├── config/
│   └── default_config.py
├── scripts/
│   ├── tephra_inversion.py
│   ├── config_io.py
│   ├── download_cerro_negro.py
│   ├── core/
│   │   ├── mcmc.py
│   │   ├── sa.py
│   │   ├── enkf.py
│   │   ├── forward.py
│   │   └── tephra2_utils.py
│   └── visualization/
│       ├── diagnostic_plots.py
│       ├── observation_plots.py
│       ├── wind_plots.py
│       ├── summary.py
│       ├── comparative_plots.py
│       └── enkf_pass_plots.py
├── data/
│   ├── input/          # prepared inputs (created by --prepare)
│   └── output/         # chains, plots, outputs
└── Tephra2/
    └── tephra2_2020    # compiled executable
```

## Configuration

All defaults live in `config/default_config.py`. Highlights:

* **Paths**: Where to find inputs/outputs and the Tephra2 executable
* **Parameters**: Variable (plume height, mass) and fixed plume settings
* **Inversion Backends**:

  * `mcmc`: iterations, burn‑in, proposal scale, likelihood sigma
  * `sa`: runs, temperature schedule (adaptive if `alpha=None`), restarts
  * `enkf`: ensemble size, number of assimilations, inflation, etc.

You can override *any* default from the notebook or by passing a `config={...}` dict when instantiating `TephraInversion`.

---

## What the Three Methods Do

* **MCMC** samples a posterior in log‑space (robust to heavy‑tailed mass). Great for uncertainty.
* **SA** is mode‑seeking; we record the full trajectory so you can still use MCMC‑style plots.
* **EnKF** updates an ensemble across multiple passes; you get per‑pass clouds and marginals.

All three share the same **`likelihood_sigma`** (default `0.6`), which sets how tightly the model is penalized for log‑space misfit. Lower means stricter fit; higher means more tolerance.

---

## Outputs

Generated figures land in `data/output/plots/`, including:

* Trace & marginals for each method
* Cross‑method scatter in parameter space
* EnKF per‑pass **traces** (5 stacked rows with very low opacity)
* EnKF per‑pass **marginals** for plume height and log mass in a single 2‑panel figure

Tables avoid scientific notation by default in the notebook.

---

## Notes & Tips

* If you crank up runs (e.g., MCMC 100k), consider tuning `likelihood_sigma`. Try `0.4–0.9`:

  * `~0.4–0.5`: tighter fit, slower/stricter exploration
  * `~0.6` (default): balanced on short‑course data
  * `~0.8–0.9`: looser fit, faster exploration but broader posteriors
* EnKF pass plots assume the ensemble member order is consistent across passes (our implementation preserves order).

## License

This project is licensed under the MIT License.

## Acknowledgements

This project builds significantly upon the foundational work of **Qingyuan Yang**, who developed the original Metropolis-Hastings algorithm for Tephra2 inversion. His codebases ([`tephra2_mcmc`](https://github.com/yiqioyang/tephra2_mcmc) and [`mh_tephra2`](https://github.com/yiqioyang/mh_tephra2)) served as the core inspiration and basis for this framework, and his mentorship throughout the project was invaluable.

Special thanks are extended to my PI **Professor Einat Lev** and **Software Engineer Samuel Krasnoff** at Columbia University's Lamont-Doherty Earth Observatory (LDEO). Their close collaboration, guidance, and support during the Spring 2025 semester were instrumental in the development of this project.

This work was conducted as part of the **Columbia Data Science Institute Scholar Program**, and their support is gratefully acknowledged.

We also thank:

- The developers and maintainers of the Tephra2 model.
- The developers of the Victor Platform.