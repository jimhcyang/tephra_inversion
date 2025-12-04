# scripts/sim/sim_types.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


@dataclass
class GroupSpec:
    """
    Specification of a group of runs to aggregate / plot.

    Backwards-compatible:

    - Old style: pass prior_factor only
        GroupSpec(model="sa", prior_factor=2.5, runs=1000, ...)

    - New style: pass scale_plume_factor / scale_mass_factor explicitly
        GroupSpec(model="mcmc",
                  scale_plume_factor=2.5,
                  scale_mass_factor=0.5,
                  n_iter=10000, ...)

    In __post_init__, if prior_factor is provided but scale_* are None,
    we set both scales to prior_factor.
    """

    model: str

    # Old interface (used in plots_demo.py, old notebook code)
    prior_factor: Optional[float] = None

    # New interface for possibly different plume vs mass scalings
    scale_plume_factor: Optional[float] = None
    scale_mass_factor: Optional[float] = None

    # Model-specific hyperparameters
    n_iter: Optional[int] = None           # MCMC
    runs: Optional[int] = None             # SA / PSO
    restarts: Optional[int] = None         # SA / PSO
    n_ens: Optional[int] = None            # ES-MDA
    n_assimilations: Optional[int] = None  # ES-MDA

    # Purely for naming / bookkeeping
    config_index: int = 0

    def __post_init__(self) -> None:
        # If user passed prior_factor (old style) but no explicit scales,
        # assume both plume and mass use that same factor.
        if self.prior_factor is not None:
            if self.scale_plume_factor is None:
                self.scale_plume_factor = self.prior_factor
            if self.scale_mass_factor is None:
                self.scale_mass_factor = self.prior_factor


@dataclass
class GroupResult:
    """
    Bookkeeping record for a plotted group (used in notebooks).

    Backwards-compatible:

    - Old code expects: gr.prior_factor
    - New code might prefer: gr.scale_plume_factor / gr.scale_mass_factor
    """

    model: str

    # Old-style single prior factor (for labels, etc.)
    prior_factor: Optional[float] = None

    config_index: int = 0
    trace_path: Path | None = None
    marginals_path: Path | None = None
    n_runs: int = 0
    n_steps: int = 0
    param_cols: Sequence[str] = ()

    # New-style separate scalings (optional)
    scale_plume_factor: Optional[float] = None
    scale_mass_factor: Optional[float] = None

    def __post_init__(self) -> None:
        # If only prior_factor was given, propagate to scales.
        if self.prior_factor is not None:
            if self.scale_plume_factor is None:
                self.scale_plume_factor = self.prior_factor
            if self.scale_mass_factor is None:
                self.scale_mass_factor = self.prior_factor

        # If only scales were given and they match, we can set a single
        # prior_factor so old code can still display something sensible.
        if (
            self.prior_factor is None
            and self.scale_plume_factor is not None
            and self.scale_mass_factor is not None
            and self.scale_plume_factor == self.scale_mass_factor
        ):
            self.prior_factor = self.scale_plume_factor
