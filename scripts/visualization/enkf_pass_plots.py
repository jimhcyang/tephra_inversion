# ─────────────────────────────────────────────────────────────
# scripts/visualization/enkf_pass_plots.py
# 2×2: traces (top) + per-pass marginals (bottom) for two params
# ─────────────────────────────────────────────────────────────
from __future__ import annotations
from pathlib import Path
from typing import List, Sequence, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


def _stack_history(ensemble_history: List[np.ndarray]) -> np.ndarray:
    """
    Stack a list of (Nens x P) arrays into (Npass x Nens x P).
    """
    if not ensemble_history:
        raise ValueError("ensemble_history is empty.")
    shapes = {X.shape for X in ensemble_history}
    if len(shapes) != 1:
        raise ValueError(f"All arrays in ensemble_history must have the same shape; got {shapes}.")
    return np.stack(ensemble_history, axis=0)  # (K, N, P)


def _pick_subset(n_ens: int, k: int, rng: np.random.Generator) -> np.ndarray:
    k = max(0, min(int(k), int(n_ens)))
    if k == 0:
        return np.array([], dtype=int)
    return rng.choice(n_ens, size=k, replace=False)


def enkf_pass_2x2(
    ensemble_history: List[np.ndarray],
    *,
    param_indices: Tuple[int, int] = (0, 1),
    param_labels: Sequence[str] = ("Plume Height (m)", "Log Eruption Mass (ln kg)"),
    title: str = "ES-MDA: per-pass traces (top) and marginals (bottom)",
    save_path: str | Path = "data/output/plots/enkf_pass_2x2.png",
    show: bool = True,
    # traces:
    max_color_series: int = 128,   # how many colored members to highlight
    gray_alpha: float = 0.02,      # opacity for the gray "all members" lines
    color_alpha: float = 0.10,     # opacity for the colored highlighted subset
    line_width: float = 0.8,
    seed: Optional[int] = 42,
    # marginals:
    bins: int = 50,
    base_hist_alpha: float = 0.12  # very low; later passes ramp slightly darker
) -> str:
    """
    Make a single 2×2 figure:
      • Top-left  : traces for param_indices[0]   (gray for all, blue for highlighted subset)
      • Top-right : traces for param_indices[1]   (gray for all, green for highlighted subset)
      • Bottom-left  : per-pass hist overlays for param_indices[0]   (blue hues)
      • Bottom-right : per-pass hist overlays for param_indices[1]   (green hues)

    The “traces” show each ensemble member’s value across passes (x-axis = pass index).
    Only the highlighted subset is colored; all others are faint gray to show overall scale.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    H = _stack_history(ensemble_history)  # (K, N, P)
    K, N, P = H.shape

    i0, i1 = param_indices
    if not (0 <= i0 < P and 0 <= i1 < P):
        raise IndexError(f"param_indices out of range, P={P}: got {param_indices}")

    label0, label1 = param_labels
    rng = np.random.default_rng(seed)

    # consistent colors: blue for param 0, green for param 1
    color0 = "C0"   # blue
    color1 = "C2"   # green (C2 is a nice green in Matplotlib default palette)

    # x-axis for traces (1..K)
    x = np.arange(1, K + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    ax_t0, ax_t1 = axes[0, 0], axes[0, 1]
    ax_h0, ax_h1 = axes[1, 0], axes[1, 1]

    # -------------------- TOP: traces (gray + highlighted subset) --------------------
    for ax, pi, color, ylab in ((ax_t0, i0, color0, label0), (ax_t1, i1, color1, label1)):
        # All members in gray (very low alpha)
        # Plot as many lines as N, but keep it lightweight (thin lines, no markers)
        y_all = H[:, :, pi]  # (K, N)
        for j in range(N):
            ax.plot(x, y_all[:, j], lw=line_width, color=(0.2, 0.2, 0.2), alpha=gray_alpha)

        # Highlight a random subset in color
        idx = _pick_subset(N, max_color_series, rng)
        for j in idx:
            ax.plot(x, y_all[:, j], lw=line_width, color=color, alpha=color_alpha)

        ax.set_ylabel(ylab)
        ax.grid(True, ls="--", alpha=0.25)

    ax_t0.set_title("Traces — highlighted subset vs all (gray)", fontsize=11)
    ax_t1.set_title("Traces — highlighted subset vs all (gray)", fontsize=11)

    # -------------------- BOTTOM: per-pass marginals (stacked alphas) ----------------
    # Slight alpha ramp so later passes are a bit darker
    def _alpha_for_pass(k: int) -> float:
        if K <= 1:
            return base_hist_alpha
        return base_hist_alpha * (1.0 + 0.75 * (k / (K - 1)))

    for k in range(K):
        a0 = _alpha_for_pass(k)
        a1 = _alpha_for_pass(k)
        ax_h0.hist(H[k, :, i0], bins=bins, density=True, alpha=a0, color=color0, edgecolor=None)
        ax_h1.hist(H[k, :, i1], bins=bins, density=True, alpha=a1, color=color1, edgecolor=None)

    ax_h0.set_xlabel(label0)
    ax_h1.set_xlabel(label1)
    ax_h0.set_ylabel("Density")
    ax_h1.set_ylabel("Density")
    ax_h0.set_title("ES-MDA per-pass marginals", fontsize=11)
    ax_h1.set_title("ES-MDA per-pass marginals", fontsize=11)

    # shared x label on top row
    ax_t0.set_xlabel("Assimilation pass")
    ax_t1.set_xlabel("Assimilation pass")

    # tidy
    for ax in (ax_t0, ax_t1, ax_h0, ax_h1):
        ax.tick_params(axis="both", which="both", labelsize=9)

    fig.suptitle(title.replace("\u2011", "-"))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return str(save_path)
