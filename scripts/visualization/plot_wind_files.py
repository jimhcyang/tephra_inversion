#!/usr/bin/env python3
"""
plot_wind_files.py

Reads Tephra2-format wind files in data/input/winds and produces, for each dataset:
  (A) color-gradient plot (timestamp -> sequential colormap)
  (B) grey-background plot with one highlighted timestamp (red + thicker)

Changes vs prior version:
  - Colorbar ticks only at 00:00 UTC of each day within the dataset window.
  - Gradient colorbar placed below the subplots (under x-axes).
  - Highlight legend (red line) also placed below the subplots.

Output:
  data/output/wind_plots/
    cerro_negro_1992_gradient.png
    cerro_negro_1992_highlight_cerro_19920410_0600.png
    kilauea_1924_gradient.png
    kilauea_1924_highlight_kilauea_19240518_2100.png
    kilauea_1992_validation_gradient.png
    kilauea_1992_validation_highlight_kilauea92_19920518_2100.png

Run from repo root:
  python scripts/visualization/plot_wind_files.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


# ----------------------------
# Config
# ----------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]  # scripts/visualization -> repo root
IN_DIR = REPO_ROOT / "data" / "input" / "winds"
OUT_DIR = REPO_ROOT / "data" / "output" / "wind_plots"

# Single-file highlight picks (UTC stamps embedded in filenames)
HIGHLIGHTS: Dict[str, str] = {
    "cerro_negro_1992": "cerro_19920410_0600.dat",
    "kilauea_1924": "kilauea_19240518_2100.dat",
    "kilauea_1992_validation": "kilauea92_19920518_2100.dat",
}

# Dataset file globs
DATASETS: Dict[str, str] = {
    "cerro_negro_1992": "cerro_*.dat",
    "kilauea_1924": "kilauea_*.dat",
    "kilauea_1992_validation": "kilauea92_*.dat",
}

# Reader-friendly dataset titles (for figures)
DATASET_TITLES: Dict[str, str] = {
    "cerro_negro_1992": "Cerro Negro (1992) wind profiles",
    "kilauea_1924": "Kīlauea (1924) wind profiles",
    "kilauea_1992_validation": "Kīlauea (1992, validation) wind profiles",
}

# Single-hue sequential colormap
CMAP_NAME = "Blues"


# ----------------------------
# Parsing helpers
# ----------------------------
_TS_RE = re.compile(r"_(\d{8})_(\d{4})\.dat$")


def parse_timestamp_utc_from_filename(p: Path) -> datetime:
    """
    Expects filenames like:
      cerro_19920410_0600.dat
      kilauea_19240518_2100.dat
      kilauea92_19920518_2100.dat
    Returns tz-aware UTC datetime.
    """
    m = _TS_RE.search(p.name)
    if not m:
        raise ValueError(f"Could not parse timestamp from filename: {p.name}")
    ymd, hm = m.group(1), m.group(2)
    dt = datetime.strptime(ymd + hm, "%Y%m%d%H%M")
    return dt.replace(tzinfo=timezone.utc)


def read_wind_file(p: Path) -> pd.DataFrame:
    """
    Tephra2 wind file format:
      #HEIGHT SPEED DIRECTION
      74.38 4.35 267.3
      ...
    """
    df = pd.read_csv(
        p,
        sep=r"\s+",
        comment="#",
        names=["HEIGHT", "SPEED", "DIRECTION"],
        engine="python",
    )
    if df.empty:
        raise ValueError(f"No data read from {p}")
    return df.sort_values("HEIGHT").reset_index(drop=True)


# ----------------------------
# Plotting helpers
# ----------------------------
@dataclass
class WindSeries:
    path: Path
    t_utc: datetime
    df: pd.DataFrame


def load_series(files: List[Path]) -> List[WindSeries]:
    out: List[WindSeries] = []
    for f in sorted(files):
        out.append(WindSeries(path=f, t_utc=parse_timestamp_utc_from_filename(f), df=read_wind_file(f)))
    return out


def _format_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _make_two_panel_figure(title: str) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    # Give extra bottom room for legends/colorbar placed under axes.
    fig, (ax_spd, ax_dir) = plt.subplots(
        1, 2, figsize=(12, 7), sharey=True, constrained_layout=False
    )
    fig.suptitle(title, fontsize=14)

    ax_spd.set_xlabel("Wind speed (m/s)")
    ax_dir.set_xlabel("Wind direction (degrees, toward)")
    ax_spd.set_ylabel("Height (m)")

    ax_dir.set_xlim(0, 360)
    ax_spd.grid(True, alpha=0.25)
    ax_dir.grid(True, alpha=0.25)

    # Reserve space at bottom for colorbar / legend
    fig.subplots_adjust(bottom=0.22, wspace=0.12)
    return fig, ax_spd, ax_dir


def _midnight_ticks(times_utc: List[datetime]) -> List[datetime]:
    """
    Return 00:00 UTC timestamps for each day present in the data window.
    Only includes midnights that fall within [min_time, max_time].
    """
    if not times_utc:
        return []

    tmin = min(times_utc).astimezone(timezone.utc)
    tmax = max(times_utc).astimezone(timezone.utc)

    # Start at next midnight at/after tmin's date start
    start_day = tmin.date()
    start_midnight = datetime(start_day.year, start_day.month, start_day.day, 0, 0, tzinfo=timezone.utc)
    if start_midnight < tmin:
        start_midnight += timedelta(days=1)

    ticks: List[datetime] = []
    cur = start_midnight
    while cur <= tmax:
        ticks.append(cur)
        cur += timedelta(days=1)

    return ticks


# ----------------------------
# Plotting
# ----------------------------
def plot_gradient(series: List[WindSeries], dataset_name: str, out_path: Path) -> None:
    if not series:
        raise ValueError(f"No files for dataset={dataset_name}")

    cmap = mpl.colormaps.get_cmap(CMAP_NAME)

    times = np.array([s.t_utc.timestamp() for s in series], dtype=float)
    norm = mpl.colors.Normalize(vmin=float(times.min()), vmax=float(times.max()))

    dataset_title = DATASET_TITLES.get(dataset_name, dataset_name)
    t0 = _format_utc(series[0].t_utc)
    t1 = _format_utc(series[-1].t_utc)

    fig, ax_spd, ax_dir = _make_two_panel_figure(
        f"{dataset_title}\nProfiles across synoptic times ({t0} → {t1})"
    )

    for s in series:
        color = cmap(norm(s.t_utc.timestamp()))
        ax_spd.plot(s.df["SPEED"].values, s.df["HEIGHT"].values, color=color, linewidth=1.0, alpha=0.9)
        ax_dir.plot(s.df["DIRECTION"].values, s.df["HEIGHT"].values, color=color, linewidth=1.0, alpha=0.9)

    # Colorbar below both subplots
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # Place a horizontal colorbar beneath the axes
    cbar = fig.colorbar(
        sm,
        ax=[ax_spd, ax_dir],
        orientation="horizontal",
        fraction=0.06,
        pad=0.12,
        aspect=40,
    )
    cbar.set_label("Time progression (UTC)")

    # Ticks only at 00:00 UTC for each day in the range
    tick_dts = _midnight_ticks([s.t_utc for s in series])
    tick_vals = [dt.timestamp() for dt in tick_dts]

    # If the window is < 1 day and yields no midnights, fall back to no ticks (clean)
    if tick_vals:
        cbar.set_ticks(tick_vals)
        cbar.set_ticklabels([dt.strftime("%m-%d 00:00Z") for dt in tick_dts])
    else:
        cbar.set_ticks([])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_highlight(series: List[WindSeries], dataset_name: str, highlight_filename: str, out_path: Path) -> None:
    if not series:
        raise ValueError(f"No files for dataset={dataset_name}")

    highlight_series: Optional[WindSeries] = None
    for s in series:
        if s.path.name == highlight_filename:
            highlight_series = s
            break

    if highlight_series is None:
        available = ", ".join(s.path.name for s in series[:8]) + ("..." if len(series) > 8 else "")
        raise ValueError(
            f"Highlight file '{highlight_filename}' not found for {dataset_name}. "
            f"Example available files: {available}"
        )

    dataset_title = DATASET_TITLES.get(dataset_name, dataset_name)
    hl_time = _format_utc(highlight_series.t_utc)

    fig, ax_spd, ax_dir = _make_two_panel_figure(
        f"{dataset_title}\nHighlighted profile at {hl_time}"
    )

    # Background grey
    for s in series:
        ax_spd.plot(s.df["SPEED"].values, s.df["HEIGHT"].values, color="0.75", linewidth=1.0, alpha=0.5)
        ax_dir.plot(s.df["DIRECTION"].values, s.df["HEIGHT"].values, color="0.75", linewidth=1.0, alpha=0.5)

    # Highlight (red + thicker)
    line_spd, = ax_spd.plot(
        highlight_series.df["SPEED"].values,
        highlight_series.df["HEIGHT"].values,
        color="red",
        linewidth=3.0,
        alpha=0.95,
    )
    line_dir, = ax_dir.plot(
        highlight_series.df["DIRECTION"].values,
        highlight_series.df["HEIGHT"].values,
        color="red",
        linewidth=3.0,
        alpha=0.95,
    )

    # One legend centered below both subplots (under x-axis)
    label = f"Highlighted time: {hl_time}  ({highlight_filename})"
    fig.legend(
        [line_dir],
        [label],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=1,
        frameon=False,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    if not IN_DIR.exists():
        raise SystemExit(f"[ERROR] Input dir not found: {IN_DIR}")

    for dataset, pattern in DATASETS.items():
        files = list(IN_DIR.glob(pattern))
        if not files:
            print(f"[WARN] No files found for {dataset} ({pattern}) in {IN_DIR}")
            continue

        series = load_series(files)

        # (A) gradient
        out_a = OUT_DIR / f"{dataset}_gradient.png"
        print(f"[OK] writing {out_a}")
        plot_gradient(series, dataset, out_a)

        # (B) highlight
        hl = HIGHLIGHTS[dataset]
        out_b = OUT_DIR / f"{dataset}_highlight_{hl.replace('.dat','')}.png"
        print(f"[OK] writing {out_b}")
        plot_highlight(series, dataset, hl, out_b)

    print(f"[DONE] plots in: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
