#!/usr/bin/env python3
"""
cerro_negro_loader.py
Standardize the Cerro Negro short-course bundle into working inputs.

Creates under data/input/:
  - tephra2.conf        (copied from tephra2.conf)
  - wind.txt            (copied from wind or wind1, any extension/case)
  - sites.csv           (E N Z from cerro_negro_92.dat)
  - observations.csv    (mass column from cerro_negro_92.dat)
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple
import pandas as pd


def _pick_shortest(root: Path, paths: list[Path]) -> Path | None:
    return min(paths, key=lambda p: len(p.relative_to(root).parts)) if paths else None


def _check_bundle(cerro_dir: Path) -> Tuple[Path, Path, Path]:
    """
    Search the ENTIRE cerro_dir tree (recursively) for the 3 required files,
    case-insensitively and with forgiving patterns.
    """
    root = Path(cerro_dir)

    # tephra2.conf (exact name, any case)
    conf_hits = [p for p in root.rglob("*") if p.is_file() and p.name.lower() == "tephra2.conf"]
    conf = _pick_shortest(root, conf_hits)

    # wind file: accept stem "wind" or "wind1" with any extension/case
    wind_hits = [
        p for p in root.rglob("*")
        if p.is_file() and p.stem.lower() in {"wind", "wind1"}
    ]
    wind = _pick_shortest(root, wind_hits)

    # data file: prefer exact name; otherwise try a fuzzy fallback
    dat_hits = [p for p in root.rglob("*") if p.is_file() and p.name.lower() == "cerro_negro_92.dat"]
    if not dat_hits:
        dat_hits = [
            p for p in root.rglob("*.dat")
            if "cerro" in p.name.lower() and "negro" in p.name.lower() and "92" in p.name
        ]
    dat = _pick_shortest(root, dat_hits)

    missing = []
    if conf is None: missing.append("tephra2.conf")
    if wind is None: missing.append("wind / wind1")
    if dat  is None: missing.append("cerro_negro_92.dat")
    if missing:
        raise FileNotFoundError(
            "Expected files not found under "
            f"{cerro_dir} -> {', '.join(missing)}.\n"
            "Tip: run the downloader first. This loader now searches recursively, "
            "so nested folders are fine."
        )

    return conf, wind, dat


def prepare_cerro_negro(
    cerro_dir: str | Path = "data/input/cerro_negro",
    work_dir: str | Path = "data/input",
) -> Tuple[Path, Path, Path, Path]:
    """
    Read the raw files anywhere under cerro_dir and write standardized copies into work_dir.
    Returns (obs_csv, sites_csv, wind_txt, conf_path).
    """
    cerro_dir = Path(cerro_dir)
    work_dir  = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    src_conf, src_wind, src_dat = _check_bundle(cerro_dir)

    # 1) Copy tephra2.conf and wind.txt
    conf_path = work_dir / "tephra2.conf"
    wind_txt  = work_dir / "wind.txt"
    conf_path.write_text(src_conf.read_text())
    wind_txt.write_text(src_wind.read_text())

    # 2) Parse observations/sites from cerro_negro_92.dat
    # file format: easting northing elevation total(kg/m2)
    df = pd.read_csv(
        src_dat, sep=r"\s+", comment="#", header=None,
        names=["E", "N", "Z", "mass"], engine="python"
    )

    sites_csv = work_dir / "sites.csv"
    obs_csv   = work_dir / "observations.csv"

    df[["E", "N", "Z"]].to_csv(
        sites_csv, sep=" ", header=False, index=False, float_format="%.3f"
    )
    df["mass"].to_csv(obs_csv, header=False, index=False, float_format="%.6f")

    return obs_csv, sites_csv, wind_txt, conf_path


if __name__ == "__main__":
    try:
        out = prepare_cerro_negro()
        print(
            "Prepared working inputs:\n"
            "  observations:", out[0],
            "\n  sites:", out[1],
            "\n  wind:", out[2],
            "\n  conf:", out[3]
        )
    except Exception as e:
        import sys
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
