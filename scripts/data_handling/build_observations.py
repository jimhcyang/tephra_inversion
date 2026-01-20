#!/usr/bin/env python3
"""
build_observations.py

Creates standardized observation CSVs for:
  (A) Cerro Negro 1992 (CN92) with layer handling (a, b, m) + ratio-splitting of m
  (B) Kilauea 1924 (KL24) by aggregating tephra *profiles* (sum thickness across subsamples)
      then optionally converting thickness (cm) -> mass loading (kg/m^2) using a fixed density.

Outputs are written to: root/data/observations/

CN92 outputs (5 tests):
  cn92_a.csv
  cn92_a_plus_splitm.csv
  cn92_b.csv
  cn92_b_plus_splitm.csv
  cn92_ab_total.csv

KL24 output:
  kl24_1924_profiles.csv               (Easting, Northing, Thickness_cm, Observations)
  kl24_1924_profiles_tephra2.csv       (Easting, Northing, Observations)  # minimal

All outputs use columns:
  Easting, Northing, Observations
(with optional Thickness_cm for KL24 profiles.csv)

Author: Jim Yang

python scripts/data_handling/build_observations.py \
  --cn "data_std/cerro_negro.csv" \
  --kl "data_std/1924 Tephra Physical Volcanology Data.csv" \
  --outdir "data/input/observations" \
  --kl-density 1000

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CSV_FLOAT_FMT = "%.2f"

def _round_for_output(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(2)
    return df

def repo_root_from_this_file() -> Path:
    # root/scripts/data_handling/build_observations.py -> parents[2] = root
    return Path(__file__).resolve().parents[2]


def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip(), errors="coerce")


def _std_obs_df(e: pd.Series, n: pd.Series, obs: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "Easting": _to_numeric_series(e),
            "Northing": _to_numeric_series(n),
            "Observations": _to_numeric_series(obs),
        }
    ).dropna(subset=["Easting", "Northing", "Observations"])

    # keep as ints for coordinates (UTM-like)
    out["Easting"] = out["Easting"].round().astype("int64")
    out["Northing"] = out["Northing"].round().astype("int64")
    return out


# -----------------------
# CN92
# -----------------------
def load_cn92(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _clean_cols(df)

    required = {"layer", "easting", "northing", "total(kg/m2)"}
    missing = sorted(required - set(map(str.lower, df.columns)))
    if missing:
        # try a friendlier lookup (case-insensitive)
        cols_lower = {c.lower(): c for c in df.columns}
        needed = [cols_lower.get(k) for k in required if k in cols_lower]
        raise KeyError(
            f"CN92 file missing required columns (case-insensitive): {missing}. "
            f"Available: {list(df.columns)}. Found: {needed}"
        )

    # normalize column access case-insensitively
    cols_lower = {c.lower(): c for c in df.columns}
    df = df.rename(
        columns={
            cols_lower["layer"]: "layer",
            cols_lower["easting"]: "easting",
            cols_lower["northing"]: "northing",
            cols_lower["total(kg/m2)"]: "total_kg_m2",
        }
    )

    df["layer"] = df["layer"].astype(str).str.strip().str.lower()
    return df


def cn92_build_outputs(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    df_a = df[df["layer"] == "a"].copy()
    df_b = df[df["layer"] == "b"].copy()
    df_m = df[df["layer"] == "m"].copy()

    # a-only / b-only / m-only (m is not one of the 5 tests but is useful to save/debug)
    a_only = _std_obs_df(df_a["easting"], df_a["northing"], df_a["total_kg_m2"])
    b_only = _std_obs_df(df_b["easting"], df_b["northing"], df_b["total_kg_m2"])
    m_only = _std_obs_df(df_m["easting"], df_m["northing"], df_m["total_kg_m2"])

    # compute global ratio from paired sites (same easting/northing appearing in both a and b)
    a_keyed = a_only.rename(columns={"Observations": "a_obs"})
    b_keyed = b_only.rename(columns={"Observations": "b_obs"})
    paired = a_keyed.merge(b_keyed, on=["Easting", "Northing"], how="inner")

    if len(paired) == 0:
        raise ValueError(
            "CN92: Found no paired (a,b) sites to estimate a:b ratio. "
            "Cannot split 'm' by ratio."
        )

    sum_a = paired["a_obs"].sum()
    sum_b = paired["b_obs"].sum()
    if sum_a <= 0 or (sum_a + sum_b) <= 0:
        raise ValueError("CN92: Non-positive sums when computing a:b ratio.")

    p_a = float(sum_a / (sum_a + sum_b))
    p_b = 1.0 - p_a

    # split m by global ratio
    m_split_a = m_only.copy()
    m_split_b = m_only.copy()
    m_split_a["Observations"] = m_split_a["Observations"] * p_a
    m_split_b["Observations"] = m_split_b["Observations"] * p_b

    # test datasets
    a_plus_splitm = pd.concat([a_only, m_split_a], ignore_index=True)
    b_plus_splitm = pd.concat([b_only, m_split_b], ignore_index=True)

    a_plus_splitm = a_plus_splitm.groupby(["Easting","Northing"], as_index=False)["Observations"].sum()
    b_plus_splitm = b_plus_splitm.groupby(["Easting","Northing"], as_index=False)["Observations"].sum()

    # a+b total:
    # - if both a & b exist at a site, sum them
    # - else if only one of {a,b} exists, use it
    # - if m exists, use m (assuming layer not distinguished)
    # This yields a “total deposit” dataset.
    ab = pd.concat(
        [
            a_only.assign(_layer="a"),
            b_only.assign(_layer="b"),
            m_only.assign(_layer="m"),
        ],
        ignore_index=True,
    )
    ab_total = (
        ab.groupby(["Easting", "Northing"], as_index=False)["Observations"].sum()
    )

    # write files
    # round only at output time
    a_only_out        = _round_for_output(a_only, ["Observations"])
    a_plus_splitm_out = _round_for_output(a_plus_splitm, ["Observations"])
    b_only_out        = _round_for_output(b_only, ["Observations"])
    b_plus_splitm_out = _round_for_output(b_plus_splitm, ["Observations"])
    ab_total_out      = _round_for_output(ab_total, ["Observations"])

    a_only_out.to_csv(outdir / "cn92_a.csv", index=False, float_format=CSV_FLOAT_FMT)
    a_plus_splitm_out.to_csv(outdir / "cn92_a_plus_splitm.csv", index=False, float_format=CSV_FLOAT_FMT)
    b_only_out.to_csv(outdir / "cn92_b.csv", index=False, float_format=CSV_FLOAT_FMT)
    b_plus_splitm_out.to_csv(outdir / "cn92_b_plus_splitm.csv", index=False, float_format=CSV_FLOAT_FMT)
    ab_total_out.to_csv(outdir / "cn92_ab_total.csv", index=False, float_format=CSV_FLOAT_FMT)

    # small provenance note
    (outdir / "cn92_splitm_ratio.txt").write_text(
        f"Computed split ratio from paired (a,b) sites:\n"
        f"  p_a = {p_a:.6f}\n"
        f"  p_b = {p_b:.6f}\n"
        f"  paired_sites = {len(paired)}\n"
        f"  m_sites = {len(m_only)}\n"
    )

    print("[CN92] wrote:",
          "cn92_a.csv, cn92_a_plus_splitm.csv, cn92_b.csv, cn92_b_plus_splitm.csv, cn92_ab_total.csv")


# -----------------------
# KL24
# -----------------------
def load_kl24(path: Path) -> pd.DataFrame:
    # your file has two header-like rows before the real header
    df = pd.read_csv(path, skiprows=2, encoding="utf-8", encoding_errors="replace", engine="python")
    df = _clean_cols(df)

    required = {"Eruption", "Easting", "Northing", "Thickness (cm)", "Sample Number"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(
            f"KL24 file missing columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    return df


def kl24_build_outputs(df: pd.DataFrame, outdir: Path, deposit_density_kg_m3: float) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["Eruption"] = df["Eruption"].astype(str).str.strip()
    df1924 = df[df["Eruption"] == "1924"].copy()

    if df1924.empty:
        raise ValueError("KL24: No rows found with Eruption == 1924.")

    # thickness per subsample
    df1924["Thickness (cm)"] = _to_numeric_series(df1924["Thickness (cm)"])
    df1924["Easting"] = _to_numeric_series(df1924["Easting"])
    df1924["Northing"] = _to_numeric_series(df1924["Northing"])
    df1924 = df1924.dropna(subset=["Thickness (cm)", "Easting", "Northing"])

    # Aggregate to tephra-profile level (each profile = one coordinate)
    prof = (
        df1924.groupby(["Easting", "Northing"], as_index=False)
        .agg(
            Thickness_cm=("Thickness (cm)", "sum"),
            n_subsamples=("Thickness (cm)", "size"),
        )
    )

    # Convert to mass loading kg/m^2 if you want Tephra2-like input
    prof["Observations"] = (prof["Thickness_cm"] / 100.0) * float(deposit_density_kg_m3)

    # standardize coords to int
    prof["Easting"] = prof["Easting"].round().astype("int64")
    prof["Northing"] = prof["Northing"].round().astype("int64")

    # Save richer + minimal Tephra2-like
    prof_out = _round_for_output(prof, ["Thickness_cm", "Observations"])
    prof_out.to_csv(outdir / "kl24_1924_profiles.csv", index=False, float_format=CSV_FLOAT_FMT)

    prof_min_out = _round_for_output(prof[["Easting", "Northing", "Observations"]], ["Observations"])
    prof_min_out.to_csv(outdir / "kl24_1924_profiles_tephra2.csv", index=False, float_format=CSV_FLOAT_FMT)

    print("[KL24] wrote: kl24_1924_profiles.csv, kl24_1924_profiles_tephra2.csv")


def main() -> int:
    root = repo_root_from_this_file()

    ap = argparse.ArgumentParser()
    ap.add_argument("--cn", type=str, default=str(root / "data" / "raw" / "cerro_negro.csv"),
                    help="Path to Cerro Negro 1992 CSV")
    ap.add_argument("--kl", type=str, default=str(root / "data" / "raw" / "1924 Tephra Physical Volcanology Data.csv"),
                    help="Path to Kilauea 1924 physical volcanology CSV")
    ap.add_argument("--outdir", type=str, default=str(root / "data" / "input" / "observations"),
                    help="Output directory for observation CSVs")
    ap.add_argument("--kl-density", type=float, default=1000.0,
                    help="Deposit density (kg/m^3) used to convert thickness->kg/m^2 for KL24")
    args = ap.parse_args()

    outdir = Path(args.outdir)

    # CN92
    cn_path = Path(args.cn)
    if not cn_path.exists():
        raise FileNotFoundError(f"CN92 input not found: {cn_path}")
    cn = load_cn92(cn_path)
    cn92_build_outputs(cn, outdir)

    # KL24
    kl_path = Path(args.kl)
    if not kl_path.exists():
        raise FileNotFoundError(f"KL24 input not found: {kl_path}")
    kl = load_kl24(kl_path)
    kl24_build_outputs(kl, outdir, args.kl_density)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
