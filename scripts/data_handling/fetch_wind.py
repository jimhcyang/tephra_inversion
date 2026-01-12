#!/usr/bin/env python3
"""
fetch_wind.py

Examples (relative paths go under root/data/input/):
  python scripts/data_handling/fetch_wind.py --event kilauea_1924 --out winds/kilauea_{ts}.dat --keep-raw
  python scripts/data_handling/fetch_wind.py --event cerro_negro_1992 --out winds/cerro_{ts}.dat --keep-raw
  python scripts/data_handling/fetch_wind.py --event TEST_20CR_KILAUEA_1924 --out winds/test_kilauea_{ts}.dat --keep-raw
  python scripts/data_handling/fetch_wind.py --event TEST_ERA5_CERRO_1992 --out winds/test_cerro_{ts}.dat --keep-raw
  python scripts/data_handling/fetch_wind.py --event kilauea_1992_validation --out winds/kilauea92_{ts}.dat --keep-raw
  python scripts/data_handling/fetch_wind.py --event TEST_ERA5_KILAUEA_1992 --out winds/test_kilauea92_{ts}.dat --keep-raw


Notes:
  - Internally uses UTC and enforces synoptic 3-hour grid (00/03/06/... UTC) via snapping.
  - Optional local-time input via --input-tz, converted to UTC then snapped.
  - If --keep-raw: ERA5 keeps NetCDF; 20CR keeps a compact NPZ profile for validation.
  - By default, does NOT overwrite existing outputs (use --overwrite to force).
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Repo paths (script is expected at root/scripts/data_handling/fetch_wind.py)
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_BASE = REPO_ROOT / "data" / "input"
DEFAULT_RAW_DIR = DEFAULT_OUT_BASE / "winds_raw"

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
G0 = 9.80665  # m/s^2 (ERA5 geopotential -> height)

DEFAULT_PRESSURES_ERA5 = [
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175,
    200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700,
    750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000
]

# Use dap2:// to reduce PyDAP protocol warnings; also filter warnings in main().
U20CR_URL = "dap2://psl.noaa.gov/thredds/dodsC/Datasets/20thC_ReanV3/prsSI/uwnd.{year}.nc"
V20CR_URL = "dap2://psl.noaa.gov/thredds/dodsC/Datasets/20thC_ReanV3/prsSI/vwnd.{year}.nc"
H20CR_URL = "dap2://psl.noaa.gov/thredds/dodsC/Datasets/20thC_ReanV3/prsSI/hgt.{year}.nc"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _bearing_toward_from_uv(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """0°=toward North, 90°=toward East. u=eastward, v=northward."""
    return (np.degrees(np.arctan2(u, v)) + 360.0) % 360.0


def _write_tephra2_wind_file(out_path: Path, heights: np.ndarray, speed: np.ndarray, direction: np.ndarray) -> Path:
    df = pd.DataFrame({"HEIGHT": heights, "SPEED": speed, "DIRECTION": direction}).sort_values("HEIGHT")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("#HEIGHT SPEED DIRECTION\n")
        for _, r in df.iterrows():
            f.write(f"{float(r['HEIGHT']):.6f} {float(r['SPEED']):.6f} {float(r['DIRECTION']):.6f}\n")
    return out_path


def _parse_dt_naive(s: str) -> datetime:
    return datetime.strptime(s.strip(), "%Y-%m-%dT%H:%M")


def _to_utc_naive(dt_naive: datetime, input_tz: str) -> datetime:
    """Interpret dt_naive in input_tz, convert to UTC, then drop tzinfo (naive UTC)."""
    if input_tz.upper() == "UTC":
        return dt_naive
    from zoneinfo import ZoneInfo  # py3.9+
    tz = ZoneInfo(input_tz)
    dt_local = dt_naive.replace(tzinfo=tz)
    dt_utc = dt_local.astimezone(timezone.utc)
    return dt_utc.replace(tzinfo=None)


def _snap_to_step_utc(dt_utc: datetime, step_hours: int, mode: str) -> datetime:
    """Snap a naive-UTC datetime to a UTC grid with spacing step_hours."""
    if step_hours <= 0:
        return dt_utc

    step = step_hours * 3600
    epoch = datetime(1970, 1, 1)
    sec = int((dt_utc - epoch).total_seconds())

    if mode == "floor":
        snapped = (sec // step) * step
    elif mode == "ceil":
        snapped = ((sec + step - 1) // step) * step
    elif mode == "nearest":
        down = (sec // step) * step
        up = ((sec + step - 1) // step) * step
        snapped = down if (sec - down) <= (up - sec) else up
    else:
        raise ValueError(f"Unknown snap mode: {mode}")

    return epoch + timedelta(seconds=snapped)


def _make_times(start_utc: datetime, end_utc: datetime, step_hours: int) -> List[datetime]:
    if end_utc < start_utc:
        raise ValueError("--end must be >= --start (after timezone conversion)")
    out: List[datetime] = []
    t = start_utc
    step = timedelta(hours=step_hours)
    while t <= end_utc:
        out.append(t)
        t += step
    return out


def _parse_pressures(s: str) -> List[int]:
    vals: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            vals.append(int(part))
    if not vals:
        raise ValueError("Empty --pressures list.")
    return vals


def _resolve_under_base(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (base_dir / p)


def _format_out(out_tmpl: str, t_utc: datetime, out_base: Path) -> Path:
    """
    If out_tmpl is relative, it is resolved under out_base.
    Supports {ts} placeholder.
    """
    ts = t_utc.strftime("%Y%m%d_%H%M")  # UTC stamp

    # If template contains {ts}
    if "{ts}" in out_tmpl:
        return _resolve_under_base(out_tmpl.format(ts=ts), out_base)

    # If user points to a dir
    out_path = _resolve_under_base(out_tmpl, out_base)
    if out_tmpl.endswith("/") or out_tmpl.endswith(os.sep) or out_path.is_dir():
        return out_path / f"wind_{ts}.dat"

    root, ext = os.path.splitext(str(out_path))
    ext = ext or ".dat"
    return Path(f"{root}_{ts}{ext}")


def _raw_tag(src: str, t_utc: datetime) -> str:
    return f"{src}_{t_utc.strftime('%Y%m%d_%H%MZ')}"


# -----------------------------------------------------------------------------
# ERA5 (CDS) fetching
# -----------------------------------------------------------------------------
def _cds_retrieve_netcdf(
    out_nc: Path,
    year: str,
    month: str,
    day: str,
    hour: str,  # "HH:MM"
    pressure_levels: List[int],
    area: List[float],  # [N, W, S, E]
) -> Path:
    try:
        import cdsapi  # type: ignore
    except Exception as e:
        raise ImportError(
            "cdsapi is required for ERA5.\n"
            "Install: pip install cdsapi\n"
            "And ensure ~/.cdsapirc is configured."
        ) from e

    out_nc.parent.mkdir(parents=True, exist_ok=True)

    dataset = "reanalysis-era5-pressure-levels"
    base_req = {
        "product_type": ["reanalysis"],
        "variable": ["geopotential", "u_component_of_wind", "v_component_of_wind"],
        "pressure_level": [str(p) for p in pressure_levels],
        "year": [year],
        "month": [month],
        "day": [day],
        "time": [hour],
        "area": area,
        "download_format": "unarchived",
    }

    client = cdsapi.Client()

    req1 = dict(base_req)
    req1["data_format"] = "netcdf"
    try:
        client.retrieve(dataset, req1, str(out_nc))
        return out_nc
    except Exception:
        req2 = dict(base_req)
        req2["format"] = "netcdf"
        client.retrieve(dataset, req2, str(out_nc))
        return out_nc


def _load_profile_from_era5_nc(nc_path: Path, lat: float, lon: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import xarray as xr  # type: ignore
    except Exception as e:
        raise ImportError("xarray is required to read ERA5 NetCDF. Install: pip install xarray") from e

    try:
        ds = xr.open_dataset(nc_path)
    except Exception as e:
        raise RuntimeError(
            "Failed to open ERA5 NetCDF. You may need a NetCDF engine.\n"
            "Try: pip install netCDF4  (or)  pip install h5netcdf\n"
            f"Original error: {e}"
        ) from e

    if "longitude" in ds.coords:
        lon_vals = ds["longitude"].values
        if np.nanmin(lon_vals) >= 0 and lon < 0:
            lon = lon % 360.0

    pt = ds.sel(latitude=lat, longitude=lon, method="nearest")

    if "time" in pt.dims:
        pt = pt.isel(time=0)
    elif "valid_time" in pt.dims:
        pt = pt.isel(valid_time=0)

    def pick_var(*names: str) -> str:
        for n in names:
            if n in pt.variables:
                return n
        raise KeyError(f"None of {names} found. Variables={list(pt.variables)}")

    u_name = pick_var("u", "u_component_of_wind")
    v_name = pick_var("v", "v_component_of_wind")
    z_name = pick_var("z", "geopotential")

    u = np.asarray(pt[u_name].values).squeeze()
    v = np.asarray(pt[v_name].values).squeeze()
    z = np.asarray(pt[z_name].values).squeeze()
    ds.close()

    if u.ndim != 1 or v.ndim != 1 or z.ndim != 1:
        raise ValueError(f"ERA5 profile not 1D after selection: u{u.shape}, v{v.shape}, z{z.shape}")

    heights = z / G0
    speed = np.sqrt(u**2 + v**2)
    direction = _bearing_toward_from_uv(u, v)

    order = np.argsort(heights)
    return heights[order], speed[order], direction[order]


def fetch_era5_one_time(
    lat: float,
    lon: float,
    t_utc: datetime,  # naive UTC
    out_path: Path,
    pressure_levels: List[int],
    bbox_deg: float,
    keep_raw: bool,
    raw_dir: Path,
    temp_nc: Optional[Path],
) -> Path:
    area = [lat + bbox_deg, lon - bbox_deg, lat - bbox_deg, lon + bbox_deg]

    if temp_nc is not None:
        nc_path = temp_nc
    elif keep_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)
        nc_path = raw_dir / f"{_raw_tag('era5', t_utc)}.nc"
    else:
        nc_path = out_path.with_suffix(".era5.nc")

    _cds_retrieve_netcdf(
        out_nc=nc_path,
        year=f"{t_utc.year:04d}",
        month=f"{t_utc.month:02d}",
        day=f"{t_utc.day:02d}",
        hour=t_utc.strftime("%H:%M"),
        pressure_levels=pressure_levels,
        area=area,
    )

    heights, speed, direction = _load_profile_from_era5_nc(nc_path, lat=lat, lon=lon)
    _write_tephra2_wind_file(out_path, heights, speed, direction)

    # If not keeping raw and not using user-provided temp_nc, clean up
    if (not keep_raw) and (temp_nc is None):
        try:
            nc_path.unlink()
        except OSError:
            pass

    return out_path


# -----------------------------------------------------------------------------
# 20CRv3 (NOAA PSL) fetching
# -----------------------------------------------------------------------------
def _open_20cr_dataset(url: str):
    try:
        import xarray as xr  # type: ignore
    except Exception as e:
        raise ImportError("xarray is required for 20CR OPeNDAP reads. Install: pip install xarray pydap") from e

    try:
        return xr.open_dataset(url, engine="pydap", decode_times=True)
    except Exception:
        return xr.open_dataset(url, decode_times=True)


@dataclass
class _CRCache:
    u: Dict[int, object]
    v: Dict[int, object]
    h: Dict[int, object]

    def __init__(self) -> None:
        self.u = {}
        self.v = {}
        self.h = {}

    def get(self, year: int):
        if year not in self.u:
            self.u[year] = _open_20cr_dataset(U20CR_URL.format(year=year))
            self.v[year] = _open_20cr_dataset(V20CR_URL.format(year=year))
            self.h[year] = _open_20cr_dataset(H20CR_URL.format(year=year))
        return self.u[year], self.v[year], self.h[year]

    def close_all(self) -> None:
        for d in (self.u, self.v, self.h):
            for _, ds in d.items():
                try:
                    ds.close()
                except Exception:
                    pass


def fetch_20crv3_one_time(
    cache: _CRCache,
    lat: float,
    lon: float,
    t_utc: datetime,  # naive UTC
    out_path: Path,
    pressure_levels: List[int],
    keep_raw: bool,
    raw_dir: Path,
) -> Path:
    lon_use = lon % 360.0 if lon < 0 else lon
    year = t_utc.year

    du, dv, dh = cache.get(year)

    pu = du.sel(lat=lat, lon=lon_use, method="nearest").sel(time=t_utc, method="nearest")
    pv = dv.sel(lat=lat, lon=lon_use, method="nearest").sel(time=t_utc, method="nearest")
    ph = dh.sel(lat=lat, lon=lon_use, method="nearest").sel(time=t_utc, method="nearest")

    chosen_time = None
    try:
        chosen_time = str(np.asarray(pu["time"].values).item())
    except Exception:
        chosen_time = None

    if "level" in pu.coords and pressure_levels:
        avail = set(np.asarray(pu["level"].values).tolist())
        keep = [p for p in pressure_levels if p in avail]
        if not keep:
            keep = np.asarray(pu["level"].values).tolist()
        pu = pu.sel(level=keep)
        pv = pv.sel(level=keep)
        ph = ph.sel(level=keep)

    levels = np.asarray(pu["level"].values).squeeze()
    u = np.asarray(pu["uwnd"].values).squeeze()
    v = np.asarray(pv["vwnd"].values).squeeze()
    hgt = np.asarray(ph["hgt"].values).squeeze()

    if u.ndim != 1 or v.ndim != 1 or hgt.ndim != 1:
        raise ValueError(f"20CR profile not 1D after selection: u{u.shape}, v{v.shape}, hgt{hgt.shape}")

    if keep_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"{_raw_tag('20crv3_profile', t_utc)}.npz"
        np.savez_compressed(
            raw_path,
            lat=float(lat),
            lon=float(lon),
            lon_use=float(lon_use),
            requested_time=t_utc.strftime("%Y-%m-%d %H:%MZ"),
            chosen_time=chosen_time if chosen_time is not None else "",
            level=levels,
            uwnd=u,
            vwnd=v,
            hgt=hgt,
        )

    speed = np.sqrt(u**2 + v**2)
    direction = _bearing_toward_from_uv(u, v)

    order = np.argsort(hgt)
    return _write_tephra2_wind_file(out_path, hgt[order], speed[order], direction[order])


# -----------------------------------------------------------------------------
# Event presets (major activity windows)
# -----------------------------------------------------------------------------
@dataclass
class EventPreset:
    name: str
    lat: float
    lon: float
    source: str
    input_tz: str
    start_local: str
    end_local: str
    step_hours: int = 3


EVENTS: Dict[str, EventPreset] = {
    "kilauea_1924": EventPreset(
        name="kilauea_1924",
        lat=19.421, lon=-155.287,
        source="20crv3",
        input_tz="Pacific/Honolulu",
        start_local="1924-05-13T00:00",
        end_local="1924-05-19T00:00",
        step_hours=3,
    ),
    "cerro_negro_1992": EventPreset(
        name="cerro_negro_1992",
        lat=12.5078, lon=-86.7022,
        source="era5",
        input_tz="America/Managua",
        start_local="1992-04-09T23:20",
        end_local="1992-04-12T18:00",
        step_hours=3,
    ),
    "TEST_20CR_KILAUEA_1924": EventPreset(
        name="TEST_20CR_KILAUEA_1924",
        lat=19.421, lon=-155.287,
        source="20crv3",
        input_tz="Pacific/Honolulu",
        start_local="1924-05-18T09:00",
        end_local="1924-05-18T12:00",
        step_hours=3,
    ),
    "TEST_ERA5_CERRO_1992": EventPreset(
        name="TEST_ERA5_CERRO_1992",
        lat=12.5078, lon=-86.7022,
        source="era5",
        input_tz="UTC",
        start_local="1992-04-10T06:00",
        end_local="1992-04-10T09:00",
        step_hours=3,
    ),
    
    # Kīlauea validation window (1992): same local window as 1924 preset,
    # but in a modern year where we can use ERA5.
    "kilauea_1992_validation": EventPreset(
        name="kilauea_1992_validation",
        lat=19.421, lon=-155.287,
        source="era5",
        input_tz="Pacific/Honolulu",   # HST
        start_local="1992-05-13T00:00",
        end_local="1992-05-19T00:00",
        step_hours=3,
    ),
    # ERA5 quick test: match the 1924 test's local window but in 1992
    "TEST_ERA5_KILAUEA_1992": EventPreset(
        name="TEST_ERA5_KILAUEA_1992",
        lat=19.421, lon=-155.287,
        source="era5",
        input_tz="Pacific/Honolulu",
        start_local="1992-05-18T09:00",
        end_local="1992-05-18T12:00",
        step_hours=3,
    ),
}


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> int:
    # Reduce PyDAP chatter
    warnings.filterwarnings(
        "ignore",
        message=r".*DAP2.*legacy.*|PyDAP was unable to determine the DAP protocol.*",
        category=UserWarning,
        module=r"pydap\..*",
    )

    p = argparse.ArgumentParser(description="Fetch reanalysis wind profiles and write Tephra2 wind files.")
    p.add_argument("--event", choices=sorted(EVENTS.keys()), default=None)
    p.add_argument("--lat", type=float, default=None)
    p.add_argument("--lon", type=float, default=None)
    p.add_argument("--source", choices=["auto", "era5", "20crv3"], default="auto")

    p.add_argument("--time", default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--step-hours", type=int, default=3)

    p.add_argument("--input-tz", default="UTC")
    p.add_argument("--snap-mode", choices=["expand", "contract", "none"], default="expand")

    p.add_argument("--out", required=True, help="Output template/path. If relative, saved under root/data/input/ by default.")
    p.add_argument("--out-base", default=str(DEFAULT_OUT_BASE), help="Base dir for relative --out paths.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-files", type=int, default=1000)

    # Validation / raw retention
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing wind .dat files.")
    p.add_argument("--keep-raw", action="store_true", help="Keep raw inputs for validation (ERA5 NetCDF; 20CR NPZ profile).")
    p.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="Where raw files are written when --keep-raw is set.")

    # ERA5 settings
    p.add_argument("--bbox-deg", type=float, default=0.25)
    p.add_argument("--temp-nc", default=None, help="Fixed temp NetCDF path (single-time only).")
    p.add_argument("--pressures", default=",".join(str(x) for x in DEFAULT_PRESSURES_ERA5))

    args = p.parse_args()

    out_base = Path(args.out_base)
    raw_dir = Path(args.raw_dir)
    pressures = _parse_pressures(args.pressures)

    # Apply preset if requested
    if args.event:
        preset = EVENTS[args.event]
        lat = args.lat if args.lat is not None else preset.lat
        lon = args.lon if args.lon is not None else preset.lon
        source = preset.source if args.source == "auto" else args.source

        # Use preset tz unless user overrides away from default UTC.
        input_tz = preset.input_tz if args.input_tz == "UTC" else args.input_tz
        step_hours = preset.step_hours if args.step_hours == 3 else args.step_hours

        start_utc = _to_utc_naive(_parse_dt_naive(preset.start_local), input_tz)
        end_utc = _to_utc_naive(_parse_dt_naive(preset.end_local), input_tz)
    else:
        if args.lat is None or args.lon is None:
            raise SystemExit("Provide --lat and --lon (or use --event preset).")
        lat, lon = args.lat, args.lon
        source = args.source
        input_tz = args.input_tz
        step_hours = args.step_hours

        if args.time:
            t_local = _parse_dt_naive(args.time)
            start_utc = _to_utc_naive(t_local, input_tz)
            end_utc = start_utc
        elif args.start and args.end:
            start_utc = _to_utc_naive(_parse_dt_naive(args.start), input_tz)
            end_utc = _to_utc_naive(_parse_dt_naive(args.end), input_tz)
        else:
            raise SystemExit("Provide either --time or (--start and --end), or use --event.")

    # Snap to UTC grid
    if args.snap_mode != "none":
        if args.snap_mode == "expand":
            start_utc = _snap_to_step_utc(start_utc, step_hours, "floor")
            end_utc = _snap_to_step_utc(end_utc, step_hours, "ceil")
        else:
            start_utc = _snap_to_step_utc(start_utc, step_hours, "ceil")
            end_utc = _snap_to_step_utc(end_utc, step_hours, "floor")

    times = _make_times(start_utc, end_utc, step_hours)

    if args.max_files and args.max_files > 0 and len(times) > args.max_files:
        raise SystemExit(f"Refusing to generate {len(times)} files (cap={args.max_files}). Use --max-files 0 for unlimited.")

    if args.dry_run:
        print(f"Planned {len(times)} times (UTC), step={step_hours}h:")
        for t in times:
            print("  ", t.strftime("%Y-%m-%d %H:%M UTC"))
        return 0

    cache = _CRCache()
    try:
        for t_utc in times:
            src = source
            if src == "auto":
                src = "era5" if t_utc.year >= 1940 else "20crv3"

            out_path = _format_out(args.out, t_utc, out_base)

            if out_path.exists() and (not args.overwrite):
                print(f"[SKIP] exists: {out_path}")
                continue

            if src == "era5":
                temp_nc = Path(args.temp_nc) if (args.temp_nc and len(times) == 1) else None
                fetch_era5_one_time(
                    lat=lat, lon=lon, t_utc=t_utc, out_path=out_path,
                    pressure_levels=pressures,
                    bbox_deg=args.bbox_deg,
                    keep_raw=args.keep_raw,
                    raw_dir=raw_dir,
                    temp_nc=temp_nc,
                )
            else:
                fetch_20crv3_one_time(
                    cache=cache,
                    lat=lat, lon=lon, t_utc=t_utc, out_path=out_path,
                    pressure_levels=pressures,
                    keep_raw=args.keep_raw,
                    raw_dir=raw_dir,
                )

            print(f"[OK] {src} {t_utc.strftime('%Y-%m-%d %H:%MZ')} -> {out_path}")

        return 0

    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    finally:
        cache.close_all()


if __name__ == "__main__":
    raise SystemExit(main())
