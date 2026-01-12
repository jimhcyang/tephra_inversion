#!/usr/bin/env python3
"""
run_fetch_winds.py

Convenience runner for fetch_wind.py presets.

Usage
-----
# run default (real) events only
python scripts/data_handling/run_fetch_winds.py

# run default + tests
python scripts/data_handling/run_fetch_winds.py --include-tests

# run one preset (can be TEST_* too)
python scripts/data_handling/run_fetch_winds.py --event TEST_ERA5_CERRO_1992

# show commands without running
python scripts/data_handling/run_fetch_winds.py --dry-run

# safety: cap max files per event (passed through)
python scripts/data_handling/run_fetch_winds.py --max-files 200
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List, Tuple


# IMPORTANT:
# fetch_wind.py already saves relative --out paths under: <repo>/data/input/
# So we pass "winds/..." here (NOT "data/input/winds/...") to avoid double-prefix.
DEFAULT_PRESETS: List[Tuple[str, str]] = [
    ("kilauea_1924", "winds/kilauea_{ts}.dat"),
    ("cerro_negro_1992", "winds/cerro_{ts}.dat"),
    ("kilauea_1992_validation", "winds/kilauea92_{ts}.dat"),
]

TEST_PRESETS: List[Tuple[str, str]] = [
    ("TEST_20CR_KILAUEA_1924", "winds/test_kilauea_{ts}.dat"),
    ("TEST_ERA5_CERRO_1992", "winds/test_cerro_{ts}.dat"),
    ("TEST_ERA5_KILAUEA_1992", "winds/test_kilauea92_{ts}.dat"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--event",
        default=None,
        help="Run only one event preset (must match a preset name in fetch_wind.py). "
             "If omitted, runs DEFAULT presets (and optionally tests).",
    )
    ap.add_argument("--include-tests", action="store_true", help="Also run TEST_* presets.")
    ap.add_argument("--dry-run", action="store_true", help="Print commands only (no execution).")

    ap.add_argument(
        "--keep-raw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep raw downloaded files (default: True). Use --no-keep-raw to disable.",
    )

    ap.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Pass --max-files to fetch_wind.py (per event).",
    )

    args = ap.parse_args()

    fetch_wind_py = os.path.join("scripts", "data_handling", "fetch_wind.py")
    if not os.path.isfile(fetch_wind_py):
        print(f"[ERROR] Could not find {fetch_wind_py}. Run from repo root or fix path in runner.", file=sys.stderr)
        return 1

    # Ensure the *actual* target directory exists:
    # fetch_wind.py writes relative paths under <repo>/data/input/
    os.makedirs(os.path.join("data", "input", "winds"), exist_ok=True)

    all_presets = list(DEFAULT_PRESETS)
    if args.include_tests:
        all_presets += TEST_PRESETS

    if args.event is not None:
        searchable = DEFAULT_PRESETS + TEST_PRESETS
        to_run = [(e, out) for (e, out) in searchable if e == args.event]
        if not to_run:
            known = ", ".join(e for e, _ in searchable)
            print(f"[ERROR] Unknown --event '{args.event}'. Known here: {known}", file=sys.stderr)
            return 2
    else:
        to_run = all_presets

    for event, out_tmpl in to_run:
        cmd = [sys.executable, fetch_wind_py, "--event", event, "--out", out_tmpl]
        if args.keep_raw:
            cmd.append("--keep-raw")
        if args.max_files is not None:
            cmd += ["--max-files", str(args.max_files)]

        print("▶", " ".join(cmd))
        if args.dry_run:
            continue

        subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())