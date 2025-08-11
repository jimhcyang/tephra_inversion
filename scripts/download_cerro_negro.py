#!/usr/bin/env python3
"""
download_cerro_negro.py
Quick downloader for the Tephra2 inversion short-course data.

• Saves ZIP to:   <repo_root>/data/input/cerro_negro/tephra2-inversion3-victor.zip
• Unzips into:    <repo_root>/data/input/cerro_negro/
• Optional:       --prepare standardizes working copies into <repo_root>/data/input/
"""

from __future__ import annotations
import sys
import zipfile
import argparse
import shutil
from pathlib import Path
from urllib.request import urlopen

URL = "https://gscommunitycodes.usf.edu/geoscicommunitycodes/public/inversion-shortcourse/tephra2-inv3/tephra2-inversion3-victor.zip"
ZIP_NAME = "tephra2-inversion3-victor.zip"
CHUNK = 1024 * 1024  # 1 MB


def repo_root_from_this_file() -> Path:
    # script path: <repo_root>/scripts/download_cerro_negro.py
    here = Path(__file__).resolve()
    return here.parent.parent


def ensure_sys_path(root: Path) -> None:
    """Ensure <repo_root> is on sys.path so 'import scripts.⋯' works when run as a file."""
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def download_to_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading:\n  {url}\n→ {dest}")
    with urlopen(url) as r, open(dest, "wb") as f:
        total = r.length or 0
        downloaded = 0
        while True:
            chunk = r.read(CHUNK)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {downloaded/1e6:.1f} / {total/1e6:.1f} MB ({pct:.1f}%)", end="")
        print()
    print("Download complete.")


def unzip_to_dir(zip_path: Path, out_dir: Path) -> None:
    print(f"Unzipping:\n  {zip_path}\n→ {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    print("Unzip complete.")


def maybe_flatten_single_subdir(parent: Path) -> None:
    """
    If the zip created a single subfolder (e.g., .../cerro_negro/<unzipped_name>/files),
    move its contents up one level into parent and remove the subfolder.
    """
    entries = [p for p in parent.iterdir() if not p.name.startswith(".")]
    if len(entries) == 1 and entries[0].is_dir():
        sub = entries[0]
        for item in sub.iterdir():
            target = parent / item.name
            if target.exists():
                # overwrite to be robust
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))
        shutil.rmtree(sub)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fetch Cerro Negro short-course bundle.")
    parser.add_argument("--prepare", action="store_true",
                        help="After download+unzip, standardize working copies into data/input/.")
    args = parser.parse_args(argv)

    root = repo_root_from_this_file()
    ensure_sys_path(root)  # make 'scripts' importable

    dest_dir = root / "data" / "input" / "cerro_negro"
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / ZIP_NAME

    try:
        download_to_file(URL, zip_path)
        unzip_to_dir(zip_path, dest_dir)
        # flatten possible nested folder from the zip
        maybe_flatten_single_subdir(dest_dir)

        # remove zip to keep tree tidy
        try:
            zip_path.unlink()
            print(f"Removed ZIP: {zip_path}")
        except Exception as e:
            print(f"[WARN] Could not remove zip: {e}")

        if args.prepare:
            # now import and run the standardizer
            from scripts.data_handling.cerro_negro_loader import prepare_cerro_negro
            obs_csv, sites_csv, wind_txt, conf_path = prepare_cerro_negro(
                cerro_dir=dest_dir,
                work_dir=root / "data" / "input",
            )
            print("\nPrepared working inputs:")
            print("  observations:", obs_csv)
            print("  sites:", sites_csv)
            print("  wind:", wind_txt)
            print("  conf:", conf_path)

        print("\nDone.")
        return 0
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
