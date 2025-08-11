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
    """
    This file lives at: <repo_root>/scripts/data_handling/download_cerro_negro.py
    So the repo root is 2 levels up.
    """
    return Path(__file__).resolve().parents[2]

def ensure_sys_path(root: Path) -> None:
    """
    Ensure <repo_root> is on sys.path so we can import:
        from scripts.data_handling.cerro_negro_loader import prepare_cerro_negro
    """
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

def download_to_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading:\n  {url}\n→ {dest}")
    with urlopen(url) as r, open(dest, "wb") as f:
        total = getattr(r, "length", None) or 0
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
        if total:
            print()
    print("Download complete.")

def unzip_to_dir(zip_path: Path, out_dir: Path) -> None:
    print(f"Unzipping:\n  {zip_path}\n→ {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    print("Unzip complete.")

def flatten_zip_payload(parent: Path) -> None:
    """
    Flatten archives that unpack to a single real folder (plus optional __MACOSX).
    Moves the contents of that folder up into `parent` and removes the folder.
    Repeats if there are nested single folders.
    """
    def visible_entries(p: Path):
        return [x for x in p.iterdir() if not x.name.startswith(".") and x.name != "__MACOSX"]

    while True:
        entries = visible_entries(parent)

        # Case 1: exactly one directory → flatten it
        if len(entries) == 1 and entries[0].is_dir():
            sub = entries[0]
            print(f"Flattening single folder: {sub.name} → {parent}")
            for item in list(sub.iterdir()):
                target = parent / item.name
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                shutil.move(str(item), str(target))
            sub.rmdir()
            continue  # re-check in case of nested single folders

        # Case 2: one real directory + __MACOSX → flatten the real one
        entries_all = [x for x in parent.iterdir()]
        real_dirs = [x for x in entries_all if x.is_dir() and x.name != "__MACOSX"]
        others    = [x for x in entries_all if x.name == "__MACOSX" or x.name.startswith(".")]

        if len(real_dirs) == 1 and not [e for e in entries_all if e not in real_dirs + others]:
            sub = real_dirs[0]
            print(f"Flattening folder (with __MACOSX present): {sub.name} → {parent}")
            for item in list(sub.iterdir()):
                target = parent / item.name
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                shutil.move(str(item), str(target))
            shutil.rmtree(sub)
            continue
        break


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fetch Cerro Negro short-course bundle.")
    parser.add_argument("--prepare", action="store_true",
                        help="After download+unzip, standardize working copies into data/input/.")
    args = parser.parse_args(argv)

    root = repo_root_from_this_file()
    ensure_sys_path(root)  # make '<repo_root>' importable

    dest_dir = root / "data" / "input" / "cerro_negro"
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / ZIP_NAME

    try:
        download_to_file(URL, zip_path)
        unzip_to_dir(zip_path, dest_dir)
        flatten_zip_payload(dest_dir)

        # remove zip to keep tree tidy
        try:
            zip_path.unlink()
            print(f"Removed ZIP: {zip_path}")
        except Exception as e:
            print(f"[WARN] Could not remove zip: {e}")

        if args.prepare:
            # Import with the correct package path based on your structure
            try:
                from scripts.data_handling.cerro_negro_loader import prepare_cerro_negro
            except Exception as e:
                print(
                    "[ERROR] Could not import 'prepare_cerro_negro' from scripts.data_handling.cerro_negro_loader.\n"
                    f"  Details: {e}\n"
                    "  Tip: run from repo root:\n"
                    "       python -m scripts.data_handling.download_cerro_negro --prepare"
                )
                return 1

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
