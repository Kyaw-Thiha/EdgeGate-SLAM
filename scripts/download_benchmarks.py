from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from urllib.request import urlretrieve

DATASETS = {
    "intel": {
        "url": "https://raw.githubusercontent.com/david-m-rosen/SE-Sync/master/data/intel.g2o",
        "filename": "intel.g2o",
    },
    "m3500": {
        "url": "https://dl.dropboxusercontent.com/s/gmdzo74b3tzvbrw/input_M3500_g2o.g2o",
        "filename": "M3500.g2o",
    },
    "mit": {
        "url": "https://raw.githubusercontent.com/david-m-rosen/SE-Sync/master/data/MIT.g2o",
        "filename": "MIT.g2o",
    },
    "csail": {
        "url": "https://raw.githubusercontent.com/david-m-rosen/SE-Sync/master/data/CSAIL.g2o",
        "filename": "CSAIL.g2o",
    },
    "manhattan": {
        "url": "https://raw.githubusercontent.com/david-m-rosen/SE-Sync/master/data/manhattan.g2o",
        "filename": "manhattan.g2o",
    },
    "city10000": {
        "url": "https://raw.githubusercontent.com/david-m-rosen/SE-Sync/master/data/city10000.g2o",
        "filename": "city10000.g2o",
    },
    "parking-garage": {
        "url": "https://raw.githubusercontent.com/david-m-rosen/SE-Sync/master/data/parking-garage.g2o",
        "filename": "parking-garage.g2o",
    },
}

EXTRAS = {
    "sphere2500": {
        "url": "https://raw.githubusercontent.com/david-m-rosen/SE-Sync/master/data/sphere2500.g2o",
        "filename": "sphere2500.g2o",
        "note": "SE(3) quaternion — not usable until SE3 parser is added (Phase 2)",
    },
    "m3500a": {
        "url": "https://dl.dropboxusercontent.com/s/m9e866tdr2jlhf6/input_M3500a_g2o.g2o",
        "filename": "M3500a.g2o",
        "note": "M3500 variant with 0.1rad extra noise on orientations",
    },
    "intel-raw": {
        "url": "http://www2.informatik.uni-freiburg.de/~stachnis/datasets/datasets/intel-lab/intel.log.gz",
        "filename": "intel.log.gz",
        "note": "Intel Carmen raw log (gzipped) — needed for laser scan overlay in demo_rerun.py",
    },
}


def _format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KiB"
    else:
        return f"{num_bytes / (1024 * 1024):.1f} MiB"


def _download_one(name: str, url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        size = _format_size(dest.stat().st_size)
        print(f"  {name:20s}  {size:>10s}  [skipped, already downloaded]")
        return True

    last_report = [0.0]

    def _progress(block_count: int, block_size: int, total_size: int) -> None:
        now = time.time()
        if now - last_report[0] < 0.3:
            return
        last_report[0] = now
        downloaded = block_count * block_size
        pct = (downloaded / total_size * 100) if total_size > 0 else 0
        d_size = _format_size(downloaded)
        t_size = _format_size(total_size) if total_size > 0 else "?"
        print(f"\r  {name:20s}  {d_size:>10s} / {t_size:>10s}  {pct:3.0f}%", end="")

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(url, dest, _progress)
        size = _format_size(dest.stat().st_size)
        print(f"\r  {name:20s}  {size:>10s}  {'done':>14s}")
        return True
    except Exception as exc:
        print(f"\r  {name:20s}  {'FAILED':>10s}")
        print(f"      {exc}")
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False


def main() -> None:
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    include_extras = "--extras" in sys.argv or "-e" in sys.argv

    print("Downloading benchmark datasets to data/raw/\n")

    core_ok = 0
    for name, info in DATASETS.items():
        dest = out_dir / info["filename"]
        if _download_one(name, info["url"], dest):
            core_ok += 1

    print()

    extra_ok = 0
    if include_extras:
        print("Downloading extra datasets to data/raw/\n")
        for name, info in EXTRAS.items():
            dest = out_dir / info["filename"]
            if _download_one(name, info["url"], dest):
                extra_ok += 1
            note = info.get("note")
            if note:
                print(f"      NOTE: {note}")
        print()

    print(f"Core:   {core_ok}/{len(DATASETS)} downloaded")
    if include_extras:
        print(f"Extras: {extra_ok}/{len(EXTRAS)} downloaded")

    if core_ok == len(DATASETS):
        print("\nAll benchmarks ready.")
    else:
        print(f"\n{len(DATASETS) - core_ok} dataset(s) failed to download.", file=sys.stderr)
        print("Re-run to retry — already-downloaded files will be skipped.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
