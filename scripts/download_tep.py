#!/usr/bin/env python3
"""Download the Rieth et al. 2017 Tennessee Eastman dataset.

Source: Harvard Dataverse — https://doi.org/10.7910/DVN/6C3JR1
Size: roughly 5 GB across four ``.RData`` files (fault-free / faulty,
train / test).

The files land in ``data/raw/tep/``. Convert them to parquet with
``python scripts/prepare_tep.py`` afterwards.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import sensorlab  # noqa: F401  — env var setup
from sensorlab.config import RAW_DIR, ensure_dirs

# Harvard Dataverse direct-download URLs (file IDs from the published dataset).
FILES: dict[str, str] = {
    "TEP_FaultFree_Training.RData": "https://dataverse.harvard.edu/api/access/datafile/3500113",
    "TEP_FaultFree_Testing.RData": "https://dataverse.harvard.edu/api/access/datafile/3500111",
    "TEP_Faulty_Training.RData": "https://dataverse.harvard.edu/api/access/datafile/3500112",
    "TEP_Faulty_Testing.RData": "https://dataverse.harvard.edu/api/access/datafile/3500110",
}


def _download(url: str, target: Path) -> None:
    """Streaming download with a tqdm progress bar."""
    from tqdm import tqdm

    req = urllib.request.Request(url, headers={"User-Agent": "sensorlab-fetch/0.1"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        with (
            tqdm(total=total, unit="B", unit_scale=True, desc=target.name) as bar,
            target.open("wb") as out,
        ):
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
                bar.update(len(chunk))


def main() -> int:
    ensure_dirs()
    target_dir = RAW_DIR / "tep"
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Rieth 2017 TEP release into {target_dir}\n")
    for name, url in FILES.items():
        path = target_dir / name
        if path.exists() and path.stat().st_size > 0:
            print(f"  ✓ {name} already present, skipping")
            continue
        try:
            _download(url, path)
        except Exception as e:
            print(f"  ✗ failed to download {name}: {e}", file=sys.stderr)
            print(
                "    You can download the files manually from "
                "https://doi.org/10.7910/DVN/6C3JR1 and place them in "
                f"{target_dir}",
                file=sys.stderr,
            )
            return 1
    print("\nDone. Next: `python scripts/prepare_tep.py` to convert to parquet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
