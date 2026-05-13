#!/usr/bin/env python3
"""Convert downloaded Rieth TEP RData files into parquet.

The Rieth release stores each combination as a single nested R dataframe.
This script flattens them and emits one parquet per source file with the
columns sensorlab.data.loader._load_real expects.
"""

from __future__ import annotations

import sys

import sensorlab  # noqa: F401
from sensorlab.config import PROCESSED_DIR, RAW_DIR, ensure_dirs


def main() -> int:
    try:
        import pyreadr
    except ImportError:
        print(
            "pyreadr not installed. Run `pip install pyreadr` to convert the RData files.\n"
            "(It is intentionally not in the default deps because not every user runs the "
            "real-data pipeline.)",
            file=sys.stderr,
        )
        return 1

    ensure_dirs()
    src_dir = RAW_DIR / "tep"
    out_dir = PROCESSED_DIR / "tep"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list(src_dir.glob("*.RData"))
    if not files:
        print(f"No RData files in {src_dir}. Run `make download-tep` first.", file=sys.stderr)
        return 1

    for fp in files:
        print(f"  reading {fp.name}…")
        bundle = pyreadr.read_r(str(fp))
        for key, df in bundle.items():
            df.columns = [str(c) for c in df.columns]
            out = out_dir / f"{fp.stem}__{key}.parquet"
            df.to_parquet(out, index=False)
            print(f"    -> {out.name}  ({df.shape[0]:,} rows)")
    print(f"\nDone. Parquet files in {out_dir}")
    print("Now you can run: python scripts/train_all.py --data real")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
