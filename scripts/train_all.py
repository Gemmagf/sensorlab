#!/usr/bin/env python3
"""Backwards-compatible entry point — delegates to ``sensorlab train``.

Prefer the CLI directly::

    sensorlab train --data synthetic --seed 0
    sensorlab evaluate --seeds 0 1 2
"""

from __future__ import annotations

import sys

from sensorlab.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
